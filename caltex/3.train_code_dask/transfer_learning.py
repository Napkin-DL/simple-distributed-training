import argparse
import json
import logging
import os
import socket
import sys
import time

import lightgbm as lgb
from constants import constants
from distributed import Client
from sagemaker_dask_utils import dask_scheduler
from sagemaker_dask_utils import data_prep
from sagemaker_dask_utils import model_info
from sagemaker_dask_utils import utils
from utils import configure_parameters
from utils import infer_problem_type


logging.basicConfig(level=logging.INFO)


def _parse_args():
    parser = argparse.ArgumentParser()

    parser.add_argument("--model-dir", type=str, default=os.environ.get("SM_MODEL_DIR"))
    parser.add_argument("--train", type=str, default=os.environ.get("SM_CHANNEL_TRAINING"))
    parser.add_argument("--train-alt", type=str, default=os.environ.get("SM_CHANNEL_TRAIN"))
    parser.add_argument("--validation", type=str, default=os.environ.get("SM_CHANNEL_VALIDATION"))
    parser.add_argument("--input-data-config", type=str, default=os.environ.get("SM_INPUT_DATA_CONFIG"))
    parser.add_argument("--hosts", type=list, default=json.loads(os.environ.get("SM_HOSTS")))
    parser.add_argument("--current-host", type=str, default=os.environ.get("SM_CURRENT_HOST"))
    parser.add_argument("--pretrained-model", type=str, default=os.environ.get("SM_CHANNEL_MODEL"))
    parser.add_argument("--num_boost_round", type=int, default=constants.DEFAULT_NUM_BOOST_ROUND)
    parser.add_argument("--early_stopping_rounds", type=int, default=constants.DEFAULT_EARLY_STOPPING_ROUNDS)
    parser.add_argument("--metric", type=str, default=constants.DEFAULT_METRIC)
    parser.add_argument("--learning_rate", type=float, default=constants.DEFAULT_LEARNING_RATE)
    parser.add_argument("--num_leaves", type=int, default=constants.DEFAULT_NUM_LEAVES)
    parser.add_argument("--feature_fraction", type=float, default=constants.DEFAULT_FEATURE_FRACTION)
    parser.add_argument("--bagging_fraction", type=float, default=constants.DEFAULT_BAGGING_FRACTION)
    parser.add_argument("--bagging_freq", type=int, default=constants.DEFAULT_BAGGING_FREQUENCE)
    parser.add_argument("--max_depth", type=int, default=constants.DEFAULT_MAX_DEPTH)
    parser.add_argument("--min_data_in_leaf", type=int, default=constants.DEFAULT_MIN_DATA_IN_LEAF)
    parser.add_argument("--max_delta_step", type=float, default=constants.DEFAULT_MAX_DELTA_STEP)
    parser.add_argument("--lambda_l1", type=float, default=constants.DEFAULT_LAMBDA_L1)
    parser.add_argument("--lambda_l2", type=float, default=constants.DEFAULT_LAMBDA_L2)
    parser.add_argument("--boosting", type=str, default=constants.DEFAULT_BOOSTING)
    parser.add_argument("--min_gain_to_split", type=float, default=constants.DEFAULT_MIN_GAIN_TO_SPLIT)
    parser.add_argument("--scale_pos_weight", type=float, default=constants.DEFAULT_SCALE_POS_WEIGHT)
    parser.add_argument("--tree_learner", type=str, default=constants.DEFAULT_TREE_LEARNER)
    parser.add_argument("--feature_fraction_bynode", type=float, default=constants.DEFAULT_FEATURE_FRACTION_BYNODE)
    parser.add_argument("--is_unbalance", type=str, default=constants.DEFAULT_IS_UNBALANCE)
    parser.add_argument("--max_bin", type=int, default=constants.DEFAULT_MAX_BIN)
    parser.add_argument("--num_threads", type=int, default=constants.DEFAULT_NUM_THREADS)
    parser.add_argument("--verbosity", type=int, default=constants.DEFAULT_VERBOSITY)
    parser.add_argument("--use_dask", type=str, default=constants.DEFAULT_USE_DASK)

    return parser.parse_known_args()


def run_with_args(args):
    """Run training."""

    utils.assign_train_argument(args)
    input_data_config = json.loads(args.input_data_config)
    if not (args.use_dask == "True" or len(args.hosts) > 1):  # this corresponds to single instance training
        logging.info("Loading data")
        X_train, y_train, X_val, y_val = data_prep.prepare_data(
            train_dir=os.path.join(args.train),
            validation_dir=args.validation,
            use_dask_data_loader=False,
            content_type=utils.get_content_type(input_data_config=input_data_config),
        )

        # create dataset for lightgbm
        lgb_train = lgb.Dataset(X_train, y_train)
        lgb_eval = lgb.Dataset(X_val, y_val, reference=lgb_train)

        # get problem type (binary classification or multi-class classification) from y_train
        problem_type, num_classes_y = infer_problem_type(y_train)

        if args.metric == constants.AUTO:
            metric = constants.CLASSIFICATION_PROBLEM_TYPE_METRIC_MAPPING[problem_type]
        else:
            metric = args.metric

        params = configure_parameters(args=args, problem_type=problem_type, is_for_dask_train=False)
        params.update(metric=metric)

        if problem_type == constants.MULTI_CLASSIFICATION:
            params.update(num_class=num_classes_y)

        categorical_feature = data_prep.get_categorical_features_index(
            train_dir=args.train, feature_dim=X_train.shape[1]
        )

        booster = utils.get_pretrained_model(pretrained_model_dir=args.pretrained_model)

        logging.info("Beginning training")
        gbm = lgb.train(
            params,
            lgb_train,
            num_boost_round=args.num_boost_round,
            valid_sets=[lgb_eval, lgb_train],
            valid_names=["val", "train"],
            early_stopping_rounds=args.early_stopping_rounds,
            categorical_feature=categorical_feature,
            init_model=booster,
        )

        utils.save_model(model=gbm, model_dir=args.model_dir)
        model_info.save_model_info(
            input_model_untarred_path=constants.INPUT_MODEL_UNTARRED_PATH, model_dir=args.model_dir
        )

    else:  # this corresponds to distributed training using dask
        logging.info("Initializing a Dask cluster")

        main_host = args.hosts[0]  # choose the first host of the host list as the main host
        scheduler_ip = dask_scheduler.get_ip_from_host(main_host)
        current_host = args.current_host

        logging.info("Start Dask cluster in all nodes")
        dask_scheduler.start_daemons(scheduler_ip, main_host, current_host)

        time.sleep(constants.DASK_WAITTIME_SECONDS)
        if current_host == main_host:
            with Client(constants.SCHEDULER_IP_TEMPLATE.format(ip=scheduler_ip)) as client:
                try:
                    logging.info(f"Client summary: {client}.")

                    logging.info("Loading data")
                    X_train, y_train, X_val, y_val = data_prep.prepare_data(
                        train_dir=os.path.join(args.train),
                        validation_dir=args.validation,
                        use_dask_data_loader=True,
                        content_type=utils.get_content_type(input_data_config=input_data_config),
                    )

                    # get problem type (binary classification or multi-class classification) from y_train
                    problem_type, num_classes_y = infer_problem_type(y_train)

                    if args.metric == constants.AUTO:
                        metric = constants.CLASSIFICATION_PROBLEM_TYPE_METRIC_MAPPING[problem_type]
                    else:
                        metric = args.metric

                    params = configure_parameters(args=args, problem_type=problem_type, is_for_dask_train=True)
                    params.update(client=client)

                    if problem_type == constants.MULTI_CLASSIFICATION:
                        params.update(num_class=num_classes_y)

                    categorical_feature = data_prep.get_categorical_features_index(
                        train_dir=args.train, feature_dim=X_train.shape[1]
                    )

                    booster = utils.get_pretrained_model(pretrained_model_dir=args.pretrained_model)

                    dask_model = lgb.DaskLGBMClassifier(**params)

                    if args.use_dask == "True" and len(args.hosts) == 1:
                        callbacks = [
                            lgb.early_stopping(stopping_rounds=args.early_stopping_rounds),
                            lgb.log_evaluation(period=1, show_stdv=True),
                        ]
                    else:
                        callbacks = None
                        logging.warning(
                            "Disable early stopping feature in multi-instance dask training "
                            "due to an issue in the open sourced LightGBM repository. "
                            "For details, see https://github.com/microsoft/SynapseML/issues/728#issuecomment-1221599961"
                        )
                    logging.info("Beginning training")
                    dask_model.fit(
                        X=X_train,
                        y=y_train,
                        eval_set=[(X_val, y_val), (X_train, y_train)],
                        eval_names=["val", "train"],
                        callbacks=callbacks,
                        categorical_feature=categorical_feature,
                        eval_metric=metric,
                        init_model=booster,
                    )

                    if not dask_model.fitted_:
                        raise RuntimeError("Unexpected error detected. Model is not fitted.")
                    logging.info("Done training")

                    utils.save_model(model=dask_model.booster_, model_dir=args.model_dir)
                    model_info.save_model_info(
                        input_model_untarred_path=constants.INPUT_MODEL_UNTARRED_PATH, model_dir=args.model_dir
                    )
                finally:
                    # shutdown scheduler, retire and close workers
                    dask_scheduler.retire_workers(client=client)
                    client.shutdown()
                    sys.exit(0)
        else:
            while True:
                scheduler = (scheduler_ip, constants.SCHEDULER_PORT)
                alive_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
                alive_check = alive_socket.connect_ex(scheduler)
                if alive_check == 0:
                    alive_socket.close()
                    time.sleep(constants.DASK_WAITTIME_SECONDS)
                else:
                    logging.info("Received a shutdown signal from Dask cluster")
                    sys.exit(0)


if __name__ == "__main__":
    args, unknown = _parse_args()
    run_with_args(args)
