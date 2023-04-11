import argparse
import logging
from typing import Dict
from typing import Tuple
from typing import Union

import dask.dataframe as dd
import pandas as pd
from constants import constants


logging.basicConfig(level=logging.INFO)


def infer_problem_type(y_train: Union[dd.core.Series, pd.core.series.Series]) -> Tuple[str, int]:
    """Determine the problem type based on the number of unique values in the target.

    If the number of unique values in the target variable equals to 2, then it is binary classification type;
    if it is more than 2, it is multiclass classification type.

    Args:
        y_train (Union[dd.core.Series, pd.core.series.Series]): the target variable in the training set.

    Returns:
        Problem type of either binary classification or multiclass classification and
        number of unique values in the target.
    """

    num_classes_y = len(y_train.unique())
    if num_classes_y == 2:
        problem_type = constants.BINARY_CLASSIFICATION
    else:
        problem_type = constants.MULTI_CLASSIFICATION
    return problem_type, num_classes_y


def configure_parameters(args: argparse.Namespace, problem_type: str, is_for_dask_train: bool) -> Dict[str, str]:
    """Configure parameters for single instance and distributed training.

    Args:
        args (argparse.Namespace): model arguments passed by SageMaker DLC.
        problem_type (str): problem type of either binary or multiclass classification.
        is_for_dask_train (bool): whether the parameters are used for dask distributed training o not.

    Returns:
        A parameter-value pair dictionary.
    """

    params = {
        "device_type": "cpu",  # TODO: support GPU training.
        "boosting_type": "gbdt",
        "objective": constants.CLASSIFICATION_PROBLEM_TYPE_OBJECTIVE_MAPPING[problem_type],
        "learning_rate": args.learning_rate,
        "num_leaves": args.num_leaves,
        "feature_fraction": args.feature_fraction,
        "bagging_fraction": args.bagging_fraction,
        "bagging_freq": args.bagging_freq,
        "max_depth": args.max_depth,
        "min_data_in_leaf": args.min_data_in_leaf,
        "max_delta_step": args.max_delta_step,
        "lambda_l1": args.lambda_l1,
        "lambda_l2": args.lambda_l2,
        "boosting": args.boosting,
        "min_gain_to_split": args.min_gain_to_split,
        "scale_pos_weight": args.scale_pos_weight,
        "tree_learner": args.tree_learner,
        "feature_fraction_bynode": args.feature_fraction_bynode,
        "is_unbalance": args.is_unbalance == "True",
        "max_bin": args.max_bin,
        "num_threads": args.num_threads,
        "verbosity": args.verbosity,
    }
    if is_for_dask_train:
        params.update(
            n_estimators=args.num_boost_round,
        )
    return params
