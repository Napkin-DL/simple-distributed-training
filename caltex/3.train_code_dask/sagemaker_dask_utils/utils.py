import argparse
import logging
import os
import pathlib
import tarfile
from typing import Any
from typing import Optional
from typing import Union

import joblib
from sagemaker_dask_utils import model_info


logging.basicConfig(level=logging.INFO)


TRAIN = "train"
TRAIN_ALT = "training"
VALIDATION = "validation"
CONTENT_TYPE = "ContentType"
DEFAULT_CONTENT_TYPE = "text/csv"
CONTENT_TYPE_PARQUET = "application/x-parquet"
INPUT_MODEL_UNTARRED_PATH = "_input_model_extracted/"
SAVE_MODEL_FILE_NAME = "model.pkl"


def assign_train_argument(args: argparse.Namespace) -> None:
    """Determine training data directory from the argument, mutate args.train if necessary.

    Allows users to pass in training data through either 'train' or 'training' channel.

    Args:
        args (argparse.Namespace): model arguments passed by SageMaker DLC.

    Raises:
        ValueError if both training and train channels are set, or if neither of them are set.
    """

    if args.train is not None and args.train_alt is not None:
        raise ValueError(
            "Both 'train' and 'training' channel are identified. "
            "Could not resolve ambiguity on which channel to use to read training data. "
            "Please specify only one of 'train' or 'training' channel and try again."
        )
    if args.train is None and args.train_alt is not None:
        args.train = args.train_alt
    if args.train is None and args.train_alt is None:
        raise ValueError(
            "Neither 'train' nor 'training' channel is identified. "
            "Please specify one of 'train' or 'training' channel to pass in training data and try again."
        )


def validate_content_type(content_type: str) -> None:
    """Validate the content type to be one of 'text/csv' and 'application/x-parquet'.

    Args:
        content_type (str):  content type of input data.
    Raises:
        ValueError if the content type is neither 'text/csv' or 'application/x-parquet'.
    """

    if content_type not in [DEFAULT_CONTENT_TYPE, CONTENT_TYPE_PARQUET]:
        raise ValueError(
            f"Content type of input data should be either 'text/csv' or 'application/x-parquet'. However, "
            f"'{content_type}' is found. "
            f"Please specify the content type to be either 'text/csv' or 'application/x-parquet'"
        )


def get_content_type(input_data_config: dict) -> str:
    """Extract the content type for train and validation channel (if existing).

    If sagemaker.inputs.TrainingInput are used for train and validation channel, the content types for train and
    validation channel should be consistent.

    Args:
        input_data_config (dict): input configuration for train and validation data.

    Returns:
        content type (str): either 'text/csv' or 'application/x-parquet'.

    Raises:
        ValueError if the content types of training and validation data are inconsistent.
    """

    if TRAIN in input_data_config:  # this corresponds to two scenarios:  train channel is named as 1/ 'train';
        # and 2/ 'training'.
        print("aaa1")
        train_data_content_type = input_data_config[TRAIN].get(CONTENT_TYPE, None)
    elif TRAIN_ALT in input_data_config:
        print("aaa2")
        train_data_content_type = input_data_config[TRAIN_ALT].get(CONTENT_TYPE, None)

    if train_data_content_type:  # Content type being None is resulted from two cases:
        # 1/ sagemaker.inputs.TrainingInput is not used to configure train and validation channel;
        # 2/ sagemaker.inputs.TrainingInput is used to configure
        # train and validation channel but ContentType argument is not specified.
        print(f"train_data_content_type1 : {train_data_content_type}")
        validate_content_type(train_data_content_type)

    if VALIDATION in input_data_config:
        print("aaa4")
        validation_data_content_type = input_data_config[VALIDATION].get(CONTENT_TYPE, None)
        if validation_data_content_type:
            print(f"validation_data_content_type1 : {validation_data_content_type}")
            validate_content_type(validation_data_content_type)
        if train_data_content_type != validation_data_content_type:
            raise ValueError(
                f"Content type for train channel is '{train_data_content_type}' "
                f"while that for validation channel is '{validation_data_content_type}'. "
                f"Please specify a consistent value of either 'text/csv' or 'application/x-parquet' and retry."
            )
    if train_data_content_type:
        print(f"train_data_content_type2 : {train_data_content_type}")
        content_type = train_data_content_type
    else:
        logging.info(
            f"'ContentType' is not identified in either training or validation data channel. "
            f"Default ContentType '{DEFAULT_CONTENT_TYPE}' is used to read the train and validation data."
        )
        content_type = DEFAULT_CONTENT_TYPE
        print(f"content_type : {content_type}")
    return content_type


def save_model(model: Any, model_dir: str) -> None:
    """Save the trained model to disk as pickled lightgbm.basic.Booster object.

    Args:
        model (lgb.basic.Booster): the trained model object.
        model_dir (str): the directory where model is saved.
    """

    logging.info("Saving model...")
    # save model to file
    export_model_path = os.path.join(model_dir, SAVE_MODEL_FILE_NAME)
    joblib.dump(model, export_model_path)


def get_pretrained_model(pretrained_model_dir: str) -> Union[None, Any]:
    """Read the trained model object for incremental training.

    Args:
        pretrained_model_dir (str): the directory where the previously trained model is saved.

    Return:
        If the pretrained model is not available, return None; otherwise return a lgb.basic.Booster object.
    """

    input_model_path = next(pathlib.Path(pretrained_model_dir).glob("*.tar.gz"))
    if not os.path.exists(INPUT_MODEL_UNTARRED_PATH):
        os.mkdir(INPUT_MODEL_UNTARRED_PATH)
    with tarfile.open(input_model_path, "r") as saved_model_tar:
        saved_model_tar.extractall(INPUT_MODEL_UNTARRED_PATH)

    is_partially_trained = model_info.is_input_model_partially_trained(
        input_model_untarred_path=INPUT_MODEL_UNTARRED_PATH
    )
    pretrained_model: Optional[Any] = None
    if is_partially_trained:
        pretrained_model = joblib.load(os.path.join(INPUT_MODEL_UNTARRED_PATH, SAVE_MODEL_FILE_NAME))
    return pretrained_model
