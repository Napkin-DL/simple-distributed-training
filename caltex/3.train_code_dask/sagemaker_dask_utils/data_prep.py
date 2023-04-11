import glob
import json
import logging
import os
import pathlib
from typing import List
from typing import Optional
from typing import Tuple
from typing import Union

import dask.dataframe as dd
import pandas as pd


logging.basicConfig(level=logging.INFO)

CSV_EXTENSION = ".csv"
PARQUET_EXTENSION = ".parquet"
INPUT_DATA_FILENAME_CSV = "*.csv"
INPUT_DATA_FILENAME_PARQUET = "*.parquet"
TRAIN_CHANNEL = "train"
VALIDATION_CHANNEL = "validation"
BLOCK_SIZE = "16MB"
TEXT_CSV = "text/csv"
APPLICATION_PARQUET = "application/x-parquet"
TRAIN_VALIDATION_FRACTION = 0.2
RANDOM_STATE_SAMPLING = 200
DEFAULT_AUTO = "auto"


def validate_file_path(path: str, file_extension: Optional[str] = CSV_EXTENSION) -> None:
    """Validate the file path based on argument file_extension under the directory of depth one.

    If the path does not exist, raise FileNotFoundError.
    If the path is a directory, raise FileNotFoundError if no file that ends with file_extension
     was found in the directory (non-recursive);
    If the path points to a file, raise ValueError if the path points to a a file that does not end with file_extension.

    Args:
        path (str): either a directory or file.
        file_extension (str): file extension used to identify a file.

    Raises:
        ValueError if the path points to a file which does not end with file_extension.
        FileNotFoundError if no file that ends with file_extension was found in the directory (non-recursive).
    """

    if not os.path.exists(path):
        raise FileNotFoundError(f"Path '{path}' does not exist.")

    if os.path.isfile(path):
        if not path.endswith(file_extension):
            raise ValueError(f"specified file at path '{path}' must ends with '{file_extension}'.")
    else:
        file_paths = [file_path for file_path in os.listdir(path) if file_path.endswith(file_extension)]
        if len(file_paths) == 0:
            raise FileNotFoundError(f"Could not find any valid file ending in '{file_extension}' under '{path}'.")
        if len(file_paths) > 1:
            logging.info(
                f"Found more than 1 file ending in '{file_extension}' under '{path}'. "
                f"All of the files ending in '{file_extension}' will be used as training or validation data."
            )


def import_cat_index_list(cat_index_path: List[str], feature_dim: int) -> List[int]:
    """Read the categorical column index from a json file, returns the list of categorical column index.

    The JSON file should be formatted such that the key is 'cat_index' and value is a list of categorical column index.

    Args:
        cat_index_path: a list of path string that indicates the directory saving the categorical column index info.
        feature_dim (int): dimension of predicting features.

    Raises:
        ValueError if the categorical feature (column) index is not integer.
        ValueError if the categorical feature (column) index is smaller than 1 or
        greater than the index of last feature dimension.

    Returns:
        list of categorical columns indexes.
    """
    assert (
        len(cat_index_path) == 1
    ), "Found json files for categorical indexes more than 1. Please ensure there is only one json file."

    with open(cat_index_path[0], "r") as f:
        cat_index_dict = json.loads(f.read())

    if cat_index_dict is not None:
        cat_index_list = list(cat_index_dict.values())[0]  # convert dict.values() from an iterable to a python list.
        if not cat_index_list:  # abort early if it is an empty list with no indexes for categorical columns
            return []

        if all(isinstance(index, int) for index in cat_index_list) is False:
            raise ValueError(
                f"Found non-integer index for categorical feature. "
                f"Please ensure each index is an integer from 1 to {feature_dim}."
            )
        if all(1 <= index <= feature_dim for index in cat_index_list) is False:
            raise ValueError(
                f"Found index for categorical feature smaller than 1 or greater than the index of "
                f"last feature {feature_dim}."
            )
        return [index - 1 for index in cat_index_list]  # offset by 1 as the first column in the input is the target.
    else:
        return []


def get_categorical_features_index(train_dir: str, feature_dim: int) -> Union[str, List[int]]:
    """Read the categorical features indexes file (*.json) used for model fitting.

    Args:
        train_dir (str): the training directory where categorical feature indexes file (*.json)
        is saved along with the training data.

        feature_dim (int): the number of columns.

    Returns:
        A string 'auto' or a list of integers indicating the categorical feature (column) indexes.
    """

    cat_index_path = list(pathlib.Path(train_dir).glob("*.json"))
    categorical_feature = DEFAULT_AUTO  # default setting
    if cat_index_path:
        cat_index_list = import_cat_index_list(cat_index_path=cat_index_path, feature_dim=feature_dim)
        if cat_index_list:
            categorical_feature = cat_index_list
            logging.info(
                f"Found categorical feature indexes file. The categorical column indexes are: {cat_index_list}."
            )
    return categorical_feature


def split_train_validation_data(
    df_train: Union[pd.DataFrame, dd.DataFrame],
    train_validation_fraction: Optional[int] = TRAIN_VALIDATION_FRACTION,
    random_state_sampling: Optional[int] = RANDOM_STATE_SAMPLING,
) -> Union[Tuple[pd.DataFrame, pd.DataFrame], Tuple[dd.DataFrame, dd.DataFrame]]:
    """Split the data into training and validation data based on the data type.

    Args:
        df_train (Union[pd.DataFrame, dd.DataFrame]): the dataset to split into training and validation.
        train_validation_fraction (int): the fraction of train and validation data split.
        random_state_sampling (int): random state value for random sampling.
    Returns:
        Tuple of training and validation data.
    """

    if isinstance(df_train, dd.DataFrame):
        df_train, df_validation = df_train.random_split(
            frac=[1 - train_validation_fraction, train_validation_fraction],
            random_state=random_state_sampling,
        )
    else:
        df_validation = df_train.sample(
            frac=train_validation_fraction,
            random_state=random_state_sampling,
        )
        df_train.drop(df_validation.index, inplace=True)
    return df_train, df_validation


def load_data_csv(
    data_dir: str,
    use_dask_data_loader: Optional[bool] = False,
    is_validation: Optional[bool] = False,
    block_size: Optional[str] = BLOCK_SIZE,
    input_data_filename: Optional[str] = INPUT_DATA_FILENAME_CSV,
) -> Union[pd.DataFrame, dd.DataFrame]:
    """Read CSV-format data using either pandas or dask data loader.

    Load the training and validation data using either pandas for single instance training or dask
    for distributed training.

    Args:
        data_dir (str): directory which to save the data.
        use_dask_data_loader (bool): whether to use dask data loader for distributed training or not.
        is_validation (bool): whether the data to read is validation data or not (used by dask.dataframe.read_csv only).
        block_size (str): number of bytes by which to cut up larger files for dask.dataframe.read_csv.
        input_data_filename (str): file names that are used to identify input data.

    Returns:
        Either pandas.dataframe or dask.dataframe object.
    """

    validate_file_path(path=data_dir, file_extension=CSV_EXTENSION)
    is_file = os.path.isfile(data_dir)
    if use_dask_data_loader:  # this corresponds to dask distributed training setting
        block_size = None if is_validation is True else block_size
        data = dd.read_csv(
            data_dir if is_file is True else os.path.join(data_dir, input_data_filename),
            header=None,  # We force validation data to have 0 partition
            # such that the entire validation data are sent to a single machine, the log of which can be used by AMT.
            blocksize=block_size,
        )

        if is_validation and data.npartitions > 1:  # this allows only one instance to receive all validation data
            data = data.repartition(npartitions=1)
    else:  # this corresponds to single instance training without using dask
        if is_file:
            data = pd.read_csv(data_dir, header=None)
        else:
            all_files = glob.glob(os.path.join(data_dir, input_data_filename))
            print(f"data frame {all_files}??? {pd.read_csv(all_files[0], header=None)}")
            data = pd.concat((pd.read_csv(f, header=None) for f in all_files), axis=0, ignore_index=True)
    return data


def load_data_parquet(
    data_dir: str,
    use_dask_data_loader: Optional[bool] = False,
    is_validation: Optional[bool] = False,
    block_size: Optional[str] = BLOCK_SIZE,
    input_data_filename: Optional[str] = INPUT_DATA_FILENAME_PARQUET,
) -> Union[pd.DataFrame, dd.DataFrame]:
    """Read PARQUET-format data using either pandas or dask data loader.

    Load the training and validation data using either pandas for single instance training or dask
    for distributed training.

    Args:
        data_dir (str): directory where to save the data.
        use_dask_data_loader (bool): whether to use dask data loader for distributed training or not.
        is_validation (bool): whether the data to read is validation data or not (used by dask.dataframe.read_csv only).
        block_size (str): number of bytes by which to cut up larger files for dask.dataframe.read_csv.
        input_data_filename (str): file names that are used to identify input data.

    Returns:
        Either pandas.dataframe or dask.dataframe object.
    """

    validate_file_path(path=data_dir, file_extension=PARQUET_EXTENSION)
    is_file = os.path.isfile(data_dir)
    if use_dask_data_loader:  # this corresponds to dask distributed training setting
        if not is_validation:
            data = dd.read_parquet(
                data_dir if is_file is True else os.path.join(data_dir, input_data_filename),
            ).repartition(
                partition_size=block_size
            )  # read_parquet function cannot partition the data well.
            # A repartition is needed after the read_parquet function call.
        else:
            data = dd.read_parquet(
                data_dir if is_file is True else os.path.join(data_dir, input_data_filename),
            ).repartition(npartitions=1)
    else:  # this corresponds to single instance training without using dask
        data = pd.read_parquet(data_dir)
    return data


def load_data(
    data_dir: str,
    content_type: str,
    use_dask_data_loader: Optional[bool] = False,
    is_validation: Optional[bool] = False,
    block_size: Optional[str] = BLOCK_SIZE,
) -> Union[pd.DataFrame, dd.DataFrame]:
    """Load data based on the content type.

    If content type is 'text/csv', use function load_data_csv; if content type is 'application/x-parquet', use function
    load_data_parquet.

    Args:
        data_dir (str): directory where to save the data.
        content_type (str): content type of train and validation data.
        use_dask_data_loader (bool): whether to use dask data loader for distributed training or not.
        is_validation (bool): whether the data to read is validation data or not (used by dask.dataframe.read_csv only).
        block_size (str): number of bytes by which to cut up larger files for dask.dataframe.read_csv.

    Returns:
        Either pandas.dataframe or dask.dataframe object.
    """

    if content_type == TEXT_CSV:
        data = load_data_csv(
            data_dir=data_dir,
            use_dask_data_loader=use_dask_data_loader,
            is_validation=is_validation,
            block_size=block_size,
            input_data_filename=INPUT_DATA_FILENAME_CSV,
        )
    elif content_type == APPLICATION_PARQUET:
        data = load_data_parquet(
            data_dir=data_dir,
            use_dask_data_loader=use_dask_data_loader,
            is_validation=is_validation,
            block_size=block_size,
            input_data_filename=INPUT_DATA_FILENAME_PARQUET,
        )
    else:
        raise ValueError(
            f"Content types for train and validation channel are found inconsistent. "
            f"Please specify a consistent value of either '{TEXT_CSV}' or '{APPLICATION_PARQUET}' and retry."
        )
    return data


def prepare_data(
    train_dir: str,
    validation_dir: Optional[str],
    use_dask_data_loader: Optional[bool] = False,
    content_type: Optional[str] = TEXT_CSV,
    train_channel: Optional[str] = TRAIN_CHANNEL,
    validation_channel: Optional[str] = VALIDATION_CHANNEL,
    train_validation_fraction: Optional[int] = TRAIN_VALIDATION_FRACTION,
    random_state_sampling: Optional[int] = RANDOM_STATE_SAMPLING,
) -> Union[
    Tuple[pd.DataFrame, pd.core.series.Series, pd.DataFrame, pd.core.series.Series],
    Tuple[dd.DataFrame, dd.core.Series, dd.DataFrame, dd.core.Series],
]:
    """Prepare and read train and validation data.

    Read data from train and validation channel, and return predicting features and target variables.

    Args:
        train_dir (str): directory which saves the training data.
        validation_dir (str): directory which saves the validation data.
        use_dask_data_loader (bool): whether to use dask data loader for distributed training or not.
        content_type (str): content type of train and validation data.
        train_channel (str): directory to save train data.
        validation_channel (str): directory to save validation data.
        train_validation_fraction (int): the fraction of train and validation data split.
        random_state_sampling (int): random state value for random sampling.

    Returns:
        Tuple of training features, training target, validation features, validation target.
    """

    if validation_dir is not None:
        logging.info(
            "Found data in the validation channel. "
            "Reading the train and validation data from the training and validation channel, respectively."
        )
        df_train = load_data(
            data_dir=train_dir,
            content_type=content_type,
            use_dask_data_loader=use_dask_data_loader,
            is_validation=False,
        )
        df_validation = load_data(
            data_dir=validation_dir,
            content_type=content_type,
            use_dask_data_loader=use_dask_data_loader,
            is_validation=True,
        )
    else:
        if os.path.exists(os.path.join(train_dir, train_channel)):
            df_train = load_data(
                data_dir=os.path.join(train_dir, train_channel),
                content_type=content_type,
                use_dask_data_loader=use_dask_data_loader,
                is_validation=False,
            )
            try:
                df_validation = load_data(
                    data_dir=os.path.join(train_dir, validation_channel),
                    content_type=content_type,
                    use_dask_data_loader=use_dask_data_loader,
                    is_validation=True,
                )
            except FileNotFoundError:  # when validation data is not available in the directory
                logging.info(
                    f"Validation data is not found. {train_validation_fraction*100}% of training data is "
                    f"randomly selected as validation data. The seed for random sampling is "
                    f"{random_state_sampling}."
                )
                df_train, df_validation = split_train_validation_data(
                    df_train=df_train,
                    train_validation_fraction=train_validation_fraction,
                    random_state_sampling=random_state_sampling,
                )
        else:
            df_train = load_data(
                data_dir=train_dir,
                content_type=content_type,
                use_dask_data_loader=use_dask_data_loader,
                is_validation=False,
            )
            logging.info(
                f"Validation data is not found. {train_validation_fraction * 100}% of training data is "
                f"randomly selected as validation data. The seed for random sampling is "
                f"{random_state_sampling}."
            )
            df_train, df_validation = split_train_validation_data(
                df_train=df_train,
                train_validation_fraction=train_validation_fraction,
                random_state_sampling=random_state_sampling,
            )
    X_train, y_train = df_train.iloc[:, 1:], df_train.iloc[:, 0]
    X_val, y_val = df_validation.iloc[:, 1:], df_validation.iloc[:, 0]
    return X_train, y_train, X_val, y_val
