import json
import logging
import os
from typing import Any
from typing import Dict


MODEL_INFO_FILE_NAME = "__models_info__.json"
IS_TRAINED_ON_INPUT_DATA = "is_trained_on_input_data"


logging.basicConfig(level=logging.INFO)


def is_input_model_partially_trained(input_model_untarred_path: str) -> bool:
    """Read the model_info file from file system to check if the input model was already trained.

    If the model_info file does not exists or if it does not contain information about previous training, we assume that
    the model was not previously trained.
    """
    input_model_info_file_path = os.path.join(input_model_untarred_path, MODEL_INFO_FILE_NAME)
    try:
        with open(input_model_info_file_path, "r") as f:
            model_info = json.load(f)
            is_trained_on_input_data = model_info[IS_TRAINED_ON_INPUT_DATA]
            assert type(is_trained_on_input_data) == bool, (
                "model_info file corrupted "
                f"{IS_TRAINED_ON_INPUT_DATA} parameter must be "
                f"either True or False, got {is_trained_on_input_data}."
            )
            return is_trained_on_input_data
    except FileNotFoundError:
        logging.info(f"'{input_model_info_file_path}' file could not be found.")
        return False
    except KeyError:
        logging.info(f"'{IS_TRAINED_ON_INPUT_DATA}' not in model info file. Assuming model was not fine-tuned.")
        return False
    except Exception as e:
        logging.error(
            f"Could not read or parse model_info file, exception: '{e}'. To continue training on previously "
            f"trained model, please create a json file {input_model_info_file_path} with"
            f" {IS_TRAINED_ON_INPUT_DATA} parameter set to True. To start training from scratch,"
            f"please create a json file {input_model_info_file_path} with"
            f" {IS_TRAINED_ON_INPUT_DATA} parameter set to False."
        )
        raise


def save_model_info(input_model_untarred_path: str, model_dir: str) -> None:
    """Save model info to the output directory along with is_trained_on_input_data parameter set to True.

    Read the existing model_info file in input_model directory if exists, set is_trained_on_input_data parameter to
    True and saves it in the output model directory.
    Args:
        input_model_untarred_path: Input model is untarred into this directory.
        model_dir: Output model directory.
    """

    input_model_info_file_path = os.path.join(input_model_untarred_path, MODEL_INFO_FILE_NAME)
    try:
        with open(input_model_info_file_path, "r") as f:
            model_info: Dict[str, Any] = json.load(f)
    except FileNotFoundError:
        logging.info(f"Info file not found at '{input_model_info_file_path}'.")
        model_info: Dict[str, Any] = {}
    except Exception as e:
        logging.error(
            f"Could not read or parse model_info file, exception: '{e}'. To continue training on previously "
            f"trained model, please create a json file {input_model_info_file_path} with"
            f" {IS_TRAINED_ON_INPUT_DATA} parameter set to True. To start training from scratch,"
            f"please create a json file {input_model_info_file_path} with"
            f" {IS_TRAINED_ON_INPUT_DATA} parameter set to False."
        )
        raise

    model_info[IS_TRAINED_ON_INPUT_DATA] = True

    output_model_info_file_path = os.path.join(model_dir, MODEL_INFO_FILE_NAME)
    with open(output_model_info_file_path, "w") as f:
        f.write(json.dumps(model_info))
