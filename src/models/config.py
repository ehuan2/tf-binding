"""config.py.

This module provides a config class to be used for both the parser as well as
for providing the model specific classes a way to access the parsed arguments.
"""
import argparse
import os
from enum import Enum

from models.base import BaseModel
import yaml


class ModelSelection(str, Enum):
    """Enum that contains all possible model choices."""

    SIMPLE = "simple"


class PredStructFeature(str, Enum):
    """Enum that contains all possible predicted structure features."""

    MGW = "MGW"
    HelT = "HelT"
    ProT = "ProT"
    Roll = "Roll"
    OC2 = "OC2"


class Config:
    """Config class for both yaml and cli arguments."""

    def __init__(self):
        """Verifies the passed arguments while populating config fields.

        Args:
            architecture (Optional[str]): The model architecture to use.

        Raises:
            ValueError: If any required argument is missing.
        """
        # Initiate parser and parse arguments
        parser = argparse.ArgumentParser()
        parser.add_argument("-c", "--config", type=str, help="")

        model_sel = [model.value for model in list(ModelSelection)]
        parser.add_argument(
            "-a",
            "--architecture",
            type=ModelSelection,
            choices=list(ModelSelection),
            metavar=f"{model_sel}",
            default=None,
            help="The model architecture family to extract the embeddings from",
        )
        parser.add_argument(
            "--tf",
            type=str,
            help="The transcription factor to use for training and evaluation",
        )
        parser.add_argument(
            "--preprocess_data_dir",
            type=str,
            default="data/tf_sites",
            help="The directory containing preprocessed data",
        )
        parser.add_argument(
            "--train_split",
            type=float,
            help="The proportion of data to use for training",
        )
        parser.add_argument(
            "--batch_size",
            type=int,
            help="The batch size to use for training",
        )

        # ---- PWM features ----
        parser.add_argument(
            "--pwm_file",
            type=str,
            help="The file name of the probability weight matrix (PWM) file",
        )
        parser.add_argument(
            "--use_probs",
            action="store_true",
            default=None,
            help="Whether to use the probability vector of the sequence in the model",
        )

        # ---- Predicted Structure files ----
        parser.add_argument(
            "--pred_struct_data_dir",
            type=str,
            help="The directory containing DNA predicted structure data as bigWig files (MGW, ProT, Roll/OC2, HelT)",
        )
        parser.add_argument(
            "--pred_struct_features",
            nargs="+",
            type=PredStructFeature,
            choices=list(PredStructFeature),
            metavar=f"{[feature.value for feature in list(PredStructFeature)]}",
            default=None,
            help="List of predicted structure features to use (e.g., MGW, HelT, ProT, Roll, OC2)",
        )
        parser.add_argument(
            "--use_probs",
            action="store_true",
            default=None,
            help="Whether to use the probability vector of the sequence in the model",
        )

        # then we finally add on arguments for each structure feature
        # so others can specify paths if they want
        for feature in PredStructFeature:
            parser.add_argument(
                f"--{feature.value.lower()}_file_name",
                type=str,
                help=f"The file name of the {feature.value} predicted structure file",
            )

        # only parse the args that we know, and throw out what we don't know
        args = parser.parse_known_args()[0]

        # the set of potential keys should be defined by the config + any
        # other special ones here (such as the model args)
        config_keys = list(args.__dict__.keys())
        print(config_keys)

        # first read the config file and set the current attributes to it
        # then parse through the other arguments as that's what we want use to
        # override the config file if supplied
        if args.config:
            with open(args.config, "r") as file:
                data = yaml.safe_load(file)

            print(data.keys())
            for key in config_keys:
                if key in data.keys():
                    setattr(self, key, data[key])
            # for key,value in data.items():
            #     setattr(self, key, value)

        # now we take all the arguments we want and we copy it over!
        for key, value in args._get_kwargs():
            if value is not None:
                setattr(self, key, value)

        print("Loaded preprocess_data_dir:", self.preprocess_data_dir)

        # require that the architecture and data path must exist
        assert all(
            hasattr(self, attr) and getattr(self, attr) is not None
            for attr in ("architecture", "preprocess_data_dir", "pred_struct_data_dir")
        ), (
            "Fields `architecture`, `preprocess_data_dir`, and `pred_struct_data_dir` in yaml config must exist, "
            "otherwise, --architecture, --preprocess_data_dir, and --pred_struct_data_dir must be set"
        )

        # change the architecture type to an enum
        if not isinstance(self.architecture, ModelSelection):
            assert self.architecture in model_sel, (
                f"Architecture {self.architecture} not supported, "
                f"use one of {model_sel}"
            )
            self.architecture = ModelSelection(self.architecture)

        assert os.path.exists(
            self.preprocess_data_dir
        ), f"Preprocessed data directory {self.preprocess_data_dir} does not exist"

        assert os.path.exists(
            self.pred_struct_data_dir
        ), f"DNA predicted structure data directory {self.pred_struct_data_dir} does not exist"

        assert self.tf is not None, "Transcription factor (--tf) must be specified"

        # now let's rewrite the config's keys that are not defined (boolean):
        for key in ["use_probs"]:
            if not hasattr(self, key) or getattr(self, key) is None:
                setattr(self, key, False)

        # now we set the defaults
        defaults = {
            "batch_size": 32,
            "train_split": 0.8,
            "pwm_file": "data/factorbookMotifPwm.txt",
            "pred_struct_features": [],

            # SVM Defaults
            "window_size": 101,
            "svm_kernal": "linear",
            "svm_c": 1.0,
            "svm_gamma": "scale"
        }

        for feature in PredStructFeature:
            defaults[
                f"{feature.value.lower()}_file_name"
            ] = f"hg19.{feature.value}.wig.bw"

        for key, value in defaults.items():
            if not hasattr(self, key) or getattr(self, key) is None:
                setattr(self, key, value)

        # now let's check that the predicted structure features to be used
        # are valid
        for feature in self.pred_struct_features:
            assert feature in list(PredStructFeature), (
                f"Predicted structure feature {feature} not supported, "
                f"use one of {[f.value for f in list(PredStructFeature)]}"
            )


def get_model_instance(config: Config) -> BaseModel:
    """
    Factory function to get the model instance based on the architecture specified in the config.
    """
    if config.architecture == "simple":
        from models.simple import SimpleModel

        return SimpleModel(config)
    
    elif config.architecture == ModelSelection.SVM:
        from models.svm import SVMModel
        return SVMModel(config)
    
    else:
        raise ValueError(f"Unknown architecture: {config.architecture}")
