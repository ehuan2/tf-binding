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
            required=True,
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
            default=0.8,
            help="The proportion of data to use for training",
        )
        parser.add_argument(
            "--batch_size",
            type=int,
            default=32,
            help="The batch size to use for training",
        )

        # only parse the args that we know, and throw out what we don't know
        args = parser.parse_known_args()[0]

        # the set of potential keys should be defined by the config + any
        # other special ones here (such as the model args)
        config_keys = list(args.__dict__.keys())

        # first read the config file and set the current attributes to it
        # then parse through the other arguments as that's what we want use to
        # override the config file if supplied
        if args.config:
            with open(args.config, "r") as file:
                data = yaml.safe_load(file)

            for key in config_keys:
                if key in data.keys():
                    setattr(self, key, data[key])

        # now we take all the arguments we want and we copy it over!
        for key, value in args._get_kwargs():
            if value is not None:
                setattr(self, key, value)

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


def get_model_instance(config: Config) -> BaseModel:
    """
    Factory function to get the model instance based on the architecture specified in the config.
    """
    if config.architecture == "simple":
        from models.simple import SimpleModel

        return SimpleModel(config)
    else:
        raise ValueError(f"Unknown architecture: {config.architecture}")
