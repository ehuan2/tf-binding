"""
dataloaders.py

Custom dataset classes for loading transcription factor binding site data.
"""

from helpers import (
    read_samples,
    TFColumns,
)
from models.config import Config
from preprocess.preprocess import (
    get_negative_strand_subsequence,
    get_pwm,
    get_ind_score,
)

import torch
from torch.utils.data import Dataset, random_split, ConcatDataset
import os
import pyBigWig


class IntervalDataset(Dataset):
    """
    Dataset class for loading intervals of transcription factor binding sites.
    Each interval is represented by its chromosomes, start, and end positions,
    as well as its labels (positive or negative).
    """

    def __init__(self, pr, is_tf_site, config: Config):
        """
        Args:
            pr (pr.PyRanges): Pyranges containing the intervals.
            is_tf_site (int): Label indicating if the interval is a transcription factor binding site (1) or not (0).
        """
        self.pr = pr
        self.is_tf_site = is_tf_site
        self.config = config

        # let's open up all the bigwig files that are specified
        if config.use_mgws:
            assert (
                config.pred_struct_data_dir is not None
            ), "pred_struct_data_dir must be specified to use mgw features"
            mgw_bigwig_path = os.path.join(
                config.pred_struct_data_dir,
                config.mgw_file_name,
            )
            self.bw_file = pyBigWig.open(mgw_bigwig_path)

        if self.config.use_probs:
            assert (
                config.pwm_file is not None
            ), "pwm_file must be specified to use probability vector features"
            # read in the PWM file
            self.pwm = get_pwm(config.pwm_file, config.tf)

    def __len__(self):
        return len(self.pr)

    def __getitem__(self, idx):
        interval = self.pr.iloc[idx]

        interval_dict = {
            TFColumns.SEQ.value: interval[TFColumns.SEQ.value],
            TFColumns.LOG_PROB.value: interval[TFColumns.LOG_PROB.value],
        }

        structure_features = {}

        if self.config.use_probs:
            seq = interval[TFColumns.SEQ.value]
            strand = interval[TFColumns.STRAND.value]
            # now, depending on the strand, we may need to reverse complement
            if strand == "-":
                seq = get_negative_strand_subsequence(seq)

            pwm_scores = torch.tensor(get_ind_score(self.pwm, seq), dtype=torch.float32)
            structure_features["pwm_scores"] = pwm_scores

        if self.config.use_mgws:
            # extract the mgw features from the bigwig file
            mgw_values = self.bw_file.values(
                interval[TFColumns.CHROM.value],
                interval[TFColumns.START.value],
                interval[TFColumns.END.value],
            )
            structure_features["mgw"] = torch.tensor(mgw_values, dtype=torch.float32)

        return {
            "interval": interval_dict,
            "structure_features": structure_features,
            "label": torch.tensor(self.is_tf_site, dtype=torch.float32),
        }


def get_data_splits(config: Config):
    """
    Factory function to get the dataset instance based on the config.
    """
    tf_dir = os.path.join(config.preprocess_data_dir, config.tf)
    pos_file = os.path.join(tf_dir, "positive", "overlap.txt")
    neg_file = os.path.join(tf_dir, "negative", "overlap.txt")

    assert os.path.exists(pos_file), f"Positive examples file not found: {pos_file}"
    assert os.path.exists(neg_file), f"Negative examples file not found: {neg_file}"

    columns = [
        TFColumns.CHROM.value,
        TFColumns.START.value,
        TFColumns.END.value,
        TFColumns.STRAND.value,
        TFColumns.SEQ.value,
        TFColumns.LOG_PROB.value,
    ]

    pos_df = read_samples(pos_file, names=columns)
    neg_df = read_samples(neg_file, names=columns)

    # now let's create the positive and negative training/testing splits
    pos_dataset = IntervalDataset(pos_df, is_tf_site=1, config=config)
    neg_dataset = IntervalDataset(neg_df, is_tf_site=0, config=config)

    train_size_pos = int(config.train_split * len(pos_dataset))
    test_size_pos = len(pos_dataset) - train_size_pos
    train_size_neg = int(config.train_split * len(neg_dataset))
    test_size_neg = len(neg_dataset) - train_size_neg

    pos_train, pos_test = random_split(pos_dataset, [train_size_pos, test_size_pos])
    neg_train, neg_test = random_split(neg_dataset, [train_size_neg, test_size_neg])

    train_dataset = ConcatDataset([pos_train, neg_train])
    test_dataset = ConcatDataset([pos_test, neg_test])

    return train_dataset, test_dataset
