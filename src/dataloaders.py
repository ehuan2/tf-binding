"""
dataloaders.py

Custom dataset classes for loading transcription factor binding site data.
"""

from helpers import (
    read_positive_samples,
    read_negative_samples,
    get_subsequence,
    TFColumns,
)
from models.config import Config

from torch.utils.data import Dataset, random_split, ConcatDataset
import os


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

    def __len__(self):
        return len(self.pr)

    def __getitem__(self, idx):
        interval = self.pr.iloc[idx]
        chrom, start, end = (
            interval[TFColumns.CHROM.value],
            interval[TFColumns.START.value],
            interval[TFColumns.END.value],
        )

        # also return the specified numbers depending on config, such as the sequence
        structure_features = {}
        if self.config.use_seq:
            # read from the genome file to get the sequence
            structure_features["sequence"] = get_subsequence(
                fasta_file=os.path.join(self.config.fasta_data_dir, f"{chrom}.fa"),
                start=start,
                end=end,
            )

        return (chrom, start, end), structure_features, self.is_tf_site


def get_data_splits(config: Config):
    """
    Factory function to get the dataset instance based on the config.
    """
    tf_dir = os.path.join(config.preprocess_data_dir, config.tf)
    pos_file = os.path.join(tf_dir, "positive_examples.txt")
    neg_file = os.path.join(tf_dir, "negative_examples.txt")

    assert os.path.exists(pos_file), f"Positive examples file not found: {pos_file}"
    assert os.path.exists(neg_file), f"Negative examples file not found: {neg_file}"

    pos_df = read_positive_samples(pos_file, include_index=False)
    neg_df = read_negative_samples(neg_file, include_chrom_index=False)

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
