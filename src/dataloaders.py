"""
dataloaders.py

Custom dataset classes for loading transcription factor binding site data.
"""

from helpers import (
    read_samples,
    TFColumns,
    one_hot_encode,
)
from models.config import (
    Config,
    ModelSelection,
)
from preprocess.preprocess import (
    get_negative_strand_subsequence,
    get_pwm,
    get_ind_score,
)

import torch
from torch.utils.data import Dataset, random_split, ConcatDataset
import os
import pyBigWig
import numpy as np


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
        if config.pred_struct_features:
            assert (
                config.pred_struct_data_dir is not None
            ), "pred_struct_data_dir must be specified to use structural features"

            self.bw_files = {}
            for feature in config.pred_struct_features:
                bigwig_path = os.path.join(
                    config.pred_struct_data_dir,
                    getattr(config, f"{feature.lower()}_file_name"),
                )
                assert os.path.exists(
                    bigwig_path
                ), f"BigWig file for {feature} not found: {bigwig_path}"

                self.bw_files[feature] = pyBigWig.open(bigwig_path)

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

        # extract the predicted features from the bigwig file
        for feature, bw_file in self.bw_files.items():
            values = bw_file.values(
                interval[TFColumns.CHROM.value],
                interval[TFColumns.START.value],
                interval[TFColumns.END.value],
            )
            structure_features[feature] = torch.tensor(values, dtype=torch.float32)

        return {
            "interval": interval_dict,
            "structure_features": structure_features,
            "label": torch.tensor(self.is_tf_site, dtype=torch.float32),
        }

class SVMDataset:
    """
    Dataset that prepares sequence + GBshape features for SVM classification.
    """

    def __init__(self, pr, is_tf_site, config):
        self.pr = pr
        self.label = is_tf_site
        self.config = config

        # directory with all 4 GBshape files
        shape_dir = config.pred_struct_data_dir

        # Load shape tracks (mgw, prot, roll, helt)
        self.shape_tracks = {}
        for shape_name in ["mgw", "prot", "roll", "helt"]:
            fname = getattr(config, f"{shape_name}_file_name", None)
            if fname is None:
                continue

            path = os.path.join(shape_dir, fname)
            if shape_name == 'prot':
                path = '/home/mcb/users/cclark6/comp561/hg19.ProT.wig.bw'
            if not os.path.exists(path):
                raise FileNotFoundError(f"Struct file not found: {path}")

            self.shape_tracks[shape_name] = pyBigWig.open(path)

        self.window = config.window_size

    def __len__(self):
        return len(self.pr)

    def __getitem__(self, idx):
        row = self.pr.iloc[idx]

        chrom = row[TFColumns.CHROM.value]
        start = int(row[TFColumns.START.value])
        end   = int(row[TFColumns.END.value])
        strand = row[TFColumns.STRAND.value]

        seq = row[TFColumns.SEQ.value].upper()

        if strand == "-":
            seq = get_negative_strand_subsequence(seq)

        seq_len = len(seq)

        if self.window is not None and self.window < seq_len:
            center = seq_len // 2
            half = self.window // 2
            seq = seq[center - half : center + half + 1]

            win_start = start + (center - half)
            win_end   = start + (center + half + 1)
        else:
            win_start = start
            win_end   = end

        seq_encoded = one_hot_encode(seq).flatten()

        shape_features = []

        for name, bw in self.shape_tracks.items():
            vals = bw.values(chrom, win_start, win_end)
            vals = np.nan_to_num(vals)

            if len(vals) != len(seq):
                # pad or truncate if shape doesnt match
                if len(vals) < len(seq):
                    pad_len = len(seq) - len(vals)
                    vals = np.concatenate([vals, np.zeros(pad_len)])
                else:
                    vals = vals[:len(seq)]

            shape_features.append(vals)

        if len(shape_features) > 0:
            shape_features = np.concatenate(shape_features)
        else:
            shape_features = np.array([], dtype=np.float32)
        
        # final svm input
        X = np.concatenate([seq_encoded, shape_features]).astype(np.float32)
        y = self.label

        # ---- DEBUGGING ----
        if idx < 1 and self.config.debug:
            print("----- DEBUG SAMPLE -----")
            print("Seq:", seq)
            print("Seq encoded shape:", seq_encoded.shape)
            print("Shape features:", {k: v.shape if hasattr(v, 'shape') else len(v) 
                                    for k, v in self.shape_tracks.items()})
            print("X shape:", X.shape)
            print("label:", y)
            print("-------------------------")

        return X, y

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
    if config.architecture == ModelSelection.SIMPLE:
        DatasetClass = IntervalDataset
    elif config.architecture == ModelSelection.SVM:
        DatasetClass = SVMDataset
    else:
        DatasetClass = None

    assert DatasetClass is not None, f'Dataset Class {DatasetClass} not implemented.'

    # ----------- DEBUGGING -----------
    print(f"[DEBUG] Using dataset class: {DatasetClass.__name__}")


    pos_dataset = DatasetClass(pos_df, is_tf_site=1, config=config)
    neg_dataset = DatasetClass(neg_df, is_tf_site=0, config=config)

    train_size_pos = int(config.train_split * len(pos_dataset))
    test_size_pos = len(pos_dataset) - train_size_pos
    train_size_neg = int(config.train_split * len(neg_dataset))
    test_size_neg = len(neg_dataset) - train_size_neg

    pos_train, pos_test = random_split(pos_dataset, [train_size_pos, test_size_pos])
    neg_train, neg_test = random_split(neg_dataset, [train_size_neg, test_size_neg])

    train_dataset = ConcatDataset([pos_train, neg_train])
    test_dataset = ConcatDataset([pos_test, neg_test])

    return train_dataset, test_dataset
