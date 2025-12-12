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
import numpy as np
from tqdm import tqdm


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

        self.bw_files = {}

        # let's open up all the bigwig files that are specified
        if config.pred_struct_features:
            assert (
                config.pred_struct_data_dir is not None
            ), "pred_struct_data_dir must be specified to use structural features"

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
            TFColumns.LOG_PROB.value: torch.tensor(
                interval[TFColumns.LOG_PROB.value]
            ).to(device=self.config.device, dtype=self.config.dtype),
        }

        structure_features = {}

        if self.config.use_probs:
            seq = interval[TFColumns.SEQ.value]
            strand = interval[TFColumns.STRAND.value]
            # now, depending on the strand, we may need to reverse complement
            if strand == "-":
                seq = get_negative_strand_subsequence(seq)

            pwm_scores = torch.tensor(get_ind_score(self.pwm, seq)).to(
                device=self.config.device, dtype=self.config.dtype
            )
            structure_features["pwm_scores"] = pwm_scores

        # extract the predicted features from the bigwig file
        for feature, bw_file in self.bw_files.items():
            values = bw_file.values(
                interval[TFColumns.CHROM.value],
                interval[TFColumns.START.value] - self.config.context_window,
                interval[TFColumns.END.value] + self.config.context_window,
            )

            if np.any(np.isnan(values)):
                raise RuntimeError(
                    f"""
NaN values found in feature {feature} for interval {interval}.
Please make sure that you are using the correct bigwig files with appropriate context window preprocessing.
Values: {values}
                """
                )

            structure_features[feature] = torch.tensor(values).to(
                device=self.config.device, dtype=self.config.dtype
            )

        return {
            "interval": interval_dict,
            "structure_features": structure_features,
            "label": torch.tensor(self.is_tf_site).to(
                device=self.config.device, dtype=self.config.dtype
            ),
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

    tf_len = len(pos_df.iloc[0][TFColumns.SEQ.value]) + 2 * config.context_window

    # now let's create the positive and negative training/testing splits
    pos_dataset = IntervalDataset(pos_df, is_tf_site=1, config=config)
    neg_dataset = IntervalDataset(neg_df, is_tf_site=0, config=config)

    train_size_pos = int(config.train_split * len(pos_dataset))
    test_size_pos = len(pos_dataset) - train_size_pos
    train_size_neg = int(config.train_split * len(neg_dataset))
    test_size_neg = len(neg_dataset) - train_size_neg

    # then split based on train and test
    pos_train, pos_test = random_split(pos_dataset, [train_size_pos, test_size_pos])
    neg_train, neg_test = random_split(neg_dataset, [train_size_neg, test_size_neg])

    # finally split on train and validation
    val_size_pos = int(config.validation_split * len(pos_train))
    train_size_pos = len(pos_train) - val_size_pos
    val_size_neg = int(config.validation_split * len(neg_train))
    train_size_neg = len(neg_train) - val_size_neg

    pos_train, pos_val = random_split(pos_train, [train_size_pos, val_size_pos])
    neg_train, neg_val = random_split(neg_train, [train_size_neg, val_size_neg])

    train_dataset = ConcatDataset([pos_train, neg_train])
    test_dataset = ConcatDataset([pos_test, neg_test])
    val_dataset = ConcatDataset([pos_val, neg_val])

    return train_dataset, test_dataset, val_dataset, tf_len


def one_hot_encode(seq: str) -> np.ndarray:
    """
    One-hot encode a DNA sequence.
    Returns a (4, L) array with rows [A, C, G, T].
    Ambiguous bases (e.g. N) get all zeros at that position.
    """
    NUC_TO_IDX = {"A": 0, "C": 1, "G": 2, "T": 3}
    seq = seq.upper()
    L = len(seq)
    arr = np.zeros((4, L), dtype=np.float32)
    for i, base in enumerate(seq):
        idx = NUC_TO_IDX.get(base)
        if idx is not None:
            arr[idx, i] = 1.0
    return arr


def batch_to_scikit(config, item):
    # --- sequence (optional for ablations) ---
    seq_vec = np.array([], dtype=np.float32)
    if config.use_seq:
        seq = item["interval"][TFColumns.SEQ.value]
        seq_vec = one_hot_encode(seq).reshape(-1)  # (4*L,)

    # --- Add in the score as well ---
    score = [item["interval"][TFColumns.LOG_PROB.value].detach().cpu().numpy()]

    # --- Add in the probability vector features if applicable ---
    if config.use_probs:
        pwm_scores = item["structure_features"]["pwm_scores"]
        pwm_arr = pwm_scores.detach().cpu().numpy().astype(np.float32)
        score.extend(pwm_arr.tolist())

    score = np.array(score, dtype=np.float32)

    # --- structure features (from DNAshape + PWM/etc.) ---
    struct_feats = []
    for name, tensor_val in item["structure_features"].items():
        if name == "pwm_scores":
            continue  # already added in score part
        arr = tensor_val.detach().cpu().numpy().astype(np.float32)
        struct_feats.append(arr)

    if struct_feats:
        struct_vec = np.concatenate(struct_feats)
    else:
        struct_vec = np.array([], dtype=np.float32)

    # --- concatenate only the non-empty parts ---
    parts = []
    if seq_vec.size > 0:
        parts.append(seq_vec)
    if struct_vec.size > 0:
        parts.append(struct_vec)
    parts.append(score)

    X = np.concatenate(parts).astype(np.float32)
    y = float(item["label"].item())
    return X, y


def dataset_to_scikit(config, dataset):
    """
    Convert an entire dataset to X, y arrays for scikit-learn.
    """
    X_list, y_list = [], []
    for i in tqdm(range(len(dataset))):
        X_i, y_i = batch_to_scikit(config, dataset[i])
        X_list.append(X_i)
        y_list.append(y_i)

    X = np.vstack(X_list)
    y = np.array(y_list)
    print(f"Transformed data into the following shape: {X.shape, y.shape}")
    return X, y
