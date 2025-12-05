"""
helpers.py
Helper functions and classes for preprocessing transcription factor binding site data.
"""
# change the pythonpath to the src/ directory
import os
import sys

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import argparse
from helpers import (
    TFColumns,
    read_samples,
)


def get_args():
    parser = argparse.ArgumentParser(
        description="Preprocess script for getting positive and negative examples of binding sites."
    )
    parser.add_argument(
        "--chip_seq_file",
        type=str,
        default="data/wgEncodeRegTfbsClusteredV3.GM12878.merged.bed",
        help="Path to the ChIP-seq data file.",
    )
    parser.add_argument(
        "--true_tf_file",
        type=str,
        default="data/factorbookMotifPos.txt",
        help="Path to the file containing true transcription factor binding sites.",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="data/tf_sites",
        help="Directory to save the output files.",
    )
    parser.add_argument(
        "--fasta_data_dir",
        type=str,
        default="data/fasta",
        help="Directory containing fasta files.",
    )
    parser.add_argument(
        "--tf",
        type=str,
        default=None,
        help="Specific transcription factor to process (default: all).",
    )
    parser.add_argument(
        "--pwm_file",
        type=str,
        default="data/factorbookMotifPwm.txt",
        help="The probability weight matrix file to read from for negative sequence processing.",
    )
    parser.add_argument(
        "--bigwig_dir",
        type=str,
        help="Directory containing bigwig files for genomic features.",
    )
    parser.add_argument(
        "--bigwigs",
        nargs="+",
        help="List of bigwig filenames to use for genomic features.",
    )
    return parser.parse_args()


def read_scores(args):
    pos_seqs_file = os.path.join(args.output_dir, args.tf, "positive", "sequences.txt")
    neg_intervals_file = os.path.join(
        args.output_dir, args.tf, "negative", "best_negative_sequences.txt"
    )
    rev_neg_intervals_file = os.path.join(
        args.output_dir, args.tf, "negative", "reverse_best_negative_sequences.txt"
    )

    pos_names = [
        TFColumns.CHROM.value,
        TFColumns.START.value,
        TFColumns.END.value,
        TFColumns.STRAND.value,
        TFColumns.SEQ.value,
        TFColumns.LOG_PROB.value,
    ]
    positive_pr = read_samples(pos_seqs_file, names=pos_names)
    positive_scores = positive_pr[TFColumns.LOG_PROB.value].tolist()

    neg_names = [
        TFColumns.CHROM.value,
        TFColumns.START.value,
        TFColumns.END.value,
        TFColumns.SEQ.value,
        TFColumns.LOG_PROB.value,
    ]
    neg_pr = read_samples(neg_intervals_file, names=neg_names)
    rev_neg_pr = read_samples(rev_neg_intervals_file, names=neg_names)

    negative_scores = neg_pr[TFColumns.LOG_PROB.value].tolist()
    rev_negative_scores = rev_neg_pr[TFColumns.LOG_PROB.value].tolist()

    return positive_scores, negative_scores, rev_negative_scores


def get_overlap_range(pos_scores, neg_scores, rev_neg_scores):
    """
    Get the overlapping score range between positive and negative samples.
    """

    def get_range(scores):
        """
        Get the min and max range of the scores.
        """
        return min(scores), max(scores)

    # now let's filter the scores based on the ranges
    pos_range = get_range(pos_scores)
    neg_range = get_range(neg_scores)
    rev_neg_range = get_range(rev_neg_scores)

    # now let's get the overlap, i.e. the lower bound is the min of the positive scores,
    # the max is the max of the negative/rev negative scores
    overall_min = pos_range[0]
    overall_max = max(neg_range[1], rev_neg_range[1])
    return overall_min, overall_max


def proportion_in_range(scores, overall_min, overall_max):
    """
    Calculate the proportion of scores that fall within the specified range.
    """
    count_in_range = sum(1 for score in scores if overall_min <= score <= overall_max)
    return count_in_range / len(scores) if scores else 0


def subset_scores_in_range(scores, overall_min, overall_max):
    """
    Subset the scores to only those within the specified range.
    """
    return [score for score in scores if overall_min <= score <= overall_max]
