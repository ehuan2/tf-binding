"""
score_dist.py.

Just as with the preprocessing script, we will read through the negative and
positive examples to compute the score distributions.

Then, we will plot this and show that the score distributions are the same,
meaning that we require some other type of information to distinguish positive
and negative examples.
"""
from preprocess import get_args
import matplotlib.pyplot as plt
import seaborn as sns
import os
from helpers import (
    TFColumns,
    read_samples,
)


def plot_score_distributions(
    positive_scores, negative_scores, rev_negative_scores, tf_name
):
    """
    Plot the score distributions for positive, negative, and reverse negative samples.
    """
    plt.figure(figsize=(10, 6))

    sns.kdeplot(
        positive_scores, color="blue", label="Positive Samples", fill=True, alpha=0.5
    )
    sns.kdeplot(
        negative_scores, color="red", label="Negative Samples", fill=True, alpha=0.5
    )
    sns.kdeplot(
        rev_negative_scores,
        color="green",
        label="Reverse Negative Samples",
        fill=True,
        alpha=0.5,
    )
    plt.title(f"Score Distributions for {tf_name}")
    plt.xlabel("Score")
    plt.ylabel("Density")
    plt.legend()
    plt.savefig("./figs/score_distributions_" + tf_name + ".png")


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


if __name__ == "__main__":
    args = get_args()

    # first let's plot just the top scores of the negatives/rev strands
    # and the positives as well
    positive_scores, negative_scores, rev_negative_scores = read_scores(args)
    plot_score_distributions(
        positive_scores, negative_scores, rev_negative_scores, args.tf
    )
