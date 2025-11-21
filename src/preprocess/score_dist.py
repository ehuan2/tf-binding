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
    positive_scores, negative_scores, rev_negative_scores, tf_name, subset=False
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

    save_png = "./figs/score_distributions_" + tf_name
    if subset:
        save_png += "_subset"

    plt.savefig(save_png + ".png")


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


if __name__ == "__main__":
    args = get_args()

    # first let's plot just the top scores of the negatives/rev strands
    # and the positives as well
    positive_scores, negative_scores, rev_negative_scores = read_scores(args)
    plot_score_distributions(
        positive_scores, negative_scores, rev_negative_scores, args.tf
    )

    overall_min, overall_max = get_overlap_range(
        positive_scores, negative_scores, rev_negative_scores
    )
    print(f"Overall score range: {overall_min} to {overall_max}")

    print(
        "Proportion of positive scores in range:",
        proportion_in_range(positive_scores, overall_min, overall_max),
    )

    print(
        "Proportion of negative scores in range:",
        proportion_in_range(negative_scores, overall_min, overall_max),
    )

    print(
        "Proportion of reverse negative scores in range:",
        proportion_in_range(rev_negative_scores, overall_min, overall_max),
    )

    pos_subset = subset_scores_in_range(positive_scores, overall_min, overall_max)
    neg_subset = subset_scores_in_range(negative_scores, overall_min, overall_max)
    rev_neg_subset = subset_scores_in_range(
        rev_negative_scores, overall_min, overall_max
    )

    plot_score_distributions(
        pos_subset, neg_subset, rev_neg_subset, args.tf, subset=True
    )
