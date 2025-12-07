"""
score_dist.py.

Just as with the preprocessing script, we will read through the negative and
positive examples to compute the score distributions.

Then, we will plot this and show that the score distributions are the same,
meaning that we require some other type of information to distinguish positive
and negative examples.
"""
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.neighbors import KernelDensity
import numpy as np
from preprocess_helpers import (
    read_scores,
    get_args,
    get_overlap_range,
    proportion_in_range,
    subset_scores_in_range,
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


def get_kde(scores):
    """
    Given the scores, return the KDE fit on the scores.
    """
    kde_data = np.array(scores).reshape(-1, 1)
    kde = KernelDensity(kernel="gaussian", bandwidth=0.3).fit(kde_data)
    return kde


def classify_scores(pos_densities, neg_densities, rev_neg_densities, threshold=1.0):
    """
    Classify scores based on the ratio of positive to negative and reverse negative densities.

    Returns:
        pos: number of scores classified as positive
        neg: number of scores classified as negative
    """
    pos = sum(
        1
        for i in range(len(pos_densities))
        if pos_densities[i] > threshold * neg_densities[i]
        and pos_densities[i] > threshold * rev_neg_densities[i]
    )
    neg = len(pos_densities) - pos

    return pos, neg


def get_classifier_rates(pos_scores, neg_scores, rev_neg_scores, tf_name):
    # now let's go over all the positive scores and calculate their densities
    kde = get_kde(pos_scores)
    kde_neg = get_kde(neg_scores)
    kde_rev_neg = get_kde(rev_neg_scores)

    def get_score_densities(scores):
        pos_densities = kde.score_samples(np.array(scores).reshape(-1, 1))
        neg_densities = kde_neg.score_samples(np.array(scores).reshape(-1, 1))
        rev_neg_densities = kde_rev_neg.score_samples(np.array(scores).reshape(-1, 1))
        return pos_densities, neg_densities, rev_neg_densities

    (
        pos_score_pos_densities,
        pos_score_neg_densities,
        pos_score_rev_neg_densities,
    ) = get_score_densities(pos_scores)

    (
        neg_score_pos_densities,
        neg_score_neg_densities,
        neg_score_rev_neg_densities,
    ) = get_score_densities(neg_scores)

    (
        rev_neg_score_pos_densities,
        rev_neg_score_rev_neg_densities,
        rev_neg_score_rev_rev_neg_densities,
    ) = get_score_densities(rev_neg_scores)

    def get_rates(threshold):
        """
        Get the true positive, false negative, false positive, and true negative rates
        based on the provided threshold.
        """
        tp, fn = classify_scores(
            pos_score_pos_densities,
            pos_score_neg_densities,
            pos_score_rev_neg_densities,
            threshold=threshold,
        )
        fp, tn = classify_scores(
            neg_score_pos_densities,
            neg_score_neg_densities,
            neg_score_rev_neg_densities,
            threshold=threshold,
        )
        fp_rev, tn_rev = classify_scores(
            rev_neg_score_pos_densities,
            rev_neg_score_rev_neg_densities,
            rev_neg_score_rev_rev_neg_densities,
            threshold=threshold,
        )
        return tp, fn, fp + fp_rev, tn + tn_rev

    tp, fn, fp, tn = get_rates(threshold=1.0)
    print(f"True Positives: {tp}, False Negatives: {fn}")
    print(f"False Positives: {fp}, True Negatives: {tn}")
    print(f"Total samples: {tp + fn + fp + tn}")

    # now we want to iterate over different thresholds to see how the rates change
    tprs = []
    fprs = []

    threshold = 0.1

    while len(tprs) == 0 or tprs[-1] < 1.0:
        tp, fn, fp, tn = get_rates(threshold=threshold)
        print(f"Threshold: {threshold}")
        print(f"  True Positives: {tp}, False Negatives: {fn}")
        print(f"  False Positives: {fp}, True Negatives: {tn}")
        tprs.append(tp / (tp + fn) if (tp + fn) > 0 else 0)
        fprs.append(fp / (fp + tn) if (fp + tn) > 0 else 0)
        threshold += 0.05

    # now let's plot the ROC curve
    plt.figure(figsize=(8, 6))
    plt.plot(fprs, tprs)
    plt.title("ROC Curve")
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.0])
    plt.grid()
    plt.savefig(f"./figs/roc_curve_{tf_name}.png")
    plt.close()

    # finally, calculate the AUROC
    auroc = 0.0
    for i in range(1, len(tprs)):
        # calculate the trapezoid area and add to auroc
        auroc += (fprs[i] - fprs[i - 1]) * (tprs[i] + tprs[i - 1]) / 2.0
    print(f"AUROC: {auroc}")


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

    print(
        f"Total new samples: {len(pos_subset) + len(neg_subset) + len(rev_neg_subset)}"
    )
    get_classifier_rates(pos_subset, neg_subset, rev_neg_subset, args.tf)
