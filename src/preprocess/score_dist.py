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

from sklearn.metrics import (
    roc_curve,
    roc_auc_score,
    precision_recall_curve,
    auc,
    f1_score,
    confusion_matrix,
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

    """
    Quick estimates for TF PAX5 at threshold -15.5
    print(f'False Neg. Estimate: {sum(np.array(pos_scores) <= -15.5)}, {len(pos_scores)}')
    print(f'True Pos. Estimate: {sum(np.array(pos_scores) >= -15.5)}, {len(pos_scores)}')

    print(f'True Neg. Estimate: {sum(np.array(neg_scores) <= -15.5) + sum(np.array(rev_neg_scores) <= -15.5)}, {len(neg_scores)}')
    print(f'False Pos. Estimate: {sum(np.array(neg_scores) >= -15.5) + sum(np.array(rev_neg_scores) >= -15.5)}, {len(neg_scores)}')
    """

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
        rev_neg_score_neg_densities,
        rev_neg_score_rev_neg_densities,
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
            rev_neg_score_neg_densities,
            rev_neg_score_rev_neg_densities,
            threshold=threshold,
        )
        return tp, fn, fp + fp_rev, tn + tn_rev

    tp, fn, fp, tn = get_rates(threshold=1.0)
    print(f"True Positives: {tp}, False Negatives: {fn}")
    print(f"False Positives: {fp}, True Negatives: {tn}")
    print(f"Total samples: {tp + fn + fp + tn}")

    # now we calculate this using scikit learn instead
    # First, let's create the scores for each sample instead -- combining them together
    scores = []
    for i in range(len(pos_scores)):
        pos_score = pos_score_pos_densities[i]
        neg_score = pos_score_neg_densities[i]
        rev_neg_score = pos_score_rev_neg_densities[i]
        scores.append(pos_score - max(neg_score, rev_neg_score))

    for i in range(len(neg_scores)):
        pos_score = neg_score_pos_densities[i]
        neg_score = neg_score_neg_densities[i]
        rev_neg_score = neg_score_rev_neg_densities[i]
        scores.append(pos_score - max(neg_score, rev_neg_score))

    for i in range(len(rev_neg_scores)):
        pos_score = rev_neg_score_pos_densities[i]
        neg_score = rev_neg_score_neg_densities[i]
        rev_neg_score = rev_neg_score_rev_neg_densities[i]
        scores.append(pos_score - max(neg_score, rev_neg_score))

    predictions = (np.array(scores) >= 0.0).astype(float)
    labels = np.array(
        [1] * len(pos_scores) + [0] * (len(neg_scores) + len(rev_neg_scores))
    )

    f1 = f1_score(labels, predictions)
    cm = confusion_matrix(labels, predictions)
    tn, fp, fn, tp = cm.ravel()
    print("Using sklearn metrics:")
    print(f"F1 Score: {f1}")
    print(f"True Positives: {tp}, False Negatives: {fn}")
    print(f"False Positives: {fp}, True Negatives: {tn}")
    print(f"Total samples: {tp + fn + fp + tn}")

    fpr, tpr, _ = roc_curve(labels, scores)
    roc_auc = roc_auc_score(labels, scores)
    precision, recall, _ = precision_recall_curve(labels, scores)
    pr_auc = auc(recall, precision)

    # now we plot the ROC and PR curves
    plt.figure()
    plt.plot(fpr, tpr, label=f"ROC curve (area = {roc_auc:.2f})")
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title("Receiver Operating Characteristic (ROC) Curve")
    plt.legend(loc="lower right")
    plt.tight_layout()
    plt.savefig(f"./figs/kde_classifier_roc_curve_{tf_name}.png")
    plt.close()

    # PR curve
    plt.figure()
    plt.plot(recall, precision, label=f"PR curve (area = {pr_auc:.2f})")
    plt.xlabel("Recall")
    plt.ylabel("Precision")
    plt.title("Precision-Recall (PR) Curve")
    plt.legend(loc="lower left")
    plt.tight_layout()
    plt.savefig(f"./figs/kde_classifier_pr_curve_{tf_name}.png")
    plt.close()


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
