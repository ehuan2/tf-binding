"""
preprocess.py.

Preprocess script for getting positive and negative examples
of binding sites, having specified the active regulatory regions
and the true regions of binding sites.

Outputs to data/tf_sites/ by default, where there is a folder per each TF.
"""
import os
import sys

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from tqdm import tqdm
from preprocess_helpers import get_args, read_scores, get_overlap_range
from helpers import (
    TFColumns,
    read_negative_samples,
    read_positive_samples,
    read_samples,
)
from Bio import SeqIO
import numpy as np
import pyBigWig


def generate_positive_examples(true_tf_file, output_dir, specific_tf):
    """
    Generate positive examples from the true TF binding sites file.
    """
    pos_tf_sites = read_positive_samples(true_tf_file)

    if specific_tf:
        tf_names = [specific_tf]
    else:
        tf_names = pos_tf_sites[TFColumns.TF_NAME.value].unique().tolist()

    for tf_name in tqdm(tf_names):
        dir_path = os.path.join(output_dir, f"{tf_name}/positive")

        if os.path.exists(f"{dir_path}/intervals.txt"):
            continue

        # now we select for this TF
        tf_sites = pos_tf_sites[pos_tf_sites[TFColumns.TF_NAME.value] == tf_name]

        os.makedirs(dir_path, exist_ok=True)

        with open(f"{dir_path}/intervals.txt", "w") as out_f:
            # let's iterate through the chromosomes one by one
            for chrom in tf_sites.chromosomes:
                chrom_pos_sites = tf_sites[tf_sites[TFColumns.CHROM.value] == chrom]

                for _, site in chrom_pos_sites.iterrows():
                    chrom = site[TFColumns.CHROM.value]
                    start = site[TFColumns.START.value]
                    end = site[TFColumns.END.value]
                    score = site[TFColumns.SCORE.value]
                    strand = site[TFColumns.STRAND.value]
                    out_f.write(f"{chrom}\t{start}\t{end}\t{score}\t{strand}\n")

    print(f"Generated positive examples for {len(tf_names)} transcription factors.")
    return pos_tf_sites


def generate_negative_examples(pos_tf_sites, chip_seq_file, output_dir, specific_tf):
    """
    Generate negative examples from the ChIP-seq data file.
    """
    chip_seq_sites = read_negative_samples(chip_seq_file)
    if specific_tf:
        tf_names = [specific_tf]
    else:
        tf_names = pos_tf_sites[TFColumns.TF_NAME.value].unique().tolist()

    for tf_name in tqdm(tf_names):
        dir_path = os.path.join(output_dir, f"{tf_name}/negative")

        if os.path.exists(f"{dir_path}/intervals.txt"):
            continue

        # now we select for this TF
        tf_pos_sites = pos_tf_sites[pos_tf_sites[TFColumns.TF_NAME.value] == tf_name]

        # find all the sites in chip-seq that don't overlap
        # with some positive site
        negative_sites = chip_seq_sites.overlap(
            tf_pos_sites,
            invert=True,
        )

        os.makedirs(dir_path, exist_ok=True)

        with open(f"{dir_path}/intervals.txt", "w") as out_f:
            # now let's iterate through the chromosomes one by one
            for chrom in negative_sites.chromosomes:
                chrom_neg_sites = negative_sites[
                    negative_sites[TFColumns.CHROM.value] == chrom
                ]
                for _, site in chrom_neg_sites.iterrows():
                    chrom = site.Chromosome
                    start = site.Start
                    end = site.End
                    out_f.write(f"{chrom}\t{start}\t{end}\n")

    print(f"Generated negative examples for {len(tf_names)} transcription factors.")


def preprocess_range(output_dir, tf_name, output_name, is_pos, handle_seq_fn):
    """
    Given the output directory and TF name, output whatever is specified from the handle_seq_fn
    """
    tf_output_dir = os.path.join(
        output_dir, f'{tf_name}/{"positive" if is_pos else "negative"}'
    )

    tf_file = os.path.join(tf_output_dir, "intervals.txt")
    assert os.path.exists(tf_file), f"TF file does not exist: {tf_file}"

    pos_tf_sites = read_positive_samples(tf_file, include_index=False)

    if os.path.exists(os.path.join(tf_output_dir, f"{output_name}.txt")):
        return

    with open(os.path.join(tf_output_dir, f"{output_name}.txt"), "w") as out_f:
        for chrom in pos_tf_sites.chromosomes:
            chrom_pos_sites = pos_tf_sites[pos_tf_sites[TFColumns.CHROM.value] == chrom]
            fasta_file = os.path.join(args.fasta_data_dir, f"{chrom}.fa")
            assert os.path.exists(
                fasta_file
            ), f"Fasta file does not exist: {fasta_file}"

            # read the fasta file
            record = SeqIO.read(fasta_file, "fasta")

            for _, site in chrom_pos_sites.iterrows():
                subsequence = handle_seq_fn(site, record)
                out_f.write(f"{subsequence}\n")


def get_pwm(pwm_file, tf_name):
    """
    Given a PWM file, and a specified TF,
    return the probability weight matrix as a numpy array.
    """
    # assume that these files exist
    # first we should get the probability weight matrix
    with open(pwm_file, "r") as pwm_f:
        for line in pwm_f:
            # parse the pwm file here
            vals = line.strip().split()
            if tf_name not in vals:
                continue

            tf_len = int(vals[1])

            pwm = np.zeros((4, tf_len))
            for i in range(4):
                nuc_probs = vals[2 + i].split(",")[:-1]
                pwm[i, :] = np.array([float(x) for x in nuc_probs])
            break
    return pwm


def get_ind_score(pwm, seq):
    """
    Given a PWM and a sequence, return the individual scores of the sequence.
    """
    scores = np.zeros(len(seq))
    for i in range(len(seq)):
        nucleotide = seq[i]
        nucleotides = ["A", "C", "G", "T"]
        if nucleotide not in nucleotides:
            scores[i] = np.log(1e-6)  # unknown nucleotide
            continue
        index = nucleotides.index(nucleotide)
        scores[i] = np.log(pwm[index, i] + 1e-6)  # avoid log(0)
    return scores


def score_seq(pwm, seq):
    """
    Given a PWM and a sequence, return the score of the sequence.
    """
    score = 0.0
    for j, nucleotide in enumerate(seq):
        nucleotides = ["A", "C", "G", "T"]
        if nucleotide not in nucleotides:
            score += np.log(1e-6)  # small probability for unknown nucleotides
            continue
        index = nucleotides.index(nucleotide)
        score += np.log(pwm[index, j] + 1e-6)  # avoid log(0)
    return score


def preprocess_seq(output_dir, tf_name, pwm_file):
    """
    Preprocess the sequences for a given TF name.
    """

    def no_score_seq_fn(site, record):
        start = site[TFColumns.START.value]
        end = site[TFColumns.END.value]
        return str(record.seq[start:end]).upper()

    def pwm_wrapper(pwm):
        def handle_seq_fn(site, record):
            chrom = site[TFColumns.CHROM.value]
            start = site[TFColumns.START.value]
            end = site[TFColumns.END.value]
            strand = site[TFColumns.STRAND.value]

            subsequence = str(record.seq[start:end]).upper()

            if strand == "-":
                subsequence = get_negative_strand_subsequence(subsequence)

            subsequence = subsequence.upper()

            score = score_seq(pwm, str(subsequence))

            return f"{chrom}\t{start}\t{end}\t{strand}\t{str(subsequence)}\t{score}"

        return handle_seq_fn

    if tf_name is None:
        tf_names = [
            name
            for name in os.listdir(output_dir)
            if os.path.isdir(os.path.join(output_dir, name))
        ]
    else:
        tf_names = [tf_name]

    for tf in tf_names:
        pwm = get_pwm(pwm_file, tf)
        preprocess_range(
            output_dir,
            tf,
            "sequences",
            is_pos=True,
            handle_seq_fn=pwm_wrapper(pwm),
        )
        preprocess_range(
            output_dir,
            tf,
            "sequences",
            is_pos=False,
            handle_seq_fn=no_score_seq_fn,
        )

    print(
        f"Preprocessed sequences for positive and negative examples of TFs: {tf_names}"
    )


def get_negative_strand_subsequence(sequence):
    """
    Given a sequence, return its negative strand subsequence.
    """
    complement = {"A": "T", "T": "A", "C": "G", "G": "C"}
    neg_strand = "".join(complement.get(base, "N") for base in reversed(sequence))
    return neg_strand


def get_top_scoring_subsequence(pwm, sequence):
    """
    Given a probability weight matrix and a sequence, return the top scoring subsequence and its interval.
    """
    tf_len = pwm.shape[1]
    max_score = float("-inf")
    best_subseq = None

    for i in range(len(sequence) - tf_len + 1):
        subseq = sequence[i : i + tf_len]
        # the scores should be the log probabilities
        score = 0.0
        for j, nucleotide in enumerate(subseq):
            nucleotides = ["A", "C", "G", "T"]
            if nucleotide not in nucleotides:
                score += np.log(1e-6)  # small probability for unknown nucleotides
                continue
            index = nucleotides.index(nucleotide)
            score += np.log(pwm[index, j] + 1e-6)  # avoid log(0)

        if score > max_score:
            max_score = score
            best_subseq = subseq
            best_interval = (i, i + tf_len)

    return best_subseq, best_interval, max_score


def preprocess_neg_seq(output_dir, tf_name, pwm_file, reverse=False):
    """
    Preprocesses the resultant sequence and interval files, s.t. we can generate
    the best negative examples based on the probability weight matrix file.
    """
    tf_output_dir = os.path.join(output_dir, f"{tf_name}/negative")
    tf_file = os.path.join(tf_output_dir, "intervals.txt")
    seq_file = os.path.join(tf_output_dir, "sequences.txt")
    output_file = os.path.join(
        tf_output_dir, f"{'reverse_' if reverse else ''}best_negative_sequences.txt"
    )

    # skip if it already exists
    if os.path.exists(output_file):
        return

    pwm = get_pwm(pwm_file, tf_name)

    # now that we have the probability weight matrix, we want to measure
    # the sliding window score for each sequence, and return the best
    with open(seq_file, "r") as seq_f, open(tf_file, "r") as interval_f, open(
        output_file, "w"
    ) as out_f:
        for seq_line, interval_line in tqdm(zip(seq_f, interval_f)):
            sequence = seq_line.strip().upper()
            chrom, start_str, end_str = interval_line.strip().split()
            start = int(start_str)
            end = int(end_str)

            # simply reverse it if it's reversed
            if reverse:
                sequence = get_negative_strand_subsequence(sequence)

            (
                best_subseq,
                (
                    sub_start_offset,
                    sub_end_offset,
                ),
                max_score,
            ) = get_top_scoring_subsequence(pwm, sequence)

            # then we adjust the start and end positions based on the strand
            if reverse:
                # now we get the original positions of the forward strand that
                # correspond to this reverse subsequence
                best_start = end - sub_end_offset
                best_end = end - sub_start_offset
            else:
                best_start = start + sub_start_offset
                best_end = start + sub_end_offset

            out_f.write(
                f"{chrom}\t{best_start}\t{best_end}\t{best_subseq}\t{max_score}\n"
            )


def filter_scores(args):
    pos_seqs_file = os.path.join(args.output_dir, args.tf, "positive", "sequences.txt")
    neg_intervals_file = os.path.join(
        args.output_dir, args.tf, "negative", "best_negative_sequences.txt"
    )
    rev_neg_intervals_file = os.path.join(
        args.output_dir, args.tf, "negative", "reverse_best_negative_sequences.txt"
    )

    # now we filter based on the scores and write out new files
    filtered_pos_file = os.path.join(
        args.output_dir, args.tf, "positive", "overlap.txt"
    )
    filtered_neg_file = os.path.join(
        args.output_dir, args.tf, "negative", "overlap.txt"
    )

    if os.path.exists(filtered_pos_file) and os.path.exists(filtered_neg_file):
        return

    # now that we score the positive, and forward, reverse sequences for negatives,
    # we should preprocess based on the overlapping score distributions
    positive_scores, negative_scores, rev_negative_scores = read_scores(args)
    overall_min, overall_max = get_overlap_range(
        positive_scores, negative_scores, rev_negative_scores
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

    neg_names = [
        TFColumns.CHROM.value,
        TFColumns.START.value,
        TFColumns.END.value,
        TFColumns.SEQ.value,
        TFColumns.LOG_PROB.value,
    ]
    neg_pr = read_samples(neg_intervals_file, names=neg_names)
    rev_neg_pr = read_samples(rev_neg_intervals_file, names=neg_names)

    positive_subset = positive_pr[
        (positive_pr[TFColumns.LOG_PROB.value] >= overall_min)
        & (positive_pr[TFColumns.LOG_PROB.value] <= overall_max)
    ]
    neg_subset = neg_pr[
        (neg_pr[TFColumns.LOG_PROB.value] >= overall_min)
        & (neg_pr[TFColumns.LOG_PROB.value] <= overall_max)
    ]
    rev_neg_subset = rev_neg_pr[
        (rev_neg_pr[TFColumns.LOG_PROB.value] >= overall_min)
        & (rev_neg_pr[TFColumns.LOG_PROB.value] <= overall_max)
    ]

    with open(filtered_pos_file, "w") as out_f:
        for _, row in positive_subset.iterrows():
            chrom = row[TFColumns.CHROM.value]
            start = row[TFColumns.START.value]
            end = row[TFColumns.END.value]
            strand = row[TFColumns.STRAND.value]
            seq = row[TFColumns.SEQ.value]
            score = row[TFColumns.LOG_PROB.value]
            out_f.write(f"{chrom}\t{start}\t{end}\t{strand}\t{seq}\t{score}\n")

    with open(filtered_neg_file, "w") as out_f:

        def write_out_file(subset, strand):
            for _, row in subset.iterrows():
                chrom = row[TFColumns.CHROM.value]
                start = row[TFColumns.START.value]
                end = row[TFColumns.END.value]
                seq = row[TFColumns.SEQ.value]
                score = row[TFColumns.LOG_PROB.value]
                out_f.write(f"{chrom}\t{start}\t{end}\t{strand}\t{seq}\t{score}\n")

        write_out_file(neg_subset, "+")
        write_out_file(rev_neg_subset, "-")


def preprocess_bigwigs(args):
    assert args.bigwig_dir is not None, "bigwig_dir must be specified to preprocess"

    # first, let's read the intervals that we have
    pos_file = os.path.join(args.output_dir, args.tf, "positive", "overlap.txt")
    neg_file = os.path.join(args.output_dir, args.tf, "negative", "overlap.txt")
    names = [
        TFColumns.CHROM.value,
        TFColumns.START.value,
        TFColumns.END.value,
        TFColumns.STRAND.value,
        TFColumns.SEQ.value,
        TFColumns.LOG_PROB.value,
    ]
    pos_df = read_samples(pos_file, names=names)
    neg_df = read_samples(neg_file, names=names)

    # then we need to preprocess the pos_df, neg_df to get:
    # 1. the chromosomes and their total lengths that we care about
    # 2. the intervals that we care about for each bigwig file
    # We can do this by merging the two dataframes
    intervals = pos_df.set_union_overlaps(neg_df, strand_behavior="ignore")
    chromosomes = intervals.chromosomes

    output_bigwig_dir = os.path.join(args.bigwig_dir, args.tf, "bigwig_processed")
    os.makedirs(output_bigwig_dir, exist_ok=True)

    for bigwig_file in args.bigwigs:
        mgw_bigwig_path = os.path.join(
            args.bigwig_dir,
            bigwig_file,
        )

        output_bigwig_file = os.path.join(
            output_bigwig_dir,
            bigwig_file,
        )

        if os.path.exists(output_bigwig_file):
            continue

        # then let's read the bigwig file and open the new preprocessed file
        bw_input = pyBigWig.open(mgw_bigwig_path)
        bw_output = pyBigWig.open(output_bigwig_file, "w")

        # first, we need to define the header for the new bigwig file
        bw_output.addHeader(
            [(chrom, bw_input.chroms()[chrom]) for chrom in chromosomes]
        )

        # now let's iterate through the intervals and extract the values
        for chrom in tqdm(chromosomes):
            chrom_intervals = intervals[intervals[TFColumns.CHROM.value] == chrom]
            for _, site in chrom_intervals.iterrows():
                start = site[TFColumns.START.value]
                end = site[TFColumns.END.value]
                values = bw_input.values(chrom, start, end)
                # we have to add everything one by one
                bw_output.addEntries(
                    [chrom] * (end - start),
                    [start + idx for idx in range(end - start)],
                    [start + idx + 1 for idx in range(end - start)],
                    values=values,
                )
                assert len(values) == end - start

        print(f"Finished preprocessing bigwig file: {bigwig_file}")


if __name__ == "__main__":
    args = get_args()
    pos_tf_sites = generate_positive_examples(
        args.true_tf_file, args.output_dir, args.tf
    )
    generate_negative_examples(
        pos_tf_sites, args.chip_seq_file, args.output_dir, args.tf
    )

    preprocess_seq(args.output_dir, args.tf, args.pwm_file)

    # do the negative sequence preprocessing -- can only do if a TF name is given
    if args.tf is not None:
        preprocess_neg_seq(args.output_dir, args.tf, args.pwm_file)
        preprocess_neg_seq(args.output_dir, args.tf, args.pwm_file, reverse=True)

        # now we will read the files again, and only keep those
        # that are in the overlapping range
        filter_scores(args)

        # now let's preprocess the big wig files to only include the
        # regions defined in the negative, reverse negative, and positive files
        preprocess_bigwigs(args)
