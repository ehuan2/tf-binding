"""
preprocess.py.

Preprocess script for getting positive and negative examples
of binding sites, having specified the active regulatory regions
and the true regions of binding sites.

Outputs to data/tf_sites/ by default, where there is a folder per each TF.
"""

import argparse
import os
from tqdm import tqdm
from helpers import (
    TFColumns,
    read_negative_samples,
    read_positive_samples,
)
from Bio import SeqIO
import numpy as np


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
        default=None,
        help="The probability weight matrix file to read from for negative sequence processing.",
    )
    return parser.parse_args()


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
                start = site[TFColumns.START.value]
                end = site[TFColumns.END.value]
                subsequence = handle_seq_fn(record, start, end)
                out_f.write(f"{subsequence}\n")


def preprocess_seq(output_dir, tf_name):
    """
    Preprocess the sequences for a given TF name.
    """

    def handle_seq_fn(record, start, end):
        subsequence = record.seq[start:end]
        return str(subsequence)

    if tf_name is None:
        tf_names = [
            name
            for name in os.listdir(output_dir)
            if os.path.isdir(os.path.join(output_dir, name))
        ]
        for tf_name in tf_names:
            preprocess_range(
                output_dir,
                tf_name,
                "sequences",
                is_pos=True,
                handle_seq_fn=handle_seq_fn,
            )
            preprocess_range(
                output_dir,
                tf_name,
                "sequences",
                is_pos=False,
                handle_seq_fn=handle_seq_fn,
            )
    else:
        preprocess_range(
            output_dir, tf_name, "sequences", is_pos=True, handle_seq_fn=handle_seq_fn
        )
        print(f"Preprocessed sequences for positive examples of TF: {tf_name}")
        preprocess_range(
            output_dir, tf_name, "sequences", is_pos=False, handle_seq_fn=handle_seq_fn
        )
        print(f"Preprocessed sequences for negative examples of TF: {tf_name}")


def get_top_scoring_subsequence(pwm, sequence):
    """
    Given a probability weight matrix and a sequence, return the top scoring subsequence and its interval.
    """
    tf_len = pwm.shape[1]
    max_score = float("-inf")
    best_subseq = None

    for i in range(len(sequence) - tf_len + 1):
        subseq = sequence[i : i + tf_len].upper()
        # the scores should be the log probabilities
        score = 0.0
        for j, nucleotide in enumerate(subseq):
            nucleotides = ["A", "C", "G", "T"]
            if nucleotide not in nucleotides:
                score += np.log(1e-6)  # small probability for unknown nucleotides
                continue
            index = nucleotides.index(nucleotide)
            score += np.log(pwm[index, j])

        if score > max_score:
            max_score = score
            best_subseq = subseq
            best_interval = (i, i + tf_len)

    return best_subseq, best_interval


def preprocess_neg_seq(output_dir, tf_name, pwm_file):
    """
    Preprocesses the resultant sequence and interval files, s.t. we can generate
    the best negative examples based on the probability weight matrix file.
    """
    tf_output_dir = os.path.join(output_dir, f"{tf_name}/negative")
    tf_file = os.path.join(tf_output_dir, "intervals.txt")
    seq_file = os.path.join(tf_output_dir, "sequences.txt")

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

    # now that we have the probability weight matrix, we want to measure
    # the sliding window score for each sequence, and return the best
    with open(seq_file, "r") as seq_f, open(
        os.path.join(tf_output_dir, "best_negative_sequences.txt"), "w"
    ) as out_f:
        for line in seq_f:
            pass


if __name__ == "__main__":
    args = get_args()
    pos_tf_sites = generate_positive_examples(
        args.true_tf_file, args.output_dir, args.tf
    )
    generate_negative_examples(
        pos_tf_sites, args.chip_seq_file, args.output_dir, args.tf
    )

    preprocess_seq(args.output_dir, args.tf)
    # preprocess_neg_seq(args.output_dir, args.tf, args.pwm_file)

    example_pwm = np.array(
        [
            [0.2, 0.3, 0.3, 0.2],  # A
            [0.3, 0.2, 0.2, 0.3],  # C
            [0.2, 0.2, 0.3, 0.3],  # G
            [0.3, 0.3, 0.2, 0.2],  # T
        ]
    )
    example_sequence = "TTGGACGTACGTGACTTGA"
    print(get_top_scoring_subsequence(example_pwm, example_sequence))
