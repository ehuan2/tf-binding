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
from helpers import TFColumns, read_negative_samples, read_positive_samples


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
    return parser.parse_args()


def generate_positive_examples(true_tf_file, output_dir):
    """
    Generate positive examples from the true TF binding sites file.
    """
    pos_tf_sites = read_positive_samples(true_tf_file)
    tf_names = pos_tf_sites[TFColumns.TF_NAME.value].unique().tolist()

    for tf_name in tqdm(tf_names):
        dir_path = os.path.join(output_dir, tf_name)

        if os.path.exists(f"{dir_path}/positive_examples.txt"):
            continue

        # now we select for this TF
        tf_sites = pos_tf_sites[pos_tf_sites[TFColumns.TF_NAME.value] == tf_name]

        os.makedirs(dir_path, exist_ok=True)

        with open(f"{dir_path}/positive_examples.txt", "w") as out_f:
            for _, site in tf_sites.iterrows():
                chrom = site[TFColumns.CHROM.value]
                start = site[TFColumns.START.value]
                end = site[TFColumns.END.value]
                score = site[TFColumns.SCORE.value]
                strand = site[TFColumns.STRAND.value]
                out_f.write(f"{chrom}\t{start}\t{end}\t{score}\t{strand}\n")

    print(f"Generated positive examples for {len(tf_names)} transcription factors.")
    return pos_tf_sites


def generate_negative_examples(pos_tf_sites, chip_seq_file, output_dir):
    """
    Generate negative examples from the ChIP-seq data file.
    """
    chip_seq_sites = read_negative_samples(chip_seq_file)
    tf_names = pos_tf_sites[TFColumns.TF_NAME.value].unique().tolist()

    for tf_name in tqdm(tf_names):
        dir_path = os.path.join(output_dir, tf_name)

        if os.path.exists(f"{dir_path}/negative_examples.txt"):
            continue

        # now we select for this TF
        tf_pos_sites = pos_tf_sites[pos_tf_sites[TFColumns.TF_NAME.value] == tf_name]

        # find all the sites in chip-seq that don't overlap
        # with some positive site
        negative_sites = chip_seq_sites.overlap(
            tf_pos_sites,
            invert=True,
        )

        with open(f"{dir_path}/negative_examples.txt", "w") as out_f:
            for _, site in negative_sites.iterrows():
                chrom = site.Chromosome
                start = site.Start
                end = site.End
                out_f.write(f"{chrom}\t{start}\t{end}\n")

    print(f"Generated negative examples for {len(tf_names)} transcription factors.")


if __name__ == "__main__":
    args = get_args()
    pos_tf_sites = generate_positive_examples(args.true_tf_file, args.output_dir)
    generate_negative_examples(pos_tf_sites, args.chip_seq_file, args.output_dir)
