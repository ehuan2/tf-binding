"""
preprocess.py.

Preprocess script for getting positive and negative examples
of binding sites, having specified the active regulatory regions
and the true regions of binding sites.

Outputs to data/tf_sites/ by default, where there is a folder per each TF.
"""

import argparse
import os
import pyranges as pr
import pandas as pd
from enum import Enum
from tqdm import tqdm


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


# create an enum for the column names to reuse
class TFColumns(Enum):
    INDEX = "Index"
    CHROM = "Chromosome"
    START = "Start"
    END = "End"
    TF_NAME = "TF_Name"
    SCORE = "Score"
    STRAND = "Strand"
    CHROM_INDEX = "Chrom_Index"  # for the chip-seq data, chrx.y, meaningless


def generate_positive_examples(true_tf_file, output_dir):
    """
    Generate positive examples from the true TF binding sites file.
    """
    pos_tf_sites = pr.PyRanges(
        pd.read_table(
            true_tf_file,
            names=[
                TFColumns.INDEX.value,
                TFColumns.CHROM.value,
                TFColumns.START.value,
                TFColumns.END.value,
                TFColumns.TF_NAME.value,
                TFColumns.SCORE.value,
                TFColumns.STRAND.value,
            ],
        )
    )

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
    chip_seq_sites = pr.PyRanges(
        pd.read_table(
            chip_seq_file,
            names=[
                TFColumns.CHROM.value,
                TFColumns.START.value,
                TFColumns.END.value,
                TFColumns.CHROM_INDEX.value,
            ],
        )
    )

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
