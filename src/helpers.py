"""
helpers.py

This module contains helper functions for various tasks.

In particular, we will have:
- a function that can take a fasta file that corresponds with a chromosome,
  a range of sequences, and returns the resulting subsequence.
- functions that will return the pyranges for the positive and negative examples
"""
from Bio import SeqIO
from enum import Enum
import pyranges as pr
import pandas as pd
import numpy as np


def get_subsequence(fasta_file, start, end):
    """
    Extracts a subsequence from a fasta file given start and end positions.

    Parameters:
    fasta_file (str): Path to the fasta file.
    start (int): Start position (1-based).
    end (int): End position (1-based).

    Returns:
    str: The extracted subsequence.
    """
    # Read the fasta file
    record = SeqIO.read(fasta_file, "fasta")
    # Extract the subsequence, it is 0-based indexed and end-exclusive!
    # So it works super nicely in python :)
    subsequence = record.seq[start:end]
    return str(subsequence)


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
    SEQ = "Sequence"
    SEQ_ENCODED = "Sequence_Encoded"
    LOG_PROB = "Log_Prob"
    MGW = "MGW"


def read_positive_samples(true_tf_file, include_index=True):
    """
    Reads the true transcription factor binding sites from a file
    and returns them as a PyRanges object.

    Parameters:
    true_tf_file (str): Path to the file containing true TF binding sites.
    include_index (bool): Whether to include the Index column.

    Returns:
    pr.PyRanges: PyRanges object containing positive examples.
    """
    names = [TFColumns.INDEX.value] if include_index else []
    names += [
        TFColumns.CHROM.value,
        TFColumns.START.value,
        TFColumns.END.value,
    ]

    if include_index:
        names.append(TFColumns.TF_NAME.value)

    names += [
        TFColumns.SCORE.value,
        TFColumns.STRAND.value,
    ]

    return pr.PyRanges(pd.read_table(true_tf_file, names=names))


def read_negative_samples(chip_seq_file, include_chrom_index=True):
    """
    Generates the

    Parameters:
    chip_seq_file (str): Path to the ChIP-seq data file.
    include_chrom_index (bool): Whether to include the Chrom_Index column.

    Returns:
    pr.PyRanges: PyRanges object containing negative examples.
    """
    names = [
        TFColumns.CHROM.value,
        TFColumns.START.value,
        TFColumns.END.value,
    ]

    if include_chrom_index:
        names.append(TFColumns.CHROM_INDEX.value)

    return pr.PyRanges(pd.read_table(chip_seq_file, names=names))


def read_samples(file_path, names):
    """
    Generic function to read samples from a file into a PyRanges object.

    Parameters:
    file_path (str): Path to the input file.
    names (list): List of column names.

    Returns:
    pr.PyRanges: PyRanges object containing the samples.
    """
    return pr.PyRanges(pd.read_table(file_path, names=names))

def one_hot_encode(seq):
    """
    One-hot encode DNA sequence into (len(seq), 4) np.array. Convert torch.tensor after call if needed.
    'N' or other unknown bases will have 0's across given column.
    Maps according to:
        'A': [1, 0, 0, 0],
        'C': [0, 1, 0, 0],
        'G': [0, 0, 1, 0],
        'T': [0, 0, 0, 1]

    Parameters:
    seq (str): DNA sequence to be encoded.

    Returns:
    np.array: Numpy array of one-hot encoded sequence
    """

    seq = seq.upper()
    mapping = {
        'A': [1, 0, 0, 0],
        'C': [0, 1, 0, 0],
        'G': [0, 0, 1, 0],
        'T': [0, 0, 0, 1]
    }

    encoded = np.zeros((len(seq), 4), dtype=np.float32)

    for i, base in enumerate(seq):
        if base in mapping:
            encoded[i] = mapping[base]
    
    return encoded
