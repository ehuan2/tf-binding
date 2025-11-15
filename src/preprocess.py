'''
preprocess.py.

Preprocess script for getting positive and negative examples
of binding sites, having specified the active regulatory regions
and the true regions of binding sites.

Outputs to data/tf_sites/ by default, where there is a folder per each TF.

** TODO **
For now, we will only have the positive examples, but eventually,
we will also add negative examples. The negative examples need to be
found from motifs as well.
'''

import argparse
import os

def get_args():
    parser = argparse.ArgumentParser(
        description='Preprocess script for getting positive and negative examples of binding sites.'
    )
    parser.add_argument(
        '--chip_seq_file',
        type=str,
        default='data/wgEncodeRegTfbsClusteredV3.GM12878.merged.bed',
        help='Path to the ChIP-seq data file.'
    )
    parser.add_argument(
        '--true_tf_file',
        type=str,
        default='data/factorbookMotifPos.txt',
        help='Path to the file containing true transcription factor binding sites.'
    )
    parser.add_argument(
        '--output_dir',
        type=str,
        default='data/tf_sites',
        help='Directory to save the output files.'
    )
    return parser.parse_args()


def generate_positive_examples(true_tf_file, output_dir):
    '''
    Generate positive examples from the true TF binding sites file.
    '''
    tf_sites = {}
    with open(true_tf_file, 'r') as f:
        for line in f:
            parts = line.strip().split('\t')
            
            chrom = parts[1]
            start = int(parts[2])
            end = int(parts[3])
            tf_name = parts[4]
            score = parts[5]
            strand = parts[6]

            if tf_name not in tf_sites:
                tf_sites[tf_name] = []
            tf_sites[tf_name].append((chrom, start, end, score, strand))

    for tf_name, sites in tf_sites.items():

        dir_path = os.path.join(output_dir, tf_name)
        os.makedirs(dir_path, exist_ok=True)

        with open(f'{dir_path}/positive_examples.txt', 'w') as out_f:
            for site in sites:
                chrom, start, end, score, strand = site
                out_f.write(f'{chrom}\t{start}\t{end}\t{score}\t{strand}\n')

    print(f'Generated positive examples for {len(tf_sites)} transcription factors.')

if __name__ == '__main__':
    args = get_args()
    generate_positive_examples(args.true_tf_file, args.output_dir)
