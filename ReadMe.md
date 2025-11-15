# Transcription Binding Prediction
## Data setup
You will need the following files:
1. Human genome data found from: http://hgdownload.soe.ucsc.edu/goldenPath/hg19/bigZips/chromFa.tar.gz. Move this under the folder of `data/fasta`.
2. An active regulatory region file in `bed` format. See https://en.wikipedia.org/wiki/BED_(file_format) for more information.
3. Set of genomic coordinates for true transcription factor binding sites in a txt file.
4. Set of motifs for each transcription factor that suggests a sequence should be technically open, also in a txt file.

Move all of these under `data/`.
