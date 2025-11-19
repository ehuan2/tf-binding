# Transcription Binding Prediction
## Data setup
You will need the following files:
1. Human genome data found from: http://hgdownload.soe.ucsc.edu/goldenPath/hg19/bigZips/chromFa.tar.gz. Move this under the folder of `data/fasta`.
2. An active regulatory region file in `bed` format. See https://en.wikipedia.org/wiki/BED_(file_format) for more information.
3. Set of genomic coordinates for true transcription factor binding sites in a txt file.
4. Set of motifs for each transcription factor that suggests a sequence should be technically open, also in a txt file.

Move all of these under `data/`.

## Requirements setup
Due to the `pyranges` environment, we suggest to use conda, with the following:
```
conda create -n tfbinding python=3.12
```
then, you can install the requirements using:
```
pip install -r requirements.txt
```
Note: we require python version 3.12 for the pyranges1 package.

## Contributing
We use some autoformatters, such as `black` to ensure readability.
Run:
```
pip install pre-commit
pre-commit install
```

## Config Setup
Take the `configs/simple.yaml` as an example, where you have the following to define:
1. The type of architecture to train.
2. The transcription factor to use for training.
3. The predicted structure data directory.
4. Whether or not to include the sequence information.
