# Transcription Binding Prediction

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


## Data setup and Preprocessing
You will need the following files:
1. Human genome data found from: http://hgdownload.soe.ucsc.edu/goldenPath/hg19/bigZips/chromFa.tar.gz. Move this under the folder of `data/fasta`, and unzip.
2. An active regulatory region file in `bed` format. See https://en.wikipedia.org/wiki/BED_(file_format) for more information.
3. Set of genomic coordinates for true transcription factor binding sites in a txt file.
4. Set of motifs for each transcription factor, encoded as a probability weight matrix, that suggests a sequence should be technically open, also in a txt file.

Make sure to set them all under the same directory, ideally under `data/`, however you can specify during preprocessing to use any path with:
```
python src/preprocess/preprocess.py --fasta_data_dir <str> --chip_seq_file <str> --true_tf_file <str> --pwm_file <str>
```
or if you have everything under the correct locations of `data/fasta, data/wgEncodeRegTfbsClusteredV3.GM12878.merged.bed, data/factorbookMotifPos.txt, data/factorbookMotifPwm.txt` respectively, then you can simply run:
```
make preprocess
```

This will preprocess the data and spit out to what is specified by `--output_dir`, which is `data/tf_sites` by default.

Then, to prepare for a specific TF with other structural features, use:
```
python src/preprocess/preprocess.py --tf <str> --bigwig_dir <str> --bigwigs <filename for MGW> <filename for HelT> ...
```

## Training
To run a training loop, simply run:
```
python src/main.py -c configs/<config.yaml>
```
Or add any of the flags found under `src/models/config.py` to override any values in the yaml file.

### Configs
Take the `configs/simple.yaml` as an example, where you can set the following:
1. `architecture`, must be set. See `src/models/config.py` for a list of all models.
2. `tf`, must be specified.
3. `preprocess_data_dir`, default is `data/tf_sites`,
4. `train_split`, default is 0.8.
5. `batch_size`, default is 32.
6. `pwm_file`, file containing the Position Weight Matrix. Should be the same used in the preprocessing step and look like:
```
GFI1	8	1.000000,1.000000,0.000000,0.000000,0.626866,0.000000,0.671642,0.000000,	0.000000,0.000000,0.298507,1.000000,0.000000,0.716418,0.000000,0.000000,	0.000000,0.000000,0.000000,0.000000,0.000000,0.283582,0.000000,1.000000,	0.000000,0.000000,0.701493,0.000000,0.373134,0.000000,0.328358,0.000000,
```
where the tabs delimit the nucleotide probabilities.
7. `pred_struct_data_dir`, the directory containing the BigWig files. **Note these files must all be in the same directory**.
8. `pred_struct_features`, a list input for the types of structural features to include.
9. `<feat>_file_name` the file name for the specified structural feature. Defaults to `<pred_struct_data_dir>/hg19.<pred_struct>.wig.bw`.
10. `use_probs`, whether or not we will use the probability scores from the position weight matrix per nucleotide.


## Contributing
We use some autoformatters, such as `black` to ensure readability.
Run:
```
pip install pre-commit
pre-commit install
```
