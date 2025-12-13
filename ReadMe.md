# 🧬 Transcription Factor Binding Prediction 🧬

TF-binding prediction can be easily achieved through the motif scorings of the sequence, however the sequences that remain overlapping are difficult to classify with accuracy. This GitHub repository is the implementation of the paper: https://drive.google.com/file/d/16eQ9wBZkh2MYc5stWbwQUWuM3819rsWp/view?usp=sharing.

## Requirements setup
We setup everything with conda as shown below:
```
conda create -n tfbinding python=3.12
```
Then, you can install the requirements using:
```
pip install -r requirements.txt
```
Note: we require python version 3.12 for the pyranges1 package.


## Data setup and Preprocessing
For all this existing data, we host the preprocessed data (alongside the other information) on our Google Drive here:
1. Sequence information and TF regions needed for the models: https://drive.google.com/file/d/1UgGh8bTUN7pOOCwPaKt5_kgtzi7GYpJd/view
2. The structural features used to enhance the model are found here: https://drive.google.com/file/d/19oz42DGXyzThQAhL74M-4sPO18xmgEuk/view?usp=sharing
3. The structural features but slightly larger to incorporate a 200 context window around the TF region, found here: https://drive.google.com/file/d/1FVmIdu91k1Ggo26NnNs3C4KCil62o_PE/view?usp=sharing

### Sequence information and TF Region Data Preprocessing
Otherwise, if you wish to do it yourself, follow these instructions:
1. Human genome sequence data found from: http://hgdownload.soe.ucsc.edu/goldenPath/hg19/bigZips/chromFa.tar.gz. Move this under the folder of `data/fasta`, and unzip.
2. Active regulatory regions in `bed` format. See https://en.wikipedia.org/wiki/BED_(file_format) for more information.
3. Set of genomic coordinates for true transcription factor binding sites in a txt file. This is usually from a factorbook.
4. Set of motifs for each transcription factor, encoded as a probability weight matrix, that suggests a sequence is likely to be binded by a certain transcription factor, also in a txt file.

Make sure to set them all under the same directory. The default directory that the preprocessing scripts are under `data`, however, feel free to specify the following:
```
python src/preprocess/preprocess.py --fasta_data_dir <str> --chip_seq_file <str> --true_tf_file <str> --pwm_file <str>
```

If you are unzipping from the Google Drive, and have everything under the correct locations of `data/fasta, data/wgEncodeRegTfbsClusteredV3.GM12878.merged.bed, data/factorbookMotifPos.txt, data/factorbookMotifPwm.txt` respectively, then you can simply run:
```
python src/preprocess/preprocess.py --tf <str>
```

However, if you wish to preprocess for all the TFs, you can use the following command. This will preprocess the data and spit out to what is specified by `--output_dir`, which is `data/tf_sites` by default. Warning: this will likely run a long time as the default is to run over every TF. **We suggest to only run the preprocessing script on TFs that you need**.
```
make preprocess
```
### Structural Data Preprocessing
The currently supported structural information is:

1. MGW: https://rohslab.usc.edu/ftp/hg19/hg19.MGW.wig.bw
2. PrOT: https://rohslab.usc.edu/ftp/hg19/hg19.ProT.wig.bw
3. HelT: https://rohslab.usc.edu/ftp/hg19/hg19.HelT.wig.bw
4. Roll: https://rohslab.usc.edu/ftp/hg19/hg19.Roll.wig.bw
5. OC2: https://rohslab.usc.edu/ftp/hg19/hg19.OC2.wig.bw

Either download to the same directory, which then you can further preprocess (as these files are huge) for faster training or use the preprocessed files from: https://drive.google.com/file/d/19oz42DGXyzThQAhL74M-4sPO18xmgEuk/view?usp=sharing as above. You can add more structural compatability by adding to `src/models/config.py`.

If you wish to preprocess your own, simply run:
```
python src/preprocess/preprocess.py --tf PAX5 --bigwig_dir <str> --bigwigs hg19.MGW.wig.bw hg19.HelT.wig.bw hg19.ProT.wig.bw hg19.OC2.wig.bw hg19.Roll.wig.bw --context_window <int>
```
This will be outputted to the specified bigwig directory under `<tf>/bigwig_processed`. The context window flag is used if you wish to preprocess the bigwigs so that they include more structural information before and after the

**Note: These preprocessed bigwig files are not compatible with certain context window lengths. To make them compatible, be sure to preprocess them beforehand using the flag.**.

## Training
To run a training loop, simply run:
```
python src/main.py -c configs/<config.yaml>
```
Or add any of the flags found under `src/models/config.py` to override any values in the yaml file.

### MLFlow
For training, we also use mlflow, a framework used for organizing different ML training runs. To access the dashboard, simply run:
```
mlflow server --port 5000
```
or run:
```
make mlflow
```
and access the dashboard through the URL shown in the terminal.


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
11. `restart_train` specifies whether or not to use the previous found model given the exact same parametrization.
12. `random_seed` specifies the random seeding to use to ensure the same data splits across runs and model comparisons.
13. `epochs` specifies the number of epochs to train the model for.
14. `context_window` specifies the extra context window for the model to use. Be sure that the proper bigwig files can handle the context window.
15. `device` specifies the torch device to use.
16. `dtype` specifies the data type to use for training the MLP model.
17. `use_seq` specifies whether to use the one-hot encoding from the sequence itself.

Many others are included in the config file itself, and are model dependent. Read `src/models/config.py` for more information.


## Contributing
We use some autoformatters, such as `black` to ensure readability.
Run:
```
pip install pre-commit
pre-commit install
```
