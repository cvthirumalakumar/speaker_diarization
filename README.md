# Speaker Diarization

This repository contains code to evaluate dataset using publicly available speaker diarization models. Details about the models can be found in the report.

## Data preparation

1. First (PriMock57)[https://github.com/babylonhealth/primock57] dataset has to be downloaded and processed for evaluation.
2. Follow below steps for downloading and preprocessing.

```bash
git lfs install
git clone https://github.com/babylonhealth/primock57.git
cd primock57/scripts
./mix_audio.sh
cd ..
python data_preparation/generate_rttm_files.py --transcript_path=<text grids path> --output_path=<output rttm path>
cd ..
```
Above code will download dataset and mix individual audios of doctro and patient audios into single audio. Then `generate_rttm_files.py` will generate rttm files from individual text grid files, which will be used as reference for evaluation.

## Inference

change working directory to `inference` folder by using following command.
```bash
cd inference
```
`inference` folder contains inference scripts for 6 different diarisation models/methods. Run all scripts to generate prediction rttm files. Please check the requirements based on the imports in each script.

## Evaluation

Run `outputs_and_evaluation/evaluate.py` to calculate DER (Diarization Error Rate)

For reference, outputs of all 6 models and ground truth rttm files have been kept in `outputs_and_evaluation` folder.