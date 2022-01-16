# AV-HuBERT (Audio-Visual Hidden Unit BERT)
[Learning Audio-Visual Speech Representation by Masked Multimodal Cluster Prediction](https://arxiv.org/abs/2201.02184)

[Robust Self-Supervised Audio-Visual Speech Recognition](https://arxiv.org/abs/2201.01763)

![lip-reading](assets/lipreading.gif)

## Introduction
AV-HuBERT is a self-supervised representation learning framework for audio-visual speech. It achieves state-of-the-art results in lip reading, ASR and audio-visual speech recognition on the LRS3 audio-visual speech benchmark.

If you find AV-HuBERT useful in your research, please use the following BibTeX entry for citation.
```BibTeX
@article{shi2022avhubert,
    author  = {Bowen Shi and Wei-Ning Hsu and Kushal Lakhotia and Abdelrahman Mohamed},
    title = {Learning Audio-Visual Speech Representation by Masked Multimodal Cluster Prediction},
    journal = {arXiv preprint arXiv:2201.02184}
    year = {2022}
}

@article{shi2022avsr,
    author  = {Bowen Shi and Wei-Ning Hsu and Abdelrahman Mohamed},
    title = {Robust Self-Supervised Audio-Visual Speech Recognition},
    journal = {arXiv preprint arXiv:2201.01763}
    year = {2022}
}
```

## License

AV-HuBERT LICENSE AGREEMENT

This License Agreement (as may be amended in accordance with this License
Agreement, “License”), between you (“Licensee” or “you”) and Meta Platforms,
Inc. (“Meta” or “we”) applies to your use of any computer program, algorithm,
source code, object code, or software that is made available by Meta under this
License (“Software”) and any specifications, manuals, documentation, and other
written information provided by Meta related to the Software (“Documentation”).

By using the Software, you agree to the terms of [this
License](https://github.com/facebookresearch/av_hubert/blob/main/LICENSE). If
you do not agree to this License, then you do not have any rights to use the
Software or Documentation (collectively, the “Software Products”), and you must
immediately cease using the Software Products.

## Pre-trained and fine-tuned models

Please find the checkpoints [here](http://facebookresearch.github.io/av_hubert)

## Demo
Run our lip-reading demo using Colab: [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/drive/1bNXkfpHiVHzXQH8WjGhzQ-fsDxolpUjD)

## Installation
First, create a conda virtual environment and activate it:
```
conda create -n avhubert python=3.8 -y
conda activate avhubert
```
Then, clone this directory:
```
git clone https://github.com/facebookresearch/av_hubert.git
cd avhubert
git submodule init
git submodule update
```

Lastly, install Fairseq and the other packages:
```
pip install -r requirements.txt
cd fairseq
pip install --editable ./
```

## Load a pretrained model
```sh
$ cd avhubert
$ python
>>> import fairseq
>>> import hubert_pretraining, hubert
>>> ckpt_path = "/path/to/the/checkpoint.pt"
>>> models, cfg, task = fairseq.checkpoint_utils.load_model_ensemble_and_task([ckpt_path])
>>> model = models[0]
```

## Train a new model

### Data preparation

Follow the steps in [`preparation`](avhubert/preparation/) to pre-process:
- LRS3 and VoxCeleb2 datasets

Follow the steps in [`clustering`](avhubert/clustering/) (pre-train only) to create:
- `{train,valid}.km` frame-aligned pseudo label files.
The `label_rate` is the same as the feature frame rate used for clustering,
which is 100Hz for MFCC features and 25Hz for AV-HuBERT features by default.

### Pre-train an AV-HuBERT model

Suppose `{train,valid}.tsv` are saved at `/path/to/data`, `{train,valid}.km`
are saved at `/path/to/labels`, the configuration file is saved at `/path/to/conf/conf-name`, and the label rate is 100Hz.

To train a model, run:
```sh
$ cd avhubert
$ fairseq-hydra-train --config-dir /path/to/conf/ --config-name conf-name \
  task.data=/path/to/data task.label_dir=/path/to/label \
  model.label_rate=100 hydra.run.dir=/path/to/experiment/pretrain/ \
  common.user_dir=`pwd`
```

### Finetune an AV-HuBERT model with Seq2Seq
Suppose `{train,valid}.tsv` are saved at `/path/to/data`, `{train,valid}.wrd`
are saved at `/path/to/labels`, the configuration file is saved at `/path/to/conf/conf-name`.

To fine-tune a pre-trained HuBERT model at `/path/to/checkpoint`, run:
```sh
$ cd avhubert
$ fairseq-hydra-train --config-dir /path/to/conf/ --config-name conf-name \
  task.data=/path/to/data task.label_dir=/path/to/label \
  task.tokenizer_bpe_model=/path/to/tokenizer model.w2v_path=/path/to/checkpoint \
  hydra.run.dir=/path/to/experiment/finetune/ common.user_dir=`pwd`
```

### Decode an AV-HuBERT model
Suppose the `test.tsv` and `test.wrd` are the video list and transcripts of
the split to be decoded, saved at `/path/to/data`, and the fine-tuned model is
saved at `/path/to/checkpoint`.

#### Seq2Seq decoding

`task.normalize` needs to be consistent with the value used during fine-tuning.
Decoding results will be saved at
`/path/to/experiment/decode/s2s/test`.

```sh
$ cd avhubert
$ python -B infer_s2s.py --config-dir ./conf/ --config-name conf-name \
  dataset.gen_subset=test common_eval.path=/path/to/checkpoint \
  common_eval.results_path=/path/to/experiment/decode/s2s/test \
  override.modalities=['video'] common.user_dir=`pwd`
```

The command above uses the default decoding hyperparameter, which can be found
in `conf/s2s_decode.yaml`. `override.modalities` can be set to `['video']` (for lip reading),
or `['audio']` (for ASR) or `['audio','video']` (for audio-visual speech recognition).These parameters can be
configured from the command line. For example, to search with a beam size of
20, we can append the command above with `generation.beam=20`.
Important parameters include:
- generation.beam
- generation.lenpen

#### Different test set
If your test data are stored in a different directory with the training data, append the following to the above command.

`+override.data=/path/to/test +override.label_dir=/path/to/test`

, where `/path/to/test` contains `test.{tsv,wrd}`. This is useful when you want to test with the fine-tuned checkpoints we provide.

#### Test under noisy environment
If you want to test your model under noisy environment, append the following to the above command.

`+override.noise_wav=/path/to/noise override.noise_prob=1 override.noise_snr={snr}` 

 `{snr}` is the signal-to-noise ratio (SNR) and `/path/to/noise` is a folder containing noise manifest files (`/path/to/noise/{valid,test}.tsv`). See [`preparation`](avhubert/preparation/) for setting up this folder.

