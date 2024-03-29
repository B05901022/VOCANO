# VOCANO: A note transcription framework for singing voice in polyphonic music

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/B05901022/VOCANO/blob/main/VOCANO.ipynb)

Inference code of our work **VOCANO: A note transcription framework for singing voice in polyphonic music**. For training details, please check [this repo][Note-Segmentation-SSL]. **This work is now also working as part of the [Omnizart][Omnizart] project, which is able to transcribe pitched instruments, vocal melody, chords, drum events, and beat.** Complete demo including the whole Omnizart transcription pipeline is available on the [Omnizart Website][Omnizart Website]. Omnizart is powered by the research outcomes from [Music and Culture Technology (MCT) Lab][Music and Culture Technology (MCT) Lab].

## Overview

In this repository, we present a VOCAl NOte transcription tool, named VOCANO, which can help transcribing vocal into MIDI files. We decompose singing voice transcription(SVT) into two parts: pitch extraction and note segmentation. Our pitch extraction model is a pre-trained Patch-CNN, which was from [our previous work][Vocal melody extraction using patch-based CNN]; our note segmentation is a PyramidNet-110 with ShakeDrop regularization, which is an improvement from [Hierarchical classification networks][Hierarchical classification networks for singing voice segmentation and transcription] trained by semi-supervised learning. See our paper for more details.

## Demo

Examples using CMedia dataset and ISMIR2014 dataset are shown [here][Our demo googledrive].

## Requirements

Our testing is performed under Python3 and CUDA10.1, under PyTorch framework. You can either use our `requirements.txt` or manually install packages needed. For faster inferencing under NVIDIA GTX/RTX devices, it is recommended to install [CuPy][CuPy] and use `-use_cp` flag while running our code.

### Clone this repository

```bash
$ git clone https://github.com/B05901022/VOCANO.git
$ cd VOCANO
```

### Install [NVIDIA-apex][NVIDIA-apex]

```bash
$ git clone https://github.com/NVIDIA/apex.git
$ pip install -v --disable-pip-version-check --no-cache-dir ./apex
$ bash setup.sh
```

### Install requirements

```bash
$ pip install -r requirements.txt -f https://download.pytorch.org/whl/torch_stable.html
```

## How to run

We provided a preprocessing code and a transcription which includes the whole pipeline. Preprocessing can extract numpy feature and pitch contour for our second phase note segmentation, which you can set `-use_pre` flag for using pre-extracted informations. Note that our note segmentation model is trained on monophonic data, thus using an additional source separation model is recommended for polyphonic cases. See appendix for more information.

### Preprocessing(Optional)

The preprocessing code extracts CFP feature which is used in our note segmentation model and pitch contour extracted by Patch-CNN.

```python
python -m vocano.preprocess -n [output_file_name] -wd [input_wavfile] -s
```

Use `-d` flag to choose which devices to be used. The default device is `-d cpu`, but you can also choose `-d [device id]` to select your gpu or `-d auto` to automatically select the least used gpu (only for Linux). Use `-use_cp` to enable gpu acceleration for feature extraction when cupy is installed. Set `-bz [batchsize] -nw [num_workers] -pn` to fit your device while feeding data.

### Transcription

The transcription code executes the full pipeline which includes preprocessing and note segmentation.

```python
python -m vocano.transcription -n [output_file_name] -wd [input_wavfile] 
```

If preprocessing is done before transcription, set `use_pre` flag to skip the step. Device can also be selected by setting the `-d` flag. Cupy can also be enabled by `-use_cp` flag. Set `-bz [batchsize] -nw [num_workers] -pn -al [amp_level]` to fit your device while feeding data and inferencing. 

If you have a better melody transcription model or groundtruth melody, you can also convert extracted melody into our form (see Appendix), and set `-pd [your_pitch_directory]`.

Transcribed results can be found under `VOCANO/generated/wav/` and `VOCANO/generated/midi/`.

## Appendix

### Source separation for polyphonic vocal

Recently, multiple source separation models like [Demucs][Demucs] are developed which gives clear vocal separation results that fits in our model. Vocal files preprocessed by source separation methods are expected to give more promising results on polyphonic audio files. Note that extracted vocal files should be in .wav format. Results transcribed with demucs and our work can be seen in our demo section.

### Using your own melody extraction model

If you wish to use your own melody extraction model, please export your extracted pitch in length of our extracted feature (feature.shape[1]), and data in Hertz unit. The dtype of exported pitch should be np.float64.

## Citation

If you find our work useful, please consider citing our paper.

* VOCANO
```
@inproceedings{vocano,
	title={{VOCANO}: A Note Transcription Framework For Singing Voice In Polyphonic Music},
	author={Hsu, Jui-Yang and Su, Li},
	booktitle={Proc. International Society of Music Information Retrieval Conference (ISMIR)},
	year={2021}
}
``` 

* Omnizart
```
@article{wu2021omnizart,
	title={Omnizart: A General Toolbox for Automatic Music Transcription},
	author={Wu, Yu-Te and Luo, Yin-Jyun and Chen, Tsung-Ping and Wei, I-Chieh and Hsu, Jui-Yang and Chuang, Yi-Chin and Su, Li},
	journal={arXiv preprint arXiv:2106.00497},
	year={2021}
}
```

[Note-Segmentation-SSL]: https://github.com/B05901022/Note-Segmentation-SSL
[Omnizart]: https://github.com/Music-and-Culture-Technology-Lab/omnizart
[Omnizart Website]: https://music-and-culture-technology-lab.github.io/omnizart-doc/
[Music and Culture Technology (MCT) Lab]: https://sites.google.com/view/mctl/home
[NVIDIA-apex]: https://github.com/NVIDIA/apex
[Vocal melody extraction using patch-based CNN]: https://arxiv.org/abs/1804.09202
[Hierarchical classification networks for singing voice segmentation and transcription]: http://archives.ismir.net/ismir2019/paper/000111.pdf
[Our demo googledrive]: https://drive.google.com/drive/folders/1Ebao0fih7JtXHNZ1XCu6WHYQTVsl7c8J?usp=sharing
[CuPy]: https://github.com/cupy/cupy
[Demucs]: https://github.com/facebookresearch/demucs