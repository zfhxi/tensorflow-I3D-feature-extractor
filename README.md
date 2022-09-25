
# tensorflow-I3D-feature-extractor

Based on:

1. [deepmind/kinetics-i3d](https://github.com/deepmind/kinetics-i3d)
2. [JaywongWang/I3D-Feature-Extractor](https://github.com/JaywongWang/I3D-Feature-Extractor)
3. [Finspire13/pytorch-i3d-feature-extraction](https://github.com/Finspire13/pytorch-i3d-feature-extraction)

Currently, I only tested on Thumos14 datasets for the temporal action localization task.

## Setup

My environment:

ArchLinux 5.15.70-1-lts, RTX 3090, nvidia 515.76, CUDA 11, cuDNN 8.5, python3.6, tensorflow1.15, ...

To install the dependencies (the requirements.txt is just for reference only):

```bash
pip install --upgrade pip
pip install nvidia-pyindex
pip install "nvidia-tensorflow[horovod]"
pip install nvidia-tensorboard==1.15
pip install "dm-sonnet<2" "tensorflow-probability==0.8.0"
```

To test your environment:

```bash
# testing the tensorflow
python -c 'import tensorflow as tf;print(tf.test.is_gpu_available())'
# testing the i3d model
python evaluate_sample.py
```
If there're only information or warning messages, don't worry, just ignore them.

Refer to 

1. https://stackoverflow.com/questions/67815471/training-with-tf-1-15-on-rtx-3090
2. https://developer.nvidia.com/blog/accelerating-tensorflow-on-a100-gpus/
3. https://github.com/deepmind/kinetics-i3d/issues/121#issuecomment-1046269913

## Run code

You may need to download checkpoints and test samples from https://github.com/deepmind/kinetics-i3d/tree/master/data.

Assuming you have extracted rgb frames and optical flows of videos, to extract rgb features from thumos14:

```bash
python extracting_feats.py -g=0 -vd=frames_demo_dir -vpf=video_list.txt -of=feat_demo_dir -et=rgb
``` 

* `-g` : the gpu id,
* `-vd` : video frames directory,
* `-vpf` : the file contains the video names,
* `-of` : output directory of feaures,
* `-et` : evaluation type, also means the feature modality.



