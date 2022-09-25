# Copyright 2017 Google Inc.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ============================================================================
"""Loads a sample video and classifies using a trained Kinetics checkpoint."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import os
import argparse

import logging

logging.getLogger("tensorflow").disabled = True

import numpy as np
import tensorflow as tf
import i3d
from PIL import Image

_IMAGE_SIZE = 224
_SAMPLE_VIDEO_FRAMES = 16
_CHECKPOINT_PATHS = {
    "rgb_imagenet": "data/checkpoints/rgb_imagenet/model.ckpt",
    "flow_imagenet": "data/checkpoints/flow_imagenet/model.ckpt",
}
_VARIABLE_SCOPE = {"rgb": "RGB", "flow": "Flow"}


def resize_image(image):
    image = image.resize((_IMAGE_SIZE, _IMAGE_SIZE), Image.BILINEAR)
    image = np.array(image, dtype="float32")
    return image


def main(unused_argv):
    # >>>>>>>>>>>>>>>>>>>> config >>>>>>>>>>>>>>>>>>>>
    parser = argparse.ArgumentParser()

    print("******--------- Extract I3D features ------*******")
    parser.add_argument("-g", "--GPU", type=int, default=2, help="GPU id")
    parser.add_argument("-of", "--OUTPUT_FEAT_DIR", dest="OUTPUT_FEAT_DIR", type=str, default="feat_demo_dir", help="Output feature path")
    parser.add_argument("-vpf", "--VIDEO_PATH_FILE", type=str, default="video_list.txt", help="input video list")
    parser.add_argument("-vd", "--VIDEO_DIR", type=str, default="frames_demo_dir", help="frame directory")
    parser.add_argument("-et", "--EVAL_TYPE", type=str, choices=["rgb", "flow"])
    parser.add_argument("-bchs", "--BATCH_SIZE", type=int, default=40)
    parser.add_argument("-l", "--L", type=int, default=_SAMPLE_VIDEO_FRAMES, help="it seems to mean chunk size")

    args = parser.parse_args()
    # <<<<<<<<<<<<<<<<<<<< config <<<<<<<<<<<<<<<<<<<<
    # >>>>>>>>>>>>>>>>>>>> environment initializing >>>>>>>>>>>>>>>>>>>>
    if not os.path.isdir(args.OUTPUT_FEAT_DIR):
        os.makedirs(args.OUTPUT_FEAT_DIR)
    os.environ["CUDA_VISIBLE_DEVICES"] = str(args.GPU)

    tf.logging.set_verbosity(tf.logging.INFO)
    assert args.EVAL_TYPE in ["rgb", "flow"]

    modality_input = tf.placeholder(tf.float32, shape=(args.BATCH_SIZE, _SAMPLE_VIDEO_FRAMES, _IMAGE_SIZE, _IMAGE_SIZE, 3 if args.EVAL_TYPE == "rgb" else 2))
    with tf.variable_scope(_VARIABLE_SCOPE[args.EVAL_TYPE]):
        modality_model = i3d.InceptionI3d(400, spatial_squeeze=True, final_endpoint="Logits")
        _, end_points = modality_model(modality_input, is_training=False, dropout_keep_prob=1.0)
        end_feature = end_points["avg_pool3d"]
    modality_variable_map = {}
    for variable in tf.global_variables():
        if variable.name.split("/")[0] == _VARIABLE_SCOPE[args.EVAL_TYPE]:
            modality_variable_map[variable.name.replace(":0", "")] = variable
    modality_saver = tf.train.Saver(var_list=modality_variable_map, reshape=True)

    gpu_options = tf.GPUOptions(allow_growth=True)
    with tf.Session(config=tf.ConfigProto(gpu_options=gpu_options)) as sess:
        modality_saver.restore(sess, _CHECKPOINT_PATHS[f"{args.EVAL_TYPE}_imagenet"])
        tf.logging.info(f"{args.EVAL_TYPE} checkpoint restored")
        # >>>>>>>>>>>>>>>>>>>> read videonames >>>>>>>>>>>>>>>>>>>>
        video_list = open(args.VIDEO_PATH_FILE).readlines()
        video_list = [name.strip() for name in video_list]
        for cnt, video_name in enumerate(video_list):
            video_path = os.path.join(args.VIDEO_DIR, video_name)
            feat_path = os.path.join(args.OUTPUT_FEAT_DIR, video_name + ".npy")
            n_frame = len([ff for ff in os.listdir(video_path) if ff.endswith(".jpg")])
            if args.EVAL_TYPE == "flow":
                assert n_frame % 2 == 0
                n_frame = n_frame // 2
            print(f"Total frames: {n_frame}")

            features = []
            n_feat = int(n_frame // 16)
            n_batch = n_feat // args.BATCH_SIZE + 1
            print(f"n_frame: {n_frame}; n_feat: {n_feat}")
            print(f"n_batch: {n_batch}")

            for i in range(n_batch):
                # >>>>>>>>>>>>>>>>>>>> load frames >>>>>>>>>>>>>>>>>>>>
                input_blobs = []
                for j in range(args.BATCH_SIZE):
                    input_blob = []
                    for k in range(args.L):
                        idx = i * args.BATCH_SIZE * args.L + j * args.L + k
                        idx = int(idx)
                        idx = idx % n_frame + 1
                        if args.EVAL_TYPE == "rgb":
                            image = Image.open(os.path.join(video_path, "img_%05d.jpg" % idx))
                            image = resize_image(image)
                        else:
                            image_x = Image.open(os.path.join(video_path, f"flow_x_{idx-1:05d}.jpg"))
                            image_y = Image.open(os.path.join(video_path, f"flow_y_{idx-1:05d}.jpg"))
                            image_x = resize_image(image_x)
                            image_y = resize_image(image_y)
                            image = np.concatenate([np.expand_dims(image_x, axis=-1), np.expand_dims(image_y, axis=-1)], axis=-1)
                        # """
                        # image[:, :, 0] -= 104.0
                        # image[:, :, 1] -= 117.0
                        # image[:, :, 2] -= 123.0
                        ## """
                        # image[:, :, :] -= 127.5
                        # image[:, :, :] /= 127.5
                        image = (image * 2 / 255) - 1  # normalizing to [-1,1]

                        input_blob.append(image)

                    input_blob = np.array(input_blob, dtype="float32")  # _SAMPLE_VIDEO_FRAMES,224,224,3

                    input_blobs.append(input_blob)

                input_blobs = np.array(input_blobs, dtype="float32")  # BATCH_SIZE,SAMPLE_VIDEO_FRAMES,224,224,3
                # >>>>>>>>>>>>>>>>>>>> forward n batch >>>>>>>>>>>>>>>>>>>>
                clip_feature = sess.run(end_feature, feed_dict={modality_input: input_blobs})  # 1,9,1,1,1024
                # <<<<<<<<<<<<<<<<<<<< forward n batch <<<<<<<<<<<<<<<<<<<<
                clip_feature = np.reshape(clip_feature, (-1, clip_feature.shape[-1]))  # 9,1024

                features.append(clip_feature)
                # <<<<<<<<<<<<<<<<<<<< load frames <<<<<<<<<<<<<<<<<<<<

            features = np.concatenate(features, axis=0)
            # features = features[:n_feat:2]  # 16 frames per feature  (since 64-frame snippet corresponds to 8 features in I3D)

            feat_path = os.path.join(args.OUTPUT_FEAT_DIR, video_name + f"-{args.EVAL_TYPE}.npy")

            print(f"Saving features and probs for video: {video_name} ...")
            print(features.shape)
            np.save(feat_path, features)

            print(f"{cnt}: {video_name} has been processed...")


if __name__ == "__main__":
    tf.app.run(main)
