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
import math
import cv2

import logging

logging.getLogger("tensorflow").disabled = True

import numpy as np
import tensorflow as tf
from PIL import Image
import i3d

_IMAGE_SIZE = 224
_CHUNK_SIZE = 16
_CHECKPOINT_PATHS = {
    "rgb_imagenet": "data/checkpoints/rgb_imagenet/model.ckpt",
    "flow_imagenet": "data/checkpoints/flow_imagenet/model.ckpt",
    # "rgb_imagenet": "data/checkpoints/rgb_scratch/model.ckpt",
    # "flow_imagenet": "data/checkpoints/flow_scratch/model.ckpt",
}
_VARIABLE_SCOPE = {"rgb": "RGB", "flow": "Flow"}


def load_rgb_frames(vn_frames_path):
    frames = []
    imgfiles = sorted(os.listdir(vn_frames_path))
    for imgname in imgfiles:
        img = cv2.imread(os.path.join(vn_frames_path, imgname))[:, :, [2, 1, 0]]
        # img = cv2.resize(img, (224, 224), cv2.INTER_LINEAR)
        img = (img / 255.0) * 2 - 1
        frames.append(img)
    return np.asarray(frames, dtype=np.float32), len(imgfiles)


def load_flow_frames(vn_frames_path):
    frames = []
    allimgfiles = sorted(os.listdir(vn_frames_path))
    ximgs = sorted(list(filter(lambda x: x.startswith("flow_x"), allimgfiles)))
    yimgs = sorted(list(filter(lambda x: x.startswith("flow_y"), allimgfiles)))
    n_frame = len(ximgs)
    assert n_frame == len(yimgs)
    for i in range(n_frame):
        imgx = cv2.imread(os.path.join(vn_frames_path, ximgs[i]), cv2.IMREAD_GRAYSCALE)
        imgy = cv2.imread(os.path.join(vn_frames_path, yimgs[i]), cv2.IMREAD_GRAYSCALE)
        imgx = (imgx / 255.0) * 2 - 1
        imgy = (imgy / 255.0) * 2 - 1
        img = np.asarray([imgx, imgy]).transpose([1, 2, 0])
        frames.append(img)
    return np.asarray(frames, dtype=np.float32), n_frame


def main(unused_argv):
    # >>>>>>>>>>>>>>>>>>>> config >>>>>>>>>>>>>>>>>>>>
    parser = argparse.ArgumentParser()

    print("******--------- Extract I3D features ------*******")
    parser.add_argument("-g", "--GPU", type=int, default=2, help="GPU id")
    parser.add_argument("-of", "--OUTPUT_FEAT_DIR", dest="OUTPUT_FEAT_DIR", type=str, default="feat_demo_dir", help="Output feature path")
    parser.add_argument("-vpf", "--VIDEO_PATH_FILE", type=str, default="video_list.txt", help="input video list")
    parser.add_argument("-vd", "--VIDEO_DIR", type=str, default="frames_demo_dir", help="frame directory")
    parser.add_argument("-et", "--EVAL_TYPE", type=str, default="rgb", choices=["rgb", "flow"])
    parser.add_argument("-bchs", "--BATCH_SIZE", type=int, default=1)

    args = parser.parse_args()
    # <<<<<<<<<<<<<<<<<<<< config <<<<<<<<<<<<<<<<<<<<
    # >>>>>>>>>>>>>>>>>>>> environment initializing >>>>>>>>>>>>>>>>>>>>
    if not os.path.isdir(args.OUTPUT_FEAT_DIR):
        os.makedirs(args.OUTPUT_FEAT_DIR)
    os.environ["CUDA_VISIBLE_DEVICES"] = str(args.GPU)

    tf.logging.set_verbosity(tf.logging.INFO)
    assert args.EVAL_TYPE in ["rgb", "flow"]

    modality_input = tf.placeholder(tf.float32, shape=(args.BATCH_SIZE, _CHUNK_SIZE, _IMAGE_SIZE, _IMAGE_SIZE, 3 if args.EVAL_TYPE == "rgb" else 2))
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
            # n_frame = len([ff for ff in os.listdir(video_path) if ff.endswith(".jpg")])
            # if args.EVAL_TYPE == "flow":
            #     assert n_frame % 2 == 0
            #     n_frame = n_frame // 2
            # print(f"Total frames: {n_frame}")

            if args.EVAL_TYPE == "rgb":
                images_array, n_frame = load_rgb_frames(video_path)
            elif args.EVAL_TYPE == "flow":
                images_array, n_frame = load_flow_frames(video_path)

            features = []
            n_feat = int(n_frame // _CHUNK_SIZE)
            n_batch = math.ceil(n_feat / args.BATCH_SIZE)
            # print(f"n_frame: {n_frame}; n_feat: {n_feat}")
            # print(f"n_batch: {n_batch}")

            for i in range(n_batch):
                s = i * args.BATCH_SIZE * _CHUNK_SIZE
                e = (i + 1) * args.BATCH_SIZE * _CHUNK_SIZE
                if e <= n_frame:
                    input_blobs = images_array[s:e].reshape(args.BATCH_SIZE, _CHUNK_SIZE, _IMAGE_SIZE, _IMAGE_SIZE, -1)  # BATCH_SIZE,_CHUNK_SIZE,224,224,3
                else:  # n_frame may be not divisible by BATCH_SIZE, so to form batches, some frames are reused
                    delta_frame = e - n_frame
                    input_blobs_part1 = images_array[s:]
                    input_blobs_part2 = np.zeros((delta_frame,) + input_blobs_part1.shape[1:])  # images_array[:delta_frame]
                    input_blobs = np.concatenate([input_blobs_part1, input_blobs_part2], axis=0).reshape(args.BATCH_SIZE, _CHUNK_SIZE, _IMAGE_SIZE, _IMAGE_SIZE, -1)
                # >>>>>>>>>>>>>>>>>>>> forward each batch >>>>>>>>>>>>>>>>>>>>
                clip_feature = sess.run(end_feature, feed_dict={modality_input: input_blobs})  # (BATCH_SIZE, 1, 1, 1, 1024)
                clip_feature = np.reshape(clip_feature, (-1, clip_feature.shape[-1]))  # BATCH_SIZE,1024
                features.append(clip_feature)
                pass

            features = np.concatenate(features, axis=0)
            features = features[:n_feat]  # some frames are reused, we should cut

            feat_path = os.path.join(args.OUTPUT_FEAT_DIR, video_name + f"-{args.EVAL_TYPE}.npy")

            print(f"Saving features and probs for video: {video_name} ...")
            # print(features.shape)
            np.save(feat_path, features)

            # print(f"{cnt}: {video_name} has been processed...")
    print("finished!")


if __name__ == "__main__":
    tf.app.run(main)
