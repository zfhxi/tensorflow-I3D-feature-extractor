# for extracting frames
# ffmpeg -i /data/czm/THUMOS14_VIDEOS/validation_reduced_videos_fps30/video_validation_0000051.mp4  -q:v 1 -r 10 -loglevel error -s 340x256 -vf fps=25 /home/czm/kinetics-i3d-master-mod/frames_demo_dir/video_validation_0000051/img_%05d.jpg
# ffmpeg -i /data/czm/THUMOS14_VIDEOS/validation_reduced_videos_fps30/video_validation_0000051.mp4  -q:v 1 -r 25 -loglevel error -s 340x256 /home/czm/kinetics-i3d-master-mod/frames_demo_dir/video_validation_0000051/img_%05d.jpg

#"""
#python extracting_feats.py -g=0 -of=/data/czm/THUMOS14_VIDEOS/validation_reduced_features_rgbflow -vpf=/data/czm/DatasetProcessing/split_train.txt -vd=/data/czm/THUMOS14_VIDEOS/validation_reduced_frames_rgb_fps25 -et=rgb
#python extracting_feats.py -g=1 -of=/data/czm/THUMOS14_VIDEOS/validation_reduced_features_rgbflow -vpf=/data/czm/DatasetProcessing/split_train.txt -vd=/data/czm/THUMOS14_VIDEOS/validation_reduced_frames_flow_fps25 -et=flow
#
#python extracting_feats.py -g=2 -of=/data/czm/THUMOS14_VIDEOS/test_reduced_features_rgbflow -vpf=/data/czm/DatasetProcessing/split_test.txt -vd=/data/czm/THUMOS14_VIDEOS/test_reduced_frames_rgb_fps25 -et=rgb
#python extracting_feats.py -g=3 -of=/data/czm/THUMOS14_VIDEOS/test_reduced_features_rgbflow -vpf=/data/czm/DatasetProcessing/split_test.txt -vd=/data/czm/THUMOS14_VIDEOS/test_reduced_frames_flow_fps25 -et=flow
#"""