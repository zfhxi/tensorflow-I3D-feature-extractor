# for valset on rgb
# python extracting_feats.py -g=1 -of=/data/czm/THUMOS14_VIDEOS/tf_validation_rgbflow_feats -vd=/data/czm/THUMOS14_VIDEOS/validation_reduced_frames_rgb_fps25 -vpf=/data/czm/DatasetProcessing/split_train.txt -et=rgb
# for valset on flow
# python extracting_feats.py -g=1 -of=/data/czm/THUMOS14_VIDEOS/tf_validation_rgbflow_feats -vd=/data/czm/THUMOS14_VIDEOS/validation_reduced_frames_flow_fps25 -vpf=/data/czm/DatasetProcessing/split_train.txt -et=flow

# for testset on rgb
# python extracting_feats.py -g=0 -of=/data/czm/THUMOS14_VIDEOS/tf_test_rgbflow_feats -vd=/data/czm/THUMOS14_VIDEOS/test_reduced_frames_rgb_fps25 -vpf=/data/czm/DatasetProcessing/split_test.txt -et=rgb
# for testset on flow
# python extracting_feats.py -g=2 -of=/data/czm/THUMOS14_VIDEOS/tf_test_rgbflow_feats -vd=/data/czm/THUMOS14_VIDEOS/test_reduced_frames_flow_fps25 -vpf=/data/czm/DatasetProcessing/split_test.txt -et=flow