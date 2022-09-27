import os
import numpy as np
import re

# rgb_path = '/data/czm/THUMOS14_VIDEOS/validation_reduced_hw_features_rgb_fps25'
# flow_path = '/data/czm/THUMOS14_VIDEOS/validation_reduced_hw_features_flow_fps25'
subset = ["validation", "test"][1]
HW_FLAG = False
if HW_FLAG:
    # rgb_flow_path = f"/data/czm/THUMOS14_VIDEOS/{subset}_reduced_hw_features_rgbflow_fps25"
    # dest_file = f"/data/czm/THUMOS14_VIDEOS/Thumos14reduced-I3D-{subset}setv2HW.npy"
    pass
else:
    rgb_flow_path = f"/data/czm/THUMOS14_VIDEOS/tf_{subset}_rgbflow_feats"
dest_file = f"/data/czm/THUMOS14_VIDEOS/tf-I3D-{subset}set.npy"
if subset == "validation":
    video_names = np.load("/data/czm/ReproduceData/co2/Thumos14reduced-Annotations/videoname.npy")[:200]
else:
    video_names = np.load("/data/czm/ReproduceData/co2/Thumos14reduced-Annotations/videoname.npy")[200:]


# file_list=sorted(os.listdir(rgb_path))

# for file in file_list:
# vn=re.search(r'(.*)\.',file).group(1)
all_feats = []

for i, vn in enumerate(video_names):
    print(i)
    rgb_feat = np.load(os.path.join(rgb_flow_path, f"{vn.decode()}-rgb.npy"))
    flow_feat = np.load(os.path.join(rgb_flow_path, f"{vn.decode()}-flow.npy"))
    cat_feat = np.concatenate([rgb_feat, flow_feat], axis=1)
    all_feats.append(cat_feat)

all_feats = np.array(all_feats, dtype=object)
np.save(dest_file, all_feats)
print("finished!")
