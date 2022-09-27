import numpy as np

# a = np.load("/data/czm/ReproduceData/co2/Thumos14reduced-I3D-validationset.npy", allow_pickle=True)[0][:, :1024]
a = np.load("/data/czm/ReproduceData/UM/THUMOS14/features/train/rgb/video_validation_0000051.npy", allow_pickle=True)
# b = np.load("/data/czm/ReproduceData/co2/Thumos14reduced-I3D-validationsetv2.npy", allow_pickle=True)[0][:, :1024]
b = np.load("feat_demo_dir/video_validation_0000051-rgb.npy", allow_pickle=True)

norm_a = np.linalg.norm(a, ord=2, axis=-1, keepdims=False)
norm_b = np.linalg.norm(b, ord=2, axis=-1, keepdims=False)
print(norm_a[:10])
print(norm_b[:10])
delta0 = np.sum(np.abs(a[0] - b[0]))
delta1 = np.sum(np.abs(a - b))
delta2 = np.sum(np.abs(a / norm_a.reshape(-1, 1) - b / norm_b.reshape(-1, 1)))
print(f"delta0: {delta0}")
print(f"delta1: {delta1}")
print(f"delta2: {delta2}")
