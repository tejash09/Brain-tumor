import nibabel as nib
import numpy as np

flair_path = 'C:/Users/reddy/Downloads/BraTS2020_TrainingData/MICCAI_BraTS2020_TrainingData/BraTS20_Training_005/BraTS20_Training_005_flair.nii'
t1_path = 'C:/Users/reddy/Downloads/BraTS2020_TrainingData/MICCAI_BraTS2020_TrainingData/BraTS20_Training_005/BraTS20_Training_005_t1.nii'
t2_path = 'C:/Users/reddy/Downloads/BraTS2020_TrainingData/MICCAI_BraTS2020_TrainingData/BraTS20_Training_005/BraTS20_Training_005_t2.nii'
t1ce_path = 'C:/Users/reddy/Downloads/BraTS2020_TrainingData/MICCAI_BraTS2020_TrainingData/BraTS20_Training_005/BraTS20_Training_005_t1ce.nii'
seg_path = 'C:/Users/reddy/Downloads/BraTS2020_TrainingData/MICCAI_BraTS2020_TrainingData/BraTS20_Training_005/BraTS20_Training_005_seg.nii'


def getBrainSizeForVolume(image_volume):
    return np.sum(image_volume > 0)


def getMaskSizesForVolume(image_volume):
    totals = {1: 0, 2: 0, 3: 0, 4: 0}
    for k in range(1, 5):
        totals[k] = np.sum(image_volume == k)
    return totals


def calculateVolumes(flair_path, t1_path, t2_path, t1ce_path, seg_path):
    flair = nib.load(flair_path)
    t1 = nib.load(t1_path)
    t2 = nib.load(t2_path)
    t1ce = nib.load(t1ce_path)
    seg = nib.load(seg_path)

    voxel_volume = np.prod(flair.header.get_zooms())

    flair_volume = getBrainSizeForVolume(flair.get_fdata()) * voxel_volume
    t1_volume = getBrainSizeForVolume(t1.get_fdata()) * voxel_volume
    t2_volume = getBrainSizeForVolume(t2.get_fdata()) * voxel_volume
    t1ce_volume = getBrainSizeForVolume(t1ce.get_fdata()) * voxel_volume

    total_brain_volume = flair_volume

    seg_volumes = getMaskSizesForVolume(seg.get_fdata())
    necrotic_core_volume = seg_volumes[1] * voxel_volume
    edema_volume = seg_volumes[2] * voxel_volume
    enhancing_tumor_volume = seg_volumes[4] * voxel_volume
    total_tumor_volume = sum(seg_volumes.values()) * voxel_volume

    results = {
        "flair_volume": flair_volume,
        "t1_volume": t1_volume,
        "t2_volume": t2_volume,
        "t1ce_volume": t1ce_volume,
        "necrotic_core_volume": necrotic_core_volume,
        "edema_volume": edema_volume,
        "enhancing_tumor_volume": enhancing_tumor_volume,
        "total_tumor_volume": total_tumor_volume,
        "flair_percentage": (flair_volume / total_brain_volume) * 100,
        "t1_percentage": (t1_volume / total_brain_volume) * 100,
        "t2_percentage": (t2_volume / total_brain_volume) * 100,
        "t1ce_percentage": (t1ce_volume / total_brain_volume) * 100,
        "necrotic_core_percentage": (necrotic_core_volume / total_brain_volume) * 100,
        "edema_percentage": (edema_volume / total_brain_volume) * 100,
        "enhancing_tumor_percentage": (enhancing_tumor_volume / total_brain_volume) * 100,
        "total_tumor_percentage": (total_tumor_volume / total_brain_volume) * 100
    }

    return results


volumes = calculateVolumes(flair_path, t1_path, t2_path, t1ce_path, seg_path)
for key, value in volumes.items():
    if 'percentage' in key:
        print(f"{key}: {value:.2f}%")
    else:
        print(f"{key}: {value:.2f}")
