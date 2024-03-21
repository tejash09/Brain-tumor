import nilearn as nl
import matplotlib.pyplot as plt
import nilearn.plotting as nlplt
import SimpleITK as sitk

niimg = nl.image.load_img("C:\\Users\\reddy\\Downloads\\BraTS2020_TrainingData\\MICCAI_BraTS2020_TrainingData\\BraTS20_Training_170\\BraTS20_Training_170_flair.nii")
nimask = nl.image.load_img("C:\\Users\\reddy\\Downloads\\BraTS2020_TrainingData\\MICCAI_BraTS2020_TrainingData\\BraTS20_Training_170\\BraTS20_Training_170_seg.nii")
#html_view = nlplt.view_img(nimask, bg_img=niimg, cmap='Paired', threshold=0.5)
#html_view.save_as_html('brain_view.html')
import nibabel as nib
import numpy as np

nifti_file_path = "C:\\Users\\reddy\\Downloads\\BraTS2020_TrainingData\\MICCAI_BraTS2020_TrainingData\\BraTS20_Training_170\\BraTS20_Training_170_seg.nii"
nifti_image = nib.load(nifti_file_path)

image_data = nifti_image.get_fdata()

tumor_mask = image_data == 1

num_tumor_voxels = np.sum(tumor_mask)

voxel_dims = nifti_image.header.get_zooms()  
voxel_volume = np.prod(voxel_dims)  

tumor_volume = num_tumor_voxels * voxel_volume
print(f'The volume of the tumor is: {tumor_volume} cubic millimeters')


tumor_mask = image_data == 1

tumor_indices = np.argwhere(tumor_mask)

min_idx = tumor_indices.min(axis=0)
max_idx = tumor_indices.max(axis=0)


tumor_dimensions = max_idx - min_idx + 1  


voxel_dims = nifti_image.header.get_zooms()  


physical_dimensions = tumor_dimensions * voxel_dims

print(f'The dimensions of the tumor are: {physical_dimensions} millimeters')
