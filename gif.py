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

# Load the NIFTI file
nifti_file_path = "C:\\Users\\reddy\\Downloads\\BraTS2020_TrainingData\\MICCAI_BraTS2020_TrainingData\\BraTS20_Training_170\\BraTS20_Training_170_seg.nii"
nifti_image = nib.load(nifti_file_path)

# Get the data from the NIFTI file
image_data = nifti_image.get_fdata()

# Assuming the tumor mask is binary (1 for tumor, 0 for non-tumor)
tumor_mask = image_data == 1

# Count the number of voxels in the tumor mask
num_tumor_voxels = np.sum(tumor_mask)

# Get voxel dimensions from the header to calculate voxel volume
voxel_dims = nifti_image.header.get_zooms()  # (x_dim, y_dim, z_dim)
voxel_volume = np.prod(voxel_dims)  # Multiply the dimensions to get the volume

# Calculate the tumor volume
tumor_volume = num_tumor_voxels * voxel_volume
print(f'The volume of the tumor is: {tumor_volume} cubic millimeters')
# Load the NIFTI file

# Get the data from the NIFTI file

# Assuming the tumor mask is binary (1 for tumor, 0 for non-tumor)
tumor_mask = image_data == 1

# Get the indices where the tumor exists
tumor_indices = np.argwhere(tumor_mask)

# Get the minimum and maximum indices along each axis
min_idx = tumor_indices.min(axis=0)
max_idx = tumor_indices.max(axis=0)

# Calculate the dimensions of the tumor
tumor_dimensions = max_idx - min_idx + 1  # Add 1 because indices are zero-based

# Get voxel dimensions from the header
voxel_dims = nifti_image.header.get_zooms()  # (x_dim, y_dim, z_dim)

# Calculate the physical dimensions of the tumor
physical_dimensions = tumor_dimensions * voxel_dims

print(f'The dimensions of the tumor are: {physical_dimensions} millimeters')

'''
fig, axes = plt.subplots(nrows=4, figsize=(30, 40))


nlplt.plot_anat(niimg,
      
                axes=axes[0])

nlplt.plot_epi(niimg,

               axes=axes[1])

nlplt.plot_img(niimg,
               axes=axes[2])

nlplt.plot_roi(nimask,
               bg_img=niimg, 
               axes=axes[3], cmap='Paired')

plt.show()'''