import numpy as np
import matplotlib.pyplot as plt
from nibabel import load


def load_brats_data(filepath):
    return load(filepath).get_fdata()


def preprocess(image):
    return (image - np.min(image)) / (np.max(image) - np.min(image))


def create_multi_slice_view(image, num_slices=36):

    slices = [image[:, :, i] for i in range(
        0, min(140, image.shape[2]), image.shape[2] // num_slices)]

    fig, axes = plt.subplots(5, 5, figsize=(12.5, 12.5))
    plt.subplots_adjust(wspace=-1.2, hspace=-1.2)

    for i, slice_img in enumerate(slices[:25]):
        ax = axes[i // 5, i % 5]
        ax.imshow(slice_img, cmap='gray')
        ax.axis('off')

    plt.tight_layout(pad=0)
    return fig


if __name__ == "__main__":
    filepath = "C:/Users/reddy/Downloads/BraTS2020_TrainingData/MICCAI_BraTS2020_TrainingData/BraTS20_Training_005/BraTS20_Training_005_flair.nii"

    brain_image = load_brats_data(filepath)
    preprocessed_image = preprocess(brain_image)

    fig = create_multi_slice_view(preprocessed_image)
    plt.savefig(f'brainaxialview.png', dpi=300,
                bbox_inches='tight', pad_inches=0)
    plt.close()
