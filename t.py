import numpy as np
import matplotlib.pyplot as plt
from nilearn import datasets, surface, plotting
from nibabel import load

# Load the brain image data from a given filepath


def load_brats_data(filepath):
    return load(filepath).get_fdata()

# Preprocess the image data by normalizing it


def preprocess(image):
    return (image - np.min(image)) / (np.max(image) - np.min(image))

# Create a multi-slice view of the brain image


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


# Main execution
if __name__ == "_main_":
    filepath = "C:/Users/reddy/Downloads/BraTS2020_TrainingData/MICCAI_BraTS2020_TrainingData/BraTS20_Training_005/BraTS20_Training_005_flair.nii"

    # Load and preprocess the brain image
    brain_image = load_brats_data(filepath)
    preprocessed_image = preprocess(brain_image)

    # Create a multi-slice view and save it as an image
    fig = create_multi_slice_view(preprocessed_image)
    plt.savefig('brainaxialview.png', dpi=300,
                bbox_inches='tight', pad_inches=0)
    plt.close()

    # Get a sample motor activation statistical map
    stat_img = datasets.load_sample_motor_activation_image()

    # Get the fsaverage cortical mesh
    fsaverage = datasets.fetch_surf_fsaverage()

    # Load the curvature data and determine sign for the background map
    curv_right = surface.load_surf_data(fsaverage.curv_right)
    curv_right_sign = np.sign(curv_right)

    # Project the statistical map onto the cortical surface
    texture = surface.vol_to_surf(stat_img, fsaverage.pial_right)

    # Plot the surface statistical map
    fig = plotting.plot_surf_stat_map(
        fsaverage.infl_right, texture, hemi='right',
        title='Surface right hemisphere', colorbar=True,
        threshold=1., bg_map=curv_right_sign,
    )
    fig.show()

    # Check if Plotly is available for interactive plotting
    engine = 'plotly'
    try:
        import plotly.graph_objects as go
    except ImportError:
        engine = 'matplotlib'

    print(f"Using plotting engine {engine}.")

    fig = plotting.plot_surf_stat_map(
        fsaverage.infl_right, texture, hemi='right',
        title='Surface right hemisphere', colorbar=True,
        threshold=1., bg_map=curv_right_sign, bg_on_data=True,
        engine=engine
    )
    fig.show()

    # Plot 3D images for comparison
    plotting.plot_glass_brain(stat_img, display_mode='r', plot_abs=False,
                              title='Glass brain', threshold=2.)

    plotting.plot_stat_map(stat_img, display_mode='x', threshold=1.,
                           cut_coords=range(0, 51, 10), title='Slices')

    # Use an atlas to outline specific regions of interest
    destrieux_atlas = datasets.fetch_atlas_surf_destrieux()
    parcellation = destrieux_atlas['map_right']

    regions_dict = {b'G_postcentral': 'Postcentral gyrus',
                    b'G_precentral': 'Precentral gyrus'}

    regions_indices = [
        np.where(np.array(destrieux_atlas['labels']) == region)[0][0]
        for region in regions_dict
    ]
    labels = list(regions_dict.values())

    # Display outlines of regions of interest on top of the statistical map
    figure = plotting.plot_surf_stat_map(fsaverage.infl_right,
                                         texture, hemi='right',
                                         title='Surface right hemisphere',
                                         colorbar=True, threshold=1.,
                                         bg_map=fsaverage.sulc_right)

    plotting.plot_surf_contours(fsaverage.infl_right, parcellation, labels=labels,
                                levels=regions_indices, figure=figure,
                                legend=True, colors=['g', 'k'])
    plotting.show()

    # Plot with a higher-resolution mesh
    big_fsaverage = datasets.fetch_surf_fsaverage('fsaverage')
    big_texture = surface.vol_to_surf(stat_img, big_fsaverage.pial_right)

    plotting.plot_surf_stat_map(big_fsaverage.infl_right,
                                big_texture, hemi='right', colorbar=True,
                                title='Surface right hemisphere: fine mesh',
                                threshold=1., bg_map=big_fsaverage.sulc_right)

    # Plot multiple views of the 3D volume on a surface
    plotting.plot_img_on_surf(stat_img,
                              views=['lateral', 'medial'],
                              hemispheres=['left', 'right'],
                              colorbar=True)
    plotting.show()

    # 3D visualization in a web browser
    view = plotting.view_surf(fsaverage.infl_right, texture, threshold='90%',
                              bg_map=fsaverage.sulc_right)
    

    view = plotting.view_img_on_surf(stat_img, threshold='90%')

    # Visualize the impact of plot parameters on surface projection
    destrieux = datasets.fetch_atlas_destrieux_2009(legacy_format=False)
    view = plotting.view_img_on_surf(
        destrieux.maps,
        surf_mesh="fsaverage",
        vol_to_surf_kwargs={"n_samples": 1, "radius": 0.0,
                            "interpolation": "nearest"},
        symmetric_cmap=False,
    )
