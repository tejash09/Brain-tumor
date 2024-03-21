import sys
import numpy as np
import plotly.graph_objects as go
import plotly.subplots as subplots
from skimage import measure
import nibabel as nib
import os
# Read patient ID from command-line arguments
patient_id = sys.argv[1]

# Define the base path for the data
base_path = "C:\\Users\\reddy\\Downloads\\BraTS2020_TrainingData\\MICCAI_BraTS2020_TrainingData\\"

# Update the brain_path and seg_path variables using the patient ID
brain_path = f"{base_path}BraTS20_Training_{patient_id}\\BraTS20_Training_{patient_id}_flair.nii"
seg_path = f"{base_path}BraTS20_Training_{patient_id}\\BraTS20_Training_{patient_id}_seg.nii"

# Load the image data
im = nib.load(brain_path).get_fdata()
seg = nib.load(seg_path).get_fdata()

fig = subplots.make_subplots(
    rows=1, cols=3,
    specs=[[{"type": "scene", "rowspan": 1, "colspan": 1},
            {"type": "scene", "rowspan": 1, "colspan": 1},
            {"type": "scene", "rowspan": 1, "colspan": 1}]],
    subplot_titles=["Brain Surface", "Tumor", "Complete"]
)

brain_parts = [
    {'img': im, 'color': 'gray', 'level': 0},
    {'img': seg, 'color': 'purple', 'level': 0},
    {'img': seg, 'color': 'blue', 'level': 1},
    {'img': seg, 'color': 'yellow', 'level': 2},
    {'img': seg, 'color': 'red', 'level': 3}
]
meshes = []
legend_labels = {
    'gray': 'Normal Brain Tissue',
    'purple': 'Necrotic Tumor Core',
    'blue': 'Peritumoral Edema',
    'yellow': 'GD-enhancing Tumor',
    'red': 'Non-enhancing Tumor Core'
}

for part in brain_parts:
    verts, faces, normals, values = measure.marching_cubes(part['img'], part['level'])
    x, y, z = verts.T
    i, j, k = faces.T
    mesh = go.Mesh3d(x=x, y=y, z=z, color=part['color'], opacity=0.5, i=i, j=j, k=k,name=legend_labels[part['color']],showlegend=True)
    fig.add_trace(mesh, row=1, col=3)

verts, faces, normals, values = measure.marching_cubes(im, 0)
x, y, z = verts.T
i, j, k = faces.T

full_mesh = go.Mesh3d(x=x, y=y, z=z, color='gray',opacity=0.5, i=i, j=j, k=k)
fig.add_trace(full_mesh,row = 1,col = 1)

verts, faces, normals, values = measure.marching_cubes(im, 0)
x, y, z = verts.T
i, j, k = faces.T

tumor_mesh1 = go.Mesh3d(x=x, y=y, z=z,color='gray', opacity=0.5, i=i, j=j, k=k)

verts, faces, normals, values = measure.marching_cubes(seg, 2)
x, y, z = verts.T
i, j, k = faces.T

tumor_mesh2 = go.Mesh3d(x=x, y=y, z=z, color='red', opacity=0.5, i=i, j=j, k=k)
fig.add_trace(tumor_mesh1,row = 1,col = 2)
fig.add_trace(tumor_mesh2,row = 1,col = 2)

for mesh in meshes:
    fig.add_trace(mesh, row=1, col=3)

fig.update_layout(
    scene_aspectmode='data',
    legend=dict(
        title='Tissue Types',
        orientation='v',
        yanchor='middle',
        y=0.5,
        xanchor='left',
        x=1.15
    )
)
html_filename = f"brain_tumor_visualization_patient_{patient_id}.html"
fig.write_html(html_filename)

output_dir = "results"
os.makedirs(output_dir, exist_ok=True)
html_filename = os.path.join(output_dir, f"brain_tumor_visualization_patient_{patient_id}.html")
