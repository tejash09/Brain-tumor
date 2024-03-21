import numpy as np
import plotly.graph_objects as go
import plotly.subplots as subplots
from skimage import measure
from skimage import measure
from skimage.draw import ellipsoid

import nibabel as nib
import plotly.express as px

brain_path = "C:\\Users\\reddy\\Downloads\\BraTS2020_TrainingData\\MICCAI_BraTS2020_TrainingData\\BraTS20_Training_170\\BraTS20_Training_170_flair.nii"
seg_path = "C:\\Users\\reddy\\Downloads\\BraTS2020_TrainingData\\MICCAI_BraTS2020_TrainingData\\BraTS20_Training_170\\BraTS20_Training_170_seg.nii"
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
fig.write_html("brain_tumor_visualization.html")