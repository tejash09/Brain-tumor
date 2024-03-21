import numpy as np
import plotly.graph_objects as go
import plotly.subplots as subplots
from skimage import measure
from skimage.draw import ellipsoid
import nibabel as nib

# Load brain image and segmentation
brain_path = "C:\\Users\\reddy\\Downloads\\BraTS2020_TrainingData\\MICCAI_BraTS2020_TrainingData\\BraTS20_Training_170\\BraTS20_Training_170_flair.nii"
seg_path = "C:\\Users\\reddy\\Downloads\\BraTS2020_TrainingData\\MICCAI_BraTS2020_TrainingData\\BraTS20_Training_170\\BraTS20_Training_170_seg.nii"
im = nib.load(brain_path).get_fdata()
seg = nib.load(seg_path).get_fdata()

# Create a figure with three 3D subplots
fig = subplots.make_subplots(
    rows=1, cols=3,
    specs=[[{"type": "scene", "rowspan": 1, "colspan": 1},
            {"type": "scene", "rowspan": 1, "colspan": 1},
            {"type": "scene", "rowspan": 1, "colspan": 1}]],
    subplot_titles=["Outline", "Tumor", "Complete"]
)

# Create meshes for different parts of the brain
brain_parts = [
    {'name': 'Brain', 'img': im, 'color': 'gray', 'level': 0},
    {'name': 'Tumor Level 0', 'img': seg, 'color': 'purple', 'level': 0},
    {'name': 'Tumor Level 1', 'img': seg, 'color': 'blue', 'level': 1},
    {'name': 'Tumor Level 2', 'img': seg, 'color': 'yellow', 'level': 2},
    {'name': 'Tumor Level 3', 'img': seg, 'color': 'red', 'level': 3}
]
meshes = []
for part in brain_parts:
    verts, faces, normals, values = measure.marching_cubes(part['img'], part['level'])
    x, y, z = verts.T
    i, j, k = faces.T
    mesh = go.Mesh3d(x=x, y=y, z=z, color=part['color'], opacity=0.5, i=i, j=j, k=k, name=part['name'])
    meshes.append(mesh)
    if 'Brain' not in part['name']:  # Add traces to main figure only if not 'Brain'
        fig.add_trace(mesh, row=1, col=3)

verts, faces, normals, values = measure.marching_cubes(im, 0)
x, y, z = verts.T
i, j, k = faces.T
full_mesh = go.Mesh3d(x=x, y=y, z=z, color='gray', opacity=0.5, i=i, j=j, k=k, name='Brain')
fig.add_trace(full_mesh, row=1, col=1)

verts, faces, normals, values = measure.marching_cubes(seg, 2)
x, y, z = verts.T
i, j, k = faces.T
tumor_mesh = go.Mesh3d(x=x, y=y, z=z, color='red', opacity=0.5, i=i, j=j, k=k, name='Tumor')
fig.add_trace(tumor_mesh, row=1, col=2)

# Define the layout for the legend
legend_layout = go.Layout(
    title="Legend",
    showlegend=True,
    scene=dict(
        xaxis=dict(showticklabels=False),
        yaxis=dict(showticklabels=False),
        zaxis=dict(showticklabels=False)
    )
)

# Add legend traces
legend_traces = [go.Scatter3d(x=[0], y=[0], z=[0], mode='markers', marker=dict(color=part['color']), name=part['name']) for part in brain_parts]
# Add a trace for the outline in the legend
legend_traces.append(go.Scatter3d(x=[0], y=[0], z=[0], mode='markers', marker=dict(color='gray'), name='Outline'))
# Add a trace for the tumor in the legend
legend_traces.append(go.Scatter3d(x=[0], y=[0], z=[0], mode='markers', marker=dict(color='red'), name='Tumor'))

# Create legend figure
legend_fig = go.Figure(data=legend_traces, layout=legend_layout)

# Combine main figure and legend figure
combined_fig = subplots.make_subplots(rows=1, cols=2, column_widths=[0.7, 0.3], subplot_titles=["Main", "Legend"])
combined_fig.add_trace(fig.data, row=1, col=1)
combined_fig.add_trace(legend_fig.data, row=1, col=2)
combined_fig.update_layout(legend=dict(orientation="h", x=0.5, y=-0.1))

# Save the combined layout as an HTML file
combined_fig.write_html("brain_visualization_with_legend.html")
