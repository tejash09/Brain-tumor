import numpy as np
import plotly.graph_objects as go
import plotly.subplots as subplots
from skimage import measure
from skimage import measure
from skimage.draw import ellipsoid
import webbrowser
import os
import nilearn as nl
import matplotlib.pyplot as plt
import nilearn.plotting as nlplt
import json
import requests
import nibabel as nib
import plotly.express as px
from flask import Flask,request,render_template,jsonify
import SimpleITK as sitk

app = Flask(__name__,template_folder='templates')

@app.route('/')
def brain():
    return render_template('brain.html')

@app.route('/about')
def about():
    return render_template('about.html')

@app.route('/patients')
def patients():
    return render_template('patients.html')

@app.route('/upload')
def upload():
    return render_template('upload.html')

@app.route('/brain')
def rain():
    return render_template('brain.html')

@app.route('/c')
def c():
    return render_template('c.html')

@app.route('/receive_data',methods = ['POST'])

#brain_path = "C:\\Users\\srira\\Downloads\\BraTS2020_TrainingData\\MICCAI_BraTS2020_TrainingData\\BraTS20_Training_170\\BraTS20_Training_170_flair.nii"
#seg_path = "C:\\Users\\srira\\Downloads\\BraTS2020_TrainingData\\MICCAI_BraTS2020_TrainingData\\BraTS20_Training_170\\BraTS20_Training_170_seg.nii"
def receive_data():
    data = request.get_json()
    im = nib.load(data['brain_path']).get_fdata()
    seg = nib.load(data['seg_path']).get_fdata()
    
    brain_nifti = nib.load(data['brain_path'])
    seg_nifti = nib.load(data['seg_path'])

    brain_data = brain_nifti.get_fdata()
    seg_data = seg_nifti.get_fdata()

    tumor_mask = seg_data == 1

    num_tumor_voxels = np.sum(tumor_mask)
    voxel_dims = seg_nifti.header.get_zooms()  
    voxel_volume = np.prod(voxel_dims)  
    tumor_volume = num_tumor_voxels * voxel_volume

    tumor_indices = np.argwhere(tumor_mask)
    min_idx = tumor_indices.min(axis=0)
    max_idx = tumor_indices.max(axis=0)
    tumor_dimensions = max_idx - min_idx + 1  
    physical_dimensions = tumor_dimensions * voxel_dims

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
    webbrowser.open_new_tab('brain_tumor_visualization.html')
    niimg = nl.image.load_img(data['brain_path'])
    nimask = nl.image.load_img(data['seg_path'])
    html_view = nlplt.view_img(nimask, bg_img=niimg, cmap='Paired', threshold=0.5)
    html_view.save_as_html('brain_view.html')
    webbrowser.open_new_tab('brain_view.html')
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

    plt.show()


app.run(debug=True)