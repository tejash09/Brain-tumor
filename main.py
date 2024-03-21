import numpy as np
import plotly.graph_objects as go
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d.art3d import Poly3DCollection

from skimage import measure
import nibabel as nib
import plotly.express as px
import os
import cv2
import glob
import PIL
import shutil
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from skimage import data
from skimage.util import montage 
import skimage.transform as skTrans
from skimage.transform import rotate
from skimage.transform import resize
from PIL import Image, ImageOps  


# neural imaging
import nilearn as nl
import nibabel as nib
import nilearn.plotting as nlplt
#pip install git+https://github.com/miykael/gif_your_nifti # nifti to gif 
import gif_your_nifti.core as gif2nif
from plotly.offline import plot

# ml libs
import keras
import keras.backend as K
from keras.callbacks import CSVLogger
import tensorflow as tf
from keras.utils import plot_model
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from keras.models import *
from keras.layers import *
from keras.optimizers import *
from keras.callbacks import ModelCheckpoint, ReduceLROnPlateau, EarlyStopping, TensorBoard
#from tf.keras.layers.experimental import preprocessing


brain_path = "C:\\Users\\reddy\\Downloads\\BraTS2020_TrainingData\\MICCAI_BraTS2020_TrainingData\\BraTS20_Training_170\\BraTS20_Training_170_flair.nii"
seg_path = "C:\\Users\\reddy\\Downloads\\BraTS2020_TrainingData\\MICCAI_BraTS2020_TrainingData\\BraTS20_Training_170\\BraTS20_Training_170_seg.nii"
im = nib.load(brain_path).get_fdata()
seg = nib.load(seg_path).get_fdata()
brain_parts = [
    {'img':im, 'color':'gray', 'level':0},
    {'img':seg, 'color':'purple', 'level':0},
    {'img':seg, 'color':'red', 'level':1},
    {'img':seg, 'color':'yellow', 'level':2},
    {'img':seg, 'color':'blue', 'level':3}
]

meshes = []
for part in brain_parts:
    print(part['color'],part['level'])
    verts, faces, normals, values = measure.marching_cubes(part['img'], part['level'])
    x, y, z = verts.T
    i, j, k = faces.T

    mesh = go.Mesh3d(x=x, y=y, z=z,color=part['color'], opacity=0.5, i=i, j=j, k=k)
    meshes.append(mesh)
bfig = go.Figure(data=meshes)


plot(bfig, filename='brain_plot.html', auto_open=True)
model = keras.models.load_model('../input/modelperclasseval/model_per_class.h5', 
                                   custom_objects={ 'accuracy' : tf.keras.metrics.MeanIoU(num_classes=4),
                                                   "dice_coef": dice_coef,
                                                   "precision": precision,
                                                   "sensitivity":sensitivity,
                                                   "specificity":specificity,
                                                   "dice_coef_necrotic": dice_coef_necrotic,
                                                   "dice_coef_edema": dice_coef_edema,
                                                   "dice_coef_enhancing": dice_coef_enhancing
                                                  }, compile=False)
plt.show()