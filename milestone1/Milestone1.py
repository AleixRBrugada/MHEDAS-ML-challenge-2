import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from skimage.filters import threshold_otsu
from skimage import morphology as Morpho
from scipy.ndimage import filters as filt
from NiftyIO import readNifty
from VolumeCutBrowser import VolumeCutBrowser

import SimpleITK as sitk 
from radiomics import imageoperations

proj_dir = os.getcwd() #/home/aleix/MHEDAS/ML/challenge2
voi_dir = '/Challenge2_DataSets/LUNA_DataSet/Full_LUNA16_Dataset/VOIs/'
annot_excel = f'{proj_dir}/Challenge2_DataSets/LUNA_DataSet/Sample_Dataset/MetadatabyAnnotation.xlsx'

### Extract VOIs

## First we have to loop through the CTs
image_dir = f'{proj_dir}{voi_dir}image/'
mask_dir = f'{proj_dir}{voi_dir}nodule_mask/'

p_list = []
n_list = []
v_list = []
for file in os.listdir(image_dir):
    s = file.split('_R_')
    patient_id = s[0]
    nodule_id = s[1].split('.')[0]
    
    p_list.append(patient_id)
    n_list.append(nodule_id)
    
    _, metadata = readNifty(f'{image_dir}{file}')
    image = sitk.ReadImage(f'{image_dir}{file}')
    mask = sitk.ReadImage(f'{mask_dir}{file}')
    
    bbox_tuple = imageoperations.checkMask(image, mask)
    bbox_voxel, _ = bbox_tuple
    
    spacing = metadata.spacing           # (x_spacing, y_spacing, z_spacing)
    origin = metadata.origen             # (x_origin, y_origin, z_origin)
    
    # Convert voxel indices to world (mm) coordinates
    x_min_mm = origin[0] + bbox_voxel[0] * spacing[0]
    x_max_mm = origin[0] + bbox_voxel[1] * spacing[0]
    y_min_mm = origin[1] + bbox_voxel[2] * spacing[1]
    y_max_mm = origin[1] + bbox_voxel[3] * spacing[1]
    z_min_mm = origin[2] + bbox_voxel[4] * spacing[2]
    z_max_mm = origin[2] + bbox_voxel[5] * spacing[2]
    
    bbox_physical = [(x_min_mm, x_max_mm), (y_min_mm, y_max_mm), (z_min_mm, z_max_mm)]
    
    volume = (x_max_mm - x_min_mm)*(y_max_mm - y_min_mm)*(z_max_mm - z_min_mm)
    
    v_list.append(volume)
    
# Now we use the lists and convert them to a dataframe.

df = pd.DataFrame({'patient_id': p_list,'nodule_id': n_list, 'VOI': v_list})
df = df.sort_values(by=['patient_id', 'nodule_id'])

def max_voting(group):
    consensus = {}
    for col in group.columns:
        if col not in ['patient_id', 'nodule_id']:  # Skip grouping columns
            mode_result = group[col].mode()
            consensus[col] = mode_result.iloc[0]
            
    # Count malignant votes (Malignancy_value > 3)
    malignant_votes = sum(group['Malignancy_value'] > 3)
    
    # Diagnosis rule: >= 2 votes for malignancy -> Diagnosis=1
    diagnosis = 1 if malignant_votes >= 2 else 0
    
    consensus['Diagnosis'] = diagnosis
    return pd.Series(consensus)

annot_df = pd.read_excel(annot_excel)
grouped = annot_df.groupby(['patient_id', 'nodule_id']).apply(max_voting, include_group=False).reset_index()

grouped.to_excel(f'{proj_dir}/Milestone1.xlsx')

    
    
