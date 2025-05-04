"""
This is the source code for volume visualization

Machine Learning for Precision Medicine
Universitat Autonoma de Barcelona

__author__ = "Debora Gil, Guillermo Torres, Pau Cano"
__license__ = "GPL"
__email__ = "debora,gtorres,pcano@cvc.uab.es"
__year__ = "2023"
"""

### IMPORT PY LIBRARIES
# Python Library 2 manage volumetric data
import numpy as np
# Pyhton standard Visualization Library
import matplotlib.pyplot as plt
# Pyhton standard IOs Library
import os

### IMPORT SESSION FUNCTIONS
#### Session Code Folder (change to your path)
SessionPyFolder=os.getcwd()
os.chdir(SessionPyFolder) #Change Dir 2 load session functions
# .nii Read Data
from NiftyIO import readNifty
# Volume Visualization
from VolumeCutBrowser import VolumeCutBrowser

import SimpleITK as sitk 
from radiomics import imageoperations
import nibabel as nib

######## LOAD DATA

#### Data Folders (change to your path)
SessionDataFolder='/home/aleix/MHEDAS/ML/challenge2/Challenge2_DataSets/LUNA_DataSet/Sample_Dataset'
os.chdir(SessionDataFolder)


CaseFolder='CT'
NiiFile='LIDC-IDRI-0003.nii.gz'


#### Load Intensity Volume
NiiFile=os.path.join(SessionDataFolder,CaseFolder,'image',NiiFile)
niivol,niimetada=readNifty(NiiFile)
#### Load Nodule Mask
NiiFile=os.path.join(SessionDataFolder,CaseFolder,'nodule_mask',NiiFile)
niimask,niimetada=readNifty(NiiFile)

######## VOLUME METADATA
print('Voxel Resolution (mm): ', niimetada.spacing)
print('Volume origin (mm): ', niimetada.origen)
print('Axes direction: ', niimetada.direction)
######## VISUALIZE VOLUMES

### Interactive Volume Visualization 
# Short Axis View
VolumeCutBrowser(niivol)
VolumeCutBrowser(niivol, IMSSeg=niimask)
# Coronal View
VolumeCutBrowser(niivol, Cut='Cor')
# Sagital View
VolumeCutBrowser(niivol, Cut='Sag')

### Generate Bounding Boxes
image = sitk.ReadImage(os.path.join(SessionDataFolder,CaseFolder,'image',NiiFile))
mask = sitk.ReadImage(os.path.join(SessionDataFolder,CaseFolder,'nodule_mask',NiiFile))

bounding_box = imageoperations.checkMask(image, mask)
#print(bounding_box)
# Extract and print the bounding box coordinates
min_x, max_x, min_y, max_y, min_z, max_z = bounding_box[0]
print(bounding_box[0])

### Use inverse of affine Matrix to convert bb coords to voxel coords
nii = nib.load(NiiFile)
affine = nii.affine

# Example world coordinates (replace with your actual ones)
bounding_box_world = np.array([
    [min_x, min_y, min_z, 1],
    [max_x, max_y, max_z, 1]
])

# Invert affine to go from world to voxel
inv_affine = np.linalg.inv(affine)

# Apply inverse affine to world coordinates
bounding_box_voxel = (inv_affine @ bounding_box_world.T).T[:, :3]
bounding_box_voxel = np.round(bounding_box_voxel).astype(int)

print("Bounding box (voxel):", bounding_box_voxel)



### Short Axis (SA) Image 
# Define SA cut
k=int(niivol.shape[2]/2) # Cut at the middle of the volume 
SA=niivol[:,:,k]
# Image
fig1=plt.figure()
plt.imshow(SA,cmap='gray')
plt.close(fig1) #close figure fig1

# Cut Level Sets
levels=[400]
fig1 = plt.figure()
ax1 = fig1.add_subplot(111,aspect='equal') 
ax1.imshow(SA,cmap='gray')
plt.contour(SA,levels,colors='r',linewidths=2)
plt.close("all") #close all plt figures




