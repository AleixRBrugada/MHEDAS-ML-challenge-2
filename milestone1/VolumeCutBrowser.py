# -*- coding: utf-8 -*-
"""
Created on Tue Nov 13 17:52:57 2018

Course on Deep Learning for BioMedicine
Computer Vision Center
Universitat Autonoma de Barcelona

__authors__ = Debora Gil,Esmitt Ramirez, Carles Sánchez,Guillermo Eduardo Torres
__email__ = {debora,csanchez,esmitt.ramirez,gtorres@cvc.uab.es}
__year__ = 2019

"""
import numpy as np
import matplotlib.pyplot as plt
import sys



################################################################################
# 
# EXAMPLE:
# # Load NII Volume
# from BasicVisualization.DICOMViewer import VolumeSlicer
# import BasicIO.NiftyIO
# import os
# ServerDir='Y:\Shared\Guille'; NIIFile='LIDC-IDRI-0305_GT1_1.nii.gz'
# niivol,_=NiftyIO.readNifty(os.path.join(ServerDir,NIIFile))
#
# VolumeCutBrowser(niivol)
        
class VolumeCutBrowser():
 """
   
     # EXAMPLE:
# # Load NII Volume
# from BasicVisualization.DICOMViewer import VolumeSlicer
# import BasicIO.NiftyIO
# import os
# DataDir='C://Data_Session1//Case0016'; 
# NIIFile='LIDC-IDRI-0016_GT1.nii.gz'
# niivol,_=NiftyIO.readNifty(os.path.join(ServerDir,NIIFile))
#
# VolumeCutBrowser(niivol)
 """
 def __init__(self,IMS,idx=None,IMSSeg=None,Cut='SA'):
     self.IMS=IMS
     self.idx=idx
     self.IMSSeg=IMSSeg
     self.drawContour=True
     self.Cut=Cut
     if IMSSeg is None:
         self.drawContour=False
         
     if idx is None:
        if self.Cut=='SA':
           self.idx=int(np.round(self.IMS.shape[2]/2))
        elif self.Cut=='Sag':
           self.idx=int(np.round(self.IMS.shape[0]/2))
        elif self.Cut=='Cor': 
           self.idx=int(np.round(self.IMS.shape[1]/2))
           
         
     
     self.fig, self.ax = plt.subplots()
     self.fig.canvas.mpl_connect('key_press_event', self.press)
     self.DrawScene()
   
 def press(self,event):
    sys.stdout.flush()
    if event.key == 'x':
        self.idx-=1
        self.idx=max(0,self.idx)
        self.DrawScene()
    if event.key == 'z':
        self.idx+=1
        
        if self.Cut=='SA':
           Mx=self.IMS.shape[2]-1
        elif self.Cut=='Sag':
           Mx=self.IMS.shape[0]-1
        elif self.Cut=='Cor': 
           Mx=self.IMS.shape[1]-1
           
        self.idx=min(Mx,self.idx)
        self.DrawScene()

     
 def DrawScene(self):
     self.ax.cla()
     
     if self.Cut=='SA':
         Im=self.IMS[:,:,self.idx]
     elif self.Cut=='Sag':
         Im=np.squeeze(self.IMS[self.idx,:,:])
     elif self.Cut=='Cor':    
         Im=np.squeeze(self.IMS[:,self.idx,:])
         
 
     self.ax.imshow(Im,cmap='gray')
     self.ax.set_title('Cut: '+ str(self.idx) +' Press "x" to decrease; "z" to increase')
     if self.drawContour:
        if self.Cut=='SA':
          Im=self.IMSSeg[:,:,self.idx]
        elif self.Cut=='Sag':
          Im=np.squeeze(self.IMSSeg[self.idx,:,:])
        elif self.Cut=='Cor':    
          Im=np.squeeze(self.IMSSeg[:,self.idx,:])
        self.ax.contour(Im,[0.5],colors='r')
        
     self.fig.canvas.draw()
     plt.show()

#########################################################
def ShowMosaic(images,NRow,NCol):
    fig = plt.figure(figsize=(14, 14))
    NIm=images.shape[2]
    
    for cnt in np.arange(NIm):
        y = fig.add_subplot(NRow, NCol, cnt+1)
        img = images[:,:,cnt]
        y.imshow(img, cmap='gray')
        
        y.axes.get_xaxis().set_visible(False)
        y.axes.get_yaxis().set_visible(False)
