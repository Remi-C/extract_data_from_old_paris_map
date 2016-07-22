# -*- coding: utf-8 -*-
"""
Created on Fri Jul 22 16:35:20 2016

@author: remi
"""

# testing the skimage.morphology.label function

#importing the test image

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

from scipy.misc import imread,imresize

from skimage.morphology import label
from skimage.measure import regionprops

path = '/media/sf_RemiCura/PROJETS/belleepoque/creation_donnees/jacoubet/Feuille31_extract.tif'

image = imread(path ,1)

bw = image > 120

label_image= label(bw,neighbors=8, background=1)


fig, ax = plt.subplots(ncols=1, nrows=1, figsize=(6, 6))
ax.imshow(label_image, vmin=-1, vmax=1)

for region in regionprops(label_image, ['Area', 'BoundingBox']):
    # skip small images
    if region['Area'] > 0:

        # draw rectangle around segmented coins
        minr, minc, maxr, maxc = region['BoundingBox']
        rect = mpatches.Rectangle((minc, minr), maxc - minc, maxr - minr,
                              fill=False, edgecolor='red', linewidth=2)
        ax.add_patch(rect)


plt.show()