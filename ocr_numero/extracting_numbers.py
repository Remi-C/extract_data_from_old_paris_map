# -*- coding: utf-8 -*-
"""
Spyder Editor

This is a temporary script file.
"""

from PIL import Image
import sys

import pyocr
import pyocr.builders


import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import matplotlib.patches as patches
import numpy as np


tools = pyocr.get_available_tools()
if len(tools) == 0:
    print("No OCR tool found")
    sys.exit(1)
# The tools are returned in the recommended order of usage
tool = tools[0]
print("Will use tool '%s'" % (tool.get_name()))
# Ex: Will use tool 'libtesseract'

langs = tool.get_available_languages()
print("Available languages: %s" % ", ".join(langs))
lang = langs[3]

print("Will use lang '%s'" % (lang))
# Ex: Will use lang 'fra'
imgpath = '/media/sf_RemiCura/PROJETS/belleepoque/creation_donnees/ocr_numero/j29B_inverted.tif'
#txt = tool.image_to_string(
#    Image.open(imgpath),
#    lang=lang,
#    builder=pyocr.builders.TextBuilder()
#)

#importing image 
src_im = None
src_im = Image.open(imgpath)  
new_size_percent = 0.25
src_im = src_im.resize((int(src_im.size[0]*new_size_percent),int(src_im.size[1]*new_size_percent)), Image.ANTIALIAS)
#rotating image
src_im = src_im.rotate(22, expand=True)


word_boxes = tool.image_to_string(
    src_im,
    lang=lang,
    builder=pyocr.builders.WordBoxBuilder()
)
#
#line_and_word_boxes = tool.image_to_string(
#    Image.open(imgpath), lang="fra",
#    builder=pyocr.builders.LineBoxBuilder()
#)
#
#for w in word_boxes:
#    print(w.position, w.content)
#    print(w.position[1][0]-w.position[0][0],          # width
#            w.position[1][1]-w.position[0][1])
  
fig1 = plt.figure()
ax1 = fig1.add_subplot(111, aspect='equal') 
plt.imshow(src_im)
for w in word_boxes:
    if w.position[1][0]-w.position[0][0] < 500 and w.position[1][1]-w.position[0][1] < 100:
        ax1.add_patch(
            patches.Rectangle(
                (w.position[0][0],w.position[0][1]),   # (x,y)
                w.position[1][0]-w.position[0][0],          # width
                w.position[1][1]-w.position[0][1],          # height
                linewidth=1
            )
        )



#addign plot for found word box


# Digits - Only Tesseract (not 'libtesseract' yet !)
#digits = tool.image_to_string(
#    Image.open('test-digits.png'),
#    lang=lang,
#    builder=pyocr.tesseract.DigitBuilder()
#)