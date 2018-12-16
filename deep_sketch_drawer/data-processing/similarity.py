# -*- coding: utf-8 -*-

import os
import shutil
from scipy.misc import imread
from scipy import spatial

photoPath = './Edges'
sketchPath = './allSketches'
sketchTargetPath = './Sketches'

for root, dirs, files in os.walk(photoPath):  
    for f in files:
        prefix, suffix = os.path.splitext(f)[0], os.path.splitext(f)[1]
        if suffix == '.jpg':
            selected = None
            cos = -1.0
            print(photoPath + "/" + f)
            assert(os.path.isfile(photoPath + "/" + f))
            edge = imread(photoPath + "/" + f)
            print(edge.shape)
            edge = edge.flatten()
            for i in range(1, 10):
                sketchName = prefix + "-" + str(i) + ".png"
                if os.path.isfile(os.path.join(sketchPath,sketchName)):
                    sketch = imread(sketchPath + "/" + sketchName)
                    print(sketch.shape)
                    sketch = sketch[:,:,0]
                    sketch = sketch.flatten()
                    similarity = spatial.distance.cosine(edge, sketch)
                    print(similarity)
                    if selected is None or similarity > cos:
                        cos = similarity
                        selected = sketchName
            shutil.copy2(sketchPath + "/" + selected, sketchTargetPath)
