#!/usr/bin/env python
# coding: utf-8

# In[2]:


import numpy as np
import cv2
import os

class SimpleDatasetLoader:
    # store the image preprocessor
    def __init__(self, preprocessors=None):
        self.preprocessors = preprocessors
        
    # if the preprocessors are None, initialize them as an empty list
        if self.preprocessors is None:
            self.preprocessors = []
            
    def load(self, imagePaths, color=0, verbose=-1):
            # initialize the list of features and labels
            # color = 1 for grayscale else read as color
            data = []
            labels = []
            # loop over the input images
            for (i, imagePath) in enumerate(imagePaths):
                # load the image and extract the class label assuming
                # that our path has the following format:
                # /path/to/dataset/{class}/{image}.jpg
                if color==1:
                    image = cv2.imread(imagePath, cv2.IMREAD_GRAYSCALE)
                    label = imagePath.split(os.path.sep)[-2]
                else:
                    image = cv2.imread(imagePath)
                    label = imagePath.split(os.path.sep)[-2]
            
                
                # check to see if our preprocessors are not None
                if self.preprocessors is not None:
                    # loop over the preprocessors and apply each to
                    # the image
                    for p in self.preprocessors:
                        image = p.preprocess(image)
                # treat our processed image as a "feature vector"
                # by updating the data list followed by the labels
                data.append(image)
                labels.append(label)
                
                # show an update every `verbose` images
                if verbose > 0 and i > 0 and (i + 1) % verbose == 0:
                    print("[INFO] processed {}/{}".format(i + 1,
                        len(imagePaths)))
                    
            return (np.array(data), np.array(labels))
    


# In[ ]:




