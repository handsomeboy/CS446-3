"""Input and output helpers to load in data.
(This file will not be graded.)
"""

import numpy as np
import skimage
import os
from skimage import io


def read_dataset(data_txt_file, image_data_path):
    """Read data into a Python dictionary.

    Args:
        data_txt_file(str): path to the data txt file.
        image_data_path(str): path to the image directory.

    Returns:
        data(dict): A Python dictionary with keys 'image' and 'label'.
            The value of dict['image'] is a numpy array of dimension (N,8,8,3)
            containing the loaded images.

            The value of dict['label'] is a numpy array of dimension (N,1)
            containing the loaded label.

            N is the number of examples in the data split, the exampels should
            be stored in the same order as in the txt file.
    """
    data = {'label':None, 'image':None}
    imgs = []
    with open(data_txt_file, 'r') as f:
        sample_path_label = f.read().splitlines() 
        labels = np.array([int(sample_path_label[i].split(',')[1]) for i in range(len(sample_path_label))])
        samples_fname = np.array([str(sample_path_label[i].split(',')[0]) for i in range(len(sample_path_label))])
    
    for i in range(len(samples_fname)):
        file_path = image_data_path+'/'+samples_fname[i]+'.jpg'
        imgs.append(io.imread(file_path))
    
    data.update({'label': labels.reshape(-1,1), 'image': np.array(imgs)})

    return data

