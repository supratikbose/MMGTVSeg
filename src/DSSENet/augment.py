import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.utils import Sequence


import os
import sys
sys.path.append(os.path.join(os.path.dirname(__file__), "..", ".."))
import src
from src.DSSENet import volume #from pipeline import volume #from vmsseg import volume

from scipy import ndimage
import glob

class DSSENet_Generator(Sequence):
    def __init__(self,
                trainConfigFilePath,
                useDataAugmentationDuringTraining = True,
                batch_size = 1,
                isTestOrValidationFlag = False
                ):
        # Read config file
        
        pass

class images_and_labels_sequence(Sequence):
    def __init__(self, 
                 data_path = '',
                 batch_size = 1, 
                 out_size = [128, 128, 128],
                 lr_flip = False,
                 translate_random = 32.0,   # +/- translation in voxels 
                 rotate_random = 20.0,      # +/- rotation in degrees  
                 scale_random = 0.2,        # +/- spatial scale ratio 
                 change_intensity = 0.1,    # +/- intensity change ratio (after normalization)
                 label_symmetry_map = [[1,1]],
                 labels_to_train = [1]
                 ):        
        self.data_path = data_path
        self.img_fnames = sorted(glob.glob(data_path + '/' + "img*.nii.gz"))
        self.lbl_fnames = sorted(glob.glob(data_path + '/' + "lbl*.nii.gz"))
        self.num_cases = len(self.img_fnames)
        self.out_size = out_size
        self.batch_size = batch_size
        self.translate_random = translate_random
        self.rotate_random = rotate_random
        self.scale_random = scale_random
        self.change_intensity = change_intensity
        self.lr_flip = lr_flip
        self.label_symmetry_map = label_symmetry_map
        self.labels_to_train = labels_to_train

    def __len__(self):
        return self.num_cases // self.batch_size

    def __getitem__(self, idx):

        # keras sequence returns a batch of datasets, not a single case like generator
        batch_X = np.zeros(shape = (self.batch_size, self.out_size[0], self.out_size[1], self.out_size[2]), dtype = np.float32)
        batch_y = np.zeros(shape = (self.batch_size, self.out_size[0], self.out_size[1], self.out_size[2]), dtype = np.int16)

        for i in range(0, self.batch_size):           
            # load case from disk
            X = volume.load_nii_nib(self.img_fnames[idx * self.batch_size + i])
            y = volume.load_nii_nib(self.lbl_fnames[idx * self.batch_size + i])

            # translate, scale, and rotate volume
            if self.translate_random > 0 or self.rotate_random > 0 or self.scale_random > 0:
                X, y = random_transform(X, y, self.rotate_random, self.scale_random, self.translate_random, fast_mode=True)

            # center-crop volume to match model input dimensions
            X = volume.centered_crop(X, self.out_size)
            y = volume.centered_crop(y, self.out_size)

            # pick specific labels to train (if training labels other than 1s and 0s)
            if self.labels_to_train != [1]:
                temp = np.zeros(shape=y.shape, dtype=y.dtype)
                new_label_value = 1
                for lbl in self.labels_to_train:
                    ti = (y == lbl)
                    temp[ti] = new_label_value
                    new_label_value += 1
                y = temp

            # left/right symmetry flip (use only in symmetric anatomical regions, e.g. brain, H&N, pelvis)
            if self.lr_flip and (np.random.random()>0.5):
                X = np.flip(X, 0)
                y = np.flip(y, 0)
                temp = np.zeros(y.shape,dtype=y.dtype)
                for ii in range(len(self.label_symmetry_map)):
                    ti = (y == self.label_symmetry_map[ii][0])
                    temp[ti] = self.label_symmetry_map[ii][1]
                    ti = (y == self.label_symmetry_map[ii][1])
                    temp[ti] = self.label_symmetry_map[ii][0]
                y = temp                              

            # normalize image intensities to zero mean and unit variance
            X = volume.normalize_intensities_ct(X, simple_normalization=True)

            # perturb image intensities
            if self.change_intensity > 0.0:
                X = X * (1 + np.random.uniform(-self.change_intensity,self.change_intensity)) + np.std(X)*np.random.uniform(-self.change_intensity,self.change_intensity)            

            batch_X[i, :] = X
            batch_y[i, :] = y

        batch_X = np.expand_dims(batch_X, -1)
        batch_y = np.expand_dims(batch_y, -1)
        return batch_X, batch_y

def generate_rotation_matrix(rotation_angles_deg):    
    R = np.zeros((3,3))
    theta_x, theta_y, theta_z  = (np.pi / 180.0) * rotation_angles_deg.astype('float64') # convert from degrees to radians
    c_x, c_y, c_z = np.cos(theta_x), np.cos(theta_y), np.cos(theta_z)
    s_x, s_y, s_z = np.sin(theta_x), np.sin(theta_y), np.sin(theta_z)   
    R[0, :] = [c_z*c_y, c_z*s_y*s_x - s_z*c_x, c_z*s_y*c_x + s_z*s_x]
    R[1, :] = [s_z*c_y, s_z*s_y*s_x + c_z*c_x, s_z*s_y*c_x - c_z*s_x]    
    R[2, :] = [-s_y, c_y*s_x, c_y*c_x]    
    return R

def random_transform(img, label, rot_angle = 15.0, scale = 0.05, translation = 0.0, fast_mode=False):
    angles = np.random.uniform(-rot_angle, rot_angle, size = 3) 
    R = generate_rotation_matrix(angles)   
    S = np.diag(1 + np.random.uniform(-scale, scale, size = 3)) 
    A = np.dot(R, S)
    t = np.array(img.shape) / 2.
    t = t - np.dot(A,t) + np.random.uniform(-translation, translation, size=3)
    # interpolate the image channel
    if fast_mode:
        # nearest neighbor (use when CPU is the bottleneck during training)
        img = ndimage.affine_transform(img, matrix = A, offset = t, prefilter = False, mode = 'nearest', order = 0)
    else:
        # linear interpolation
        img = ndimage.affine_transform(img, matrix = A, offset = t, prefilter = False, mode = 'nearest', order = 1)          
    # interpolate the label channel
    label = ndimage.affine_transform(label, matrix = A, offset = t, prefilter = False, mode = 'nearest', order = 0) 
    return (img, label)