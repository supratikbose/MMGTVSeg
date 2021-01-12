# Implementation of DSSE-VNet-AC
# Ref 1: P. Liu et al.: Encoder-Decoder Neural Network With 3D SE and Deep Supervision
# Ref 2: Xu Chen et.al: Learning Active Contour Models for Medical Image Segmentation
# Ref 3: Squeeze and Excitation Networks https://arxiv.org/abs/1709.01507

#Note 1: https://stackoverflow.com/questions/47538391/keras-batchnormalization-axis-clarification
# A note about difference in meaning of axis in np.mean versus in BatchNormalization. When we take the mean along an axis, we collapse
# that dimension and preserve all other dimensions. In your example data.mean(axis=0) collapses the 0-axis, which is the vertical 
# dimension of data. When we compute a BatchNormalization along an axis, we preserve the dimensions of the array, and we normalize 
# with respect to the mean and standard deviation over every other axis. So in your 2D example BatchNormalization with axis=1 is subtracting 
# the mean for axis=0, just as you expect. This is why bn.moving_mean has shape (4,).
# In our case we we should do the batch_normalization along (i.e., preserving) the channel axis

import os
import json
import numpy as np
import sys
import glob
from datetime import datetime

sys.path.append(os.path.join(os.path.dirname(__file__), "..", ".."))
import src

import nibabel as nib
from scipy import ndimage
from scipy.ndimage import morphology
import SimpleITK

import tensorflow as tf
import tensorflow.keras.backend as K
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Conv3D, UpSampling3D, Conv3DTranspose, Activation, Add, Concatenate, BatchNormalization, ELU, SpatialDropout3D, GlobalAveragePooling3D, Reshape, Dense, Multiply,  Permute
from tensorflow.keras import regularizers, metrics
from tensorflow.keras.utils import Sequence
#import tensorflow_addons as tfa

import matplotlib.pyplot as plt
import time

######### Loss functions ########## 
##mean Dice loss (mean of multiple labels with option to ignore zero (background) label)
def dice_coef(y_true, y_pred, smooth = 0.00001, squared_denominator = False, ignore_zero_label = True, data_format = 'channels_last'):
    #For two labels shape=(None, 144, 144, 144, 2)
    num_dim = len(K.int_shape(y_pred)) #Should be 5
    if 'channels_last' == data_format:
        num_labels = K.int_shape(y_pred)[-1] #Should be 2        
    else:
        num_labels = K.int_shape(y_pred)[1] #Should be 2

    reduce_axis = list(range(1, num_dim - 1)) #should be [1, 2, 3]

    #... means as many : as needed 
    # Also  fixing channel index (here channel_last is assumed hard-coded) removes that dimension
    # So y_true : (None, 144, 144, 144, 1) --> (None, 144, 144, 144)
    y_true = y_true[..., 0] if 'channels_last' == data_format else  y_true[:, 0, ...]
    dice = 0.0

    if (ignore_zero_label == True):
        label_range = range(1, num_labels)
    else:
        label_range = range(0, num_labels)
    # In this case zero_label is not ignored and label_range = [0,1]
    for i in label_range:
        #For label 0, softmax output at channel 0 and for label 1, softmax output at channel 0
        #Once again shape of y_pred_b = (None, 144, 144, 144)
        y_pred_b = y_pred[..., i] if 'channels_last' == data_format else  y_pred[:, i, ...]
        # y_true_b is the mask corresponding to label = i, here 0 or 1
        # y_true_b is type casted to type of y_pred
        y_true_b = K.cast(K.equal(y_true, i), K.dtype(y_pred))
        intersection = K.sum(y_true_b * y_pred_b, axis = reduce_axis)        
        if squared_denominator: 
            y_pred_b = K.square(y_pred_b)
        y_true_o = K.sum(y_true_b, axis = reduce_axis)
        y_pred_o =  K.sum(y_pred_b, axis = reduce_axis)     
        d = (2. * intersection + smooth) / (y_true_o + y_pred_o + smooth)
        #K.mean(d) is the dice for label=i, i = 0, 1 
        dice = dice + K.mean(d)
    dice = dice / len(label_range)
    return dice

def dice_loss(data_format = 'channels_last'):
    def loss(y_true, y_pred):
        f = 1 - dice_coef(y_true, y_pred, smooth = 0.00001, squared_denominator = False, ignore_zero_label = False, data_format=data_format)
        return f    
    return loss

def dice_loss_fg(data_format = 'channels_last'):
    def loss(y_true, y_pred):
        f = 1 - dice_coef(y_true, y_pred, smooth = 0.00001, squared_denominator = False, ignore_zero_label = True, data_format=data_format)
        return f
    return loss

def modified_dice_loss(data_format = 'channels_last'):
    def loss(y_true, y_pred):
        f = 1 - dice_coef(y_true, y_pred, smooth = 0.00001, squared_denominator = True, ignore_zero_label = False, data_format=data_format)
        return f
    return loss

def modified_dice_loss_fg(data_format = 'channels_last'):
    def loss(y_true, y_pred):
        f = 1 - dice_coef(y_true, y_pred, smooth = 0.00001, squared_denominator = True, ignore_zero_label = True, data_format=data_format)
        return f
    return loss

def customMultiLabelJaccardDiceAndCategoricalCrossEntropyLossWrapper(alpha_dice = 0.4, smooth = 0.00001,data_format = 'channels_last'):
    def loss(y_true, y_pred):
        #First compute jaccard dice loss  which is dice loss with squared_denominator
        f=1 - dice_coef(y_true, y_pred, smooth = smooth, squared_denominator = True, ignore_zero_label = False, data_format=data_format)
        #Then compute SparseCategoricalCrossentropy loss Using this loss is valid as 
        # y_true is expected to be an integer corresponding to the label [0 for label 0, 1 for label 1,...]
        # and not an one-hot-encoding representation. y_pred here is the predicted probability since it is 
        #output of a softmax layer. So it is NOT logit.
        g = K.sparse_categorical_crossentropy(K.cast(y_true,'int8'), y_pred,
                                            from_logits=False, 
                                            axis=-1 if 'channels_last' == data_format else 1)
        return alpha_dice * f + (1.0 - alpha_dice)*g
    return loss

#################   Metrics ##############
def surface_distance_array(test_labels, gt_labels, sampling=1, connectivity=1):
    input_1 = np.atleast_1d(test_labels.astype(np.bool))
    input_2 = np.atleast_1d(gt_labels.astype(np.bool))      
    conn = morphology.generate_binary_structure(input_1.ndim, connectivity)
    s = input_1 ^ morphology.binary_erosion(input_1, conn)         # ^ is the logical XOR operator
    s_prime = input_2 ^ morphology.binary_erosion(input_2, conn)   # ^ is the logical XOR operator     
    dta = morphology.distance_transform_edt(~s, sampling)
    dtb = morphology.distance_transform_edt(~s_prime, sampling)
    sds = np.concatenate([np.ravel(dta[s_prime!=0]), np.ravel(dtb[s!=0])])        
    msd = sds.mean()
    sd_stdev=sds.std()
    rms = np.sqrt((sds**2).mean())
    hd  = sds.max()
    return msd,sd_stdev,rms, hd, sds

def surface_distance(test_labels, gt_labels, sampling=1, connectivity=1):
    input_1 = np.atleast_1d(test_labels.astype(np.bool))
    input_2 = np.atleast_1d(gt_labels.astype(np.bool))      
    conn = morphology.generate_binary_structure(input_1.ndim, connectivity)
    s = input_1 ^ morphology.binary_erosion(input_1, conn)         # ^ is the logical XOR operator
    s_prime = input_2 ^ morphology.binary_erosion(input_2, conn)   # ^ is the logical XOR operator     
    dta = morphology.distance_transform_edt(~s, sampling)
    dtb = morphology.distance_transform_edt(~s_prime, sampling)
    sds = np.concatenate([np.ravel(dta[s_prime!=0]), np.ravel(dtb[s!=0])])        
    msd = sds.mean()
    sd_stdev=sds.std()
    rms = np.sqrt((sds**2).mean())
    hd  = sds.max()
    return msd, sd_stdev, rms, hd

def surface_distance_multi_label(test, gt, sampling=1):
    labels = np.unique(gt)
    ti = labels > 0
    unique_lbls = labels[ti]
    msd = np.zeros(len(unique_lbls))
    sd_stdev = np.zeros(len(unique_lbls))
    rms = np.zeros(len(unique_lbls))
    hd = np.zeros(len(unique_lbls))
    i = 0
    for lbl_num in unique_lbls:
            ti = (test == lbl_num)
            ti2 = (gt == lbl_num)
            test_mask = np.zeros(test.shape, dtype=np.uint8)
            test_mask[ti] = 1
            gt_mask = np.zeros(gt.shape, dtype=np.uint8)
            gt_mask[ti2] = 1
            msd[i], sd_stdev[i], rms[i], hd[i] = surface_distance(test_mask, gt_mask, sampling)
            i = i + 1
    return unique_lbls, msd, rms, hd

def dice_coef_func(a,b):
    a = a.astype(np.uint8).flatten()
    b = b.astype(np.uint8).flatten()
    dice = (2 * np.sum(np.multiply(a,b))) / (np.sum(a) + np.sum(b))
    return dice
        
def dice_multi_label(test, gt):
    labels = np.unique(gt)
    ti = labels > 0
    unique_lbls = labels[ti]
    dice = np.zeros(len(unique_lbls))
    i = 0
    for lbl_num in unique_lbls:
            ti = (test == lbl_num)
            ti2 = (gt == lbl_num)
            test_mask = np.zeros(test.shape, dtype=np.uint8)
            test_mask[ti] = 1
            gt_mask = np.zeros(gt.shape, dtype=np.uint8)
            gt_mask[ti2] = 1
            dice[i] = dice_coef_func(test_mask, gt_mask)
            i = i + 1
    return dice

def surface_distance_from_nii(test_file, gt_file):    
        test = SimpleITK.ReadImage(test_file)
        test_lbl = np.transpose(SimpleITK.GetArrayFromImage(test), (2,1,0))
        gt = SimpleITK.ReadImage(gt_file)
        gt_lbl = np.transpose(SimpleITK.GetArrayFromImage(gt), (2,1,0))
        lbls, msd, rms, hd = surface_distance_multi_label(test_lbl, gt_lbl, gt.GetSpacing())
        return lbls, msd, rms, hd

def dice_from_nii(test_file, gt_file):    
        test = SimpleITK.ReadImage(test_file)
        test_lbl = np.transpose(SimpleITK.GetArrayFromImage(test), (2,1,0))
        gt = SimpleITK.ReadImage(gt_file)
        gt_lbl = np.transpose(SimpleITK.GetArrayFromImage(gt), (2,1,0))
        dice = dice_multi_label(test_lbl, gt_lbl)
        return dice

def volume(label):
        volume=np.sum(label)
        return volume

def COM(label):
        #Get the coordinates of the nonzero indices
        indices=np.nonzero(label)
        #Take the average of the index values in each direction to get the center of mass
        COMx=np.mean(indices[0])
        COMy=np.mean(indices[1])
        COMz=np.mean(indices[2])
        return COMx, COMy, COMz

###############  data generator ##############
class DSSENet_Generator(Sequence): 
    def __init__(self,
                trainConfigFilePath,
                useDataAugmentationDuringTraining = True, #True for training, not true for CV  specially if we want to merge prediction
                batch_size = 1,
                numCVFolds = 5,
                cvFoldIndex = 0, #Can be between 0 to numCVFolds -1
                isValidationFlag = False, # True for validation
                verbose = False):
        # Read config file
        with open(trainConfigFilePath) as fp:
                self.trainConfig = json.load(fp)
                fp.close() 
        self.useDataAugmentationDuringTraining = useDataAugmentationDuringTraining
        self.batch_size = batch_size
        self.numCVFolds = numCVFolds
        self.cvFoldIndex = cvFoldIndex
        self.isValidationFlag = isValidationFlag
        self.verbose = verbose
        
  
        
        # self.TrainCVPatientList = [(os.path.basename(f)).replace('_ct.nii.gz','') \
        #         for f in glob.glob(self.trainConfig["resampledFilesLocation"] + '/*_ct.nii.gz', recursive=False) ]
        self.TrainCVPatientList = self.trainConfig["trainCVPatientList"]        
        #DO NOT Ranomize patient list as we want to call the same generator for different CV index
        # and do not want to shuffle patient list between different calls
        #######random.shuffle(self.TrainCVPatientList)   
        
        #Based on numCVFolds, CV index and whether this is a training set generator of validation generator
        #determine list of patient names to be used in this data generator
        self.numTrainCVPatients = len(self.TrainCVPatientList)
        self.numCVPatients = self.numTrainCVPatients // self.numCVFolds
        self.numTrainPatients = self.numTrainCVPatients - self.numCVPatients
        
        #Assert cvFoldIndex is between 0 to numCVFolds -1
        assert(self.cvFoldIndex >= 0 and self.cvFoldIndex < self.numCVFolds)
        startIdx_cv = self.cvFoldIndex * self.numCVPatients
        endIdx_cv = (self.cvFoldIndex+1) * self.numCVPatients
        self.list_cvIdx = [*range(startIdx_cv, endIdx_cv)]
        self.list_trainIdx =  [*range(0, startIdx_cv)] + [*range(endIdx_cv, self.numTrainCVPatients)]
        self.cVPatients = [self.TrainCVPatientList[i] for i in self.list_cvIdx]
        self.trainPatients = [self.TrainCVPatientList[i] for i in self.list_trainIdx]
        
        #Is current one a trainData generator or a validation data generator
        if isValidationFlag:
            self.patientNames =  self.cVPatients 
        else:
            self.patientNames =  self.trainPatients 
        self.num_cases = len(self.patientNames)
            
        #The cubes are going to be arranged as depth-index, rowIndex, col-Index    
        self.cube_size = [self.trainConfig["patientVol_Depth"], self.trainConfig["patientVol_Height"], self.trainConfig["patientVol_width"]]
        self.DepthRange = slice(0, self.cube_size[0])
        self.RowRange = slice(0, self.cube_size[1])
        self.ColRange = slice(0, self.cube_size[2])
        if 'channels_last' == self.trainConfig['data_format']:
            self.X_size = self.cube_size+[2] # 2 channel CT and PET
            self.y_size = self.cube_size+[1] # 1 channel output
        else: # 'channels_first'
            self.X_size = [2] + self.cube_size # 2 channel CT and PET
            self.y_size = [1] + self.cube_size # 1 channel output

        if self.verbose:
            print('trainConfigFilePath: ', trainConfigFilePath)            
            print('resampledFilesLocation: ', self.trainConfig["resampledFilesLocation"])
            print('suffixList: ', self.trainConfig["suffixList"])
            print('patientVol_width: ', self.trainConfig["patientVol_width"])
            print('patientVol_Height: ', self.trainConfig["patientVol_Height"])
            print('patientVol_Depth: ', self.trainConfig["patientVol_Depth"]) 
            print('ct_low: ', self.trainConfig["ct_low"])
            print('ct_high: ', self.trainConfig["ct_high"])
            print('pt_low: ', self.trainConfig["pt_low"])
            print('pt_high: ', self.trainConfig["pt_high"])   
            print('labels_to_train: ', self.trainConfig["labels_to_train"])                     
            print('label_names: ', self.trainConfig["label_names"])
            print('lr_flip: ', self.trainConfig["lr_flip"])
            print('label_symmetry_map: ', self.trainConfig["label_names"])
            print('translate_random: ', self.trainConfig["translate_random"])
            print('rotate_random: ', self.trainConfig["rotate_random"])
            print('scale_random: ', self.trainConfig["scale_random"])
            print('change_intensity: ', self.trainConfig["change_intensity"])                 
            print('data_format: ', self.trainConfig['data_format'])

            print('useDataAugmentationDuringTraining: ', self.useDataAugmentationDuringTraining)
            print('batch_size: ', self.batch_size)
            print('numCVFolds: ', self.numCVFolds)
            print('cvFoldIndex: ', self.cvFoldIndex)
            print('isValidationFlag: ', self.isValidationFlag)            
          
            print('TrainCVPatientList: ', self.TrainCVPatientList)        
            print('numTrainCVPatients: ', self.numTrainCVPatients)
            print('numCVPatients: ', self.numCVPatients)
            print('numTrainPatients: ', self.numTrainPatients)
            print('list_cvIdx: ', self.list_cvIdx)
            print('list_trainIdx: ', self.list_trainIdx)
            print('cVPatients: ', self.cVPatients)
            print('trainPatients: ', self.trainPatients)
            if isValidationFlag:
                print('USING VALIDATION SET: ', self.patientNames)
                print('num_validation_cases: ', self.num_cases)
            else:
                print('USING TRAINING SET: ', self.patientNames)
                print('num_training_cases: ', self.num_cases)
            print('DepthRange: ', self.DepthRange)
            print('RowRange: ', self.RowRange)
            print('ColRange: ', self.ColRange)
            

    
    def __len__(self):
        #Note here, in this implementation data augmentation is not actually increasing 
        #number of original cases; instead it is applying random transformation on one
        #of the original case before using it in a training batch. That is why the 
        # # the __len()__ function is not dependent on data augmentation 
        return self.num_cases // self.batch_size

    def getitemExtended(self, idx):
        # keras sequence returns a batch of datasets, not a single case like generator
        #Note that _getitem__() gets called __len__() number of times, passing idx in range 0 <= idx < __len__()
        batch_X = np.zeros(shape = tuple([self.batch_size] + self.X_size), dtype = np.float32)
        batch_y = np.zeros(shape = tuple([self.batch_size] + self.y_size), dtype = np.int16) #np.int16 #np.float32
        ctFiles=[]
        ptFiles=[]
        gtvFiles=[]
        returnNow = False

        for i in range(0, self.batch_size):  
            X = np.zeros(shape = tuple(self.X_size), dtype = np.float32)
            y = np.zeros(shape = tuple(self.y_size), dtype = np.int16) #np.int16     #np.float32   
            # load case from disk
            overallIndex = idx * self.batch_size + i
            fileIndex = overallIndex 

            ctFileName = self.patientNames[fileIndex] + self.trainConfig["suffixList"][0]
            ptFileName = self.patientNames[fileIndex] + self.trainConfig["suffixList"][1]
            gtvFileName = self.patientNames[fileIndex] + self.trainConfig["suffixList"][2]
                        
            #check file existence            
            if os.path.exists(os.path.join(self.trainConfig["resampledFilesLocation"], ctFileName)):
                pass
            else:
                print(os.path.join(self.trainConfig["resampledFilesLocation"], ctFileName), ' does not exist')  
                returnNow = True  
            if os.path.exists(os.path.join(self.trainConfig["resampledFilesLocation"], ptFileName)):
                pass
            else:
                print(os.path.join(self.trainConfig["resampledFilesLocation"], ptFileName), ' does not exist')  
                returnNow = True
            if os.path.exists(os.path.join(self.trainConfig["resampledFilesLocation"], gtvFileName)):
                pass
            else:
                print(os.path.join(self.trainConfig["resampledFilesLocation"], gtvFileName), ' does not exist')  
                returnNow = True

            if returnNow:
                sys.exit() # return batch_X, batch_y, False, 0, 0, 0, 0, 0, 0

            #We are here => returnNow = False
            #Also note #axes: depth, height, width
            ctFiles.append(ctFileName)
            ctData = np.transpose(nib.load(os.path.join(self.trainConfig["resampledFilesLocation"], ctFileName)).get_fdata(), axes=(2,1,0))  
            ptFiles.append(ptFileName)           
            ptData = np.transpose(nib.load(os.path.join(self.trainConfig["resampledFilesLocation"], ptFileName)).get_fdata(), axes=(2,1,0)) 
            gtvFiles.append(gtvFileName)
            gtvData = np.transpose(nib.load(os.path.join(self.trainConfig["resampledFilesLocation"], gtvFileName)).get_fdata(), axes=(2,1,0))   
            
            #Debug code
            if self.verbose:
               minCT = ctData.min()
               maxCT = ctData.max()                 
               minPT = ptData.min()
               maxPT = ptData.max()
               minGTV = gtvData.min()
               maxGTV = gtvData.max()         
               print('BatchId ', idx, ' sampleInBatchId ', i, ' ', ctFileName, ' ', ptFileName, ' ', gtvFileName)
               print('ctData shape-type-min-max: ', ctData.shape, ' ', ctData.dtype, ' ', minCT, ' ', maxCT)
               print('ptData shape-type-min-max: ', ptData.shape, ' ', ptData.dtype, ' ', minPT, ' ', maxPT)
               print('gtvtData shape-type-min-max: ', gtvData.shape, ' ', gtvData.dtype, ' ', minGTV, ' ', maxGTV)  
               print('########################################################################')
            
            #Clamp and normalize CT data <- simple normalization, just divide by 1000
            np.clip(ctData, self.trainConfig["ct_low"], self.trainConfig["ct_high"], out= ctData)
            ctData = ctData / 1000.0 #<-- This will put values between -1 and 3.1
            ctData = ctData.astype(np.float32)        
            #Clamp and normalize PET Data
            np.clip(ptData, self.trainConfig["pt_low"], self.trainConfig["pt_high"], out= ptData)
            #ptData = ptData / 1.0 #<-- Dividing by 10 will put values between 0 and 2.5
            ptData = (ptData - np.mean(ptData))/np.std(ptData)
            ptData = ptData.astype(np.float32)
            #For gtv mask make it integer
            gtvData = gtvData.astype(np.int16) #int16 #float32

            #Apply Data augmentation
            if self.useDataAugmentationDuringTraining:
                # translate, scale, and rotate volume
                if self.trainConfig["translate_random"] > 0 or self.trainConfig["rotate_random"] > 0 or self.trainConfig["scale_random"] > 0:
                    ctData, ptData, gtvData = self.random_transform(ctData, ptData, gtvData, self.trainConfig["rotate_random"], self.trainConfig["scale_random"], self.trainConfig["translate_random"], fast_mode=True)
                # No flipping or intensity modification

            # pick specific labels to train (if training labels other than 1s and 0s)
            if self.trainConfig["labels_to_train"] != [1]:
                temp = np.zeros(shape=gtvData.shape, dtype=gtvData.dtype)
                new_label_value = 1
                for lbl in self.trainConfig["labels_to_train"]:
                    ti = (gtvData == lbl)
                    temp[ti] = new_label_value
                    new_label_value += 1
                gtvData = temp

            #Concatenate CT and PET data  in X and put X  in batch_X; Put GTV in Y and Y in batch_Y
            if 'channels_last' == self.trainConfig['data_format']:
                #Some of the files have extra slices so we fix the range
                X[:,:,:,0] = ctData[self.DepthRange, self.RowRange, self.ColRange]
                X[:,:,:,1] = ptData[self.DepthRange, self.RowRange, self.ColRange]
                y[:,:,:,0] = gtvData[self.DepthRange, self.RowRange, self.ColRange]
            else:
                X[0,:,:,:] = ctData[self.DepthRange, self.RowRange, self.ColRange]
                X[1,:,:,:] = ptData[self.DepthRange, self.RowRange, self.ColRange]
                y[0,:,:,:] = gtvData[self.DepthRange, self.RowRange, self.ColRange]
            
            batch_X[i,:,:,:,:] = X
            batch_y[i,:,:,:,:] = y

        #return batch_X, batch_y, True, minCT, maxCT, minPT, maxPT, minGTV, maxGTV      
        return batch_X, batch_y, ctFiles, ptFiles, gtvFiles

    def __getitem__(self, idx):
        batch_X, batch_y, ctFiles, ptFiles, gtvFiles = self.getitemExtended(idx)
        return batch_X, batch_y

    def generate_rotation_matrix(self,rotation_angles_deg):    
        R = np.zeros((3,3))
        theta_x, theta_y, theta_z  = (np.pi / 180.0) * rotation_angles_deg.astype('float64') # convert from degrees to radians
        c_x, c_y, c_z = np.cos(theta_x), np.cos(theta_y), np.cos(theta_z)
        s_x, s_y, s_z = np.sin(theta_x), np.sin(theta_y), np.sin(theta_z)   
        R[0, :] = [c_z*c_y, c_z*s_y*s_x - s_z*c_x, c_z*s_y*c_x + s_z*s_x]
        R[1, :] = [s_z*c_y, s_z*s_y*s_x + c_z*c_x, s_z*s_y*c_x - c_z*s_x]    
        R[2, :] = [-s_y, c_y*s_x, c_y*c_x]    
        return R

    def random_transform(self, img1, img2, label, rot_angle = 15.0, scale = 0.05, translation = 0.0, fast_mode=False):
        angles = np.random.uniform(-rot_angle, rot_angle, size = 3) 
        R = self.generate_rotation_matrix(angles)   
        S = np.diag(1 + np.random.uniform(-scale, scale, size = 3)) 
        A = np.dot(R, S)
        t = np.array(img1.shape) / 2.
        t = t - np.dot(A,t) + np.random.uniform(-translation, translation, size=3)
        # interpolate the image channel
        if fast_mode:
            # nearest neighbor (use when CPU is the bottleneck during training)
            img1 = ndimage.affine_transform(img1, matrix = A, offset = t, prefilter = False, mode = 'nearest', order = 0)
            img2 = ndimage.affine_transform(img2, matrix = A, offset = t, prefilter = False, mode = 'nearest', order = 0)
        else:
            # linear interpolation
            img1 = ndimage.affine_transform(img1, matrix = A, offset = t, prefilter = False, mode = 'nearest', order = 1)  
            img2 = ndimage.affine_transform(img2, matrix = A, offset = t, prefilter = False, mode = 'nearest', order = 1)        
        # interpolate the label channel
        label = ndimage.affine_transform(label, matrix = A, offset = t, prefilter = False, mode = 'nearest', order = 0) 
        return (img1, img2, label)   


    def displayBatchData(self, batchX, batchY, sampleInBatchId = 0, startSliceId = 26, endSliceId = 31, pauseTime_sec = 0.5):
        numSamplesInBatch = batchX.shape[0]
        depth = batchX.shape[1] if 'channels_last' == self.trainConfig['data_format'] else batchX.shape[2]
        numSlicesToDisplay = endSliceId - startSliceId + 1
        plt.figure(1)#sampleInBatchId+1
        for sliceId in range(startSliceId, endSliceId):
            offset = sliceId - startSliceId
            #Display CT        
            plt.subplot(3, numSlicesToDisplay, offset+1)
            plt.axis('off')
            plt.title('CT_'+str(sliceId), fontsize=8) 
            if 'channels_last' == self.trainConfig['data_format']:
                plt.imshow(batchX[sampleInBatchId, sliceId,:, :, 0])
            else: # 'channel_first'
                plt.imshow(batchX[sampleInBatchId, 0, sliceId,:, :])
            #Display PET        
            plt.subplot(3, numSlicesToDisplay, numSlicesToDisplay + offset+1)
            plt.axis('off')
            plt.title('PT_'+str(sliceId), fontsize=8)
            if 'channels_last' == self.trainConfig['data_format']:
                plt.imshow(batchX[sampleInBatchId, sliceId,:, :, 1])
            else: # 'channel_first'
                plt.imshow(batchX[sampleInBatchId, 1, sliceId,:, :])
            #Display GTV        
            plt.subplot(3, numSlicesToDisplay, 2*numSlicesToDisplay + offset+1)
            plt.axis('off')
            plt.title('GTV_'+str(sliceId), fontsize=8)
            if 'channels_last' == self.trainConfig['data_format']:
                plt.imshow(batchY[sampleInBatchId, sliceId,:, :, 0])
            else: # 'channel_first'
                plt.imshow(batchY[sampleInBatchId, 0, sliceId,:, :])
        plt.show()
        plt.pause(pauseTime_sec)




################### DSSE_VNet ##################
def shape(tensor):
    s = tensor.get_shape()
    return tuple([s[i].value for i in range(0, len(s))])

def getBNAxis(data_format='channels_last'):
    if 'channels_last' == data_format:
        bnAxis = -1 #last axis is channel : (batch, slice, row, col, channel)
    else:
        bnAxis = -4 #4th-last axis is channel : (batch, channel, slice, row, col)
    return bnAxis

#D x H x W x ? ==> D x H x W x f and  strides = (1,1,1)
def ConvBnElu(x, filters, kernel_size = (3,3,3), strides = (1,1,1), kernel_initializer = 'he_normal', padding = 'same', use_bias = False, data_format='channels_last'):
    x = Conv3D(filters = filters, kernel_size = kernel_size, strides = strides, kernel_initializer = kernel_initializer, padding = padding, data_format=data_format, use_bias = use_bias)(x)
    #See note 1 
    x = BatchNormalization(axis=getBNAxis(data_format))(x)
    x=  ELU()(x)
    return x

#D x H x W x f_in ==> D/2 x H/2 x W/2 x 2f_in 
def DownConvBnElu(x, in_filters,  kernel_initializer = 'he_normal', padding = 'same', use_bias = False, data_format='channels_last'):
    x = Conv3D(filters = 2 * in_filters, kernel_size = (2,2,2), strides = (2,2,2), kernel_initializer = kernel_initializer, padding = padding, data_format=data_format, use_bias = use_bias)(x)
    #See note 1 
    x = BatchNormalization(axis=getBNAxis(data_format))(x)
    x=  ELU()(x)
    return x

#D x H x W x f_in ==> 2D x 2H x 2W x f_in/2
def UpConvBnElu(x, in_filters,  kernel_initializer = 'he_normal', padding = 'same', use_bias = False, data_format='channels_last'):
    x = Conv3DTranspose(filters = in_filters // 2, kernel_size = (2,2,2), strides = (2,2,2), kernel_initializer = kernel_initializer, padding = padding, data_format=data_format, use_bias = use_bias)(x)
    #See note 1 
    x = BatchNormalization(axis=getBNAxis(data_format))(x)
    x=  ELU()(x)
    return x

#This is essentially the orange block in upper layers of  Figure 1 of Ref 1:
# Note  2nd ConvBnElu has kernel size = 3x3x3
#D x H x W x f ==> D x H x W x f  
def UpperLayerSingleResidualBlock(x,data_format='channels_last'):
    filters = x.get_shape().as_list()[getBNAxis(data_format)] #x._keras_shape[getBNAxis(data_format)]
    shortcut = x     
    x = ConvBnElu(x, filters=filters, kernel_size = (5,5,5), strides = (1,1,1), data_format=data_format)
    x = ConvBnElu(x, filters=filters, kernel_size = (3,3,3), strides = (1,1,1), data_format=data_format)
    x = ELU()(Add()([x,shortcut]))
    return x

#This is essentially the concatenation of orange (single Residual) block and pink (bottom double residual) block in Figure 1 of Ref 1: 
#D x H x W x f ==> D x H x W x f 
def SingleAndDoubleResidualBlock(x,data_format='channels_last'):
    filters = x.get_shape().as_list()[getBNAxis(data_format)] #x._keras_shape[getBNAxis(data_format)]
    #First single Res block
    shortcut_1 = x
    #Residualblock1:Conv1
    x = ConvBnElu(x, filters=filters, kernel_size = (5,5,5), strides = (1,1,1), data_format=data_format)
    #Residualblock1:Conv2
    x = ConvBnElu(x, filters=filters, kernel_size = (5,5,5), strides = (1,1,1), data_format=data_format)
    x = ELU()(Add()([x,shortcut_1]))
    #Then bottom double Res block
    shortuct_2 = x
    #BottResidualBlock:Conv11
    x = ConvBnElu(x, filters=filters//4, kernel_size = (1,1,1), strides = (1,1,1), data_format=data_format)
    #BottResidualBlock:Conv12
    x = ConvBnElu(x, filters=filters//4, kernel_size = (3,3,3), strides = (1,1,1), data_format=data_format)
    #BottResidualBlock:Conv13
    x = ConvBnElu(x, filters=filters, kernel_size = (3,3,3), strides = (1,1,1), data_format=data_format)
    #BottResidualBlock:Conv21
    x = ConvBnElu(x, filters=filters//4, kernel_size = (1,1,1), strides = (1,1,1), data_format=data_format)
    #BottResidualBlock:Conv22
    x = ConvBnElu(x, filters=filters//4, kernel_size = (3,3,3), strides = (1,1,1), data_format=data_format)
    #BottResidualBlock:Conv23
    x = ConvBnElu(x, filters=filters, kernel_size = (3,3,3), strides = (1,1,1), data_format=data_format)
    x = ELU()(Add()([x, shortuct_2]))
    return x

def DeepSupervision(x, filters=2, upSamplingRatio=1,data_format='channels_last'):
    x = Conv3D(filters = filters, kernel_size = (1,1,1), strides = (1,1,1), kernel_initializer = 'he_normal', padding = 'same', data_format=data_format, use_bias = False)(x)
    x = UpSampling3D(size=(upSamplingRatio, upSamplingRatio, upSamplingRatio), data_format=data_format)(x)
    return x

# https://github.com/titu1994/keras-squeeze-excite-network/blob/master/se.py : Used
def Squeeze_Excite_block(x, ratio=16, data_format='channels_last'):
    ''' Create a channel-wise squeeze-excite block
    Args:
        input: input tensor
        ratio: number of output filters
    Returns: a keras tensor
    '''
    #init = input
    #channel_axis = getBNAxis(data_format) #1 if K.image_data_format() == "channels_first" else -1
    filters =  x.get_shape().as_list()[getBNAxis(data_format)] #x._keras_shape[getBNAxis(data_format)] #init._keras_shape[channel_axis] ##x.get_shape().as_list()[getBNAxis(data_format)] 
    # Note se_shape is the target shape with out considering the batch_size rank
    # In the 2D case it was : se_shape = (1, 1, filters)  
    se_shape = (1, 1, 1, filters) 

    se = GlobalAveragePooling3D(data_format=data_format)(x) #In the 2D case it was : GlobalAveragePooling2D()(init)
    se = Reshape(se_shape)(se)
    se = Dense(filters // ratio, activation='relu', kernel_initializer='he_normal', use_bias=False)(se)
    se = Dense(filters, activation='sigmoid', kernel_initializer='he_normal', use_bias=False)(se)

    if 'channels_first' == data_format: #K.image_data_format() == 'channels_first':
        # Note : dims: Tuple of integers. Permutation pattern, does not include the  samples (i.e., batch) dimension. Indexing starts at 1.
        # In the 2D case it was : se = Permute((3, 1, 2))(se)
        se = Permute((4, 1, 2, 3))(se)

    x = Multiply()([x, se])
    return x

# 16*9 = 144
# 144 => 72  => 36  => 18  => 9
#  32 => 16  => 8   => 4   => 2
# OR
#  16 => 8  => 4   => 2   => 1
def DSSEVNet(input_shape, dropout_prob = 0.25, data_format='channels_last'):

    ########## Encode path ##########       (using the terminology from Ref1)
    #InTr  D x H x W x C ==> D x H x W x 16
    #  16 x 144 x 144 x 2 channel
	
	#>>>>>>>> <tf.Tensor 'input_1:0' shape=(None, 16, 144, 144, 2) dtype=float32>
    img_input = Input(shape = input_shape) # (Nc, D, H, W) if channels_first else  (D, H, W, Nc) Note batch_size is not included
    # if the input has more than 1 channel it has to be expanded because broadcasting only works for 1 input
    # channel
	#>>>>>>> (144, 144, 144, 2)
    input_channels =  input_shape[-1] if 'channels_last' == data_format else input_shape[-4] #config["inputChannels"]
    tile_tensor    =  [1,1,1,1,16]    if 'channels_last' == data_format else [1,16,1,1,1]    #config["inputChannels"]
    if 1 == input_channels:
        sixteen_channelInput = tf.tile(img_input,tile_tensor)
    else:
		#>>>>>> <tf.Tensor 'elu/Identity:0' shape=(None, 144, 144, 144, 16) dtype=float32>
        sixteen_channelInput = ConvBnElu(img_input, filters=16, kernel_size = (5,5,5), strides = (1,1,1),  data_format=data_format)
    #In Table 1 of Ref 1, stride of 1x1x5 was mentioned for conv1, but we are sicking to stride 1x1x1; And here conv1 step includes add + elu
	#>>>>>>> <tf.Tensor 'elu_1/Identity:0' shape=(None, 144, 144, 144, 16) dtype=float32>
    _InTr =  ELU()(Add()([ConvBnElu(sixteen_channelInput, filters=16, kernel_size = (5,5,5), strides = (1,1,1),  data_format=data_format), sixteen_channelInput]))
	#>>>>>> <tf.Tensor 'spatial_dropout3d/Identity:0' shape=(None, 16, 144, 144, 16) dtype=float32>
    _InTrDropout = SpatialDropout3D(rate=dropout_prob, data_format='channels_last')(_InTr)

    #DownTr32  D x H x W x 16 ==> D/2 x H/2 x W/2 x 32  
	#>>>>>> <tf.Tensor 'elu_3/Identity:0' shape=(None, 72, 72, 72, 32) dtype=float32>
    _DownTr32  =  DownConvBnElu(x=_InTr, in_filters=16,  data_format=data_format)
	#>>>>>> <tf.Tensor 'elu_6/Identity:0' shape=(None, 72, 72, 72, 32) dtype=float32>
    _DownTr32  =  UpperLayerSingleResidualBlock(x=_DownTr32, data_format=data_format)
	#>>>>>> <tf.Tensor 'elu_9/Identity:0' shape=(None, 72, 72, 72, 32) dtype=float32>
    _DownTr32  =  UpperLayerSingleResidualBlock(x=_DownTr32, data_format=data_format)
	#>>>>>> <tf.Tensor 'multiply/Identity:0' shape=(None, 72, 72, 72, 32) dtype=float32>
    _DownTr32  =  Squeeze_Excite_block(x=_DownTr32, ratio=8, data_format=data_format)
	#>>>>>> <tf.Tensor 'spatial_dropout3d_1/Identity:0' shape=(None, 72, 72, 72, 32) dtype=float32>
    _DownTr32Dropout = SpatialDropout3D(rate=dropout_prob, data_format='channels_last')(_DownTr32)

    #DownTr64   D/2 x H/2 x W/2 x 32 ==> D/4 x H/4 x W/4  x 64  
	#>>>>>> <tf.Tensor 'elu_10/Identity:0' shape=(None, 36, 36, 36, 64) dtype=float32>
    _DownTr64  =  DownConvBnElu(x=_DownTr32, in_filters=32,  data_format=data_format)
    #>>>>>> <tf.Tensor 'elu_20/Identity:0' shape=(None, 36, 36, 36, 64) dtype=float32>	
    _DownTr64  =  SingleAndDoubleResidualBlock(x=_DownTr64,  data_format=data_format)
	#>>>>>> <tf.Tensor 'multiply_1/Identity:0' shape=(None, 36, 36, 36, 64) dtype=float32>
    _DownTr64  =  Squeeze_Excite_block(x=_DownTr64, ratio=8, data_format=data_format)
	#>>>>>> <tf.Tensor 'spatial_dropout3d_2/Identity:0' shape=(None, 36, 36, 36, 64) dtype=float32>
    _DownTr64Dropout = SpatialDropout3D(rate=dropout_prob, data_format='channels_last')(_DownTr64)

     #DownTr128   D/4 x H/4 x W/4  x 64 ==> D/8 x H/8 x W/8 x 128
	#>>>>>> <tf.Tensor 'elu_21/Identity:0' shape=(None, 18, 18, 18, 128) dtype=float32>
    _DownTr128  =  DownConvBnElu(x=_DownTr64, in_filters=64,  data_format=data_format)  
	#>>>>>> <tf.Tensor 'elu_31/Identity:0' shape=(None, 18, 18, 18, 128) dtype=float32>
    _DownTr128  =  SingleAndDoubleResidualBlock(x=_DownTr128,  data_format=data_format)
	#>>>>>> <tf.Tensor 'multiply_2/Identity:0' shape=(None, 18, 18, 18, 128) dtype=float32>
    _DownTr128  =  Squeeze_Excite_block(x=_DownTr128, ratio=8, data_format=data_format)
	#>>>>>> <tf.Tensor 'spatial_dropout3d_3/Identity:0' shape=(None, 18, 18, 18, 128) dtype=float32>
    _DownTr128Dropout = SpatialDropout3D(rate=dropout_prob, data_format='channels_last')(_DownTr128)

    #DownTr256   D/8 x H/8 x W/8 x 128 ==> D/16 x H/16 x W/16 x 256
	#>>>>>> <tf.Tensor 'elu_32/Identity:0' shape=(None, 9, 9, 9, 256) dtype=float32>
    _DownTr256  =  DownConvBnElu(x=_DownTr128, in_filters=128,  data_format=data_format)  
	#>>>>>> <tf.Tensor 'elu_42/Identity:0' shape=(None, 9, 9, 9, 256) dtype=float32>
    _DownTr256  =  SingleAndDoubleResidualBlock(x=_DownTr256, data_format=data_format)
	#>>>>>> <tf.Tensor 'multiply_3/Identity:0' shape=(None, 9, 9, 9, 256) dtype=float32>
    _DownTr256  =  Squeeze_Excite_block(x=_DownTr256, ratio=8, data_format=data_format)       


    ########## Dncode path ##########
    #UpTr256    D/16 x H/16 x W/16 x 256 ==> D/8 x H/8 x W/8 x 128 => D/8 x H/8 x W/8 x 256 (due to concatenation)
	#>>>>>> <tf.Tensor 'elu_43/Identity:0' shape=(None, 18, 18, 18, 128) dtype=float32>
    _UpTr256  = UpConvBnElu(_DownTr256, in_filters=256, data_format=data_format)
	#>>>>>> <tf.Tensor 'concatenate/Identity:0' shape=(None, 18, 18, 18, 256) dtype=float32>
    _UpTr256  = Concatenate(axis = getBNAxis(data_format))([_UpTr256,_DownTr128Dropout])
	#>>>>>> <tf.Tensor 'elu_53/Identity:0' shape=(None, 18, 18, 18, 256) dtype=float32>
    _UpTr256  =  SingleAndDoubleResidualBlock(x=_UpTr256, data_format=data_format)
	#>>>>>> <tf.Tensor 'multiply_4/Identity:0' shape=(None, 18, 18, 18, 256) dtype=float32>
    _UpTr256  =  Squeeze_Excite_block(x=_UpTr256, ratio=8, data_format=data_format)
    #Also Dsv4 D/8 x H/8 x W/8 x 256 => D x H x W x 4
	#>>>>>> <tf.Tensor 'up_sampling3d/Identity:0' shape=(None, 144, 144, 144, 4) dtype=float32>
    _Dsv4 = DeepSupervision(_UpTr256, filters=4, upSamplingRatio=8, data_format=data_format)


    #UpTr128    D/8 x H/8 x W/8 x 256 ==> D/4 x H/4 x W/4 x 64 => D/4 x H/4 x W/4 x 128 (due to concatenation)
	#>>>>>> <tf.Tensor 'elu_54/Identity:0' shape=(None, 36, 36, 36, 64) dtype=float32>
    _UpTr128  = UpConvBnElu(_UpTr256, in_filters=128, data_format=data_format)
	#>>>>>> <tf.Tensor 'concatenate_1/Identity:0' shape=(None, 36, 36, 36, 128) dtype=float32>
    _UpTr128  = Concatenate(axis = getBNAxis(data_format))([_UpTr128,_DownTr64Dropout])
	#>>>>>> <tf.Tensor 'elu_64/Identity:0' shape=(None, 36, 36, 36, 128) dtype=float32>
    _UpTr128  =  SingleAndDoubleResidualBlock(x=_UpTr128, data_format=data_format)
	#>>>>>> <tf.Tensor 'multiply_5/Identity:0' shape=(None, 36, 36, 36, 128) dtype=float32>
    _UpTr128  =  Squeeze_Excite_block(x=_UpTr128, ratio=8, data_format=data_format)
    #Also Dsv3 D/4 x H/4 x W/4 x 128 => D x H x W x 4
	#>>>>>> <tf.Tensor 'up_sampling3d_1/Identity:0' shape=(None, 144, 144, 144, 4) dtype=float32>
    _Dsv3 = DeepSupervision(_UpTr128, filters=4, upSamplingRatio=4, data_format=data_format)

    #UpTr64    D/4 x H/4 x W/4 x 128 ==> D/2 x H/2 x W/2 x 32 => D/2 x H/2 x W/2 x 64 (due to concatenation)
	#>>>>>> <tf.Tensor 'elu_65/Identity:0' shape=(None, 72, 72, 72, 32) dtype=float32>
    _UpTr64  = UpConvBnElu(_UpTr128, in_filters=64, data_format=data_format)
	#>>>>>> <tf.Tensor 'concatenate_2/Identity:0' shape=(None, 8, 72, 72, 64) dtype=float32>
    _UpTr64  = Concatenate(axis = getBNAxis(data_format))([_UpTr64,_DownTr32Dropout])
	#>>>>>> <tf.Tensor 'elu_75/Identity:0' shape=(None, 72, 72, 72, 64) dtype=float32>
    _UpTr64  =  SingleAndDoubleResidualBlock(x=_UpTr64, data_format=data_format)
	#>>>>>> <tf.Tensor 'multiply_6/Identity:0' shape=(None, 72, 72, 72, 64) dtype=float32>
    _UpTr64  =  Squeeze_Excite_block(x=_UpTr64, ratio=8, data_format=data_format)
    #Also Dsv2 D/2 x H/2 x W/2 x 64 => D x H x W x 4
	#>>>>>> <tf.Tensor 'up_sampling3d_2/Identity:0' shape=(None, 144, 144, 144, 4) dtype=float32>
    _Dsv2 = DeepSupervision(_UpTr64, filters=4, upSamplingRatio=2, data_format=data_format)

    #UpTr32    D/2 x H/2 x W/2 x 64 ==> D x H x W x 16 => D x H x W x 32 (due to concatenation)
	#>>>>>> <tf.Tensor 'elu_76/Identity:0' shape=(None, 144, 144, 144, 16) dtype=float32>
    _UpTr32  = UpConvBnElu(_UpTr64, in_filters=32, data_format=data_format)
	#>>>>>> <tf.Tensor 'concatenate_3/Identity:0' shape=(None, 144, 144, 144, 32) dtype=float32>
    _UpTr32  = Concatenate(axis = getBNAxis(data_format))([_UpTr32,_InTrDropout])
	#>>>>>> <tf.Tensor 'elu_79/Identity:0' shape=(None, 144, 144, 144, 32) dtype=float32>
    _UpTr32  =   UpperLayerSingleResidualBlock(x=_UpTr32, data_format=data_format)
	#>>>>>> <tf.Tensor 'elu_82/Identity:0' shape=(None, 144, 144, 144, 32) dtype=float32>
    _UpTr32  =   UpperLayerSingleResidualBlock(x=_UpTr32, data_format=data_format)
	#>>>>>> <tf.Tensor 'multiply_7/Identity:0' shape=(None, 144, 144, 144, 32) dtype=float32>
    _UpTr32  =  Squeeze_Excite_block(x=_UpTr32, ratio=8, data_format=data_format)
    #Also Dsv1 D x H x W x 32 => D x H x W x 4
	#>>>>>> <tf.Tensor 'up_sampling3d_3/Identity:0' shape=(None, 144, 144, 144, 4) dtype=float32>
    _Dsv1 = DeepSupervision(_UpTr32, filters=4, upSamplingRatio=1, data_format=data_format)

    #Final concatenation and convolution
    #144 x 144 x 144 x 4 ==> 144 x 144 x 144 x 16
	#>>>>>> <tf.Tensor 'concatenate_4/Identity:0' shape=(None, 144, 144, 144, 16) dtype=float32>
    _DsvConcat = Concatenate(axis = getBNAxis(data_format))([_Dsv1, _Dsv2, _Dsv3, _Dsv4])
    #144 x 144 x 144 x 16 ==> 144 x 144 x 144 x 2
	#>>>>>> <tf.Tensor 'conv3d_66/Identity:0' shape=(None, 144, 144, 144, 2) dtype=float32>
    #We are going to use filters = 2 for two classes (0 and 1) and softmax as the activation
    #And we will use categorical accuracy for metric and modified dice loss as loss
    _Final = Conv3D(filters = 2, kernel_size = (1,1,1), strides = (1,1,1), kernel_initializer = 'he_normal', padding = 'same', activation='softmax', data_format=data_format, use_bias = False)(_DsvConcat)

    # model instantiation
    model = Model(img_input, _Final)
    return model

def sanityCheckTrainParams(trainInputParams):
    if 'loss_func' not in trainInputParams:
        trainInputParams['loss_func'] = customMultiLabelJaccardDiceAndCategoricalCrossEntropyLossWrapper #modified_dice_loss
    elif trainInputParams['loss_func'].lower() == 'dice_loss':
        trainInputParams['loss_func'] = dice_loss
    elif trainInputParams['loss_func'].lower() == 'modified_dice_loss':
        trainInputParams['loss_func'] = modified_dice_loss  
    elif trainInputParams['loss_func'].lower() == 'dice_loss_fg':
        trainInputParams['loss_func'] = dice_loss_fg
    elif trainInputParams['loss_func'].lower() == 'modified_dice_loss_fg':
        trainInputParams['loss_func'] = modified_dice_loss_fg  
    if 'acc_func' not in trainInputParams:
        trainInputParams['acc_func'] = metrics.categorical_accuracy
    elif trainInputParams['acc_func'].lower() == 'categorical_accuracy':
        trainInputParams['acc_func'] = metrics.categorical_accuracy
    if 'labels_to_train' not in trainInputParams:
        trainInputParams['labels_to_train'] = [1]
    if 'asymmetric' not in trainInputParams:
        trainInputParams['asymmetric'] = True
    ############## NOT USED, HARD CODED WITHIN DSSE-NET, CHANGE LATER #########
    if 'group_normalization' not in trainInputParams:
        trainInputParams['group_normalization'] = False
    if 'activation_type' not in trainInputParams:
        trainInputParams['activation_type'] = 'relu'
    if 'final_activation_type' not in trainInputParams:
        trainInputParams['final_activation_type'] = 'softmax'
    ############################################################################
    if 'AMP' not in trainInputParams:
        trainInputParams['AMP'] = False
    if 'XLA' not in trainInputParams:
        trainInputParams['XLA'] = False   

    return trainInputParams

############### Model train and evaluate function #################
def trainFold(trainConfigFilePath = '/home/user/DMML/CodeAndRepositories/MMGTVSeg/input/trainInput_DSSENet.json',
              saveModelDirectory = '/home/user/DMML/CodeAndRepositories/MMGTVSeg/output/DSSEModels',
              logDir= '/home/user/DMML/CodeAndRepositories/MMGTVSeg',
              cvFoldIndex=0, 
              numCVFolds = 5):
    # load trainInputParams  from JSON config files
    with open(trainConfigFilePath) as f:
        trainInputParams = json.load(f)
        f.close()

    trainInputParams = sanityCheckTrainParams(trainInputParams)

    #Original
    # determine number of available GPUs
    gpus = tf.config.list_physical_devices('GPU') 
    num_gpus = len(gpus)    
    print('Number of GPUs available  for training: ', num_gpus)
    #If using CPUs only for training (set it true if getting GPU memory allocation error)
    if True == trainInputParams['cpuOnlyFlag']:
        #Hide GPUs
        if num_gpus > 0: #gpus:
            print("Restricting TensorFlow to use CPU only by hiding GPUs.")
            ####Alternate: print("Restricting TensorFlow to only use the first GPU.")
            try:
                tf.config.experimental.set_visible_devices([], 'GPU')
                ####Alternate: tf.config.experimental.set_visible_devices(gpus[0], 'GPU')
                logical_gpus = tf.config.experimental.list_logical_devices('GPU')
                print(len(gpus), "Physical GPUs,", len(logical_gpus), "Logical GPU")
            except RuntimeError as e:
                # Visible devices must be set before GPUs have been initialized
                print(e)       
            pass
    else:    
        # prevent tensorflow from allocating all available GPU memory
        if (num_gpus > 0):
            for gpu in gpus:
                tf.config.experimental.set_memory_growth(gpu, True)

    num_cpus = min(os.cpu_count(), 24)   # don't use more than 16 CPU threads
    print('Number of CPUs used for training: ', num_cpus)

    if (trainInputParams['AMP']):
        os.environ['TF_ENABLE_AUTO_MIXED_PRECISION'] = '1'
        os.environ['CUDNN_TENSOROP_MATH'] = '1'
        print('Using Automatic Mixed Precision (AMP) Arithmentic...')

    if (trainInputParams['XLA']):
        tf.config.optimizer.set_jit(True)

    #Make sure saveModelDirectory exists and it is a directory
    #If it does not exist, create it.
    if os.path.exists(saveModelDirectory):
        #Check if it is a directory or not
        if os.path.isfile(saveModelDirectory): 
            print('Error: ', saveModelDirectory,  ' points to a file. It should be a directory. Exisitng')
            sys.exit()    
    else:
        #create 
        os.makedirs(saveModelDirectory)
    #We are here - so saveModelDirectory is a directory

    train_sequence = DSSENet_Generator(trainConfigFilePath = trainConfigFilePath, 
                                        useDataAugmentationDuringTraining = True,
                                        batch_size = 1,
                                        numCVFolds = numCVFolds,
                                        cvFoldIndex = cvFoldIndex, #Can be between 0 to 4
                                        isValidationFlag = False,
                                        verbose=False
                                                )

    val_sequence = DSSENet_Generator(trainConfigFilePath = trainConfigFilePath, 
                                        useDataAugmentationDuringTraining = False,
                                        batch_size = 1,
                                        numCVFolds = numCVFolds,
                                        cvFoldIndex = cvFoldIndex, #Can be between 0 to 4
                                        isValidationFlag = True,
                                        verbose=False
                                        )
    
    # count number of training and test cases
    num_train_cases = train_sequence.__len__()
    num_val_cases = val_sequence.__len__()

    print('Number of train cases: ', num_train_cases)
    print('Number of test cases: ', num_val_cases)
    print("labels to train: ", trainInputParams['labels_to_train'])
    
    sampleCube_dim = [trainInputParams["patientVol_Depth"], trainInputParams["patientVol_Height"], trainInputParams["patientVol_width"]]
    if 'channels_last' == trainInputParams['data_format']:
        input_shape = tuple(sampleCube_dim+[2]) # 2 channel CT and PET
        output_shape = tuple(sampleCube_dim+[1]) # 1 channel output
    else: # 'channels_first'
        input_shape = tuple([2] + sampleCube_dim) # 2 channel CT and PET
        output_shape = tuple([1] + sampleCube_dim) # 1 channel output

    # # distribution strategy (multi-GPU or TPU training), disabled because model.fit 
    # strategy = tf.distribute.experimental.MultiWorkerMirroredStrategy()
    # with strategy.scope():
    
    # load existing or create new model for this folder
    thisFoldIntermediateModelFileName = "{:>02d}InterDSSENetModel.h5".format(cvFoldIndex)
    thisFoldFinalModelFileName = "{:>02d}FinalDSSENetModel.h5".format(cvFoldIndex)
    thisFoldIntermediateModelPath = os.path.join(saveModelDirectory,thisFoldIntermediateModelFileName)
    thisFoldFinalModelPath = os.path.join(saveModelDirectory,thisFoldFinalModelFileName)
    print(thisFoldIntermediateModelPath)
    print(thisFoldFinalModelPath)

    if os.path.exists(thisFoldIntermediateModelPath):
        #This line is giving bug with custom loss objects
        # model = tf.keras.models.load_model(thisFoldIntermediateModelPath, custom_objects={'dice_loss_fg': dice_loss_fg, 'modified_dice_loss': modified_dice_loss, 'customMultiLabelJaccardDiceAndCategoricalCrossEntropyLossWrapper': customMultiLabelJaccardDiceAndCategoricalCrossEntropyLossWrapper})
        model = tf.keras.models.load_model(thisFoldIntermediateModelPath,compile=False)
        optimizer = tf.keras.optimizers.Adam(lr=0.0001, beta_1=0.9, beta_2=0.999, epsilon=1e-8, decay=0.0)        
        if trainInputParams['AMP']:
            optimizer = tf.compat.v1.train.experimental.enable_mixed_precision_graph_rewrite(optimizer)
        model.compile(optimizer = optimizer, loss = trainInputParams['loss_func'](data_format=trainInputParams['data_format']), metrics = [trainInputParams['acc_func']])        
        print('Loaded model: ' + thisFoldIntermediateModelPath)
    else:
        model = DSSEVNet(input_shape=input_shape, dropout_prob = 0.25, data_format=trainInputParams['data_format'])                              
        optimizer = tf.keras.optimizers.Adam(lr=0.0001, beta_1=0.9, beta_2=0.999, epsilon=1e-8, decay=0.0)        
        if trainInputParams['AMP']:
            optimizer = tf.compat.v1.train.experimental.enable_mixed_precision_graph_rewrite(optimizer)
        model.compile(optimizer = optimizer, loss = trainInputParams['loss_func'](data_format=trainInputParams['data_format']), metrics = [trainInputParams['acc_func']])
        model.summary(line_length=140)
        
    # TODO: clean up the evaluation callback
    #tb_logdir = './logs/' + os.path.basename(trainInputParams['fname'])
    #tb_logdir = './logs/' + os.path.splitext(os.path.basename(thisFoldIntermediateModelPath))[0] + '/' + datetime.now().strftime("%Y%m%d-%H%M%S")
    tb_logdir = os.path.join(logDir, os.path.splitext(os.path.basename(thisFoldIntermediateModelPath))[0] + '/' + datetime.now().strftime("%Y%m%d-%H%M%S"))
    train_callbacks = [tf.keras.callbacks.TensorBoard(log_dir = tb_logdir),
                            tf.keras.callbacks.ModelCheckpoint(thisFoldIntermediateModelPath, 
                                monitor = "loss", save_best_only = True, mode='min')]

    model.fit(x=train_sequence,
                        steps_per_epoch = num_train_cases,
                        max_queue_size = 40,
                        epochs = trainInputParams['num_training_epochs'],
                        validation_data = val_sequence,
                        validation_steps = num_val_cases,
                        callbacks = train_callbacks,
                        use_multiprocessing = False,
                        workers = num_cpus, 
                        shuffle = True)
    model.save(thisFoldFinalModelPath,save_format='h5')



def evaluateFold(trainConfigFilePath = '/home/user/DMML/CodeAndRepositories/MMGTVSeg/input/trainInput_DSSENet.json',
                 saveModelDirectory = '/home/user/DMML/CodeAndRepositories/MMGTVSeg/output/DSSEModels',
                 cvFoldIndex = 0,
                 numCVFolds = 5,
                 savePredictions = True,
                 out_dir = '/home/user/DMML/CodeAndRepositories/MMGTVSeg/output/evaluate_test/',
                 thisFoldFinalModelPath = "",
                 verbose=False):
    
    #Open configuration
    with open(trainConfigFilePath) as f:
        trainInputParams = json.load(f)
        f.close()
    trainInputParams = sanityCheckTrainParams(trainInputParams)

    #Get location of model file based on the CVFold index if it is not already provided
    if 0 == len(thisFoldFinalModelPath):
        thisFoldFinalModelFileName = "{:>02d}FinalDSSENetModel.h5".format(cvFoldIndex)
        thisFoldFinalModelPath = os.path.join(saveModelDirectory,thisFoldFinalModelFileName)
    print(thisFoldFinalModelPath)
    
    #Make sure model file exists
    if  os.path.exists(thisFoldFinalModelPath) and os.path.isfile(thisFoldFinalModelPath):
        pass
    else:
        sys.exit('No file exists at ', thisFoldFinalModelPath)

    #Make sure output directory exists 
    if os.path.exists(out_dir):
        #Check if it is a directory or not
        if os.path.isfile(out_dir): 
            sys.exit(out_dir, ' is a file and not directory. Exiting.') 
    else:
        #create 
        os.makedirs(out_dir)

    #Create test data generator
    batch_size =1
    val_sequence = DSSENet_Generator(trainConfigFilePath = trainConfigFilePath, 
                                    useDataAugmentationDuringTraining = False,
                                    batch_size = batch_size,
                                    numCVFolds = numCVFolds,
                                    cvFoldIndex = cvFoldIndex, #Can be between 0 to 4
                                    isValidationFlag = True,
                                    verbose=verbose
                                    )
    numBatches = val_sequence.__len__()

    #load model
    #This line is giving bug with custom loss objects
    # model = tf.keras.models.load_model(thisFoldFinalModelPath, custom_objects={'dice_loss_fg': dice_loss_fg, 'modified_dice_loss': modified_dice_loss, 'customMultiLabelJaccardDiceAndCategoricalCrossEntropyLossWrapper': customMultiLabelJaccardDiceAndCategoricalCrossEntropyLossWrapper})
    model = tf.keras.models.load_model(thisFoldFinalModelPath,compile=False)
    optimizer = tf.keras.optimizers.Adam(lr=0.0001, beta_1=0.9, beta_2=0.999, epsilon=1e-8, decay=0.0)        
    if trainInputParams['AMP']:
        optimizer = tf.compat.v1.train.experimental.enable_mixed_precision_graph_rewrite(optimizer)
    model.compile(optimizer = optimizer, loss = trainInputParams['loss_func'](data_format=trainInputParams['data_format']), metrics = [trainInputParams['acc_func']])        
    print('Loaded model: ' + thisFoldFinalModelPath)

    #Evaluation and writing loop
    test_msd  = np.zeros((numBatches*batch_size, len(trainInputParams['label_names'])))
    test_dice = np.zeros((numBatches*batch_size, len(trainInputParams['label_names'])))
    for idx in range(0,numBatches):
            #With channel_last data format batch_X: (batch_size, 144, 144, 144, 2), batch_y_gt: (batch_size, 144, 144, 144, 1),
            batch_X, batch_y_gt, ctFiles, ptFiles, gtvFiles = val_sequence.getitemExtended(idx)
            t = time.time()
            #With channel_last data format batch_X: (batch_size, 144, 144, 144, 2), batch_y_gt: (batch_size, 144, 144, 144, 1),
            batch_y_pred = model.predict(batch_X, batch_size=batch_size)
            print('\nInference time for ', batch_size, ' samples: ', time.time() - t)
            #Convert softmax output for  N-classes (here N= 2, 0 and 1) into the class with highest probability
            if 'channels_last' == trainInputParams['data_format']:
                batch_y_pred = np.argmax(batch_y_pred, axis=-1).astype('int16')
                batch_y_gt = np.squeeze(batch_y_gt,axis=-1)
            else:
                batch_y_pred = np.argmax(batch_y_pred, axis=1).astype('int16')
                batch_y_gt = np.squeeze(batch_y_gt,axis=1)
                
            for i in range(0, batch_size):
                saveIndex = idx * batch_size + i
                print('Batch ', idx, ' sampleInBatch ', i, ' ', ctFiles[i], ' ', ptFiles[i], ' ', gtvFiles[i])
                #get spacing
                orgSpacing = SimpleITK.ReadImage(os.path.join(trainInputParams["resampledFilesLocation"], ctFiles[i])).GetSpacing()
                tarnsposedSpacing = (orgSpacing[2], orgSpacing[1], orgSpacing[0])  
                [lbls, msd, rms, hd] = surface_distance_multi_label(batch_y_pred[i,:], batch_y_gt[i,:], sampling=tarnsposedSpacing)
                test_msd[saveIndex, lbls-1] = np.transpose(msd)
                print('Surface to surface MSD [mm]: ', np.transpose(msd))
                dice = dice_multi_label(batch_y_pred[i,:], batch_y_gt[i,:])
                test_dice[saveIndex, lbls-1] = np.transpose(dice)
                print('Dice: ', np.transpose(dice))
                # save results to NIFTI 
                if savePredictions:
                    #Load original GTV data <--- remember its size can be larger than the sample batch_y_gt[i,:] 
                    srcImage_nii = nib.load(os.path.join(trainInputParams["resampledFilesLocation"], gtvFiles[i]))
                    srcImage_nii_data = srcImage_nii.get_fdata()
                    srcImage_nii_aff  = srcImage_nii.affine   
                    #Transpose  and  create  buffer of same size as srcImage 
                    transposed_srcImage_nii_data = np.transpose(srcImage_nii_data, axes=(2,1,0))
                    transposed_gtv_pred_data = np.zeros(shape = transposed_srcImage_nii_data.shape, dtype = np.int16)
                    predY_shape= batch_y_pred[i,:].shape
                    #debug
                    if predY_shape != transposed_srcImage_nii_data.shape:
                        print('***********************************************')
                        print("predY_shape ", predY_shape, " transpose_GTV shape: ", transposed_srcImage_nii_data.shape)
                    transposed_gtv_pred_data[slice(0, predY_shape[0]), slice(0, predY_shape[1]), slice(0, predY_shape[2])] = batch_y_pred[i,:]
                    #Transpose back to have same alignment as srcImage
                    gtv_pred_data = np.transpose(transposed_gtv_pred_data, axes=(2,1,0)).astype(np.int8)#srcImage_nii_data.dtype
                    desImage_nii = nib.Nifti1Image(gtv_pred_data, affine=srcImage_nii_aff)
                    desFileName = "predFold{:>02d}_".format(cvFoldIndex) + gtvFiles[i]
                    nib.save(desImage_nii, os.path.join(out_dir,desFileName))
    
    return thisFoldFinalModelPath, test_msd, test_dice

def train(trainConfigFilePath = '/home/user/DMML/CodeAndRepositories/MMGTVSeg/input/trainInput_DSSENet.json',
          saveModelDirectory = '/home/user/DMML/CodeAndRepositories/MMGTVSeg/output/DSSEModels',
          logDir= '/home/user/DMML/CodeAndRepositories/MMGTVSeg',
          numCVFolds = 5,
          rangeCVFoldIdStart=0,
          rangeCVFoldIdEnd=1 ):
    #Run CV Folds
    for cvFoldIndex in rangerange(rangeCVFoldIdStart,rangeCVFoldIdEnd):
        trainFold(trainConfigFilePath=trainConfigFilePath,
                  saveModelDirectory = saveModelDirectory,
                  logDir=logDir,
                  cvFoldIndex=cvFoldIndex, 
                  numCVFolds = numCVFolds )    

def evaluate(trainConfigFilePath = '/home/user/DMML/CodeAndRepositories/MMGTVSeg/input/trainInput_DSSENet.json',
             saveModelDirectory = '/home/user/DMML/CodeAndRepositories/MMGTVSeg/output/DSSEModels',
             out_dir = '/home/user/DMML/CodeAndRepositories/MMGTVSeg/output/evaluate_test/',
             numCVFolds = 5,
             trainModelEnsembleJsonPath_out = '/home/user/DMML/CodeAndRepositories/MMGTVSeg/output/trainModelCVEval_DSSENet.json'):
        listOfModelPaths = []
        listOfAverageDice = []  
        listOfAverageMSD = []      
        for cvFoldIndex in range(0,numCVFolds):
            thisFoldFinalModelPath, thisFold_test_msd, thisFold_test_dice = evaluateFold(
                trainConfigFilePath = trainConfigFilePath,
                saveModelDirectory = saveModelDirectory,
                cvFoldIndex = cvFoldIndex,        
                numCVFolds = numCVFolds,                        
                savePredictions = True,
                out_dir = out_dir,
                thisFoldFinalModelPath = "",
                verbose=False)
            listOfModelPaths.append(thisFoldFinalModelPath)
            listOfAverageDice.append(np.mean(thisFold_test_dice))
            listOfAverageMSD.append(np.mean(thisFold_test_msd))
        listOfEnsembleWeights =  [r/sum(listOfAverageDice) for r in listOfAverageDice]
        for idx in range(0,numCVFolds):
            print(listOfModelPaths[idx], ' AvgDice: ',  listOfAverageDice[idx], ' AvgMSD: ',  
                         listOfAverageMSD[idx],' ensembleWt ', listOfEnsembleWeights[idx])
        evalDict = {}
        evalDict['listOfModelPaths'] = listOfModelPaths
        evalDict['listOfEnsembleWeights'] = listOfEnsembleWeights
        evalDict['listOfAverageDice'] = listOfAverageDice
        evalDict['listOfAverageMSD'] = listOfAverageMSD
        with open(trainModelEnsembleJsonPath_out, 'w') as fp:
                json.dump(evalDict, fp, ) #, indent='' #, indent=4
                fp.close()
        return listOfModelPaths, listOfAverageDice, listOfAverageMSD, listOfEnsembleWeights

def ensembleBasedPrediction(listOfTestPatientNames,
        resampledTestDataLocation = '/home/user/DMML/CodeAndRepositories/MMGTVSeg/data/hecktor_train/resampled',
        groundTruthPresent = False,
        trainConfigFilePath = '/home/user/DMML/CodeAndRepositories/MMGTVSeg/input/trainInput_DSSENet.json',
        trainModelEnsembleJsonPath_in = '/home/user/DMML/CodeAndRepositories/MMGTVSeg/output/trainModelCVEval_DSSENet.json',
        out_dir = '/home/user/DMML/CodeAndRepositories/MMGTVSeg/output/evaluate_test/'):

    #Open configuration
    with open(trainConfigFilePath) as f:
        trainInputParams = json.load(f)
        f.close()
    trainInputParams = sanityCheckTrainParams(trainInputParams)
    #Open ensembles
    with open(trainModelEnsembleJsonPath_in) as f:
        trainModelEnsembleParams = json.load(f)
        f.close()

    if groundTruthPresent:
        groundTruthTestComparison = []
        diceSum = 0.0
    #Load list of models
    listOfModels = []
    for modelPath in trainModelEnsembleParams['listOfModelPaths']:
        model = tf.keras.models.load_model(modelPath,compile=False)
        optimizer = tf.keras.optimizers.Adam(lr=0.0001, beta_1=0.9, beta_2=0.999, epsilon=1e-8, decay=0.0)        
        if trainInputParams['AMP']:
            optimizer = tf.compat.v1.train.experimental.enable_mixed_precision_graph_rewrite(optimizer)
        model.compile(optimizer = optimizer, loss = trainInputParams['loss_func'](data_format=trainInputParams['data_format']), metrics = [trainInputParams['acc_func']])        
        print('Loaded model: ' + modelPath)
        listOfModels.append(model)

    #The cubes are going to be arranged as depth-index, rowIndex, col-Index    
    cube_size = [trainInputParams["patientVol_Depth"], trainInputParams["patientVol_Height"], trainInputParams["patientVol_width"]]
    DepthRange = slice(0, cube_size[0])
    RowRange = slice(0, cube_size[1])
    ColRange = slice(0, cube_size[1])
    numLabels = 1 + len(trainInputParams["labels_to_train"]) 
    if 'channels_last' == trainInputParams['data_format']:
        X_size = cube_size+[2] # 2 channel CT and PET
        y_size = cube_size+[numLabels] #  Output 2 labels including 0
        if groundTruthPresent:
            y_gt_size = cube_size+[1] #  ground truth has one channel identifying  label of voxel

    else: # 'channels_first'
        X_size = [2] + cube_size # 2 channel CT and PET
        y_size = [numLabels] + cube_size # output 2 labels including 0
        if groundTruthPresent:
            y_gt_size = [1] + cube_size #  ground truth has one channel identifying  label of voxel


    #Read each input and then predict using each model from the list of models
    for patientName in listOfTestPatientNames:
        ctFilePath = os.path.join(resampledTestDataLocation, patientName + trainInputParams["suffixList"][0])
        ptFilePath = os.path.join(resampledTestDataLocation,patientName + trainInputParams["suffixList"][1])
        if groundTruthPresent:
            gtvFilePath = os.path.join(resampledTestDataLocation,patientName + trainInputParams["suffixList"][2])
        print('################################')
        print(ctFilePath)
        print(ptFilePath)
        if groundTruthPresent:
            print(gtvFilePath)


        ctData = np.transpose(nib.load(ctFilePath).get_fdata(), axes=(2,1,0))
        ptData = np.transpose(nib.load(ptFilePath).get_fdata(), axes=(2,1,0))
        if groundTruthPresent:
            gtvData = np.transpose(nib.load(gtvFilePath).get_fdata(), axes=(2,1,0))

        #Clamp and normalize CT data <- simple normalization, just divide by 1000
        np.clip(ctData, trainInputParams["ct_low"], trainInputParams["ct_high"], out= ctData)
        ctData = ctData / 1000.0 #<-- This will put values between -1 and 3.1
        ctData = ctData.astype(np.float32)
        #Clamp and normalize PET Data
        np.clip(ptData, trainInputParams["pt_low"], trainInputParams["pt_high"], out= ptData)
        ptData = ptData / 1.0 #<-- If 10.0 is used This will put values between 0 and 2.5
        ptData = ptData.astype(np.float32)
        if groundTruthPresent:
            #For gtv mask make it integer
            gtvData = gtvData.astype(np.int16) #int16 #float32
            # pick specific labels to train (if training labels other than 1s and 0s)
            if trainInputParams["labels_to_train"] != [1]:
                temp = np.zeros(shape=gtvData.shape, dtype=gtvData.dtype)
                new_label_value = 1
                for lbl in trainInputParams["labels_to_train"]:
                    ti = (gtvData == lbl)
                    temp[ti] = new_label_value
                    new_label_value += 1
                gtvData = temp

        
        X = np.zeros(shape = tuple(X_size), dtype = np.float32)    
        y_pred_softmaxEnsemble = np.zeros(shape = tuple(y_size), dtype = np.float32)
        if groundTruthPresent:
            y_gt = np.zeros(shape = tuple(y_gt_size), dtype = np.int16)

        #Concatenate CT and PET data  in X and put X  in batch_X; 
        # Put GTV (if present) in Y_GT 
        if 'channels_last' == trainInputParams['data_format']:
            #Some of the files have extra slices so we fix the range
            X[:,:,:,0] = ctData[DepthRange, RowRange, ColRange]
            X[:,:,:,1] = ptData[DepthRange, RowRange, ColRange]
            if groundTruthPresent:
                y_gt[:,:,:,0] = gtvData[DepthRange, RowRange, ColRange]
        else:
            X[0,:,:,:] = ctData[DepthRange, RowRange, ColRange]
            X[1,:,:,:] = ptData[DepthRange, RowRange, ColRange]
            if groundTruthPresent:
                y_gt[0,:,:,:] = gtvData[DepthRange, RowRange, ColRange]
        
        #batch_size of 1. Since batch size is 1 so we are not using batch_Y or batch_Y_GT explicitly anymore
        batch_X = np.zeros(shape = tuple([1] + X_size), dtype = np.float32)
        batch_X[0,:,:,:,:] = X

        #predict with each model, multiply by ensemble weight and then add
        #http://rasbt.github.io/mlxtend/user_guide/classifier/EnsembleVoteClassifier/
        t = time.time()
        for model, ensembleWeight in zip(listOfModels,trainModelEnsembleParams['listOfEnsembleWeights']):
            batch_y_pred_softmax = model.predict(batch_X, batch_size=1)
            y_pred_softmax = batch_y_pred_softmax[0,...]
            #y_pred_softmax_list.append(y_pred_softmax)
            y_pred_softmaxEnsemble += ensembleWeight * y_pred_softmax
        print('\nInference time for 1 sample with ', len(listOfModels), ' models: ', time.time() - t)
        #Now do the voting : convert weighted ensemble softmax output for  N-classes (here N= 2, 0 and 1) 
        # into the class with highest probability
        if 'channels_last' == trainInputParams['data_format']:
            y_pred = np.argmax(y_pred_softmaxEnsemble, axis=-1).astype('int16')            
        else:
            y_pred = np.argmax(y_pred_softmaxEnsemble, axis=0).astype('int16')

        #Compute msd dice etc if possible
        if groundTruthPresent:
            if 'channels_last' == trainInputParams['data_format']:
                y_gt = np.squeeze(y_gt,axis=-1)
            else:
                y_gt = np.squeeze(y_gt,axis=0)
            #get spacing
            orgSpacing = SimpleITK.ReadImage(ctFilePath).GetSpacing()
            tarnsposedSpacing = (orgSpacing[2], orgSpacing[1], orgSpacing[0])
            [lbls, msd, rms, hd] = surface_distance_multi_label(y_pred, y_gt, sampling=tarnsposedSpacing)
            dice = dice_multi_label(y_pred, y_gt)
            print(patientName, ' Surface to surface MSD [mm]: ', np.transpose(msd), ' dice: ', np.transpose(dice))
            groundTruthTestComparison.append({ 'patientName':patientName, 'SSMSD':np.transpose(msd),  'dice':np.transpose(dice) })
            diceSum += np.sum(np.transpose(dice))

        #Save output : Use the same affine as ct data as availability of true GTV is not guaranteed
        #Load original GTV data <--- remember its size can be larger than the sample batch_y_gt[i,:] 
        srcImage_nii = nib.load(ctFilePath)
        srcImage_nii_data = srcImage_nii.get_fdata()
        srcImage_nii_aff  = srcImage_nii.affine   
        #Transpose  and  create  buffer of same size as srcImage 
        transposed_srcImage_nii_data = np.transpose(srcImage_nii_data, axes=(2,1,0))
        transposed_gtv_pred_data = np.zeros(shape = transposed_srcImage_nii_data.shape, dtype = np.int16)
        predY_shape= y_pred.shape #batch_y_pred[i,:].shape
        #debug
        if predY_shape != transposed_srcImage_nii_data.shape:
            print('***********************************************')
            print("predY_shape ", predY_shape, " transpose_GTV shape: ", transposed_srcImage_nii_data.shape)
        transposed_gtv_pred_data[slice(0, predY_shape[0]), slice(0, predY_shape[1]), slice(0, predY_shape[2])] = y_pred #batch_y_pred[i,:]
        #Transpose back to have same alignment as srcImage
        gtv_pred_data = np.transpose(transposed_gtv_pred_data, axes=(2,1,0)).astype(np.int8)#srcImage_nii_data.dtype
        desImage_nii = nib.Nifti1Image(gtv_pred_data, affine=srcImage_nii_aff)
        desFileName = 'predEnsemble_' + patientName + trainInputParams["suffixList"][2]
        nib.save(desImage_nii, os.path.join(out_dir,desFileName))

    if groundTruthPresent:
        print('Average Dice: ', diceSum/len(groundTruthTestComparison))
        for result in groundTruthTestComparison :
            print(result)
    return
        
