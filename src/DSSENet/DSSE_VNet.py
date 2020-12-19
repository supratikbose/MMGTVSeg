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
import tensorflow_addons as tfa

######### Loss functions ##########
# mean Dice loss (mean of multiple labels with option to ignore zero (background) label)
def dice_coef(y_true, y_pred, smooth = 0.00001, squared_denominator = False, ignore_zero_label = True):
    num_dim = len(K.int_shape(y_pred)) 
    num_labels = K.int_shape(y_pred)[-1]
    reduce_axis = list(range(1, num_dim - 1))
    y_true = y_true[..., 0]
    dice = 0.0

    if (ignore_zero_label == True):
        label_range = range(1, num_labels)
    else:
        label_range = range(0, num_labels)

    for i in label_range:
        y_pred_b = y_pred[..., i]
        y_true_b = K.cast(K.equal(y_true, i), K.dtype(y_pred))
        intersection = K.sum(y_true_b * y_pred_b, axis = reduce_axis)        
        if squared_denominator: 
            y_pred_b = K.square(y_pred_b)
        y_true_o = K.sum(y_true_b, axis = reduce_axis)
        y_pred_o =  K.sum(y_pred_b, axis = reduce_axis)     
        d = (2. * intersection + smooth) / (y_true_o + y_pred_o + smooth) 
        dice = dice + K.mean(d)
    dice = dice / len(label_range)
    return dice

def dice_loss(y_true, y_pred):
    f = 1 - dice_coef(y_true, y_pred, smooth = 0.00001, squared_denominator = False, ignore_zero_label = False)
    return f

def dice_loss_fg(y_true, y_pred):
    f = 1 - dice_coef(y_true, y_pred, smooth = 0.00001, squared_denominator = False, ignore_zero_label = True)
    return f

def modified_dice_loss(y_true, y_pred):
    f = 1 - dice_coef(y_true, y_pred, smooth = 0.00001, squared_denominator = True, ignore_zero_label = False)
    return f

def modified_dice_loss_fg(y_true, y_pred):
    f = 1 - dice_coef(y_true, y_pred, smooth = 0.00001, squared_denominator = True, ignore_zero_label = True)
    return f

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
                data_format='channels_last',
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
        self.data_format = data_format
        self.useDataAugmentationDuringTraining = useDataAugmentationDuringTraining
        self.batch_size = batch_size
        self.numCVFolds = numCVFolds
        self.cvFoldIndex = cvFoldIndex
        self.isValidationFlag = isValidationFlag
        self.verbose = verbose
        
        #Read individual members of trainConfig <-- Is it needed?
        
        self.AllPatientList = [(os.path.basename(f)).replace('_ct.nii.gz','') \
                for f in glob.glob(self.trainConfig["resampledFilesLocation"] + '/*_ct.nii.gz', recursive=False) ]
        #DO NOT Ranomize patient list as we want to call the same generator for different CV index
        # and do not want to shuffle patient list between different calls
        #######random.shuffle(self.AllPatientList)   
        
        #Based on numCVFolds, CV index and whether this is a training set generator of validation generator
        #determine list of patient names to be used in this data generator
        self.numAllPatients = len(self.AllPatientList)
        self.numCVPatients = self.numAllPatients // self.numCVFolds
        self.numTrainPatients = self.numAllPatients - self.numCVPatients
        
        #Assert cvFoldIndex = 0, #Can be between 0 to numCVFolds -1
        assert(self.cvFoldIndex >= 0 and self.cvFoldIndex < self.numCVFolds -1)
        startIdx_cv = self.cvFoldIndex * self.numCVPatients
        endIdx_cv = (self.cvFoldIndex+1) * self.numCVPatients
        self.list_cvIdx = [*range(startIdx_cv, endIdx_cv)]
        self.list_trainIdx =  [*range(0, startIdx_cv)] + [*range(endIdx_cv, self.numAllPatients)]
        self.cVPatients = [self.AllPatientList[i] for i in self.list_cvIdx]
        self.trainPatients = [self.AllPatientList[i] for i in self.list_trainIdx]
        
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
        self.ColRange = slice(0, self.cube_size[1])
        if 'channels_last' == self.data_format:
            self.X_size = self.cube_size+[2] # 2 channel CT and PET
            self.y_size = self.cube_size+[1] # 1 channel output
        else: # 'channels_first'
            self.X_size = [2] + self.cube_size # 2 channel CT and PET
            self.y_size = [1] + self.cube_size # 1 channel output

        if self.verbose:
            print('trainConfigFilePath: ', trainConfigFilePath)
            print('data_format: ', self.data_format)
            print('useDataAugmentationDuringTraining: ', self.useDataAugmentationDuringTraining)
            print('batch_size: ', self.batch_size)
            print('numCVFolds: ', self.numCVFolds)
            print('cvFoldIndex: ', self.cvFoldIndex)
            print('isValidationFlag: ', self.isValidationFlag)            
            print('labels_to_train: ', self.trainConfig["labels_to_train"])
            print('label_names: ', self.trainConfig["label_names"])
            print('lr_flip: ', self.trainConfig["lr_flip"])
            print('label_symmetry_map: ', self.trainConfig["label_names"])
            print('translate_random: ', self.trainConfig["translate_random"])
            print('rotate_random: ', self.trainConfig["rotate_random"])
            print('scale_random: ', self.trainConfig["scale_random"])
            print('change_intensity: ', self.trainConfig["change_intensity"])            
            print('ct_low: ', self.trainConfig["ct_low"])
            print('ct_high: ', self.trainConfig["ct_high"])
            print('pt_low: ', self.trainConfig["pt_low"])
            print('pt_high: ', self.trainConfig["pt_high"])
            print('resampledFilesLocation: ', self.trainConfig["resampledFilesLocation"])
            print('suffixList: ', self.trainConfig["suffixList"])
            print('patientVol_width: ', self.trainConfig["patientVol_width"])
            print('patientVol_Height: ', self.trainConfig["patientVol_Height"])
            print('change_intensity: ', self.trainConfig["patientVol_Depth"])            
            print('AllPatientList: ', self.AllPatientList)        
            print('numAllPatients: ', self.numAllPatients)
            print('numCVPatients: ', self.numCVPatients)
            print('numTrainPatients: ', self.numTrainPatients)
            print('list_cvIdx: ', self.list_cvIdx)
            print('list_trainIdx: ', self.list_trainIdx)
            print('cVPatients: ', self.cVPatients)
            print('trainPatients: ', self.trainPatients)
            if isValidationFlag:
                print('Using VALIDATION SET: ', self.patientNames)
                print('num_validation_cases: ', self.num_cases)
            else:
                print('sing TRAINING SET: ', self.patientNames)
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

    def __getitem__(self, idx):
        # keras sequence returns a batch of datasets, not a single case like generator
        #Note that _getitem__() gets called __len__() number of times, passing idx in range 0 <= idx < __len__()
        batch_X = np.zeros(shape = tuple([self.batch_size] + self.X_size), dtype = np.float32)
        batch_y = np.zeros(shape = tuple([self.batch_size] + self.y_size), dtype = np.int16)
        returnNow = False
        for i in range(0, self.batch_size):  
            X = np.zeros(shape = tuple(self.X_size), dtype = np.float32)
            y = np.zeros(shape = tuple(self.y_size), dtype = np.int16)         
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
            ctData = np.transpose(nib.load(os.path.join(self.trainConfig["resampledFilesLocation"], ctFileName)).get_fdata(), axes=(2,1,0)) #axes: depth, height, width            
            ptData = np.transpose(nib.load(os.path.join(self.trainConfig["resampledFilesLocation"], ptFileName)).get_fdata(), axes=(2,1,0)) 
            gtvData = np.transpose(nib.load(os.path.join(self.trainConfig["resampledFilesLocation"], gtvFileName)).get_fdata(), axes=(2,1,0))   
            
            #Debug code
            if self.verbose:
               minCT = ctData.min()
               maxCT = ctData.max()                 
               minPT = ptData.min()
               maxPT = ptData.max()
               minGTV = gtvData.min()
               maxGTV = gtvData.max()         
               print('BatchId ', idx, ' sampleInBatchId ', i, ' ', ctFileName, ' ', ptFileName, ' ', gtvFileName, )
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
            ptData = ptData / 1.0 #<-- This will put values between 0 and 2.5
            ptData = ptData.astype(np.float32)
            #For gtv mask make it integer
            gtvData = gtvData.astype(np.int16)

            #Apply Data augmentation
            if self.useDataAugmentationDuringTraining:
                # translate, scale, and rotate volume
                if self.trainConfig["translate_random"] > 0 or self.trainConfig["rotate_random"] > 0 or self.trainConfig["scale_random"] > 0:
                    ctData, ptData, gtvData = self.random_transform(ctData, ptData, gtvData, self.trainConfig["rotate_random"], self.trainConfig["scale_random"], self.trainConfig["translate_random"], fast_mode=True)
                # No flipping or intensity modification

            #Concatenate CT and PET data  in X and put X  in batch_X; Put GTV in Y and Y in batch_Y
            if 'channels_last' == self.data_format:
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

import matplotlib.pyplot as plt
def displayBatchData(batchX, batchY, sampleInBatchId = 0, startSliceId = 26, endSliceId = 31, data_format='channels_last',pauseTime_sec = 0.5):
    numSamplesInBatch = batchX.shape[0]
    depth = batchX.shape[1] if 'channels_last' == data_format else batchX.shape[2]
    numSlicesToDisplay = endSliceId - startSliceId + 1
    plt.figure(1)#sampleInBatchId+1
    for sliceId in range(startSliceId, endSliceId):
        offset = sliceId - startSliceId
        #Display CT        
        plt.subplot(3, numSlicesToDisplay, offset+1)
        plt.axis('off')
        plt.title('CT_'+str(sliceId), fontsize=8) 
        if 'channels_last' == data_format:
            plt.imshow(batchX[sampleInBatchId, sliceId,:, :, 0])
        else: # 'channel_first'
            plt.imshow(batchX[sampleInBatchId, 0, sliceId,:, :])
        #Display PET        
        plt.subplot(3, numSlicesToDisplay, numSlicesToDisplay + offset+1)
        plt.axis('off')
        plt.title('PT_'+str(sliceId), fontsize=8)
        if 'channels_last' == data_format:
            plt.imshow(batchX[sampleInBatchId, sliceId,:, :, 1])
        else: # 'channel_first'
            plt.imshow(batchX[sampleInBatchId, 1, sliceId,:, :])
        #Display GTV        
        plt.subplot(3, numSlicesToDisplay, 2*numSlicesToDisplay + offset+1)
        plt.axis('off')
        plt.title('GTV_'+str(sliceId), fontsize=8)
        if 'channels_last' == data_format:
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
	#>>>>>>> (16, 144, 144, 2)
    input_channels =  input_shape[-1] if 'channels_last' == data_format else input_shape[-4] #config["inputChannels"]
    tile_tensor    =  [1,1,1,1,16]    if 'channels_last' == data_format else [1,16,1,1,1]    #config["inputChannels"]
    if 1 == input_channels:
        sixteen_channelInput = tf.tile(img_input,tile_tensor)
    else:
		#>>>>>> <tf.Tensor 'elu/Identity:0' shape=(None, 16, 144, 144, 16) dtype=float32>
        sixteen_channelInput = ConvBnElu(img_input, filters=16, kernel_size = (5,5,5), strides = (1,1,1),  data_format=data_format)
    #In Table 1 of Ref 1, stride of 1x1x5 was mentioned for conv1, but we are sicking to stride 1x1x1; And here conv1 step includes add + elu
	#>>>>>>> <tf.Tensor 'elu_1/Identity:0' shape=(None, 16, 144, 144, 16) dtype=float32>
    _InTr =  ELU()(Add()([ConvBnElu(sixteen_channelInput, filters=16, kernel_size = (5,5,5), strides = (1,1,1),  data_format=data_format), sixteen_channelInput]))
	#>>>>>> <tf.Tensor 'spatial_dropout3d/Identity:0' shape=(None, 16, 144, 144, 16) dtype=float32>
    _InTrDropout = SpatialDropout3D(rate=dropout_prob, data_format='channels_last')(_InTr)

    #DownTr32  D x H x W x 16 ==> D/2 x H/2 x W/2 x 32  
	#>>>>>> <tf.Tensor 'elu_3/Identity:0' shape=(None, 8, 72, 72, 32) dtype=float32>
    _DownTr32  =  DownConvBnElu(x=_InTr, in_filters=16,  data_format=data_format)
	#>>>>>> <tf.Tensor 'elu_6/Identity:0' shape=(None, 8, 72, 72, 32) dtype=float32>
    _DownTr32  =  UpperLayerSingleResidualBlock(x=_DownTr32, data_format=data_format)
	#>>>>>> <tf.Tensor 'elu_9/Identity:0' shape=(None, 8, 72, 72, 32) dtype=float32>
    _DownTr32  =  UpperLayerSingleResidualBlock(x=_DownTr32, data_format=data_format)
	#>>>>>> <tf.Tensor 'multiply/Identity:0' shape=(None, 8, 72, 72, 32) dtype=float32>
    _DownTr32  =  Squeeze_Excite_block(x=_DownTr32, ratio=8, data_format=data_format)
	#>>>>>> <tf.Tensor 'spatial_dropout3d_1/Identity:0' shape=(None, 8, 72, 72, 32) dtype=float32>
    _DownTr32Dropout = SpatialDropout3D(rate=dropout_prob, data_format='channels_last')(_DownTr32)

    #DownTr64   D/2 x H/2 x W/2 x 32 ==> D/4 x H/4 x W/4  x 64  
	#>>>>>> <tf.Tensor 'elu_10/Identity:0' shape=(None, 4, 36, 36, 64) dtype=float32>
    _DownTr64  =  DownConvBnElu(x=_DownTr32, in_filters=32,  data_format=data_format)
    #>>>>>> <tf.Tensor 'elu_20/Identity:0' shape=(None, 4, 36, 36, 64) dtype=float32>	
    _DownTr64  =  SingleAndDoubleResidualBlock(x=_DownTr64,  data_format=data_format)
	#>>>>>> <tf.Tensor 'multiply_1/Identity:0' shape=(None, 4, 36, 36, 64) dtype=float32>
    _DownTr64  =  Squeeze_Excite_block(x=_DownTr64, ratio=8, data_format=data_format)
	#>>>>>> <tf.Tensor 'spatial_dropout3d_2/Identity:0' shape=(None, 4, 36, 36, 64) dtype=float32>
    _DownTr64Dropout = SpatialDropout3D(rate=dropout_prob, data_format='channels_last')(_DownTr64)

     #DownTr128   D/4 x H/4 x W/4  x 64 ==> D/8 x H/8 x W/8 x 128
	#>>>>>> <tf.Tensor 'elu_21/Identity:0' shape=(None, 2, 18, 18, 128) dtype=float32>
    _DownTr128  =  DownConvBnElu(x=_DownTr64, in_filters=64,  data_format=data_format)  
	#>>>>>> <tf.Tensor 'elu_31/Identity:0' shape=(None, 2, 18, 18, 128) dtype=float32>
    _DownTr128  =  SingleAndDoubleResidualBlock(x=_DownTr128,  data_format=data_format)
	#>>>>>> <tf.Tensor 'multiply_2/Identity:0' shape=(None, 2, 18, 18, 128) dtype=float32>
    _DownTr128  =  Squeeze_Excite_block(x=_DownTr128, ratio=8, data_format=data_format)
	#>>>>>> <tf.Tensor 'spatial_dropout3d_3/Identity:0' shape=(None, 2, 18, 18, 128) dtype=float32>
    _DownTr128Dropout = SpatialDropout3D(rate=dropout_prob, data_format='channels_last')(_DownTr128)

    #DownTr256   D/8 x H/8 x W/8 x 128 ==> D/16 x H/16 x W/16 x 256
	#>>>>>> <tf.Tensor 'elu_32/Identity:0' shape=(None, 1, 9, 9, 256) dtype=float32>
    _DownTr256  =  DownConvBnElu(x=_DownTr128, in_filters=128,  data_format=data_format)  
	#>>>>>> <tf.Tensor 'elu_42/Identity:0' shape=(None, 1, 9, 9, 256) dtype=float32>
    _DownTr256  =  SingleAndDoubleResidualBlock(x=_DownTr256, data_format=data_format)
	#>>>>>> <tf.Tensor 'multiply_3/Identity:0' shape=(None, 1, 9, 9, 256) dtype=float32>
    _DownTr256  =  Squeeze_Excite_block(x=_DownTr256, ratio=8, data_format=data_format)       


    ########## Dncode path ##########
    #UpTr256    D/16 x H/16 x W/16 x 256 ==> D/8 x H/8 x W/8 x 128 => D/8 x H/8 x W/8 x 256 (due to concatenation)
	#>>>>>> <tf.Tensor 'elu_43/Identity:0' shape=(None, 2, 18, 18, 128) dtype=float32>
    _UpTr256  = UpConvBnElu(_DownTr256, in_filters=256, data_format=data_format)
	#>>>>>> <tf.Tensor 'concatenate/Identity:0' shape=(None, 2, 18, 18, 256) dtype=float32>
    _UpTr256  = Concatenate(axis = getBNAxis(data_format))([_UpTr256,_DownTr128Dropout])
	#>>>>>> <tf.Tensor 'elu_53/Identity:0' shape=(None, 2, 18, 18, 256) dtype=float32>
    _UpTr256  =  SingleAndDoubleResidualBlock(x=_UpTr256, data_format=data_format)
	#>>>>>> <tf.Tensor 'multiply_4/Identity:0' shape=(None, 2, 18, 18, 256) dtype=float32>
    _UpTr256  =  Squeeze_Excite_block(x=_UpTr256, ratio=8, data_format=data_format)
    #Also Dsv4 D/8 x H/8 x W/8 x 256 => D x H x W x 4
	#>>>>>> <tf.Tensor 'up_sampling3d/Identity:0' shape=(None, 16, 144, 144, 4) dtype=float32>
    _Dsv4 = DeepSupervision(_UpTr256, filters=4, upSamplingRatio=8, data_format=data_format)


    #UpTr128    D/8 x H/8 x W/8 x 256 ==> D/4 x H/4 x W/4 x 64 => D/4 x H/4 x W/4 x 128 (due to concatenation)
	#>>>>>> <tf.Tensor 'elu_54/Identity:0' shape=(None, 4, 36, 36, 64) dtype=float32>
    _UpTr128  = UpConvBnElu(_UpTr256, in_filters=128, data_format=data_format)
	#>>>>>> <tf.Tensor 'concatenate_1/Identity:0' shape=(None, 4, 36, 36, 128) dtype=float32>
    _UpTr128  = Concatenate(axis = getBNAxis(data_format))([_UpTr128,_DownTr64Dropout])
	#>>>>>> <tf.Tensor 'elu_64/Identity:0' shape=(None, 4, 36, 36, 128) dtype=float32>
    _UpTr128  =  SingleAndDoubleResidualBlock(x=_UpTr128, data_format=data_format)
	#>>>>>> <tf.Tensor 'multiply_5/Identity:0' shape=(None, 4, 36, 36, 128) dtype=float32>
    _UpTr128  =  Squeeze_Excite_block(x=_UpTr128, ratio=8, data_format=data_format)
    #Also Dsv3 D/4 x H/4 x W/4 x 128 => D x H x W x 4
	#>>>>>> <tf.Tensor 'up_sampling3d_1/Identity:0' shape=(None, 16, 144, 144, 4) dtype=float32>
    _Dsv3 = DeepSupervision(_UpTr128, filters=4, upSamplingRatio=4, data_format=data_format)

    #UpTr64    D/4 x H/4 x W/4 x 128 ==> D/2 x H/2 x W/2 x 32 => D/2 x H/2 x W/2 x 64 (due to concatenation)
	#>>>>>> <tf.Tensor 'elu_65/Identity:0' shape=(None, 8, 72, 72, 32) dtype=float32>
    _UpTr64  = UpConvBnElu(_UpTr128, in_filters=64, data_format=data_format)
	#>>>>>> <tf.Tensor 'concatenate_2/Identity:0' shape=(None, 8, 72, 72, 64) dtype=float32>
    _UpTr64  = Concatenate(axis = getBNAxis(data_format))([_UpTr64,_DownTr32Dropout])
	#>>>>>> <tf.Tensor 'elu_75/Identity:0' shape=(None, 8, 72, 72, 64) dtype=float32>
    _UpTr64  =  SingleAndDoubleResidualBlock(x=_UpTr64, data_format=data_format)
	#>>>>>> <tf.Tensor 'multiply_6/Identity:0' shape=(None, 8, 72, 72, 64) dtype=float32>
    _UpTr64  =  Squeeze_Excite_block(x=_UpTr64, ratio=8, data_format=data_format)
    #Also Dsv2 D/2 x H/2 x W/2 x 64 => D x H x W x 4
	#>>>>>> <tf.Tensor 'up_sampling3d_2/Identity:0' shape=(None, 16, 144, 144, 4) dtype=float32>
    _Dsv2 = DeepSupervision(_UpTr64, filters=4, upSamplingRatio=2, data_format=data_format)

    #UpTr32    D/2 x H/2 x W/2 x 64 ==> D x H x W x 16 => D x H x W x 32 (due to concatenation)
	#>>>>>> <tf.Tensor 'elu_76/Identity:0' shape=(None, 16, 144, 144, 16) dtype=float32>
    _UpTr32  = UpConvBnElu(_UpTr64, in_filters=32, data_format=data_format)
	#>>>>>> <tf.Tensor 'concatenate_3/Identity:0' shape=(None, 16, 144, 144, 32) dtype=float32>
    _UpTr32  = Concatenate(axis = getBNAxis(data_format))([_UpTr32,_InTrDropout])
	#>>>>>> <tf.Tensor 'elu_79/Identity:0' shape=(None, 16, 144, 144, 32) dtype=float32>
    _UpTr32  =   UpperLayerSingleResidualBlock(x=_UpTr32, data_format=data_format)
	#>>>>>> <tf.Tensor 'elu_82/Identity:0' shape=(None, 16, 144, 144, 32) dtype=float32>
    _UpTr32  =   UpperLayerSingleResidualBlock(x=_UpTr32, data_format=data_format)
	#>>>>>> <tf.Tensor 'multiply_7/Identity:0' shape=(None, 16, 144, 144, 32) dtype=float32>
    _UpTr32  =  Squeeze_Excite_block(x=_UpTr32, ratio=8, data_format=data_format)
    #Also Dsv1 D x H x W x 32 => D x H x W x 4
	#>>>>>> <tf.Tensor 'up_sampling3d_3/Identity:0' shape=(None, 16, 144, 144, 4) dtype=float32>
    _Dsv1 = DeepSupervision(_UpTr32, filters=4, upSamplingRatio=1, data_format=data_format)

    #Final concatenation and convolution
    #128 x 128 x 128 x 4 ==> 128 x 128 x 128 x 16
	#>>>>>> <tf.Tensor 'concatenate_4/Identity:0' shape=(None, 16, 144, 144, 16) dtype=float32>
    _DsvConcat = Concatenate(axis = getBNAxis(data_format))([_Dsv1, _Dsv2, _Dsv3, _Dsv4])
    #128 x 128 x 128 x 1 ==> 128 x 128 x 128 x 1
	#>>>>>> <tf.Tensor 'conv3d_66/Identity:0' shape=(None, 16, 144, 144, 1) dtype=float32>
    _Final = Conv3D(filters = 1, kernel_size = (1,1,1), strides = (1,1,1), kernel_initializer = 'he_normal', padding = 'same', activation='sigmoid', data_format=data_format, use_bias = False)(_DsvConcat)

    # model instantiation
    model = Model(img_input, _Final)
    return model


############### Model train and test function #################
def train(trainConfigFilePath, data_format='channels_last', cpuOnlyFlag = False):
    # load trainInputParams  from JSON config files
    with open(trainConfigFilePath) as f:
        trainInputParams = json.load(f)
        f.close()

    trainInputParams['loss_func'] = dice_loss
    trainInputParams['acc_func'] = metrics.categorical_accuracy
    trainInputParams['group_normalization'] = False
    trainInputParams['activation_type'] = 'relu'
    trainInputParams['final_activation_type'] = 'softmax'
    trainInputParams['AMP'] = False
    trainInputParams['XLA'] = False

    if 'labels_to_train' not in trainInputParams:
        trainInputParams['labels_to_train'] = [1]
    if 'asymmetric' not in trainInputParams:
        trainInputParams['asymmetric'] = True
    
    #Original
    # determine number of available
    gpus = tf.config.list_physical_devices('GPU') 
    num_gpus = len(gpus)    
    print('Number of GPUs available  for training: ', num_gpus)
    #If using CPUs only for training (set it true if getting GPU memory allocation error)
    if True == cpuOnlyFlag:
        #Hide GPUs
        if num_gpus > 0: #gpus:
            print("Restricting TensorFlow to use CPU only by hiding GPUs.")
            try:
                tf.config.experimental.set_visible_devices([], 'GPU')
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

    # #Limit GPU use to a single GPU as I am not sure whether that messed up tensorboard
    # # I earlier saw an error message about multiGPU and tensorboard
    # gpus = tf.config.experimental.list_physical_devices('GPU')
    # num_gpus = len(gpus)    
    # print('Number of GPUs AVAILABLE for training: ', num_gpus)
    # if gpus:
    #     print("Restricting TensorFlow to only use the first GPU.")
    #     try:
    #         tf.config.experimental.set_visible_devices(gpus[0], 'GPU')
    #         logical_gpus = tf.config.experimental.list_logical_devices('GPU')
    #         print(len(gpus), "Physical GPUs,", len(logical_gpus), "Logical GPU")
    #     except RuntimeError as e:
    #         # Visible devices must be set before GPUs have been initialized
    #         print(e)

    num_cpus = min(os.cpu_count(), 24)   # don't use more than 16 CPU threads
    print('Number of CPUs used for training: ', num_cpus)

    if (trainInputParams['AMP']):
        os.environ['TF_ENABLE_AUTO_MIXED_PRECISION'] = '1'
        os.environ['CUDNN_TENSOROP_MATH'] = '1'
        print('Using Automatic Mixed Precision (AMP) Arithmentic...')

    if (trainInputParams['XLA']):
        tf.config.optimizer.set_jit(True)
    

    train_sequence = DSSENet_Generator(trainConfigFilePath = trainConfigFilePath, 
                                        data_format=data_format,
                                        useDataAugmentationDuringTraining = True,
                                        batch_size = 1,
                                        numCVFolds = 5,
                                        cvFoldIndex = 0, #Can be between 0 to 4
                                        isValidationFlag = False,
                                        verbose=False
                                                )

    test_sequence = DSSENet_Generator(trainConfigFilePath = trainConfigFilePath, 
                                        data_format=data_format,
                                        useDataAugmentationDuringTraining = False,
                                        batch_size = 1,
                                        numCVFolds = 5,
                                        cvFoldIndex = 0, #Can be between 0 to 4
                                        isValidationFlag = True,
                                        verbose=False
                                        )
    
    # count number of training and test cases
    num_train_cases = train_sequence.__len__()
    num_test_cases = test_sequence.__len__()

    print('Number of train cases: ', num_train_cases)
    print('Number of test cases: ', num_test_cases)
    print("labels to train: ", trainInputParams['labels_to_train'])
    
    sampleCube_dim = [trainInputParams["patientVol_Depth"], trainInputParams["patientVol_Height"], trainInputParams["patientVol_width"]]
    if 'channels_last' == data_format:
        input_shape = tuple(sampleCube_dim+[2]) # 2 channel CT and PET
        output_shape = tuple(sampleCube_dim+[1]) # 1 channel output
    else: # 'channels_first'
        input_shape = tuple([2] + sampleCube_dim) # 2 channel CT and PET
        output_shape = tuple([1] + sampleCube_dim) # 1 channel output

    # # distribution strategy (multi-GPU or TPU training), disabled because model.fit 
    # strategy = tf.distribute.experimental.MultiWorkerMirroredStrategy()
    # with strategy.scope():
    
    # load existing or create new model
    if os.path.exists(trainInputParams["lastSavedModel"]):
        #from tensorflow.keras.models import load_model        
        #model = load_model(trainInputParams['fname'], custom_objects={'dice_loss_fg': loss.dice_loss_fg, 'modified_dice_loss': loss.modified_dice_loss})
        model = tf.keras.models.load_model(trainInputParams['fname'], custom_objects={'dice_loss_fg': dice_loss_fg, 'modified_dice_loss': modified_dice_loss})
        print('Loaded model: ' + trainInputParams["lastSavedModel"])
    else:
        model = DSSEVNet(input_shape=input_shape, dropout_prob = 0.25, data_format=data_format)                              

        optimizer = tf.keras.optimizers.Adam(lr=0.0001, beta_1=0.9, beta_2=0.999, epsilon=1e-8, decay=0.0)        
        if trainInputParams['AMP']:
            optimizer = tf.compat.v1.train.experimental.enable_mixed_precision_graph_rewrite(optimizer)

        model.compile(optimizer = optimizer, loss = trainInputParams['loss_func'], metrics = [trainInputParams['acc_func']])
        model.summary(line_length=140)
        
    # TODO: clean up the evaluation callback
    #tb_logdir = './logs/' + os.path.basename(trainInputParams['fname'])
    tb_logdir = './logs/' + os.path.basename(trainInputParams["lastSavedModel"]) + '/' + datetime.now().strftime("%Y%m%d-%H%M%S")
    train_callbacks = [tf.keras.callbacks.TensorBoard(log_dir = tb_logdir),
                tf.keras.callbacks.ModelCheckpoint(trainInputParams["lastSavedModel"], 
                monitor = "loss", save_best_only = True, mode='min')]


    model.fit_generator(train_sequence,
                        steps_per_epoch = num_train_cases,
                        max_queue_size = 40,
                        epochs = trainInputParams['num_training_epochs'],
                        validation_data = test_sequence,
                        validation_steps = num_test_cases,
                        callbacks = train_callbacks,
                        use_multiprocessing = False,
                        workers = num_cpus, 
                        shuffle = True)

    model.save(trainInputParams['lastSavedModel'] + '_final')