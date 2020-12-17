import os
import json
import glob
import sys
import nibabel as nib
import numpy as np
import SimpleITK as sitk
from scipy import ndimage
import random
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.utils import Sequence

# sys.path.append(os.path.join(os.path.dirname(__file__), "..", ".."))
#import src
# from src.DSSENet import volume #from pipeline import volume #from vmsseg import volume




######################################################################################################

def mergeSubFilesIntoPatientImage(srcFolderPath, baseFileName, numDepthSplits, desFolderPath,
                                    desFilePrefix, verbose = False):
    """
    Merge  00K_<srcFileName> into original nii.gz file 
    srcFolderPath: Complete path to where 00K_<srcFileName> files are present
                    0 <= K <= numDepthSplits-1
    numDepthSplits: Number of  splits; 
    desFolderPath : destination folder to store merged
    verbose: flag to show extra debug message
    """
    success = False
    #check existence of split file names
    for k in range(0,numDepthSplits):
        splitFileName = '{:>03d}_{}'.format(k,baseFileName)
        if os.path.exists(os.path.join(srcFolderPath,splitFileName)):
            if verbose:
                print(splitFileName, ' exists in ', srcFolderPath)
            continue
        else:
            print(splitFileName, ' does not exist in ', srcFolderPath, ' - Exiting.')
            success = False
            return success

    #We are here - so all split file exists
    for k in range(0,numDepthSplits):
        splitFileName = '{:>03d}_{}'.format(k,baseFileName)
        splitFilePathName = os.path.join(srcFolderPath,splitFileName)
        srcImage_nii = nib.load(splitFilePathName)
        srcImage_nii_data = srcImage_nii.get_fdata()
        srcImage_nii_aff  = srcImage_nii.affine
        if 0 == k:
            desImg_nii_aff = srcImage_nii_aff
            desImg_nii_data = srcImage_nii_data
            if verbose:
                print('Shape :', desImg_nii_data.shape)     
        else:
            desImg_nii_data = np.concatenate((desImg_nii_data,srcImage_nii_data),axis=2)
            if verbose:
                print('Shape :', desImg_nii_data.shape)
    output = nib.Nifti1Image(desImg_nii_data, affine=desImg_nii_aff)
    desFileName = desFilePrefix + baseFileName
    nib.save(output, os.path.join(desFolderPath,desFileName))
    if True: #verbose:
        print('Finished Merging')
    success = True  
    return success  


# Initial Test code for split and merge - changed quite a lot
# laptopFlag = True
# if laptopFlag:
#     splitPatientImageIntoSubFiles('U:/UDrive/Shared/InformationRetrieval/DeepLearning/CodeAndRepositories/MMSegWithDSSEVNetAndACLoss/data/Temp/CHGJ029_Rsmpl/CHGJ029_ct_gtvt.nii.gz',\
#         12,'U:/UDrive/Shared/InformationRetrieval/DeepLearning/CodeAndRepositories/MMSegWithDSSEVNetAndACLoss/data/Temp/CHGJ029_Split', False)
#     mergeSubFilesIntoPatientImage('U:/UDrive/Shared/InformationRetrieval/DeepLearning/CodeAndRepositories/MMSegWithDSSEVNetAndACLoss/data/Temp/CHGJ029_Split', 
#         'CHGJ029_ct_gtvt.nii.gz', 12, 'U:/UDrive/Shared/InformationRetrieval/DeepLearning/CodeAndRepositories/MMSegWithDSSEVNetAndACLoss/data/Temp/CHGJ029_Merge',
#         'merged_', False)
# else:
#     splitPatientImageIntoSubFiles('/home/user/Desktop/Temp/CHGJ029_Rsmpl/CHGJ029_ct.nii.gz',\
#         12,'/home/user/Desktop/Temp/CHGJ029_Split', False)


#Get list of patients
# pattern1 = 'data/resampled/'
# pattern2 = '_ct.nii.gz'
# s='data/resampled/CHGJ069_ct.nii.gz'
# s1 = (s.replace(pattern1,'')).replace(pattern2,'')
# print(s1)
# pattern1 = 'data/resampled/'
# pattern2 = '_ct.nii.gz'
# dataFolderRelativePath = 'data/resampled/'
# patientList = [(f.replace(pattern1,'')).replace(pattern2,'') \
#   for f in glob.glob(dataFolderRelativePath + '*_ct.nii.gz', recursive=False) ]
# print(patientList)

#Writing JSON files
# import os
# import json
# patientInfoDict = dict()
# patientInfoDict['patientList'] = ['AAA', 'BBB', 'CCC', 'DDD', 'EEE']
# patientInfoDict['suffixList']  = ['_c.gz', '_d.gz'] 
# patientInfoDict['listPatientsWithExtraSlice'] = ['AAA', 'CCC']
# patientVol_Depth = 144
# sampleInput_Depth = 6
# numDepthSplits =   patientVol_Depth //  sampleInput_Depth
# outputJsonFilePath = 'input/tmp.json'
# patientInfoDict['prefixList'] = ['{:>03d}'.format(k) for k in range(numDepthSplits)]
# with open(outputJsonFilePath, 'w') as fp:
#     json.dump(patientInfoDict,fp, indent = 4)


# patientList = [
#         "CHUM013",
#         "CHGJ089",
#         "CHUM055",
#         "CHGJ048",
#         "CHGJ057",
#         "CHUS090",
#         "CHMR005",
#         "CHGJ072",
#         "CHUM062",
#         "CHGJ088",
#         "CHUS008",
#         "CHUM002",
#         "CHGJ038"]
# numCVFold = 5
# numPatients = len(patientList)
# numCVPatients = numPatients // numCVFold
# numTrainPatients = numPatients - numCVPatients
# print('numPatients ', numPatients, ' numTrainPatients ', numTrainPatients, 'numCVPatients ', numCVPatients)
# for cvIndex in range(numCVFold):
#     #newFileName = '{:>03d}_{}'.format(k,srcImgFileName)
#     trainKey = 'train_{:>03d}_Patients'.format(cvIndex)
#     cVKey = 'cv_{:>03d}_Patients'.format(cvIndex)
#     startIdx_cv = cvIndex * numCVPatients
#     endIdx_cv = (cvIndex+1) * numCVPatients
#     #Note  argument-unpacking operator i.e. *.
#     list_cvIdx = [*range(startIdx_cv, endIdx_cv)]
#     list_trainIdx =  [*range(0, startIdx_cv)] + [*range(endIdx_cv, numPatients)]    
#     cVPatients = [patientList[i] for i in list_cvIdx]
#     trainPatients = [patientList[i] for i in list_trainIdx]
#     print('cvIndex ', cvIndex, ' list_cvIdx ', list_cvIdx, ' list_trainIdx ', list_trainIdx)
#     print(cVKey, ' ', cVPatients, ' ', trainKey, ' ',trainPatients)


# #randomize patient list
# random.shuffle(patientList)    
# numPatients = len(patientList)
# if verbose:
#     print(patientList)
#     print('numPatients: ', numPatients)
#     unique_patientList = list(set(patientList))
#     print('numUniquePatients: ', len(unique_patientList))


# numCVFolds = preproc_config["numCVFolds"]
# numCVPatients = numPatients // numCVFolds
# numTrainPatients = numPatients - numCVPatients
# if verbose:
#     print('numCVFolds: ', numCVFolds, ' numCVPatients: ', numCVPatients, ' numTrainPatients: ', numTrainPatients)

#####################
#minC, maxC, minP, maxP, minG, maxG  -17466.458984375   32306.255859375   -0.17402704060077667   38.8752326965332   0.0   1.0    
#####################
# #Check why there is CRC check error <--- This was successful
# fileList = glob.glob('data/hecktor_train/split' + '/*.nii.gz', recursive=False) 
# numFilesInList = len(fileList)
# for id in range(0, numFilesInList):
#     filePath = fileList[id]
#     fileData = np.transpose(nib.load(filePath).get_fdata(), axes=(2,1,0))
#     print(filePath, ' ', fileData.min(), ' ', fileData.max())


#This will be used for train and validation but not for testing where ground truth is not present
# class DSSENet_Generator(Sequence): 
#     def __init__(self,
#                 trainConfigFilePath,
#                 data_format='channels_last',
#                 useDataAugmentationDuringTraining = True, #True for training, not true for CV  specially if we want to merge prediction
#                 batch_size = 1,
#                 cvFoldIndex = 0, #Can be between 0 to 4
#                 isValidationFlag = False # True for validation
#                 ):
#         # Read config file
#         with open(trainConfigFilePath) as fp:
#                 self.trainConfig = json.load(fp)
#                 fp.close() 
#         self.data_format = data_format
#         self.useDataAugmentationDuringTraining = useDataAugmentationDuringTraining
#         self.batch_size = batch_size
#         self.cvFoldIndex = cvFoldIndex
#         self.isValidationFlag = isValidationFlag

#         self.patientVol_width = self.trainConfig["patientVol_width"]
#         self.patientVol_Height = self.trainConfig["patientVol_Height"]
#         self.patientVol_Depth = self.trainConfig["patientVol_Depth"]
#         self.sampleInput_Depth = self.trainConfig["sampleInput_Depth"]
#         self.splitFilesLocation = self.trainConfig["splitFilesLocation"]
#         self.numDepthSplits = self.trainConfig["numDepthSplits"]
#         self.prefixList = self.trainConfig["prefixList"]
#         assert(self.numDepthSplits == len(self.prefixList))
#         self.suffixList = self.trainConfig['suffixList'] 
#         if self.isValidationFlag:
#             foldKey = 'cv_{:>03d}_Patients'.format(cvFoldIndex)
#         else:
#             foldKey = 'train_{:>03d}_Patients'.format(cvFoldIndex)
#         self.patientNames =  self.trainConfig[foldKey] 
#         self.num_cases = self.numDepthSplits * len(self.patientNames)

#         self.labels_to_train = self.trainConfig["labels_to_train"]
#         self.label_names = self.trainConfig["label_names"]
#         self.lr_flip = self.trainConfig["lr_flip"]
#         self.label_symmetry_map = self.trainConfig["label_symmetry_map"]
#         self.translate_random = self.trainConfig["translate_random"]
#         self.rotate_random = self.trainConfig["rotate_random"]
#         self.scale_random = self.trainConfig["scale_random"]
#         self.change_intensity = self.trainConfig["change_intensity"]
#         self.ct_low = self.trainConfig["ct_low"]        
#         self.ct_high = self.trainConfig["ct_high"]
#         self.pt_low = self.trainConfig["pt_low"]
#         self.pt_high = self.trainConfig["pt_high"]

#         self.cube_size = [self.sampleInput_Depth, self.patientVol_Height, self.patientVol_width]
#         if 'channels_last' == self.data_format:
#             self.X_size = self.cube_size+[2] # 2 channel CT and PET
#             self.y_size = self.cube_size+[1] # 1 channel output
#         else: # 'channel+first'
#             self.X_size = [2] + self.cube_size # 2 channel CT and PET
#             self.y_size = [1] + self.cube_size # 1 channel output
    
#     def __len__(self):
#         #Note here, in this implementation data augmentation is not actually increasing 
#         #number of original cases; instead it is applying random transformation on one
#         #of the original case before using it in a training batch. That is why the 
#         # # the __len()__ function is not dependent on data augmentation 
#         return self.num_cases // self.batch_size

#     def __getitem__(self, idx):
#         # keras sequence returns a batch of datasets, not a single case like generator
#         #Note that _getitem__() gets called __len__() number of times, passing idx in range 0 <= idx < __len__()
#         batch_X = np.zeros(shape = tuple([self.batch_size] + self.X_size), dtype = np.float32)
#         batch_y = np.zeros(shape = tuple([self.batch_size] + self.y_size), dtype = np.int16)
#         returnNow = False
#         for i in range(0, self.batch_size):  
#             X = np.zeros(shape = tuple(self.X_size), dtype = np.float32)
#             y = np.zeros(shape = tuple(self.y_size), dtype = np.int16)         
#             # load case from disk
#             overallIndex = idx * self.batch_size + i
#             fileIndex = overallIndex // self.numDepthSplits
#             splitIndex = overallIndex % self.numDepthSplits

#             ctFileName = self.prefixList[splitIndex] + '_' + self.patientNames[fileIndex] + self.suffixList[0]
#             ptFileName = self.prefixList[splitIndex] + '_' + self.patientNames[fileIndex] + self.suffixList[1]
#             gtvFileName = self.prefixList[splitIndex] + '_' + self.patientNames[fileIndex] + self.suffixList[2]
                        
#             #check file existence            
#             if os.path.exists(os.path.join(self.splitFilesLocation, ctFileName)):
#                 pass
#             else:
#                 print(os.path.join(self.splitFilesLocation, ctFileName), ' does not exist')  
#                 returnNow = True  
#             if os.path.exists(os.path.join(self.splitFilesLocation, ptFileName)):
#                 pass
#             else:
#                 print(os.path.join(self.splitFilesLocation, ptFileName), ' does not exist')  
#                 returnNow = True
#             if os.path.exists(os.path.join(self.splitFilesLocation, gtvFileName)):
#                 pass
#             else:
#                 print(os.path.join(self.splitFilesLocation, gtvFileName), ' does not exist')  
#                 returnNow = True

#             if returnNow:
#                 sys.exit() # return batch_X, batch_y, False, 0, 0, 0, 0, 0, 0

#             #We are here => returnNow = False
#             ctData = np.transpose(nib.load(os.path.join(self.splitFilesLocation, ctFileName)).get_fdata(), axes=(2,1,0)) #axes: depth, height, width            
#             ptData = np.transpose(nib.load(os.path.join(self.splitFilesLocation, ptFileName)).get_fdata(), axes=(2,1,0)) 
#             gtvData = np.transpose(nib.load(os.path.join(self.splitFilesLocation, gtvFileName)).get_fdata(), axes=(2,1,0))            
            
#             # Debug code
#             # minCT = ctData.min()
#             # maxCT = ctData.max()                 
#             # minPT = ptData.min()
#             # maxPT = ptData.max()
#             # if gtvData is not None:
#             #     minGTV = gtvData.min()
#             #     maxGTV = gtvData.max()
#             # else: 
#             #     minGTV = 0
#             #     maxGTV = 1
#             #print('BatchId ', idx, ' sampleInBatchId ', i, ' ', ctFileName, ' ', ptFileName, ' ', gtvFileName, )
#             # print('ctData shape-type-min-max: ', ctData.shape, ' ', ctData.dtype, ' ', minCT, ' ', maxCT)
#             # print('ptData shape-type-min-max: ', ptData.shape, ' ', ptData.dtype, ' ', minPT, ' ', maxPT)
#             # if gtvData is not None:
#             #     print('gtvtData shape-type-min-max: ', gtvData.shape, ' ', gtvData.dtype, ' ', minGTV, ' ', maxGTV)

#             #Clamp and normalize CT data <- simple normalization, just divide by 1000
#             np.clip(ctData, self.ct_low, self.ct_high, out= ctData)
#             ctData = ctData / 1000.0 #<-- This will put values between -1 and 3.1
#             ctData = ctData.astype(np.float32)        
#             #Clamp and normalize PET Data
#             np.clip(ptData, self.pt_low, self.pt_high, out= ptData)
#             ptData = ptData / 1.0 #<-- This will put values between 0 and 2.5
#             ptData = ptData.astype(np.float32)
#             #For gtv mask make it integer
#             gtvData = gtvData.astype(np.int16)

#             #Apply Data augmentation
#             if self.useDataAugmentationDuringTraining:
#                 # translate, scale, and rotate volume
#                 if self.translate_random > 0 or self.rotate_random > 0 or self.scale_random > 0:
#                     ctData, ptData, gtvData = self.random_transform(ctData, ptData, gtvData, self.rotate_random, self.scale_random, self.translate_random, fast_mode=True)
#                 # No flipping or intensity modification

#             #Concatenate CT and PET data  in X and put X  in batch_X; Put GTV in Y and Y in batch_Y
#             if 'channels_last' == self.data_format:
#                 X[:,:,:,0] = ctData
#                 X[:,:,:,1] = ptData
#                 y[:,:,:,0] = gtvData
#             else:
#                 X[0,:,:,:] = ctData
#                 X[1,:,:,:] = ptData
#                 y[0,:,:,:] = gtvData
            
#             batch_X[i,:,:,:,:] = X
#             batch_y[i,:,:,:,:] = y

#         #return batch_X, batch_y, True, minCT, maxCT, minPT, maxPT, minGTV, maxGTV        
#         return batch_X, batch_y


#     def generate_rotation_matrix(self,rotation_angles_deg):    
#         R = np.zeros((3,3))
#         theta_x, theta_y, theta_z  = (np.pi / 180.0) * rotation_angles_deg.astype('float64') # convert from degrees to radians
#         c_x, c_y, c_z = np.cos(theta_x), np.cos(theta_y), np.cos(theta_z)
#         s_x, s_y, s_z = np.sin(theta_x), np.sin(theta_y), np.sin(theta_z)   
#         R[0, :] = [c_z*c_y, c_z*s_y*s_x - s_z*c_x, c_z*s_y*c_x + s_z*s_x]
#         R[1, :] = [s_z*c_y, s_z*s_y*s_x + c_z*c_x, s_z*s_y*c_x - c_z*s_x]    
#         R[2, :] = [-s_y, c_y*s_x, c_y*c_x]    
#         return R

#     def random_transform(self, img1, img2, label, rot_angle = 15.0, scale = 0.05, translation = 0.0, fast_mode=False):
#         angles = np.random.uniform(-rot_angle, rot_angle, size = 3) 
#         R = self.generate_rotation_matrix(angles)   
#         S = np.diag(1 + np.random.uniform(-scale, scale, size = 3)) 
#         A = np.dot(R, S)
#         t = np.array(img1.shape) / 2.
#         t = t - np.dot(A,t) + np.random.uniform(-translation, translation, size=3)
#         # interpolate the image channel
#         if fast_mode:
#             # nearest neighbor (use when CPU is the bottleneck during training)
#             img1 = ndimage.affine_transform(img1, matrix = A, offset = t, prefilter = False, mode = 'nearest', order = 0)
#             img2 = ndimage.affine_transform(img2, matrix = A, offset = t, prefilter = False, mode = 'nearest', order = 0)
#         else:
#             # linear interpolation
#             img1 = ndimage.affine_transform(img1, matrix = A, offset = t, prefilter = False, mode = 'nearest', order = 1)  
#             img2 = ndimage.affine_transform(img2, matrix = A, offset = t, prefilter = False, mode = 'nearest', order = 1)        
#         # interpolate the label channel
#         label = ndimage.affine_transform(label, matrix = A, offset = t, prefilter = False, mode = 'nearest', order = 0) 
#         return (img1, img2, label)   

# import matplotlib.pyplot as plt    
# def displayBatchData(batchX, batchY, data_format='channels_last',pauseTime_sec = 0.5):
#     numSamplesInBatch = batchX.shape[0]
#     depth = batchX.shape[1] if 'channels_last' == data_format else batchX.shape[2]
#     for sampleId in range(0,numSamplesInBatch):
#         plt.figure(sampleId+1)
#         for sliceId in range(0,depth):
#             #Display CT        
#             plt.subplot(3, depth, sliceId+1)
#             plt.axis('off')
#             plt.title('CT_'+str(sliceId), fontsize=8) 
#             if 'channels_last' == data_format:
#                 plt.imshow(batchX[sampleId, sliceId,:, :, 0])
#             else: # 'channel_first'
#                 plt.imshow(batchX[sampleId, 0, sliceId,:, :])
#             #Display PET        
#             plt.subplot(3, depth, depth + sliceId+1)
#             plt.axis('off')
#             plt.title('PT_'+str(sliceId), fontsize=8)
#             if 'channels_last' == data_format:
#                 plt.imshow(batchX[sampleId, sliceId,:, :, 1])
#             else: # 'channel_first'
#                 plt.imshow(batchX[sampleId, 1, sliceId,:, :])
#             #Display GTV        
#             plt.subplot(3, depth, 2*depth + sliceId+1)
#             plt.axis('off')
#             plt.title('GTV_'+str(sliceId), fontsize=8)
#             if 'channels_last' == data_format:
#                 plt.imshow(batchY[sampleId, sliceId,:, :, 0])
#             else: # 'channel_first'
#                 plt.imshow(batchY[sampleId, 0, sliceId,:, :])
#         plt.show()
#         plt.pause(pauseTime_sec)


# #Test code
# trainGenerator = DSSENet_Generator(
#     trainConfigFilePath = '/home/user/DMML/CodeAndRepositories/MMGTVSeg/input/trainInput_DSSENet.json',
#     data_format='channels_last',
#     useDataAugmentationDuringTraining = False,
#     batch_size = 2,
#     cvFoldIndex = 1, #Can be between 0 to 4
#     isValidationFlag = False
#     )
# numBatches = trainGenerator.__len__()

#Debug
#minC, maxC, minP, maxP, minG, maxG = 50000, -50000, 50000, -50000, 50000, -50000
# for idx in range(0,numBatches):
#     batchX, batchY = trainGenerator.__getitem__(idx)
    #Debug
    # bX, bY, success, minCT, maxCT, minPT, maxPT, minGTV, maxGTV = trainGenerator.__getitem__(idx)
    # if not success:
    #     print('ERRRRRRRRRRRRRRRRRROR')
    #     break
    # #if minCT < -3000 or maxCT > 5000:
    # if minPT < -0.1 or maxPT > 20:
    #     print('Idx ', idx, ' minPT ', minPT, ' maxPT ', maxPT,  ' ------------ABNORMAL---------')
    #     #break        
    # if minCT < minC:
    #     minC = minCT
    # if maxCT > maxC:
    #     maxC = maxCT
    # if minPT < minP:
    #     minP = minPT
    # if maxPT > maxP:
    #     maxP = maxPT
    # if minGTV < minG:
    #     minG = minGTV
    # if maxGTV > maxG:
    #     maxG = maxGTV  
#print('minC, maxC, minP, maxP, minG, maxG ', minC, ' ', maxC, ' ', minP, ' ',  maxP, ' ', minG, ' ', maxG)

# batchX, batchY = trainGenerator.__getitem__(5)
# displayBatchData(batchX, batchY, data_format='channels_last',pauseTime_sec = 0.5)

sys.path.append('/home/user/DMML/CodeAndRepositories/MMGTVSeg')
import src
from src.DSSENet import model
model.train('/home/user/DMML/CodeAndRepositories/MMGTVSeg/input/trainInput_DSSENet.json')
