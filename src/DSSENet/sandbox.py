import os
import json
import glob
import sys
import nibabel as nib
import numpy as np
import SimpleITK as sitk
import random

    

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



# Cross validation plan:
# List of patient =>  Split Files (CT, PT, GTV), NumSplits
# N-Fold Cross Validation : 
# Fold_i : Train_Patient Names & Split Files, CVPatient Name & Split Files
# Train generator : Batch of Split Files with random shuffle + On the fly Data Augmentation
# Validation generator: Batch size 1 of Split file, no data augmentation,
#                       Merging of prediction result

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

