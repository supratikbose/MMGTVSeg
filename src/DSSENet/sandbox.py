import os
import json
import glob
import sys
import nibabel as nib
import numpy as np
import SimpleITK as sitk
import random

def splitPatientImageIntoSubFiles(
                srcFolder,
                srcImgFileName,
                patientVol_width,
                patientVol_Height,
                patientVol_Depth,
                sampleInput_Depth,
                desFolderPath,
                verbose):
# def splitPatientImageIntoSubFiles(srcFilePath, numDepthSplits,desFolderPath, 
#                                     verbose = False):
    """
    Split original nii.gz file into  a number of files containing subset of 
    contiguous slices. Split files  will be named as 00K_<baseFileName>
    srcFolder: Folder containing source files
    srcImgFileName: File that will be split
    patientVol_width : width 
    patientVol_Height : height
    patientVol_Depth: number of slices
    sampleInput_Depth: depth of input sample
    desFolderPath : destination folder to store split files 
    verbose: flag to show extra debug message
    """
    success = False
    #srcImgFileName = os.path.basename(srcFilePath)
    srcFilePath = os.path.join(srcFolder, srcImgFileName)
    srcImage_itk = sitk.ReadImage(srcFilePath)
    origin_spcaing = srcImage_itk.GetSpacing()
    origin_size = srcImage_itk.GetSize()
    origin_origin = srcImage_itk.GetOrigin()
    expected_size = (patientVol_Height, patientVol_width, patientVol_Depth)
    if verbose:
        print(srcFilePath)
        print('origin_spcaing: ', origin_spcaing)
        print('origin_size: ', origin_size)
        print('origin_origin: ', origin_origin)    
    if origin_size != expected_size:
        print(srcImgFileName, ' : ', origin_size, ' different than expected ', expected_size,  ' Exiting')
        success = False
        return success
    if  0 != (patientVol_Depth %   sampleInput_Depth):
        print(srcImgFileName, ' : depth ', patientVol_Depth, ' not divisible by sample depth ', sampleInput_Depth,  ' Exiting')
        success = False
        return success
    numDepthSplits =   patientVol_Depth //  sampleInput_Depth
    if verbose:
        print('patient depth: ', patientVol_Depth, ' sampleInput_Depth: ', sampleInput_Depth, ' numDepthSplits: ', numDepthSplits)              
    
    srcImage_nii = nib.load(srcFilePath)
    srcImage_nii_data = srcImage_nii.get_fdata()
    srcImage_nii_aff  = srcImage_nii.affine
    srcImage_nii_hdr  = srcImage_nii.header
    if verbose:
        print('srcImage_nii_aff :')
        print(srcImage_nii_aff)    
        print('srcImage_nii_hdr :')
        print(srcImage_nii_hdr)
        print('srcImage_nii_data.shape :', srcImage_nii_data.shape)

    for k in range(0,numDepthSplits):
        # extraction ranges
        x_idx_range = slice(0, origin_size[0])
        y_idx_range = slice(0, origin_size[1])
        z_idx_range = slice(k*sampleInput_Depth, (k+1)*sampleInput_Depth)
        if verbose:
            print('Extracting and writing in separate file...')
            print('Slice range: ', k*sampleInput_Depth, ' ', (k+1)*sampleInput_Depth)   
        roi = srcImage_nii_data[x_idx_range,y_idx_range,z_idx_range]
        newFileName = '{:>03d}_{}'.format(k,srcImgFileName)
        if True: # verbose: # Print always
            print(newFileName)
        # output = sitk.GetImageFromArray(roi, isVector=False)
        # output.SetOrigin([0,0,0])
        # output.SetSpacing([1,1,1])
        # sitk.WriteImage(output, os.path.join(desFolderPath,newFileName)) 
        #https://gist.github.com/ofgulban/285ef3df135a9fadd7cf7dca984b7409           
        output = nib.Nifti1Image(roi, affine=srcImage_nii_aff)
        nib.save(output, os.path.join(desFolderPath,newFileName))
    if verbose:
        print('Finished')
    success = True     
    return success

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


# #Test code for split and merge
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

# Cross validation plan:
# List of patient =>  Split Files (CT, PT, GTV), NumSplits
# N-Fold Cross Validation : 
# Fold_i : Train_Patient Names & Split Files, CVPatient Name & Split Files
# Train generator : Batch of Split Files with random shuffle + On the fly Data Augmentation
# Validation generator: Batch size 1 of Split file, no data augmentation,
#                       Merging of prediction result

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

def preproc(preproc_config_file, splitFilesAlreadyCreated = False, verbose=False):
    #Read preprocessing patients
    with open(preproc_config_file) as f:
        preproc_config = json.load(f)
        f.close()
    #Get unique patient names from resampled directory
    resampledFilesLocation = preproc_config["resampledFilesLocation"]
    patientVol_width = preproc_config["patientVol_width"]
    patientVol_Height = preproc_config["patientVol_Height"]
    patientVol_Depth = preproc_config["patientVol_Depth"]
    sampleInput_Depth = preproc_config["sampleInput_Depth"]
    splitFilesLocation = preproc_config["splitFilesLocation"]
    #Randomize patient list : First we got file list, then dropped
    # the '_ct.nii.gz' to get patient name and then shuffled it
    patientList = [(os.path.basename(f)).replace('_ct.nii.gz','') \
      for f in glob.glob(resampledFilesLocation + '/*_ct.nii.gz', recursive=False) ]      
    #For each patient, split files to generate   <00k>_<patient><_ct/_pt/_ct_gtvt>.nii.gz
    # where 0 < k < numDepthSplits-1,   numDepthSplits = patientVol_Depth / sampleInput_Depth     
    if False == splitFilesAlreadyCreated:
        #First check is splitFilesLocation is empty or not.
        if os.path.exists(splitFilesLocation):
            #Check if it is a directory or not
            if os.path.isfile(splitFilesLocation): 
                print('Error: splitFilesLocation is a file.')
                return
            #We are here - so it is a directory - See if it is empty
            if 0 != len(os.listdir(splitFilesLocation)):
                print('Error: splitFilesLocation is a non-empty directory.')
                return    
        else:
            #create 
            os.mkdir(splitFilesLocation)    
        #We are here so splitFilesLocation is an empty directory.        
        #Create split files
        suffixList = ['_ct.nii.gz', '_pt.nii.gz', '_ct_gtvt.nii.gz']
        for patientName in patientList:
            for suffix in suffixList:
                baseFileName = patientName + suffix
                success = splitPatientImageIntoSubFiles(
                    resampledFilesLocation,
                    baseFileName,
                    patientVol_width,
                    patientVol_Height,
                    patientVol_Depth,
                    sampleInput_Depth,
                    splitFilesLocation,
                    verbose)
                if not success:
                    print('Failed in splitting ', baseFileName, ' Exiting.')
                    return False
    
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
    
    return True


preproc('input/preprocInput_DSSENet.json', splitFilesAlreadyCreated = False, verbose=False)