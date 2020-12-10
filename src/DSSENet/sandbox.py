import os
import glob
import sys
import nibabel as nib
import numpy as np
import SimpleITK as sitk

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



def splitPatientImageIntoSubFiles(srcFilePath, numDepthSplits,desFolderPath, 
                                    verbose = False):
    """
    Split original nii.gz file into  a number of files containing subset of 
    contiguous slices. Split files files will be named as 00K_<srcFileName>
    srcFilePath: Complete path to the source file
    numDepthSplits: Number of desired splits; Number of original slices should be
                    divisible by  numDepthSplits
    desFolderPath : destination folder to store split files 
    verbose: flag to show extra debug message
    """
    success = False
    srcImgFileName = os.path.basename(srcFilePath)
    print(srcImgFileName)

    srcImage_itk = sitk.ReadImage(srcFilePath)
    origin_spcaing = srcImage_itk.GetSpacing()
    if verbose:
        print('origin_spcaing: ', origin_spcaing)
    origin_size = srcImage_itk.GetSize()
    if verbose:
        print('origin_size: ', origin_size)
    origin_origin = srcImage_itk.GetOrigin()
    if verbose:
        print('origin_origin: ', origin_origin)   
    numSliceOrigin = origin_size[2]
    if verbose:
        print('numSliceOrigin: ', numSliceOrigin)
    
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
    if 0 == (numSliceOrigin % numDepthSplits):
        if verbose:
            print(numSliceOrigin, ' is divisible by: ', numDepthSplits)
        numSlicePerSplit = numSliceOrigin // numDepthSplits
        if verbose:
            print('numSlicePerSplit: ', numSlicePerSplit)                    
        for k in range(0,numDepthSplits):
            # extraction ranges
            x_idx_range = slice(0, origin_size[0])
            y_idx_range = slice(0, origin_size[1])
            z_idx_range = slice(k*numSlicePerSplit, (k+1)*numSlicePerSplit)
            if verbose:
                print('Extracting and writing in separate file...')
                print('Slice range: ', k*numSlicePerSplit, ' ', (k+1)*numSlicePerSplit)   
            roi = srcImage_nii_data[x_idx_range,y_idx_range,z_idx_range]
            if verbose:
                print('Dim :', roi.ndim)
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
    else:
        print(numSliceOrigin, ' is NOT divisible by: ', numDepthSplits, '. No split file generated ')
        success = False    
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