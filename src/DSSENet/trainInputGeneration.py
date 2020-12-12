#training and cross validation input generation
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

    return values:
    successFlag : if splitting was successful
    oneSliceExtraFlag: if original image contained one slice extra
    """
    successFlag = False
    oneSliceExtraFlag = False 
    if  0 != (patientVol_Depth %   sampleInput_Depth):
        print('Expected  depth ', patientVol_Depth, ' not divisible by sample depth ', sampleInput_Depth,  ' Exiting')
        successFlag = False
        return successFlag, oneSliceExtraFlag
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
        print(srcImgFileName, ' : ', origin_size, ' different than expected ', expected_size)
        if (origin_size[0] == expected_size[0]) \
        and (origin_size[1] == expected_size[1]) \
        and (origin_size[2] == expected_size[2]+1) :
            oneSliceExtraFlag = True
            print('But we can still continue')
        else:
            print('We will have to exit')
            successFlag = False
            return successFlag, oneSliceExtraFlag
    
    numDepthSplits =   patientVol_Depth //  sampleInput_Depth
    if oneSliceExtraFlag: #verbose
        print(srcImgFileName, ' Actual depth ', origin_size[2], ' Expected: ', patientVol_Depth, 
               ' sampleInput_Depth: ', sampleInput_Depth, ' numDepthSplits: ', numDepthSplits)              
    
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
        if verbose: # verbose: # True Print always
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
    successFlag = True     
    return successFlag, oneSliceExtraFlag

def createSplitFiles(preprocConfigfilePath, outputJsonFilePath, verbose=False):
    #Read preprocessing patients
    successFlag = False
    patientList = []
    listPatientsWithExtraSlice = []
    with open(preprocConfigfilePath) as f:
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

    #First check is splitFilesLocation is empty or not.
    if os.path.exists(splitFilesLocation):
        #Check if it is a directory or not
        if os.path.isfile(splitFilesLocation): 
            print('Error: splitFilesLocation is a file.')
            successFlag = False
            return successFlag, patientList, listPatientsWithExtraSlice
        #We are here - so it is a directory - See if it is empty
        if 0 != len(os.listdir(splitFilesLocation)):
            print('Error: splitFilesLocation is a non-empty directory.')
            successFlag = False
            return successFlag, patientList, listPatientsWithExtraSlice    
    else:
        #create 
        os.mkdir(splitFilesLocation)    
    #We are here so splitFilesLocation is an empty directory.        
    #Create split files
    suffixList = ['_ct.nii.gz', '_pt.nii.gz', '_ct_gtvt.nii.gz']
    for patientName in patientList:
        if True: # verbose: # True Print always
            print('Splitting ', patientName)        
        for suffix in suffixList:
            baseFileName = patientName + suffix
            successFlag = False
            oneSliceExtraFlag = False

            successFlag, oneSliceExtraFlag = splitPatientImageIntoSubFiles(
                    resampledFilesLocation,
                    baseFileName,
                    patientVol_width,
                    patientVol_Height,
                    patientVol_Depth,
                    sampleInput_Depth,
                    splitFilesLocation,
                    verbose)
            if ('_ct.nii.gz' == suffix) and successFlag and oneSliceExtraFlag:
                listPatientsWithExtraSlice.append(patientName)
            if not successFlag:
                print('Failed in splitting ', baseFileName, ' Exiting.')
                return successFlag, patientList, listPatientsWithExtraSlice
    
    patientInfoDict = dict()
    patientInfoDict['patientList'] = patientList
    patientInfoDict['suffixList']  = suffixList 
    patientInfoDict['listPatientsWithExtraSlice'] = listPatientsWithExtraSlice
    numDepthSplits =   patientVol_Depth //  sampleInput_Depth
    patientInfoDict['suffixList'] = ['{:>03d}'.format(k) for k in range(numDepthSplits)]
    with open(outputJsonFilePath, 'w') as fp:
        json.dump(patientInfoDict, fp, indent=4)
    print('createSplitFiles Finshed.')
    return successFlag, patientList, listPatientsWithExtraSlice
            
    
successFlag, patientList, listPatientsWithExtraSlice = \
    createSplitFiles('input/preprocInput_DSSENet.json', 'input/patientInfoDict_DSSENet.json', verbose=False)   