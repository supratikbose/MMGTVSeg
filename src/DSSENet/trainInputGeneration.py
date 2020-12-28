############# NOT BEING USED ANYMORE ###########
################################################
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

def createSplitFiles(trainConfigFilePath,  verbose=False):
    #Read preprocessing patients
    successFlag = False
    patientList = []
    listPatientsWithExtraSlice = []
    with open(trainConfigFilePath) as fp:
        preproc_config = json.load(fp)
        fp.close()
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
    #Ranomize patient list
    random.shuffle(patientList)      
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
    #patientInfoDict is a superset of preproc_config to which new keys are added
    patientInfoDict = preproc_config #dict()
    patientInfoDict['patientList'] = patientList
    patientInfoDict['suffixList']  = suffixList 
    patientInfoDict['listPatientsWithExtraSlice'] = listPatientsWithExtraSlice
    numDepthSplits =   patientVol_Depth //  sampleInput_Depth
    patientInfoDict['numDepthSplits'] = numDepthSplits
    patientInfoDict['prefixList'] = ['{:>03d}'.format(k) for k in range(numDepthSplits)]
    with open(trainConfigFilePath, 'w') as fp:
        json.dump(patientInfoDict, fp) # indent=4
        fp.close()
    print('createSplitFiles Finshed.')
    return successFlag, patientList, listPatientsWithExtraSlice

# Cross validation plan:
# List of patient =>  Split Files (CT, PT, GTV), NumSplits
# N-Fold Cross Validation : 
# Fold_i : Train_Patient Names & Split Files, CVPatient Name & Split Files
# Train generator : Batch of Split Files with random shuffle + On the fly Data Augmentation
# Validation generator: Batch size 1 of Split file, no data augmentation,
#                       Merging of prediction result            
def generateNFoldCVnput(trainConfigFilePath, numCVFold=5, verbose=False):
    #Read preprocessing patients
    with open(trainConfigFilePath) as fp:
        previousConfig = json.load(fp)
        fp.close()    
    # resampledFilesLocation = previousConfig["resampledFilesLocation"]
    # patientVol_width = previousConfig["patientVol_width"]
    # patientVol_Height = previousConfig["patientVol_Height"]
    # patientVol_Depth = previousConfig["patientVol_Depth"]
    # sampleInput_Depth = previousConfig["sampleInput_Depth"]
    # splitFilesLocation = previousConfig["splitFilesLocation"]
    # patientList = previousConfig['patientList']
    # suffixList = previousConfig['suffixList']
    # listPatientsWithExtraSlice = previousConfig['listPatientsWithExtraSlice']
    # numDepthSplits = previousConfig['numDepthSplits'] 
    # prefixList = previousConfig['prefixList']
    #For N fold cross validation, we will only use those patients which do not 
    #have extra slices This will make later merging of split files easier. 
    #Though in truth, during testing we will surely encounter patients with 
    #extra slices, we might as well take care of them even now during cross validation.
    #Note that in truth, none of the original _ct.nii.gz or  _pt.nii.gz or 
    #_ct_gtvt.nii.gz files needs to be merged as they may be obtained from resampled 
    # directory. Only, for predicted _gtvt files, while merging the predicted split files
    #we might need an extra zero valued slice at the end of merging if the original
    #file had an extra slice before merging.
    newConfig = previousConfig
    patientList = newConfig['patientList']
    numPatients = len(patientList)
    numCVPatients = numPatients // numCVFold
    numTrainPatients = numPatients - numCVPatients
    if verbose:
        print('numPatients ', numPatients, ' numTrainPatients ', numTrainPatients, 'numCVPatients ', numCVPatients)
    newConfig['numCVFold'] = numCVFold
    newConfig['numTrainPatients'] = numTrainPatients
    newConfig['numCVPatients'] = numCVPatients
    newConfig['numPatients'] = numPatients
    for cvIndex in range(numCVFold):
        #newFileName = '{:>03d}_{}'.format(k,srcImgFileName)
        trainKey = 'train_{:>03d}_Patients'.format(cvIndex)
        cVKey = 'cv_{:>03d}_Patients'.format(cvIndex)
        startIdx_cv = cvIndex * numCVPatients
        endIdx_cv = (cvIndex+1) * numCVPatients
        #Note  argument-unpacking operator i.e. *.
        list_cvIdx = [*range(startIdx_cv, endIdx_cv)]
        list_trainIdx =  [*range(0, startIdx_cv)] + [*range(endIdx_cv, numPatients)]    
        cVPatients = [patientList[i] for i in list_cvIdx]
        trainPatients = [patientList[i] for i in list_trainIdx]
        if verbose:
            print('cvIndex ', cvIndex, ' list_cvIdx ', list_cvIdx, ' list_trainIdx ', list_trainIdx)
            print(cVKey, ' ', cVPatients, ' ', trainKey, ' ',trainPatients)
        newConfig[trainKey] = trainPatients
        newConfig[cVKey] = cVPatients

        # #Add other training parameters <-- Commented as they are hardcoded at the very beginning
        # newConfig['labels_to_train'] = [1]
        # newConfig['label_names'] = {"1": "GTV"}
        # newConfig['lr_flip'] = False
        # newConfig['label_symmetry_map'] = [[1,1]]
        # newConfig['translate_random']= 30.0       
        # newConfig['rotate_random']= 15.0          
        # newConfig['scale_random']= 0.2            
        # newConfig['change_intensity']= 0.05
        # newConfig['ct_low']= -1000       
        # newConfig['ct_high']= 3095          
        # newConfig['pt_low']= 0.0           
        # newConfig['pt_high']= 20.0
        # newConfig['data_format']= "channels_last"
        # newConfig['num_training_epochs']= 50

    with open(trainConfigFilePath, 'w') as fp:
        json.dump(newConfig, fp) #, indent='' #, indent=4
        fp.close()
    print('generateNFoldCVnput Finshed.')
    return
####################### Test code #####################
#createSplitFiles('/home/user/DMML/CodeAndRepositories/MMGTVSeg/input/trainInput_DSSENet.json',  verbose=False)
#generateNFoldCVnput('/home/user/DMML/CodeAndRepositories/MMGTVSeg/input/trainInput_DSSENet.json', numCVFold=5, verbose=False)

# with open('input/trainInput_DSSENet.json') as fp:
#         previousConfig = json.load(fp)
#         fp.close()   
# with open('input/trainInput_DSSENet.json', 'w') as fp:
#         json.dump(previousConfig, fp, ) #, indent='' #, indent=4
#         fp.close()

############################# FINAL ###################################

def createTrainInputJsonFile(jsonPath):
    dataLocation = "/home/user/DMML/CodeAndRepositories/MMGTVSeg/data/hecktor_train/resampled"
    #Empty dictionary
    trainConfig = {}
    #Insert members
    trainConfig['resampledFilesLocation'] = dataLocation
    trainConfig['suffixList'] = ["_ct.nii.gz", "_pt.nii.gz", "_ct_gtvt.nii.gz"]
    trainConfig["patientVol_width"] = 144
    trainConfig["patientVol_Height"] = 144
    trainConfig["patientVol_Depth"] = 144
    trainConfig['ct_low']= -1000       
    trainConfig['ct_high']= 3095          
    trainConfig['pt_low']= 0.0           
    trainConfig['pt_high']= 20.0
    trainConfig['labels_to_train'] = [1]
    trainConfig['label_names'] = {"1": "GTV"}
    trainConfig['lr_flip'] = False
    trainConfig['label_symmetry_map'] = [[1,1]]
    trainConfig['translate_random']= 30.0       
    trainConfig['rotate_random']= 15.0          
    trainConfig['scale_random']= 0.2            
    trainConfig['change_intensity']= 0.05
    trainConfig['num_training_epochs']= 25    
    trainConfig['data_format']= "channels_last"
    trainConfig['cpuOnlyFlag'] =  False
    trainConfig['lastSavedModelFolder'] = "/home/user/DMML/CodeAndRepositories/MMGTVSeg/output/DSSEModels"
    #Create train_CV patient list and test patient list
    #Randomize patient list : First we got file list, then dropped
    # the '_ct.nii.gz' to get patient name and then shuffled it
    patientList = [(os.path.basename(f)).replace('_ct.nii.gz','') \
      for f in glob.glob(dataLocation + '/*_ct.nii.gz', recursive=False) ]
    #Ranomize patient list
    random.shuffle(patientList)
    numAllPatients = len(patientList)
    numTrainCVPatients =  round(0.85*numAllPatients)
    numTestPatients = numAllPatients - numTrainCVPatients
    trainConfig['numTrainCVPatients'] = numTrainCVPatients
    trainConfig['trainCVPatientList'] = patientList[0:numTrainCVPatients]
    trainConfig['numTestPatients'] = numTestPatients
    trainConfig['testPatientList'] = patientList[numTrainCVPatients:numAllPatients]
    with open(jsonPath, 'w') as fp:
            json.dump(trainConfig, fp, ) #, indent='' #, indent=4
            fp.close()

#Generate trainInputJson file: Remember to call only once sice due to 
#random.shuffle trainCVPatientList and testPatientList changes every time 
# method is invoked.
createTrainInputJsonFile('/home/user/DMML/CodeAndRepositories/MMGTVSeg/input/temp.json')