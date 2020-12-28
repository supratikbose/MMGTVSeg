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
from datetime import datetime

################################### Test code related with split file and TrainInputJason generation #######################################



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

# #Check why there is CRC check error <--- This was successful
# fileList = glob.glob('data/hecktor_train/split' + '/*.nii.gz', recursive=False) 
# numFilesInList = len(fileList)
# for id in range(0, numFilesInList):
#     filePath = fileList[id]
#     fileData = np.transpose(nib.load(filePath).get_fdata(), axes=(2,1,0))
#     print(filePath, ' ', fileData.min(), ' ', fileData.max())

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

#Test code for above methods
# createSplitFiles('/home/user/DMML/CodeAndRepositories/MMGTVSeg/input/trainInput_DSSENet.json',  verbose=False)
# generateNFoldCVnput('/home/user/DMML/CodeAndRepositories/MMGTVSeg/input/trainInput_DSSENet.json', numCVFold=5, verbose=False)

############################# Generate TrainInput.jason file including split information ######################################
# {
#     "lastSavedModel": "output/lastSaved.h5", 
#     "num_training_epochs": 50, 
#     "labels_to_train": [1], 
#     "label_names": {"1": "GTV"}, 
#     "lr_flip": false, 
#     "label_symmetry_map": [[1, 1]], 
#     "translate_random": 30.0, 
#     "rotate_random": 15.0, 
#     "scale_random": 0.2, 
#     "change_intensity": 0.05, 
#     "ct_low": -1000, 
#     "ct_high": 3095, 
#     "pt_low": 0.0, 
#     "pt_high": 25.0, 
#     "data_format": "channels_last", 
#     "resampledFilesLocation": "/home/user/DMML/CodeAndRepositories/MMGTVSeg/data/hecktor_train/resampled", 
#     "patientVol_width": 144, 
#     "patientVol_Height": 144, 
#     "patientVol_Depth": 144, 
#     "sampleInput_Depth": 16, 
#     "splitFilesLocation": "/home/user/DMML/CodeAndRepositories/MMGTVSeg/data/hecktor_train/split", 
#     "patientList": ["CHUM017", "CHGJ052", "CHUS097", "CHGJ089", "CHUS027", "CHGJ072", "CHMR001", "CHUM033", "CHUS057", "CHUM030", "CHGJ071", "CHUS087", "CHUM032", "CHUM007", "CHGJ050", "CHGJ038", "CHGJ070", "CHUS010", "CHUS043", "CHUS048", "CHUS064", "CHUM018", "CHUS007", "CHMR011", "CHUM014", "CHUM012", "CHUS060", "CHUS013", "CHUM021", "CHUM036", "CHGJ090", "CHUM010", "CHGJ082", "CHUM059", "CHUM026", "CHUS074", "CHGJ016", "CHUM038", "CHUS091", "CHGJ053", "CHGJ080", "CHGJ083", "CHUS076", "CHUS090", "CHUM013", "CHGJ029", "CHUS009", "CHUS006", "CHGJ078", "CHUS021", "CHUS031", "CHMR030", "CHUS046", "CHGJ073", "CHUM029", "CHUS083", "CHUM064", "CHGJ069", "CHUM037", "CHUM040", "CHGJ018", "CHGJ055", "CHMR014", "CHMR021", "CHGJ028", "CHUM002", "CHUS081", "CHGJ076", "CHUM006", "CHUM065", "CHUS096", "CHMR029", "CHUS067", "CHGJ017", "CHUM027", "CHUS051", "CHUM019", "CHUM056", "CHGJ035", "CHUM048", "CHUS080", "CHGJ048", "CHGJ065", "CHGJ066", "CHUS005", "CHUS047", "CHGJ030", "CHUS065", "CHUS042", "CHUM011", "CHUM008", "CHGJ085", "CHUS026", "CHGJ062", "CHUM023", "CHUM041", "CHGJ058", "CHUM055", "CHUM058", "CHUS061", "CHGJ025", "CHGJ039", "CHUS100", "CHMR040", "CHUS039", "CHUS035", "CHUS004", "CHUS008", "CHUM053", "CHUM016", "CHUS055", "CHMR013", "CHGJ057", "CHUM039", "CHUS053", "CHUM051", "CHUS088", "CHUS098", "CHUM060", "CHUS077", "CHUM044", "CHUS038", "CHGJ037", "CHUM022", "CHUS040", "CHUM049", "CHUM024", "CHUM001", "CHUS036", "CHUS049", "CHUS086", "CHMR020", "CHUM043", "CHUS058", "CHUS056", "CHGJ088", "CHUS052", "CHUS069", "CHUM054", "CHGJ007", "CHMR004", "CHUS073", "CHGJ086", "CHUM050", "CHUS101", "CHUS095", "CHUM035", "CHUS020", "CHUM034", "CHUS022", "CHUS003", "CHUS028", "CHUS089", "CHGJ032", "CHUM063", "CHMR016", "CHMR034", "CHUS045", "CHGJ077", "CHGJ087", "CHGJ031", "CHGJ074", "CHGJ008", "CHGJ034", "CHGJ010", "CHUS041", "CHUS085", "CHUS068", "CHUM046", "CHGJ046", "CHUS033", "CHGJ081", "CHGJ043", "CHGJ036", "CHUM062", "CHUM042", "CHUS015", "CHMR025", "CHMR005", "CHMR012", "CHUS016", "CHUM015", "CHUS078", "CHMR024", "CHUM057", "CHUS050", "CHUS094", "CHUS030", "CHUM045", "CHGJ026", "CHGJ091", "CHGJ015", "CHUS019", "CHUM061", "CHUS066", "CHMR023", "CHGJ092", "CHUM047", "CHGJ067", "CHMR028", "CHGJ013"], 
#     "suffixList": ["_ct.nii.gz", "_pt.nii.gz", "_ct_gtvt.nii.gz"], 
#     "listPatientsWithExtraSlice": ["CHUM017", "CHUS097", "CHMR011", "CHUM012", "CHUS060", "CHUM036", "CHUM038", "CHUM064", "CHUS096", "CHUM027", "CHUM048", "CHUM058", "CHGJ039", "CHMR040", "CHUM053", "CHUM043", "CHGJ088", "CHGJ008", "CHUM046", "CHGJ081", "CHUM062", "CHMR005", "CHMR012", "CHMR024", "CHGJ091", "CHUS019", "CHUM061", "CHMR023"], 
#     "numDepthSplits": 9, 
#     "prefixList": ["000", "001", "002", "003", "004", "005", "006", "007", "008"], 
#     "numCVFold": 5, 
#     "numTrainPatients": 161, 
#     "numCVPatients": 40, 
#     "numPatients": 201, 
#     "train_000_Patients": ["CHGJ080", "CHGJ083", "CHUS076", "CHUS090", "CHUM013", "CHGJ029", "CHUS009", "CHUS006", "CHGJ078", "CHUS021", "CHUS031", "CHMR030", "CHUS046", "CHGJ073", "CHUM029", "CHUS083", "CHUM064", "CHGJ069", "CHUM037", "CHUM040", "CHGJ018", "CHGJ055", "CHMR014", "CHMR021", "CHGJ028", "CHUM002", "CHUS081", "CHGJ076", "CHUM006", "CHUM065", "CHUS096", "CHMR029", "CHUS067", "CHGJ017", "CHUM027", "CHUS051", "CHUM019", "CHUM056", "CHGJ035", "CHUM048", "CHUS080", "CHGJ048", "CHGJ065", "CHGJ066", "CHUS005", "CHUS047", "CHGJ030", "CHUS065", "CHUS042", "CHUM011", "CHUM008", "CHGJ085", "CHUS026", "CHGJ062", "CHUM023", "CHUM041", "CHGJ058", "CHUM055", "CHUM058", "CHUS061", "CHGJ025", "CHGJ039", "CHUS100", "CHMR040", "CHUS039", "CHUS035", "CHUS004", "CHUS008", "CHUM053", "CHUM016", "CHUS055", "CHMR013", "CHGJ057", "CHUM039", "CHUS053", "CHUM051", "CHUS088", "CHUS098", "CHUM060", "CHUS077", "CHUM044", "CHUS038", "CHGJ037", "CHUM022", "CHUS040", "CHUM049", "CHUM024", "CHUM001", "CHUS036", "CHUS049", "CHUS086", "CHMR020", "CHUM043", "CHUS058", "CHUS056", "CHGJ088", "CHUS052", "CHUS069", "CHUM054", "CHGJ007", "CHMR004", "CHUS073", "CHGJ086", "CHUM050", "CHUS101", "CHUS095", "CHUM035", "CHUS020", "CHUM034", "CHUS022", "CHUS003", "CHUS028", "CHUS089", "CHGJ032", "CHUM063", "CHMR016", "CHMR034", "CHUS045", "CHGJ077", "CHGJ087", "CHGJ031", "CHGJ074", "CHGJ008", "CHGJ034", "CHGJ010", "CHUS041", "CHUS085", "CHUS068", "CHUM046", "CHGJ046", "CHUS033", "CHGJ081", "CHGJ043", "CHGJ036", "CHUM062", "CHUM042", "CHUS015", "CHMR025", "CHMR005", "CHMR012", "CHUS016", "CHUM015", "CHUS078", "CHMR024", "CHUM057", "CHUS050", "CHUS094", "CHUS030", "CHUM045", "CHGJ026", "CHGJ091", "CHGJ015", "CHUS019", "CHUM061", "CHUS066", "CHMR023", "CHGJ092", "CHUM047", "CHGJ067", "CHMR028", "CHGJ013"], 
#     "cv_000_Patients": ["CHUM017", "CHGJ052", "CHUS097", "CHGJ089", "CHUS027", "CHGJ072", "CHMR001", "CHUM033", "CHUS057", "CHUM030", "CHGJ071", "CHUS087", "CHUM032", "CHUM007", "CHGJ050", "CHGJ038", "CHGJ070", "CHUS010", "CHUS043", "CHUS048", "CHUS064", "CHUM018", "CHUS007", "CHMR011", "CHUM014", "CHUM012", "CHUS060", "CHUS013", "CHUM021", "CHUM036", "CHGJ090", "CHUM010", "CHGJ082", "CHUM059", "CHUM026", "CHUS074", "CHGJ016", "CHUM038", "CHUS091", "CHGJ053"], 
#     "train_001_Patients": ["CHUM017", "CHGJ052", "CHUS097", "CHGJ089", "CHUS027", "CHGJ072", "CHMR001", "CHUM033", "CHUS057", "CHUM030", "CHGJ071", "CHUS087", "CHUM032", "CHUM007", "CHGJ050", "CHGJ038", "CHGJ070", "CHUS010", "CHUS043", "CHUS048", "CHUS064", "CHUM018", "CHUS007", "CHMR011", "CHUM014", "CHUM012", "CHUS060", "CHUS013", "CHUM021", "CHUM036", "CHGJ090", "CHUM010", "CHGJ082", "CHUM059", "CHUM026", "CHUS074", "CHGJ016", "CHUM038", "CHUS091", "CHGJ053", "CHUS080", "CHGJ048", "CHGJ065", "CHGJ066", "CHUS005", "CHUS047", "CHGJ030", "CHUS065", "CHUS042", "CHUM011", "CHUM008", "CHGJ085", "CHUS026", "CHGJ062", "CHUM023", "CHUM041", "CHGJ058", "CHUM055", "CHUM058", "CHUS061", "CHGJ025", "CHGJ039", "CHUS100", "CHMR040", "CHUS039", "CHUS035", "CHUS004", "CHUS008", "CHUM053", "CHUM016", "CHUS055", "CHMR013", "CHGJ057", "CHUM039", "CHUS053", "CHUM051", "CHUS088", "CHUS098", "CHUM060", "CHUS077", "CHUM044", "CHUS038", "CHGJ037", "CHUM022", "CHUS040", "CHUM049", "CHUM024", "CHUM001", "CHUS036", "CHUS049", "CHUS086", "CHMR020", "CHUM043", "CHUS058", "CHUS056", "CHGJ088", "CHUS052", "CHUS069", "CHUM054", "CHGJ007", "CHMR004", "CHUS073", "CHGJ086", "CHUM050", "CHUS101", "CHUS095", "CHUM035", "CHUS020", "CHUM034", "CHUS022", "CHUS003", "CHUS028", "CHUS089", "CHGJ032", "CHUM063", "CHMR016", "CHMR034", "CHUS045", "CHGJ077", "CHGJ087", "CHGJ031", "CHGJ074", "CHGJ008", "CHGJ034", "CHGJ010", "CHUS041", "CHUS085", "CHUS068", "CHUM046", "CHGJ046", "CHUS033", "CHGJ081", "CHGJ043", "CHGJ036", "CHUM062", "CHUM042", "CHUS015", "CHMR025", "CHMR005", "CHMR012", "CHUS016", "CHUM015", "CHUS078", "CHMR024", "CHUM057", "CHUS050", "CHUS094", "CHUS030", "CHUM045", "CHGJ026", "CHGJ091", "CHGJ015", "CHUS019", "CHUM061", "CHUS066", "CHMR023", "CHGJ092", "CHUM047", "CHGJ067", "CHMR028", "CHGJ013"], 
#     "cv_001_Patients": ["CHGJ080", "CHGJ083", "CHUS076", "CHUS090", "CHUM013", "CHGJ029", "CHUS009", "CHUS006", "CHGJ078", "CHUS021", "CHUS031", "CHMR030", "CHUS046", "CHGJ073", "CHUM029", "CHUS083", "CHUM064", "CHGJ069", "CHUM037", "CHUM040", "CHGJ018", "CHGJ055", "CHMR014", "CHMR021", "CHGJ028", "CHUM002", "CHUS081", "CHGJ076", "CHUM006", "CHUM065", "CHUS096", "CHMR029", "CHUS067", "CHGJ017", "CHUM027", "CHUS051", "CHUM019", "CHUM056", "CHGJ035", "CHUM048"], 
#     "train_002_Patients": ["CHUM017", "CHGJ052", "CHUS097", "CHGJ089", "CHUS027", "CHGJ072", "CHMR001", "CHUM033", "CHUS057", "CHUM030", "CHGJ071", "CHUS087", "CHUM032", "CHUM007", "CHGJ050", "CHGJ038", "CHGJ070", "CHUS010", "CHUS043", "CHUS048", "CHUS064", "CHUM018", "CHUS007", "CHMR011", "CHUM014", "CHUM012", "CHUS060", "CHUS013", "CHUM021", "CHUM036", "CHGJ090", "CHUM010", "CHGJ082", "CHUM059", "CHUM026", "CHUS074", "CHGJ016", "CHUM038", "CHUS091", "CHGJ053", "CHGJ080", "CHGJ083", "CHUS076", "CHUS090", "CHUM013", "CHGJ029", "CHUS009", "CHUS006", "CHGJ078", "CHUS021", "CHUS031", "CHMR030", "CHUS046", "CHGJ073", "CHUM029", "CHUS083", "CHUM064", "CHGJ069", "CHUM037", "CHUM040", "CHGJ018", "CHGJ055", "CHMR014", "CHMR021", "CHGJ028", "CHUM002", "CHUS081", "CHGJ076", "CHUM006", "CHUM065", "CHUS096", "CHMR029", "CHUS067", "CHGJ017", "CHUM027", "CHUS051", "CHUM019", "CHUM056", "CHGJ035", "CHUM048", "CHUM044", "CHUS038", "CHGJ037", "CHUM022", "CHUS040", "CHUM049", "CHUM024", "CHUM001", "CHUS036", "CHUS049", "CHUS086", "CHMR020", "CHUM043", "CHUS058", "CHUS056", "CHGJ088", "CHUS052", "CHUS069", "CHUM054", "CHGJ007", "CHMR004", "CHUS073", "CHGJ086", "CHUM050", "CHUS101", "CHUS095", "CHUM035", "CHUS020", "CHUM034", "CHUS022", "CHUS003", "CHUS028", "CHUS089", "CHGJ032", "CHUM063", "CHMR016", "CHMR034", "CHUS045", "CHGJ077", "CHGJ087", "CHGJ031", "CHGJ074", "CHGJ008", "CHGJ034", "CHGJ010", "CHUS041", "CHUS085", "CHUS068", "CHUM046", "CHGJ046", "CHUS033", "CHGJ081", "CHGJ043", "CHGJ036", "CHUM062", "CHUM042", "CHUS015", "CHMR025", "CHMR005", "CHMR012", "CHUS016", "CHUM015", "CHUS078", "CHMR024", "CHUM057", "CHUS050", "CHUS094", "CHUS030", "CHUM045", "CHGJ026", "CHGJ091", "CHGJ015", "CHUS019", "CHUM061", "CHUS066", "CHMR023", "CHGJ092", "CHUM047", "CHGJ067", "CHMR028", "CHGJ013"], 
#     "cv_002_Patients": ["CHUS080", "CHGJ048", "CHGJ065", "CHGJ066", "CHUS005", "CHUS047", "CHGJ030", "CHUS065", "CHUS042", "CHUM011", "CHUM008", "CHGJ085", "CHUS026", "CHGJ062", "CHUM023", "CHUM041", "CHGJ058", "CHUM055", "CHUM058", "CHUS061", "CHGJ025", "CHGJ039", "CHUS100", "CHMR040", "CHUS039", "CHUS035", "CHUS004", "CHUS008", "CHUM053", "CHUM016", "CHUS055", "CHMR013", "CHGJ057", "CHUM039", "CHUS053", "CHUM051", "CHUS088", "CHUS098", "CHUM060", "CHUS077"], 
#     "train_003_Patients": ["CHUM017", "CHGJ052", "CHUS097", "CHGJ089", "CHUS027", "CHGJ072", "CHMR001", "CHUM033", "CHUS057", "CHUM030", "CHGJ071", "CHUS087", "CHUM032", "CHUM007", "CHGJ050", "CHGJ038", "CHGJ070", "CHUS010", "CHUS043", "CHUS048", "CHUS064", "CHUM018", "CHUS007", "CHMR011", "CHUM014", "CHUM012", "CHUS060", "CHUS013", "CHUM021", "CHUM036", "CHGJ090", "CHUM010", "CHGJ082", "CHUM059", "CHUM026", "CHUS074", "CHGJ016", "CHUM038", "CHUS091", "CHGJ053", "CHGJ080", "CHGJ083", "CHUS076", "CHUS090", "CHUM013", "CHGJ029", "CHUS009", "CHUS006", "CHGJ078", "CHUS021", "CHUS031", "CHMR030", "CHUS046", "CHGJ073", "CHUM029", "CHUS083", "CHUM064", "CHGJ069", "CHUM037", "CHUM040", "CHGJ018", "CHGJ055", "CHMR014", "CHMR021", "CHGJ028", "CHUM002", "CHUS081", "CHGJ076", "CHUM006", "CHUM065", "CHUS096", "CHMR029", "CHUS067", "CHGJ017", "CHUM027", "CHUS051", "CHUM019", "CHUM056", "CHGJ035", "CHUM048", "CHUS080", "CHGJ048", "CHGJ065", "CHGJ066", "CHUS005", "CHUS047", "CHGJ030", "CHUS065", "CHUS042", "CHUM011", "CHUM008", "CHGJ085", "CHUS026", "CHGJ062", "CHUM023", "CHUM041", "CHGJ058", "CHUM055", "CHUM058", "CHUS061", "CHGJ025", "CHGJ039", "CHUS100", "CHMR040", "CHUS039", "CHUS035", "CHUS004", "CHUS008", "CHUM053", "CHUM016", "CHUS055", "CHMR013", "CHGJ057", "CHUM039", "CHUS053", "CHUM051", "CHUS088", "CHUS098", "CHUM060", "CHUS077", "CHGJ031", "CHGJ074", "CHGJ008", "CHGJ034", "CHGJ010", "CHUS041", "CHUS085", "CHUS068", "CHUM046", "CHGJ046", "CHUS033", "CHGJ081", "CHGJ043", "CHGJ036", "CHUM062", "CHUM042", "CHUS015", "CHMR025", "CHMR005", "CHMR012", "CHUS016", "CHUM015", "CHUS078", "CHMR024", "CHUM057", "CHUS050", "CHUS094", "CHUS030", "CHUM045", "CHGJ026", "CHGJ091", "CHGJ015", "CHUS019", "CHUM061", "CHUS066", "CHMR023", "CHGJ092", "CHUM047", "CHGJ067", "CHMR028", "CHGJ013"], 
#     "cv_003_Patients": ["CHUM044", "CHUS038", "CHGJ037", "CHUM022", "CHUS040", "CHUM049", "CHUM024", "CHUM001", "CHUS036", "CHUS049", "CHUS086", "CHMR020", "CHUM043", "CHUS058", "CHUS056", "CHGJ088", "CHUS052", "CHUS069", "CHUM054", "CHGJ007", "CHMR004", "CHUS073", "CHGJ086", "CHUM050", "CHUS101", "CHUS095", "CHUM035", "CHUS020", "CHUM034", "CHUS022", "CHUS003", "CHUS028", "CHUS089", "CHGJ032", "CHUM063", "CHMR016", "CHMR034", "CHUS045", "CHGJ077", "CHGJ087"], 
#     "train_004_Patients": ["CHUM017", "CHGJ052", "CHUS097", "CHGJ089", "CHUS027", "CHGJ072", "CHMR001", "CHUM033", "CHUS057", "CHUM030", "CHGJ071", "CHUS087", "CHUM032", "CHUM007", "CHGJ050", "CHGJ038", "CHGJ070", "CHUS010", "CHUS043", "CHUS048", "CHUS064", "CHUM018", "CHUS007", "CHMR011", "CHUM014", "CHUM012", "CHUS060", "CHUS013", "CHUM021", "CHUM036", "CHGJ090", "CHUM010", "CHGJ082", "CHUM059", "CHUM026", "CHUS074", "CHGJ016", "CHUM038", "CHUS091", "CHGJ053", "CHGJ080", "CHGJ083", "CHUS076", "CHUS090", "CHUM013", "CHGJ029", "CHUS009", "CHUS006", "CHGJ078", "CHUS021", "CHUS031", "CHMR030", "CHUS046", "CHGJ073", "CHUM029", "CHUS083", "CHUM064", "CHGJ069", "CHUM037", "CHUM040", "CHGJ018", "CHGJ055", "CHMR014", "CHMR021", "CHGJ028", "CHUM002", "CHUS081", "CHGJ076", "CHUM006", "CHUM065", "CHUS096", "CHMR029", "CHUS067", "CHGJ017", "CHUM027", "CHUS051", "CHUM019", "CHUM056", "CHGJ035", "CHUM048", "CHUS080", "CHGJ048", "CHGJ065", "CHGJ066", "CHUS005", "CHUS047", "CHGJ030", "CHUS065", "CHUS042", "CHUM011", "CHUM008", "CHGJ085", "CHUS026", "CHGJ062", "CHUM023", "CHUM041", "CHGJ058", "CHUM055", "CHUM058", "CHUS061", "CHGJ025", "CHGJ039", "CHUS100", "CHMR040", "CHUS039", "CHUS035", "CHUS004", "CHUS008", "CHUM053", "CHUM016", "CHUS055", "CHMR013", "CHGJ057", "CHUM039", "CHUS053", "CHUM051", "CHUS088", "CHUS098", "CHUM060", "CHUS077", "CHUM044", "CHUS038", "CHGJ037", "CHUM022", "CHUS040", "CHUM049", "CHUM024", "CHUM001", "CHUS036", "CHUS049", "CHUS086", "CHMR020", "CHUM043", "CHUS058", "CHUS056", "CHGJ088", "CHUS052", "CHUS069", "CHUM054", "CHGJ007", "CHMR004", "CHUS073", "CHGJ086", "CHUM050", "CHUS101", "CHUS095", "CHUM035", "CHUS020", "CHUM034", "CHUS022", "CHUS003", "CHUS028", "CHUS089", "CHGJ032", "CHUM063", "CHMR016", "CHMR034", "CHUS045", "CHGJ077", "CHGJ087", "CHGJ013"], 
#     "cv_004_Patients": ["CHGJ031", "CHGJ074", "CHGJ008", "CHGJ034", "CHGJ010", "CHUS041", "CHUS085", "CHUS068", "CHUM046", "CHGJ046", "CHUS033", "CHGJ081", "CHGJ043", "CHGJ036", "CHUM062", "CHUM042", "CHUS015", "CHMR025", "CHMR005", "CHMR012", "CHUS016", "CHUM015", "CHUS078", "CHMR024", "CHUM057", "CHUS050", "CHUS094", "CHUS030", "CHUM045", "CHGJ026", "CHGJ091", "CHGJ015", "CHUS019", "CHUM061", "CHUS066", "CHMR023", "CHGJ092", "CHUM047", "CHGJ067", "CHMR028"]
# }

###################################  Test code using above json files #####################################
# with open('input/trainInput_DSSENet.json') as fp:
#         previousConfig = json.load(fp)
#         fp.close()   
# with open('input/trainInput_DSSENet.json', 'w') as fp:
#         json.dump(previousConfig, fp, ) #, indent='' #, indent=4
#         fp.close()
#This will be used for train and validation but not for testing where ground truth is not present
# class DSSENet_Generator(Sequence): 
#     def __init__(self,
#                 trainConfigFilePath,
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


    
#####################
#minC, maxC, minP, maxP, minG, maxG  -17466.458984375   32306.255859375   -0.17402704060077667   38.8752326965332   0.0   1.0    
#####################

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

# #Right-click on the file in File window. and select Run Current File in Python Python Interactive window?
# import matplotlib.pyplot as plt
# plt.plot([1, 2, 3, 4])
# plt.ylabel('some numbers')
# plt.show()

#Test code
import sys
sys.path.append('/home/user/DMML/CodeAndRepositories/MMGTVSeg')
import src
from src import  DSSENet
from DSSENet import DSSE_VNet

# import pprint
# pprint.pprint(sys.path)
#pprint.pprint(sys.modules)

##################################################
# trainGenerator = DSSE_VNet.DSSENet_Generator(
#     trainConfigFilePath = '/home/user/DMML/CodeAndRepositories/MMGTVSeg/input/trainInput_DSSENet.json',
#     useDataAugmentationDuringTraining = False,
#     batch_size = 1,
#     numCVFolds = 5,
#     cvFoldIndex = 2, #Can be between 0 to 4
#     isValidationFlag = False,
#     verbose=True
#     )

# numBatches = trainGenerator.__len__()
# batchX, batchY = trainGenerator.__getitem__(5)
# trainGenerator.displayBatchData(batchX, batchY, sampleInBatchId = 0, 
#                           startSliceId = 70, 
#                           endSliceId = 75, 
#                           pauseTime_sec = 0.5)

# for idx in range(0,numBatches):
#    batchX, batchY = trainGenerator.__getitem__(idx)    

##################################################################
# import tensorflow as tf
# import tensorflow.keras.backend as K
# from tensorflow.keras.models import Model
# from tensorflow.keras.layers import Input, Conv3D, UpSampling3D, Conv3DTranspose, Activation, Add, Concatenate, BatchNormalization, ELU, SpatialDropout3D, GlobalAveragePooling3D, Reshape, Dense, Multiply,  Permute
# from tensorflow.keras import regularizers, metrics
# from tensorflow.keras.utils import Sequence
# import tensorflow_addons as tfa

# trainConfigFilePath = '/home/user/DMML/CodeAndRepositories/MMGTVSeg/input/trainInput_DSSENet.json'
# cvFoldIndex = 0
# with open(trainConfigFilePath) as f:
#     trainInputParams = json.load(f)
#     f.close()

# thisFoldIntermediateModelFileName = "{:>02d}InterDSSENetModel.h5".format(cvFoldIndex)
# thisFoldFinalModelFileName = "{:>02d}FinalDSSENetModel.h5".format(cvFoldIndex)
# thisFoldIntermediateModelPath = os.path.join(trainInputParams["lastSavedModelFolder"],thisFoldIntermediateModelFileName)
# thisFoldFinalModelPath = os.path.join(trainInputParams["lastSavedModelFolder"],thisFoldFinalModelFileName)
# print(thisFoldIntermediateModelPath)
# print(thisFoldFinalModelPath)
# tb_logdir = './logs/' + os.path.splitext(os.path.basename(thisFoldIntermediateModelPath))[0] + '/' + datetime.now().strftime("%Y%m%d-%H%M%S")
# print(tb_logdir)

# ##### Check if you can load model that was saved into Tensorflow SavedModel #######
# ##### format and write it back into h5 format ########

# def dice_coef(y_true, y_pred, smooth = 0.00001, squared_denominator = False, ignore_zero_label = True):
#     num_dim = len(K.int_shape(y_pred)) 
#     num_labels = K.int_shape(y_pred)[-1]
#     reduce_axis = list(range(1, num_dim - 1))
#     y_true = y_true[..., 0]
#     dice = 0.0

#     if (ignore_zero_label == True):
#         label_range = range(1, num_labels)
#     else:
#         label_range = range(0, num_labels)

#     for i in label_range:
#         y_pred_b = y_pred[..., i]
#         y_true_b = K.cast(K.equal(y_true, i), K.dtype(y_pred))
#         intersection = K.sum(y_true_b * y_pred_b, axis = reduce_axis)        
#         if squared_denominator: 
#             y_pred_b = K.square(y_pred_b)
#         y_true_o = K.sum(y_true_b, axis = reduce_axis)
#         y_pred_o =  K.sum(y_pred_b, axis = reduce_axis)     
#         d = (2. * intersection + smooth) / (y_true_o + y_pred_o + smooth) 
#         dice = dice + K.mean(d)
#     dice = dice / len(label_range)
#     return dice

# def dice_loss(y_true, y_pred):
#     f = 1 - dice_coef(y_true, y_pred, smooth = 0.00001, squared_denominator = False, ignore_zero_label = False)
#     return f

# def dice_loss_fg(y_true, y_pred):
#     f = 1 - dice_coef(y_true, y_pred, smooth = 0.00001, squared_denominator = False, ignore_zero_label = True)
#     return f

# def modified_dice_loss(y_true, y_pred):
#     f = 1 - dice_coef(y_true, y_pred, smooth = 0.00001, squared_denominator = True, ignore_zero_label = False)
#     return f

# def modified_dice_loss_fg(y_true, y_pred):
#     f = 1 - dice_coef(y_true, y_pred, smooth = 0.00001, squared_denominator = True, ignore_zero_label = True)
#     return f
# #########
# model = tf.keras.models.load_model(thisFoldFinalModelPath, custom_objects={'dice_loss_fg': dice_loss_fg, 'modified_dice_loss': modified_dice_loss})
# print('Loaded model: ' + thisFoldFinalModelPath)
# model.save('/home/user/DMML/CodeAndRepositories/MMGTVSeg/output/00FinalDSSENetModel.h5',save_format='h5')
# newModel = tf.keras.models.load_model('/home/user/DMML/CodeAndRepositories/MMGTVSeg/output/00FinalDSSENetModel.h5', custom_objects={'dice_loss_fg': dice_loss_fg, 'modified_dice_loss': modified_dice_loss})
# print('Loaded NEW model: ' + '/home/user/DMML/CodeAndRepositories/MMGTVSeg/output/00FinalDSSENetModel.h5')
#############################################

##################################################
thisFoldFinalModelPath, test_msd, test_dice = DSSE_VNet.evaluateFold(
    trainConfigFilePath = '/home/user/DMML/CodeAndRepositories/MMGTVSeg/input/trainInput_DSSENet.json', 
    cvFoldIndex = 1,        
    numCVFolds = 5,                        
    savePredictions = True,
    out_dir = '/home/user/DMML/CodeAndRepositories/MMGTVSeg/output/evaluate_test/',
    thisFoldFinalModelPath = "",
    verbose=False)


# listOfModelPaths, listOfAverageDice, listOfAverageMSD, ensembleWeight = DSSE_VNet.evaluate(
#     trainConfigFilePath = '/home/user/DMML/CodeAndRepositories/MMGTVSeg/input/trainInput_DSSENet.json', 
#             numCVFolds = 5)
####################################### 

# TODO Changes to be copied into VM 


