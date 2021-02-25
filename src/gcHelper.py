#GraphCut Heleper

import os
import numpy as np
import json
import nibabel as nib
from scribbleHelper import readAndScaleImageData, dice_multi_label

def generateGrahcutSegmentationFromScribble(predFromNN, binLimit,\
         fgScribble, bgScribble, imgForGC, segparams):
    """
    predFromNN : int8 binary volume of prediction from Neural Net with slice as first dimension    
    
    binLimit: axial, corronal and sagittal limits of the bounding box to be used in graphcut, 
    
    fgScribble: fgScribble  for imcut() : 0: unknown, 1 FG,
    bgScribble: bgScribble for imcut() : 0: unknown, 1 BG
    
    imgForGC: This image may be pet or CT or the softmax map or some layer to represent the
     multimodal combination. It is the same shape as that of gt (and predFromNN)  
     
    segparams: graphCut configuration  as defined in :
    1. https://github.com/mjirik/imcut#one-component-gaussian-model-and-one-object
    One component Gaussian model and one object
    pairwise_alpha control the complexity of the object shape. Higher pairwise_alpha => more compact shape.
    segparams = {
        'method': 'graphcut',
        "pairwise_alpha": 20,
        'modelparams': {
                'cvtype': 'full',
                "params": {"covariance_type": "full", "n_components": 1},
        },
        "return_only_object_with_seeds": True,
    }
    2. https://github.com/mjirik/imcut#gaussian-mixture-distribution-model-with-extern-feature-vector-function
    Gaussian mixture distribution model with extern feature vector function
    segparams = {
    'method': 'graphcut',
    'use_boundary_penalties': False,
    'boundary_dilatation_distance': 2,
    'boundary_penalties_weight': 1,
    'modelparams': {
        'type': 'gmmsame',
        'fv_type': "fv_extern",
        'fv_extern': fv_function,
        'adaptation': 'original_data',
    }
    'mdl_stored_file': False,
    }
    mdl_stored_file: if this is set, load model from file, you can see more in 
    function test_external_fv_with_save in pycut_test.py
    
    Returns  segmentation result
    """    
    import imcut.pycut
    #imCut requires fgSeeds to be 1 and bgSeeds to be 2
    #Convert scribbles into seeds for imcut() 
    # FG and BG seeds  are supposed to be disjoint; Still clip to 1
    # Multiply bgSribble by 2 to make bgSeeds
    fgSeeds = np.clip(fgScribble, 0, 1)
    bgSeeds = 2 * np.clip(bgScribble, 0, 1)
    #fgSeeds and bgSeeds are supposed to be disjoint. Still clip to 2
    seedsForGC = np.clip(fgSeeds + bgSeeds, 0, 2)    
    #Setup graph cut by cropping everything to bounding box limit
    [a_min, a_max, c_min, c_max, s_min, s_max] = binLimit
    seedsForGC = seedsForGC[a_min:a_max+1, c_min:c_max+1, s_min:s_max+1]
    imgForGC = imgForGC[a_min:a_max+1, c_min:c_max+1, s_min:s_max+1]
    gc = imcut.pycut.ImageGraphCut(img=imgForGC, segparams=segparams)
    gc.set_seeds(seedsForGC)
    gc.run()
    #segmentationResult with 1 as FG : note original gc.segmentation returns FG as 0
    boundedSegmentationResult = 1 - gc.segmentation
    gcSegResult = np.zeros_like(predFromNN)
    gcSegResult[a_min:a_max+1, c_min:c_max+1, s_min:s_max+1] = boundedSegmentationResult
    return gcSegResult      


def generateGrahcutSegmentationAndDiceFromJson(graphCutInputConfig_JsonFilePath):
    gcAndDiceResult = {}
    
    try:
        with open(graphCutInputConfig_JsonFilePath) as fp:
            graphCutInputConfig = json.load(fp)
            fp.close()
        
        expFolder = graphCutInputConfig["expFolder"]
        expPatName = graphCutInputConfig["expPatName"]
        binLimit = graphCutInputConfig["binLimit"]
        patDataConfig = graphCutInputConfig["patDataConfig"]
        autoScribbleAndGCConfig = graphCutInputConfig["autoScribbleAndGCConfig"]
        verbose=False
        
        ctFileName = expPatName + patDataConfig['ctSuffix']
        ctData = readAndScaleImageData(fileName=ctFileName,\
            folderName=expFolder, clipFlag = patDataConfig['ct_clipImages'],\
            clipLow=patDataConfig['ct_low'], clipHigh =patDataConfig['ct_high'],\
            scaleFlag=patDataConfig['ct_scaleImages'], scaleFactor=patDataConfig['ct_scaleFactor'],\
            meanSDNormalizeFlag = patDataConfig['ct_normalizeImages'], finalDataType = np.float32,\
            isLabelData=False, labels_to_train_list=None, verbose=verbose)
        
        ptFileName = expPatName + patDataConfig['ptSuffix']
        ptData = readAndScaleImageData(fileName=ptFileName,\
            folderName=expFolder, clipFlag = patDataConfig['pt_clipImages'],\
            clipLow=patDataConfig['pt_low'], clipHigh =patDataConfig['pt_high'],\
            scaleFlag=patDataConfig['pt_scaleImages'], scaleFactor=patDataConfig['pt_scaleFactor'],\
            meanSDNormalizeFlag = patDataConfig['pt_normalizeImages'], finalDataType = np.float32,\
            isLabelData=False, labels_to_train_list=None, verbose=verbose)
    
        softmaxFileName = expPatName + patDataConfig['softmaxSuffix']
        softmaxData = readAndScaleImageData(fileName=softmaxFileName,\
            folderName=expFolder, clipFlag = False,\
            clipLow=0.0, clipHigh =1.0,\
            scaleFlag=False, scaleFactor=1.0, meanSDNormalizeFlag = False, finalDataType = np.float32,\
            isLabelData=False, labels_to_train_list=None, verbose=verbose)
        
        gtFileName = expPatName + patDataConfig['gtSuffix']
        gtData = readAndScaleImageData(fileName=gtFileName,\
            folderName=expFolder, clipFlag = False,\
            clipLow=0, clipHigh = 0,\
            scaleFlag=False, scaleFactor=1, meanSDNormalizeFlag = False, finalDataType = np.int8, \
            isLabelData=True, labels_to_train_list=patDataConfig['labels_to_train'], verbose=verbose)
        
        predFileName = expPatName + patDataConfig['predSuffix']
        predFromNN = readAndScaleImageData(fileName=predFileName,\
            folderName=expFolder, clipFlag = False,\
            clipLow=0, clipHigh = 0,\
            scaleFlag=False, scaleFactor=1, meanSDNormalizeFlag = False, finalDataType = np.int8, \
            isLabelData=True, labels_to_train_list=patDataConfig['labels_to_train'], verbose=verbose)

        fgScribbleFileName = expPatName + patDataConfig['fgScribbleSuffix']
        fgScribble = readAndScaleImageData(fileName=fgScribbleFileName,\
            folderName=expFolder, clipFlag = False,\
            clipLow=0, clipHigh = 0,\
            scaleFlag=False, scaleFactor=1, meanSDNormalizeFlag = False, finalDataType = np.int8, \
            isLabelData=True, labels_to_train_list=patDataConfig['labels_to_train'], verbose=verbose)

        bgScribbleFileName = expPatName + patDataConfig['bgScribbleSuffix']
        bgScribble = readAndScaleImageData(fileName=bgScribbleFileName,\
            folderName=expFolder, clipFlag = False,\
            clipLow=0, clipHigh = 0,\
            scaleFlag=False, scaleFactor=1, meanSDNormalizeFlag = False, finalDataType = np.int8, \
            isLabelData=True, labels_to_train_list=patDataConfig['labels_to_train'], verbose=verbose)
        
        gcSegResult_ct =  generateGrahcutSegmentationFromScribble(\
                predFromNN=predFromNN, binLimit=binLimit,\
                fgScribble=fgScribble, bgScribble=bgScribble,\
                imgForGC=ctData, segparams=autoScribbleAndGCConfig['segparams_ssgc'])
        gcSegResult_pet =  generateGrahcutSegmentationFromScribble(\
                predFromNN=predFromNN, binLimit=binLimit,\
                fgScribble=fgScribble, bgScribble=bgScribble,\
                imgForGC=ptData, segparams=autoScribbleAndGCConfig['segparams_ssgc'])
        gcSegResult_softmax =  generateGrahcutSegmentationFromScribble(\
                predFromNN=predFromNN, binLimit=binLimit,\
                fgScribble=fgScribble, bgScribble=bgScribble,\
                imgForGC=softmaxData, segparams=autoScribbleAndGCConfig['segparams_ssgc'])
        
        modelImage_nii_aff = nib.load(os.path.join(expFolder, ctFileName)).affine
        
        gcSegResult_ct_FileName = expPatName + patDataConfig['ct_gCutSuffix']
        gcSegResult_ct_FilePath = os.path.join(expFolder, gcSegResult_ct_FileName)
        nib.save(nib.Nifti1Image(np.transpose(gcSegResult_ct, axes=(2,1,0)),\
                                 affine=modelImage_nii_aff),gcSegResult_ct_FilePath)
        
        gcSegResult_pet_FileName = expPatName + patDataConfig['pet_gCutSuffix']
        gcSegResult_pet_FilePath = os.path.join(expFolder, gcSegResult_pet_FileName)
        nib.save(nib.Nifti1Image(np.transpose(gcSegResult_pet, axes=(2,1,0)),\
                                 affine=modelImage_nii_aff),gcSegResult_pet_FilePath)
         
        gcSegResult_softmax_FileName = expPatName + patDataConfig['softmax_gCutSuffix']
        gcSegResult_softmax_FilePath = os.path.join(expFolder, gcSegResult_softmax_FileName)
        nib.save(nib.Nifti1Image(np.transpose(gcSegResult_softmax, axes=(2,1,0)),\
                                 affine=modelImage_nii_aff),gcSegResult_softmax_FilePath)
                   
        gcDice_ct = dice_multi_label(gcSegResult_ct, gtData)[0]
        gcDice_pet = dice_multi_label(gcSegResult_pet, gtData)[0]
        gcDice_softmax = dice_multi_label(gcSegResult_softmax, gtData)[0]

        gcAndDiceResult["successFlag"] = True
        gcAndDiceResult["patientName"] = graphCutInputConfig["patientName"]
        gcAndDiceResult["gcPath_ct"] = gcSegResult_ct_FilePath
        gcAndDiceResult["gcPath_pet"] = gcSegResult_pet_FilePath
        gcAndDiceResult["gcPath_softmax"] = gcSegResult_softmax_FilePath
        gcAndDiceResult["originalDice"] = dice_multi_label(predFromNN, gtData)[0]
        gcAndDiceResult["numFGS"] = graphCutInputConfig["numFGS"]
        gcAndDiceResult["numBGS"] = graphCutInputConfig["numBGS"]
        gcAndDiceResult["gcDice_ct"] = gcDice_ct
        gcAndDiceResult["gcDice_pet"] = gcDice_pet
        gcAndDiceResult["gcDice_softmax"] = gcDice_softmax
        
    except:
        gcAndDiceResult["successFlag"] = False
        gcAndDiceResult["patientName"] = ""
        gcAndDiceResult["gcPath_ct"] = ""
        gcAndDiceResult["gcPath_pet"] = ""
        gcAndDiceResult["gcPath_softmax"] = ""
        gcAndDiceResult["originalDice"] = 0.0
        gcAndDiceResult["numFGS"] = 0
        gcAndDiceResult["numBGS"] = 0
        gcAndDiceResult["gcDice_ct"] = 0.0
        gcAndDiceResult["gcDice_pet"] = 0.0
        gcAndDiceResult["gcDice_softmax"] = 0.0
        
    return gcAndDiceResult

def local_generateGrahcutSegmentationAndDiceFromJson(graphCutInputConfig_JsonFilePath):
    return generateGrahcutSegmentationAndDiceFromJson(graphCutInputConfig_JsonFilePath)

def remote_generateGrahcutSegmentationAndDiceFromJson(graphCutInputConfig_JsonFilePath):
    from graphCutClient import sendImCutRqstAndReceiveResult
    resultAvailable, gcAndDiceResult = \
        sendImCutRqstAndReceiveResult(graphCutInputConfig_JsonFilePath)
    return gcAndDiceResult