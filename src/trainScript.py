import sys
import os
rootPath = '/home/user/DMML/CodeAndRepositories/MMGTVSeg'
sys.path.append('/home/user/DMML/CodeAndRepositories/MMGTVSeg')
import src
from src import  DSSENet
from DSSENet import DSSE_VNet
#import pprint
#pprint.pprint(sys.path)
#pprint.pprint(sys.modules)

#Train folds and save model
DSSE_VNet.train(trainConfigFilePath = os.path.join(rootPath, 'input/trainInput_DSSENet.json'),
                saveModelDirectory = os.path.join(rootPath, 'output/DSSEModels'),
                logDir = os.path.join(rootPath, 'logs'),
                numCVFolds=5)

#Evaluate folds and create ensemble weight
listOfModelPaths, listOfAverageDice, listOfAverageMSD, listOfEnsembleWeights = DSSE_VNet.evaluate(
    trainConfigFilePath = os.path.join(rootPath, 'input/trainInput_DSSENet.json'),
    saveModelDirectory = os.path.join(rootPath, 'output/DSSEModels'),
    out_dir = os.path.join(rootPath, 'output/evaluate_test'),
    numCVFolds = 5,
    trainModelEnsembleJsonPath_out = os.path.join(rootPath, 'output/trainModelCVEval_DSSENet.json')
    )

listOfTestPatientNames = ["CHUS097", "CHUM021", "CHGJ036", "CHUS026", "CHUM019", "CHUS015", "CHUM036", "CHUM022", "CHUM038", "CHUS013"]


DSSE_VNet.ensembleBasedPrediction(listOfTestPatientNames,
            resampledTestDataLocation = '/home/user/DMML/CodeAndRepositories/MMGTVSeg/data/hecktor_train/resampled',
            groundTruthPresent = True,
            trainConfigFilePath = os.path.join(rootPath, 'input/trainInput_DSSENet.json'),
            trainModelEnsembleJsonPath_in = os.path.join(rootPath, 'output/trainModelCVEval_DSSENet.json'),
            out_dir = os.path.join(rootPath, 'output/evaluate_test')
            )