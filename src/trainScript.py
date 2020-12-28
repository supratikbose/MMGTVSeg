import sys
sys.path.append('/home/user/DMML/CodeAndRepositories/MMGTVSeg')
import src
from src import  DSSENet
from DSSENet import DSSE_VNet
#import pprint
#pprint.pprint(sys.path)
#pprint.pprint(sys.modules)
DSSE_VNet.train(trainConfigFilePath = '/home/user/DMML/CodeAndRepositories/MMGTVSeg/input/trainInput_DSSENet.json',
                numCVFolds=5)