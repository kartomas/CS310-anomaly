import time
import pathlib
from matplotlib import rcParams
boundries = [[60,152],
        [50,175],
        [91,200],
        [31,168],
        [5,90, 140,200],
        [1,100, 110,200],
        [1,175],
        [1,94],
        [1,48],
        [1,140],
        [70,165],
        [130,200],
        [1,156],
        [1,200],
        [138,200],
        [123,200],
        [1,47],
        [54,120],
        [64,138],
        [45,175],
        [31,200],
        [16,107],
        [8,165],
        [50,171],
        [40,135],
        [77,144],
        [10,122],
        [105,200],
        [1,15, 45,113],
        [175,200],
        [1,180],
        [1,52, 65,115],
        [5,165],
        [1,121],
        [86,200],
        [15,108],
        ]
BATCH_SIZE = 1
IMG_HEIGHT = 176
IMG_WIDTH = 176
SHUFFLE_BUFFER_SIZE = 1000
SUBTRACT_MEAN_IMAGE = True
NUM_CHANNELS = 10
DATADIR = pathlib.Path('UCSD_Anomaly_Dataset.v1p2/UCSDped1back/Train/')
TESTPATH = pathlib.Path('UCSD_Anomaly_Dataset.v1p2/UCSDped1back/Test/')

rcParams.update({'figure.autolayout': True})
def timeIt(t):
    current = time.time()
    if t!=0:
        print(current-t)    
    return current