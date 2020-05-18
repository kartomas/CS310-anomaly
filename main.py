import tensorflow as tf
from keras.models import load_model
from keras.datasets import cifar10
from keras.models import Model
import preprocessing
import config
import models
import testing
import numpy as np
TF_FORCE_GPU_ALLOW_GROWTH = True

def trainAndTest(filters, latent, trainingSet, fit=True, test2d=False, maxpool=False, batch=False, drop=False, customPath=None,stacked=False, test3d=False, use3dCNN=False, filter_size=3):
    global count
    # count+=1
    if(fit):
        if(use3dCNN): trainingSet=np.expand_dims(trainingSet, axis=4)
        print(trainingSet.shape)
        # if(for3d):
        X_train = tf.data.Dataset.from_tensor_slices((trainingSet, trainingSet))
        X_train = preprocessing.prepare_for_training(X_train, cache=None, shuffle_buffer_size=100) #"./c.tfcache"
    t = config.timeIt(0)
    folderName = ''
    for f in filters:
        folderName+=str(f)+'-'
    if latent is None:
        folderName+='no latent'
    # else:
    #     folderName+='l'+str(latent)+str(count)
    print(folderName)
    if(customPath is None):
        checkpoint_path = folderName+"/cp-x.ckpt"
    else:
        checkpoint_path=customPath+"/cp-x.ckpt"
    if(use3dCNN):
       autoen1, cp_callback = models.conv3dTuning(filters, latent, checkpoint_path, filter_size=filter_size, batch=batch, dropout=drop)
    elif(stacked):
        autoen1, cp_callback = models.modular2d3dims(filters, latent, checkpoint_path, filter_size=filter_size, batch=batch, dropout=drop)
    else:
        # autoen1, cp_callback = models.modular(filters, latent, checkpoint_path, batch=batch, dropout=drop)
        autoen1, cp_callback = models.final2D(checkpoint_path)
    # autoen1, cp_callback = modular([128,64,32], 16, checkpoint_path)
    print("time to set up model")
    config.timeIt(t)
    if(fit):
        trained1 = autoen1.fit(X_train, epochs=10, verbose=1, callbacks=[cp_callback])
    t = config.timeIt(0)
    X_train = None

    if(test2d):
        # testing.testSceneBySceneReg(autoen1, checkpoint_path, trainingSet)
        testing.testIt(autoen1, checkpoint_path)
    elif(test3d):
        testing.test3d(autoen1, checkpoint_path, extra_dim=use3dCNN)
    print("time to test model")
    config.timeIt(t)

    


def main():
    
  

    # checkpoint_dir = os.path.dirname(checkpoint_path)

    print("Num GPUs Available: ", len(tf.config.list_physical_devices('GPU')))
    t = config.timeIt(0)

    print('Time to create training data')
    config.timeIt(t)

    trainingSet = preprocessing.create_training_data3d(strided=True)

    trainAndTest([128,64],None, trainingSet, customPath='finalStacked', fit=False, stacked=True, test3d=True)
    trainAndTest([128,64],None, trainingSet, fit=False, customPath='final3d', use3dCNN=True, test3d=True)   


main()