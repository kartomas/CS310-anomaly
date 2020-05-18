import config
import os 
import numpy as np
import tensorflow as tf
import preprocessing
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from sklearn.base import TransformerMixin
from sklearn.metrics import confusion_matrix
from sklearn.metrics import roc_curve
from sklearn.metrics import roc_auc_score
from sklearn.metrics import precision_recall_curve
from sklearn.metrics import f1_score, auc
from scipy.optimize import brentq
from scipy.interpolate import interp1d



def test3d(model, name, extra_dim=False, savePicsNormalised=True):
    name = name.split("/")[0]+'/'
    reconstructionErrors = []
    trueLabels = []
    t = config.timeIt(0)
    scaleEachScene = True
    for i in range(1,37):
        # print('scene ' + str(i))
        scene, bound = preprocessing.loadScene3d(i)
        # if(extra_dim): scene=np.expand_dims(scene, axis=3)
        recErrorFrame, labels = predictScene3d(model, scene, bound, name+'scene'+str(i), useSequences=True, extra_dim=extra_dim, savePics=True, scaleEachScene=scaleEachScene)
        # print('scene dims')
        # print(scene.shape)

        # sequences_reconstruction_cost = np.array([np.linalg.norm(np.subtract(sequences[i],reconstructed_sequences[i])) for i in range(0,sz)])
        # print(sequences_reconstruction_cost.shape)
        # print(scene.shape)
        reconstructionErrors.extend(recErrorFrame)
        trueLabels.extend(labels)
    reconstructionErrors = np.array(reconstructionErrors)
    if(scaleEachScene):
        anomalyScores = reconstructionErrors
    else:
        anomalyScores = (reconstructionErrors-np.min(reconstructionErrors))/(np.max(reconstructionErrors)-np.min(reconstructionErrors))
    
    print(np.max(anomalyScores))
    
    rocAUC, prAUC, eer, thresh =getMetricsNew(trueLabels, anomalyScores, name, printConfusion=True)
    generatePictures(anomalyScores, trueLabels, name, thresh)
    print('for whole dataset ROC AUC is ' + str(rocAUC))
    print('for whole dataset PR AUC is ' + str(prAUC))
    print('for whole dataset eer is ' + str(eer))
    print('for whole dest thresh is ' + str(thresh))
    print('iterating trough scenes took')
    config.timeIt(t)

def generatePictures(scores, labels, name, thresh):

    t1 = np.arange(1,201,1)
    a=1
    for i in range(0,len(scores),200):
        sceneScores = scores[i:i+200]
        sceneLabels = labels[i:i+200]
        # print(sceneLabels)
        fig, ax = plt.subplots()
        b = np.min(sceneScores)
        t = np.max(sceneScores)

        lower1= None
        lower2= None
        upper1= None
        upper2= None
        bound = config.boundries[a-1]
        if((len(bound)) < 3):
            lower1 = bound[0]
            upper1 = bound[1]

        else:
            lower1 = bound[0]
            upper1 = bound[1]
            lower2 = bound[2]
            upper2 = bound[3]


        # print(lower1, upper1, lower2, upper2)
        if lower2 is None:
            plt.fill_between([lower1, upper1],b,t, color='red', alpha = 0.3, label='Frames that contain Anomalies')
        else:
            plt.fill_between([lower1, upper1],b,t, color='red', alpha = 0.3, label='Frames that contain Anomalies')
            plt.fill_between([lower2, upper2],b,t, color='red', alpha = 0.3)

        supper = np.ma.masked_where(sceneScores <= thresh, sceneScores)
        # print(supper)
        # supper = np.where(total <= thresh,)
        # print(supper)
        #normal?
        slower = np.ma.masked_where(sceneScores > thresh, sceneScores)

        plt.plot(t1,slower, 'ro', label='Frames classified as Anomalies')
        plt.plot(t1,supper, 'go', label='Frames classified as Normal')
        # ax.plot(t1,total, 'b-')
        plt.axhline(y=thresh, color='r', linestyle='-', label='Threshold')
        
        # # b, t = plt.ylim() # discover the values for bottom and top
        # # b -= 0.5 # Add 0.5 to the bottom
        # # t += 0.5 # Subtract 0.5 from the top
        # plt.fill_between([lower, upper],b,t, color='red', alpha = 0.3, label='Frames that contain Anomalies')
        plt.legend()
        fig.tight_layout()
        plt.xlabel('Frame Number')
        plt.ylabel('Regularity Score')

        # plt.plot(t1,lower, 'r-')
        # plt.plot(t1,upper, 'r-')
        # plt.axis([1,200])
        # plt.show()
        plt.savefig(name+'scene'+str(a)+'graphDS.jpg')
        plt.close('all')
        
        # b, t = plt.ylim() # discover the values for bottom and top
        # b -= 0.5 # Add 0.5 to the bottom
        # t += 0.5 # Subtract 0.5 from the top
        a+=1




def testIt(model, name):
    name = name.split("/")[0]+'/'
    reconstructionErrors = []
    trueLabels = []
    t = config.timeIt(0)
    for i in range(1,37):
            scene, bound = preprocessing.loadScene(i)
            thresh=None
            re, labels = testSceneReg(model, scene, bound, name+'scene'+str(i), thresh, savePics=True)
            # re, tl = predictScene(model, scene, bound)
            reconstructionErrors.extend(re)
            trueLabels.extend(labels)
            # print(reconstructionErrors)
            # print(trueLabels)
            # regScores.extend(regScore)
    reconstructionErrors = np.array(reconstructionErrors)
    # anomalyScores = (reconstructionErrors-np.min(reconstructionErrors))/(np.max(reconstructionErrors)-np.min(reconstructionErrors))

    rocAUC, prAUC, eer, thresh =getMetricsNew(trueLabels, reconstructionErrors, name, printConfusion=True)
    generatePictures(reconstructionErrors, trueLabels, name, thresh)
    print('for whole dataset ROC AUC is ' + str(rocAUC))
    print('for whole dataset PR AUC is ' + str(prAUC))
    print('for whole dataset eer is ' + str(eer))
    print('for whole dest thresh is ' + str(thresh))
    print('iterating trough scenes took')
    config.timeIt(t)
   
def predictScene3d(model, scene, bound, name, useSequences=False, extra_dim=False, savePics=False, scaleEachScene=False):
    # sz = len(scene[0]) * len(scene[1])
    # print('turetu but 200 ' + str(sz))
    # print(name)
    if(useSequences):
        reshaped = scene.reshape(200,config.IMG_HEIGHT,config.IMG_WIDTH)
        
        sz = reshaped.shape[0]
        # print('sz yra ' +str(sz))
        sequences = np.zeros((sz, config.NUM_CHANNELS, config.IMG_HEIGHT, config.IMG_WIDTH))
        
        for i in range(0, sz):
            clip = np.zeros((config.NUM_CHANNELS, config.IMG_HEIGHT, config.IMG_WIDTH))
            for j in range(0, config.NUM_CHANNELS):
                if i>=(sz-config.NUM_CHANNELS):
                    clip[j]=reshaped[i-j,:,:]
                else:
                    clip[j] = reshaped[i + j, :, :]
            sequences[i] = clip
        # print('nu ka cia ddaro paziurim')
        # print(sequences.shape)
        if(extra_dim): sequences = np.expand_dims(sequences, axis=4)
        sceneDS = tf.data.Dataset.from_tensor_slices(sequences).batch(config.BATCH_SIZE)
        reconstructed = model.predict(sceneDS)
        if(extra_dim):
            reconstructed = reconstructed.squeeze()
            sequences = sequences.squeeze()
        # reconstructed = reconstructed.reshape(200,256,256)
        # print(reconstructed.shape)
        recErrorFrame = np.array([np.linalg.norm(np.subtract(sequences[i],reconstructed[i])) for i in range(0,200)])
        # recErrorFrame = np.linalg.norm(sequences-reconstructed, axis=(1,2,3)) #(np.square(sequences-reconstructed)).sum(axis=(1,2,3))
        # extraFrames = 
        # for i in range(0,10):
        #     extracted = np.square(scene[190+i,:,:]-reconstructed[190+i,:,:])).mean(axis=(1,2))
        #     recErrorFrame.append(extracted)
        # print('koks sheipas paapendinus? ' + recErrorFrame.shape)
    else:
        if(extra_dim): scene = np.expand_dims(scene, axis=4)
        sceneDS = tf.data.Dataset.from_tensor_slices(scene).batch(config.BATCH_SIZE)
        reconstructed = model.predict(sceneDS)
        reconstructed = reconstructed.reshape(200,config.IMG_HEIGHT,config.IMG_WIDTH)
        scene = scene.reshape(200,config.IMG_HEIGHT,config.IMG_WIDTH)
        # print(reconstructed.shape)
        recErrorFrame = (np.square(scene-reconstructed)).sum(axis=(1,2))



    if(savePics or scaleEachScene):
        t1 = np.arange(1,201,1)
        if(useSequences):
            if(extra_dim): scene = np.expand_dims(scene, axis=4)
            sceneDS = tf.data.Dataset.from_tensor_slices(scene).batch(config.BATCH_SIZE)
            reconstructed = model.predict(sceneDS)
            reconstructed = reconstructed.reshape(200,config.IMG_HEIGHT,config.IMG_WIDTH)
            scene = scene.reshape(200,config.IMG_HEIGHT,config.IMG_WIDTH)

        localScale = (recErrorFrame-np.min(recErrorFrame))/(np.max(recErrorFrame))
        localScale = 1- localScale
        saveScenePics(scene, reconstructed, name+'local')

        #split anomalies and
        # print(total.shape)
        # print(total)
        fig, ax = plt.subplots()
        b = np.min(localScale)
        t = np.max(localScale)

    trueLabels = np.empty(200)
    if((len(bound)) < 3):
        lower = bound[0]
        upper = bound[1]
        # print(upper)
        # print(lower)
        trueLabels[:lower-1] = 0
        trueLabels[upper:] = 0
        trueLabels[lower-1:upper] = 1

    else:
        lower1 = bound[0]
        upper1 = bound[1]
        lower2 = bound[2]
        upper2 = bound[3]

        # print(lower1)
        # print(lower2)
        # print(upper1)
        # print(upper2)

        trueLabels[:lower1-1] = 0
        trueLabels[upper1:lower2-1] = 0
        trueLabels[upper2:] = 0

        trueLabels[lower1-1:upper1] = 1
        trueLabels[lower2-1:upper2] = 1

    # print(trueLabels)
    trueLabels = 1- trueLabels

    if((np.any(trueLabels)) and (scaleEachScene or savePics)): rocAUC, prAUC, eer, thresh = getMetricsNew(trueLabels, localScale, name)
    else:
        rocAUC =0
        prAUC =0
        eer =0
        thresh=0.5
    # print('for scene bst thr is ' + str(thresh))
    # print('for scene  ROC AUC is ' + str(rocAUC))
    # print('for scene  PR AUC is ' + str(prAUC))
    # print('for scene eer is ' + str(eer))

    
    if(savePics):
        if((len(bound)) < 3):
            plt.fill_between([lower, upper],b,t, color='red', alpha = 0.3, label='Frames that contain Anomalies')
        else:
            plt.fill_between([lower1, upper1],b,t, color='red', alpha = 0.3, label='Frames that contain Anomalies')
            plt.fill_between([lower2, upper2],b,t, color='red', alpha = 0.3)

        supper = np.ma.masked_where(localScale <= thresh, localScale)
        # print(supper)
        # supper = np.where(total <= thresh,)
        # print(supper)
        #normal?
        slower = np.ma.masked_where(localScale > thresh, localScale)

        plt.plot(t1,slower, 'ro', label='Frames classified as Anomalies')
        plt.plot(t1,supper, 'go', label='Frames classified as Normal')
        # ax.plot(t1,total, 'b-')
        plt.axhline(y=thresh, color='r', linestyle='-', label='Threshold')
        
        # # b, t = plt.ylim() # discover the values for bottom and top
        # # b -= 0.5 # Add 0.5 to the bottom
        # # t += 0.5 # Subtract 0.5 from the top
        # plt.fill_between([lower, upper],b,t, color='red', alpha = 0.3, label='Frames that contain Anomalies')
        plt.legend()
        fig.tight_layout()
        plt.xlabel('Frame Number')
        plt.ylabel('Regularity Score')

        # plt.plot(t1,lower, 'r-')
        # plt.plot(t1,upper, 'r-')
        # plt.axis([1,200])
        # plt.show()
        plt.savefig(name+'graphlocal.jpg')
        plt.close('all')
    # acc = findAccuracy(normal, anomaly, thresh)

    # getMetricsNew(trueLabels, localScale, name+'local')
    if(scaleEachScene):
        return localScale.squeeze(), trueLabels
    else:
         return recErrorFrame.squeeze(), trueLabels

def predictScene(model, scene, bound):
    sceneDS = tf.data.Dataset.from_tensor_slices(scene).batch(config.BATCH_SIZE)
    reconstructed = model.predict(sceneDS)
    # recErrorFrame = (np.square(scene-reconstructed)).mean(axis=(1,2))
    recErrorFrame = np.array([np.linalg.norm(np.subtract(sequences[i],reconstructed[i])) for i in range(0,200)])

    # recErrorFrame = np.linalg.norm(scene-reconstructed,ord=2, axis=(1,2))
    trueLabels = np.empty(200)
    if((len(bound)) < 3):
        lower = bound[0]-1
        upper = bound[1]
        trueLabels[:lower] = 0
        trueLabels[upper:] = 0
        trueLabels[lower:upper] = 1
        # normal = np.concatenate((recErrorFrame[:lower], recErrorFrame[upper:]), axis=0)
        # anomaly = recErrorFrame[lower:upper]
    else:
        lower1 = bound[0]-1
        upper1 = bound[1]
        lower2 = bound[2]-1
        upper2 = bound[3]

        trueLabels[:lower1] = 0
        trueLabels[upper1:lower2] = 0
        trueLabels[upper2:] = 0

        trueLabels[lower1:upper1] = 1
        trueLabels[lower2:upper2] = 1

    return recErrorFrame.squeeze(), trueLabels

    
def testSceneReg(model, scene, bound, name, thresh=None, savePics=False):

    sceneDS = tf.data.Dataset.from_tensor_slices(scene).batch(config.BATCH_SIZE)
    reconstructed = model.predict(sceneDS)

    t1 = np.arange(1,201,1)
    # print(t1)
    #generate an array of MSEs for all frames of the scene
    # recErrorFrame = (np.square(scene-reconstructed)).mean(axis=(1,2))
    recErrorFrame = np.array([np.linalg.norm(np.subtract(scene[i],reconstructed[i])) for i in range(0,200)])
    
    localScale = (recErrorFrame-np.min(recErrorFrame))/(np.max(recErrorFrame))
    localScale = 1- localScale


    if(savePics):
        # print('saving pics')
        saveScenePics(scene, reconstructed, name+'local')

        #split anomalies and
        # print(total.shape)
        # print(total)
        fig, ax = plt.subplots()
        b = np.min(localScale)
        t = np.max(localScale)

   
    trueLabels = np.empty(200)

    if((len(bound)) < 3):
        lower = bound[0]
        upper = bound[1]
        # print(upper)
        # print(lower)
        trueLabels[:lower-1] = 0
        trueLabels[upper:] = 0
        trueLabels[lower-1:upper] = 1

        # if(savePics): ax.fill_between([lower, upper],b,t, color='red', alpha = 0.3, label='Frames that contain Anomalies')
    else:
        lower1 = bound[0]
        upper1 = bound[1]
        lower2 = bound[2]
        upper2 = bound[3]

        # print(lower1)
        # print(lower2)
        # print(upper1)
        # print(upper2)

        trueLabels[:lower1-1] = 0
        trueLabels[upper1:lower2-1] = 0
        trueLabels[upper2:] = 0

        trueLabels[lower1-1:upper1] = 1
        trueLabels[lower2-1:upper2] = 1
     
    trueLabels = 1- trueLabels
    
    
    if(np.any(trueLabels) and savePics): rocAUC, prAUC, eer, thresh = getMetricsNew(trueLabels, localScale, name)
    else:
        rocAUC =0
        prAUC =0
        eer =0
        thresh=0.5
    # if(thresh is None):
    #      thresh, acc = findThreshold(normalScaled, anomalyScaled)
    # else:  
    #     localThresh, acc = findThreshold(normal, anomaly)
    if(savePics):
        if((len(bound)) < 3):
            plt.fill_between([lower, upper],b,t, color='red', alpha = 0.3, label='Frames that contain Anomalies')
        else:
            plt.fill_between([lower1, upper1],b,t, color='red', alpha = 0.3, label='Frames that contain Anomalies')
            plt.fill_between([lower2, upper2],b,t, color='red', alpha = 0.3)

        supper = np.ma.masked_where(localScale <= thresh, localScale)
        # print(supper)
        # supper = np.where(total <= thresh,)
        # print(supper)
        #normal?
        slower = np.ma.masked_where(localScale > thresh, localScale)

        plt.plot(t1,slower, 'ro', label='Frames classified as Anomalies')
        plt.plot(t1,supper, 'go', label='Frames classified as Normal')
        # ax.plot(t1,total, 'b-')
        plt.axhline(y=thresh, color='r', linestyle='-', label='Threshold')
        
        # # b, t = plt.ylim() # discover the values for bottom and top
        # # b -= 0.5 # Add 0.5 to the bottom
        # # t += 0.5 # Subtract 0.5 from the top
        # plt.fill_between([lower, upper],b,t, color='red', alpha = 0.3, label='Frames that contain Anomalies')
        plt.legend()
        fig.tight_layout()
        plt.xlabel('Frame Number')
        plt.ylabel('Regularity Score')

        # plt.plot(t1,lower, 'r-')
        # plt.plot(t1,upper, 'r-')
        # plt.axis([1,200])
        # plt.show()
        plt.savefig(name+'graphlocal.jpg')
        plt.close('all')
    # print(normal.squeeze().shape)
    # print(anomaly.squeeze().shape)
    return localScale.squeeze(), trueLabels



def findThreshold(normal, anomaly, steps=100):

    total = np.concatenate((normal, anomaly), axis=0)
    minVal = np.min(total)
    maxVal = np.max(total)
    rangeVal = maxVal - minVal
    stepSize = rangeVal/steps
    # print(rangeVal, stepSize)

    normalSize = len(normal)
    anomalySize = len(anomaly)
    bestAcc = 0
    bestThresh = 0
    thr = 0

    for a in range(steps):
        thr = minVal+a*stepSize
        acc = findAccuracy(normal,anomaly,thr)
        if acc>bestAcc:
            bestAcc = acc
            bestThresh = thr

    # print('Best threshold is ' + str(bestThresh))
    # print('Best accuracy is ' + str(bestAcc))
    return bestThresh, bestAcc

def plot_confusion_matrix(cm, classes,
                          title='Confusion matrix',
                          cmap=plt.cm.Blues, normalize=False):

    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation='vertical')
    plt.yticks(tick_marks, classes)
    b, t = plt.ylim() # discover the values for bottom and top
    # b += 0.5 # Add 0.5 to the bottom
    # t -= 0.5 # Subtract 0.5 from the top
    plt.ylim(b, t) # update the ylim(bottom, top) values

    #Normalize the matrix
    # print(cm.astype('float'))
    # print(cm.sum(axis=1)[:, np.newaxis])
    if(normalize):
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
    print(cm)
    #round numbers
    cm = np.round(cm,2)
    #print matrix in a simple way
    print(cm)

    thresh = cm.max() * 0.7
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            plt.text(j, i, cm[i, j],
                     ha='center',
                     va='center',
                     color="white" if cm[i, j] > thresh else "black")


    plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predicted label')

def getMetricsNew(trueVals, predictedVals, path, printConfusion=False):
   
    fullCount = len(trueVals)

    normalCount = np.count_nonzero(trueVals)
    anomalyCount = fullCount-normalCount
    # print(fullCount)
    # print(anomalyCount)
    # print(normalCount)
    largerDS = normalCount if normalCount >= anomalyCount else anomalyCount


    ns_probs = [1 for _ in range(len(trueVals))]
    roc_AUC = roc_auc_score(trueVals, predictedVals)
    # nsAUC = roc_auc_score(trueVals, ns_probs)
    # print('auc score ' + str(modelAUC))
    ns_fpr, ns_tpr, _ = roc_curve(trueVals, ns_probs)
    model_fpr, model_tpr, thresholds = roc_curve(trueVals, predictedVals)
    # tp = model_tpr * anomalyCount
    # tn = (1-model_fpr) * normalCount
    tnr = 1-model_fpr
    # accs = (tp+tn) / fullCount
    balancedAccs = (tnr + model_tpr )/2

    # print(model_fpr)
    # print(model_tpr)
    # print(balancedAccs)
    bestIndex = np.argmax(balancedAccs)
    best_thresh = thresholds[bestIndex]
    # print('best thresh from roc curve is ' + str(best_thresh))
    print('best thresh gives balanced acc ' + str(balancedAccs[bestIndex]))
    # print('best thresh gives regular acc ' + str(accs[bestIndex]))

    plt.figure()
    plt.plot(ns_fpr, ns_tpr, linestyle='--', label='No Skill')
    plt.plot(model_fpr, model_tpr, marker='.', label='Model')
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.legend()
    plt.savefig(path+'auc.jpg')
    plt.close('all')

    precision, recall, thresholds = precision_recall_curve(trueVals, predictedVals)

    pr_AUC = auc(recall, precision)
    # print('approx auc score ' + str(apporxAUC))

    no_skill=normalCount/fullCount
    plt.plot([0, 1], [no_skill, no_skill], linestyle='--', label='No Skill')
    plt.plot(recall, precision, marker='.', label='Model')
        # axis labels
    plt.xlabel('Recall')
    plt.ylabel('Precision')
    # show the legend
    plt.legend()
    plt.savefig(path+'precision-recall.jpg')
    plt.close('all')
    eer = brentq(lambda x : 1. - x - interp1d(model_fpr, model_tpr)(x), 0., 1.)
    if(printConfusion):
        # normalTrue = np.full(normalSize, 0)
        # anomalyTrue = np.full(anomalySize, 1)
        # trueVals = np.concatenate((anomalyTrue, normalTrue))
        predictions = np.where(predictedVals <= best_thresh,1,0)
        # print(normalPred)
        # anomalyPred = np.where(anomaly > thresh,1,0)
        # predictedVals = np.concatenate((anomalyPred, normalPred))
        cm =confusion_matrix(trueVals,predictions)
        print(cm)
        plt.figure(figsize=(5,5))
        plot_confusion_matrix(cm, ['Anomaly', 'Normal'])
        plt.savefig(path+'conf.jpg')
        plt.close('all')
    # print('eer is ' + str(eer))
    return roc_AUC, pr_AUC, eer, best_thresh

def getMetricsReg(normal, anomaly, path):
    normalSize = len(normal)
    anomalySize = len(anomaly)
    if(normalSize==0): return
    # print(normalSize)
    # print(anomalySize)
    largerDS = normalSize if normalSize >= anomalySize else anomalySize

    normalTrue = np.full(normalSize, 0)
    anomalyTrue = np.full(anomalySize, 1)
    trueVals = np.concatenate((anomalyTrue, normalTrue))
    # normalPred = np.where(normal <= thresh,1,0)
    # # print(normalPred)
    # anomalyPred = np.where(anomaly > thresh,0,1)
    predictedVals = np.concatenate((anomaly, normal))
    ns_probs = [1 for _ in range(len(trueVals))]
    roc_AUC = roc_auc_score(trueVals, predictedVals)
    # nsAUC = roc_auc_score(trueVals, ns_probs)
    # print('auc score ' + str(modelAUC))
    ns_fpr, ns_tpr, _ = roc_curve(trueVals, ns_probs)
    model_fpr, model_tpr, _ = roc_curve(trueVals, predictedVals)
    plt.figure()
    plt.plot(ns_fpr, ns_tpr, linestyle='--', label='No Skill')
    plt.plot(model_fpr, model_tpr, marker='.', label='Model')
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.legend()
    plt.savefig(path+'confusion.jpg')
    plt.close('all')

    precision, recall, thresholds = precision_recall_curve(trueVals, predictedVals)

    print(np.max(predictedVals))
    print(thresholds)
    tp = recall*anomalySize
    fp = (tp/precision) - tp
    tn = normalSize -fp
    acc = (tp+tn) / (anomalySize+normalSize)
    best_thresh = thresholds[np.argmax(acc)]
    print('best thresh from pc curve is ' + str(best_thresh))
    # modelF1 = f1_score(trueVals, predictedVals)
    # print('f1 score ' + str(modelF1))
    pr_AUC = auc(recall, precision)
    # print('approx auc score ' + str(apporxAUC))

    no_skill=largerDS/len(trueVals)
    plt.plot([0, 1], [no_skill, no_skill], linestyle='--', label='No Skill')
    plt.plot(recall, precision, marker='.', label='Model')
        # axis labels
    plt.xlabel('Recall')
    plt.ylabel('Precision')
    # show the legend
    plt.legend()
    plt.savefig(path+'precision-recall.jpg')
    plt.close('all')
    eer = brentq(lambda x : 1. - x - interp1d(model_fpr, model_tpr)(x), 0., 1.)
    # print('eer is ' + str(eer))
    return roc_AUC, pr_AUC, eer


def findAccuracy(normal, anomaly, thresh, printConfusion=False, calculateMetrics=False, path=None):
    normalSize = len(normal)
    anomalySize = len(anomaly)
    # print(normalSize)
    # print(anomalySize)
    tp = 0
    fp = 0
    tn = 0
    fn = 0

    for a in normal:
        if a<=thresh:
            tn+=1
        else:
            fp+=1

    for a in anomaly:
       if a>thresh:
           tp+=1
       else:
           fn+=1

    if printConfusion:
        normalTrue = np.full(normalSize, 0)
        anomalyTrue = np.full(anomalySize, 1)
        trueVals = np.concatenate((anomalyTrue, normalTrue))
        normalPred = np.where(normal <= thresh,0,1)
        # print(normalPred)
        anomalyPred = np.where(anomaly > thresh,1,0)
        predictedVals = np.concatenate((anomalyPred, normalPred))
        cm =confusion_matrix(trueVals,predictedVals)
        print(cm)
        plt.figure(figsize=(5,5))
        plot_confusion_matrix(cm, ['Anomaly', 'Normal'])
        plt.show()
        tpr= (tp/anomalySize)*100
        tnr= (tn/normalSize)*100
        acc= (tpr + tnr)/2
        print("true positive " + str(tpr))
        print("true negative " + str(tnr))
        print("thresh " + str(thresh))
        
        print("balanced accuracy" + str(acc))

    if(normalSize==0):
        acc=tp/anomalySize
    else:
        tpr= (tp/anomalySize)*100
        tnr= (tn/normalSize)*100
        acc= (tpr + tnr)/2
   
    return acc

def top10errors(normal, anomaly, name):

    print('saving picture')
    print(name)
    plt.figure(figsize=(30,10))
    # plt.title(name)

    for i in range(0,10):
        ax = plt.subplot(4, 10, i+1,)
        ax.set_title(normal[i][3], fontsize=6)
        plt.imshow(normal[i][1].reshape(config.IMG_HEIGHT, config.IMG_WIDTH), interpolation='nearest')
        plt.gray()
        ax.get_xaxis().set_visible(False)
        ax.get_yaxis().set_visible(False)

    for i in range(0,10):
        ax = plt.subplot(4, 10, i+10+1)
        ax.set_title('mse ' + str(normal[i][0]), fontsize=6)
        plt.imshow(normal[i][2].reshape(config.IMG_HEIGHT, config.IMG_WIDTH), interpolation='nearest')
        plt.gray()
        ax.get_xaxis().set_visible(False)
        ax.get_yaxis().set_visible(False)

    for i in range(0,10):
        ax = plt.subplot(4, 10, i+20+1)
        ax.set_title(anomaly[i][3], fontsize=6)
        plt.imshow(anomaly[i][1].reshape(config.IMG_HEIGHT, config.IMG_WIDTH), interpolation='nearest')
        plt.gray()
        ax.get_xaxis().set_visible(False)
        ax.get_yaxis().set_visible(False)

    for i in range(0,10):
        ax = plt.subplot(4, 10, i+30+1)
        ax.set_title('mse ' + str(anomaly[i][0]), fontsize=6)
        plt.imshow(anomaly[i][2].reshape(config.IMG_HEIGHT, config.IMG_WIDTH), interpolation='nearest')
        plt.gray()
        ax.get_xaxis().set_visible(False)
        ax.get_yaxis().set_visible(False)

    plt.tight_layout()
    plt.savefig(name+'.jpg')
    plt.close('all')

def saveScenePics(scene, reconstructed, name):
    total = np.absolute(scene-reconstructed)
    total = np.squeeze(total)
    # print(name)
    plt.figure(figsize=(30,10))
    # print(total.shape)
    # print(scene.shape)
    scaler = NDStandardScaler()
    scaledData = scaler.fit_transform(total)
   
    a=0
    for i in range(0,10):        
        ax = plt.subplot(6, 10, i+1)
        ax.set_title(a)
        plt.imshow(scene[a].reshape(config.IMG_HEIGHT, config.IMG_WIDTH), interpolation='nearest')
        plt.gray()
        ax.get_xaxis().set_visible(False)
        ax.get_yaxis().set_visible(False)
        a+=10

    for i in range(0,10):
        ax = plt.subplot(6, 10, i+30+1)
        ax.set_title(a)
        plt.imshow(scene[a].reshape(config.IMG_HEIGHT, config.IMG_WIDTH), interpolation='nearest')
        plt.gray()
        ax.get_xaxis().set_visible(False)
        ax.get_yaxis().set_visible(False)
        a+=10

    a=0
    for i in range(0,10):
        ax = plt.subplot(6, 10, i+10+1)
        plt.imshow(reconstructed[a].reshape(config.IMG_HEIGHT, config.IMG_WIDTH), interpolation='nearest')
        plt.gray()
        ax.get_xaxis().set_visible(False)
        ax.get_yaxis().set_visible(False)
        a+=10

    for i in range(0,10):
        ax = plt.subplot(6, 10, i+40+1)
        plt.imshow(reconstructed[a].reshape(config.IMG_HEIGHT, config.IMG_WIDTH), interpolation='nearest')
        plt.gray()
        ax.get_xaxis().set_visible(False)
        ax.get_yaxis().set_visible(False)
        a+=10

    a=0

    for i in range(0,10):
        ax = plt.subplot(6, 10, i+20+1)
        # ax.set_title(total[a], fontsize=6)
        plt.imshow(scaledData[a].reshape(config.IMG_HEIGHT, config.IMG_WIDTH), cmap='viridis', interpolation='nearest')
        # plt.gray()
        ax.get_xaxis().set_visible(False)
        ax.get_yaxis().set_visible(False)
        a+=10

    for i in range(0,10):
        ax = plt.subplot(6, 10, i+50+1)
        # ax.set_title(total[a], fontsize=6)
        plt.imshow(scaledData[a].reshape(config.IMG_HEIGHT, config.IMG_WIDTH), cmap='viridis',interpolation='nearest')
        # plt.gray()
        ax.get_xaxis().set_visible(False)
        ax.get_yaxis().set_visible(False)
        a+=10

    plt.tight_layout()
    plt.savefig(name+'reconstruction.jpg')
    plt.close('all')

class NDStandardScaler(TransformerMixin):
    def __init__(self, **kwargs):
        self._scaler = StandardScaler(copy=True, **kwargs)
        self._orig_shape = None

    def fit(self, X, **kwargs):
        X = np.array(X)
        # Save the original shape to reshape the flattened X later
        # back to its original shape
        if len(X.shape) > 1:
            self._orig_shape = X.shape[1:]
        X = self._flatten(X)
        self._scaler.fit(X, **kwargs)
        return self

    def transform(self, X, **kwargs):
        X = np.array(X)
        X = self._flatten(X)
        X = self._scaler.transform(X, **kwargs)
        X = self._reshape(X)
        return X

    def _flatten(self, X):
        # Reshape X to <= 2 dimensions
        if len(X.shape) > 2:
            n_dims = np.prod(self._orig_shape)
            X = X.reshape(-1, n_dims)
        return X

    def _reshape(self, X):
        # Reshape X back to it's original shape
        if len(X.shape) >= 2:
            X = X.reshape(-1, *self._orig_shape)
        return X