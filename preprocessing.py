import config
import os
import cv2
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt

AUTOTUNE = tf.data.experimental.AUTOTUNE
def create_training_data():
    try:
        X_train = np.load('np_files/trainingSet2d.npy')
    except IOError as e:
        print('saved training setnot found, generating new one')
        image_count = len(list(config.DATADIR.glob('**/*.jpg')))
        global DATASET_SIZE
        DATASET_SIZE = image_count
        images= list(config.DATADIR.glob('**/*.jpg'))
        cwd = os.getcwd()
        paths = []
        training_data= []
        # print(os.getcwd())
        for img in images:  # iterate over each image per dogs and cats
            # print(img)
            folName = img.parts[-2]
            # print(folName)
            index = int(folName[-3:])-1
            imgName = img.parts[-1]
            path = os.path.join(folName,imgName)
            img_array = cv2.imread(os.path.join(cwd, img) ,cv2.IMREAD_GRAYSCALE)  # convert to array
            img_array = cv2.resize(img_array,(config.IMG_HEIGHT,config.IMG_WIDTH))
            training_data.append([img_array])  # add this to our training_data
            paths.append(path)
        if(config.SUBTRACT_MEAN_IMAGE):
            X_train = np.array(training_data).reshape(-1,config.IMG_HEIGHT, config.IMG_WIDTH)

            # print('mean and stdev before norm')

            # print('mean and stdev before norm')
            meanImage = X_train.mean(axis=0)
            plt.imshow(meanImage, cmap='gray')
            plt.axis("off")
            plt.savefig('meanImage.jpg')
            

            plt.imshow(X_train[0], cmap='gray')
            plt.axis("off")

            plt.savefig('before_sub.jpg')

            plt.imshow(X_train[0]-meanImage, cmap='gray')
            plt.axis("off")

            plt.savefig('after_sub.jpg')

            plt.close('all')

            print('mean  before norm')
            print(X_train.mean())
            print(X_train.std())

            print(X_train.min())
            print(X_train.max())

            X_train = (X_train - meanImage)

            print('mean and stdev after sub')
            print(X_train.mean())
            print(X_train.std())

            print(X_train.min())
            print(X_train.max())

            X_train = (X_train-X_train.min())/(X_train.max()-X_train.min())

            print('mean and stdev after norm')
            print(X_train.mean())
            print(X_train.std())
            print(X_train.min())
            print(X_train.max())

            np.save('np_files/meanImage.npy',np.array(meanImage))

        else:
            X_train = np.array(training_data).reshape(-1,config.IMG_HEIGHT, config.IMG_HEIGHT,1)/255.0
        # print(len(training_data))
        
        # print(X.shape)
        # X_train = X.reshape(-1,config.IMG_HEIGHT, config.IMG_HEIGHT,1)/255
        X_train = np.array(X_train).reshape(-1,config.IMG_HEIGHT, config.IMG_HEIGHT,1)
        print(X_train.shape)
        pathToFolder='np_files/'
        np.save(pathToFolder+'trainingSet2d.npy',X_train) 
    
    return X_train

def get_clips_by_stride(stride, frames_list, sequence_size):
    """ For data augmenting purposes.
    Parameters
    ----------
    stride : int
        The distance between two consecutive frames
    frames_list : list
        A list of sorted frames of shape 256 X 256
    sequence_size: int
        The size of the lstm sequence
    Returns
    -------
    list
        A list of clips , 10 frames each
    """
    clips = []
    sz = len(frames_list)
    clip = np.zeros(shape=(sequence_size, config.IMG_HEIGHT, config.IMG_WIDTH))
    # print(clip[0, :, :, 0].shape)
    cnt = 0
    for start in range(0, stride):
        for i in range(start, sz, stride):
            test =np.array(frames_list[i])
            # print(test.shape)
            # print(clip[cnt, :, :, 0].shape)
            clip[cnt, :, :] = frames_list[i]
            cnt = cnt + 1
            if cnt == sequence_size:
                clips.append(clip)
                cnt = 0
    return clips

def create_training_data3d(strided=False):
    
    image_count = len(list(config.DATADIR.glob('**/*.jpg')))
    global DATASET_SIZE
    DATASET_SIZE = image_count
    images= list(config.DATADIR.glob('**/*.jpg'))
    cwd = os.getcwd()
    paths = []
    clips = []
    training_data= []
    try:
        clips = np.load('np_files/clips.npy')
    except IOError as e:
        print('saved training setnot found, generating new one')
    # print(os.getcwd())
        for img in images:  # iterate over each image per dogs and cats
            # print(img)
            folName = img.parts[-2]
            # print(folName)
            index = int(folName[-3:])-1
            imgName = img.parts[-1]
            path = os.path.join(folName,imgName)
            img_array = cv2.imread(os.path.join(cwd, img) ,cv2.IMREAD_GRAYSCALE)  # convert to array
            img_array = cv2.resize(img_array,(config.IMG_HEIGHT,config.IMG_WIDTH))
            training_data.append([img_array])  # add this to our training_data
            # paths.append(path)

        # print('x shape')
        # print(X_train.shape)
        if(config.SUBTRACT_MEAN_IMAGE):
            X_train = np.array(training_data).reshape(-1,config.IMG_HEIGHT, config.IMG_HEIGHT)

            # print('mean and stdev before norm')

            # print('mean and stdev before norm')
            meanImage = X_train.mean(axis=0)
            plt.imshow(meanImage, cmap='gray')
            plt.savefig('meanImage.jpg')
            

            plt.imshow(X_train[0], cmap='gray')
            plt.savefig('before_sub.jpg')

            plt.imshow(X_train[0]-meanImage, cmap='gray')
            plt.savefig('after_sub.jpg')

            plt.close('all')

            print('mean  before norm')
            print(X_train.mean())
            print(X_train.std())

            print(X_train.min())
            print(X_train.max())

            X_train = (X_train - meanImage)

            print('mean and stdev after sub')
            print(X_train.mean())
            print(X_train.std())

            print(X_train.min())
            print(X_train.max())

            X_train = (X_train-X_train.min())/(X_train.max()-X_train.min())

            print('mean and stdev after norm')
            print(X_train.mean())
            print(X_train.std())
            print(X_train.min())
            print(X_train.max())

            np.save('np_files/meanImage.npy',np.array(meanImage))

        else:
            X_train = np.array(training_data).reshape(-1,config.IMG_HEIGHT, config.IMG_HEIGHT)/255.0

        if(strided):
            for stride in range(1,3):
                clips.extend(get_clips_by_stride(stride, X_train, config.NUM_CHANNELS))
            clips = np.array(clips)
        else:
            clips = np.array(combineIntoChannels(X_train, config.NUM_CHANNELS))
        pathToFolder='np_files/'
        np.save(pathToFolder+'clips.npy',clips) 
        

    print('clips shape')
    print(clips.shape)
    # print(len(training_data))
    
 
    # X_train = X.reshape(-1,config.IMG_HEIGHT, config.IMG_HEIGHT,1)/255
    # print(X_train.shape)
    
    return clips


def combineIntoChannels(ds, channels):
    clips = np.zeros((int(len(ds)/channels), channels, config.IMG_HEIGHT, config.IMG_WIDTH))
    idx = 0
    ch = 0

    for i in range(0, len(ds)):
        clips[idx,ch,:,:] = ds[i]
        ch = ch + 1
        if  ch>= channels:
            idx = idx + 1
            ch = 0
    
    return clips
    


def prepare_for_training(ds, cache=False, shuffle_buffer_size=1000):
  # This is a small dataset, only load it once, and keep it in memory.
  # use `.cache(filename)` to cache preprocessing work for datasets that don't
  # fit in memory.
  if cache:
    if isinstance(cache, str):
      ds = ds.cache(cache)
    else:
      ds = ds.cache()

  ds = ds.shuffle(buffer_size=shuffle_buffer_size)

  # Repeat forever
  # ds = ds.repeat()

  ds = ds.batch(config.BATCH_SIZE)

  # `prefetch` lets the dataset fetch batches in the background while the model
  # is training.
  # ds = ds.prefetch(buffer_size=AUTOTUNE)

  return ds

def loadScene3d(num):
    folderNum= str(num).zfill(3)
    images= list(config.TESTPATH.glob('Test'+folderNum+'/*.jpg'))
    # print('before sort')
    # print(images)
    # images = images.sort()
    # print('after sort')
    # print(images)
    cwd = os.getcwd()
    scene = []
    for img in images:
        folName = img.parts[-2]
        # print(folName)
        index = int(folName[-3:])-1
        imgName = img.parts[-1]
        path = os.path.join(folName,imgName)
        # print(path)
        imgNum = int(imgName[:-4])
        # print(os.path.join(cwd, img))
        img_array = cv2.imread(os.path.join(cwd, img) ,cv2.IMREAD_GRAYSCALE)
        img_array = cv2.resize(img_array,(config.IMG_HEIGHT,config.IMG_WIDTH))
        scene.append(img_array)
     # convert to array
        #     training_data.append([img_array])  # add this to our training_data
    #     if(checkBoundries(index, imgNum)):
    #         tAnomaly.append(img_array)
    #         tAnomalyPath.append(path)
    #     else:
    #         tNormal.append(img_array)
    #         tNormalPath.append(path)

    # print(images)
    if(config.SUBTRACT_MEAN_IMAGE):
        scene = np.array(scene).reshape(-1,config.IMG_HEIGHT, config.IMG_WIDTH)
        meanImage = np.load('np_files/meanImage.npy')
        scene = scene - meanImage
        scene = (scene-scene.min())/(scene.max()-scene.min())

    else:
        scene = np.array(scene).reshape(-1,config.IMG_HEIGHT, config.IMG_WIDTH)/255
    scene = combineIntoChannels(scene, config.NUM_CHANNELS)
    # print(scene.shape)
    return scene, config.boundries[num-1]
  
def loadScene(num):
    folderNum= str(num).zfill(3)
    images= list(config.TESTPATH.glob('Test'+folderNum+'/*.jpg'))
    # print('before sort')
    # print(images)
    # images = images.sort()
    # print('after sort')
    # print(images)
    cwd = os.getcwd()
    scene = []
    for img in images:
        folName = img.parts[-2]
        # print(folName)
        index = int(folName[-3:])-1
        imgName = img.parts[-1]
        path = os.path.join(folName,imgName)
        # print(path)
        imgNum = int(imgName[:-4])
        # print(os.path.join(cwd, img))
        img_array = cv2.imread(os.path.join(cwd, img) ,cv2.IMREAD_GRAYSCALE)
        img_array = cv2.resize(img_array,(config.IMG_HEIGHT,config.IMG_WIDTH))
        scene.append(img_array)
     # convert to array
        #     training_data.append([img_array])  # add this to our training_data
    #     if(checkBoundries(index, imgNum)):
    #         tAnomaly.append(img_array)
    #         tAnomalyPath.append(path)
    #     else:
    #         tNormal.append(img_array)
    #         tNormalPath.append(path)

    # print(images)

    if(config.SUBTRACT_MEAN_IMAGE):
        scene = np.array(scene).reshape(-1,config.IMG_HEIGHT, config.IMG_WIDTH)
        meanImage = np.load('np_files/meanImage.npy')
        scene = scene - meanImage
        scene = (scene-scene.min())/(scene.max()-scene.min())
        scene = scene.reshape(-1,config.IMG_HEIGHT, config.IMG_WIDTH,1)
    else:
        scene = np.array(scene).reshape(-1,config.IMG_HEIGHT, config.IMG_WIDTH)/255
    # print(scene.shape)
    return scene, config.boundries[num-1]

def checkBoundries(index, name):
    if((len(config.boundries[index])) < 3):
        lower = config.boundries[index][0]
        upper = config.boundries[index][1]
        return (lower <= name and name <= upper)
    else:
        lower = config.boundries[index][0]
        upper = config.boundries[index][1]
        first = (lower <= name and name <= upper)
        lower = config.boundries[index][2]
        upper = config.boundries[index][3]
        sec = (lower <= name and name <= upper)
        return first or sec


def loadTest():
    image_count = len(list(config.TESTPATH.glob('**/*.jpg')))
    images= list(config.TESTPATH.glob('**/*.jpg'))
    # print(len(images))
    cwd = os.getcwd()
    tAnomaly = []
    tNormal = []
    tAnomalyPath = []
    tNormalPath = []
    for img in images:
        folName = img.parts[-2]
        # print(folName)
        index = int(folName[-3:])-1
        imgName = img.parts[-1]
        path = os.path.join(folName,imgName)
        # print(path)
        imgNum = int(imgName[:-4])
        img_array = cv2.imread(os.path.join(cwd, img) ,cv2.IMREAD_GRAYSCALE)  # convert to array
        img_array = cv2.resize(img_array,(config.IMG_HEIGHT,config.IMG_WIDTH))
        #     training_data.append([img_array])  # add this to our training_data
        if(checkBoundries(index, imgNum)):
            tAnomaly.append(img_array)
            tAnomalyPath.append(path)
        else:
            tNormal.append(img_array)
            tNormalPath.append(path)


    print('tanomaly len' + str(len(tAnomaly)))
    print('tnormal len' + str(len(tNormal)))

    tAnomaly = np.array(tAnomaly).reshape(-1,config.IMG_HEIGHT, config.IMG_WIDTH, 1)
    # print(tAnomaly.shape)
    tAnomaly = tAnomaly.reshape(-1,config.IMG_HEIGHT, config.IMG_WIDTH,1)/255
    # print(tAnomaly.shape)

    tNormal = np.array(tNormal).reshape(-1,config.IMG_HEIGHT, config.IMG_WIDTH, 1)/255
    # tNormal = tNormal.reshape(-1,IMG_HEIGHT, IMG_WIDTH,1)/255

    return tAnomaly, tNormal, tAnomalyPath, tNormalPath