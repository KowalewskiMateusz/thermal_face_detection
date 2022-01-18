from collections import Counter
from typing import Collection
import cv2 
import numpy as np
from glob import glob
import pandas as pd  
from os.path import join
import random
from tqdm import tqdm
from tensorflow import keras 
from tensorflow.keras.applications.mobilenet_v2 import MobileNetV2
from tensorflow.keras.applications.resnet50 import ResNet50

from tensorflow.keras.models import Model
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout, InputLayer
from tensorflow.keras.models import Sequential
from tensorflow.keras import optimizers
from tensorflow import one_hot
from plantcv import plantcv as pcv

from sklearn.model_selection import train_test_split
from sklearn import metrics

import os

from tensorflow.keras.callbacks import Callback
import matplotlib.pyplot as plt
import numpy as np
import ast 
from scikitplot.metrics import plot_confusion_matrix


IMG_HEIGHT=48
IMG_WIDTH=48
INPUT_SHAPE = ((IMG_HEIGHT,IMG_WIDTH,3))



class PerformanceVisualizationCallback(Callback):
    def __init__(self, model, validation_data, image_dir):
        super().__init__()
        self.model = model
        self.validation_data = validation_data
        
        os.makedirs(image_dir, exist_ok=True)
        self.image_dir = image_dir

    def on_epoch_end(self, epoch, logs={}):
        y_pred = np.asarray(self.model.predict(self.validation_data[0]))
        y_true = self.validation_data[1]      
        
        y_pred_class = [1 if a_ > 0.5 else 0 for a_ in y_pred]

        # plot and save confusion matrix
        fig, ax = plt.subplots(figsize=(16,12))
        plot_confusion_matrix(y_true, y_pred_class, ax=ax)
        fig.savefig(os.path.join(self.image_dir, f'confusion_matrix_epoch_{epoch}'))


def train_ResNet50(X_train,y_train,X_test,y_test):
    restnet = ResNet50(include_top=False, weights='imagenet', input_shape=INPUT_SHAPE)

    for layer in restnet.layers:
        layer.trainable = False
    
    model = Sequential()
    model.add(restnet)
    model.add(Flatten())
    model.add(Dense(2048, activation='relu'))
    model.add(Dropout(0.2))
    model.add(Dense(1024, activation='relu'))
    model.add(Dropout(0.2))
    model.add(Dense(512, activation='relu'))
    model.add(Dropout(0.2))
    model.add(Dense(1, activation='sigmoid'))
    model.summary()
    model.compile(loss='binary_crossentropy',
                optimizer=optimizers.Adam(lr=2e-3),
                metrics=['accuracy'])


    X_train_np = np.empty([len(X_train),IMG_HEIGHT , IMG_WIDTH, 3])
    for i, img in enumerate(X_train):
        X_train_np[i] = img

    performance_cbk = PerformanceVisualizationCallback(
                      model=model,
                      validation_data=(X_test,y_test),
                      image_dir='performance_vizualizations')


    model.fit(X_train_np, np.array(y_train),
                                batch_size=256, 
                                epochs=10 ,
                                validation_data=(X_test,y_test),
                                callbacks=[performance_cbk])

    model.save('MobileNet_v1')
    return model 

def get_tufts():
    files = glob(r'TD_IR_A/**/*.jpg',recursive=True)
    labels = pd.read_csv(r'TD_IR_A/bounding-boxes.csv')

    X_train = list()
    y_train = list()

    for file in files:
        img = cv2.imread(file, cv2.IMREAD_GRAYSCALE)
        img = pcv.transform.rescale(gray_img=img)
        img = np.repeat(img[:, :, np.newaxis], 3, axis=2)

        if file.split('\\')[1] == 'n':
            img = cv2.resize(img, dsize=(IMG_HEIGHT, IMG_WIDTH), interpolation=cv2.INTER_CUBIC)


            X_train.append(img)
            y_train.append(0)
            continue

        participant = int(file.split('\\')[1])
        file_name = file.split('\\')[2] 
        file_row = labels.loc[(labels['Participant'] == participant) & (labels['File'] == file_name)].iloc[0]
        left = file_row['Left']
        top = file_row['Top']
        width = file_row['Width']
        height = file_row['Height']
        crop_img = img[top:top+height, left:left+width]
        crop_img = cv2.resize(crop_img, dsize=(IMG_HEIGHT, IMG_WIDTH), interpolation=cv2.INTER_CUBIC)


        X_train.append(crop_img)
        y_train.append(1)

    return X_train,y_train

def get_flir():
    files = pd.read_csv(r'FLIR_ADAS_1_3/bounding-boxes.csv')

    X_train,X_test = list(),list()
    y_train,y_test = list(),list()

    for _,file in files.iterrows():
        file_path = join('FLIR_ADAS_1_3',file['Set'],'thermal_8_bit',f'{file["File"][:-3]}jpeg')
        img = cv2.imread(file_path, cv2.IMREAD_GRAYSCALE)
        img = pcv.transform.rescale(gray_img=img)

        img = np.repeat(img[:, :, np.newaxis], 3, axis=2)

       
        left = file['Left']
        top = file['Top']
        width = file['Width']
        height = file['Height']
        crop_img = img[top:top+height, left:left+width]
        crop_img = cv2.resize(crop_img, dsize=(IMG_HEIGHT, IMG_WIDTH), interpolation=cv2.INTER_CUBIC)
        if file['Set'] == 'val':
            X_test.append(crop_img)
            y_test.append(1)
            continue

        X_train.append(crop_img)
        y_train.append(1)
    return X_train,y_train,X_test,y_test

def get_negatives():
    def random_crop(img):
        x = random.randint(0, img.shape[1] - IMG_WIDTH)
        y = random.randint(0, img.shape[0] - IMG_HEIGHT)
        img = img[y:y+IMG_HEIGHT, x:x+IMG_WIDTH]
        return img

    
    files_pos = pd.read_csv(r'FLIR_ADAS_1_3/bounding-boxes.csv')
    files_pos = files_pos['File'].values
    files_pos = [file[:-3] + 'jpeg' for file in files_pos]

    files = glob(r'FLIR_ADAS_1_3/train/**/*.jpeg',recursive=True)
    files = [file for file in files if file.split('\\')[-1] not in files_pos] 
    X_train,y_train = list(),list()
    for file in files[1000:]:
        img = cv2.imread(file, cv2.IMREAD_GRAYSCALE)
        img = pcv.transform.rescale(gray_img=img)

        img = np.repeat(img[:, :, np.newaxis], 3, axis=2)

        img = random_crop(img)
        X_train.append(img)
        y_train.append(0)
    return X_train,y_train

def get_dataset(positive):
    if positive:
        path = 'datasets/eti'
        label = 1
    else:
        path = 'datasets/eti_negs'
        label = 0
    files = glob(f'{path}/*.png')
    X_train,y_train = list(),list()
    for file in files:
        img = cv2.imread(file, cv2.IMREAD_GRAYSCALE)
        img = np.repeat(img[:, :, np.newaxis], 3, axis=2)
        img = cv2.resize(img, dsize=(IMG_HEIGHT, IMG_WIDTH), interpolation=cv2.INTER_CUBIC)

        X_train.append(img)
        y_train.append(label)
    return X_train ,y_train

def get_our_dataset():
    X_pos ,y_pos = get_dataset(positive = True)
    X_neg ,y_neg = get_dataset(positive = False)
    X_pos.extend(X_neg)
    y_pos.extend(y_neg)
    return X_pos,y_pos



# X_train,y_train,X_test,y_test = get_flir()
# X_train_neg,y_train_neg = get_negatives()
# X_train_tufts,y_train_tufts= get_tufts()

# X_train.extend(X_train_tufts)
# X_train.extend(X_train_neg)

# y_train.extend(y_train_tufts)
# y_train.extend(y_train_neg)



# with open('dataset.npy', 'wb') as f:
#     np.save(f, X_train)
#     np.save(f, y_train)
#     np.save(f,X_test)
#     np.save(f,y_test)


with open('saved_datasets/dataset.npy', 'rb') as f:
    X_train = np.load(f)
    y_train = np.load(f)
    X_test = np.load(f)
    y_test = np.load(f)

X_our,y_our = get_our_dataset()

# x_test_neg = list()
# y_test_neg = list()

# X_train_neg = list()
# y_train_neg = list()

# for i,j in zip(X_train,y_train):
#     if j == 0 and len(y_test_neg) < 1000:
#         x_test_neg.append(i)
#         y_test_neg.append(j)
#     if j == 0 and  len(y_test_neg) >=1000:
#         X_train_neg.append(i)
#         y_train_neg.append(j)
#     if len(y_train_neg) > 1000:
#         break


X_train_our, X_test_our, y_train_our, y_test_our = train_test_split(np.array(X_our),np.array(y_our), test_size=0.2)



# X_test = np.concatenate((X_test,X_our))
# y_test = np.concatenate((y_test,y_our))

# X_train = np.concatenate((X_train_our,X_train))
# y_train = np.concatenate((y_train_our,y_train))
 
from collections import Counter 

model = train_ResNet50(X_train_our, y_train_our,X_test_our, y_test_our)




# X_train, X_test, y_train, y_test = train_test_split(X_train, y_train, test_size=0.33)


# model = train_ResNet50(X_train,y_train,X_test,y_test)


