import os
import argparse
os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"]="0"  # specify which GPU(s) to be used
import keras
import cv2
import tensorflow as tf
import numpy as np
import numpy as np
from keras import backend as K
from keras.models import Sequential
from keras.layers.core import Dense, Dropout, Activation, Flatten
from keras.layers.convolutional import Convolution2D, MaxPooling2D
from keras.preprocessing.image import ImageDataGenerator
from sklearn.metrics import classification_report, confusion_matrix
from tensorflow.keras.preprocessing import image
from glob import glob                                                           
import cv2 
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score
from sklearn.metrics import f1_score
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score, balanced_accuracy_score
from sklearn.metrics import confusion_matrix,roc_auc_score
import timeit
import datetime
import shutil

def seperate_val_data():
    
    for i in range(10000,200001,10000):
        print(i)
        catgeories = ["pre_CHF","post_CHF"]
        
        src_dir = f"./boiling/results_{i}"

        file_names = os.listdir(src_dir)

        for file_name in file_names:

            if "pre_CHF" in file_name:
                sub_dir = f'{src_dir}/{catgeories[0]}'
            else:
                sub_dir = f'{src_dir}/{catgeories[1]}'
                
            if not os.path.exists(sub_dir):
                os.makedirs(sub_dir)
            shutil.copy(os.path.join(src_dir, file_name), sub_dir)
            print(f'copying {file_name} from {src_dir} to {sub_dir}')

            # Delete image after moving it to categroy file
            if file_name.endswith('.jpg'):
                os.remove((os.path.join(src_dir, file_name)))


def test_val_data():

    base_DS = "DS3"


    All_Models = []
    All_Metrics = []
    All_CMs = []


    model_name = f"./base_classifier_training/CNN DS3 Binary - Base Model - epoch 88.hdf5"
    print(model_name)


    for exp in range(10000,200001,10000):

        categories = ['pre_CHF','post_CHF']
        images = []
        y_true = []
        
        print(f"getting data for {exp} ......")
        
        for j,category in enumerate (categories): 
            im_files = glob(f'./boiling/results_{exp}/{category}/*.j*')

            for i,im_file in enumerate (im_files):
                
                if category == 'post_CHF':
                    y_true.append(0)
                elif category == 'pre_CHF':
                    y_true.append(1)
                

                img1 = image.load_img(im_file)
                img1 = image.img_to_array(img1)
                img1 = np.expand_dims(img1, axis=0)
                img1 /= 255.
                images.append(img1)
                
            
        begin_time = datetime.datetime.now()

        model = keras.models.load_model(model_name)
        
        imagesNP = np.vstack(images)
        y_pred = model.predict(imagesNP)
        # y_pred_prob = model.predict_proba(imagesNP)[:,1]
        y_pred_prob = y_pred[:,1]
        y_pred = np.argmax(y_pred,axis=1)

        balanced_acc = balanced_accuracy_score(y_true, y_pred)
        f1_weighted = f1_score(y_true, y_pred, average='weighted')
        precision_weighted = precision_score(y_true, y_pred, average='weighted')
        recall_weighted = recall_score(y_true, y_pred, average='weighted')
        ROC_AUC = roc_auc_score(y_true, y_pred_prob)
        
        CM = confusion_matrix(y_true, y_pred)
        
        testing_time = datetime.datetime.now() - begin_time

        metrics = [exp,balanced_acc,f1_weighted,precision_weighted,recall_weighted,ROC_AUC]
        metrics_names = ["GAN Model","Balanced Accuracy","F1_weighted","Precision_weighted","Recall_weighted","ROC_AUC"]

        print (CM)
        print ("      ")
        print(f"{balanced_acc = }")
        print ("      ")
        print(f"{ROC_AUC = }")
        All_Metrics.append(metrics)
        All_CMs.append(CM)

    #Send Metrics To Excel Sheet

    df = pd.DataFrame(All_Metrics,columns=metrics_names)

    df.to_excel (f'./Val_{base_DS}_Base Model_Metrics.xlsx', index = False, header=True)

    print(df.shape)



    frames = []

    for cm in All_CMs:
        df = pd.DataFrame(cm)
        frames.append(df)

    final = pd.concat(frames)
    frames = []

    for cm in All_CMs:
        df = pd.DataFrame(cm)
        frames.append(df)

    final = pd.concat(frames)

    final.to_excel (f'./Val_{base_DS}_Base Model_CMs2.xlsx', index = False, header=True)





