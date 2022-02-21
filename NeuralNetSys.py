# _*_ coding:gbk _*_
import tensorflow as tf

from matplotlib import pyplot as plt
import numpy as np
from tensorflow.keras.layers import TextVectorization
import string
import re
from tensorflow.keras import layers
import os
import PIL
import PIL.Image
import random
from tensorflow import keras
import Read_file
import pathlib
from keras.callbacks import ReduceLROnPlateau
AUTOTUNE = tf.data.experimental.AUTOTUNE

class KERAS_NeuralNet:
    def __init__(self,dataset,labels,init_lr=0.001,Data_type="NUM",Batch_size=32,epochs=10,div_prop=0.1,Dense_num=[2,128,10,2],opti='adam',plot_al=True):
        '''
        Dense_num:not available for Data_type'TEXT'
        Data_type:NUM,IMAGE,TEXT(IMAGE/TEXT are unavailable in this program)
        opti:optimizer of molel,available choices:adam/sgd/rms/adag/adad/adamax/nadam
        '''
        self.Data_type=Data_type#Just a useless parameter,I created it just for text and image neural networks
        self.dataset=dataset
        self.labels=labels
        self.epochs=epochs
        self.Dense_num=Dense_num
        self.plot_All=plot_al
        self.batch_size=Batch_size
        self.div=div_prop
        self.init_lr=init_lr
        self.opti=opti
    def KERASNeuralEngine(self,max_features=0,embedding_dim=128,sequence_length = 500):
        '''
        max_features,embedding_dim,sequence_length are only available for Data_type 'TEXT' 
        '''
        if self.Data_type=='NUM':
            #import data
            x_data=np.array(self.dataset)
            y_data=np.array(self.labels)
            # To improve the accurancy we randomlize the data
            # seed
            np.random.seed(116)  # using same seed to make sure the labels and features are corresponded
            np.random.shuffle(x_data)
            np.random.seed(116)
            np.random.shuffle(y_data)
            tf.random.set_seed(116)

            # divide data into test group and train group
            train_features = x_data[0:-int(len(self.dataset)*self.div)]
            train_labels = y_data[0:-int(len(self.dataset)*self.div)]
            test_features = x_data[-int(len(self.dataset)*self.div):]
            test_labels = y_data[-int(len(self.dataset)*self.div):]
            
            #normalization
            train_features=(train_features-train_features.min(axis=0))/(train_features.max(axis=0)-train_features.min(axis=0))
            test_features=(test_features-test_features.min(axis=0))/(test_features.max(axis=0)-test_features.min(axis=0))
            #Neural network
            Dense_layers=[]
            for i in range(len(self.Dense_num)):
                if i!=0 and i!=len(self.Dense_num):
                    Dense_layers.append(keras.layers.Dense(self.Dense_num[i],activation='relu'))
                else:
                    Dense_layers.append(keras.layers.Dense(self.Dense_num[i]))
            model=keras.Sequential(Dense_layers)
            #key parameters
            if self.opti=='adam':
                self.opti=tf.keras.optimizers.Adam(learning_rate=self.init_lr)
            if self.opti=='sgd':
                self.opti=tf.keras.optimizers.SGD(learning_rate=self.init_lr)
            if self.opti=='rms':
                self.opti=tf.keras.optimizers.RMSprop(learning_rate=self.init_lr)
            if self.opti=='adag':
                self.opti=tf.keras.optimizers.Adagrad(learning_rate=self.init_lr)
            if self.opti=='adad':
                self.opti=tf.keras.optimizers.Adadelta(learning_rate=self.init_lr)
            if self.opti=='adamax':
                self.opti=tf.keras.optimizers.Adamax(learning_rate=self.init_lr)
            if self.opti=='nadam':
                self.opti=tf.keras.optimizers.Nadam(learning_rate=self.init_lr)
            model.compile(optimizer=self.opti,
              loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
              metrics=['accuracy'])
            #train
            #Dynamical learn rate
            reduce_lr=ReduceLROnPlateau(monitor='loss',patience=int(self.epochs/5),mode='auto')
            #training
            train_history=model.fit(train_features, train_labels, epochs=self.epochs,verbose=2,callbacks=[reduce_lr])
            #train over
            #calculate total accuracy
            test_loss, test_acc = model.evaluate(test_features,  test_labels, verbose=2)
            print('\nTotal Test accuracy:', test_acc)
            if self.plot_All==True:
                #plot accuracy
                #print(train_history.history.keys())
                plt.plot(train_history.history['accuracy'])
                #plt.plot(train_history.history['val_acc'])
                plt.title('Model accuracy')
                plt.ylabel('Accuracy')
                plt.xlabel('Epoch')
                plt.legend(['Train', 'Test'], loc='upper left')
                plt.show()
                #plot loss
                plt.plot(train_history.history['loss'])
                #plt.plot(train_history.history['val_loss'])
                plt.title('Model loss')
                plt.ylabel('Loss')
                plt.xlabel('Epoch')
                plt.legend(['Train', 'Test'], loc='upper left')
                plt.show()
            #os.system("pause")
            return test_acc




#o1=KERAS_NeuralNet()
#o1.KERASNeuralEngine()