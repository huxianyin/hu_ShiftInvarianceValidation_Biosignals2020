#!/usr/bin/env python
# coding: utf-8

# In[1]:


import os
import sys
import tensorflow as tf
from tensorflow.contrib import rnn
import pickle
import numpy as np
from keras.utils import to_categorical
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix
import pandas as pd
import keras
from keras.models import Model
from keras.layers import Input, Conv1D, Dense, Flatten, Dropout,MaxPooling1D, Activation, BatchNormalization
from keras.callbacks import EarlyStopping, ModelCheckpoint
from keras.utils import plot_model
from keras import backend as K
from keras.layers import Layer


# In[2]:


class BlurPooling1D(Layer):
    def __init__(self, filt_size=5, stride=2,**kwargs):
        self.stride = stride
        self.filt_size = filt_size
        self.padding = (int(1.*(filt_size-1)/2), int(np.ceil(1.*(filt_size-1)/2)))
        if(self.filt_size==1):
            self.a = np.array([1.,])
        elif(self.filt_size==2):
            self.a = np.array([1., 1.])
        elif(self.filt_size==3):
            self.a = np.array([1., 2., 1.])
        elif(self.filt_size==4):    
            self.a = np.array([1., 3., 3., 1.])
        elif(self.filt_size==5):    
            self.a = np.array([1., 4., 6., 4., 1.])
        elif(self.filt_size==6):    
            self.a = np.array([1., 5., 10., 10., 5., 1.])
        elif(self.filt_size==7):    
            self.a = np.a
            rray([1., 6., 15., 20., 15., 6., 1.])
        super(BlurPooling1D, self).__init__(**kwargs)
        
    def compute_output_shape(self, input_shape):
        win_len = input_shape[1] // self.stride
        channels = input_shape[2]
        return (input_shape[0], win_len, channels)
        
    def call(self, x):
        k = self.a
        k = k / np.sum(k)
        k = np.tile(k[:,None,None], (1,x.shape[-1],1) )                
        k = K.constant (k, dtype=K.floatx() )

        x = K.temporal_padding(x, padding=self.padding)
        x = K.conv1d(x,k,strides=self.stride,padding='valid')
        
        #x = MaxPooling1D(pool_size=2,strides=2,padding="same")(x)
        return x


# In[3]:


def plot_cmx(true,pred):
    cm = confusion_matrix(np.argmax(true,axis=1), np.argmax(pred,axis=1))
    classes = ["N","AF"]
    df_cm = pd.DataFrame(cm, index = classes,
                  columns = classes)
    plt.figure(figsize = (8,6))
    sns.set(font_scale=1.4)#for label size
    sns.heatmap(df_cm, annot=True,cmap="Blues",fmt="d",annot_kws={"size": 16})# font size
    plt.xlabel("Predict Label")
    plt.ylabel("True Label")
    plt.ylim(len(cm),0)
    plt.tight_layout()


# In[4]:


def load_dataset(cross_idx):
    file = dataset_files[cross_idx]
    with open(file,"rb") as f:
        dataset = pickle.load(f)
    test_keys = dataset["test_key"]
    X_train = dataset["X_train"]
    y_train = dataset["y_trian"]
    X_test = dataset["X_test"]
    y_test = dataset["y_test"]
    print("test keys:",test_keys)
    print(X_train.shape)
    print(y_train.shape)
    print(X_test.shape)
    print(y_test.shape)
    return X_train,y_train,X_test,y_test


# In[5]:


def ResNet_model_MaxBlurPool(blur_size):
    # Add CNN layers left branch (higher frequencies)
    # Parameters from paper
    WINDOW_SIZE = segment_len
    INPUT_FEAT = 1
    OUTPUT_CLASS = 2    # output classes

    k = 1    # increment every 4th residual block
    p = False # pool toggle every other residual block (end with 2^8)
    convfilt = 32
    convstr = 1
    ksize = 16
    poolsize = 2
    poolstr  = 2
    drop = 0.5
    
    # Modelling with Functional API
    #input1 = Input(shape=(None,1), name='input')
    input1 = Input(shape=(WINDOW_SIZE,INPUT_FEAT), name='input')
    
    ## First convolutional block (conv,BN, relu)
    x = Conv1D(filters=convfilt,
               kernel_size=ksize,
               padding='same',
               strides=convstr,
               kernel_initializer='he_normal')(input1)                
    x = BatchNormalization()(x)        
    x = Activation('relu')(x)  
    
    ## Second convolutional block (conv, BN, relu, dropout, conv) with residual net
    # Left branch (convolutions)
    x1 =  Conv1D(filters=convfilt,
               kernel_size=ksize,
               padding='same',
               strides=convstr,
               kernel_initializer='he_normal')(x)      
    x1 = BatchNormalization()(x1)    
    x1 = Activation('relu')(x1)
    x1 = Dropout(drop)(x1)
    x1 =  Conv1D(filters=convfilt,
               kernel_size=ksize,
               padding='same',
               strides=convstr,
               kernel_initializer='he_normal')(x1)
    #x1 = MaxPooling1D(pool_size=poolsize,strides=poolstr)(x1)
    #BlurMaxPooling
    x1 = BlurPooling1D(filt_size=blur_size)(x1)
    
    # Right branch, shortcut branch pooling
    #x2 = MaxPooling1D(pool_size=poolsize,strides=poolstr)(x)
    x2 = BlurPooling1D(filt_size=blur_size)(x)
    
    # Merge both branches
    x = keras.layers.add([x1, x2])
    del x1,x2
    
    ## Main loop
    p = not p 
    for l in range(2):
        if (l%4 == 0) and (l>0): # increment k on every fourth residual block
            k += 1
             # increase depth by 1x1 Convolution case dimension shall change
            xshort = Conv1D(filters=convfilt*k,kernel_size=1)(x)
        else:
            xshort = x        
        # Left branch (convolutions)
        # notice the ordering of the operations has changed        
        x1 = BatchNormalization()(x)
        x1 = Activation('relu')(x1)
        x1 = Dropout(drop)(x1)
        x1 =  Conv1D(filters=convfilt*k,
               kernel_size=ksize,
               padding='same',
               strides=convstr,
               kernel_initializer='he_normal')(x1)        
        x1 = BatchNormalization()(x1)
        x1 = Activation('relu')(x1)
        x1 = Dropout(drop)(x1)
        x1 =  Conv1D(filters=convfilt*k,
               kernel_size=ksize,
               padding='same',
               strides=convstr,
               kernel_initializer='he_normal')(x1)        
        if p:
            #x1 = MaxPooling1D(pool_size=poolsize,strides=poolstr)(x1)  
            x1 = BlurPooling1D(filt_size=blur_size)(x1)    
        # Right branch: shortcut connection
        if p:
            #x2 = MaxPooling1D(pool_size=poolsize,strides=poolstr)(xshort)
            x2 = BlurPooling1D(filt_size=blur_size)(xshort)
        else:
            x2 = xshort  # pool or identity            
        # Merging branches
        x = keras.layers.add([x1, x2])
        # change parameters
        p = not p # toggle pooling

    
    # Final bit    
    x = BatchNormalization()(x)
    x = Activation('relu')(x) 
    x = Flatten()(x)
    x = Dense(1000)(x)
    #x = Dense(1000)(x)
    out = Dense(OUTPUT_CLASS, activation='softmax')(x)
    model = Model(inputs=input1, outputs=out)
    model.compile(optimizer='adam',
                  loss='categorical_crossentropy',
                  metrics=['accuracy'])
    #model.summary()
    #sequential_model_to_ascii_printout(model)
    plot_model(model, to_file='model.png')
    return model


# In[6]:


def save_graphs(idx,history,prediction,y_test,fig_save_dir="./figs"):
    if not os.path.exists(fig_save_dir):
        os.mkdir(fig_save_dir)
    cmx_save_path = os.path.join(fig_save_dir,"cmx-"+str(idx)+".png")
    acc_save_path = os.path.join(fig_save_dir,"acc-"+str(idx)+".png")
    loss_save_path = os.path.join(fig_save_dir,"loss-"+str(idx)+".png")
    
    plot_cmx(pred=prediction,true=y_test)
    plt.savefig(cmx_save_path)
    plt.clf()

    plt.plot(history.history["loss"])
    plt.xlabel("Epochs")
    plt.ylabel("loss")
    plt.savefig(loss_save_path)
    plt.clf()

    plt.plot(history.history["accuracy"])
    plt.xlabel("Epochs")
    plt.ylabel("accuracy")
    plt.savefig(acc_save_path)
    plt.clf()
    plt.close()


# In[7]:


def save_model(idx,model,model_save_dir="./models"):
    if not os.path.exists(model_save_dir):
        os.mkdir(model_save_dir)
    model_save_path = os.path.join(model_save_dir,str(idx)+"-model.h5")
    model.save(model_save_path)


# In[8]:


dataset_root = "./dataset_RRI/"
segment_len = 100
fs = 250
dataset_files = [os.path.join(dataset_root,i) for i in os.listdir(dataset_root) if "cross" in i]


# In[9]:


epoches = 50
repeat = 10
cross = 5
records = []
use_blurpool = True


# In[10]:


def main(blur_size):
    save_root = "./results_maxblur-" + str(blur_size)
    if not os.path.exists(save_root):
        os.mkdir(save_root)

    model_save_dir = os.path.join(save_root,"models")
    fig_save_dir = os.path.join(save_root,"figs")
    result_save_path = os.path.join(save_root,"result.csv")


    for cross_idx in range(cross):
        X_train,y_train,X_test,y_test = load_dataset(cross_idx)
        for repeat_idx in range(repeat):
            idx = (cross_idx)*10 + repeat_idx + 1
            if use_blurpool:
                model = ResNet_model_MaxBlurPool(blur_size)
            else:
                model = ResNet_model()
            history = model.fit(X_train,y_train,epochs=epoches,verbose=1)
            prediction = model.predict(X_test)
            loss,acc = model.evaluate(X_test,y_test)
            print("test acc={0:.2f} , test loss={1:.2f}".format(acc,loss))
            save_model(idx,model,model_save_dir)
            save_graphs(idx,history,prediction,y_test,fig_save_dir)
            records.append({"idx":idx,"cross_idx":cross_idx,"repeat_idx":repeat_idx,"loss":loss,"accuracy":acc})

    result_df = pd.DataFrame.from_records(records)
    print("5-cross-validated accuaracy =",np.mean(result_df.accuracy))
    print("5-cross-validated loss =",np.mean(result_df.loss))
    result_df.to_csv(result_save_path,header=True,index=False)


# In[ ]:


if __name__ == "__main__":
    blur_size = int(sys.argv[1])
    main(blur_size)
#    blur_size = 1
#    main(blur_size)
    


# In[ ]:





# In[ ]:




