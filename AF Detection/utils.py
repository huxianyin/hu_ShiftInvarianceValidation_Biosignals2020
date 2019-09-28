import os
import keras
import math
import pickle
import numpy as np
import pandas as pd
import seaborn as sns
import tensorflow as tf
from tensorflow.contrib import rnn
from keras.utils import to_categorical
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix
from keras.models import Model
from keras.layers import Input, Conv1D, Dense, Flatten, Dropout,MaxPooling1D, AveragePooling1D,Activation, BatchNormalization
from keras.callbacks import EarlyStopping, ModelCheckpoint
from keras.utils import plot_model
from keras import backend as K
from keras.layers import Layer

dataset_root = "./dataset_RRI/"
segment_len = 100
fs = 250
dataset_files = [os.path.join(dataset_root,i) for i in os.listdir(dataset_root) if "cross" in i]

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

    if "accuracy" in set(history.history.keys()):
        acc = history.history["accuracy"]
    else:
        acc = history.history["acc"]
    plt.plot(acc)
    plt.xlabel("Epochs")
    plt.ylabel("accuracy")
    plt.savefig(acc_save_path)
    plt.clf()
    plt.close()

def save_model(idx,model,model_save_dir="./models"):
    if not os.path.exists(model_save_dir):
        os.mkdir(model_save_dir)
    model_save_path = os.path.join(model_save_dir,str(idx)+"-model.h5")
    model.save(model_save_path)

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
            self.a = np.array([1., 6., 15., 20., 15., 6., 1.])
        else:
            self.a = None
        
        super(BlurPooling1D, self).__init__(**kwargs)
        
    def compute_output_shape(self, input_shape):
        if self.filt_size > 0:
            win_len = math.ceil(input_shape[1] / self.stride)
        else:
            win_len = input_shape[1] // self.stride
        channels = input_shape[2]
        return (input_shape[0], win_len, channels)
        
    def call(self, x):
        if self.a is not None:
            k = self.a
            k = k / np.sum(k)
            k = np.tile(k[:,None,None], (1,1,1) )                
            k = K.constant (k, dtype=K.floatx() )
            x = MaxPooling1D(pool_size=2,strides=1,padding="same")(x)

            x = K.temporal_padding(x, padding=self.padding)

            conved_channels=[]
            for c in tf.split(x,num_or_size_splits=x.get_shape()[-1],axis=2):
                conved = K.conv1d(c,k,strides=self.stride,padding='valid')
                conved_channels.append(conved)
            x = K.concatenate(conved_channels,axis=2)

        elif self.filt_size == 0:
            x = MaxPooling1D(pool_size=2,strides=2,padding="valid")(x)
        else:
            x = AveragePooling1D(pool_size=2,strides=2,padding="valid")(x)
        return x

    def get_config(self):
        config = {'filt_size': self.filt_size,'stride':self.stride}
        base_config = super(BlurPooling1D, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))

def drop_repeat(df,axis="RMSE",repeat=5):
    if "accuracy" in axis:
        func = max
    else:
        func = min
    num = int(len(df)/repeat)
    result_df = []
    for i in range(num):
        part_df = df[i*repeat:(i+1)*repeat]
        max_row = part_df[part_df[axis]==func(part_df[axis])]
        index = np.array(max_row["idx"])[0]
        result_df.append(max_row[max_row.idx==index])
    tmp = pd.concat(result_df,axis=0)
    return tmp

def cal_cross_validated(df,cross=5):
    result_df = []
    num = int(len(df)/cross)
    func = np.mean
    for i in range(num):
        part_df = df[i*cross:(i+1)*cross]
        mean_loss = np.mean(part_df["loss"])
        mean_acc = np.mean(part_df["accuracy"])
        mean_consis_b = np.mean(part_df["consis_b"])
        mean_consis_ub = np.mean(part_df["consis_ub"])
        mean_consis_n = np.mean(part_df["consis_n"])
        mean_consis_a = np.mean(part_df["consis_a"])
        mean_consis_a_s = np.mean(part_df["consis_a_s"])

        pool_factor = list(part_df["pool_factor"])[0]
        pool_type = list(part_df["pool_type"])[0]
        marker = list(part_df["marker"])[0]

        row = pd.Series(data=[pool_factor,pool_type,marker,mean_loss,mean_acc,mean_consis_b,mean_consis_ub,mean_consis_n,mean_consis_a,mean_consis_a_s],
                       index=["pool_factor","pool_type","marker","loss","accuracy","consis_b","consis_ub","consis_n","consis_a","consis_a_s"])
        
        result_df.append(row)
    tmp = pd.DataFrame(result_df)
    return tmp


def generate_marker(pooling):
    if pooling=="avg":
        return "$A$"
    elif pooling=="max":
        return "$M$"
    elif pooling=="maxblur-1":
        return "o"
    elif pooling=="maxblur-2":
        return "|"
    elif pooling=="maxblur-3":
        return "^"
    elif pooling=="maxblur-4":
        return "D"
    elif pooling=="maxblur-5":
        return "p"
    elif pooling=="maxblur-6":
        return "h"
    elif pooling=="maxblur-7":
        return "*"


def write_header(header,file_path):
    with open(file_path,"w") as f:
        string = '_,'
        for c in header:
            string = string+c+","
        f.write(string[:-1]+'\n')


def excute_command(script_path,args):
   command = "python " + script_path
   arg_str = " "
   for arg in args:
       if type(arg) == list:
           arg_str += list2str(arg,"-")
       else:
           arg_str += str(arg)
       arg_str += " "
   excute_command = command+arg_str
   os.system(command+arg_str)


   