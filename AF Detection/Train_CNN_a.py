#!/usr/bin/env python
# coding: utf-8


import os
import sys
import pandas as pd
from utils import *
from keras.models import load_model
from keras.utils import CustomObjectScope



dataset_root = "./dataset_RRI_augmented/"
dataset_files = [os.path.join(dataset_root,i) for i in os.listdir(dataset_root) if "cross" in i]


def CNN_MaxBlurPool(blur_size,pool_factor):
    assert(pool_factor >= 1)
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
    
    ## First convolutional block (conv,BN, relu,pool)
    x = Conv1D(filters=convfilt,kernel_size=ksize,padding='same',strides=convstr,kernel_initializer='he_normal')(input1)                
    x = BatchNormalization()(x)        
    x = Activation('relu')(x)  
    x = BlurPooling1D(filt_size=blur_size)(x)

    for layer in range(pool_factor-1):
        x = Conv1D(filters=convfilt,kernel_size=ksize,padding='same',strides=convstr,kernel_initializer='he_normal')(x)                
        x = BatchNormalization()(x)        
        x = Activation('relu')(x)  
        x = BlurPooling1D(filt_size=blur_size)(x)
    
    # Final bit    
    x = BatchNormalization()(x)
    x = Activation('relu')(x) 
    x = Flatten()(x)
    x = Dense(1000)(x)
    out = Dense(OUTPUT_CLASS, activation='softmax')(x)
    model = Model(inputs=input1, outputs=out)
    model.compile(optimizer='adam',
                  loss='categorical_crossentropy',
                  metrics=['accuracy'])
    
    #model.summary()
    #sequential_model_to_ascii_printout(model)
    #plot_model(model, to_file='CNN.png')
    
    return model



def main(cross_idx,repeat_idx,blur_size,pool_factor,save_root="Result_CNN_aug"):
    if not os.path.exists(save_root):
        os.mkdir(save_root)
    pool_type = "maxblur-"
    if blur_size > 0:
        pool_type = pool_type + str(blur_size)
    elif blur_size == 0:
        pool_type = "max"
    elif blur_size == -1:
        pool_tyoe = "avg"
    
    save_dir = os.path.join(save_root,"results_" + pool_type)
    if not os.path.exists(save_dir):
        os.mkdir(save_dir)

    X_train,y_train,X_test,y_test = load_dataset(cross_idx,dataset_files)
    
    model_save_dir = os.path.join(save_dir,"models")
    fig_save_dir = os.path.join(save_dir,"figs")
    result_path = os.path.join(save_dir,"result.csv")

    idx = (cross_idx)*10 + repeat_idx + 1
    model = CNN_MaxBlurPool(blur_size,pool_factor)
    #return model
    history = model.fit(X_train,y_train,epochs=epoches,verbose=1)
    prediction = model.predict(X_test)
    loss,acc = model.evaluate(X_test,y_test)
    meta_data = {"history":history.history,"prediction":prediction}
    
    print("test acc={} , test loss={}".format(acc,loss))
    
    model_path = os.path.join(model_save_dir,str(idx)+"-model.h5")
    save_model(idx,model,model_save_dir)
    del model

    #save_model_json(idx,model,model_save_dir)
    #validate for saved model
    #with CustomObjectScope({'BlurPooling1D': BlurPooling1D}):
    #    model_loaded = load_model(model_path)
    #    _,loaded_acc = model_loaded.evaluate(X_test,y_test)
    #print("diff=",acc-loaded_acc)

    columns = ["idx","pool_factor","pool_type","cross_idx","repeat_idx","loss","accuracy"]
    if idx==1:
        write_header(file_path=result_path,header=columns)
            
    #save_graphs(idx,history,prediction,y_test,fig_save_dir)
    values = [idx,pool_factor,pool_type,cross_idx,repeat_idx,loss,acc]
    df = pd.DataFrame([values],columns=columns)
    with open(result_path,"a") as f:
        df.to_csv(f, header=False)
    

epoches = 5


if __name__ == "__main__":
    cross_idx = int(sys.argv[1])
    repeat_idx = int(sys.argv[2])
    blur_size = int(sys.argv[3])
    pool_factor= int(sys.argv[4])
    save_root = "Result_CNN_aug_pool"+str(pool_factor)
    model = main(cross_idx,repeat_idx,blur_size,pool_factor,save_root)
