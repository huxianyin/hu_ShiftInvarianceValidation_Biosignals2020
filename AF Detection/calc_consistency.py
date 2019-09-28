import os
import sys
import wfdb
import pickle
import pandas as pd
import numpy as np
from keras.utils import to_categorical
from tqdm import tqdm_notebook
from sklearn.preprocessing import MinMaxScaler
import matplotlib.pyplot as plt
from keras.models import load_model
import matplotlib.animation as animation
from utils import BlurPooling1D,drop_repeat,write_header
from keras.utils import CustomObjectScope


win_len = 100
min_norm,max_norm = 47,651
dataset_root = "./dataset_RRI/"
test_keys_dict = {0: ['08434', '04746', '07162'], 
              1: ['08219', '07162', '05091'], 
              2: ['07879', '07162', '08455'], 
              3: ['05261', '06453', '07162'], 
              4: ['07162', '04043', '08405']}

def int2label(label):
    if label==0:
        return "Normal"
    elif label==1:
        return "AF"

def calc_consistency_once(X,label,seg_idx,model):
    continue_seg = np.concatenate( [X[seg_idx],X[seg_idx+1]],axis=0)
    sliding_win_start = 1
    sliding_win_end = sliding_win_start+win_len
    step = 1
    consistency_list = [] 
    slided_data = []
    while(sliding_win_end<len(continue_seg)):
        data = continue_seg[sliding_win_start:sliding_win_end,:]
        slided_data.append(data)
        sliding_win_start += step
        sliding_win_end += step
        
    slided_data = np.array(slided_data)
    logits = np.argmax(model.predict(slided_data),axis=1)
    labels = [label for i in range(len(logits))]
    consistency_list = np.array(logits==label,dtype=np.float32)
    return consistency_list

def calc_consistency(X,y,start,end,model):
    consistency = []
    consistency_normal = []
    consistency_af = []
    if len(X)==1:
        print("cosistency of can not calculate cosistency")
        return [],[],[]
        
    for seg_idx in range(len(X)-1):
        seg = X[seg_idx]
        next_seg = X[seg_idx+1]
        seg_end = end[seg_idx]
        next_seg_start = start[seg_idx+1]
        label = np.argmax(y[seg_idx])
        next_label = np.argmax(y[seg_idx+1])
        if not (seg_end == next_seg_start and label==next_label):continue
        consistency_list =  calc_consistency_once(X,label,seg_idx,model)
        consistency.append(consistency_list)
        
        if label == 0:
            consistency_normal.append(consistency_list)
        else:
            consistency_af.append(consistency_list)
        
    assert(len(consistency_normal)+len(consistency_af) == len(consistency))
    #print("cosistency of ",test_key,"=",np.mean(consistency))
    return consistency,consistency_normal,consistency_af

def calc_consistency_for_cross(idx,cross_idx,model):
    test_keys = test_keys_dict[cross_change_dict[cross_idx]]
    consistencys = []
    consistencys_a = []
    consistencys_n = []
    
    for test_key in test_keys:
        with open(os.path.join(dataset_root,test_key+".pickle"),"rb") as f:
            dataset = pickle.load(f)
        X = np.expand_dims( (dataset["X"] - min_norm) / (max_norm - min_norm) ,axis=2)
        y = to_categorical (dataset["y"],num_classes=2)
        start = dataset["start"]
        end = dataset["end"]
        consistency,consistency_normal,consistency_af = calc_consistency(X,y,start,end,model)
        consistencys += (consistency)
        consistencys_a += consistency_af
        consistencys_n += consistency_normal

    consistencys = np.concatenate(consistencys,axis=0)
    consistencys_a = np.concatenate(consistencys_a,axis=0)
    consistencys_n = np.concatenate(consistencys_n,axis=0)
    rand_indices=np.random.permutation(len(consistencys_n))
    consistencys_a_selected = consistencys_a[rand_indices]

    consistency_unbalenced = np.mean(consistencys)
    consistency_normal = np.mean(consistencys_n)
    consistency_af = np.mean(consistencys_a)
    consistency_af_selected = np.mean(consistencys_a_selected)
    consistency_balanced = (consistency_normal+consistency_af_selected)/2

    return consistency_balanced,consistency_unbalenced,consistency_normal,consistency_af,consistency_af_selected


def update_result_with_consistency(cross_idx,result_root):
    model_root = os.path.join(result_root,"models")
    result_path = os.path.join(result_root,"result.csv")
    result_df = drop_repeat(pd.read_csv(result_path),axis="loss",repeat=10)
    result_df = result_df.drop("_",axis=1)

    row = result_df.iloc[cross_idx]
    assert(cross_idx == row["cross_idx"])

    idx = int(row["idx"])
    repeat_idx = int(row["repeat_idx"])
    loss = row["loss"]
    acc = row["accuracy"]
    pool_factor = row["pool_factor"]
    pool_type = row["pool_type"]

    column_names = ["idx","pool_factor","pool_type","cross_idx","repeat_idx","loss","accuracy",
                    "consis_b","consis_ub","consis_n","consis_a","consis_a_s"]

    with CustomObjectScope({'BlurPooling1D': BlurPooling1D}):
        model = load_model(os.path.join(model_root,str(idx)+"-model.h5"))
    #if not check_model(model,cross_idx,acc):
    #    print("loaded model failed")
    #    return

    consis_b,consis_ub,consis_n,consis_a,consis_a_s = calc_consistency_for_cross(idx,cross_idx,model)
    print(cross_idx,consis_b,"\n")
    values = [idx,pool_factor,pool_type,cross_idx,repeat_idx,loss,acc,consis_b,consis_ub,consis_n,consis_a,consis_a_s]
    df = pd.DataFrame([values],columns=column_names)
    new_result_file =  os.path.join(result_root,"new_result.csv")
    
    if(cross_idx==0):
        write_header(header=column_names,file_path=new_result_file)

    with open(new_result_file, 'a') as f:
        df.to_csv(f, header=False)


def check_model(model,cross_idx,acc):
    data_cross_idx = cross_change_dict[cross_idx]
    X_test = X_test_cross[data_cross_idx]
    y_test = y_test_cross[data_cross_idx]
    loss,_acc = model.evaluate(X_test,y_test)
    flag = (_acc==acc)
    if not flag:
        print(_acc,acc)
    return flag



cross_change_dict = {
    0:3,
    1:1,
    2:4,
    3:2,
    4:0}
X_test_cross = {}
y_test_cross = {}

for i in range(5):
    with open(os.path.join("./dataset_RRI/","dataset-cross"+str(i)+".pickle"),"rb") as f:
        dataset=pickle.load(f)
    X_test_cross[i] = dataset["X_test"]
    y_test_cross[i] = dataset["y_test"]

if __name__ == '__main__':
    cross_idx = int(sys.argv[1])
    result_root = sys.argv[2]
    update_result_with_consistency(cross_idx,result_root)





