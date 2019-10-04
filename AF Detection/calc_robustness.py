#!/usr/bin/env python
# coding: utf-8

# In[1]:


import os
import sys
from tqdm import tqdm_notebook
from utils import *
from utils import BlurPooling1D,excute_command
from keras.utils import CustomObjectScope
from keras.models import load_model


# In[2]:


win_len = 100
min_norm,max_norm = 47,651
dataset_root = "./dataset_RRI/"
test_keys_dict = {0: ['08434', '04746', '07162'], 
              1: ['08219', '07162', '05091'], 
              2: ['07879', '07162', '08455'], 
              3: ['05261', '06453', '07162'], 
              4: ['07162', '04043', '08405']}



cross_change_dict = {
    0:3,
    1:1,
    2:4,
    3:2,
    4:0}


# In[7]:


X_test_cross = {}
y_test_cross = {}

for i in range(5):
    with open(os.path.join("./dataset_RRI/","dataset-cross"+str(i)+".pickle"),"rb") as f:
        dataset=pickle.load(f)
    X_test_cross[i] = dataset["X_test"]
    y_test_cross[i] = dataset["y_test"]




def generate_noised_data(X,y):
    noised_X = []
    for i in range(len(X)-1):
        if not y[i] == y[i+1]:continue
        rand_idx = np.random.randint(win_len-1)
        noised_x = list(np.squeeze(X[i]))
        removed_rri = noised_x[rand_idx]
        
        noised_x[rand_idx+1] = noised_x[rand_idx+1] + removed_rri
        
        noised_x.pop(rand_idx)
        noised_x.append(X[i+1][0])

        noised_X.append(noised_x)
    noised_X = np.expand_dims(np.array(noised_X),axis=2)
    return noised_X,to_categorical(y[:-1],num_classes=2)
 

def update_result_with_robustness(result_root):
    df = pd.read_csv(os.path.join(result_root,"new_result.csv"))
    df = df.drop("_",axis=1)

    robustness_list = []
    for row in df.iterrows():
        idx = int(row[1]["idx"])
        cross_idx = int(row[1]["cross_idx"])
        model_root = os.path.join(result_root,"models")
        model = load_model(os.path.join(model_root,str(idx)+"-model.h5"),custom_objects={"BlurPooling1D":BlurPooling1D})
        robusts = []
        for test_key in test_keys_dict[cross_change_dict[cross_idx]]:
            with open(os.path.join(dataset_root,test_key+".pickle"),"rb") as f:
                dataset = pickle.load(f)
            X = np.expand_dims( (dataset["X"] - min_norm) / (max_norm - min_norm) ,axis=2)
            y = dataset["y"]
            if len(X) == 1:
                continue
            noised_X,y = generate_noised_data(X,y)
            loss,robust = model.evaluate(noised_X,y,verbose=2)
            robusts.append(robust)
        robustness = np.mean(robusts)
        print(robustness)
        robustness_list.append(robustness)
        del model
    df["robustness"] = robustness_list
    
    new_result_file = os.path.join(result_root,"new_rseult_2.csv")
    with open(new_result_file,"w") as f:
        df.to_csv(f, header=True)




if __name__ == '__main__':
    result_root = sys.argv[1]
    update_result_with_robustness(result_root)

