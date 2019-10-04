# hu_ShiftInvarianceValidation_Biosignals_2019

## Task1: Validate on AF Detection from ECG

### What is AF Detection
- We used MIT-BIH Atrial fibrillation (AF) Dataset
- The dataset contains 23 records of 10 hour ECG with heart beat annotation and AF annotation
- 100 RRI as a Segment
- Do 2-class classification Task (Normal / AF)


### Results

#### Evaluate using Accuracy and Consistency
- Accuracy : classification accuracy on test data
- Consistency : how often the model predict the same label given 2 different shift to the same input

| Data Augmentation | 1-Layer| 2-Layer| 3-Layer|
| --- | --- | --- | --- |
| Yes | <img src="https://github.com/heilab/hu_ShiftInvarianceValidation_Biosignals_2019/blob/master/AF%20Detection/figs/Scatter_acc-consis/with%20aug/1CNN_w.png" width="260" height="200"/>|<img src="https://github.com/heilab/hu_ShiftInvarianceValidation_Biosignals_2019/blob/master/AF%20Detection/figs/Scatter_acc-consis/with%20aug/2CNN_w.png" width="260" height="200"/>|<img src="https://github.com/heilab/hu_ShiftInvarianceValidation_Biosignals_2019/blob/master/AF%20Detection/figs/Scatter_acc-consis/with%20aug/3CNN_w.png" width="260" height="200"/>|
| No | <img src="https://github.com/heilab/hu_ShiftInvarianceValidation_Biosignals_2019/blob/master/AF%20Detection/figs/Scatter_acc-consis/without%20aug/1CNN_wo_improvement.png" width="260" height="200"/>|<img src="https://github.com/heilab/hu_ShiftInvarianceValidation_Biosignals_2019/blob/master/AF%20Detection/figs/Scatter_acc-consis/without%20aug/2CNN_wo_improvement.png" width="260" height="200"/>|<img src="https://github.com/heilab/hu_ShiftInvarianceValidation_Biosignals_2019/blob/master/AF%20Detection/figs/Scatter_acc-consis/without%20aug/3CNN_wo_improvement.png" width="260" height="200"/>|


#### Evaluate using Accuracy and Robustness
- Accuracy : classification accuracy on test data
- Robustness : classification accuracy on crashed test data

| Data Augmentation | 1-Layer| 2-Layer| 3-Layer|
| --- | --- | --- | --- |
| Yes | <img src="https://github.com/heilab/hu_ShiftInvarianceValidation_Biosignals_2019/blob/master/AF%20Detection/figs/Scatter_acc-robust/with%20aug/1CNN_w.png" width="260" height="200"/>|<img src="https://github.com/heilab/hu_ShiftInvarianceValidation_Biosignals_2019/blob/master/AF%20Detection/figs/Scatter_acc-robust/with%20aug/2CNN_w.png" width="260" height="200"/>|<img src="https://github.com/heilab/hu_ShiftInvarianceValidation_Biosignals_2019/blob/master/AF%20Detection/figs/Scatter_acc-robust/with%20aug/3CNN_w.png" width="260" height="200"/>|
| No | <img src="https://github.com/heilab/hu_ShiftInvarianceValidation_Biosignals_2019/blob/master/AF%20Detection/figs/Scatter_acc-robust/without%20aug/1CNN_wo_improvement.png" width="260" height="200"/>|<img src="https://github.com/heilab/hu_ShiftInvarianceValidation_Biosignals_2019/blob/master/AF%20Detection/figs/Scatter_acc-robust/without%20aug/2CNN_wo_improvement.png" width="260" height="200"/>|<img src="https://github.com/heilab/hu_ShiftInvarianceValidation_Biosignals_2019/blob/master/AF%20Detection/figs/Scatter_acc-robust/without%20aug/3CNN_wo_improvement.png" width="260" height="200"/>|
