# hu_ShiftInvarianceValidation_Biosignals2020

- Validation of the effect of maxblur pooling method for improving temporal shift-invariance of Convolutional Neural Network (CNN) in bio-signals classification tasks.
- About maxblur pooling, please refer to [Making Convolutional Networks Shift-Invariant Again](https://richzhang.github.io/antialiased-cnns/)
- [Paper](https://www.insticc.org/node/TechnicalProgram/biostec/2020/presentationDetails/88799)
- [Best Paper Award](https://biosignals.scitevents.org/PreviousAwards.aspx?y=2021) in Biosignals2020

| Max Pooling | Maxblur Pooling|
| --- | --- |
| <img src="https://user-images.githubusercontent.com/34026599/174485379-ff9400a7-45f1-48d3-afd7-3bd6b3a9faac.gif" width="400" height="133" />| <img src="https://user-images.githubusercontent.com/34026599/174485417-c2786373-5984-4f28-b074-563882e01148.gif" width="400" height="133" /> |

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


#### Improvements of maxblur on non-augemented data in CNN with different number of pooling layers

| filter size | baseline = max | baseline = avg |
| --- | --- |--- |
| 7 | <img src="https://github.com/heilab/hu_ShiftInvarianceValidation_Biosignals_2019/blob/master/AF%20Detection/figs/compare_improv/baseline%3Dmax/maxblur-7%20vs%20max.png" width="260" height="200"/> |<img src="https://github.com/heilab/hu_ShiftInvarianceValidation_Biosignals_2019/blob/master/AF%20Detection/figs/compare_improv/baseline%3Davg/maxblur-7%20vs%20avg.png" width="260" height="200"/>|
