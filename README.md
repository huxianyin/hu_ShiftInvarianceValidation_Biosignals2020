# hu_ShiftInvarianceValidation_Biosignals_2019

## Task1: Validate on AF Detection from ECG

### What is AF Detection
- We used MIT-BIH Atrial fibrillation (AF) Dataset
- The dataset contains 23 records of 10 hour ECG with heart beat annotation and AF annotation
- 100 RRI as a Segment
- Do 2-class classification Task (Normal / AF)


### Results

- Evaluate using Accuracy and Consistency
<div align="center">
<img src="https://github.com/heilab/hu_ShiftInvarianceValidation_Biosignals_2019/blob/master/AF%20Detection/figs/Scatter_acc-consis/with%20aug/1CNN_w.png" width="200" height="250"/>
<img src="https://github.com/heilab/hu_ShiftInvarianceValidation_Biosignals_2019/blob/master/AF%20Detection/figs/Scatter_acc-consis/with%20aug/2CNN_w.png" width="200" height="250"/>
<img src="https://github.com/heilab/hu_ShiftInvarianceValidation_Biosignals_2019/blob/master/AF%20Detection/figs/Scatter_acc-consis/with%20aug/3CNN_w.png" width="200" height="250"/>
</div>


- Evaluate using Accuracy and Robustness
