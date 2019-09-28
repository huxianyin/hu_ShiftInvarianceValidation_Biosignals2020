# Validate on AF Detection from ECG

## What is AF Detection
- We used MIT-BIH Atrial fibrillation (AF) Dataset
- The dataset contains 23 records of 10 hour ECG with heart beat annotation and AF annotation
- 100 RRI as a Segment
- Do 2-class classification Task (Normal / AF)


## Results

### Evaluate using Accuracy and Consistency
<div align="center">
  <img src="https://github.com/heilab/hu_ShiftInvarianceValidation_Biosignals_2019/blob/master/AF%20Detection/figs/result_cnn_1.png" width="320" height="300"/> <img src="https://github.com/heilab/hu_ShiftInvarianceValidation_Biosignals_2019/blob/master/AF%20Detection/figs/result_cnn_2.png" width="320" height="300"/> <img src="https://github.com/heilab/hu_ShiftInvarianceValidation_Biosignals_2019/blob/master/AF%20Detection/figs/result_cnn_3.png" width="320" height="300" />
</div>


### Examples
- Confidence of model prediction on Sfhiting of RRI Segegment(No.123) in record07162
<img src="https://github.com/heilab/hu_ShiftInvarianceValidation_Biosignals_2019/blob/master/AF%20Detection/figs/samp.123.png" width="800" height="200" />

### Animated Examples
- Baseline (using max pooling)
<img src="https://github.com/heilab/hu_ShiftInvarianceValidation_Biosignals_2019/blob/master/AF%20Detection/figs/max.gif" width="600" height="200" />

- Maxblur (using maxblur pooling with filter size=7)
<img src="https://github.com/heilab/hu_ShiftInvarianceValidation_Biosignals_2019/blob/master/AF%20Detection/figs/maxblur-7.gif" width="600" height="200" />


## How to run
