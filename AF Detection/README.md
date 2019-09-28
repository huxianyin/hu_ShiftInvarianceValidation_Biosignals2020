# Validate on AF Detection from ECG

## What is AF Detection
- We used MIT-BIH Atrial fibrillation (AF) Dataset 
- The dataset contains 23 records of 10 hour ECG with heart beat annotation and AF annotation
- 100 RRI as a Segment
- Do 2-class classification Task (Normal / AF)


## Results

- Evaluate using Accuracy and Consistency
	![result_1_layer_cnn](https://github.com/heilab/hu_ShiftInvarianceValidation_Biosignals_2019/blob/master/AF%20Detection/figs/result_cnn_1.png)

- Examples
	- Confidence of model prediction on Sfhiting of RRI Segegment(No.[123]) in record07162
	![[result_123]](https://github.com/heilab/hu_ShiftInvarianceValidation_Biosignals_2019/blob/master/AF%20Detection/figs/samp.123.png)

- Animated Examples
	- Baseline (using max pooling)
	![[result_max]](https://github.com/heilab/hu_ShiftInvarianceValidation_Biosignals_2019/blob/master/AF%20Detection/figs/max.gif)
	
	- Maxblur (using maxblur pooling with filter size=7)
	![[result_maxblur-7]](https://github.com/heilab/hu_ShiftInvarianceValidation_Biosignals_2019/blob/master/AF%20Detection/figs/maxblur-7.gif)


## How to run 


