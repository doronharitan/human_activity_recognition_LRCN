# Action Recognition in Video using LRCN
### Project overview:
I implemented the LRCN model from the paper [[1]](https://arxiv.org/abs/1411.4389)
 and studied the model accuracy on variety of cases.
  The model was trained on the UCF-101 dataset that holds ~13K movies
   of 101 different human actions [[2]](https://arxiv.org/abs/1212.0402).
The following project consist of:
1.	Implementation of the LRCN model in PyTorch 1.3.4
2.	Train the model on a sub-set of 55 classes.
3.	Custom tests of the model accuracy.
4.	Pipeline for labeling an arbitrary video for human actions with single frame resolution.

# Tl;dr
1. LRCN can successfully classify what human action occur in a video.
2. LRCN classification accuracy decreases during the video fragment where the action changes. 
In this case the level of accuracy depends on the number of frames from each class. 
3. LRCN can label arbitrary video for human action.

## Table of content
- [LRCN model overview](#lrcn-model-overview)
- [Data Pre-processing](#data-pre-processing)
- [Test the accuracy of the model on test dataset](#basic-test-mode)
- [Test the accuracy of the model on a video that containe multiple human actions](#diverse-human-actions-video-test-mode)
- [Predicting human action in arbitrary videos](#predicting-human-action-in-arbitrary-videos)

## LRCN model overview
The LCRN model enables human action recognition in videos.
The video frames are embedded by a pre-trained ResNet-152 (ImageNet) 
and processed by a bi-directional LSTM layer that outputs the labels of the the preformed action .
This model implements the CNN-RNN framework.

The input of the network is a short section of a video presenting human activity. 
The output is a probability vector of possible labels.

## Data Pre-processing
Each video is downsampled and reduced to a single fragment 
(number of frames/fragment is a hyper-parameter). 

Elaboration on the preprocessing pipeline:
1. Read each video in the raw video dataset using CV2.
2. Downsample the video to 2.5 FPS (controlled by args.sampling_rate)
3. Extract N consecutive frames from a random location of the video. Discard the rest.
 
## Basic test mode:
Testing the accuracy of the model on the test dataset (comes with the UCF-101 Dataset)
The model reached a classification accuracy of **90.5%** when tested on the UCF-101 test dataset 
(Random guess baseline = 1.8%)

In order to understand the ability of the model to classify correctly each class I analyzed the 
confusion matrix.
The Confusion matrix suggests that the model confuses similar classes such as 
'Military Parade' with 'Band marching' and 'Haircut' with 'Blow Dry Hair'.

   <p align="center"><img width="650" height="550" src="https://github.com/doronharitan/human_activity_recognition_LRCN/blob/master/figuers/Normalized_confusion_matrix.png"></p>
   
   
Could the class confusion be an artifact of an unbalanced data, or is it truly due to class similarity?

First I examined how many data points are there in each class. As shown in the figure the classes are slightly imbalanced (See ‘Blowing Candles’ class)
   
   <p align="center"><img width="650" height="400" src="https://github.com/doronharitan/human_activity_recognition_LRCN/blob/master/figuers/train_val.jpg">

  To test if the classification accuracy drops with mis-represented class ,
   I plotted the accuracy of each class and marked in red all of the misrepresented ones. 
   A class is defined as misrepresented if its volume is smaller by 10% than an average..  
   We don’t see significant correlation due to contradicting examples such as ‘Jumping rope’ and
    ‘Apply Lipstick’.

   <p align="center"><img width="600" height="500" src="https://github.com/doronharitan/human_activity_recognition_LRCN/blob/master/figuers/The_accuracy_score_for_each_class.png"></p>


## Diverse human actions video test mode:
This test checks the model performance on videos that contain change of 
a human action (for example the first part describes one action 
that changes to another).
 
I trained the model to predict human action
based on 5 continues frames. Can it predict the correct action 
when 2 of the frames describes 'applying makeup' and the other 3 describes 'crawling baby'?
 
 This test was applied to 16 concatenated movie fragments (5 frames each, total test length 80 frames)
 The fragments were randomly chosen from the test dataset. The model predicts the label of each
 consecutive 5 frames of the test video (‘sliding-window’of size 5).
 The true label of each sliding window is set according the maximal frame count 
  from a single action: Each fragment can span up to 2 different actions, see image below)
   
  
   ![alt text](https://github.com/doronharitan/human_activity_recognition_LRCN/blob/master/figuers/what_is_sliding_window.gif)

In this test the model reached a classification accuracy of **73.6%**. 
The inferior accuracy is explained by the frequent action transitions.
   
<p align="center"><img width="400" height="400" src="https://github.com/doronharitan/human_activity_recognition_LRCN/blob/master/figuers/Video_with_prediction_vs_true_labels.gif"></p>
 * Correct classification are marked in green. False classification is marked in another color (each class has its own color).
    

 I analyzed the accuracy of the model as a function of the number of frames from the 
 first action in the N frame window. 
     
  <p align="center"><img width="300" height="250" src="https://github.com/doronharitan/human_activity_recognition_LRCN/blob/master/figuers/analysis_of_predicted_labels_in_sliding_window.png"></p>
        
The results indicate that the classification accuracy depend on the number of frames from each class. 
When all of the frames belong to one action (N=5 frames) the model has high classification accuracy.
The accuracy decreases with the number of frames of one class, down to ~60% in case of almost equal frame counts
(in the case of 3 frames from the first action and 2 frames from the second action) 
     
     
 **What does the model predict when it misclassifies an action?** To check this,
  I plotted the predicted label VS. the true label, colored as follows: 
1. Light orange: The true and the predicted labels are identical.2. Light-blue: 
3. when the predicted label is for the second action in the window, although the majority of the frames are from the first action.Reminder:The true label is the first action (blue and light blue respectively)
4. when the predicted label doesn't equal the true label or the second action in the window (red).
     
     <p align="center"><img src="https://github.com/doronharitan/human_activity_recognition_LRCN/blob/master/figuers/change_in_accuracy_with_the_movment_of_sliding_window.png"></p>
      
 The above results show that in 40% of the cases (6/15) the model predicts the second action in the window although the majority of the frames are from the first action. This indicates that the last frames in the window could have higher influence in the classification
      of the action. This observation can be studied by:
1. Repeat this experiment multiple times with different videos and see if the results are consistent. 
2. How the output of the LSTM changes with the sliding window.
3. Investigate the weight of the LSTM forget gate.
4. Check ohw bidirectional LSTM or attention influence the model performance.

## Predicting human action in arbitrary videos
The performance of the model is tested on videos taken from youtube. 
  
Similarly to the previous test, the video is down-sampled to match the FPS of the training data set. The model produces a sequence of labels by predicting the action accruing in each N-frames of the sliding window.

The model classified correctly 91.8% of the frames in a video of a single action. 
     <p align="center"><img src="https://github.com/doronharitan/human_activity_recognition_LRCN/blob/master/figuers/youtube_test.gif"></p>

## Referance
1. Donahue, Jeffrey, et al. "Long-term recurrent convolutional networks for visual recognition and description." Proceedings of the IEEE conference on computer vision and pattern recognition. 2015.
2. Soomro, Khurram, Amir Roshan Zamir, and Mubarak Shah. "UCF101: A dataset of 101 human actions classes from videos in the wild." arXiv preprint arXiv:1212.0402 (2012).


