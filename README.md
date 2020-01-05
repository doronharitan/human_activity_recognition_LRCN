# Action Recognition in Video using LRCN
PyTorch implementation of Long-term Recurrent Convolutional Networks (LRCN) [[1]](#referance)

## LRCN model
The following network enables human action recognition in videos.
 The network utilise frame embeddings that were extracted by a pre-trained ResNet-152 (ImageNet) and a bi-directional LSTM operating on them, in order to predict the preformed action in the input video (CNN-RNN).

The input of the network is a short 5 frames* video presenting human activity. 

In the train and in the basic test mode the frames would be extracted from raw videos taken from [UCF-101](https://www.crcv.ucf.edu/data/UCF101.php) [[2]](#referance)
dataset (using the script preprocessing_data.py, for further information see 'Data Pre-processing' paragraph below). The raw video would be sampled before the extraction of the frames to lower FPS (default from 25 FPS to 2.5 FPS). Then randomly, a start point to sample the video would be generated, and 5 continues frames (total of 2 sec) would be extracted and save as a short video which would be used as the input to the network.

*default settings, can be changed using the train and preprocessing parameters

## Installation
This implementation uses Python 3.7.4 and PyTorch.

All dependencies can be installed into a conda environment with the provided environment.yml file.


``` 
# ==== clone repository =====
git clone https://github.com/doronharitan/human_activity_recognition_LRCN.git
cd human_activity_recognition_LRCN

# ==== create conda env and activte it =====
conda env create -f environment.yml
conda activate lrcn_env

# ==== Downloads the UCF-101 dataset =====
- Linux users can use the sh file - download_ucf101.sh
$  bash download_ucf101.sh 
- Windows users can download directly form the following links:
http://crcv.ucf.edu/data/UCF101/UCF101.rar
https://www.crcv.ucf.edu/data/UCF101/UCF101TrainTestSplits-RecognitionTask.zip 

# ==== extract the data from the downloaded files ====
unrar UCF101.rar          
unzip ucfTrainTestlist.zip  
```

## Data Pre-processing
This function would sample the raw video dataset and create new down-sampled 5 Frame* videos. If processing_mode='main' This videos would be saved in a designated folder. 

Elaboration on the function steps:
1. Read each video in the raw video dataset using CV2.
2. From each video Y (args.sampling_rate) frames would be sampled, reducing the FPS by Y (args.sampling_rate, for example from 25 to 2.5 FPS).
3. The function randomly set the start point where the new sampled array would be read from, and X (args.num_frames_to_extract) continues frames would be extracted.
4. if processing_mode == 'main' The Y continues frames are extracted and save to a new video if not the data in tensor type mode is passed to the next function.

*default settings, can be changed using the preprocessing parameters: --num_frames_to_extract
 
```
python preprocessing_data.py –-sampled_data_dir dir_where_extracted_videos_would_be_saved\
                             -–row_data_dir add_dir\
                             -–ucf_list_dir add_dir\

```

#### Default args parameters for preprocessing_data.py
```
--sampling_rate             10          #what is the down-sample FPS factor 
--ucf101_fps                25          #raw videos FPS 
--num_frames_to_extract     5
--video_file_name           None        #Relevant for dataset = youtube only, The video file name we would process, if None the script would run on all of the video files in the row_data_dir folder
--dataset                   UCF101      #The dataset name. options = youtube, UCF101
```

##  Train and test modes
*Default args parameters to train and test modes are detailed below

#### Train mode
```
python train.py   --sampled_data_path dir_where_extracted_videos_were_saved\
                  --ucf_list_dir add_dir    
```

#### Test modes:
By default, model checkpoints are saved in the Saved_model_checkpoint directory using the following naming convention:
 epoch_<num_epoch>.pth.tar

#####Basic test mode: 
    
    Testing the accuracy of the model on the test dataset (comes with the UCF-101 Dataset)

```
python test.py   --sampled_data_path dir_where_extracted_videos_were_saved\
                 --ucf_list_dir add_dir\
                 --model_dir add_saved_checkpoint_dir\
                 --model_name add_saved_checkpoint_name    
```

##### Diverse human actions video test mode:

    Testing the accuracy of the model on a 80 frames** video showing diverse human actions.
The video was created by random picking and grouping 5-frame* test videos (down-sampled test videos from the UCF-101 dataset, output of the preprocessing_data.py script).

   In this test mode a sliding window of X frames (args.num_frames_to_sample) would be passed on the continues video,
   creating a stack of x-frames videos that can be used as an input to the LRCN network (note that the args.num_frames_to_sample have to be equal to the args.num_frames_to_sample used in the pre-processing of the data and in the training of the network).
   The true label for each sliding window would be set according the majority of frames we have for each action,
   meaning if the sliding window has 3 frames from the first action and two from the next action ,the label of the sliding
   window would be the first action
      
   ![alt text](https://github.com/doronharitan/human_activity_recognition_LRCN/blob/master/figuers/what_is_sliding_window.gif)

   In this mode we also test how accurate the model is when the 5-frame* input consistent from different human actions (for example the first 3 frames are form one video 
   and the last two are the start of a second video). We trained the model to predict human action
   based on 5 continues frames. Can he predict the correct action when 2 of the frames describes 'applying makeup' and the other 3 describes 'crawling baby'?
    would we see that what actually determent what the model predict is the last frame ot the first frames?
      
    *default settings, can be changed using the preprocessing parameters: --num_frames_to_extract
    ** depends on the batch size
```
python test_continues_movie.py   --sampled_data_path dir_where_extracted_videos_were_saved\
                                 --ucf_list_dir add_dir\
                                 --model_dir add_model_dir\
                                 --model_name add_model_name    
```    

- ######Random Youtube video test mode:

  Testing the accuracy of the model in predicting the human action happening in a random youtube video (downloaded from youtube in mp4 format).
  
  First the youtube video is down-sampled so the final FPS would be the same as the down-sampled train UCF-101 dataset (In this case, the data pre-precessing could be:  1.
   'live' meaning without directly running preprocessing_data.py and saving the down-sampled data. 2. It could run in steps. first running the preprocessing_data.py and saving the down-sampled data and than running the youtube test mode).
  Second, similar to what happens in 'Diverse human actions video test mode' a sliding window of X frames (args.num_frames_to_sample) would passed on the down-sampled video,
   creating a stack of X-frames videos that can be used as an input to the LRCN network (note that the args.num_frames_to_sample have to be equal to the args.num_frames_to_sample used in the training of the LRCN network).

Note: Due to the type of video dataset we trained our model on, we can't predict what human action happens in the video, if it is not happening in the center of the video (meaning in the 224X224 box we cut off from the original frames). 

```
python test_Youtube_videos.py   --row_data_dir dir_of_the_row_data_videos\
                                 --model_dir add_model_dir\
                                 --model_name add_model_name    
```    

#### Default args parameters to train and test modes
```
--epochs                    100
--batch-size                32 
--lr                        5e-4
-num_workers                4
--split_size                0.2         #set the size of the split between validation data and train data
--num_frames_video          5           #the number of frames we have in the input video
--seed                      42
--load_all_data_to_RAM      False       #load dataset directly to the RAM, for faster computation. usually use when the num of class is small and we want to debug the code
--latent_dim                512
--hidden_size               256         #LSTM hidden size
--lstm_layers               2
--bidirectional             True
--open_new_folder           'True’      #open a new folder where all of the run data would be saved at                    
--load_checkpoint           False
--checkpoint_path           ''
--checkpoint_interval       5
--val_check_interval        5           #Interval between running validation test
--local_dir                 os.getcwd()
--number_of_classes         None        #The number of classes we would train on. From the total 101 classes

# used in all the test mode

--model_dir                 model_dir   
--model_name                model_name  

# used in all the youtube test mode

--video_file_name           None        #The video file name we would process, if None the script would run on all of the video files in the folder')
--preprocessing_movie_mode  'live'      #The pre-processing of the data would be 'live' (witout directly activating the script and witout saving the dowm-sampled video) or not ('precessed option)
--dataset'                  'youtube'   #The dataset type. options = youtube, UCF101
--ucf101_fps                25          #UCF101 raw videos FPS 
```

## Results
The train model that was tested below was trained on a randomly chosen train/test set (from the 3 possible provided by UCF-101 dataset).
The model was trained and thus, tested on 55 classes from the 101 possible classes in UCF-101 dataset.
- _**Basic test mode**_:  The model reached a classification accuracy of **90.5%**.

    In order to understand the ability of the model to classify correctly each class I run a confusion matrix.
from the confusion matrix, shown below, we can learn that model something confused relatively similar classes, for example: 'Military Parade' with 'Band marching' and 'Haircut' with 'Blow Dry Hair'.

   <p align="center"><img width="650" height="550" src="https://github.com/doronharitan/human_activity_recognition_LRCN/blob/master/figuers/Normalized_confusion_matrix.png"></p>
   
   The above raised the question, Could the model 'confusion' (seen in the confusion matrix)  be caused by an unbalance representation of each class in the train dataset?
   Could it be ,that classes which have lower data points in the train dataset have higher tendency to be confused? or does the confusion comes from the fact that the classes are similar?
   
   In order to address this question, first I checked how many data point I have from each class in each mode (train, validation and test).
   The frequency of each class in the train/validation and in the test set can be seen below. We can see that their are classes that have lower representation in the dataset. for example 'Blowing Candles'
   
   <p align="center"><img width="650" height="400" src="https://github.com/doronharitan/human_activity_recognition_LRCN/blob/master/figuers/train_val.jpg">
   <p align="center"><img width="650" height="400" src="https://github.com/doronharitan/human_activity_recognition_LRCN/blob/master/figuers/test.jpg"></p>

  To test if the classification accuracy correlates with the frequency of each class, I plotted the classification accuracy of each class and marked in 'red' all of the classes that their frequency in the train dataset is lower than 90% of the average.
   From the results shown below, we can see that their isn't a tight correlation and that there are classes that their classification accuracy is low and their frequency is high (for example 'Jumping Rope') and vise versa (for example 'Apply Lipstick'). 

    <p align="center"><img width="600" height="500" src="https://github.com/doronharitan/human_activity_recognition_LRCN/blob/master/figuers/The_accuracy_score_for_each_class.png"></p>

- _**Diverse human actions video test mode**_ - In this test the model reached a classification accuracy of **73.6%**.
   
   <p align="center"><img width="400" height="400" src="https://github.com/doronharitan/human_activity_recognition_LRCN/blob/master/figuers/Video_with_prediction_vs_true_labels.gif"></p>
   
    In order to shed some light on why the model reached lower classification accuracy than the classification accuracy reached in the basic test mode,
     I analyzed the ability of the model to classify an action as a function of the number of frames we have from the first action in the sliding window. [A reminder what we did in this test mode](#diverse-human-actions-video-test-mode) 
     
     <p align="center"><img width="300" height="250" src="https://github.com/doronharitan/human_activity_recognition_LRCN/blob/master/figuers/analysis_of_predicted_labels_in_sliding_window.png"></p>
        
     The results show that the classification accuracy depend on the number of frames we have from each class. 
     When all of the frames are from one action (5 frames) the model has high classification accuracy.
     When it has 4 frames from the first action the classification accuracy decreases but not by much. 
     When the number of frames become similar (in the case of 3 frames from the first action or 3 frames 
     from the second action) the classification accuracy decrease by ~80% (in this specific run). 
     This means that the model doesn't follow the rule I set. The rule which determine that
     the true label of the window is set according to the majority of the frames.
     Could it be that in this scenario the model predicts the second action and not the first as I thought?
     (This would explain the decrease in the classification accuracy of a window with 3 frames from the first action but not the decrease of a window with 3 frames from the second action ) 
     
     <p align="center"><img width="700" height="400" src="https://github.com/doronharitan/human_activity_recognition_LRCN/blob/master/figuers/change_in_accuracy_with_the_movment_of_sliding_window.png"></p>
      
     The above results indecates that ....
## Referance
1. Donahue, Jeffrey, et al. "Long-term recurrent convolutional networks for visual recognition and description." Proceedings of the IEEE conference on computer vision and pattern recognition. 2015.
2. Soomro, Khurram, Amir Roshan Zamir, and Mubarak Shah. "UCF101: A dataset of 101 human actions classes from videos in the wild." arXiv preprint arXiv:1212.0402 (2012).