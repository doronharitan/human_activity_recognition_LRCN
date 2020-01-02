# Action Recognition in Video using LRCN
PyTorch implementation of Long-term Recurrent Convolutional Networks (LRCN) [1]

## LRCN model
The following network enables human action recognition in videos.
 The network utilies frame embeddings that were extracted by a pre-trained ResNet-152 (ImageNet) and a bi-directional LSTM operating on them in order to predict the preformed  action in the input video.

The input of the network would be a short 5 frames* video presenting human activity. 

In the train and in the basic test mode the frames would be extracted from raw video taken from UCF-101 
Dataset (using the script preprocessing_data.py, for further information see 'Data Pre-processing' paragraph below). The raw video would be sampled before the extraction of the frames to lower FPS (default from 25 FPS to 2.5 FPS). Then randomly, a start point to sample the video would be generated, and 5 continues frames (total of 2 sec) would be extracted and save as a short video which would be used as the input to the network.

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
This function would sample the raw video dataset and create new down-sampled 5 Frames* videos. If processing_mode='main' This videos would be saved in a designated folder. 

Elaboration on the function steps:
1. Read each video in the raw video dataset using CV2.
2. From each video X (args.sampling_rate) frames are sampled reducing the FPS by args.sampling_rate (for example from 25 to 2.5 FPS).
3. The function randomly set the start point where the new sampled array would be read from, and Y(args.num_frames_to_extract) continues frames are extracted.
4. if processing_mode == 'main' The Y continues frames are extracted and save to a new video if not the data in tensor tyoe mode is passed to the next function.

*default settings, can be changed using the preprocessing parameters --num_frames_to_extract
 
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

- Basic test mode - testing the accuracy of the model on the test video dataset (comes with the UCF-101 Dataset)

```
python test.py   --sampled_data_path dir_where_extracted_videos_were_saved\
                 --ucf_list_dir add_dir\
                 --model_dir add_saved_checkpoint_dir\
                 --model_name add_saved_checkpoint_name    
```

- Diverse human actions video test mode - testing the accuracy of the model on a 156 frames** video showing diverse human actions.
The video was created by random picking and grouping 5-frames* test videos (down-sampled test videos from the UCF-101 dataset, output of the preprocessing_data.py script).

![alt text](https://github.com/doronharitan/human_activity_recognition_LRCN/blob/master/figuers/raw_continues_movie_1.gif)

   In this test mode a sliding window of X frames (args.num_frames_to_sample) would be passed on the continues video,
   creating a stack of x-frames videos that can be used as an input to the LRCN network (note that the args.num_frames_to_sample have to be equal to the args.num_frames_to_sample used in the pre-processing of the data and in the training of the network).
   The true label for each sliding window would be set according the majority of frames we have for each action,
   meaning if the sliding window has 3 frames from the first action and two from the next action ,the label of the sliding
   window would be the first action
      
    ![alt text](https://github.com/doronharitan/human_activity_recognition_LRCN/blob/master/figuers/Video_with_prediction_vs_true_labels.gif)

   In this mode we also test how accurate the model is when the 5-frames* model input consistent from different human actions (for example the first 3 frames are form one video 
   and the last two are the start of a second video).
   
   We trained the model to predict the human action based on 5 continues frames. Can he predict the correct action when 2 of the frames describes 'applying makeup' and the other 3 describes 'crawling baby'? would we see that what actually determent what the model predict is the last frame?
      
    *default settings, can be changed using the preprocessing parameters --num_frames_to_extract
    ** depends on the batch size
```
python test_continues_movie.py   --sampled_data_path dir_where_extracted_videos_were_saved\
                                 --ucf_list_dir add_dir\
                                 --model_dir add_model_dir\
                                 --model_name add_model_name    
```    

- Random Youtube video test mode -  testing the accuracy of the model in predicting the human action happening in a random youtube video (downloaded from youtube in mp4 format).
  
  First the youtube video is down-sampled so the final FPS would be the same as the down-sampled train UCF-101 dataset. (This data pre-precessing could be: 1. 'live' meaning without directly running preprocessing_data.py and saving the down-sampled data. 2. It could run in steps. first running the preprocessing_data.py and saving the down-sampled data and than running the youtube test mode )
  Than, similar to what happens in 'Diverse human actions video test mode' a sliding window of X frames (args.num_frames_to_sample) would be passed on the down-sampled video,
   creating a stack of x-frames videos that can be used as an input to the LRCN network (note that the args.num_frames_to_sample have to be equal to the args.num_frames_to_sample used in the training of the LRCN network).

Note: Due to the type of video we trained our dataset on, we can't predict what human action happens in the video, if it is not happening in the center of the video (meaning in the 224X224 box we cut off from the original frames). 

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
WIP

## Conclusion?
WIP