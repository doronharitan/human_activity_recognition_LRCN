# Action Recognition in Video using LRCN
PyTorch implementation of the Long-term Recurrent Convolutional Networks (LRCN) [1] model

## LRCN model
The following network enables human action recognition in video by a bi-directional LSTM operating on frame embeddings extracted by a pre-trained ResNet-152 (ImageNet).

The input of the network would be a short 5 frames* video presenting human activity. 

In the train and in the basic test mode the frames would be extracted from raw video taken from UCF-101 Dataset (using the script preprocessing data.py, see usage in installation). The raw video would be sampled before the extraction of the frames to lower FPS (default from 25 FPS to 2.5 FPS). Then randomly, a start point to sample the video would be generated, and 5 continues frames (total of 2 sec) would be extracted and save as a short video which would be used as the input to the network.

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
extract frames from the raw video dataset and create new 5 frames* videos that would be used as the input of the train and basic test mode 

*default settings, can be changed using the preprocessing parameters --num_frames_to_extract
 
```
python preprocessing_data.py –-sampled_data_dir dir_where_extracted_videos_would_be_saved\
                             -–row_data_dir add_dir\
                             -–ucf_list_dir add_dir   

```

#### Default args parameters for preprocessing_data.py
```
--sampling_rate             10          #what is the down-sample FPS factor 
--FPS                       25          #raw videos FPS 
--num_frames_to_extract     5

```

##  Train and test modes
#### Train mode
```
python train.py   --sampled_data_path dir_where_extracted_videos_were_saved\
                  --ucf_list_dir add_dir    
```

#### Test modes:
By default, model checkpoints are saved in the Saved_model_checkpoint directory using the following naming convention:
 epoch_<num_epoch>.pth.tar

- Basic test mode - testing the accuracy of the model on the test video annotation (comes with the UCF-101 Dataset)

```
python test.py   --sampled_data_path dir_where_extracted_videos_were_saved\
                 --ucf_list_dir add_dir\
                 --model_dir add_model_dir\
                 --model_name add_model_name    
```

- Diverse human actions video test mode - testing the accuracy of the model on a 320 frames** video showing diverse human actions.
The video was created by random picking and grouping 5-frames test videos (from the test videos annotation provided by UCF-101 dataset).

    In this mode we also test how accurate the model is when the 5-frames input consistent from different human action movies (for example the first 3 frames are form one movie 
    and the last two are the start of the second movie).
     We trained the model to predict the human action based on 5 continues frames. Can he predict the correct action when 2 of the frames describes 'applying makeup' and the other 3 describes 'crawling baby'? would we see that what actually determent what the model predict is the last frame?
    
    ** depends on the batch size
```
python test_continues_movie.py   --sampled_data_path dir_where_extracted_videos_were_saved\
                                 --ucf_list_dir add_dir\
                                 --model_dir add_model_dir\
                                 --model_name add_model_name    
```    

- test random video from youtube - WIP


#### Default args parameters to train and test modes
```
--epochs                    100
--batch-size                32 
--lr                        5e-4
-num_workers                4
--split_size                0.2         #set the size of the split between validation data and train data
--num_frames_video          5           #the number of frames we have in the input video
--seed                      42
--smaller_dataset           False       #train the network on smaller dataset, mostly useful for memory problems and debug mode
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
--model_dir                 model_dir   #used in the test mode
--model_name                model_name  #used in the test mode
```

## Results
WIP

## Conclusion?
WIP