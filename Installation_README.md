## Table of content
- [Installation](#installation)
- [Data Pre-processing](#data-pre-processing)
- [How to run the train and test script](#train-and-test-modes)
- [Default args parameters](#default-args-parameters)

## Installation
This implementation uses Python 3.7.4 and PyTorch.

Run the following lines in Anaconda Prompt to install the environment, download the database (~7GB)
 and extract the data:

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
*All hyper-parameters are documented in [Default args parameters](#default-args-parameters)

```
python preprocessing_data.py –-sampled_data_dir dir_where_extracted_videos_would_be_saved\
                             -–row_data_dir add_dir\
                             -–ucf_list_dir add_dir
```

## Train and test modes
*All hyper-parameters are documented in [Default args parameters](#default-args-parameters)

### Train mode
```
python train.py   --sampled_data_path dir_where_extracted_videos_were_saved\
                  --ucf_list_dir add_dir    
```

### Test modes:
By default, model checkpoints are saved in the Saved_model_checkpoint directory using the following naming convention:
 epoch_<num_epoch>.pth.tar

- #### _Basic test mode_:
```
python test.py   --sampled_data_path dir_where_extracted_videos_were_saved\
                 --ucf_list_dir add_dir\
                 --model_dir add_saved_checkpoint_dir\
                 --model_name add_saved_checkpoint_name    
```
- #### _Diverse human actions video test mode_:
``` Run the test
python test_continues_movie.py   --sampled_data_path dir_where_extracted_videos_were_saved\
                                 --ucf_list_dir add_dir\
                                 --model_dir add_model_dir\
                                 --model_name add_model_name    
```    

```
python test_Youtube_videos.py   --row_data_dir dir_of_the_row_data_videos\
                                 --model_dir add_model_dir\
                                 --model_name add_model_name    
```    


## Default args parameters
#### Default args parameters for preprocessing_data.py
```
--sampling_rate             10          #what is the down-sample FPS factor 
--ucf101_fps                25          #raw videos FPS 
--num_frames_to_extract     5
--video_file_name           None        #Relevant for dataset = youtube only, The video file name we would process, if None the script would run on all of the video files in the row_data_dir folder
--dataset                   UCF101      #The dataset name. options = youtube, UCF101
```

#### Default args parameters for train
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
```

#### Default args parameters for test modes
```
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
--model_dir                 model_dir   
--model_name                model_name  

# used in all the youtube test mode

--video_file_name           None        #The video file name we would process, if None the script would run on all of the video files in the folder')
--preprocessing_movie_mode  'live'      #The pre-processing of the data would be 'live' (witout directly activating the script and witout saving the dowm-sampled video) or not ('precessed option)
--dataset'                  'youtube'   #The dataset type. options = youtube, UCF101
--ucf101_fps                25          #UCF101 raw videos FPS 
```