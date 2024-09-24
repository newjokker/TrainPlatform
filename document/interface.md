# 接口文档

### 启动容器

* docker run --gpus '"device=0,1,2"' -e HOST_IP=192.168.3.50 -v /etc/localtime:/etc/localtime:ro --shm-size=8g -p 60067:60067 -p 7869:7869 -v /home/suanfa-2/ucd_cache:/usr/ucd_cache -v /home/suanfa-2/ldq/YoloTrainServer/runs:/usr/src/app/runs -d auto_yolo_train_server:v1.3.1


## 使用的接口

### Start Training Process
POST /start

Starts a training process using the provided configuration.

Request Body:
model_type (str): The type of the model.
batch_size (int): The batch size for training.
device (int): The GPU device ID to use.
epochs (int): Number of training epochs.
imgsz (int): Image size for input.
workers (int): Number of worker threads for data loading.
model_name (str): Name of the model.
train_ucd (str): Path or identifier for the training dataset.
val_ucd (str): Path or identifier for the validation dataset.
labels (str): Path to the labels file.
continue_train (bool): Whether to continue training from an existing checkpoint.
hyp_dict (dict): Dictionary containing hyperparameters.
Response:
{"message": "Training started"}

### Get Training Status
GET /status

Returns the current status of the training process.

Response:
JSON object representing the status of the process.

### Terminate Training Process
GET /terminate/{model_name}

Terminates a running training process.

Path Parameters:
model_name (str): The name of the model to terminate.
Response:
{"message": "Process terminated"}

### Retrieve Training Information
GET /train_info/{model_name}

Returns information about the training logs and configuration.

Path Parameters:
model_name (str): The name of the model.
Response:
{"train_log_std": [...], "opt_yaml": [...], "train_log_err": [...]}

### Clear Training Cache
GET /clear_train_cache/{model_name}

Clears the cache files associated with a specific model.

Path Parameters:
model_name (str): The name of the model.
Response:
Confirmation message.

### List Cache Information
GET /cache_info

Lists all models and their associated folders.

Response:
{"train_info": [...], "train": [...]}

### Download Model Weights
GET /download_model/{model_name}/weights/{model_type}

Downloads the specified model weights file.

Path Parameters:
model_name (str): The name of the model.
model_type (str): Type of weights to download ('last.pt' or 'best.pt').
Response:
Binary file download or error message.
8. Get GPU Information
GET /gpu_info

Returns the current GPU usage statistics.

Response:
JSON array of GPU information objects.

### Help Documentation
GET /help

Provides a basic help page that can be expanded with more detailed documentation.

Response:
Placeholder for future documentation content.

