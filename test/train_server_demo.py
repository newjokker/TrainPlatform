
from fastapi import FastAPI, HTTPException
import subprocess
import yaml
import uuid
import gpustat
import uvicorn
import shutil
import random
import os
import signal
from pydantic import BaseModel
import argparse
from JoTools.utils.FileOperationUtil import FileOperationUtil
from fastapi.responses import FileResponse


app = FastAPI()

class TrainInfo(BaseModel):
    model_type:str
    batch_size:int
    device:int
    epochs:int
    imgsz:int
    model_name:str
    train_ucd:str
    val_ucd:str
    labels:str
    continue_train:bool=False
    hyp_dict:dict={}


@app.post("/start_train")
def start_train(train_info:TrainInfo):
    return {"status": "success", "message": f"start train success : {train_info}"}

@app.get("/stop_train/{model_name}")
def stop_train(model_name:str):
    return {"status": "success", "message": f"stop train success : {model_name}"}

@app.get("/get_model_type")
def get_model_type():
    return {"status": "success", "model_type_list": ["yolos", "yolom"]}

@app.get("/get_gpu_info")
def get_gpu_info():
    """直接返回gpu相关信息"""
    # gpu_stats = gpustat.new_query()
    gpus_info = []
    for gpu in [1,2,3]:
        gpus_info.append({
            'name': "gpu.name",
            'index': random.choice(list(range(10))),
            'memory_used': "gpu.memory_used",
            'memory_total': "gpu.memory_total",
        })
    return {"status": "success", "gpu_info": gpus_info}

@app.get("/get_model_name_has_cahce")
def get_model_name_has_cahce():
    """缓存的文件，可以是训练的可以是训练完毕的，可以是训练失败的"""
    return {"status": "success", "model_name_list":["model1", "model2", "model3"]}

@app.get("/get_model_name_in_training")
def get_model_name_in_training():
    """正在训练的模型列表"""
    return {"status": "success", "model_name_list":["model1", "model2"]}

@app.get("/get_train_process/{model_name}")
def train_process(model_name:str):
    return {"status": "success", "info": ["get_train_process"]}

@app.get("/get_train_detail/{model_name}")
def train_detail(model_name:str):
    return {"status": "success", "info": ["get_train_detail"]}

@app.get("/get_train_param/{model_name}")
def train_param(model_name:str):
    return {"status": "success", "info": ["get_train_param"]}






if __name__ == "__main__":

    # TODO: ucd 直接生成一个配置文件，去修改配置文件即可
    # TODO: 多 GPU 进行训练

    uvicorn.run(app, host="0.0.0.0", port=11223)
