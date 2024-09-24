
from fastapi import FastAPI, HTTPException
import subprocess
import yaml
import uuid
import gpustat
import uvicorn
import shutil
import os
import signal
from pydantic import BaseModel
import argparse
from JoTools.utils.FileOperationUtil import FileOperationUtil
from fastapi.responses import FileResponse
from config import ROOT_DIR, PROJECT, UCD_CACHE_DIR, UCD_APP_PATH, BATCH_SIZE, DEVICE, EPOCHS, IMGSZ, WORKES, MODEL_TYPE, PORT_TRAIN_SERVER


app = FastAPI()

class TrainInfo(BaseModel):
    model_type:str=MODEL_TYPE
    batch_size:int=BATCH_SIZE
    device:int=DEVICE
    epochs:int=EPOCHS
    imgsz:int=IMGSZ
    workers:int=WORKES
    model_name:str
    train_ucd:str
    val_ucd:str
    labels:str
    continue_train:bool=False   # 继续训练
    hyp_dict:dict={}

class SingleProcessManager:
    def __init__(self):
        self._process = {}
        self._model_name = set()
        self._safe_code = {}

    @staticmethod
    def get_hyp_yaml(yaml_path, change_info):
        """获得超参"""
        context = {"lr0": 0.01,
                   "lrf": 0.01,
                   "momentum": 0.937,
                   "weight_decay": 0.0005,
                   "warmup_epochs": 3.0,
                   "warmup_momentum": 0.8,
                   "warmup_bias_lr": 0.1,
                   "box": 0.05,
                   "cls": 0.5,
                   "cls_pw": 1.0,
                   "obj": 1.0,
                   "obj_pw": 1.0,
                   "iou_t": 0.2,
                   "anchor_t": 4.0,
                   "fl_gamma": 0.0,
                   "hsv_h": 0.015,
                   "hsv_s": 0.7,
                   "hsv_v": 0.4,
                   "degrees": 0.0,
                   "translate": 0.1,
                   "scale": 0.5,
                   "shear": 0.0,
                   "perspective": 0.0,
                   "flipud": 0.0,
                   "fliplr": 0.5,
                   "mosaic": 1.0,
                   "mixup": 0.0,
                   "copy_paste": 0.0}

        with open(yaml_path, "w", encoding="utf-8") as f:
            for each in change_info:
                if each in context:
                    context[each] = float(change_info[each])
                else:
                    return {"error": f"* {each} should not in param", "success": False}
            yaml.dump(context, f)

        if os.path.exists(yaml_path):
            return {"success": True, "error": ""}
        else:
            return {"success": False, "error": f"create yaml failed : {yaml_path}"}

    @staticmethod
    def get_dataset_yaml(yaml_path, label_list, assign_ucd_cache_dir, assign_root_dir):
        img_dir = os.path.join(assign_ucd_cache_dir, "img_cache")
        context = dict()
        with open(yaml_path, "w", encoding="utf-8") as f:
            context["path"]     = img_dir
            context["train"]    = [img_dir]
            context["val"]      = [img_dir]
            context["train_label"] = [os.path.join(assign_root_dir, r"txt_dir/train")]
            context["val_label"]    = [os.path.join(assign_root_dir, r"txt_dir/val")]
            context["nc"]           = len(label_list)
            # tags
            context["names"] = {}
            for index, each_tag in enumerate(label_list):
                context["names"][index] = each_tag
            yaml.dump(context, f)

        if os.path.exists(yaml_path):
            return {"success": True, "error": ""}
        else:
            return {"success": False, "error": f"create yaml failed : {yaml_path}"}

    def start_process(self, train_info:TrainInfo):
        """Starts the subprocess if not already running."""

        run_info = self.get_status()

        try:
            model_name      = train_info.model_name
            continue_train  = train_info.continue_train
            train_ucd       = train_info.train_ucd
            val_ucd         = train_info.val_ucd
            labels          = train_info.labels
            model_type      = train_info.model_type
            batch_size      = train_info.batch_size
            device          = train_info.device
            epochs          = train_info.epochs
            imgsz           = train_info.imgsz
            workers         = train_info.workers
            hyp_dict        = train_info.hyp_dict
            assign_proj_dir = os.path.join(project, model_name)
            assign_root_dir = os.path.join(root_dir, model_name)
            os.makedirs(assign_root_dir, exist_ok=True)

            if len(model_name) < 2:
                raise HTTPException(status_code=500, detail=f"model_name length must >= 2 : {model_name}")

            if "Running" in run_info:
                if model_name in run_info["Running"]:
                    raise HTTPException(status_code=500, detail=f"Failed to start process: model is on trainning : {model_name}")

            if continue_train:
                last_model_path = os.path.join(assign_proj_dir, r"weights/last.pt")
                if not os.path.exists(last_model_path):
                    if os.path.exists(assign_proj_dir):
                        shutil.rmtree(assign_proj_dir)
                    continue_train = False
            else:
                if os.path.exists(assign_proj_dir):
                    raise HTTPException(status_code=500, detail=f"Failed to start process: model name already exists : {model_name} and continue_train is False")

            # get hyp.yaml
            save_train_info_dir = os.path.join(root_dir, model_name)
            os.makedirs(save_train_info_dir, exist_ok=True)
            hyp_yaml_path = f"{root_dir}/{model_name}/hyp.yaml"
            create_hyp_info = SingleProcessManager.get_hyp_yaml(hyp_yaml_path, hyp_dict)
            if not create_hyp_info["success"]:
                return HTTPException(status_code=500, detail=f"* create hyp.yaml failed : {hyp_yaml_path} : {create_hyp_info['error']}")

            # get dataset yaml
            dataset_yaml_path = os.path.join(root_dir, model_name, "train.yaml")
            label_list = labels.split(",")
            label_list = [x.strip() for x in label_list if x.strip()]
            create_hyp_info = SingleProcessManager.get_dataset_yaml(dataset_yaml_path, label_list, assign_ucd_cache_dir=UCD_CACHE_DIR, assign_root_dir=os.path.join(root_dir, model_name))
            if not create_hyp_info["success"]:
                return HTTPException(status_code=500, detail=f"* create train.yaml failed : {hyp_yaml_path} : {create_hyp_info['error']}")

            # check model type
            model_type_list = ["yolov5n", "yolov5s", "yolov5m", "yolov5x"]
            if model_type not in model_type_list:
                return HTTPException(status_code=500, detail=f"model_type must in {model_type_list}")

            yaml_path   = os.path.join(assign_root_dir, "train.yaml")
            cfg_path    = os.path.join("./models", f"{model_type}.yaml")
            weight_path = os.path.join("./models", f"{model_type}.pt")
            if continue_train:
                command = f"python3 ./train.py  --resume \
                                                --data {yaml_path} \
                                                --cfg {cfg_path} \
                                                --weights {weight_path} \
                                                --project {project} \
                                                --batch-size {batch_size} \
                                                --device {device}  \
                                                --epochs {epochs} \
                                                --imgsz {imgsz} \
                                                --name {model_name} \
                                                --workers {workers} \
                                                --train_ucd {train_ucd}  \
                                                --val_ucd {val_ucd}  \
                                                --labels {labels}  \
                                                --hyp {hyp_yaml_path} \
                                                --root_dir {assign_root_dir}"
            else:
                command = f"python3 ./train.py  --data {yaml_path} \
                                                --cfg {cfg_path} \
                                                --weights {weight_path} \
                                                --project {project} \
                                                --batch-size {batch_size} \
                                                --device {device}  \
                                                --epochs {epochs} \
                                                --imgsz {imgsz} \
                                                --name {model_name} \
                                                --workers {workers} \
                                                --train_ucd {train_ucd}  \
                                                --val_ucd {val_ucd}  \
                                                --labels {labels}  \
                                                --hyp {hyp_yaml_path} \
                                                --root_dir {assign_root_dir}"
            # 使用 os.setsid 创建一个新的进程组
            save_log_dir = os.path.join(root_dir, model_name, "logs")
            os.makedirs(save_log_dir, exist_ok=True)
            err_path = os.path.join(save_log_dir, "train_err.log")
            std_path = os.path.join(save_log_dir, "train_std.log")
            self._process[model_name] = subprocess.Popen(command, shell=True, preexec_fn=os.setsid, stderr=open(err_path, "w"), stdout=open(std_path, "w"))
            # self._process = subprocess.Popen(command, shell=True)                             # 不用新的进程组的时候会出现问题
            self._safe_code = {model_name: str(uuid.uuid1())}
            self._model_name.add(model_name)
            return {"safe_code": self._safe_code, "message": f"start trainning success : {model_name}"}
        except Exception as e:
            raise HTTPException(status_code=500, detail=f"Failed to start process: {e}")

    def get_status(self):
        """Returns the status of the subprocess."""

        info = {"Running":set()}

        for model_name in self._model_name:
            if model_name in self._process:
                if self._process[model_name].poll() is None:
                    info["Running"].add(model_name)
                else:
                    self._model_name.remove(model_name)
                    self._process.pop(model_name)
        return info

    def terminate_process(self, model_name):
        """Terminates the subprocess if it is running."""

        if model_name in self._process:
            if self._process[model_name].poll() is None:
                try:
                    # 使用 os.killpg 终止整个进程组
                    os.killpg(os.getpgid(self._process[model_name].pid), signal.SIGTERM)
                    self._process[model_name].wait()  # 等待子进程终止
                    self._model_name.remove(model_name)
                    self._process.pop(model_name)
                    return {"message": f"Process terminated : {model_name}"}
                except ProcessLookupError:
                    return {"message": f"Model not trainning : {model_name}"}
            else:
                return {"message": f"Model not trainning : {model_name}"}
        else:
            return {"message": f"No trainning with : {model_name}"}

    def clear_cache(self, model_name):
        """删除训练的缓存文件"""

        def delete_model_dir(assign_model_name):
            assign_train_cache = os.path.join(project, assign_model_name)
            assign_train_info_cache = os.path.join(root_dir, assign_model_name)
            if os.path.exists(assign_train_cache):
                shutil.rmtree(assign_train_cache)

            if os.path.exists(assign_train_info_cache):
                shutil.rmtree(assign_train_info_cache)

        info = {"clear_cache_success": [], "clear_cache_failed": {}}
        if model_name == "*":
            for each_model_name in os.listdir(project):
                if not os.path.isdir(os.path.join(project, each_model_name)):
                    continue

                if each_model_name in self._process:
                    if self._process[each_model_name].poll() is None:
                        info["clear_cache_failed"][each_model_name] = f"model is trainning : {each_model_name}"
                    else:
                        delete_model_dir(each_model_name)
                        info["clear_cache_success"].append(each_model_name)
                else:
                    delete_model_dir(each_model_name)
                    info["clear_cache_success"].append(each_model_name)

        elif model_name == "":
            info["clear_cache_failed"][model_name] = f"model_name is empty"
        elif model_name in self._process:
            if self._process[model_name].poll() is None:
                info["clear_cache_failed"][model_name] = f"model is trainning : {model_name}"
            else:
                delete_model_dir(model_name)
                info["clear_cache_success"].append(model_name)
        else:
            assign_train_cache = os.path.join(project, model_name)
            assign_train_info_cache = os.path.join(root_dir, model_name)
            delete_model_dir(model_name)
            if os.path.exists(assign_train_cache) or os.path.exists(assign_train_info_cache):
                info["clear_cache_failed"][model_name] = f"clear {model_name} cache failed"
            else:
                info["clear_cache_success"].append(model_name)
        return info

# Global instance of the process manager
process_manager = SingleProcessManager()

@app.post("/start")
async def start_process(train_info:TrainInfo):
    """Starts the subprocess if not already running."""
    return process_manager.start_process(train_info)

@app.get("/status")
async def get_status():
    """Returns the status of the subprocess."""
    return process_manager.get_status()

@app.get("/terminate/{model_name}")
async def terminate_process(model_name:str):
    """Terminates the subprocess if it is running."""
    return process_manager.terminate_process(model_name)

@app.get("/train_info/{model_name}")
async def train_info(model_name:str):
    """打印模型的训练数据"""

    if model_name == "":
        return {"message": "model_name is empty", "status": False}

    return_info = {"train_log_std": None, "opt_yaml": None, "train_log_err":None}
    train_info_log = os.path.join(root_dir, model_name, "logs", "train_std.log")
    if os.path.exists(train_info_log):
        with open(train_info_log, "r") as log_file:
            info = log_file.readlines()
            info = [x.strip() for x in info if x.strip("\n")]
            return_info["train_log_std"] = info

    train_yaml_path = os.path.join(project, model_name, "opt.yaml")
    if os.path.exists(train_yaml_path):
        with open(train_yaml_path, "r") as log_file:
            info = log_file.readlines()
            info = [x.strip() for x in info if x.strip("\n")]
            return_info["opt_yaml"] = info

    train_error_log = os.path.join(root_dir, model_name, "logs", "train_err.log")
    if os.path.exists(train_error_log):
        # with open(train_error_log, 'r') as file:
        #     info = file.readlines()
        #     info = info[-20:]
        #     info = [x.strip() for x in info if x.strip("\n")]
        #     return_info["train_log_err"] = info
        last_n_char = 3000
        with open(train_error_log, 'rb') as file:
            # 将文件指针移到文件末尾
            file.seek(0, 2)
            bytes_in_file = file.tell()

            # 如果文件小于n个字节，则直接读取整个文件
            if bytes_in_file <= last_n_char:
                file.seek(0)
                info = file.read().decode('utf-8', errors='replace')
            else:
                file.seek(-last_n_char, 2)
                last_n_bytes = file.read(last_n_char)
                info = last_n_bytes.decode('utf-8', errors='replace')
            info = info.replace('\r\n', '\n').replace('\r', '\n')
            info = [line.strip() for line in info.split('\n') if line.strip()]
            return_info["train_log_err"] = info
    return return_info

@app.get("/clear_train_cache/{model_name}")
async def clear_train_cache(model_name:str):
    """删除训练的缓存文件"""
    return process_manager.clear_cache(model_name)

@app.get("/cache_info")
async def cache_info():
    """Terminates the subprocess if it is running."""
    info = {"train_info": [], "train": []}
    for each in FileOperationUtil.re_all_folder(root_dir, recurse=False):
        info["train_info"].append(os.path.split(each)[1])
    for each in FileOperationUtil.re_all_folder(project, recurse=False):
        info["train"].append(os.path.split(each)[1])
    return info

@app.get("/download_model/{model_name}/weights/{model_type}")
async def download_model(model_name:str, model_type:str):
    """download_model file"""
    if model_type not in ["last.pt", "best.pt"]:
        return {"error": "model_type must in ['last.pt', 'best.pt']"}

    model_path = os.path.join(project, model_name, "weights", model_type)
    if os.path.exists(model_path):
        return FileResponse(model_path, media_type='application/octet-stream', filename=model_type)
    else:
        return {"error": f"model not exists : {model_path}"}


@app.get("/help")
def help_doc():
    """直接返回接口文档"""
    return {}

# ---------------------------------------------------------------------------------------------------------

@app.get("/get_gpu_info")
def get_gpu_info():
    """直接返回gpu相关信息"""
    gpu_stats = gpustat.new_query()
    gpus_info = []
    for gpu in gpu_stats.gpus:
        gpus_info.append({
            'name': gpu.name,
            'index': gpu.index,
            'memory_used': gpu.memory_used,
            'memory_total': gpu.memory_total,
        })
    return gpus_info

@app.get("/get_model_name_has_cahce")
def get_model_name_has_cahce():
    """缓存的文件，可以是训练的可以是训练完毕的，可以是训练失败的"""
    return {"status": "success"}

@app.get("/get_model_name_in_training")
def get_model_name_in_training():
    """正在训练的模型列表"""
    return {}

@app.get("/get_train_process/{model_name}")
def train_process(model_name:str):
    if model_name == "test":
        return {"status": "success"}
    return {"status": "success", "info": ["get_train_process"]}

@app.get("/get_train_detail/{model_name}")
def train_detail(model_name:str):
    if model_name == "test":
        return {"status": "success"}
    return {"status": "success", "info": ["get_train_detail"]}

@app.get("/get_train_param/{model_name}")
def train_param(model_name:str):
    if model_name == "test":
        return {"status": "success"}
    return {"status": "success", "info": ["get_train_param"]}






if __name__ == "__main__":

    # TODO: ucd 直接生成一个配置文件，去修改配置文件即可
    # TODO: 多 GPU 进行训练


    root_dir    = ROOT_DIR
    project     = PROJECT

    uvicorn.run(app, host="0.0.0.0", port=PORT_TRAIN_SERVER)
