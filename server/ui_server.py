

import gradio as gr
import requests
import json
import yaml
from config import PORT_UI_SERVER, LOG_DIR, REGIST_SERVER_PORT, LOCAL_HOST, ENV_HOST, UCD_CHECK_URL
import os
from JoTools.utils.LogUtil import LogUtil
from requests.exceptions import RequestException

os.makedirs(LOG_DIR, exist_ok=True)

log_path = os.path.join(LOG_DIR, "ui_server.log")
log = LogUtil.get_log(log_path, 5, "ui_server", print_to_console=False)


global all_agent_info
all_agent_info = {}


with gr.Blocks() as demo:
    with gr.Row():

        with gr.Column(scale=1):
            with gr.Row():
                update_dataset_bt   = gr.Button('UpDate Dataset')
                update_agent_bt     = gr.Button('Update Agent')
                register_agent_bt   = gr.Button('Deregister Agent')
                gpu_info_bt         = gr.Button('Update GPU Info')

            train_dataset_dd    = gr.Dropdown(choices=[], label="train dataset", allow_custom_value=False, value="执行 UpDate Dataset 后选择数据集")
            with gr.Row():
                agent_dd            = gr.Dropdown(choices=[], label="agent", allow_custom_value=False, value="可以用于训练的 agent", interactive=True)
                model_name_bt       = gr.Textbox(value="", label="model name")
            
            with gr.Row():
                device_dd           = gr.Dropdown(choices=[], label="gpu id", allow_custom_value=True, value="执行 Update GPU Info")
                continue_train_dd   = gr.Dropdown(choices=["true", "false"], label="continue_train", allow_custom_value=False, value="false")
                epoch_dd            = gr.Dropdown(value="50", label="epoch", choices=[5,10,20,50,100,200], allow_custom_value=True)

            with gr.Row():
                batch_size_dd   = gr.Dropdown(value="10", label="batch_size", choices=[1, 2, 4, 6,10,20,50], allow_custom_value=True)
                model_type_dd   = gr.Dropdown(choices=[], label="model type", allow_custom_value=False)
                imgsz_bt        = gr.Dropdown(choices=["416", "512", "640", "832", "1280"], label="image size", allow_custom_value=True, value="640")
            
            labels_bt           = gr.Textbox(value="", label="labels")
            param_file          = gr.File(label="Upload a param json if need", file_types=[".json"])
            star_train_bt       = gr.Button('Start Train')

        with gr.Column(scale=1):
            with gr.Row():
                update_model_name_bt       = gr.Button('Update Model Name')
                update_train_model_name_bt = gr.Button('Update Train Model Name')
            model_name_dd = gr.Dropdown(choices=[], label="model name", allow_custom_value=True)
            
            with gr.Row():
                stop_train_bt   = gr.Button('Stop Train')
                Load_models_bt  = gr.Button('Load models')

            with gr.Row():
                main_process_bt     = gr.Button('MainProcess')
                param_bt            = gr.Button('Param')
                detailed_log_bt     = gr.Button('DetailedLog')
            info_text = gr.Textbox(label='info', lines=15, placeholder="wait...", interactive=False)


    def deregister_agent(agent_name):
        check_agent_name(agent_name)

        url = f"http://{LOCAL_HOST}:{REGIST_SERVER_PORT}/deregister_agent/{agent_name}"

        try:
            response = requests.get(url, headers={'Content-Type': 'application/json'})
        except RequestException as e:
            log.error(f"failed connect to : {url} , {e}")
            raise gr.Error(f"failed connect to : {url} , {e}")
        except:
            log.error(f"failed deregister_agent : {agent_name}")
            raise gr.Error(f"failed deregister_agent : {agent_name}")

        check_agent_response(response)
        log.info(f"* deregister_agent success : {agent_name}")
        return response.text, get_agent_info()

    def check_agent_response(agent_response, need_key=None):

        response_text = agent_response.text
        if agent_response.status_code != 200:
            raise gr.Error(f"response status error : {agent_response.status_code}, {response_text}")

        try:
            response_info = json.loads(response_text)
            if "status" not in response_info:
                log.error(f"no 'status' in response : {response_text}")
                raise gr.Error(f"no 'status' in response : {response_text}")
            
            if response_info["status"] == "failed":
                if "error_info" not in response_info:
                    log.error("not 'error info' in response when status is failed")
                    raise gr.Error("not 'error info' in response when status is failed")
            elif response_info["status"] == "success":
                if need_key is not None:
                    if need_key not in response_info:
                        log.error(f"no key : {need_key} in response")
                        raise gr.Error(f"no key : {need_key} in response")
            else:
                log.error("status in response should in ['failed', 'success']")
                raise gr.Error("status in response should in ['failed', 'success']")

        except Exception as e:
            log.error(f"response formart error : {e}, {response_text}")
            raise gr.Error(f"response formart error : {e}, {response_text}")

    def check_agent_name(agent_name):
        global all_agent_info
        if agent_name in [None, "", " "]:
            log.error("agent name is empty")
            raise gr.Error("agent name is empty")

        if agent_name not in all_agent_info:
            log.error(f"agent name not register, update agent info, agent_name : {agent_name}")
            raise gr.Error(f"agent name not register, update agent info, agent_name : {agent_name}")

        if ("host" not in all_agent_info[agent_name]) or ("port" not in all_agent_info[agent_name]):
            log.error(f"not 'host' or 'port' info in register info : {agent_name}, {all_agent_info[agent_name]}")
            raise gr.Error(f"not 'host' or 'port' info in register info : {agent_name}, {all_agent_info[agent_name]}") 

    def check_model_name(model_name):
        if model_name in [None, "", " "]:
            log.error("model name is empty")
            raise gr.Error("model name is empty")

    def download_model(agent_name, model_name):
        global all_agent_info

        check_agent_name(agent_name)
        check_model_name(model_name)

        now_agent_info = all_agent_info[agent_name]
    
        url = f"http://{now_agent_info['port']}:{now_agent_info['host']}/download_model/{model_name}/weights/best.pt\nhttp://{now_agent_info['port']}:{now_agent_info['host']}/download_model/{model_name}/weights/last.pt"
        log.info(f"* download model : {url}")
        return url

    def start_train(agent_name, train_ds, model_name, epoch, labels, model_type, imgsz, device, batch_size, continue_train, param_info):

        global all_agent_info

        check_agent_name(agent_name)
        check_model_name(model_name)

        now_agent_info = all_agent_info[agent_name]
    
        if train_ds in ["", " ", None]:
            raise gr.Error(f"error : train_dataset is empty")
        elif model_name in ["", " ", None]:
            raise gr.Error(f"error : model_name is empty")
        elif epoch in ["", " ", None]:
            raise gr.Error(f"error : epoch is empty")
        elif labels in ["", " ", None]:
            raise gr.Error(f"error : labels is empty")
        elif model_type in ["", " ", None]:
            raise gr.Error(f"error : model_type is empty")
        elif imgsz in ["", " ", None]:
            raise gr.Error(f"error : imgsz is empty")
        elif device in ["", " ", None]:
            raise gr.Error(f"error : device is empty")
        elif batch_size in ["", " ", None]:
            raise gr.Error(f"error : batch_size is empty")

        for each_device in device.split(","):
            if not str(each_device).isdigit():
                raise gr.Error(f"error : device is illegal : {device}")

        if continue_train == "true":
            continue_train = True
        else:
            continue_train = False

        info = {
                "model_name"    : model_name,
                "labels"        : labels,
                "continue_train": continue_train,
                "train_ucd"     : train_ds,
                "val_ucd"       : "None",
                "epochs"        : int(epoch),
                "model_type"    : model_type,
                "imgsz"         : int(imgsz),
                "batch_size"    : int(batch_size),
                "device"        : int(device),
                "hyp_dict"      : param_info
                }

        url = f"http://{now_agent_info['host']}:{now_agent_info['port']}/start_train"
        
        try:
            response = requests.post(url, headers={'Content-Type': 'application/json'}, json=info)
        except RequestException as e:
            log.error(f"failed connect to : {url} , {e}")
            raise gr.Error(f"failed connect to : {url} , {e}")
        except:
            log.error(f"failed start_train : {agent_name}")
            raise gr.Error(f"failed start_train : {agent_name}")

        check_agent_response(response, "message")
        log.info(f"* start train {agent_name} -> {model_name}, {response.text}")
        return response.text

    def stop_train(agent_name, model_name):
        global all_agent_info
        
        check_agent_name(agent_name)
        check_model_name(model_name)

        now_agent_info = all_agent_info[agent_name]
        url = f"http://{now_agent_info['host']}:{now_agent_info['port']}/stop_train/{model_name}"

        try:
            response = requests.get(url, headers={'Content-Type': 'application/json'})
        except RequestException as e:
            log.error(f"failed connect to : {url} , {e}")
            raise gr.Error(f"failed connect to : {url} , {e}")
        except:
            log.error(f"failed stop_train : {agent_name}")
            raise gr.Error(f"failed stop_train : {agent_name}")

        check_agent_response(response)
        log.info(f"* stop train {agent_name} -> {model_name}, {response.text}")
        return response.text

    def delete_train_cache(agent_name, model_name):
        global all_agent_info
        if agent_name is None:
            raise gr.Error("error : agent_name is empty")
        
        check_agent_name(agent_name)
        check_model_name(model_name)

        now_agent_info = all_agent_info[agent_name]
    
        url = f"http://{now_agent_info['port']}:{now_agent_info['host']}/clear_train_cache/{model_name}"

        try:
            response = requests.post(url, headers={'Content-Type': 'application/json'})
        except RequestException as e:
            log.error(f"failed connect to : {url} , {e}")
            raise gr.Error(f"failed connect to : {url} , {e}")
        except:
            log.error(f"failed delete_train_cache : {agent_name}")
            raise gr.Error(f"failed delete_train_cache : {agent_name}")

        check_agent_response(response)

        log.info(f"* delete train cache : {agent_name} -> {model_name}, {response.text}")
        return response.text

    def get_cache_model_name_in_assign_agent(agent_name):

        global all_agent_info

        check_agent_name(agent_name)

        now_agent_info = all_agent_info[agent_name]
    
        url = f"http://{now_agent_info['host']}:{now_agent_info['port']}/get_model_name_has_cahce"
             
        try:
            response = requests.get(url, headers={'Content-Type': 'application/json'})
        except RequestException as e:
            log.error(f"failed connect to : {url} , {e}")
            raise gr.Error(f"failed connect to : {url} , {e}")
        except:
            log.error(f"failed get gpu info : {agent_name}")
            raise gr.Error(f"failed get_cache_model_name_in_assign_agent : {agent_name}")

        check_agent_response(response, need_key="model_name_list")
        model_name_info = json.loads(response.text)
        model_name_list = model_name_info["model_name_list"]
        log.info(f"* get_cache_model_name_in_assign_agent : {model_name_list}")
        return gr.Dropdown(choices=model_name_list, interactive=True, value="")

    def get_train_model_name_in_assign_agent(agent_name):

        global all_agent_info

        check_agent_name(agent_name)
        now_agent_info = all_agent_info[agent_name]
    
        url = f"http://{now_agent_info['host']}:{now_agent_info['port']}/get_model_name_in_training" 
        
        try:
            response = requests.get(url, headers={'Content-Type': 'application/json'})
        except RequestException as e:
            log.error(f"failed connect to : {url} , {e}")
            raise gr.Error(f"failed connect to : {url} , {e}")
        except:
            log.error(f"failed get_train_model_name_in_assign_agent : {agent_name}")
            raise gr.Error(f"failed get_train_model_name_in_assign_agent : {agent_name}")

        check_agent_response(response, "model_name_list")
        model_name_list = json.loads(response.text)["model_name_list"]
        log.info(f"* get_train_model_name_in_assign_agent : {model_name_list}")
        return gr.Dropdown(choices=model_name_list, interactive=True, value="")

    def _get_train_info(agent_name, model_name, info_type):
        global all_agent_info

        check_agent_name(agent_name)
        check_model_name(model_name)

        now_agent_info = all_agent_info[agent_name]

        if info_type == "process":
            url = f"http://{now_agent_info['host']}:{now_agent_info['port']}/get_train_process/{model_name}"
        elif info_type == "param":  
            url = f"http://{now_agent_info['host']}:{now_agent_info['port']}/get_train_param/{model_name}"
        elif info_type == "detail":
            url = f"http://{now_agent_info['host']}:{now_agent_info['port']}/get_train_detail/{model_name}"
        else:
            raise gr.Error("info_type error, not in ['process', 'param', 'detail']", duration=5)
    
        response = requests.get(url, headers={'Content-Type': 'application/json'})
        check_agent_response(response, "info")

        response_info = json.loads(response.text)
        return '\n'.join(response_info["info"])

    def get_train_process(agent_name, model_name):
        check_agent_name(agent_name)
        check_model_name(model_name)
        log.info(f"* get model process info : {agent_name} -> {model_name}")
        return _get_train_info(agent_name, model_name, "process")

    def get_model_Param(agent_name, model_name):
        check_agent_name(agent_name)
        check_model_name(model_name)
        log.info(f"* get model param info : {agent_name} -> {model_name}")
        return _get_train_info(agent_name, model_name, "param")

    def get_model_train_detail(agent_name, model_name):
        check_agent_name(agent_name)
        check_model_name(model_name)
        log.info(f"* get model train detail info : {agent_name} -> {model_name}")
        return _get_train_info(agent_name, model_name, "detail")

    def update_dataset_info():

        try:
            response = requests.get(UCD_CHECK_URL, headers={'Content-Type': 'application/json'})
        except RequestException as e:
            log.error(f"failed connect to : {UCD_CHECK_URL} , {e}")
            raise gr.Error(f"failed connect to : {UCD_CHECK_URL} , {e}")
        except:
            log.error(f"failed update_dataset_info")
            raise gr.Error(f"failed update_dataset_info")

        dataset_name_list = json.loads(response.text)["customer"]
        dataset_name_list = [x.replace("\\", "/") for x in dataset_name_list]
        return gr.Dropdown(choices=dataset_name_list, interactive=True, value="")

    def update_gpu_info(agent_name):
        global all_agent_info

        check_agent_name(agent_name)
        now_agent_info = all_agent_info[agent_name]
    
        url = f"http://{now_agent_info['host']}:{now_agent_info['port']}/get_gpu_info"  # 替换为实际的接口 URL
        
        try:
            response = requests.get(url, headers={'Content-Type': 'application/json'})
        except RequestException as e:
            log.error(f"failed connect to : {url} , {e}")
            raise gr.Error(f"failed connect to : {url} , {e}")
        except:
            log.error(f"failed get gpu info : {agent_name}")
            raise gr.Error(f"failed get gpu info : {agent_name}")

        check_agent_response(response, "gpu_info")
        gpu_info = json.loads(response.text)["gpu_info"]
        device_list = []

        info = ""
        for each_gpu in gpu_info:
            device_list.append(str(each_gpu["index"]))
            info += str(each_gpu["index"]) + " : \n"
            info += f"  name         : {each_gpu['name']}\n"
            info += f"  memory_used : {each_gpu['memory_used']}\n"
            info += f"  memory_total : {each_gpu['memory_total']}\n"
        log.info(f"* update gpu info : {agent_name}, {info}")
        return gr.Dropdown(choices=device_list, interactive=True, value=""), info

    def update_after_start_train(agent_name, train_dataset_bt, model_name_bt, epoch_bt, labels_bt, model_type_dd, imgsz_bt, device_dd, batch_size_bt, continue_train_dd, param_file):

        param_info = {}
        if param_file:
            with open(param_file, 'r', encoding='utf-8') as file:
                param_info = yaml.safe_load(file)

        info = start_train(agent_name, train_dataset_bt, model_name_bt, epoch_bt, labels_bt, model_type_dd, imgsz_bt, device_dd, batch_size_bt, continue_train_dd, param_info)
        return gr.Dropdown(interactive=True, value=model_name_bt), info

    def load_tags_from_json(ucd_path):
        url = f"http://192.168.3.111:11101/ucd/check_assign_json/{ucd_path}"
        response = requests.get(url, headers={'Content-Type': 'application/json'})
        info = json.loads(response.text)
        tags            = []
        count_tags_info = info.get("count_tags_info", {})
        file_size       = info.get("size", "null")
        uc_count        = info.get("uc_count", "null")
        add_time        = info.get("add_time", "null")
        return_info  = f"add_time : {add_time}\nfile_size    : {file_size}\nuc_count : {uc_count}\ntags:\n"
        for each_tag in count_tags_info:
            return_info += f"    {each_tag} : {count_tags_info[each_tag]}\n"
            tags.append(each_tag)
        log.info(f"* load_tags_from_json : {ucd_path}")
        return return_info, ",".join(tags)

    def get_agent_info():
        global all_agent_info
        url = f"http://{LOCAL_HOST}:{REGIST_SERVER_PORT}/register_info"
        response = requests.get(url=url)

        check_agent_response(response, "agent_info")

        agent_info = json.loads(response.text)["agent_info"]
        log.info(f'* get_agent_info : {agent_info}')
        if len(agent_info) == 0:
            log.error(f"* No registered agent")
            raise gr.Error(message="No registered agent", duration=3, visible=True)
        else:
            agent_name_list = list(agent_info.keys())
            log.info(f"* agent name list : {agent_name_list}")
            all_agent_info = agent_info
            log.info(f"* get agent info : {agent_name_list}")
            return gr.Dropdown(choices=agent_name_list, label="agent", interactive=True, allow_custom_value=False)

    def select_agent(agent_name):
        global all_agent_info
        check_agent_name(agent_name)
        return_str = f'* host :  {all_agent_info[agent_name]["host"]}\n* port : {all_agent_info[agent_name]["port"]}\n* name : {all_agent_info[agent_name]["name"]}\n'
        
        now_agent_info = all_agent_info[agent_name]
        url = f"http://{now_agent_info['host']}:{now_agent_info['port']}/get_model_type"  # 替换为实际的接口 URL
        
        try:
            response = requests.get(url, headers={'Content-Type': 'application/json'})
        except RequestException as e:
            log.error(f"failed connect to : {url} , {e}")
            raise gr.Error(f"failed connect to : {url} , {e}")
        except:
            log.error(f"failed select_agent : {agent_name}")
            raise gr.Error(f"failed select_agent : {agent_name}")

        check_agent_response(response, "model_type_list")

        model_type_list = json.loads(response.text)["model_type_list"]
        log.info(f"* model type list {agent_name} -> {model_type_list}")

        if len(model_type_list) < 1:
            log.error(f"the length of model_type_list < 1 : {agent_name}")
            raise gr.Error(f"the length of model_type_list < 1 : {agent_name}")
        else:
            return_str += f"* support model type : \n"
            for each_model_type in model_type_list:
                return_str += f"         * {each_model_type}\n"

        return return_str, gr.Dropdown(choices=model_type_list, label="model type", allow_custom_value=False)

    update_model_name_bt.click(
        fn=get_cache_model_name_in_assign_agent,
        inputs=[agent_dd],
        outputs=[model_name_dd]
    )

    update_train_model_name_bt.click(
        fn=get_train_model_name_in_assign_agent,
        inputs=[agent_dd],
        outputs=[model_name_dd]
    )

    update_agent_bt.click(
        fn=get_agent_info,
        outputs=[agent_dd]
    )

    agent_dd.change(
        fn=select_agent,
        inputs=[agent_dd],
        outputs=[info_text, model_type_dd]
    )

    star_train_bt.click(
        inputs=[agent_dd, train_dataset_dd, model_name_bt, epoch_dd, labels_bt, model_type_dd, imgsz_bt, device_dd, batch_size_dd, continue_train_dd, param_file],
        fn=update_after_start_train,
        outputs=[model_name_dd, info_text]
    )

    stop_train_bt.click(
        fn=stop_train,
        inputs=[agent_dd, model_name_dd],
        outputs=[info_text]
    )

    gpu_info_bt.click(
        fn=update_gpu_info,
        inputs=[agent_dd],
        outputs=[device_dd, info_text]
    )

    update_dataset_bt.click(
        fn=update_dataset_info,
        outputs=[train_dataset_dd]
    )

    main_process_bt.click(
        fn=get_train_process,
        inputs=[agent_dd, model_name_dd],
        outputs=[info_text]
    )

    param_bt.click(
        fn=get_model_Param,
        inputs=[agent_dd, model_name_dd],
        outputs=[info_text]
    )

    detailed_log_bt.click(
        fn=get_model_train_detail,
        inputs=[agent_dd, model_name_dd],
        outputs=[info_text]
    )

    Load_models_bt.click(
        fn=download_model,
        inputs=[agent_dd, model_name_dd],
        outputs=[info_text]
    )

    train_dataset_dd.change(
        fn=load_tags_from_json,
        inputs=train_dataset_dd,
        outputs=[info_text, labels_bt]
    )

    register_agent_bt.click(
        fn=deregister_agent,
        inputs=[agent_dd],
        outputs=[info_text, agent_dd],
    )


if __name__ == "__main__":

    # TODO: 完善日志，所有的错误都记录下来，

    # FXIME: train_pre.sh 使用 python 进行编写，否则 错误无法清晰地看到
    # TODO: 将训练后的模型管理起来，管理在平台上，可以指定使用模型跑一个数据集的数据
    # TODO: 增加检测功能，指定数据集，指定模型，直接跑出一个有结果的数据集，管理在平台上
    # TODO: 管理平台的功能应该是一个单独的 docker 使用 http 和其他的服务进行交互，连数据集也管理在这个平台上
    # TODO: agent 增加自动化训练，自动纠错，自动管理出报告这些功能，当前的数据集管理过于就简单了
    # TODO: 模型管理平台会对接 svn，在 svn 上的是官方的模型，在 其他的地方的是自定义的模型集，
    # TODO: 因为每个人的权限很大了，所以一定使用密码才能删除这个功能要启动开发了，不然会非常的麻烦，没法解决误删的问题

    res = requests.post("http://192.168.3.69:11202/register_agent", json={"name": "yolo_test", "host": "192.168.3.69", "port": 11223})
    res = requests.post("http://192.168.3.69:11202/register_agent", json={"name": "other_test", "host": "192.168.3.50", "port": 60016})
    print(res.text)

    log.info(f"* start ui server {ENV_HOST}:{PORT_UI_SERVER}")
    demo.launch(server_name='0.0.0.0', server_port=PORT_UI_SERVER, share=False, debug=False)
