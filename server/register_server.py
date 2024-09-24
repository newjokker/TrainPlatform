
import os
from fastapi import FastAPI, HTTPException
from config import REGIST_SERVER_PORT, LOG_DIR
import uvicorn
from pydantic import BaseModel
from JoTools.utils.LogUtil import LogUtil

os.makedirs(LOG_DIR, exist_ok=True)


log_path = os.path.join(LOG_DIR, "register_server.log")
log = LogUtil.get_log(log_path, 5, "register_server", print_to_console=False)


app = FastAPI()


global all_register_info 
all_register_info = {}


class RegisterInfo(BaseModel):
    host:str
    port:int
    name:str


@app.post("/register_agent")
async def register_agent(register_info:RegisterInfo):
    """注册 agent"""
    global all_register_info

    if register_info.name in all_register_info:
        return {"status": "failed", "error_info": "agent name exists"}
    
    host_port = f"{register_info.host}:{register_info.port}"
    
    for each_agent_name in all_register_info:
        each_register_info = all_register_info[each_agent_name]
        each_host_port = f'{each_register_info["host"]}:{each_register_info["port"]}'
        if each_host_port == host_port:
            return {"status": "failed", "error_info": f"agent host:port exists : each_host_port"}

    all_register_info[register_info.name] = {"host": register_info.host, "port": register_info.port, "name": register_info.name}
    return {"status": "success"}

@app.get("/deregister_agent/{agent_name}")
async def deregister_agent(agent_name:str):

    global all_register_info

    if agent_name not in all_register_info:
        return {"status": "failed", "error_info": f"no agent named : {agent_name}"}

    del all_register_info[agent_name]
    return {"status": "success", "message": f"deregister agent success : {agent_name}"}

@app.get("/register_info")
async def register_agent():
    global all_register_info
    return {"status": "success", "agent_info": all_register_info}



if __name__ == "__main__":


    uvicorn.run(app, host="0.0.0.0", port=REGIST_SERVER_PORT)








