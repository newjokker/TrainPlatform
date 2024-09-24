import os



# log
LOG_DIR                 = "/usr/src/app/logs"


# server
LOCAL_HOST              = "127.0.0.1"
PORT_UI_SERVER          = 11201
REGIST_SERVER_PORT      = 11202


UCD_CHECK_URL           = f"http://192.168.3.111:11101/ucd/check"



ENV_HOST                = os.environ.get("HOST_IP", "ENV_HOST")





