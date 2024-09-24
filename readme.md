# 说明

### 镜像启动

#### server

* docker run --name auto_train_server -p 7869:7869 -v /home/ldq/Code/AutoYoloTrainServer:/usr/code -d auto_yolo_train_server:v0.0.1

* sudo docker rm -f auto_train_server

* sudo docker exec -it auto_train_server /bin/bash 

* sudo docker run -p 11202:11202 -p 11201:11201 --name auto_train_server  -v /home/ldq/Code/AutoYoloTrainServer:/usr/code -it auto_yolo_train_server:v0.0.1 /bin/bash

#### agent

* 需要指定服务端的 ip 和 端口，主动去注册，如果无法注册的话 这个镜像应该主动报错是失败的

* docker run --gpus '"device=0,1,2"' -e HOST_IP=192.168.3.50 -v /etc/localtime:/etc/localtime:ro --shm-size=8g -p 60067:60067 -p 60066:60066 -p 7869:7869 -v /home/suanfa-2/ucd_cache:/usr/ucd_cache -v /home/suanfa-2/ldq/YoloTrainServer/runs:/usr/src/app/runs -d auto_yolo_train_server:v1.2.8 


### 镜像打包

#### server 镜像

* cd server

* docker build -t auto_yolo_train_server:v0.0.1 -f Dockerfile_server . 

#### agent 镜像

* cd agent

* docker build -t auto_yolo_train_agent:version -f Dockerfile_agent . 


### 版本说明

* v1.0.x  完成第一个单模型训练版本
* v1.1.x  多模型训练版本
* v1.2.x  增加可视化界面
* v1.3.x  将服务端和客户端分开来开发，别人实现接口训练自己的代码

### 注意事项

* 指定共享内存的大小，默认 64M 非常小，不指定的话 worker 设置会很小速度非常慢， --shm-size=8g


### docker 启动命令

* -e HOST_IP=192.168.3.50  容器内部不能直接知道当前所在的宿主机的 IP 需要传入

* -v /etc/localtime:/etc/localtime:ro  容器内部的时间和外部的时间一致

* --shm-size=8g 使用共享内内存的大小

### TODO

* 将代码分为两个部分，一个是 server 一个是 agent , server 只是负责调用， agent 负责启动和绑定对应的 server ， agent 负责查看 对应的 agent 的目前的信息

* 写好通用的接口之后，我写一个 demo 让别人自己去对应着实现自己想做的功能即可

















