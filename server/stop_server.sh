#!/bin/bash

# Function to recursively terminate all child processes
terminate_children() {
    local parent_pid=$1
    local child_pids=$(pgrep -P "$parent_pid")

    if [ -n "$child_pids" ]; then
        for child_pid in $child_pids; do
            terminate_children "$child_pid"
            kill "$child_pid" &> /dev/null
        done
    fi
}

# Main script

supervisorpid="/var/run/supervisord.pid"

if [ ! -f "$supervisorpid" ]; then
  echo "PID file not exists : $supervisorpid"

  if [ $# -eq 0 ]; then
    echo "No PID provided"
    exit 1
  fi
  pid=$1
else
  pid=$(cat "$supervisorpid")
fi


# FIXME 增加发送关闭请求

curl "http://192.168.3.50:6006/terminate"

echo "supervisord pid : $pid"
terminate_children "$pid"
kill "$pid" &> /dev/null

ps -ef | grep python3

