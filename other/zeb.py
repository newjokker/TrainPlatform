import os
import signal
import subprocess
import time
import errno

def sigchld_handler(signum, frame):
    """
    信号处理函数，用于处理 SIGCHLD 信号。
    当子进程终止时，这个函数会被调用。
    """
    # 无限循环直到没有更多的子进程需要等待
    try:
        while True:
            # 调用 os.waitpid 来收集子进程的状态信息
            pid, status = os.waitpid(-1, os.WNOHANG)
            if pid == 0:
                break  # 没有更多的子进程需要等待

    except OSError as e:
        if e.errno == errno.ECHILD:
            print('current process has no existing unwaited-for child processes.')
            time.sleep(1)
        else:
            raise

# 注册信号处理函数
signal.signal(signal.SIGCHLD, sigchld_handler)

# 创建子进程
child_process = subprocess.Popen(["sleep", "5"])

# 父进程做一些其他事情
time.sleep(10)

# 检查子进程是否已经终止
if child_process.poll() is None:
    print("Child process is still running.")
else:
    print("Child process has terminated.")

# 等待一段时间，让信号处理函数有机会被调用
time.sleep(1)

# 检查子进程是否已经被清理
if child_process.poll() is None:
    print("Child process is still running.")
else:
    print("Child process has been cleaned up.")