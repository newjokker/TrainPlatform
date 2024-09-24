
import os
import time
import psutil
import subprocess
import signal
import errno
import uuid


def sigchld_handler(signum, frame):
    """
    信号处理函数，用于处理 SIGCHLD 信号。
    当子进程终止时，这个函数会被调用。
    """
    # 无限循环直到没有更多的子进程需要等待
    try:
        while True:
            pid, status = os.waitpid(-1, os.WNOHANG)
            # FIXME 这边能获取结束的子进程的 pid 和 return_code
            print(pid, status)
            if pid == 0:
                break
    except OSError as e:
        if e.errno == errno.ECHILD:
            print('* current process has no existing unwaited-for child processes.')
        else:
            raise

# # 注册信号处理函数
signal.signal(signal.SIGCHLD, sigchld_handler)


class JoProcess(object):


    def __init__(self, commant_list, log_dir="/var/log/JoProcess"):

        os.makedirs(log_dir, exist_ok=True)
        # FIXME 还是无法防止变为僵尸进程
        self.log_std_path       = os.path.join(log_dir, str(uuid.uuid1())) + "_std.txt"
        self.log_err_path       = os.path.join(log_dir, str(uuid.uuid1())) + "_err.txt"
        self.process            = subprocess.Popen(commant_list, preexec_fn=os.setsid,
                                           stdout=open(self.log_std_path, 'w'),
                                           stderr=open(self.log_err_path, 'w'))
        self.start_time = time.time()
        self.pid        = self.process.pid
        process_info    = psutil.Process(self.process.pid)
        
        # # 下面这些属性都是抄的 psutil 的
        self.name       = process_info.name()
        self.exe        = process_info.exe()
        self.ppid       = process_info.ppid()
        # self.status     = process_info.status()
        self.cwd        = process_info.cwd()
        self.cmdline    = process_info.cmdline()
        # print("* finished init")

    def status(self):
        # FIXME 如果是退出的话获取退出码，当使用了信号处理了退出信息之后就不好获取退出码了
        status = self.process.poll()
        if status is None:
            return "running", status
        elif status == 0:
            return "finished", status
        else:
            return "error_stop", status

    @staticmethod
    def _get_process_start_time(pid):
        """获取任意指定 PID 进程的开始时间"""
        try:
            p = psutil.Process(pid)
            start_time = p.create_time()
            return start_time
        except psutil.NoSuchProcess:
            return None

    def is_same_process(self, assign_th=1):
        """查看占用当前 PID 的进程和 init 启动时候使用的进程是不是同一个，通过判断启动时间是否一致（会不会存在启动花很长时间的情况啊？）"""
        now_start_time = JoProcess._get_process_start_time(self.pid)

        if now_start_time is None:
            raise ValueError("* process is killed")

        if abs(self.start_time - now_start_time) < assign_th:
            return True
        else:
            return False 

    def get_cpu_percent(self):
        """每一步都要核对一下，这个进程还是不是我之前指定的进程"""
        assert self.is_same_process(), "* process is killed"
        process_info = psutil.Process(self.pid)
        try:
            cpu_usage = process_info.cpu_percent(interval=1)  # interval 参数表示采样间隔
            return cpu_usage
        except psutil.NoSuchProcess:
            # raise ValueError("* process is killed")
            return 0

    def get_memory_info(self):
        """获取内存的使用单位是字节"""
        assert self.is_same_process(), "* process is killed"
        process_info = psutil.Process(self.pid)
        try:
            mem_info = process_info.memory_info()
            rss = mem_info.rss
            vms = mem_info.vms
            return rss, vms
        except psutil.NoSuchProcess:
            # raise ValueError("* process is killed")
            return 0, 0
    
    def terminal(self):
        """获取进程的终端信息"""
        with open(self.log_std_path, 'r') as file:
            content = file.read()
        return content

    def kill_process(self, sig=signal.SIGTERM):
        """杀掉当前进程"""
        # 信号类型: 默认情况下，我们使用 SIGTERM 信号来请求进程优雅地终止。如果进程不响应 SIGTERM，可以使用更强力的信号，如 SIGKILL，但请注意 SIGKILL 信号不能被捕获或忽略，这可能导致进程立即终止而无法进行清理操作
        try:
            os.kill(self.pid, sig)
            print(f"* Sent signal {sig} to process {self.pid}")
        except ProcessLookupError:
            print(f"* Process {self.pid} does not exist")
        except PermissionError:
            print(f"* Permission denied when trying to kill process {self.pid}")
        except Exception as e:
            print(f"* Failed to kill process {self.pid}: {e}")

    def __del__(self):
        os.remove(self.log_std_path)
        os.remove(self.log_err_path)

        if os.path.exists(self.log_err_path):
            print(f"* delete file failed : {self.log_err_path}")

        if os.path.exists(self.log_std_path):
            print(f"* delete file failed : {self.log_std_path}")


if __name__ == "__main__":

    """
    * 有时间分享一下僵尸进程
    """


    # FIXME 加了信号处理之后就不能判断是不是正常退出的了，只能根据 redis 中的信息去判断


    a = JoProcess(["python", "print.py"])

    print(a.status())

    print(a.pid)

    time.sleep(2)
    # res = os.kill(a.pid, signal.SIGTERM)
    print(a.terminal())

    print(a.get_memory_info())
    print(a.get_cpu_percent())

    for i in range(10):
        print(a.status())
        time.sleep(1)

    print(a.terminal())

    # a.kill_process()

    for i in range(10):
        print(a.status())
        time.sleep(1)




    print("* finished")












