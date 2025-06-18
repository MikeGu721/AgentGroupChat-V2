import json
from datetime import datetime
import os.path
from config import *


class Logger:
    def __init__(self, log_dir):
        '''
        初始化日志类
        Args:
            log_dir: str, 日志存放根目录
        '''
        self.log_count = 0
        self.log_dir = log_dir
        if not os.path.exists(self.log_dir): os.makedirs(self.log_dir)

        now = datetime.now()
        self.identifier = now.strftime("%Y%m%d_%H%M%S_%f")
        self.log_file = os.path.join(self.log_dir, f"{self.identifier}.json")
        self.log_fw = open(self.log_file, 'a', encoding='utf-8')

    def gprint(self, *args, **kwargs):
        '''
        打印日志，同时保存打印内容
        '''
        json_data = {"id": self.log_count,
                     "time": str(datetime.now()),
                     "args": " ".join([str(arg) for arg in args]),
                     "kwargs": json.dumps(kwargs, ensure_ascii=False)}
        self.log_fw.write(json.dumps(json_data, ensure_ascii=False) + "\n")
        self.log_count += 1
        if DEBUG:
            print(json_data)
