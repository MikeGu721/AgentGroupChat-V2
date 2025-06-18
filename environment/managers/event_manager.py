from logger import Logger
from config import *
from .actions import run_check_message


class EventManager:
    def __init__(self, engine, logger: Logger = None):
        self.listeners: dict[str, callable] = {}  # 存储每种事件的监听器函数
        self.engine = engine
        self.logger = logger

    def subscribe(self, event_name: str, listener: callable):
        """
        订阅事件，在事件发生时会收到通知，调用对应监听器。

        Args:
        - event_name (str): 事件名称
        - listener (callable): 监听器函数
        """
        self.listeners[event_name] = listener

    def publish(self, event_name: str, data: dict):
        """
        发布事件，发布的事件会触发对应的监听器

        Args:
        - event_name (str): 事件名称
        - data (dict): 事件相关数据，作为对应监听器函数的参数
        """
        if event_name in self.listeners:
            self.listeners[event_name](data)
        else:
            print(f"[Error]: No Listener Registered for Event '{event_name}'.")

    def check_messages(self, data):
        group_id = data["group_id"]
        group_messages = data["messages"]
        task_context = data["context"]
        group_message_desc = "\n".join(
            [message.get_messages_desc() for message in group_messages]
        )
        self.logger.gprint(
            "System INFO",
            message=f"EventManager checking messages for group '{group_id}'.",
        )
        # 任务完成/新任务检查
        is_completed, if_new_task, new_task = run_check_message(
            task_context["main_task_desc"],
            task_context["all_subtask_desc"],
            task_context["curr_task_desc"],
            group_message_desc,
            self.engine,
            self.logger,
        )
        # TODO: 发现新的子任务，通知TaskManager
        # if if_new_task:
        #     self.publish("new_subtask", {"task_desc": new_task})

        return is_completed
