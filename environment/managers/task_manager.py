from utils import *
from logger import Logger
from .actions import *
from .event_manager import EventManager
from ..characters.character import Character


class Task:
    def __init__(
        self,
        name,
        desc,
        task_type="SubTask",
        character_dict: dict[str, Character] = None,
    ):
        self.name = name
        self.desc = desc
        self.task_type = task_type
        self.character_dict = character_dict
        self.is_finished = False

    def get_task_desc(self):
        return f"{self.name}: {self.desc}\nTask will be completed by {','.join(self.character_dict.keys())}."


class TaskManager:
    def __init__(
        self,
        task_name: str = None,
        instruction: str = None,
        all_character_dict: dict[str, Character] = None,
        event_manager: EventManager = None,
        task_decompose: bool = True,
        all_character: bool = False,
        engine: str = None,
        logger: Logger = None,
        task_context: dict = None
    ):
        # 接收用户指令，创建主任务
        self.all_character_dict = all_character_dict
        self.main_task = Task(task_name, instruction, "MainTask", all_character_dict)
        self.sub_tasks: list[Task] = []
        self.task_context = task_context if task_context else TASK_CONTEXT
        self.engine = engine
        self.logger = logger

        # 分解主任务并识别每个任务的成员，调整子任务优先级
        self.main_task_decomposition(task_decompose, all_character)
        self.set_task_context()

        # 注册监听器，和GroupManager进行交互
        self.event_manager = event_manager
        self.event_manager.subscribe("new_subtask", self.handle_new_subtask)
        self.event_manager.subscribe("task_completed", self.handle_task_completed)

        # 开始任务
        self.start_main_task()

    def main_task_decomposition(self, task_decompose, all_character):
        self.logger.gprint(
            "System INFO",
            message=f"Main task start!\n{self.main_task.name}:\n{self.main_task.desc}",
        )
        if task_decompose:
            # 收到用户指令后分解主任务并识别每个子任务的成员
            subtasks_and_characters = run_main_task_decompose(
                f"{self.main_task.name}:\n{self.main_task.desc}",
                self.all_character_dict,
                self.engine,
                self.logger,
            )
            subtasks: list[Task] = []
            for subtask_name, subtask_desc, character_ids in subtasks_and_characters:
                character_dict = {
                    idx: self.all_character_dict[idx] for idx in character_ids
                }
                if all_character:
                    subtasks.append(
                        Task(subtask_name, subtask_desc, "SubTask", self.all_character_dict)
                    )
                else:
                    subtasks.append(Task(subtask_name, subtask_desc, "SubTask", character_dict))
                self.logger.gprint(
                    "System INFO",
                    message=f"New Subtask created! {subtask_name}: {subtask_desc}. This task will be solved by {','.join(character_ids)}.",
                )
        else:  # 不进行任务分解
            subtasks = [
                Task(
                    self.main_task.name,
                    self.main_task.desc,
                    "SubTask",
                    self.main_task.character_dict,
                )
            ]
        # 添加子任务
        self.sub_tasks.extend(subtasks)

    def set_task_context(self):
        self.task_context["main_task_desc"] = (
            f"{self.main_task.name}:\n{self.main_task.desc}"
        )
        self.task_context["all_subtask_desc"] = "\n".join(
            [
                f"{idx + 1}. {subtask.name}: {subtask.desc}"
                for idx, subtask in enumerate(self.sub_tasks)
            ]
        )
        self.task_context["all_character_desc"] = "\n".join(
            [
                f"{idx + 1}. {character.id}: {character.scratch}"
                for idx, character in enumerate(self.all_character_dict.values())
            ]
        )

    def add_subtask(self, subtask: Task):
        """
        Add new subtasks and adjust priorities.
        """
        finished_tasks = [task for task in self.sub_tasks if task.is_finished]
        unfinished_tasks = [task for task in self.sub_tasks if not task.is_finished]
        ongoing_task, todo_task = unfinished_tasks[0], unfinished_tasks[1:]
        self.sub_tasks = finished_tasks + [ongoing_task] + [subtask] + todo_task
        self.set_task_context()
        # self.sub_tasks.append(subtask)
        # self.adjust_subtask_priorities()

    def adjust_subtask_priorities(self):
        priorities: list[int] = run_task_prioritize(
            self.main_task, self.sub_tasks, self.engine, self.logger
        )
        self.sub_tasks = [
            self.sub_tasks[i - 1] for i in priorities
        ]  # prioritized subtasks for parent_task, idx从1开始
        # print log
        new_priorities_desc = "\n".join(
            [
                f"{idx + 1}. {subtask.name}: {subtask.desc}"
                for idx, subtask in enumerate(self.sub_tasks)
            ]
        )
        self.logger.gprint(
            "System INFO",
            message=f"Subtask priorities adjusted! New priorities:\n {new_priorities_desc}",
        )

    def start_main_task(self):
        # 新建主群聊
        main_group_id = MAIN_GROUP_ID
        self.event_manager.publish(
            "create_group",
            {
                "group_id": main_group_id,
                "members": self.all_character_dict,
                "context": self.task_context,
            },
        )
        self.start_next_task()

    def get_next_task(self):
        for subtask in self.sub_tasks:
            if not subtask.is_finished:
                return subtask
        # if self.sub_tasks:
        #     return self.sub_tasks[0]
        return None

    def start_next_task(self):
        next_task = self.get_next_task()
        if not next_task:
            self.finish_main_task()
            return
        self.logger.gprint(
            "System INFO", message=f"Subtask start! {next_task.name}: {next_task.desc}"
        )

        # 设置任务上下文，创建子群聊
        self.set_task_context()
        self.task_context["curr_task_desc"] = next_task.get_task_desc()
        self.task_context["curr_group_actor_ids"] = ",".join(
            next_task.character_dict.keys()
        )
        self.task_context["all_curr_group_members"] = next_task.character_dict
        subgroup_id = f"Group_{next_task.name}"
        self.event_manager.publish(
            "create_group",
            {
                "group_id": subgroup_id,
                "members": next_task.character_dict,
                "context": self.task_context,
            },
        )
        self.event_manager.publish(
            "start_group_chat", {"group_id": subgroup_id, "context": self.task_context}
        )

    def handle_new_subtask(self, data):
        task_desc = data["task_desc"]
        is_new_task, task = run_process_new_task(
            self.task_context["main_task_desc"],
            self.task_context["all_subtask_desc"],
            self.task_context["all_character_desc"],
            task_desc,
            self.engine,
            self.logger,
        )
        if is_new_task:
            character_dict = {idx: self.all_character_dict[idx] for idx in task[2]}
            new_task = Task(task[0], task[1], "SubTask", character_dict)
            self.add_subtask(new_task)

    def get_subtask(self, task_desc) -> Task:
        # 找到指定的subtask
        res = None
        for subtask in self.sub_tasks:
            if subtask.get_task_desc() == task_desc:
                res = subtask
                break
        return res

    def handle_task_completed(self, data):
        task_desc = data["subtask_desc"]
        curr_subtask = self.get_subtask(task_desc)
        # # 删除子任务
        # self.sub_tasks.remove(curr_subtask)
        curr_subtask.is_finished = True
        self.start_next_task()

    def finish_main_task(self):
        # TODO: 在这里加结算逻辑
        self.logger.gprint("System INFO", message=f"All task completed!")
        pass
