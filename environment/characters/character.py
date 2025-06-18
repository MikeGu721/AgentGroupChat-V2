import os
import json
from logger import Logger
from .actions import *


class Memory:
    def __init__(self, logger: Logger):
        self.long_memory: list[str] = (
            []
        )  # 子任务完成后，对short memory的总结、reflection、子任务完成结果
        self.short_memory: list[str] = []  # 当前子任务相关的记忆
        self.logger = logger

    def get_memory_desc(self, memory_type="short"):
        memory = self.short_memory if memory_type == "short" else self.long_memory
        return "\n".join([f"{idx + 1}. {memory}" for idx, memory in enumerate(memory)])

    def add_memory(self, character_id, content, memory_type="short"):
        memory = self.short_memory if memory_type == "short" else self.long_memory
        memory.append(content)
        self.logger.gprint(
            "Memory INFO",
            message=f"{character_id}'{memory_type} has updated: {content}",
        )

    def retrieve_memory(self, memory_type="short"):
        # TODO: 检索逻辑
        memory = self.short_memory if memory_type == "short" else self.long_memory
        return memory

    def clear_short_term_memory(self):
        self.short_memory = []


class Character:
    def __init__(self, persona_file=None, engine=None, logger: Logger = None):
        self.id: str = ""
        self.scratch: str = ""
        self.objective: str = ""
        self.message_format_desc: str = ""
        self.message_format_field: str = ""
        self.memory = Memory(logger)
        self.action_map = None
        self.engine = engine
        self.logger = logger
        self.load_persona(persona_file)
        self.task_context: dict = TASK_CONTEXT

    def load_persona(self, persona_file=None):
        if not persona_file:
            return
        with open(persona_file, encoding="utf-8") as f:
            persona: dict = json.load(f)
        # dynamically setting attributes
        for key, value in persona.items():
            setattr(self, key, value)

    def get_self_description(self):
        return self.scratch

    def set_task_context(self, context: dict):
        self.task_context = context

    def perceive(self):
        environment_desc = run_perceive(
            self, self.task_context, self.engine, self.logger
        )
        self.memory.add_memory(
            self.id,
            f"Your understanding of the current environment: {environment_desc}",
            memory_type="short",
        )
        return environment_desc

    def decide_action(self, environment_desc):
        action_name = run_decide_action(
            self,
            environment_desc,
            self.task_context,
            self.action_map,
            self.engine,
            self.logger,
        )

        target_id = run_decide_target(
            self,
            environment_desc,
            self.task_context,
            self.action_map,
            action_name,
            self.engine,
            self.logger,
        )
        return action_name, target_id

    def execute_action(self, environment_desc, action_name, target_id):
        kwargs = self.action_map[action_name]["args"]
        kwargs["environment_desc"] = environment_desc
        kwargs["target_id"] = target_id
        action_func = self.action_map[action_name]["func"]
        return action_func(**kwargs)

    def execute_code(self, message: str, test_setup_code: str = "") -> str:
        """
        代码生成任务，编译代码作为反馈
        """
        # parse message
        message = message.split("### Answer:")[-1].strip()
        func_code = extract_code(message)

        namespace = {}
        # First execute setup code if any
        if test_setup_code:
            try:
                exec(test_setup_code, namespace)
            except Exception as e:
                return f"Setup code error: {str(e)}"
        # Then execute function code
        try:
            prepare_test_environment(namespace)
            exec(func_code, namespace)
        except Exception as e:
            return f"Code execution error: {str(e)}"

        return "Code compiled successfully."

    def chat(self, environment_desc, chat_type, target_id):
        """
        chat_type: private/meeting/groupchat, action_module决定
        角色对target发起聊天；
        如果是私聊或会晤，被发起聊天者自己决定是否要回复消息。
        一次聊天最多进行MAX_CHAT_TURN轮。
        """
        turn, chat_history = 1, []
        # character发起聊天
        message = run_chat(
            self,
            environment_desc,
            self.task_context,
            chat_type,
            target_id,
            self.engine,
            self.logger,
        )

        # 如果是代码生成任务，加上编译反馈
        if "programming" in self.task_context["main_task_desc"]:
            compile_res = self.execute_code(
                message, self.task_context["test_setup_code"]
            )
            message = f"{message}\n Code compilation result: {compile_res}"

        chat_history.append((self.id, target_id, message))
        self.logger.gprint(
            "Chat INFO", message=f"{self.id} say to {target_id}: {message}"
        )
        # 如果是对所有人说，直接由GroupManager广播消息，不需要其他人回复
        if target_id == "All":
            return chat_history
        # 如果是对某人说，则target自行判断是否需要回复
        target: Character = self.task_context["all_curr_group_members"][target_id]
        responder, sender = target, self
        while turn < MAX_CHAT_TURN * 2:
            message = run_respond(
                responder,
                sender,
                self.task_context,
                chat_type,
                chat_history,
                responder.engine,
                self.logger,
            )
            if not message:
                return chat_history

            # 如果是代码生成任务，加上编译反馈
            if "programming" in self.task_context["main_task_desc"]:
                compile_res = self.execute_code(
                    message, self.task_context["test_setup_code"]
                )
                message = f"{message}\n Code compilation result: {compile_res}"

            chat_history.append((responder.id, sender.id, message))
            self.logger.gprint(
                "Chat INFO", message=f"{responder.id} say to {sender.id}: {message}"
            )
            responder, sender = self, target
            turn += 1

        return chat_history

    def reflect(self):
        # TODO: 只reflect，无update
        pass

    def summarize(self, content: str):
        # TODO: summarize short memory
        pass


class TaskCharacter(Character):
    def __init__(self, persona_file, engine, logger=None):
        super().__init__(persona_file, engine=engine, logger=logger)
        self.set_action_map()

    def set_action_map(self):
        # 任务解决默认group discussion
        self.action_map = {
            "Skip": {
                "desc": ALL_CHAT_TYPES["Skip"]["desc"],
                "func": None,
                "args": None,
            },
            "GroupChat": {
                "desc": ALL_CHAT_TYPES["GroupChat"]["desc"],
                "func": self.chat,
                "args": {"chat_type": "GroupChat"},
            },
        }


class SocialCharacter(Character):
    def __init__(self, persona_file, engine, logger=None):
        super().__init__(persona_file, engine=engine, logger=logger)
        self.support_character_id = ""
        self.is_main_character = False
        self.beliefs = {}
        self.load_persona(persona_file)
        self.set_action_map()

    def set_action_map(self):
        self.action_map = {
            "Skip": {
                "desc": ALL_CHAT_TYPES["Skip"]["desc"],
                "func": None,
                "args": None,
            },
            "Private": {
                "desc": ALL_CHAT_TYPES["Private"]["desc"],
                "func": self.chat,
                "args": {"chat_type": "Private"},
            },
            "Meeting": {
                "desc": ALL_CHAT_TYPES["Meeting"]["desc"],
                "func": self.chat,
                "args": {"chat_type": "Meeting"},
            },
            "GroupChat": {
                "desc": ALL_CHAT_TYPES["GroupChat"]["desc"],
                "func": self.chat,
                "args": {"chat_type": "GroupChat"},
            },
        }

    def save_persona(self, save_dir):
        """
        Save agent's persona to save_dir.
        """
        if not os.path.exists(save_dir):
            os.makedirs(save_dir)
        save_path = os.path.join(save_dir, f"{self.id}.json")

        persona_dict = self.__dict__.copy()  # get all attributes

        with open(save_path, "w", encoding="utf-8") as f:
            json.dump(persona_dict, f, ensure_ascii=False, indent=4)

    def get_self_description(self):
        description = [
            f"You: {self.id}.",
            f"Your character setting: {self.scratch}.",
            f"Your goal: {self.objective}.",
            (
                f"You are supporting {self.support_character_id} in achieving their goals."
                if self.support_character_id
                else "You are not supporting anyone at the moment."
            ),
        ]
        # current belief desc of main characters
        if self.is_main_character:
            description.append(f"Your current belief: {self.get_current_belief()}")

        return "\n".join(description).strip()

    def get_current_belief(self):
        """
        Get beliefs with the highest score. There may be multiple beliefs with the same score.
        """
        max_score = max(self.beliefs.values())
        main_beliefs = [
            belief for belief, score in self.beliefs.items() if score == max_score
        ]

        return "; ".join(main_beliefs) + "."

    def reflect(self):
        # TODO: reflect and update
        super().reflect()
        pass

    def vote(self):
        # TODO: 所有subtask完成后，在主任务结束前进行
        pass
