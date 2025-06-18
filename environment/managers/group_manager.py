from logger import Logger
from config import *
from utils import *
from .event_manager import EventManager
from .actions import run_summarize_group_message
from ..characters.character import Character, TaskCharacter


class GroupMessage:
    def __init__(
        self, sender_id: str, receiver_id: str, message, is_all: bool, logger: Logger
    ):
        self.sender_id = sender_id if sender_id else "System"
        self.receiver_id = receiver_id if not is_all else "All"
        self.message = message
        self.is_all = is_all  # to all character
        self.logger = logger
        self.logger.gprint(
            "Group INFO", message=f"New group message: {self.get_messages_desc()}"
        )

    def get_messages_desc(self):
        message = f"{self.sender_id} say to {self.receiver_id}: {self.message}"
        return message


class Group:
    def __init__(
        self, group_id: str, members: dict[str, Character], task_context: dict
    ):
        self.id = group_id
        self.members = members
        self.task_context = task_context
        self.task_context["all_curr_group_member_desc"] = (
            self.get_group_member_description()
        )
        self.messages: list[GroupMessage] = []

    def get_group_member_description(self):
        return "\n".join(
            [
                f"{character.id}: {character.scratch}"
                for character in self.members.values()
            ]
        )


class GroupManager:
    def __init__(
        self, event_manager: EventManager = None, engine=None, logger: Logger = None
    ):
        self.groups: dict[str, Group] = {}  # {group_id: Group}
        self.event_manager = event_manager
        self.event_manager.subscribe("create_group", self.handle_create_group)
        self.event_manager.subscribe("start_group_chat", self.handle_start_group_chat)
        self.engine = engine
        self.logger = logger

    def handle_create_group(self, data):
        group_id, members, context = data["group_id"], data["members"], data["context"]
        self.groups[group_id] = Group(group_id, members, context)
        self.logger.gprint(
            "System INFO", message=f"Group with ID '{group_id}' has been created."
        )

    def handle_start_group_chat(self, data):
        group_id, context = data["group_id"], data["context"]
        group = self.groups[group_id]
        group.task_context = context

        # 按TaskManager安排的可行动角色和顺序行动
        actor_ids = group.task_context["curr_group_actor_ids"].split(",")
        # 最多行动MAX_ACTION_NUM轮
        for turn in range(MAX_ACTION_TURN):
            group.task_context["max_act_turn"] = MAX_ACTION_TURN
            group.task_context["curr_act_turn"] = turn + 1
            self.logger.gprint(
                "System INFO",
                message=f"Group with ID '{group_id}' is chatting in turn {turn + 1}/{MAX_ACTION_TURN}",
            )
            # 子任务只需要一个人完成
            if len(actor_ids) == 1:
                actor = group.members[actor_ids[0]]
                actor.set_task_context(group.task_context)
                environment_desc = actor.perceive()
                action_name, target_id = "GroupChat", "All"
                chat_history = actor.execute_action(
                    environment_desc, action_name, target_id
                )
                self.execute_update(action_name, group, chat_history)
            else:  # 群成员按顺序轮流行动
                for actor_id in actor_ids:
                    actor = group.members[actor_id]
                    actor.set_task_context(group.task_context)
                    environment_desc = actor.perceive()
                    action_name, target_id = actor.decide_action(environment_desc)
                    self.logger.gprint(
                        "System INFO",
                        message=f"{actor.id} decided to {action_name} chat with {target_id}.",
                    )
                    if action_name == "Skip":
                        continue
                    # 执行函数
                    chat_history = actor.execute_action(
                        environment_desc, action_name, target_id
                    )
                    # 更新记忆和群消息
                    self.execute_update(action_name, group, chat_history)

            if isinstance(list(group.members.values())[0], TaskCharacter):
                # 每轮群聊结束后，EventManager判断子任务是否提前完成/有新的子任务
                is_completed = self.event_manager.check_messages(
                    {
                        "group_id": group_id,
                        "messages": group.messages,
                        "context": group.task_context,
                    }
                )
                if is_completed:
                    self.logger.gprint(
                        "System INFO",
                        message=f"Subtask `{group.task_context['curr_task_desc']}` has completed！",
                    )
                    break

        # 结束群聊
        self.finish_group_chat(group)
        # 告诉TaskManager子任务已完成
        self.event_manager.publish(
            "task_completed", {"subtask_desc": group.task_context["curr_task_desc"]}
        )

    def execute_update(self, chat_type, group, chat_history):
        kwargs = {"group": group, "chat_history": chat_history}
        update_map = {
            "Private": self.update_after_private,
            "Meeting": self.update_after_meeting,
            "GroupChat": self.update_after_groupchat,
        }
        update_map[chat_type](**kwargs)

    def update_after_groupchat(self, group: Group, chat_history):
        for message in chat_history:
            sender_id, receiver_id, message_content = message[0], message[1], message[2]
            is_all = True if receiver_id == "All" else False
            group.messages.append(
                GroupMessage(
                    sender_id, receiver_id, message_content, is_all, self.logger
                )
            )
            for member in group.members.values():
                member.memory.add_memory(
                    member.id,
                    group.messages[-1].get_messages_desc(),
                    memory_type="short",
                )

    def update_after_meeting(self, group: Group, chat_history):
        # 通知群友有人秘密会晤了
        actor, target = (
            group.members[chat_history[0][0]],
            group.members[chat_history[0][1]],
        )
        group_message = f"I have a confidential meeting with {target.id}."
        group.messages.append(
            GroupMessage(actor.id, "All", group_message, True, self.logger)
        )
        for member in group.members.values():
            member.memory.add_memory(member.id, group_message, memory_type="short")
        # 只更新双方的记忆，不更新群消息
        for message in chat_history:
            sender = group.members[message[0]]
            receiver = group.members[message[1]]
            sender.memory.add_memory(sender.id, message[2], "short")
            receiver.memory.add_memory(receiver.id, message[2], "short")

    def update_after_private(self, group: Group, chat_history):
        # 只更新双方的记忆
        for message in chat_history:
            sender = group.members[message[0]]
            receiver = group.members[message[1]]
            sender.memory.add_memory(sender.id, message[2], "short")
            receiver.memory.add_memory(receiver.id, message[2], "short")

    def finish_group_chat(self, group: Group):
        # 总结子群聊消息，添加到main group消息中，并同步到群成员的long memory
        group_summary = self.summarize_subgroup_messages(group)
        self.groups[MAIN_GROUP_ID].messages.append(
            GroupMessage("System", "All", group_summary, True, self.logger)
        )

        for member in group.members.values():
            member.memory.add_memory(member.id, group_summary, memory_type="long")

        # TODO: 社会模拟类任务中，角色需要从自己视角对short memory进行摘要和反思，并添加到long memory
        # write something here

        # 清除short memory
        for member in group.members.values():
            member.memory.clear_short_term_memory()

        # 销毁子群聊
        self.logger.gprint(
            "System INFO", message=f"Group with ID '{group.id}' has been destroyed."
        )
        self.groups.pop(group.id)

    def summarize_subgroup_messages(self, group: Group) -> str:
        return run_summarize_group_message(
            group.task_context, group.messages, self.engine, self.logger
        )
