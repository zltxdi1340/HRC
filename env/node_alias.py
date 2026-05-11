CANONICAL_NODES = [
    "near_key",
    "has_key",
    "near_target",
    "has_target",
    "opened_box",
    "opened_door",
    "has_box",
    "at_goal",
    "task_success",
]

OLD_TO_NEW = {
    "near_obj": "near_target",
    "has_obj": "has_target",
    "open_box": "opened_box",
    "door_open": "opened_door",
    "success": "task_success",
}

NEW_TO_OLD = {v: k for k, v in OLD_TO_NEW.items()}


def to_new(name: str) -> str:
    return OLD_TO_NEW.get(name, name)


def to_old(name: str) -> str:
    return NEW_TO_OLD.get(name, name)


def normalize_nodes(nodes):
    return [to_new(x) for x in nodes]