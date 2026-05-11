# env/subgoals.py
import re
from env.node_alias import to_new

OBJ_TYPES = {"key", "ball", "box"}
COLORS = {"red", "green", "blue", "purple", "yellow", "grey", "gray"}


def parse_pickup_target(env):
    """
    从 mission 文本解析 pickup 目标，例如：
      - "pick up the yellow ball"
      - "pick up a red box"
    返回 (target_type, target_color)；解析不到则 (None, None)
    """
    mission = getattr(env.unwrapped, "mission", None)
    if not mission:
        return None, None

    s = mission.lower().strip()
    m = re.search(r"(pick up|pickup)\s+(the|a)\s+((?P<color>\w+)\s+)?(?P<obj>key|ball|box)", s)
    if not m:
        return None, None

    obj = m.group("obj")
    color = m.group("color")
    if obj not in OBJ_TYPES:
        return None, None
    if color and color not in COLORS:
        color = None
    return obj, color

def parse_target_from_mission(env):
    mission = getattr(env.unwrapped, "mission", None)
    if not mission:
        return None, None

    s = mission.lower().strip()
    patterns = [
        r"(pick up|pickup)\s+(the|a)\s+((?P<color>\w+)\s+)?(?P<obj>key|ball|box)",
        r"(fetch|get|go fetch)\s+(the|a)\s+((?P<color>\w+)\s+)?(?P<obj>key|ball|box)",
        r"(go to)\s+(the|a)\s+((?P<color>\w+)\s+)?(?P<obj>key|ball|box)",
    ]
    for pat in patterns:
        m = re.search(pat, s)
        if m:
            obj = m.group("obj")
            color = m.group("color")
            if obj not in OBJ_TYPES:
                return None, None
            if color and color not in COLORS:
                color = None
            return obj, color
    return None, None

def _find_goal_pos(env):
    u = env.unwrapped
    grid = u.grid
    for idx, obj in enumerate(grid.grid):
        if obj is None:
            continue
        if getattr(obj, "type", None) == "goal":
            x = int(idx % grid.width)
            y = int(idx // grid.width)
            return (x, y)
    return None


def _any_door_open(env) -> int:
    u = env.unwrapped
    grid = u.grid
    for obj in grid.grid:
        if obj is None:
            continue
        if getattr(obj, "type", None) == "door":
            if getattr(obj, "is_open", False):
                return 1
    return 0


def _env_id(env) -> str:
    spec = getattr(env, "spec", None)
    if spec is not None and getattr(spec, "id", None):
        return str(spec.id)
    return "unknown"

def _near_target(env, target_type, target_color=None):
    if target_type is None:
        return 0
    u = env.unwrapped
    ax, ay = int(u.agent_pos[0]), int(u.agent_pos[1])
    grid = u.grid
    for idx, obj in enumerate(grid.grid):
        if obj is None:
            continue
        if getattr(obj, "type", None) != target_type:
            continue
        if target_color is not None and getattr(obj, "color", None) != target_color:
            continue
        x = int(idx % grid.width)
        y = int(idx // grid.width)
        if abs(ax - x) + abs(ay - y) == 1:
            return 1
    return 0

def _box_opened_or_key_released(env):
    u = env.unwrapped
    grid = u.grid

    # 如果已经拿到 key，认为 box 相关前置条件已完成
    carrying = getattr(u, "carrying", None)
    if carrying is not None and getattr(carrying, "type", None) == "key":
        return 1

    # 如果 grid 上已经出现 key，也认为 box 被打开过
    for obj in grid.grid:
        if obj is None:
            continue
        if getattr(obj, "type", None) == "key":
            return 1
    return 0

def extract_subgoals(env):
    """
    统一输出新节点命名体系下的 subgoals。

    节点默认以智能体为主体：
      near_key       : 智能体接近钥匙
      has_key        : 智能体持有钥匙
      near_target    : 智能体接近任务目标物体
      has_target     : 智能体持有任务目标物体
      opened_box     : 智能体已完成打开盒子 / 释放钥匙
      opened_door    : 智能体已完成开门
      has_box        : 智能体持有盒子
      has_ball       : 智能体持有球
      at_goal        : 智能体到达 goal
      task_success   : 任务完成，只作为终止变量
    """
    u = env.unwrapped
    env_id = _env_id(env)

    agent_pos = (int(u.agent_pos[0]), int(u.agent_pos[1]))

    carrying = getattr(u, "carrying", None)
    carrying_type = getattr(carrying, "type", None) if carrying is not None else None
    carrying_color = getattr(carrying, "color", None) if carrying is not None else None

    has_key = int(carrying_type == "key")
    has_ball = int(carrying_type == "ball")
    has_box = int(carrying_type == "box")

    opened_door = int(_any_door_open(env))

    # 只有 KeyInBox 任务才把“key 出现在 grid 上 / 已拿到 key”解释为 opened_box。
    # 普通 DoorKey / UnlockLocal 里 key 本来就在地图上，不能因此认为 opened_box=1。
    if "KeyInBox" in env_id:
        opened_box = int(_box_opened_or_key_released(env))
    else:
        opened_box = 0

    goal_pos = _find_goal_pos(env)
    at_goal = int(goal_pos is not None and agent_pos == goal_pos)

    # key-specific proximity
    near_key = _near_target(env, "key", None)

    # mission target parse
    target_type, target_color = parse_target_from_mission(env)
    near_target = _near_target(env, target_type, target_color)

    ok_type = carrying_type == target_type
    ok_color = (target_color is None) or (carrying_color == target_color)

    if target_type is not None:
        has_target = int(ok_type and ok_color)
    else:
        has_target = 0

    # task_success definition
    if "MiniGrid-DoorKeyBall" in env_id:
        task_success = int(at_goal and has_ball)

    elif "GoToObject" in env_id:
        task_success = int(near_target)

    elif "Fetch" in env_id or "BabyAI-Pickup" in env_id:
        task_success = int(has_target)

    elif "UnlockLocal" in env_id or "KeyInBox" in env_id:
        task_success = int(opened_door)

    elif "UnlockPickup" in env_id:
        task_success = int(has_box)

    else:
        task_success = int(at_goal)

    return {
        "near_key": int(near_key),
        "has_key": int(has_key),

        "near_target": int(near_target),
        "has_target": int(has_target),

        "opened_box": int(opened_box),
        "opened_door": int(opened_door),

        "has_box": int(has_box),
        "has_ball": int(has_ball),
        "at_goal": int(at_goal),
        "task_success": int(task_success),

        "_env_id": env_id,
        "_carrying_type": carrying_type,
        "_carrying_color": carrying_color,
        "_target_type": target_type,
        "_target_color": target_color,
    }