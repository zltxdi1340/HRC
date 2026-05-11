# interventions/do_ops.py
from __future__ import annotations

from typing import Optional, Tuple

from minigrid.core.world_object import Key, Ball, Box


def _make_obj(obj_type: str, color: str | None):
    # 兜底创建对象（很少用到：当 grid 找不到目标对象时）
    c = color or "yellow"
    if obj_type == "key":
        return Key(c)
    if obj_type == "ball":
        return Ball(c)
    if obj_type == "box":
        return Box(c)
    return None


def _iter_grid_objects(env):
    u = env.unwrapped
    grid = u.grid
    for idx, obj in enumerate(grid.grid):
        if obj is None:
            continue
        x = int(idx % grid.width)
        y = int(idx // grid.width)
        yield x, y, obj


def find_locked_doors(env):
    """返回所有 locked door，按 grid 扫描顺序排序。"""
    out = []
    for x, y, obj in _iter_grid_objects(env):
        if getattr(obj, "type", None) != "door":
            continue
        if bool(getattr(obj, "is_locked", False)):
            out.append((x, y, obj))
    return out


def choose_target_locked_door(env) -> Optional[Tuple[int, int, object]]:
    """确定性地选择目标 locked door：当前先取 grid 扫描顺序的第一扇。"""
    doors = find_locked_doors(env)
    if doors:
        return doors[0]
    return None


def choose_target_door(env, color: str | None = None) -> Optional[Tuple[int, int, object]]:
    """优先返回 locked door；若没有，则返回第一扇匹配颜色的 door。"""
    locked = choose_target_locked_door(env)
    if locked is not None:
        x, y, door = locked
        if color is None or getattr(door, "color", None) == color:
            return locked

    for x, y, obj in _iter_grid_objects(env):
        if getattr(obj, "type", None) != "door":
            continue
        if color is not None and getattr(obj, "color", None) != color:
            continue
        return (x, y, obj)
    return None


def do_have(env, obj_type: str, color: str | None = None):
    """
    do(has_<obj>=1): 保证 agent carrying 为指定 obj_type（可选 color）。
    语义：目标已经满足也算成功（返回 True），避免 P(intervened) 被“冗余 do”压低。
    """
    u = env.unwrapped
    carrying = getattr(u, "carrying", None)
    if carrying is not None:
        if getattr(carrying, "type", None) == obj_type:
            if color is None or getattr(carrying, "color", None) == color:
                return True

    grid = u.grid

    # 先在 grid 中找目标物体
    found_pos = None
    found_obj = None
    for x, y, obj in _iter_grid_objects(env):
        if getattr(obj, "type", None) != obj_type:
            continue
        if color is not None and getattr(obj, "color", None) != color:
            continue
        found_pos = (x, y)
        found_obj = obj
        break

    if found_obj is not None:
        grid.set(found_pos[0], found_pos[1], None)
        u.carrying = found_obj
        return True

    # grid 找不到则兜底创建一个
    new_obj = _make_obj(obj_type, color)
    if new_obj is None:
        return False
    u.carrying = new_obj
    return True


def do_have_matching_key_for_locked_door(env) -> bool:
    """
    对 door tasks 的 do(has_key=1):
    自动找到目标 locked door，并给 agent 一把颜色匹配的 key。
    若没有 locked door，则退化为任意 key。
    """
    target = choose_target_locked_door(env)
    if target is None:
        return do_have(env, "key")
    _x, _y, door = target
    door_color = getattr(door, "color", None)
    return do_have(env, "key", color=door_color)


def do_door_open(env, open_=True, color: str | None = None):
    """
    do(door_open=1): 优先打开目标 locked door（并解锁）；
    若没有 locked door，则打开第一扇匹配颜色的 door。
    门已经开也算成功。
    """
    target = choose_target_door(env, color=color)
    if target is None:
        return False

    _x, _y, obj = target
    if hasattr(obj, "is_open"):
        obj.is_open = bool(open_)
    if hasattr(obj, "is_locked") and open_:
        obj.is_locked = False
    return True


def do_near_obj(env, obj_type: str, color: str | None = None):
    """
    do(near_obj=1): 把 agent 放到目标物体旁边。
    """
    u = env.unwrapped
    grid = u.grid

    target_pos = None
    for idx, obj in enumerate(grid.grid):
        if obj is None:
            continue
        if getattr(obj, "type", None) != obj_type:
            continue
        if color is not None and getattr(obj, "color", None) != color:
            continue
        x = int(idx % grid.width)
        y = int(idx // grid.width)
        target_pos = (x, y)
        break

    if target_pos is None:
        return False

    tx, ty = target_pos
    candidates = [(tx-1, ty), (tx+1, ty), (tx, ty-1), (tx, ty+1)]
    for x, y in candidates:
        if 0 <= x < grid.width and 0 <= y < grid.height:
            cell = grid.get(x, y)
            if cell is None:
                u.agent_pos = (x, y)
                return True
    return False


def do_open_box(env):
    """
    do(open_box=1): 如果 box 里有 key，就把 key 释放到 grid 上。
    """
    u = env.unwrapped
    grid = u.grid

    for idx, obj in enumerate(grid.grid):
        if obj is None:
            continue
        if getattr(obj, "type", None) != "box":
            continue

        x = int(idx % grid.width)
        y = int(idx // grid.width)

        contained = getattr(obj, "contains", None)
        if contained is not None:
            grid.set(x, y, contained)
            return True

    return False
