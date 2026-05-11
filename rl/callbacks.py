# rl/callbacks.py
from collections import defaultdict, deque
from stable_baselines3.common.callbacks import BaseCallback

class GoalSuccessCallback(BaseCallback):
    def __init__(self, window=200, verbose=0):
        super().__init__(verbose)
        self.window = window
        self.buf = defaultdict(lambda: deque(maxlen=window))       # goal -> success
        self.diag_buf = defaultdict(lambda: deque(maxlen=window))  # (goal, metric) -> values

    def _on_step(self) -> bool:
        dones = self.locals["dones"]
        infos = self.locals["infos"]
        rewards = self.locals.get("rewards", None)

        for i, done in enumerate(dones):
            if not done:
                continue

            info = infos[i]
            goal = info.get("goal_node")
            if goal is None:
                continue

            # success
            if "is_success" in info:
                succ = 1.0 if info["is_success"] else 0.0
            elif rewards is not None:
                succ = 1.0 if rewards[i] > 0 else 0.0
            else:
                succ = 0.0

            self.buf[goal].append(succ)

            # diagnostics
            for k in [
                "debug_ever_target_seen",
                "debug_ever_target_front",
                "debug_pickup_attempted",
                "debug_pickup_attempt_front",
                "debug_pickup_attempt_count",
            ]:
                if k in info:
                    self.diag_buf[(goal, k)].append(float(info[k]))

        # success curves
        for g, dq in self.buf.items():
            if len(dq) > 0:
                self.logger.record(f"goal_success/{g}", sum(dq) / len(dq))

        # diagnostic curves
        for (g, k), dq in self.diag_buf.items():
            if len(dq) > 0:
                metric_name = k.replace("debug_", "")
                self.logger.record(f"diag/{g}/{metric_name}", sum(dq) / len(dq))

        return True