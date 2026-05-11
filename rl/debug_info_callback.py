from __future__ import annotations
from collections import deque
from typing import Iterable, Dict, Any

import numpy as np
from stable_baselines3.common.callbacks import BaseCallback


class DebugInfoTensorboardCallback(BaseCallback):
    """Log selected env info keys to TensorBoard with short, collision-safe names.

    For each selected key, logs up to three series:
      - debug/<alias>_s : mean across envs for the current callback step
      - debug/<alias>_e : mean over completed episodes only (when `dones` is true)
      - debug/<alias>_w : moving average over recent completed episodes

    Key aliases are intentionally short to avoid Stable-Baselines3 logger
    truncation collisions.
    """

    DEFAULT_ALIAS = {
        "debug_ever_seen_door": "seen_door",
        "debug_ever_front_of_door": "front_door",
        "debug_toggle_attempted": "toggle_any",
        "debug_toggle_attempt_front_of_door": "toggle_front",
        "debug_toggle_attempt_count": "toggle_cnt",
        "debug_seen_door_this_step": "seen_step",
        "debug_front_of_door_this_step": "front_step",
        "debug_ever_target_seen": "seen_tgt",
        "debug_ever_target_front": "front_tgt",
        "debug_pickup_attempted": "pickup_any",
        "debug_pickup_attempt_front": "pickup_front",
        "debug_pickup_attempt_count": "pickup_cnt",
        "strict_parent_mode": "strict_mode",
        "strict_target_locked_door_exists": "target_door",
        "strict_parent_failed": "parent_fail",
    }

    def __init__(
        self,
        keys: Iterable[str],
        ep_window: int = 200,
        prefix: str = "debug",
        verbose: int = 0,
        alias_map: Dict[str, str] | None = None,
    ):
        super().__init__(verbose)
        self.keys = [str(k).strip() for k in keys if str(k).strip()]
        self.ep_window = int(ep_window)
        self.prefix = str(prefix).strip().rstrip("/") or "debug"
        self.alias_map = dict(self.DEFAULT_ALIAS)
        if alias_map:
            self.alias_map.update({str(k).strip(): str(v).strip() for k, v in alias_map.items()})
        self._ep_buffers: Dict[str, deque] = {
            k: deque(maxlen=self.ep_window) for k in self.keys
        }

    @staticmethod
    def _to_float(v: Any):
        if isinstance(v, (int, float, np.integer, np.floating, bool)):
            return float(v)
        return None

    def _alias(self, key: str) -> str:
        alias = self.alias_map.get(key, key)
        # final defensive cleanup to keep names short and stable
        alias = alias.strip().replace("/", "_").replace(" ", "_")
        return alias[:24]

    def _record(self, name: str, value: float) -> None:
        # keep record names compact to avoid logger truncation collisions
        self.logger.record(name, float(value))

    def _on_step(self) -> bool:
        infos = self.locals.get("infos", []) or []
        dones = self.locals.get("dones", None)
        if dones is None:
            dones = [False] * len(infos)

        for key in self.keys:
            step_vals = []
            ep_vals = []
            for info, done in zip(infos, dones):
                if not isinstance(info, dict) or key not in info:
                    continue
                fv = self._to_float(info.get(key))
                if fv is None:
                    continue
                step_vals.append(fv)
                if bool(done):
                    ep_vals.append(fv)
                    self._ep_buffers[key].append(fv)

            alias = self._alias(key)
            if step_vals:
                self._record(f"{self.prefix}/{alias}_s", np.mean(step_vals))
            if ep_vals:
                self._record(f"{self.prefix}/{alias}_e", np.mean(ep_vals))
            if len(self._ep_buffers[key]) > 0:
                self._record(f"{self.prefix}/{alias}_w", np.mean(self._ep_buffers[key]))

        return True
