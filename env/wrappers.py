# env/wrappers.py
import gymnasium as gym

class NoDropWrapper(gym.Wrapper):
    """
    全局禁用 drop（动作4），避免资源变量不单调导致概率太小/结构不可识别。
    """
    def __init__(self, env, drop_action=4, replacement_action=6):
        super().__init__(env)
        self.drop_action = drop_action
        self.replacement_action = replacement_action  # done/no-op

    def step(self, action):
        if int(action) == self.drop_action:
            action = self.replacement_action
        return self.env.step(action)
