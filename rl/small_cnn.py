# rl/small_cnn.py
import gymnasium as gym
import torch as th
import torch.nn as nn
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor

class SmallCNN(BaseFeaturesExtractor):
    """
    For tiny images like 7x7 (MiniGrid). Output features_dim for policy/value MLP heads.
    Expects observation_space is Dict with key "image" shaped (C,H,W).
    """
    def __init__(self, observation_space: gym.spaces.Dict, features_dim: int = 128):
        # We will compute the final dim dynamically
        super().__init__(observation_space, features_dim=1)

        img_space = observation_space.spaces["image"]
        n_input_channels = img_space.shape[0]

        self.cnn = nn.Sequential(
            nn.Conv2d(n_input_channels, 32, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.Flatten(),
        )

        # Compute shape by doing one forward pass
        with th.no_grad():
            sample = th.as_tensor(img_space.sample()[None]).float()
            n_flatten = self.cnn(sample).shape[1]

        self.linear = nn.Sequential(
            nn.Linear(n_flatten, features_dim),
            nn.ReLU(),
        )

        self._features_dim = features_dim

    def forward(self, observations):
        x = observations["image"].float() / 255.0
        x = self.cnn(x)
        return self.linear(x)
    
# 追加到 rl/small_cnn.py
class SmallCNNCombined(BaseFeaturesExtractor):
    def __init__(self, observation_space: gym.spaces.Dict, features_dim: int = 256):
        super().__init__(observation_space, features_dim=1)

        img_space = observation_space.spaces["image"]
        n_input_channels = img_space.shape[0]
        goal_dim = observation_space.spaces["goal"].shape[0]

        self.cnn = nn.Sequential(
            nn.Conv2d(n_input_channels, 32, 3, 1, 1), nn.ReLU(),
            nn.Conv2d(32, 64, 3, 1, 1), nn.ReLU(),
            nn.Flatten(),
        )

        with th.no_grad():
            sample = th.as_tensor(img_space.sample()[None]).float()
            n_flatten = self.cnn(sample).shape[1]

        # direction is Discrete(4) in your env; encode as one-hot inside forward
        dir_dim = 4
        self.mlp = nn.Sequential(
            nn.Linear(n_flatten + dir_dim + goal_dim, features_dim),
            nn.ReLU(),
        )
        self._features_dim = features_dim

    def forward(self, obs):
        img = obs["image"].float()
        # SB3 preprocess 对 uint8 image 往往已经做了 /255，这里稳妥处理一次
        if img.max() > 1.0:
            img = img / 255.0
        x = self.cnn(img)  # [B, n_flatten]

        # direction: 可能是 [B,1] int，也可能已是 one-hot [B,4] 或 [B,1,4]
        d = obs["direction"]
        if d.dtype in (th.int8, th.int16, th.int32, th.int64, th.long):
            # integer -> one-hot
            if d.ndim > 1:
                d = d.squeeze(-1)
            d_onehot = th.nn.functional.one_hot(d, num_classes=4).float()
        else:
            # already one-hot float
            d_onehot = d.float()

        # squeeze possible extra dim: [B,1,4] -> [B,4]
        if d_onehot.ndim == 3 and d_onehot.shape[1] == 1:
            d_onehot = d_onehot.squeeze(1)

        g = obs["goal"].float()
        # squeeze possible extra dim: [B,1,G] -> [B,G]
        if g.ndim == 3 and g.shape[1] == 1:
            g = g.squeeze(1)

        z = th.cat([x, d_onehot, g], dim=1)
        return self.mlp(z)