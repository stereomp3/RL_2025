import torch as th
import torch.nn as nn
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor
from gymnasium import spaces


class LightWeightCNN(BaseFeaturesExtractor):
    """
    針對 64x64x4 (Frame Stack) 輸入優化的輕量級 CNN。
    結構參考 Nature CNN 但減少參數以加快運算。
    """

    def __init__(self, observation_space: spaces.Box, features_dim: int = 256):
        super().__init__(observation_space, features_dim)

        # 假設輸入是 (Batch, Channel, Height, Width) -> (B, 4, 64, 64)
        n_input_channels = observation_space.shape[0]

        self.cnn = nn.Sequential(
            # Layer 1: 捕捉大特徵 (賽道邊界)
            nn.Conv2d(n_input_channels, 32, kernel_size=8, stride=4, padding=0),
            nn.ReLU(),

            # Layer 2: 捕捉細節
            nn.Conv2d(32, 64, kernel_size=4, stride=2, padding=0),
            nn.ReLU(),

            # Layer 3: 高層特徵
            nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=0),
            nn.ReLU(),

            nn.Flatten(),
        )

        # 計算 Flatten 後的維度
        # 輸入 64x64 -> L1(15x15) -> L2(6x6) -> L3(4x4) -> 64*4*4 = 1024
        with th.no_grad():
            n_flatten = self.cnn(th.as_tensor(observation_space.sample()[None]).float()).shape[1]

        self.linear = nn.Sequential(
            nn.Linear(n_flatten, features_dim),
            nn.ReLU()
        )

    def forward(self, observations: th.Tensor) -> th.Tensor:
        return self.linear(self.cnn(observations))


# 定義 Policy 參數 (PPO 與 A2C 共用)
# 這會告訴 SB3 使用我們定義的 LightWeightCNN，而不是預設的大型網路
POLICY_KWARGS = dict(
    features_extractor_class=LightWeightCNN,
    features_extractor_kwargs=dict(features_dim=256),
    net_arch=dict(pi=[256, 128], vf=[256, 128])  # Actor 和 Critic 的全連接層
)