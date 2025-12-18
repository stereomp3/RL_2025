import torch as th
import torch.nn as nn
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor
from gymnasium import spaces


class TransformerCNN(BaseFeaturesExtractor):
    """
    混合架構:
    1. CNN: 負責從單張畫面提取特徵 (Spatial Features)
    2. Transformer: 負責處理連續幀之間的關係 (Temporal Features)
    """

    def __init__(self, observation_space: spaces.Box, features_dim: int = 512):
        # features_dim 增加到 512
        super().__init__(observation_space, features_dim)

        n_stacked_frames = observation_space.shape[0]  # 8
        self.img_h = observation_space.shape[1]
        self.img_w = observation_space.shape[2]

        # --- 1. 增強版 CNN (類似 Nature CNN，但針對單通道) ---
        # 結構: 32(8x8) -> 64(4x4) -> 64(3x3)
        self.cnn = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=8, stride=4, padding=0),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=4, stride=2, padding=0),
            nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=0),
            nn.ReLU(),
            nn.Flatten()
        )

        # 計算 CNN 輸出的維度
        with th.no_grad():
            sample_input = th.zeros(1, 1, self.img_h, self.img_w)
            cnn_out_dim = self.cnn(sample_input).shape[1]

        # --- 2. 高規格 Transformer Encoder ---
        self.embed_dim = 512  # 增加維度
        self.projection = nn.Linear(cnn_out_dim, self.embed_dim)

        # Positional Encoding
        self.pos_embedding = nn.Parameter(th.randn(1, n_stacked_frames, self.embed_dim))

        # Transformer Layer
        # nhead=8 (8個注意力頭)
        # dim_feedforward=2048 (更大的隱藏層)
        # num_layers=4 (堆疊4層 Encoder，增加深度推理能力)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=self.embed_dim,
            nhead=8,
            dim_feedforward=2048,
            dropout=0.0,  # 關閉 Dropout 以追求訓練時的最大擬合能力
            batch_first=True
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=4)

        # --- 3. 輸出層 ---
        self.linear = nn.Sequential(
            nn.Linear(self.embed_dim * n_stacked_frames, features_dim),
            nn.ReLU()
        )

    def forward(self, observations: th.Tensor) -> th.Tensor:
        batch_size = observations.shape[0]
        n_stack = observations.shape[1]

        # (Batch, 4, 64, 64) -> (Batch * 4, 1, 64, 64)
        # 使用 reshape 避免不連續記憶體報錯
        x = observations.reshape(-1, 1, self.img_h, self.img_w)

        # CNN 特徵提取
        cnn_feats = self.cnn(x)

        # 投影並還原序列維度
        cnn_feats = self.projection(cnn_feats)
        sequence = cnn_feats.reshape(batch_size, n_stack, self.embed_dim)

        # 加入位置編碼
        sequence = sequence + self.pos_embedding

        # Transformer 處理
        transformer_out = self.transformer(sequence)

        # Flatten 並輸出
        output = transformer_out.reshape(batch_size, -1)

        return self.linear(output)


# 定義 Policy 參數 (使用更深更寬的全連接層)
POLICY_KWARGS = dict(
    features_extractor_class=TransformerCNN,  # 負責「看」畫面並理解時間
    features_extractor_kwargs=dict(features_dim=512),
    # Actor 和 Critic 都用 1024x1024
    # pi = Policy (Actor), vf = Value Function (Critic)
    # net_arch=dict(pi=[1024, 1024], vf=[1024, 1024]),
    net_arch=dict(pi=[1024, 1024, 512], vf=[1024, 1024, 512]),
    activation_fn=th.nn.ReLU
    # Actor (策略網路)：負責根據看到的畫面決定「動作」(油門、方向盤)。
    # Critic (價值網路)：負責評估現在的狀況「好不好」(預測未來分數)。
)



# import torch as th
# import torch.nn as nn
# from stable_baselines3.common.torch_layers import BaseFeaturesExtractor
# from gymnasium import spaces
#
#
# class TransformerCNN(BaseFeaturesExtractor):
#     """
#     混合架構:
#     1. CNN: 負責從單張畫面提取特徵 (Spatial Features)
#     2. Transformer: 負責處理連續幀之間的關係 (Temporal Features)
#     """
#
#     def __init__(self, observation_space: spaces.Box, features_dim: int = 256):
#         super().__init__(observation_space, features_dim)
#
#         # 輸入形狀預期為 (Batch, n_stack, Height, Width) -> e.g., (B, 4, 64, 64)
#         n_stacked_frames = observation_space.shape[0]  # 4
#         self.img_h = observation_space.shape[1]
#         self.img_w = observation_space.shape[2]
#
#         # --- 1. CNN 特徵提取器 (針對"單幀"設計) ---
#         # 輸入 channel 設為 1 (單張灰階)
#         self.cnn = nn.Sequential(
#             nn.Conv2d(1, 16, kernel_size=8, stride=4, padding=0),
#             nn.ReLU(),
#             nn.Conv2d(16, 32, kernel_size=4, stride=2, padding=0),
#             nn.ReLU(),
#             nn.Flatten()
#         )
#
#         # 計算 CNN 輸出的維度
#         with th.no_grad():
#             sample_input = th.zeros(1, 1, self.img_h, self.img_w)
#             cnn_out_dim = self.cnn(sample_input).shape[1]
#
#         # --- 2. Transformer Encoder ---
#         self.embed_dim = 128
#         self.projection = nn.Linear(cnn_out_dim, self.embed_dim)
#
#         # Positional Encoding
#         self.pos_embedding = nn.Parameter(th.randn(1, n_stacked_frames, self.embed_dim))
#
#         # Transformer Encoder Layer
#         encoder_layer = nn.TransformerEncoderLayer(
#             d_model=self.embed_dim,
#             nhead=4,
#             dim_feedforward=256,
#             batch_first=True
#         )
#         self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=1)
#
#         # --- 3. 輸出層 ---
#         self.linear = nn.Sequential(
#             nn.Linear(self.embed_dim * n_stacked_frames, features_dim),
#             nn.ReLU()
#         )
#
#     def forward(self, observations: th.Tensor) -> th.Tensor:
#         batch_size = observations.shape[0]
#         n_stack = observations.shape[1]
#
#         # --- 修正點: 使用 .reshape() 取代 .view() ---
#         # .view() 在 Tensor 不連續時會報錯，.reshape() 則會自動處理
#         # (Batch, 4, 64, 64) -> (Batch * 4, 1, 64, 64)
#         x = observations.reshape(-1, 1, self.img_h, self.img_w)
#
#         # CNN 特徵提取
#         cnn_feats = self.cnn(x)
#
#         # 投影並還原序列維度
#         cnn_feats = self.projection(cnn_feats)
#         sequence = cnn_feats.reshape(batch_size, n_stack, self.embed_dim)
#
#         # 加入位置編碼
#         sequence = sequence + self.pos_embedding
#
#         # Transformer 處理
#         transformer_out = self.transformer(sequence)
#
#         # Flatten 並輸出
#         output = transformer_out.reshape(batch_size, -1)
#
#         return self.linear(output)
#
#
# # 定義 Policy 參數
# POLICY_KWARGS = dict(
#     features_extractor_class=TransformerCNN,
#     features_extractor_kwargs=dict(features_dim=256),
#     net_arch=dict(pi=[256, 128], vf=[256, 128])
# )