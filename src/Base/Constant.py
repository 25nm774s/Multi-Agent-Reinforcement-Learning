from typing import Tuple,List,Any,Dict
import torch

# 環境から受け取る座標はこっちを使用　共通の座標表現形式
PosType = Tuple[int,int]
"""共通の座標表現形式"""

GlobalState = Dict[str,PosType]
"""共通の状態表現s"""

ExperienceType = Tuple[
    torch.Tensor, # agent_obs_tensor (n_agents, num_channels, grid_size, grid_size)
    torch.Tensor, # global_state_tensor ((goals_number + n_agents) * 2,)
    torch.Tensor, # actions_tensor (n_agents,)
    torch.Tensor, # rewards_tensor (n_agents,)
    torch.Tensor, # dones_tensor (n_agents,)
    torch.Tensor, # next_agent_obs_tensor (n_agents, num_channels, grid_size, grid_size)
    torch.Tensor  # next_global_state_tensor ((goals_number + n_agents) * 2,)
]
"""経験のデータ型"""

#####################################
# Qテーブル内部の座標表現形式
QState = Tuple[int, ...]
"""Qテーブル入力用の座標表現形式(旧QState:)"""

QValues = List[float]
"""行動に対するQ値"""

QTableType = Dict[QState, QValues]
"""Qテーブルの構造"""

####################################
# DQN内部の座標表現形式
# 未定義