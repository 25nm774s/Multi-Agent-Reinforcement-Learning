import torch
from typing import Dict, List, Any
from Base.Constant import PosType

class ObsToTensorWrapper:
    """
    環境の生の状態（全体座標）を、Qネットワークが処理できるグリッド表現に変換するクラス。
    """
    def __init__(self, grid_size: int, goals_number: int, agents_number: int, device: torch.device, neighbor_distance: int = 0, agent_ids: List[str] = [], goal_ids: List[str] = []):
        """
        StateProcessor のコンストラクタ.

        Args:
            grid_size (int): グリッドのサイズ.
            goals_num (int): 環境中のゴール数.
            agents_num (int): 環境中のエージェント数.
            device (torch.device): テンソル操作に使用するデバイス (CPU/GPU).
            neighbor_distance (int): エージェントが他のエージェントを観測できる最大距離。
                                     0の場合、他のエージェントは観測されない。
            agent_ids (List[str]): エージェントのIDリスト.
            goal_ids (List[str]): ゴールのIDリスト.
        """
        self.grid_size = grid_size
        self.goals_num = goals_number
        self.agents_num = agents_number
        self.device = device
        self.num_channels = 3 # ゴール, 自身, 他者の3チャネル
        self.neighbor_distance = neighbor_distance
        self._agent_ids = agent_ids
        self._goal_ids = goal_ids

    def transform_state_batch(self, single_agent_obs_dict: Dict[str, Any]) -> torch.Tensor:
        """
        単一エージェントのローカル観測辞書を受け取り、
        そのエージェントの視点に基づいた3チャネルのグリッド表現テンソル (3, G, G) に変換する。

        Args:
            single_agent_obs_dict (Dict[str, Any]): 単一エージェントのローカル観測辞書。
                                                     キー: 'self' (PosType), 'all_goals' (List[PosType]), 'others' (Dict[str, PosType])

        Returns:
            torch.Tensor: 形状 (num_channels, grid_size, grid_size) のテンソル。
        """
        G = self.grid_size

        # 最終的な出力テンソルを初期化 (3, G, G)
        state_map = torch.zeros((self.num_channels, G, G), dtype=torch.float32, device=self.device)

        # --- 座標の抽出 ---
        my_pos: PosType = single_agent_obs_dict['self']
        all_goal_positions: List[PosType] = single_agent_obs_dict['all_goals']
        others_info: Dict[str, PosType] = single_agent_obs_dict['others']

        # --- チャネル 0: ゴールマップの設定 ---
        for gx, gy in all_goal_positions:
            if 0 <= gx < G and 0 <= gy < G: # 有効な座標のみをプロット
                state_map[0, gx, gy] = 1.0

        # --- チャネル 1: 自身のエージェントのマップの設定 ---
        mx, my = my_pos
        if 0 <= mx < G and 0 <= my < G: # 有効な座標のみをプロット
            state_map[1, mx, my] = 1.0

        # --- チャネル 2: 他のエージェントのマップの設定 ---
        for other_agent_id, (ox, oy) in others_info.items():
            # env._get_observation() が既に観測範囲外のエージェントを (-1, -1) で表現しているため、
            # ここではそれらを無視し、有効な座標のみをプロットします。
            if ox != -1 and oy != -1 and 0 <= ox < G and 0 <= oy < G:
                state_map[2, ox, oy] = 1.0

        return state_map

    def _flatten_global_state_dict(self, state_dict: Dict[str, Any]) -> torch.Tensor:
        """
        GlobalState辞書をStateProcessorが期待する順序でフラットなテンソルに変換します。
        順序はゴールID、次にエージェントIDの座標です。

        Args:
            state_dict (Dict[str, Any]): グローバル状態の辞書 (例: obj_id -> (x, y)).

        Returns:
            torch.Tensor: フラット化されたグローバル状態テンソル ((goals_num + agents_num) * 2,).
        """
        flat_coords = []
        # ゴール座標を順序通りに追加
        for goal_id in self._goal_ids:
            # 存在しない場合は (-1, -1) を使用
            pos = state_dict.get(goal_id, (-1, -1))
            flat_coords.extend(pos)

        # エージェント座標を順序通りに追加
        for agent_id in self._agent_ids:
            # 存在しない場合は (-1, -1) を使用
            pos = state_dict.get(agent_id, (-1, -1))
            flat_coords.extend(pos)

        return torch.tensor(flat_coords, dtype=torch.float32, device=self.device)