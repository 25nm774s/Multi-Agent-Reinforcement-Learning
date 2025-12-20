import torch
from typing import Tuple, List, Any

class StateProcessor:
    """
    環境の生の状態（全体座標）を、Qネットワークが処理できるグリッド表現に変換するクラス.
    """
    def __init__(self, grid_size: int, goals_num: int, agents_num: int, device: torch.device):
        """
        StateProcessor のコンストラクタ.

        Args:
            grid_size (int): グリッドのサイズ.
            goals_num (int): 環境中のゴール数.
            agents_num (int): 環境中のエージェント数.
            device (torch.device): テンソル操作に使用するデバイス (CPU/GPU).
        """
        self.grid_size = grid_size
        self.goals_num = goals_num
        self.agents_num = agents_num
        self.device = device
        self.num_channels = 3 # ゴール, 自身, 他者の3チャネル

    def transform_state_batch(self, i: int, batch_raw_data: torch.Tensor) -> torch.Tensor:
        """
        一次元の全体状態テンソル (B, feature_dim) を受け取り、
        エージェント i の視点に基づいた3チャネルのグリッド表現テンソル (B, 3, G, G) に変換する。

        Args:
            i (int): 観測を生成するエージェントのインデックス (0-indexed)。
            batch_raw_data (torch.Tensor): リプレイバッファから取り出したバッチテンソル (B, feature_dim)。
                                           全体状態 (ゴールと全エージェントの座標) を含む。

        Returns:
            torch.Tensor: 形状 (batch_size, 3, grid_size, grid_size) のテンソル。
        """
        batch_size = batch_raw_data.size(0)
        G = self.grid_size

        # 座標はインデックスとして使うため、整数型に変換 (Long型推奨)
        coords = batch_raw_data.long().to(self.device)

        # --- 座標の抽出とリシェイプ ---
        goal_coords_end = self.goals_num * 2

        # 1. ゴール座標: (B, goals_num, 2)
        goal_coords = coords[:, :goal_coords_end].reshape(batch_size, self.goals_num, 2)

        # 2. 全エージェント座標: (B, agents_num, 2)
        all_agent_coords = coords[:, goal_coords_end:].reshape(batch_size, self.agents_num, 2)

        # --- グリッドマップの作成 (3チャネル: ゴール, 自身, 他者) ---

        # 最終的な出力テンソルを初期化 (B, 3, G, G)
        state_map = torch.zeros((batch_size, self.num_channels, G, G), dtype=torch.float32, device=self.device)

        # 全バッチ、全エンティティに対応するインデックスを準備
        batch_indices_base = torch.arange(batch_size, device=self.device).repeat_interleave(self.goals_num)

        # --- チャネル 0: ゴールマップの設定 (全バッチ共通) ---

        # x座標とy座標を平坦化
        goal_x = goal_coords[:, :, 0].flatten()
        goal_y = goal_coords[:, :, 1].flatten()

        # state_map[バッチインデックス, チャネル0, x座標, y座標] = 1.0
        state_map[batch_indices_base, 0, goal_x, goal_y] = 1.0

        # --- チャネル 1: 自身のエージェント i のマップの設定 ---

        # 自身のエージェント i の座標 (B, 2) を抽出
        self_coords = all_agent_coords[:, i, :] # (B, 2)

        # 自身のエージェント i の座標を平坦化 (B, 2 -> B)
        self_x = self_coords[:, 0] # (B,)
        self_y = self_coords[:, 1] # (B,)

        # バッチインデックス (B,)
        batch_indices_self = torch.arange(batch_size, device=self.device)

        # state_map[バッチインデックス, チャネル1, iのx座標, iのy座標] = 1.0
        state_map[batch_indices_self, 1, self_x, self_y] = 1.0

        # --- チャネル 2: 他のエージェントのマップの設定 ---

        if self.agents_num > 1:
            filtered_other_coords_list = []
            filtered_batch_indices_list = []

            for j in range(self.agents_num):
                if j != i:
                    current_agent_coords = all_agent_coords[:, j, :]

                    # -1以外の有効な座標をフィルタリング
                    valid_coords_mask = (current_agent_coords[:, 0] != -1) & (current_agent_coords[:, 1] != -1)

                    # 有効な座標とそれに対応するバッチインデックスを抽出
                    filtered_coords = current_agent_coords[valid_coords_mask]
                    filtered_batch_indices = batch_indices_self[valid_coords_mask]

                    if filtered_coords.numel() > 0: # 有効な座標が存在する場合のみ追加
                        filtered_other_coords_list.append(filtered_coords)
                        filtered_batch_indices_list.append(filtered_batch_indices)

            if filtered_other_coords_list: # 他のエージェントの有効な座標が存在する場合のみ処理
                # 全ての有効な他のエージェントの座標を結合
                other_coords = torch.cat(filtered_other_coords_list, dim=0)
                # 全ての有効な他のエージェントに対応するバッチインデックスを結合
                batch_indices_other = torch.cat(filtered_batch_indices_list, dim=0)

                # x座標とy座標を抽出
                other_x = other_coords[:, 0]
                other_y = other_coords[:, 1]

                # state_map[バッチインデックス, チャネル2, 他のx座標, 他のy座標] = 1.0
                state_map[batch_indices_other, 2, other_x, other_y] = 1.0

        return state_map
