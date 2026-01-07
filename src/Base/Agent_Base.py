from abc import ABC, abstractmethod
import torch
import torch.nn as nn # オプティマイザーを設定するため。
import numpy as np
from typing import List, Optional, Tuple, Dict, Any

from utils.StateProcesser import StateProcessor
from DQN.dqn import AgentNetwork
from .Constant import GlobalState

class BaseMasterAgent(ABC):
    """
    抽象基底クラス BaseMasterAgent.
    状態処理、Q値計算、生存マスク適用などの共通ロジックをカプセル化し、
    IQLMasterAgent および QMIXMasterAgent の基盤を提供します。
    """
    def __init__(self,
                 n_agents: int,
                 action_size: int,
                 grid_size: int,
                 goals_number: int,
                 device: torch.device,
                 state_processor: StateProcessor,
                 agent_network: AgentNetwork,
                 agent_ids: List[str],
                 goal_ids: List[str]):
        """
        BaseMasterAgent のコンストラクタ.

        Args:
            n_agents (int): 環境内のエージェント数.
            action_size (int): エージェントのアクション空間のサイズ.
            grid_size (int): グリッド環境のサイズ.
            goals_number (int): 環境内のゴール数.
            device (torch.device): テンソル操作に使用するデバイス (CPU/GPU).
            state_processor (StateProcessor): 状態変換に使用するStateProcessorのインスタンス.
            agent_network (AgentNetwork): 共有AgentNetworkのインスタンス.
            agent_ids (List[str]): エージェントのIDリスト.
            goal_ids (List[str]): ゴールのIDリスト.
        """
        super().__init__()
        self.n_agents = n_agents
        self.action_size = action_size
        self.grid_size = grid_size
        self.goals_num = goals_number
        self.device = device
        self.state_processor = state_processor
        self.agent_network = agent_network
        self._agent_ids = agent_ids
        self._goal_ids = goal_ids

        self.num_channels = self.agent_network.num_channels

        self.agent_network_target = AgentNetwork(grid_size, action_size, n_agents).to(device)
        self.agent_network_target.load_state_dict(self.agent_network.state_dict())
        self.agent_network_target.eval()

    def _process_raw_observations_batch(self, obs_dicts_batch: List[Dict[str, Dict[str, Any]]]) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        ReplayBuffer.sampleから得られる観測辞書のバッチを処理し、
        AgentNetworkおよびMixingNetworkが期待するテンソル形式に変換します。

        Args:
            obs_dicts_batch (List[Dict[str, Dict[str, Any]]]):
                バッチサイズBの観測辞書のリスト。各辞書はエージェントIDをキーとし、
                そのエージェントのローカル観測（'self', 'all_goals', 'others'）を含む。

        Returns:
            Tuple[torch.Tensor, torch.Tensor]:
                - transformed_obs_batch (torch.Tensor): AgentNetworkへの入力 (B * N, C, G, G)。
                - global_state_tensor_for_mixing_network (torch.Tensor): MixingNetworkへの入力 (B, state_dim)。
        """
        batch_size = len(obs_dicts_batch)

        all_transformed_obs_for_agents = [] # (B*N, C, G, G)
        all_flattened_global_states_for_mixing = [] # (B, state_dim)

        for obs_dict_item in obs_dicts_batch:
            # AgentNetwork用のローカル観測テンソルを生成
            for agent_id in self._agent_ids:
                # StateProcessor.transform_state_batch は単一エージェントの観測辞書を期待するように変更された
                single_agent_obs = obs_dict_item[agent_id]
                transformed_obs_for_agent = self.state_processor.transform_state_batch(single_agent_obs)
                all_transformed_obs_for_agents.append(transformed_obs_for_agent)

            # MixingNetwork用のグローバル状態テンソルを生成
            # obs_dict_itemからGlobalState辞書を再構築し、それをフラット化
            # この再構築は、_flatten_global_state_dictがGlobalState dictを期待するため
            reconstructed_global_state: GlobalState = {}

            # ゴール座標を抽出
            # どのアージェントの観測辞書を使っても'all_goals'は同じはず
            if self._agent_ids and self._agent_ids[0] in obs_dict_item and 'all_goals' in obs_dict_item[self._agent_ids[0]]:
                for i, goal_pos_tuple in enumerate(obs_dict_item[self._agent_ids[0]]['all_goals']):
                    # goal_pos_tupleはリストとして読み込まれる可能性があるので、タプルに変換
                    reconstructed_global_state[self._goal_ids[i]] = tuple(goal_pos_tuple)

            # エージェント自身の座標を抽出
            for agent_id in self._agent_ids:
                if agent_id in obs_dict_item and 'self' in obs_dict_item[agent_id]:
                    reconstructed_global_state[agent_id] = tuple(obs_dict_item[agent_id]['self'])

            # 再構築したGlobalStateをフラットなNumPy配列に変換し、テンソル化
            flattened_global_state_np = self._flatten_global_state_dict(reconstructed_global_state)
            flattened_global_state_tensor = torch.tensor(flattened_global_state_np, dtype=torch.float32, device=self.device)
            all_flattened_global_states_for_mixing.append(flattened_global_state_tensor)

        # 最終的なテンソルにスタック
        transformed_obs_batch = torch.stack(all_transformed_obs_for_agents, dim=0) # (B*N, C, G, G)
        global_state_tensor_for_mixing_network = torch.stack(all_flattened_global_states_for_mixing, dim=0) # (B, state_dim)

        return transformed_obs_batch, global_state_tensor_for_mixing_network


    # --- 既存のメソッドはそのまま維持 ---

    def _flatten_global_state_dict(self, state_dict: GlobalState) -> np.ndarray:
        """
        GlobalState辞書をStateProcessorが期待する順序でフラットなNumPy配列に変換します。
        順序はゴールID、次にエージェントIDの座標です。
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

        return np.array(flat_coords, dtype=np.float32)

    # _get_agent_q_values method now expects transformed_obs_batch and agent_ids_batch directly
    def _get_agent_q_values(
        self,
        agent_network_instance: AgentNetwork,
        transformed_obs_batch_for_q_values: torch.Tensor, # (B*N, C, G, G) from _process_raw_observations_batch
        agent_ids_batch_for_q_values: torch.Tensor # (B*N,) agent_ids for batch processing, already flattened
    ) -> torch.Tensor:
        """
        与えられたAgentNetworkインスタンス（メインまたはターゲット）から、
        バッチ内の全エージェントの全アクションに対するQ値を計算します。

        Args:
            agent_network_instance (AgentNetwork): Q値を計算するAgentNetworkのインスタンス.
            transformed_obs_batch_for_q_values (torch.Tensor): AgentNetworkへの入力 (形状: (batch_size * n_agents, num_channels, grid_size, grid_size)).
            agent_ids_batch_for_q_values (torch.Tensor): バッチ内の各エージェントのID (形状: (batch_size * n_agents,)).

        Returns:
            torch.Tensor: 各エージェントの各アクションに対するQ値 (形状: (batch_size, n_agents, action_size)).
        """
        batch_size = transformed_obs_batch_for_q_values.size(0) // self.n_agents

        # AgentNetworkからQ値を計算
        # agent_network_instance((B*N, C, G, G), (B*N,)) -> (B*N, A)
        q_values_flat = agent_network_instance(transformed_obs_batch_for_q_values, agent_ids_batch_for_q_values)

        # 結果を元の形状 (B, N, A) に戻す
        q_values_reshaped = q_values_flat.view(batch_size, self.n_agents, self.action_size)

        return q_values_reshaped

    @abstractmethod
    def get_actions(self, obs_dict_for_current_step: Dict[str, Dict[str, Any]], epsilon: float) -> Dict[str, int]:
        """
        与えられたグローバル状態とイプシロンに基づいて、各エージェントのアクションを選択します。
        具体的な実装はサブクラスで提供されます。
        """
        pass

    @abstractmethod
    def evaluate_q(
        self,
        obs_dicts_batch: List[Dict[str, Dict[str, Any]]],
        actions_batch: torch.Tensor,
        rewards_batch: torch.Tensor,
        next_obs_dicts_batch: List[Dict[str, Dict[str, Any]]],
        dones_batch: torch.Tensor,
        is_weights_batch: Optional[torch.Tensor] = None
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        """
        リプレイバッファからサンプリングされたバッチデータを使用して、Q値を評価し、
        損失とTD誤差を計算します。具体的な実装はサブクラスで提供されます。
        """
        pass

    def sync_target_network(self) -> None:
        """
        ターゲットAgentNetworkをメインAgentNetworkと同期します。
        メインネットワークの重みをターゲットネットワークにコピーします。
        """
        self.agent_network_target.load_state_dict(self.agent_network.state_dict())
        self.agent_network_target.eval()

    def _huber_loss(
        self,
        q: torch.Tensor,
        target: torch.Tensor,
        is_weights: Optional[torch.Tensor] = None
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Huber Loss を計算する. PERを使用する場合は重要度サンプリング重みを適用する.
        TD誤差も計算して返す.
        """
        td_errors: torch.Tensor = target - q
        abs_td_errors: torch.Tensor = torch.abs(td_errors)

        HUBER_LOSS_DELTA = 1.0
        cond: torch.Tensor = abs_td_errors < HUBER_LOSS_DELTA
        L2: torch.Tensor = 0.5 * torch.square(td_errors)
        L1: torch.Tensor = HUBER_LOSS_DELTA * (abs_td_errors - 0.5 * HUBER_LOSS_DELTA)
        loss_per_sample: torch.Tensor = torch.where(cond, L2, L1)

        if is_weights is not None:
            weighted_loss = loss_per_sample * is_weights.squeeze(-1) if is_weights.ndim > 1 else loss_per_sample * is_weights
            final_loss = torch.mean(weighted_loss)
        else:
            final_loss = torch.mean(loss_per_sample)

        return final_loss, abs_td_errors.detach()

    @abstractmethod
    def get_optimizer_params(self) -> List[nn.Parameter]:
        """
        オプティマイザが更新すべきネットワークパラメータのリストを返します。
        """
        pass
