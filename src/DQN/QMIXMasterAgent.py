import torch
import torch.nn as nn
import numpy as np
from typing import List, Optional, Tuple, Dict, Any

from Base.Agent_Base import AgentNetwork, BaseMasterAgent, ObsToTensorWrapper

from .network import MixingNetwork

class QMIXMasterAgent(BaseMasterAgent):
    """
    QMIX (Q-value Mixing Network) の Master Agent クラス.
    各エージェントは共有の AgentNetwork を使用してQ値を学習し、
    MixingNetwork を介してチーム全体のQ値 (Q_tot) を協調的に学習します。
    """
    def __init__(self,
                 n_agents: int,
                 action_size: int,
                 grid_size: int,
                 goals_number: int,
                 device: torch.device,
                 state_processor: ObsToTensorWrapper,
                 agent_network: AgentNetwork,
                 gamma: float,
                 agent_ids: List[str],
                 goal_ids: List[str]):
        super().__init__(
            n_agents=n_agents,
            action_size=action_size,
            grid_size=grid_size,
            goals_number=goals_number,
            device=device,
            state_processor=state_processor,
            agent_network=agent_network,
            agent_ids=agent_ids,
            goal_ids=goal_ids
        )
        self.gamma = gamma

        # MixingNetworkのstate_dimを計算: グローバル状態のフラット化されたサイズ
        # (目標数 + エージェント数) * 各位置の次元 (x, y)
        mixing_network_state_dim = (self.goals_num + self.n_agents) * 2
        self.mixing_network = MixingNetwork(n_agents, mixing_network_state_dim).to(device)
        self.mixing_network_target = MixingNetwork(n_agents, mixing_network_state_dim).to(device)
        self.mixing_network_target.load_state_dict(self.mixing_network.state_dict())
        self.mixing_network_target.eval() # ターゲットネットワークは推論モード

    def get_optimizer_params(self) -> List[nn.Parameter]:
        """
        QMIXMasterAgent の場合は、agent_network と mixing_network の両方のパラメータを返します。
        """
        params = list(self.agent_network.parameters())
        params.extend(list(self.mixing_network.parameters()))
        return params

    def get_actions(self, obs_dict_for_current_step: Dict[str, Dict[str, Any]], epsilon: float) -> Dict[str, int]:
        """
        与えられたグローバル状態（ローカル観測辞書）とイプシロンに基づいて、各エージェントのアクションを選択します。
        """
        self.agent_network.eval() # ネットワークを評価モードに設定
        with torch.no_grad():
            transformed_obs_list_for_all_agents = []
            for agent_id in self._agent_ids:
                single_agent_obs = obs_dict_for_current_step[agent_id]
                # StateProcessor.transform_state_batchは単一エージェントの観測辞書を受け取る
                transformed_obs = self.state_processor.transform_state_batch(single_agent_obs)
                transformed_obs_list_for_all_agents.append(transformed_obs)

            # 全エージェントの変換済み観測をスタック: (N, C, G, G)
            transformed_obs_for_all_agents = torch.stack(transformed_obs_list_for_all_agents, dim=0)

            # エージェントIDバッチの作成 (N,)
            # 各エージェントIDは一度だけ必要。unsqueeze(0)やrepeatは不要。
            agent_ids_for_all_agents = torch.arange(self.n_agents, dtype=torch.long, device=self.device)

            # AgentNetwork からQ値を計算 (N, action_size)
            q_values_all_agents = self.agent_network(transformed_obs_for_all_agents, agent_ids_for_all_agents)

            # ε-greedyポリシーを適用
            actions: Dict[str, int] = {}
            for i, aid in enumerate(self._agent_ids):
                if np.random.rand() < epsilon:
                    actions[aid] = np.random.randint(self.action_size)
                else:
                    actions[aid] = q_values_all_agents[i].argmax().item()

        self.agent_network.train() # ネットワークを学習モードに戻す
        return actions

    def evaluate_q(
        self,
        obs_dicts_batch: List[Dict[str, Dict[str, Any]]], # List of observation dictionaries
        actions_batch: torch.Tensor, # (batch_size, n_agents)
        rewards_batch: torch.Tensor, # (batch_size, n_agents) - Changed from (batch_size,)
        next_obs_dicts_batch: List[Dict[str, Dict[str, Any]]], # List of observation dictionaries
        dones_batch: torch.Tensor, # (batch_size, n_agents) - Individual done flags (0 if alive, 1 if dead)
        is_weights_batch: Optional[torch.Tensor] = None # (batch_size,)
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        """
        QMIXモードにおけるQ値の評価と損失計算を行います。
        """
        self.agent_network.train()
        self.mixing_network.train()
        self.agent_network_target.eval()
        self.mixing_network_target.eval()

        batch_size = len(obs_dicts_batch)

        # Process raw observation dictionaries into tensors for AgentNetwork and MixingNetwork
        current_transformed_obs_batch, current_global_state_for_mixing = self._process_raw_observations_batch(obs_dicts_batch)
        next_transformed_obs_batch, next_global_state_for_mixing = self._process_raw_observations_batch(next_obs_dicts_batch)

        # Agent IDs for batch processing
        # This needs to be (B*N,) for the _get_agent_q_values method
        agent_ids_for_q_values = torch.arange(self.n_agents, dtype=torch.long, device=self.device).repeat(batch_size) # (batch_size * n_agents,)

        # STEP 1: 各エージェントの全アクションQ値の算出 (メインネットワーク)
        # _get_agent_q_values expects (B*N, C, G, G) and (B*N,) agent_ids
        # Returns (batch_size, n_agents, action_size)
        current_q_values_all_agents = self._get_agent_q_values(self.agent_network, current_transformed_obs_batch, agent_ids_for_q_values)

        # STEP 2: 選択されたアクションのQ値抽出 (メインネットワーク)
        # actions_batch (B, N) -> (B, N, 1) for gather
        chosen_q_values = current_q_values_all_agents.gather(2, actions_batch.unsqueeze(-1)) # (batch_size, n_agents, 1)

        # 生存マスクの適用 (Q_i): 脱落したエージェントのQ値を0にする
        # dones_batch (B, N) -> (B, N, 1)
        chosen_q_values_masked = chosen_q_values * (1 - dones_batch.unsqueeze(-1)) # (batch_size, n_agents, 1)

        # STEP 3: MixingNetwork への入力整形と Q_tot の算出 (メインネットワーク)
        # current_global_state_for_mixing has shape (batch_size, state_dim)
        Q_tot = self.mixing_network(chosen_q_values_masked, current_global_state_for_mixing) # (batch_size, 1)

        # ターゲットQ_totの計算 (ターゲットネットワーク)
        with torch.no_grad():
            # 次の状態の各エージェントのQ値 (ターゲットAgentNetwork)
            # _get_agent_q_values expects (B*N, C, G, G) and (B*N,) agent_ids
            # Returns (batch_size, n_agents, action_size)
            next_q_values_all_agents_target = self._get_agent_q_values(self.agent_network_target, next_transformed_obs_batch, agent_ids_for_q_values)

            # 各エージェントの次の状態での最大Q値 (ターゲットAgentNetwork)
            # (batch_size, n_agents, 1)
            next_max_q_values_target = next_q_values_all_agents_target.max(dim=2, keepdim=True)[0] # keepdim=True for (B, N, 1)

            # 生存マスクの適用 (次のQ_i): 脱落したエージェントの次のQ値を0にする
            next_max_q_values_target_masked = next_max_q_values_target * (1 - dones_batch.unsqueeze(-1)) # (batch_size, n_agents, 1)

            # ターゲットMixingNetwork を介して target_Q_tot_unmasked を算出
            # next_global_state_for_mixing has shape (batch_size, state_dim)
            target_Q_tot_unmasked = self.mixing_network_target(next_max_q_values_target_masked, next_global_state_for_mixing) # (batch_size, 1)

            # Bellman方程式による最終的なターゲットQ_tot
            # rewards_batch (B, N) を (B, 1) のチーム報酬に合計
            team_rewards_batch = rewards_batch.sum(dim=1, keepdim=True) # (batch_size, 1)
            target_Q_tot = team_rewards_batch + self.gamma * target_Q_tot_unmasked # (batch_size, 1)

        # 損失の計算
        # _huber_lossは(batch_size,)を期待するので、squeeze(-1)で整形
        loss, abs_td_errors = self._huber_loss(Q_tot.squeeze(-1), target_Q_tot.squeeze(-1), is_weights_batch)

        return loss, abs_td_errors.detach()

    def sync_target_network(self) -> None:
        """
        ターゲットAgentNetwork と ターゲットMixingNetwork をメインネットワークと同期します。
        """
        super().sync_target_network() # BaseMasterAgentのAgentNetwork同期を呼び出す
        self.mixing_network_target.load_state_dict(self.mixing_network.state_dict())
        self.mixing_network_target.eval()
