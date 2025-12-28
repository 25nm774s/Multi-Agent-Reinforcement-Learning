import torch
import numpy as np
from typing import List, Optional, Tuple

from Base.Agent_Base import AgentNetwork, GlobalState, BaseMasterAgent, StateProcessor

from utils.StateProcesser import StateProcessor
from .dqn import MixingNetwork

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
                 goals_num: int,
                 device: torch.device,
                 state_processor: StateProcessor,
                 agent_network: AgentNetwork,
                 gamma: float): # Add gamma as a parameter
        super().__init__(
            n_agents=n_agents,
            action_size=action_size,
            grid_size=grid_size,
            goals_num=goals_num,
            device=device,
            state_processor=state_processor,
            agent_network=agent_network
        )
        self.gamma = gamma

        # MixingNetworkのstate_dimを計算: グローバル状態のフラット化されたサイズ
        # (目標数 + エージェント数) * 各位置の次元 (x, y)
        mixing_network_state_dim = (self.goals_num + self.n_agents) * 2
        self.mixing_network = MixingNetwork(n_agents, mixing_network_state_dim).to(device)
        self.mixing_network_target = MixingNetwork(n_agents, mixing_network_state_dim).to(device)
        self.mixing_network_target.load_state_dict(self.mixing_network.state_dict())
        self.mixing_network_target.eval() # ターゲットネットワークは推論モード

    def get_actions(self, global_state: GlobalState, epsilon: float) -> List[int]:
        """
        与えられたグローバル状態とイプシロンに基づいて、各エージェントのアクションを選択します。
        """
        self.agent_network.eval() # ネットワークを評価モードに設定
        with torch.no_grad():
            # グローバル状態をバッチ次元を追加して StateProcessor に渡す
            flat_global_state_np = np.array(global_state).flatten() # Ensure it's flattened
            global_state_tensor = torch.tensor(flat_global_state_np, dtype=torch.float32, device=self.device).unsqueeze(0)

            transformed_obs_list = []
            for agent_idx in range(self.n_agents):
                transformed_obs_for_agent = self.state_processor.transform_state_batch(
                    agent_idx, global_state_tensor
                ) # (1, num_channels, grid_size, grid_size)
                transformed_obs_list.append(transformed_obs_for_agent)

            obs_for_all_agents = torch.cat(transformed_obs_list, dim=0) # (n_agents, C, G, G)

            agent_ids_for_all_agents = torch.arange(self.n_agents, dtype=torch.long, device=self.device)

            # AgentNetwork からQ値を計算 (n_agents, action_size)
            q_values_all_agents = self.agent_network(obs_for_all_agents, agent_ids_for_all_agents)

            # ε-greedyポリシーを適用
            actions: List[int] = []
            for i in range(self.n_agents):
                if np.random.rand() < epsilon:
                    actions.append(np.random.randint(self.action_size))
                else:
                    actions.append(q_values_all_agents[i].argmax().item()) # 最大Q値のアクションを選択
        self.agent_network.train() # ネットワークを学習モードに戻す
        return actions

    def evaluate_q(
        self,
        global_states_batch_raw: torch.Tensor, # Raw flattened global state (B, total_features)
        actions_batch: torch.Tensor, # (batch_size, n_agents)
        rewards_batch: torch.Tensor, # (batch_size,) - This is a team reward
        next_global_states_batch_raw: torch.Tensor, # Raw flattened next global state (B, total_features)
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

        batch_size = global_states_batch_raw.size(0)

        # Agent IDs for batch processing
        agent_ids_batch_for_q_values = torch.arange(self.n_agents, dtype=torch.long, device=self.device).unsqueeze(0).repeat(batch_size, 1) # (batch_size, n_agents)

        # STEP 1: 各エージェントの全アクションQ値の算出 (メインネットワーク)
        # (batch_size, n_agents, action_size)
        current_q_values_all_agents = self._get_agent_q_values(self.agent_network, global_states_batch_raw, agent_ids_batch_for_q_values)

        # STEP 2: 選択されたアクションのQ値抽出 (メインネットワーク)
        # actions_batch (B, N) -> (B, N, 1) for gather
        chosen_q_values = current_q_values_all_agents.gather(2, actions_batch.unsqueeze(-1)) # (batch_size, n_agents, 1)

        # 生存マスクの適用 (Q_i): 脱落したエージェントのQ値を0にする
        # dones_batch (B, N) -> (B, N, 1)
        chosen_q_values_masked = chosen_q_values * (1 - dones_batch.unsqueeze(-1)) # (batch_size, n_agents, 1)

        # STEP 3: MixingNetwork への入力整形と Q_tot の算出 (メインネットワーク)
        Q_tot = self.mixing_network(chosen_q_values_masked, global_states_batch_raw) # (batch_size, 1)

        # ターゲットQ_totの計算 (ターゲットネットワーク)
        with torch.no_grad():
            # 次の状態の各エージェントのQ値 (ターゲットAgentNetwork)
            # (batch_size, n_agents, action_size)
            next_q_values_all_agents_target = self._get_agent_q_values(self.agent_network_target, next_global_states_batch_raw, agent_ids_batch_for_q_values)

            # 各エージェントの次の状態での最大Q値 (ターゲットAgentNetwork)
            # (batch_size, n_agents, 1)
            next_max_q_values_target = next_q_values_all_agents_target.max(dim=2, keepdim=True)[0] # keepdim=True for (B, N, 1)

            # 生存マスクの適用 (次のQ_i): 脱落したエージェントの次のQ値を0にする
            next_max_q_values_target_masked = next_max_q_values_target * (1 - dones_batch.unsqueeze(-1)) # (batch_size, n_agents, 1)

            # ターゲットMixingNetwork を介して target_Q_tot_unmasked を算出
            target_Q_tot_unmasked = self.mixing_network_target(next_max_q_values_target_masked, next_global_states_batch_raw) # (batch_size, 1)

            # Bellman方程式による最終的なターゲットQ_tot
            # rewards_batch (B,) -> (B, 1)
            target_Q_tot = rewards_batch.unsqueeze(-1) + self.gamma * target_Q_tot_unmasked # (batch_size, 1)

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
