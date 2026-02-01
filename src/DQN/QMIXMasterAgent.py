import torch
from typing import Optional, Tuple

from Base.Agent_Base import MixerBasedMasterAgent

from .network import MixingNetwork

class QMIXMasterAgent(MixerBasedMasterAgent):
    """
    QMIX (Q-value Mixing Network) の Master Agent クラス.
    各エージェントは共有の AgentNetwork を使用してQ値を学習し,
    MixingNetwork を介してチーム全体のQ値 (Q_tot) を協調的に学習します。
    """
    def __init__(
        self,
        gamma: float,
        **kwargs
    ):
        # MixingNetworkのstate_dimを計算: グローバル状態のフラット化されたサイズ
        # (目標数 + エージェント数) * 各位置の次元 (x, y)
        mixing_network_state_dim = (kwargs['goals_number'] + kwargs['n_agents']) * 2
        mixer_network_instance = MixingNetwork(kwargs['n_agents'], mixing_network_state_dim).to(kwargs['device'])

        super().__init__(
            mixer_network_instance=mixer_network_instance,
            **kwargs
        )
        self.gamma = gamma


    # get_optimizer_params は MixerBasedMasterAgent の実装をそのまま使用

    # get_actions は BaseMasterAgent の実装をそのまま使用

    def evaluate_q(
        self,
        agent_obs_tensor_batch: torch.Tensor, # (B, N, C, G, G)
        global_state_tensor_batch: torch.Tensor, # (B, state_dim)
        actions_tensor_batch: torch.Tensor, # (B, N)
        rewards_tensor_batch: torch.Tensor, # (B, N)
        dones_tensor_batch: torch.Tensor, # (B, N)
        all_agents_done_batch: torch.Tensor, # (B, 1) - NEW
        next_agent_obs_tensor_batch: torch.Tensor, # (B, N, C, G, G)
        next_global_state_tensor_batch: torch.Tensor, # (B, state_dim)
        is_weights_batch: Optional[torch.Tensor] = None
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        """
        QMIXモードにおけるQ値の評価と損失計算を行います。
        """
        self.agent_network.train()
        self.mixer_network.train()
        self.agent_network_target.eval()
        self.mixer_network_target.eval()

        batch_size = agent_obs_tensor_batch.size(0)

        # AgentNetworkに渡すために(B*N, C, G, G)にreshape
        current_agent_obs_flat = agent_obs_tensor_batch.view(batch_size * self.n_agents, self.num_channels, self.grid_size, self.grid_size)
        next_agent_obs_flat = next_agent_obs_tensor_batch.view(batch_size * self.n_agents, self.num_channels, self.grid_size, self.grid_size)

        # Agent IDs for batch processing
        # This needs to be (B*N,) for the _get_agent_q_values method
        agent_ids_for_q_values = torch.arange(self.n_agents, dtype=torch.long, device=self.device).repeat(batch_size) # (batch_size * n_agents,)

        # STEP 1: 各エージェントの全アクションQ値の算出 (メインネットワーク)
        # Returns (batch_size, n_agents, action_size)
        current_q_values_all_agents = self._get_agent_q_values(self.agent_network, current_agent_obs_flat, agent_ids_for_q_values)

        # STEP 2: 選択されたアクションのQ値抽出 (メインネットワーク)
        # actions_tensor_batch (B, N) -> (B, N, 1) for gather
        chosen_q_values = current_q_values_all_agents.gather(2, actions_tensor_batch.unsqueeze(-1)) # (batch_size, n_agents, 1)

        # 生存マスクの適用 (Q_i): 脱落したエージェントのQ値を0にする
        # dones_tensor_batch (B, N) -> (B, N, 1)
        # NOTE: For QMIX, individual dones are used to mask *individual agent Q-values* before mixing.
        # This is distinct from the 'team_done_scalar' which applies to the total Q-value.
        chosen_q_values_masked = chosen_q_values * (1 - dones_tensor_batch.unsqueeze(-1)) # (batch_size, n_agents, 1)

        # STEP 3: MixingNetwork への入力整形と Q_tot の算出 (メインネットワーク)
        # global_state_tensor_batch has shape (batch_size, state_dim)
        # ミキサーへの入力Q値の定義: (batch_size, n_agents, 1)に統一済み
        # ミキサーへのグローバル状態の入力: 引数として受け取りつつ、内部で無視する実装を維持。QMIXは使用。
        Q_tot = self.mixer_network(chosen_q_values_masked, global_state_tensor_batch) # (batch_size, 1)

        # ターゲットQ_totの計算 (ターゲットネットワーク)
        with torch.no_grad():
            # 次の状態の各エージェントのQ値 (ターゲットAgentNetwork)
            # Returns (batch_size, n_agents, action_size)
            next_q_values_all_agents_target = self._get_agent_q_values(self.agent_network_target, next_agent_obs_flat, agent_ids_for_q_values)

            # 各エージェントの次の状態での最大Q値 (ターゲットAgentNetwork)
            # (batch_size, n_agents, 1)
            next_max_q_values_target = next_q_values_all_agents_target.max(dim=2, keepdim=True)[0] # keepdim=True for (B, N, 1)

            # 生存マスクの適用 (次のQ_i): 脱落したエージェントの次のQ値を0にする
            next_max_q_values_target_masked = next_max_q_values_target * (1 - dones_tensor_batch.unsqueeze(-1)) # (batch_size, n_agents, 1)

            # --- Reward and Done processing for QMIX using the new helper ---
            # For QMIX, we need the scalar team reward and team done status.
            _, _, team_reward_scalar, team_done_scalar = self._process_rewards_and_dones(
                rewards_tensor_batch, dones_tensor_batch, all_agents_done_batch
            )

            # ターゲットMixingNetwork を介して target_Q_tot_unmasked を算出
            # next_global_state_tensor_batch has shape (batch_size, state_dim)
            # ミキサーへの入力Q値の定義: (batch_size, n_agents, 1)に統一済み
            # ミキサーへのグローバル状態の入力: 引数として受け取りつつ、内部で無視する実装を維持。QMIXは使用。
            target_Q_tot_unmasked = self.mixer_network_target(next_max_q_values_target_masked, next_global_state_tensor_batch) # (batch_size, 1)

            # Bellman方程式による最終的なターゲットQ_tot
            # The team_reward_scalar is already (B, 1) and processed according to agent_reward_processing_mode
            target_Q_tot = team_reward_scalar + self.gamma * target_Q_tot_unmasked * (1 - team_done_scalar) # (batch_size, 1)

        # 損失の計算
        # _huber_lossは(batch_size,)を期待するので、squeeze(-1)で整形
        loss, abs_td_errors = self._huber_loss(Q_tot.squeeze(-1), target_Q_tot.squeeze(-1), is_weights_batch)

        return loss, abs_td_errors.detach()

    # sync_target_network は MixerBasedMasterAgent の実装をそのまま使用

