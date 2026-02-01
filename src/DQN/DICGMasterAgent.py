import torch
from typing import Optional, Tuple

from Base.Agent_Base import MixerBasedMasterAgent
from .network import DICGMixer

class DICGMasterAgent(MixerBasedMasterAgent):
    """
    DICG (Difference Inidividual Contribution Global) の Master Agent クラス。
    各エージェントのQ値とグローバル状態を入力として受け取り、
    DICGMixer を介してチーム全体のQ値を学習します。
    """
    def __init__(
        self,
        gamma: float,
        **kwargs
    ):
        # DICGMixerのstate_dimを計算: グローバル状態のフラット化されたサイズ
        # (目標数 + エージェント数) * 各位置の次元 (x, y)
        mixing_network_state_dim = (kwargs['goals_number'] + kwargs['n_agents']) * 2
        mixer_network_instance = DICGMixer(kwargs['n_agents'], mixing_network_state_dim).to(kwargs['device'])

        super().__init__(
            mixer_network_instance=mixer_network_instance,
            **kwargs
        )
        self.gamma = gamma

    def evaluate_q(
        self,
        agent_obs_tensor_batch: torch.Tensor, # (B, N, C, G, G)
        global_state_tensor_batch: torch.Tensor, # (B, state_dim)
        actions_tensor_batch: torch.Tensor, # (B, N)
        rewards_tensor_batch: torch.Tensor, # (B, N)
        dones_tensor_batch: torch.Tensor, # (B, N)
        all_agents_done_batch: torch.Tensor, # (B, 1)
        next_agent_obs_tensor_batch: torch.Tensor, # (B, N, C, G, G)
        next_global_state_tensor_batch: torch.Tensor, # (B, state_dim)
        is_weights_batch: Optional[torch.Tensor] = None
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        """
        DICGモードにおけるQ値の評価と損失計算を行います。
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
        agent_ids_for_q_values = torch.arange(self.n_agents, dtype=torch.long, device=self.device).repeat(batch_size)

        # STEP 1: 各エージェントの全アクションQ値の算出 (メインネットワーク)
        current_q_values_all_agents = self._get_agent_q_values(self.agent_network, current_agent_obs_flat, agent_ids_for_q_values)

        # STEP 2: 選択されたアクションのQ値抽出 (メインネットワーク)
        chosen_q_values = current_q_values_all_agents.gather(2, actions_tensor_batch.unsqueeze(-1)) # (batch_size, n_agents, 1)

        # 生存マスクの適用 (Q_i): 脱落したエージェントのQ値を0にする
        chosen_q_values_masked = chosen_q_values * (1 - dones_tensor_batch.unsqueeze(-1)) # (batch_size, n_agents, 1)

        # STEP 3: DICGMixer を介した Q_tot の算出 (メインネットワーク)
        Q_tot = self.mixer_network(chosen_q_values_masked, global_state_tensor_batch) # (batch_size, 1)

        # ターゲットQ_totの計算 (ターゲットネットワーク)
        with torch.no_grad():
            # 次の状態の各エージェントのQ値 (ターゲットAgentNetwork)
            next_q_values_all_agents_target = self._get_agent_q_values(self.agent_network_target, next_agent_obs_flat, agent_ids_for_q_values)

            # 各エージェントの次の状態での最大Q値 (ターゲットAgentNetwork)
            next_max_q_values_target = next_q_values_all_agents_target.max(dim=2, keepdim=True)[0] # keepdim=True for (B, N, 1)

            # 生存マスクの適用 (次のQ_i): 脱落したエージェントの次のQ値を0にする
            next_max_q_values_target_masked = next_max_q_values_target * (1 - dones_tensor_batch.unsqueeze(-1)) # (batch_size, n_agents, 1)

            # Reward and Done processing for DICG (similar to QMIX/VDN, uses team reward/done)
            _, _, team_reward_scalar, team_done_scalar = self._process_rewards_and_dones(
                rewards_tensor_batch, dones_tensor_batch, all_agents_done_batch
            )

            # ターゲットDICGMixer を介して target_Q_tot_unmasked を算出
            target_Q_tot_unmasked = self.mixer_network_target(next_max_q_values_target_masked, next_global_state_tensor_batch) # (batch_size, 1)

            # Bellman方程式による最終的なターゲットQ_tot
            target_Q_tot = team_reward_scalar + self.gamma * target_Q_tot_unmasked * (1 - team_done_scalar) # (batch_size, 1)

        # 損失の計算
        loss, abs_td_errors = self._huber_loss(Q_tot.squeeze(-1), target_Q_tot.squeeze(-1), is_weights_batch)

        return loss, abs_td_errors.detach()

