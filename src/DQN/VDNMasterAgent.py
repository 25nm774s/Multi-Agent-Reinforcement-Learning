import numpy as np
import torch
import torch.nn as nn
from typing import Dict, List, Tuple, Optional

from Environments.StateProcesser import ObsToTensorWrapper
from .network import IAgentNetwork
from Base.Agent_Base import BaseMasterAgent

class VDNMasterAgent(BaseMasterAgent):
    """
    Value Decomposition Networks (VDN) の Master Agent クラス.
    各エージェントは共有の AgentNetwork を使用してQ値を学習し,
    チーム全体のQ値 (Q_tot) は個々のQ値の単純な合計として算出されます。
    """
    def __init__(self,
                 n_agents: int,
                 action_size: int,
                 grid_size: int,
                 goals_number: int,
                 device: torch.device,
                 state_processor: ObsToTensorWrapper,
                 agent_network: IAgentNetwork,
                 gamma: float,
                 agent_ids: List[str],
                 goal_ids: List[str],
                 agent_reward_processing_mode: str):
        super().__init__(
            n_agents=n_agents,
            action_size=action_size,
            grid_size=grid_size,
            goals_number=goals_number,
            device=device,
            state_processor=state_processor,
            agent_network_instance=agent_network,
            agent_ids=agent_ids,
            goal_ids=goal_ids,
            agent_reward_processing_mode=agent_reward_processing_mode
        )
        self.gamma = gamma

    def get_optimizer_params(self) -> List[nn.Parameter]:
        """
        VDNMasterAgent の場合は、agent_network のパラメータのみを返します。
        (Mixing networkは学習可能な重みを持たないため)
        """
        return list(self.agent_network.parameters())

    def get_actions(self, agent_obs_tensor: torch.Tensor, epsilon: float) -> Dict[str, int]:
        """
        与えられた現在のステップの観測とイプシロンに基づいて、各エージェントのアクションを選択します。
        これはIQLMasterAgentやQMIXMasterAgentのget_actionsと同一のロジックです。
        Args:
            agent_obs_tensor (torch.Tensor): 各エージェントのグリッド観測 (n_agents, num_channels, grid_size, grid_size)。
            epsilon (float): 探索率。
        Returns:
            Dict[str, int]: 各エージェントIDをキー、選択された行動を値とする辞書。
        """
        self.agent_network.eval()
        with torch.no_grad():
            agent_ids_for_all_agents = torch.arange(self.n_agents, dtype=torch.long, device=self.device)
            q_values_all_agents = self.agent_network(agent_obs_tensor, agent_ids_for_all_agents)

            actions: Dict[str, int] = {}
            for i, aid in enumerate(self._agent_ids):
                if np.random.rand() < epsilon:
                    actions[aid] = np.random.randint(self.action_size)
                else:
                    actions[aid] = q_values_all_agents[i].argmax().item()
        self.agent_network.train()
        return actions

    def evaluate_q(
        self,
        agent_obs_tensor_batch: torch.Tensor, # (B, N, C, G, G)
        global_state_tensor_batch: torch.Tensor, # (B, state_dim) - VDNでは直接使用しないが引数として受け取る
        actions_tensor_batch: torch.Tensor, # (B, N)
        rewards_tensor_batch: torch.Tensor, # (B, N)
        dones_tensor_batch: torch.Tensor, # (B, N)
        all_agents_done_batch: torch.Tensor, # (B, 1)
        next_agent_obs_tensor_batch: torch.Tensor, # (B, N, C, G, G)
        next_global_state_tensor_batch: torch.Tensor, # (B, state_dim) - VDNでは直接使用しないが引数として受け取る
        is_weights_batch: Optional[torch.Tensor] = None
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        """
        VDNモードにおけるQ値の評価と損失計算を行います。
        Q_totは個々のQ値の単純な合計です。
        """
        # 1. Set the main agent_network to training mode and the agent_network_target to evaluation mode.
        self.agent_network.train()
        self.agent_network_target.eval()

        # 2. Get the batch_size from agent_obs_tensor_batch.
        batch_size = agent_obs_tensor_batch.size(0)

        # 3. Reshape agent_obs_tensor_batch and next_agent_obs_tensor_batch into a flat tensor.
        current_agent_obs_flat = agent_obs_tensor_batch.view(batch_size * self.n_agents, self.num_channels, self.grid_size, self.grid_size)
        next_agent_obs_flat = next_agent_obs_tensor_batch.view(batch_size * self.n_agents, self.num_channels, self.grid_size, self.grid_size)

        # 4. Create a tensor agent_ids_for_q_values.
        agent_ids_for_q_values = torch.arange(self.n_agents, dtype=torch.long, device=self.device).repeat(batch_size) # (batch_size * n_agents,)

        # 5. Calculate current_q_values_all_agents.
        current_q_values_all_agents = self._get_agent_q_values(self.agent_network, current_agent_obs_flat, agent_ids_for_q_values)

        # 6. Extract the chosen_q_values (Q-values for the actions actually taken).
        # actions_tensor_batch (B, N) -> (B, N, 1) for gather
        chosen_q_values = current_q_values_all_agents.gather(2, actions_tensor_batch.unsqueeze(-1)) # (batch_size, n_agents, 1)

        # 7. Apply a masking operation to chosen_q_values using dones_tensor_batch.
        chosen_q_values_masked = chosen_q_values * (1 - dones_tensor_batch.unsqueeze(-1)) # (batch_size, n_agents, 1)

        # 8. Calculate Q_tot by summing chosen_q_values_masked across the agents dimension.
        Q_tot = chosen_q_values_masked.sum(dim=1) # (batch_size, 1)

        # 9. Inside a torch.no_grad() context:
        with torch.no_grad():
            # a. Calculate next_q_values_all_agents_target.
            next_q_values_all_agents_target = self._get_agent_q_values(self.agent_network_target, next_agent_obs_flat, agent_ids_for_q_values)

            # b. Determine next_max_q_values_target.
            next_max_q_values_target = next_q_values_all_agents_target.max(dim=2, keepdim=True)[0] # keepdim=True for (B, N, 1)

            # c. Apply a masking operation to next_max_q_values_target using dones_tensor_batch.
            next_max_q_values_target_masked = next_max_q_values_target * (1 - dones_tensor_batch.unsqueeze(-1)) # (batch_size, n_agents, 1)

            # d. Call self._process_rewards_and_dones.
            _, _, team_reward_scalar, team_done_scalar = self._process_rewards_and_dones(
                rewards_tensor_batch, dones_tensor_batch, all_agents_done_batch
            )

            # e. Calculate target_Q_tot_unmasked.
            target_Q_tot_unmasked = next_max_q_values_target_masked.sum(dim=1) # (batch_size, 1)

            # f. Compute the target_Q_tot using the Bellman equation.
            target_Q_tot = team_reward_scalar + self.gamma * target_Q_tot_unmasked * (1 - team_done_scalar) # (batch_size, 1)

        # 10. Calculate the loss and absolute TD errors by calling self._huber_loss.
        loss, abs_td_errors = self._huber_loss(Q_tot.squeeze(-1), target_Q_tot.squeeze(-1), is_weights_batch)

        # 11. Return the computed loss and the detached abs_td_errors.
        return loss, abs_td_errors.detach()

    def sync_target_network(self) -> None:
        """
        VDNMasterAgentの場合、ターゲットAgentNetworkのみをメインネットワークと同期します。
        MixingNetworkは存在しないため、IQLMasterAgentと同様の動作になります。
        """
        super().sync_target_network() # BaseMasterAgentのAgentNetwork同期を呼び出す
