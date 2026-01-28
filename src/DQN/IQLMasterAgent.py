import torch
import torch.nn as nn
import numpy as np
from typing import List, Optional, Tuple, Dict

from Base.Agent_Base import BaseMasterAgent, ObsToTensorWrapper, IAgentNetwork

class IQLMasterAgent(BaseMasterAgent):
    """
    Independent Q-Learning (IQL) の Master Agent クラス.
    各エージェントは共有の AgentNetwork を使用してQ値を学習しますが,
    Q値の評価と行動選択は独立して行われます。
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
                 agent_reward_processing_mode: str): # Add new argument
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
            agent_reward_processing_mode=agent_reward_processing_mode # Pass to base
        )
        self.gamma = gamma

    def get_optimizer_params(self) -> List[nn.Parameter]:
        """
        IQLMasterAgent の場合は、agent_network のパラメータのみを返します。
        """
        return list(self.agent_network.parameters())

    def get_actions(self, agent_obs_tensor: torch.Tensor, epsilon: float) -> Dict[str, int]:
        """
        与えられた現在のステップの観測とイプシロンに基づいて、各エージェントのアクションを選択します。

        Args:
            agent_obs_tensor (torch.Tensor): 各エージェントのグリッド観測 (n_agents, num_channels, grid_size, grid_size)。
            epsilon (float): 探索率。

        Returns:
            Dict[str, int]: 各エージェントIDをキー、選択された行動を値とする辞書。
        """
        self.agent_network.eval() # ネットワークを評価モードに設定
        with torch.no_grad():
            # agent_obs_tensor は (N, C, G, G) の形状で既に渡される
            # agent_ids_for_all_agents は (N,) の形状
            agent_ids_for_all_agents = torch.arange(self.n_agents, dtype=torch.long, device=self.device)

            # AgentNetwork からQ値を計算 (N, action_size)
            q_values_all_agents = self.agent_network(agent_obs_tensor, agent_ids_for_all_agents)

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
        IQLモードにおけるQ値の評価と損失計算を行います。
        """
        self.agent_network.train()
        self.agent_network_target.eval()

        batch_size = agent_obs_tensor_batch.size(0)

        # AgentNetworkに渡すために(B*N, C, G, G)にreshape
        current_agent_obs_flat = agent_obs_tensor_batch.view(batch_size * self.n_agents, self.num_channels, self.grid_size, self.grid_size)
        next_agent_obs_flat = next_agent_obs_tensor_batch.view(batch_size * self.n_agents, self.num_channels, self.grid_size, self.grid_size)

        # Agent IDs for current and next state processing
        # This needs to be (B*N,) for the _get_agent_q_values method
        agent_ids_for_q_values = torch.arange(self.n_agents, dtype=torch.long, device=self.device).repeat(batch_size)

        # Current Q-values from main network
        # (batch_size, n_agents, action_size)
        current_q_values_all_agents = self._get_agent_q_values(self.agent_network, current_agent_obs_flat, agent_ids_for_q_values)

        # Q-values for the actions actually taken by each agent
        # actions_tensor_batch (B, N) -> (B, N, 1) for gather
        current_q_values_taken_actions = current_q_values_all_agents.gather(2, actions_tensor_batch.unsqueeze(-1)).squeeze(-1) # (batch_size, n_agents)

        # Next Q-values from target network
        with torch.no_grad():
            next_q_values_all_agents_target = self._get_agent_q_values(self.agent_network_target, next_agent_obs_flat, agent_ids_for_q_values)

            # Max Q-values for next states, for each agent
            next_max_q_values = next_q_values_all_agents_target.max(dim=2)[0] # (batch_size, n_agents)

            # --- Reward and Done processing for IQL using the new helper ---
            processed_rewards_for_iql, processed_dones_for_iql, _, _ = self._process_rewards_and_dones(
                rewards_tensor_batch, dones_tensor_batch, all_agents_done_batch
            )

            # Apply individual done flags to next_max_q_values
            # Target = R + gamma * max_Q(s', a') * (1 - done)
            target_q_values_all_agents = processed_rewards_for_iql + self.gamma * next_max_q_values * (1 - processed_dones_for_iql)

        # Calculate Huber Loss for each agent independently
        # Flatten Q-values and targets for loss calculation
        current_q_flat = current_q_values_taken_actions.view(-1) # (batch_size * n_agents,)
        target_q_flat = target_q_values_all_agents.view(-1)     # (batch_size * n_agents,)

        # If PER is used, IS weights need to be expanded/repeated for each agent
        expanded_is_weights = None
        if is_weights_batch is not None:
            # is_weights_batch (B,) を (B*N,) に拡張
            expanded_is_weights = is_weights_batch.unsqueeze(1).repeat(1, self.n_agents).view(-1) # (batch_size * n_agents,)

        loss, abs_td_errors_flat = self._huber_loss(current_q_flat, target_q_flat, expanded_is_weights)

        # Average abs_td_errors across agents for each sample in the batch
        # (batch_size * n_agents,) -> (batch_size, n_agents) -> (batch_size,)
        abs_td_errors_per_sample = abs_td_errors_flat.view(batch_size, self.n_agents).mean(dim=1)

        return loss, abs_td_errors_per_sample.detach()
