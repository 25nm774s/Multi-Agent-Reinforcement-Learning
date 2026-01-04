import torch
import torch.nn as nn
import numpy as np
from typing import List, Optional, Tuple, Dict, Any

from Base.Agent_Base import BaseMasterAgent, StateProcessor, AgentNetwork

class IQLMasterAgent(BaseMasterAgent):
    """
    Independent Q-Learning (IQL) の Master Agent クラス.
    各エージェントは共有の AgentNetwork を使用してQ値を学習しますが、
    Q値の評価と行動選択は独立して行われます。
    """
    def __init__(self,
                 n_agents: int,
                 action_size: int,
                 grid_size: int,
                 goals_number: int,
                 device: torch.device,
                 state_processor: StateProcessor,
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

    def get_optimizer_params(self) -> List[nn.Parameter]:
        """
        IQLMasterAgent の場合は、agent_network のパラメータのみを返します。
        """
        return list(self.agent_network.parameters())

    def get_actions(self, obs_dict_for_current_step: Dict[str, Dict[str, Any]], epsilon: float) -> List[int]:
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
            agent_ids_for_all_agents = torch.arange(self.n_agents, dtype=torch.long, device=self.device)

            # AgentNetwork からQ値を計算 (N, action_size)
            q_values_all_agents = self.agent_network(transformed_obs_for_all_agents, agent_ids_for_all_agents)

            # ε-greedyポリシーを適用
            actions: List[int] = []
            for i in range(self.n_agents):
                if np.random.rand() < epsilon:
                    actions.append(np.random.randint(self.action_size))
                else:
                    actions.append(q_values_all_agents[i].argmax().item())
        self.agent_network.train() # ネットワークを学習モードに戻す
        return actions

    def evaluate_q(
        self,
        obs_dicts_batch: List[Dict[str, Dict[str, Any]]],
        actions_batch: torch.Tensor, # (batch_size, n_agents)
        rewards_batch: torch.Tensor, # (batch_size,) - This is a team reward
        next_obs_dicts_batch: List[Dict[str, Dict[str, Any]]],
        dones_batch: torch.Tensor, # (batch_size, n_agents) - Individual done flags
        is_weights_batch: Optional[torch.Tensor] = None # (batch_size,)
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        """
        IQLモードにおけるQ値の評価と損失計算を行います。
        """
        self.agent_network.train()
        self.agent_network_target.eval()

        batch_size = len(obs_dicts_batch)

        # Process raw observation dictionaries into tensors for AgentNetwork and MixingNetwork
        current_transformed_obs_batch, _ = self._process_raw_observations_batch(obs_dicts_batch)
        next_transformed_obs_batch, _ = self._process_raw_observations_batch(next_obs_dicts_batch)

        # Agent IDs for current and next state processing
        # This needs to be (B*N,) for the _get_agent_q_values method
        # We create a base sequence of agent_ids (0 to N-1) and repeat it for each item in the batch
        agent_ids_for_q_values = torch.arange(self.n_agents, dtype=torch.long, device=self.device).repeat(batch_size)

        # Current Q-values from main network
        # (batch_size * n_agents, action_size) -> reshaped to (batch_size, n_agents, action_size)
        current_q_values_all_agents = self._get_agent_q_values(self.agent_network, current_transformed_obs_batch, agent_ids_for_q_values)

        # Q-values for the actions actually taken by each agent
        # actions_batch (B, N) -> (B, N, 1) for gather
        current_q_values_taken_actions = current_q_values_all_agents.gather(2, actions_batch.unsqueeze(-1)).squeeze(-1) # (batch_size, n_agents)

        # Next Q-values from target network
        with torch.no_grad():
            next_q_values_all_agents_target = self._get_agent_q_values(self.agent_network_target, next_transformed_obs_batch, agent_ids_for_q_values)

            # Max Q-values for next states, for each agent
            next_max_q_values = next_q_values_all_agents_target.max(dim=2)[0] # (batch_size, n_agents)

            # Apply individual done flags to next_max_q_values
            # If agent is 'done' (dones_batch[idx, agent_id] == 1), its next Q-value should be 0
            # (1 - dones_batch) will be 1 if not done, 0 if done

            # TD Target calculation for each agent
            # Target = R + gamma * max_Q(s', a') * (1 - done)
            expanded_rewards_batch = rewards_batch.unsqueeze(1).repeat(1, self.n_agents) # (batch_size, n_agents)

            # Use dones_batch as mask. 1 if done, 0 if not done. So (1 - dones_batch) becomes 0 if done, 1 if not done.
            target_q_values_all_agents = expanded_rewards_batch + self.gamma * next_max_q_values * (1 - dones_batch)

        # Calculate Huber Loss for each agent independently
        # Flatten Q-values and targets for loss calculation
        current_q_flat = current_q_values_taken_actions.view(-1) # (batch_size * n_agents,)
        target_q_flat = target_q_values_all_agents.view(-1)     # (batch_size * n_agents,)

        # If PER is used, IS weights need to be expanded/repeated for each agent
        expanded_is_weights = None
        if is_weights_batch is not None:
            expanded_is_weights = is_weights_batch.unsqueeze(1).repeat(1, self.n_agents).view(-1) # (batch_size * n_agents,)

        loss, abs_td_errors_flat = self._huber_loss(current_q_flat, target_q_flat, expanded_is_weights)

        # Average abs_td_errors across agents for each sample in the batch
        abs_td_errors_per_sample = abs_td_errors_flat.view(batch_size, self.n_agents).mean(dim=1) # (batch_size,)

        return loss, abs_td_errors_per_sample.detach()
