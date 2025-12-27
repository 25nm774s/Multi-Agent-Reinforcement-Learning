import torch
import numpy as np
from typing import List, Optional, Tuple

from ..Base.Agent_Base import BaseMasterAgent, GlobalState, StateProcessor, AgentNetwork

from utils.StateProcesser import StateProcessor

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

    def get_actions(self, global_state: GlobalState, epsilon: float) -> List[int]:
        """
        与えられたグローバル状態とイプシロンに基づいて、各エージェントのアクションを選択します。
        """
        self.agent_network.eval() # ネットワークを評価モードに設定
        with torch.no_grad():
            # グローバル状態をバッチ次元を追加して StateProcessor に渡す
            # global_state はタプルなので、まずnp.arrayに変換してからテンソル化、unsqueze(0)でバッチ次元を追加
            flat_global_state_np = np.array(global_state).flatten() # Ensure it's flattened
            global_state_tensor = torch.tensor(flat_global_state_np, dtype=torch.float32, device=self.device).unsqueeze(0)

            # 各エージェントの観測を生成するためにStateProcessorを呼び出す
            # StateProcessorはバッチ処理された生の状態を受け取ることを想定
            # ここでは単一のグローバル状態を、各エージェントの観測形式に変換
            # 各エージェントの状態を生成 (1, C, G, G) * N_agents
            obs_for_all_agents = torch.cat([
                self.state_processor.transform_state_batch(i, global_state_tensor)
                for i in range(self.n_agents)
            ], dim=0) # (N_agents, C, G, G)

            # エージェントIDバッチの作成 (N_agents,)
            agent_ids_for_all_agents = torch.arange(self.n_agents, dtype=torch.long, device=self.device)

            # AgentNetwork からQ値を計算 (N_agents, action_size)
            q_values_all_agents = self.agent_network(obs_for_all_agents, agent_ids_for_all_agents)

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
        obs_batch: torch.Tensor,
        actions_batch: torch.Tensor, # (batch_size, n_agents)
        rewards_batch: torch.Tensor, # (batch_size,) - This is a team reward
        next_obs_batch: torch.Tensor,
        dones_batch: torch.Tensor, # (batch_size, n_agents) - Individual done flags (0 if alive, 1 if dead)
        is_weights_batch: Optional[torch.Tensor] = None # (batch_size,)
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        """
        IQLモードにおけるQ値の評価と損失計算を行います。
        """
        self.agent_network.train()
        self.agent_network_target.eval()

        batch_size = obs_batch.size(0)
        
        # Agent IDs for current and next state processing
        agent_ids_batch_for_q_values = torch.arange(self.n_agents, dtype=torch.long, device=self.device).unsqueeze(0).repeat(batch_size, 1) # (batch_size, n_agents)

        # Current Q-values from main network
        # (batch_size, n_agents, action_size)
        current_q_values_all_agents = self._get_agent_q_values(self.agent_network, obs_batch, agent_ids_batch_for_q_values)

        # Q-values for the actions actually taken by each agent
        # actions_batch (B, N) -> (B, N, 1) for gather
        current_q_values_taken_actions = current_q_values_all_agents.gather(2, actions_batch.unsqueeze(-1)).squeeze(-1) # (batch_size, n_agents)

        # Next Q-values from target network
        # (batch_size, n_agents, action_size)
        with torch.no_grad():
            next_q_values_all_agents = self._get_agent_q_values(self.agent_network_target, next_obs_batch, agent_ids_batch_for_q_values)

            # Max Q-values for next states, for each agent
            next_max_q_values = next_q_values_all_agents.max(dim=2)[0] # (batch_size, n_agents)

            # Apply individual done flags to next_max_q_values
            # If agent is 'done' (dones_batch[idx, agent_id] == 1), its next Q-value should be 0
            # (1 - dones_batch) will be 1 if not done, 0 if done
            # next_max_q_values_masked = next_max_q_values * (1 - dones_batch)

            # TD Target calculation for each agent
            # rewards_batch (B,) is shared. Expand it for each agent.
            # We use team reward, but individual dones.
            # Target = R + gamma * max_Q(s', a') * (1 - done)
            expanded_rewards_batch = rewards_batch.unsqueeze(1).repeat(1, self.n_agents) # (batch_size, n_agents)
            
            # Use dones_batch as mask. 1 if done, 0 if not done. So (1 - dones_batch) becomes 0 if done, 1 if not done.
            target_q_values_all_agents = expanded_rewards_batch + self.gamma * next_max_q_values * (1 - dones_batch)

        # Calculate Huber Loss for each agent independently
        # We need to reshape current_q_values_taken_actions and target_q_values_all_agents
        # to (batch_size * n_agents,) for _huber_loss if it expects (batch_size,) input
        
        # For IQL, we calculate loss for each agent and then average, 
        # or calculate a single loss over all (batch_size * n_agents) transitions.
        # Let's calculate a single loss over all (batch_size * n_agents) transitions for simplicity and efficiency.
        
        # Flatten Q-values and targets for loss calculation
        current_q_flat = current_q_values_taken_actions.view(-1) # (batch_size * n_agents,)
        target_q_flat = target_q_values_all_agents.view(-1)     # (batch_size * n_agents,)

        # If PER is used, IS weights need to be expanded/repeated for each agent
        expanded_is_weights = None
        if is_weights_batch is not None:
            expanded_is_weights = is_weights_batch.unsqueeze(1).repeat(1, self.n_agents).view(-1) # (batch_size * n_agents,)

        loss, abs_td_errors_flat = self._huber_loss(current_q_flat, target_q_flat, expanded_is_weights)

        # Reshape TD errors back to (batch_size, n_agents) if needed, 
        # but for PER update, we generally need (batch_size,) if each sample in replay buffer is a single transition
        # If the priority is per original sample in ReplayBuffer (i.e. one global_state -> next_global_state transition),
        # then the TD error for priority update should be averaged/maxed across agents for that sample.
        
        # For IQL, usually PER operates on the whole (s,a,r,s',d) tuple from the buffer.
        # So we should average (or take max) abs_td_errors across agents for each sample in the batch.
        
        # Average abs_td_errors across agents for each sample in the batch
        abs_td_errors_per_sample = abs_td_errors_flat.view(batch_size, self.n_agents).mean(dim=1) # (batch_size,)

        return loss, abs_td_errors_per_sample.detach()
