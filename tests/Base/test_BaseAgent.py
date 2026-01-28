import unittest
from unittest.mock import MagicMock, patch
import torch
import torch.nn as nn
from typing import Dict, List, Tuple, Optional

from src.Base.Agent_Base import BaseMasterAgent
from src.DQN.network import IAgentNetwork
from src.Environments.StateProcesser import ObsToTensorWrapper

class DummyAgent(IAgentNetwork):
    def __init__(self, grid_size: int, output_size: int, total_agents: int = 1):
        super().__init__(grid_size, output_size, total_agents)
        self._grid_size = grid_size
        self._num_channels = 3 # Fixed for this environment
        self._total_agents = total_agents
        self._action_size = output_size
        # Dummy layers for state_dict to work
        self.dummy_layer = nn.Linear(1, 1)

    @property
    def grid_size(self) -> int:
        return self._grid_size

    @property
    def num_channels(self) -> int:
        return self._num_channels

    @property
    def total_agents(self) -> int:
        return self._total_agents

    @property
    def action_size(self) -> int:
        return self._action_size

    def forward(self, x: torch.Tensor, agent_ids: torch.Tensor) -> torch.Tensor:
        # Mock forward pass to return predictable Q-values
        # In real tests, this would be mocked, but here we need a callable network
        batch_n = x.size(0)
        return torch.randn(batch_n, self.action_size, device=x.device)

class DummyMasterAgent(BaseMasterAgent):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def get_actions(self, agent_obs_tensor: torch.Tensor, epsilon: float) -> Dict[str, int]:
        # Dummy implementation for abstract method
        return {aid: 0 for aid in self._agent_ids}

    def evaluate_q(
        self,
        agent_obs_tensor_batch: torch.Tensor,
        global_state_tensor_batch: torch.Tensor,
        actions_tensor_batch: torch.Tensor,
        rewards_tensor_batch: torch.Tensor,
        dones_tensor_batch: torch.Tensor,
        all_agents_done_batch: torch.Tensor,
        next_agent_obs_tensor_batch: torch.Tensor,
        next_global_state_tensor_batch: torch.Tensor,
        is_weights_batch: Optional[torch.Tensor] = None
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        # Dummy implementation for abstract method
        return torch.tensor(0.0), None

    def get_optimizer_params(self) -> List[nn.Parameter]:
        # Dummy implementation for abstract method
        return list(self.agent_network.parameters())

class TestBaseMasterAgent(unittest.TestCase):
    def setUp(self):
        self.n_agents = 2
        self.action_size = 5
        self.grid_size = 6
        self.goals_num = 2
        self.device = torch.device('cpu')
        self.agent_ids = [f'agent_{i}' for i in range(self.n_agents)]
        self.goal_ids = [f'goal_{i}' for i in range(self.goals_num)]

        self.mock_state_processor = MagicMock(spec=ObsToTensorWrapper)
        self.mock_state_processor.num_channels = 3

        self.dummy_agent_network = DummyAgent(
            grid_size=self.grid_size,
            output_size=self.action_size,
            total_agents=self.n_agents
        ).to(self.device)

        # Test each reward processing mode
        self.dummy_master_agent_individual = DummyMasterAgent(
            n_agents=self.n_agents, action_size=self.action_size, grid_size=self.grid_size, goals_number=self.goals_num,
            device=self.device, state_processor=self.mock_state_processor, agent_network_instance=self.dummy_agent_network,
            agent_ids=self.agent_ids, goal_ids=self.goal_ids, agent_reward_processing_mode='individual'
        )
        self.dummy_master_agent_sum = DummyMasterAgent(
            n_agents=self.n_agents, action_size=self.action_size, grid_size=self.grid_size, goals_number=self.goals_num,
            device=self.device, state_processor=self.mock_state_processor, agent_network_instance=self.dummy_agent_network,
            agent_ids=self.agent_ids, goal_ids=self.goal_ids, agent_reward_processing_mode='sum_and_distribute'
        )
        self.dummy_master_agent_mean = DummyMasterAgent(
            n_agents=self.n_agents, action_size=self.action_size, grid_size=self.grid_size, goals_number=self.goals_num,
            device=self.device, state_processor=self.mock_state_processor, agent_network_instance=self.dummy_agent_network,
            agent_ids=self.agent_ids, goal_ids=self.goal_ids, agent_reward_processing_mode='mean_and_distribute'
        )

    def test_init_sets_attributes(self):
        self.assertEqual(self.dummy_master_agent_individual.n_agents, self.n_agents)
        self.assertEqual(self.dummy_master_agent_individual.action_size, self.action_size)
        self.assertEqual(self.dummy_master_agent_individual.grid_size, self.grid_size)
        self.assertEqual(self.dummy_master_agent_individual.goals_num, self.goals_num)
        self.assertEqual(self.dummy_master_agent_individual.device, self.device)
        self.assertEqual(self.dummy_master_agent_individual.state_processor, self.mock_state_processor)
        self.assertEqual(self.dummy_master_agent_individual.agent_network, self.dummy_agent_network)
        self.assertIsNotNone(self.dummy_master_agent_individual.agent_network_target)
        self.assertEqual(self.dummy_master_agent_individual.agent_reward_processing_mode, 'individual')
        self.assertTrue(not self.dummy_master_agent_individual.agent_network_target.training) # Should be in eval mode

    def test_process_rewards_and_dones_individual(self):
        batch_size = 4
        rewards = torch.randn(batch_size, self.n_agents, device=self.device)
        dones = torch.randint(0, 2, (batch_size, self.n_agents), dtype=torch.float32, device=self.device)
        all_agents_done = torch.randint(0, 2, (batch_size, 1), dtype=torch.float32, device=self.device)

        proc_rewards_iql, proc_dones_iql, team_reward, team_done = \
            self.dummy_master_agent_individual._process_rewards_and_dones(rewards, dones, all_agents_done)

        self.assertTrue(torch.equal(proc_rewards_iql, rewards)) # IQL rewards should be individual
        self.assertTrue(torch.equal(proc_dones_iql, dones))     # IQL dones should be individual
        self.assertTrue(torch.equal(team_reward, rewards.sum(dim=1, keepdim=True))) # QMIX team reward is sum
        self.assertTrue(torch.equal(team_done, all_agents_done)) # QMIX team done is overall done

    def test_process_rewards_and_dones_sum_and_distribute(self):
        batch_size = 4
        rewards = torch.randn(batch_size, self.n_agents, device=self.device)
        dones = torch.randint(0, 2, (batch_size, self.n_agents), dtype=torch.float32, device=self.device)
        all_agents_done = torch.randint(0, 2, (batch_size, 1), dtype=torch.float32, device=self.device)

        proc_rewards_iql, proc_dones_iql, team_reward, team_done = \
            self.dummy_master_agent_sum._process_rewards_and_dones(rewards, dones, all_agents_done)

        expected_sum_rewards = rewards.sum(dim=1, keepdim=True).expand_as(rewards)
        self.assertTrue(torch.equal(proc_rewards_iql, expected_sum_rewards)) # IQL rewards should be summed and distributed
        self.assertTrue(torch.equal(proc_dones_iql, dones))     # IQL dones should be individual
        self.assertTrue(torch.equal(team_reward, rewards.sum(dim=1, keepdim=True))) # QMIX team reward is sum
        self.assertTrue(torch.equal(team_done, all_agents_done)) # QMIX team done is overall done

    def test_process_rewards_and_dones_mean_and_distribute(self):
        batch_size = 4
        rewards = torch.randn(batch_size, self.n_agents, device=self.device)
        dones = torch.randint(0, 2, (batch_size, self.n_agents), dtype=torch.float32, device=self.device)
        all_agents_done = torch.randint(0, 2, (batch_size, 1), dtype=torch.float32, device=self.device)

        proc_rewards_iql, proc_dones_iql, team_reward, team_done = \
            self.dummy_master_agent_mean._process_rewards_and_dones(rewards, dones, all_agents_done)

        expected_mean_rewards = rewards.mean(dim=1, keepdim=True).expand_as(rewards)
        self.assertTrue(torch.equal(proc_rewards_iql, expected_mean_rewards)) # IQL rewards should be mean and distributed
        self.assertTrue(torch.equal(proc_dones_iql, dones))     # IQL dones should be individual
        self.assertTrue(torch.equal(team_reward, rewards.mean(dim=1, keepdim=True))) # QMIX team reward is mean
        self.assertTrue(torch.equal(team_done, all_agents_done)) # QMIX team done is overall done

    def test_get_agent_q_values(self):
        batch_size = 4
        agent_obs_batch_flat = torch.randn(batch_size * self.n_agents, self.mock_state_processor.num_channels, self.grid_size, self.grid_size, device=self.device)
        agent_ids_batch_flat = torch.arange(self.n_agents, device=self.device).repeat(batch_size)

        # Mock the dummy_agent_network's forward method to return fixed values for predictability
        mock_q_values_flat = torch.randn(batch_size * self.n_agents, self.action_size, device=self.device)
        self.dummy_agent_network.forward = MagicMock(return_value=mock_q_values_flat)

        q_values_reshaped = self.dummy_master_agent_individual._get_agent_q_values(self.dummy_agent_network, agent_obs_batch_flat, agent_ids_batch_flat)

        self.assertEqual(q_values_reshaped.shape, (batch_size, self.n_agents, self.action_size))
        self.dummy_agent_network.forward.assert_called_once()

    def test_huber_loss_without_is_weights(self):
        q_values = torch.tensor([1.0, 2.0, 3.0], device=self.device)
        targets = torch.tensor([1.5, 1.0, 3.0], device=self.device)

        loss, abs_td_errors = self.dummy_master_agent_individual._huber_loss(q_values, targets)

        # Expected TD errors: [0.5, -1.0, 0.0]
        # Expected abs_td_errors: [0.5, 1.0, 0.0]
        # With HUBER_LOSS_DELTA = 1.0:
        #   - 0.5: 0.5 * 0.5^2 = 0.125
        #   - 1.0: 0.5 * 1.0^2 = 0.5
        #   - 0.0: 0.5 * 0.0^2 = 0.0
        # Mean loss = (0.125 + 0.5 + 0.0) / 3 = 0.625 / 3 = 0.208333...

        self.assertAlmostEqual(loss.item(), (0.5 * 0.5**2 + 0.5 * 1.0**2 + 0.5 * 0.0**2) / 3, places=5)
        self.assertTrue(torch.equal(abs_td_errors, torch.tensor([0.5, 1.0, 0.0], device=self.device)))

    def test_huber_loss_with_is_weights(self):
        q_values = torch.tensor([1.0, 2.0, 3.0], device=self.device)
        targets = torch.tensor([1.5, 1.0, 3.0], device=self.device)
        is_weights = torch.tensor([0.1, 0.5, 1.0], device=self.device) # Example weights

        loss, abs_td_errors = self.dummy_master_agent_individual._huber_loss(q_values, targets, is_weights)

        # Expected abs_td_errors: [0.5, 1.0, 0.0]
        # Loss per sample without weights: [0.125, 0.5, 0.0]
        # Loss with weights: [0.125 * 0.1, 0.5 * 0.5, 0.0 * 1.0] = [0.0125, 0.25, 0.0]
        # Mean loss = (0.0125 + 0.25 + 0.0) / 3 = 0.2625 / 3 = 0.0875

        expected_weighted_loss = (0.5 * 0.5**2 * 0.1 + 0.5 * 1.0**2 * 0.5 + 0.5 * 0.0**2 * 1.0) / 3
        self.assertAlmostEqual(loss.item(), expected_weighted_loss, places=5)
        self.assertTrue(torch.equal(abs_td_errors, torch.tensor([0.5, 1.0, 0.0], device=self.device)))

    def test_sync_target_network(self):
        # Modify main network weights
        self.dummy_master_agent_individual.agent_network.dummy_layer.weight.data.fill_(0.5) # type: ignore

        # Before sync, target weights should be different (initial state)
        self.assertFalse(torch.equal(
            self.dummy_master_agent_individual.agent_network.dummy_layer.weight.data, # type: ignore
            self.dummy_master_agent_individual.agent_network_target.dummy_layer.weight.data # type: ignore
        ))

        self.dummy_master_agent_individual.sync_target_network()

        # After sync, target weights should be identical to main weights
        self.assertTrue(torch.equal(
            self.dummy_master_agent_individual.agent_network.dummy_layer.weight.data, # type: ignore
            self.dummy_master_agent_individual.agent_network_target.dummy_layer.weight.data # type: ignore
        ))
