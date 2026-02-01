import torch
import torch.nn as nn
from typing import Dict
import unittest
from unittest.mock import MagicMock, patch

from src.DQN.VDNMasterAgent import VDNMasterAgent
from src.DQN.network import IAgentNetwork, AgentNetwork
from src.Environments.StateProcesser import ObsToTensorWrapper

# Dummy concrete subclass for testing BaseMasterAgent
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

# Test class for VDNMasterAgent
class TestVDNMasterAgent(unittest.TestCase):
    def setUp(self):
        self.n_agents = 2
        self.action_size = 5
        self.grid_size = 6
        self.goals_num = 2
        self.device = torch.device('cpu')
        self.gamma = 0.99
        self.agent_ids = [f'agent_{i}' for i in range(self.n_agents)]
        self.goal_ids = [f'goal_{i}' for i in range(self.goals_num)]

        self.mock_state_processor = MagicMock(spec=ObsToTensorWrapper)
        self.mock_state_processor.num_channels = 3

        self.agent_network = AgentNetwork(
            grid_size=self.grid_size,
            output_size=self.action_size,
            total_agents=self.n_agents
        ).to(self.device)

        self.vdn_master_agent = VDNMasterAgent(
            n_agents=self.n_agents,
            action_size=self.action_size,
            grid_size=self.grid_size,
            goals_number=self.goals_num,
            device=self.device,
            state_processor=self.mock_state_processor,
            agent_network_instance=self.agent_network,
            gamma=self.gamma,
            agent_ids=self.agent_ids,
            goal_ids=self.goal_ids,
            agent_reward_processing_mode='individual'
        )

    def test_get_optimizer_params(self):
        params_from_method = set(self.vdn_master_agent.get_optimizer_params())
        expected_params = set(self.agent_network.parameters())
        expected_params.update(set(self.vdn_master_agent.mixer_network.parameters())) # VDN mixer has no parameters, so it should be empty and not affect the check if it's there.

        # Convert to sets of parameter objects to compare them effectively
        self.assertEqual(params_from_method, expected_params)


    def test_get_actions_greedy(self):
        agent_obs_tensor = torch.randn(self.n_agents, self.agent_network.num_channels, self.grid_size, self.grid_size, device=self.device)
        epsilon = 0.0

        with patch.object(self.vdn_master_agent.agent_network, 'forward', return_value=torch.tensor(
            [[1.0, 2.0, 3.0, 0.5, 1.5], [5.0, 4.0, 3.0, 2.0, 1.0]], dtype=torch.float32, device=self.device
        )) as mock_forward:
            actions: Dict[str, int] = self.vdn_master_agent.get_actions(agent_obs_tensor, epsilon)

            self.assertEqual(len(actions), self.n_agents)
            self.assertEqual(actions[self.agent_ids[0]], 2) # argmax of [1.0, 2.0, 3.0, 0.5, 1.5]
            self.assertEqual(actions[self.agent_ids[1]], 0) # argmax of [5.0, 4.0, 3.0, 2.0, 1.0]
            self.assertTrue(mock_forward.called)
            self.assertTrue(self.vdn_master_agent.agent_network.training) # Should be set back to train mode

    def test_get_actions_exploratory(self):
        agent_obs_tensor = torch.randn(self.n_agents, self.agent_network.num_channels, self.grid_size, self.grid_size, device=self.device)
        epsilon = 1.0

        with patch.object(self.vdn_master_agent.agent_network, 'forward') as mock_forward:
            actions: Dict[str, int] = self.vdn_master_agent.get_actions(agent_obs_tensor, epsilon)

            self.assertEqual(len(actions), self.n_agents)
            for aid in self.agent_ids:
                self.assertTrue(0 <= actions[aid] < self.action_size)

            self.assertTrue(mock_forward.called) # Forward is still called to get action space size
            self.assertTrue(self.vdn_master_agent.agent_network.training)

    def test_evaluate_q(self):
        batch_size = 4
        agent_obs_tensor_batch = torch.randn(batch_size, self.n_agents, self.agent_network.num_channels, self.grid_size, self.grid_size, device=self.device)
        global_state_tensor_batch = torch.randn(batch_size, (self.goals_num + self.n_agents) * 2, device=self.device) # Not used by VDN, but passed
        actions_tensor_batch = torch.tensor([[0, 1], [2, 3], [4, 0], [1, 2]], dtype=torch.long, device=self.device)
        rewards_tensor_batch = torch.randn(batch_size, self.n_agents, device=self.device)
        dones_tensor_batch = torch.tensor([[0, 0], [0, 1], [1, 0], [1, 1]], dtype=torch.float32, device=self.device)
        all_agents_done_batch = torch.tensor([[0], [0], [0], [1]], dtype=torch.float32, device=self.device)
        next_agent_obs_tensor_batch = torch.randn(batch_size, self.n_agents, self.agent_network.num_channels, self.grid_size, self.grid_size, device=self.device)
        next_global_state_tensor_batch = torch.randn(batch_size, (self.goals_num + self.n_agents) * 2, device=self.device) # Not used by VDN, but passed

        is_weights_batch = torch.ones(batch_size, device=self.device)

        mock_main_agent_q_values = torch.randn(batch_size, self.n_agents, self.action_size, device=self.device)
        mock_target_agent_q_values = torch.randn(batch_size, self.n_agents, self.action_size, device=self.device)

        with patch.object(self.vdn_master_agent, '_get_agent_q_values') as mock_get_q_values:
            mock_get_q_values.side_effect = [mock_main_agent_q_values, mock_target_agent_q_values]

            loss, abs_td_errors = self.vdn_master_agent.evaluate_q(
                agent_obs_tensor_batch, global_state_tensor_batch, actions_tensor_batch, rewards_tensor_batch,
                dones_tensor_batch, all_agents_done_batch, next_agent_obs_tensor_batch, next_global_state_tensor_batch, is_weights_batch
            )

            self.assertIsInstance(loss, torch.Tensor)
            self.assertIsInstance(abs_td_errors, torch.Tensor)
            self.assertEqual(loss.shape, torch.Size(())) # Scalar loss
            self.assertEqual(abs_td_errors.shape, (batch_size,)) # type: ignore # Mean TD error per sample

            self.assertEqual(mock_get_q_values.call_count, 2)
            self.assertTrue(self.vdn_master_agent.agent_network.training)
            self.assertFalse(self.vdn_master_agent.agent_network_target.training)

            # Test with PER weights (should apply to loss)
            is_weights_batch_per = torch.rand(batch_size, device=self.device)
            mock_get_q_values.side_effect = [mock_main_agent_q_values, mock_target_agent_q_values] # Reset mock

            loss_per, abs_td_errors_per = self.vdn_master_agent.evaluate_q(
                agent_obs_tensor_batch, global_state_tensor_batch, actions_tensor_batch, rewards_tensor_batch,
                dones_tensor_batch, all_agents_done_batch, next_agent_obs_tensor_batch, next_global_state_tensor_batch, is_weights_batch_per
            )
            self.assertFalse(torch.equal(loss, loss_per)) # Loss should be different if IS weights are applied

    def test_sync_target_network(self):
        # Modify main network weights
        self.vdn_master_agent.agent_network.fc1.weight.data.fill_(0.5) # type: ignore

        # Before sync, target weights should be different (initial state)
        self.assertFalse(torch.equal(
            self.vdn_master_agent.agent_network.fc1.weight.data, # type: ignore
            self.vdn_master_agent.agent_network_target.fc1.weight.data # type: ignore
        ))

        self.vdn_master_agent.sync_target_network()

        # After sync, target weights should be identical to main weights
        self.assertTrue(torch.equal(
            self.vdn_master_agent.agent_network.fc1.weight.data, # type: ignore
            self.vdn_master_agent.agent_network_target.fc1.weight.data # type: ignore
        ))
