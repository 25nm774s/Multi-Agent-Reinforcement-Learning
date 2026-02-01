import torch
import torch.nn as nn
import unittest
from unittest.mock import MagicMock, patch

from src.DQN.DICGMasterAgent import DICGMasterAgent
from src.DQN.network import DICGMixer

class DummyAgentNetwork(nn.Module):
    def __init__(self, grid_size: int, output_size: int, total_agents: int = 1):
        super().__init__()
        self._grid_size = grid_size
        self._num_channels = 3
        self._total_agents = total_agents
        self._action_size = output_size
        # Dummy layer to simulate network output
        self.dummy_linear = nn.Linear(1, 1)

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
        # Mock forward pass for predictable Q-values
        batch_n = x.size(0)
        return torch.randn(batch_n, self.action_size, device=x.device)


class TestDICGMasterAgent(unittest.TestCase):
    def setUp(self):
        self.n_agents = 2
        self.action_size = 5
        self.grid_size = 6
        self.goals_num = 2
        self.device = torch.device('cpu')
        self.gamma = 0.99
        self.agent_ids = [f'agent_{i}' for i in range(self.n_agents)]
        self.goal_ids = [f'goal_{i}' for i in range(self.goals_num)]

        # Mock ObsToTensorWrapper
        self.mock_state_processor = MagicMock()
        self.mock_state_processor.num_channels = 3

        # Dummy AgentNetwork instance
        self.agent_network = DummyAgentNetwork(
            grid_size=self.grid_size,
            output_size=self.action_size,
            total_agents=self.n_agents
        ).to(self.device)

        # DICGMasterAgent instance
        self.dicg_master_agent = DICGMasterAgent(
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

    def test_init(self):
        self.assertIsInstance(self.dicg_master_agent.mixer_network, DICGMixer)
        self.assertIsInstance(self.dicg_master_agent.mixer_network_target, DICGMixer)
        self.assertEqual(self.dicg_master_agent.gamma, self.gamma)

    def test_get_optimizer_params(self):
        params = self.dicg_master_agent.get_optimizer_params()
        # Should contain parameters from agent_network and mixer_network
        agent_net_param_count = sum(p.numel() for p in self.agent_network.parameters())
        mixer_net_param_count = sum(p.numel() for p in self.dicg_master_agent.mixer_network.parameters())
        total_param_count = sum(p.numel() for p in params)
        self.assertEqual(total_param_count, agent_net_param_count + mixer_net_param_count)

    def test_evaluate_q(self):
        batch_size = 4
        # Dummy tensor batches
        agent_obs_tensor_batch = torch.randn(batch_size, self.n_agents, self.agent_network.num_channels, self.grid_size, self.grid_size, device=self.device)
        global_state_tensor_batch = torch.randn(batch_size, (self.goals_num + self.n_agents) * 2, device=self.device)
        actions_tensor_batch = torch.randint(0, self.action_size, (batch_size, self.n_agents), dtype=torch.long, device=self.device)
        rewards_tensor_batch = torch.randn(batch_size, self.n_agents, device=self.device)
        dones_tensor_batch = torch.randint(0, 2, (batch_size, self.n_agents), dtype=torch.float32, device=self.device)
        all_agents_done_batch = torch.randint(0, 2, (batch_size, 1), dtype=torch.float32, device=self.device)
        next_agent_obs_tensor_batch = torch.randn(batch_size, self.n_agents, self.agent_network.num_channels, self.grid_size, self.grid_size, device=self.device)
        next_global_state_tensor_batch = torch.randn(batch_size, (self.goals_num + self.n_agents) * 2, device=self.device)
        is_weights_batch = torch.ones(batch_size, device=self.device)

        # Mock _get_agent_q_values
        mock_main_agent_q_values = torch.randn(batch_size, self.n_agents, self.action_size, device=self.device)
        mock_target_agent_q_values = torch.randn(batch_size, self.n_agents, self.action_size, device=self.device)

        with patch.object(self.dicg_master_agent, '_get_agent_q_values') as mock_get_q_values:
            mock_get_q_values.side_effect = [mock_main_agent_q_values, mock_target_agent_q_values]

            # Mock DICGMixer's forward method
            mock_main_q_tot = torch.randn(batch_size, 1, device=self.device)
            mock_target_q_tot = torch.randn(batch_size, 1, device=self.device)

            with patch.object(self.dicg_master_agent.mixer_network, 'forward') as mock_mixing_forward:
                with patch.object(self.dicg_master_agent.mixer_network_target, 'forward') as mock_mixing_target_forward:
                    mock_mixing_forward.return_value = mock_main_q_tot
                    mock_mixing_target_forward.return_value = mock_target_q_tot

                    loss, abs_td_errors = self.dicg_master_agent.evaluate_q(
                        agent_obs_tensor_batch, global_state_tensor_batch, actions_tensor_batch, rewards_tensor_batch,
                        dones_tensor_batch, all_agents_done_batch, next_agent_obs_tensor_batch, next_global_state_tensor_batch, is_weights_batch
                    )

                    self.assertIsInstance(loss, torch.Tensor)
                    self.assertIsInstance(abs_td_errors, torch.Tensor)
                    self.assertEqual(loss.shape, torch.Size(()))
                    self.assertEqual(abs_td_errors.shape, (batch_size,)) # type: ignore

                    self.assertEqual(mock_get_q_values.call_count, 2)
                    self.assertEqual(mock_mixing_forward.call_count, 1)
                    self.assertEqual(mock_mixing_target_forward.call_count, 1)

                    self.assertTrue(self.dicg_master_agent.agent_network.training)
                    self.assertTrue(self.dicg_master_agent.mixer_network.training)
                    self.assertFalse(self.dicg_master_agent.agent_network_target.training)
                    self.assertFalse(self.dicg_master_agent.mixer_network_target.training)

    def test_sync_target_network(self):
        # Modify main network weights
        self.dicg_master_agent.agent_network.dummy_linear.weight.data.fill_(0.5) # type: ignore
        self.dicg_master_agent.mixer_network.query_network.weight.data.fill_(0.6) # type: ignore

        # Before sync, target weights should be different
        self.assertFalse(torch.equal(
            self.dicg_master_agent.agent_network.dummy_linear.weight.data, # type: ignore
            self.dicg_master_agent.agent_network_target.dummy_linear.weight.data # type: ignore
        ))
        self.assertFalse(torch.equal(
            self.dicg_master_agent.mixer_network.query_network.weight.data, # type: ignore
            self.dicg_master_agent.mixer_network_target.query_network.weight.data # type: ignore
        ))

        self.dicg_master_agent.sync_target_network()

        # After sync, target weights should be identical
        self.assertTrue(torch.equal(
            self.dicg_master_agent.agent_network.dummy_linear.weight.data, # type: ignore
            self.dicg_master_agent.agent_network_target.dummy_linear.weight.data # type: ignore
        ))
        self.assertTrue(torch.equal(
            self.dicg_master_agent.mixer_network.query_network.weight.data, # type: ignore
            self.dicg_master_agent.mixer_network_target.query_network.weight.data # type: ignore
        ))
