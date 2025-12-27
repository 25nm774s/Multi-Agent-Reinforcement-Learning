import torch
import unittest

from unittest.mock import MagicMock, patch

from src.DQN.QMIXMasterAgent import QMIXMasterAgent, AgentNetwork
from src.utils.StateProcesser import StateProcessor

class TestQMIXMasterAgent(unittest.TestCase):
    def setUp(self):
        self.n_agents = 2
        self.action_size = 5
        self.grid_size = 6
        self.goals_num = 2
        self.device = torch.device('cpu')
        self.gamma = 0.99

        # Mock StateProcessor
        self.mock_state_processor = MagicMock(spec=StateProcessor)
        self.mock_state_processor.num_channels = 3
        self.mock_state_processor.transform_state_batch.side_effect = \
            lambda agent_id, raw_state_batch: \
                torch.randn(raw_state_batch.shape[0], self.mock_state_processor.num_channels, self.grid_size, self.grid_size, device=self.device)

        # AgentNetwork setup (using actual class)
        self.agent_network = AgentNetwork(
            grid_size=self.grid_size,
            output_size=self.action_size,
            total_agents=self.n_agents
        ).to(self.device)

        # Create QMIXMasterAgent instance
        self.qmix_master_agent = QMIXMasterAgent(
            n_agents=self.n_agents,
            action_size=self.action_size,
            grid_size=self.grid_size,
            goals_num=self.goals_num,
            device=self.device,
            state_processor=self.mock_state_processor,
            agent_network=self.agent_network,
            gamma=self.gamma
        )

    def test_get_actions_greedy(self):
        global_state = tuple([(0,0) for _ in range(self.goals_num + self.n_agents)])
        epsilon = 0.0

        with patch.object(self.qmix_master_agent.agent_network, 'forward', return_value=torch.tensor(
            [[1.0, 2.0, 3.0, 0.5, 1.5], [5.0, 4.0, 3.0, 2.0, 1.0]], dtype=torch.float32, device=self.device
        )) as mock_forward:
            actions = self.qmix_master_agent.get_actions(global_state, epsilon)

            self.assertEqual(actions[0], 2)
            self.assertEqual(actions[1], 0)
            self.assertTrue(mock_forward.called)
            self.assertTrue(self.qmix_master_agent.agent_network.training) # Should be set back to train mode

    def test_get_actions_exploratory(self):
        global_state = tuple([(0,0) for _ in range(self.goals_num + self.n_agents)])
        epsilon = 1.0

        with patch.object(self.qmix_master_agent.agent_network, 'forward') as mock_forward:
            actions = self.qmix_master_agent.get_actions(global_state, epsilon)

            for action in actions:
                self.assertTrue(0 <= action < self.action_size)
            self.assertTrue(mock_forward.called)
            self.assertTrue(self.qmix_master_agent.agent_network.training)

    def test_evaluate_q(self):
        batch_size = 4
        flattened_state_dim = (self.goals_num + self.n_agents) * 2
        global_states_batch_raw = torch.randn(batch_size, flattened_state_dim, device=self.device)
        next_global_states_batch_raw = torch.randn(batch_size, flattened_state_dim, device=self.device)

        actions_batch = torch.tensor([[0, 1], [2, 3], [4, 0], [1, 2]], dtype=torch.long, device=self.device)
        rewards_batch = torch.randn(batch_size, device=self.device) # Team reward

        dones_batch = torch.tensor([[0, 0], [0, 1], [1, 0], [1, 1]], dtype=torch.float32, device=self.device) # Individual done flags

        is_weights_batch = torch.ones(batch_size, device=self.device) # No IS weights for simplicity

        # Mock _get_agent_q_values (from BaseMasterAgent) to return predictable Q-values
        mock_main_agent_q_values = torch.randn(batch_size, self.n_agents, self.action_size, device=self.device)
        mock_target_agent_q_values = torch.randn(batch_size, self.n_agents, self.action_size, device=self.device)

        # Mock MixingNetwork to return predictable Q_tot
        mock_main_q_tot = torch.randn(batch_size, 1, device=self.device)
        mock_target_q_tot = torch.randn(batch_size, 1, device=self.device)

        with patch.object(self.qmix_master_agent, '_get_agent_q_values') as mock_get_q_values:
            with patch.object(self.qmix_master_agent.mixing_network, 'forward') as mock_mixing_forward:
                with patch.object(self.qmix_master_agent.mixing_network_target, 'forward') as mock_mixing_target_forward:

                    mock_get_q_values.side_effect = [mock_main_agent_q_values, mock_target_agent_q_values] # Called for main_net and target_net
                    mock_mixing_forward.return_value = mock_main_q_tot
                    mock_mixing_target_forward.return_value = mock_target_q_tot

                    loss, abs_td_errors = self.qmix_master_agent.evaluate_q(
                        global_states_batch_raw, actions_batch, rewards_batch,
                        next_global_states_batch_raw, dones_batch, is_weights_batch
                    )

                    # Verify output types and shapes
                    self.assertIsInstance(loss, torch.Tensor)
                    self.assertIsInstance(abs_td_errors, torch.Tensor)
                    self.assertEqual(loss.shape, torch.Size(())) # Scalar loss
                    self.assertEqual(abs_td_errors.shape, (batch_size,)) #type:ignore 直前でNoneチェックあり。 Mean TD error per sample

                    # Verify mock calls
                    self.assertEqual(mock_get_q_values.call_count, 2)
                    self.assertEqual(mock_mixing_forward.call_count, 1)
                    self.assertEqual(mock_mixing_target_forward.call_count, 1)

                    # Verify network modes
                    self.assertTrue(self.qmix_master_agent.agent_network.training)
                    self.assertTrue(self.qmix_master_agent.mixing_network.training)
                    self.assertFalse(self.qmix_master_agent.agent_network_target.training)
                    self.assertFalse(self.qmix_master_agent.mixing_network_target.training)

                    # Check basic loss calculation
                    self.assertFalse(torch.isnan(loss))
                    self.assertFalse(torch.isinf(loss))

                    # Test with PER weights
                    is_weights_batch_per = torch.rand(batch_size, device=self.device)
                    mock_get_q_values.side_effect = [mock_main_agent_q_values, mock_target_agent_q_values]
                    mock_mixing_forward.return_value = mock_main_q_tot
                    mock_mixing_target_forward.return_value = mock_target_q_tot

                    loss_per, abs_td_errors_per = self.qmix_master_agent.evaluate_q(
                        global_states_batch_raw, actions_batch, rewards_batch,
                        next_global_states_batch_raw, dones_batch, is_weights_batch_per
                    )
                    self.assertFalse(torch.equal(loss, loss_per)) # Loss should be different if IS weights are applied

    def test_sync_target_network(self):
        # Modify main network weights
        self.qmix_master_agent.agent_network.fc1.weight.data.fill_(0.5)
        self.qmix_master_agent.mixing_network.hyper_w1.weight.data.fill_(0.5)

        # Before sync, target weights should be different (initial state)
        self.assertFalse(torch.equal(
            self.qmix_master_agent.agent_network.fc1.weight.data,
            self.qmix_master_agent.agent_network_target.fc1.weight.data
        ))
        self.assertFalse(torch.equal(
            self.qmix_master_agent.mixing_network.hyper_w1.weight.data,
            self.qmix_master_agent.mixing_network_target.hyper_w1.weight.data
        ))

        self.qmix_master_agent.sync_target_network()

        # After sync, target weights should be identical to main weights
        self.assertTrue(torch.equal(
            self.qmix_master_agent.agent_network.fc1.weight.data,
            self.qmix_master_agent.agent_network_target.fc1.weight.data
        ))
        self.assertTrue(torch.equal(
            self.qmix_master_agent.mixing_network.hyper_w1.weight.data,
            self.qmix_master_agent.mixing_network_target.hyper_w1.weight.data
        ))
