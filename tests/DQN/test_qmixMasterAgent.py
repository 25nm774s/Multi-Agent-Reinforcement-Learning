import torch
import unittest
from typing import Dict, Any

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
        self.agent_ids = [f'agent_{i}' for i in range(self.n_agents)]
        self.goal_ids = [f'goal_{i}' for i in range(self.goals_num)]

        # Mock StateProcessor
        self.mock_state_processor = MagicMock(spec=StateProcessor)
        self.mock_state_processor.num_channels = 3
        # Mock transform_state_batch to accept single_agent_obs_dict and return dummy grid states
        # Expected return shape: (C, G, G)
        self.mock_state_processor.transform_state_batch.side_effect = \
            lambda single_agent_obs_dict: \
                torch.randn(self.mock_state_processor.num_channels, self.grid_size, self.grid_size, device=self.device)

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
            goals_number=self.goals_num,
            device=self.device,
            state_processor=self.mock_state_processor,
            agent_network=self.agent_network,
            gamma=self.gamma,
            agent_ids=self.agent_ids,
            goal_ids=self.goal_ids
        )

    def _create_dummy_obs_dict(self) -> Dict[str, Dict[str, Any]]:
        """Helper to create a dummy observation dictionary."""
        obs_dict: Dict[str, Dict[str, Any]] = {}
        goals_list = [(g, g) for g in range(self.goals_num)]
        for i, agent_id in enumerate(self.agent_ids):
            others_info = {other_id: (i, i+1) for j, other_id in enumerate(self.agent_ids) if other_id != agent_id}
            obs_dict[agent_id] = {
                'self': (i, i),
                'all_goals': goals_list,
                'others': others_info
            }
        return obs_dict

    def test_get_actions_greedy(self):
        obs_dict_for_current_step = self._create_dummy_obs_dict()
        epsilon = 0.0

        with patch.object(self.qmix_master_agent.agent_network, 'forward', return_value=torch.tensor(
            [[1.0, 2.0, 3.0, 0.5, 1.5], [5.0, 4.0, 3.0, 2.0, 1.0]], dtype=torch.float32, device=self.device
        )) as mock_forward:
            actions = self.qmix_master_agent.get_actions(obs_dict_for_current_step, epsilon)

            self.assertEqual(actions[0], 2)
            self.assertEqual(actions[1], 0)
            self.assertTrue(mock_forward.called)
            self.assertTrue(self.qmix_master_agent.agent_network.training) # Should be set back to train mode

    def test_get_actions_exploratory(self):
        obs_dict_for_current_step = self._create_dummy_obs_dict()
        epsilon = 1.0

        with patch.object(self.qmix_master_agent.agent_network, 'forward') as mock_forward:
            actions = self.qmix_master_agent.get_actions(obs_dict_for_current_step, epsilon)

            for action in actions:
                self.assertTrue(0 <= action < self.action_size)
            self.assertTrue(mock_forward.called)
            self.assertTrue(self.qmix_master_agent.agent_network.training)

    def test_evaluate_q(self):
        batch_size = 4
        # Dummy observation dictionaries batch
        obs_dicts_batch = [self._create_dummy_obs_dict() for _ in range(batch_size)]
        next_obs_dicts_batch = [self._create_dummy_obs_dict() for _ in range(batch_size)]

        actions_batch = torch.tensor([[0, 1], [2, 3], [4, 0], [1, 2]], dtype=torch.long, device=self.device)
        # rewards_batch should now be (batch_size, n_agents)
        rewards_batch = torch.randn(batch_size, self.n_agents, device=self.device) # Individual rewards

        dones_batch = torch.tensor([[0, 0], [0, 1], [1, 0], [1, 1]], dtype=torch.float32, device=self.device) # Individual done flags

        is_weights_batch = torch.ones(batch_size, device=self.device) # No IS weights for simplicity

        # Mock _process_raw_observations_batch to return predictable tensors for AgentNetwork and MixingNetwork
        # It returns (B*N, C, G, G) and (B, state_dim)
        mock_transformed_obs_current = torch.randn(batch_size * self.n_agents, self.mock_state_processor.num_channels, self.grid_size, self.grid_size, device=self.device)
        mock_global_state_for_mixing_current = torch.randn(batch_size, (self.goals_num + self.n_agents) * 2, device=self.device)
        mock_transformed_obs_next = torch.randn(batch_size * self.n_agents, self.mock_state_processor.num_channels, self.grid_size, self.grid_size, device=self.device)
        mock_global_state_for_mixing_next = torch.randn(batch_size, (self.goals_num + self.n_agents) * 2, device=self.device)

        with patch.object(self.qmix_master_agent, '_process_raw_observations_batch') as mock_process_obs_batch:
            mock_process_obs_batch.side_effect = [
                (mock_transformed_obs_current, mock_global_state_for_mixing_current),
                (mock_transformed_obs_next, mock_global_state_for_mixing_next)
            ]

            # Mock _get_agent_q_values (from BaseMasterAgent) to return predictable Q-values
            mock_main_agent_q_values = torch.randn(batch_size, self.n_agents, self.action_size, device=self.device)
            mock_target_agent_q_values = torch.randn(batch_size, self.n_agents, self.action_size, device=self.device)

            with patch.object(self.qmix_master_agent, '_get_agent_q_values') as mock_get_q_values:
                mock_get_q_values.side_effect = [mock_main_agent_q_values, mock_target_agent_q_values]

                # Mock MixingNetwork to return predictable Q_tot
                mock_main_q_tot = torch.randn(batch_size, 1, device=self.device)
                mock_target_q_tot = torch.randn(batch_size, 1, device=self.device)

                with patch.object(self.qmix_master_agent.mixing_network, 'forward') as mock_mixing_forward:
                    with patch.object(self.qmix_master_agent.mixing_network_target, 'forward') as mock_mixing_target_forward:
                        mock_mixing_forward.return_value = mock_main_q_tot
                        mock_mixing_target_forward.return_value = mock_target_q_tot

                        loss, abs_td_errors = self.qmix_master_agent.evaluate_q(
                            obs_dicts_batch, actions_batch, rewards_batch,
                            next_obs_dicts_batch, dones_batch, is_weights_batch
                        )

                        # Verify output types and shapes
                        self.assertIsInstance(loss, torch.Tensor)
                        self.assertIsInstance(abs_td_errors, torch.Tensor)
                        self.assertEqual(loss.shape, torch.Size(())) # Scalar loss
                        self.assertEqual(abs_td_errors.shape, (batch_size,)) #type:ignore Mean TD error per sample

                        # Verify mock calls
                        self.assertEqual(mock_process_obs_batch.call_count, 2)
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
                        mock_process_obs_batch.side_effect = [
                            (mock_transformed_obs_current, mock_global_state_for_mixing_current),
                            (mock_transformed_obs_next, mock_global_state_for_mixing_next)
                        ]
                        mock_get_q_values.side_effect = [mock_main_agent_q_values, mock_target_agent_q_values]
                        mock_mixing_forward.return_value = mock_main_q_tot
                        mock_mixing_target_forward.return_value = mock_target_q_tot

                        loss_per, abs_td_errors_per = self.qmix_master_agent.evaluate_q(
                            obs_dicts_batch, actions_batch, rewards_batch,
                            next_obs_dicts_batch, dones_batch, is_weights_batch_per
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
