import torch
import unittest
from unittest.mock import MagicMock, patch
from typing import Dict

from src.DQN.IQLMasterAgent import IQLMasterAgent
from DQN.network import AgentNetwork
from Environments.StateProcesser import ObsToTensorWrapper

class TestIQLMasterAgent(unittest.TestCase):
    def setUp(self):
        self.n_agents = 2
        self.action_size = 5 # Define action_size
        self.grid_size = 6
        self.goals_num = 2
        self.device = torch.device('cpu')
        self.gamma = 0.99
        self.agent_ids = [f'agent_{i}' for i in range(self.n_agents)]
        self.goal_ids = [f'goal_{i}' for i in range(self.goals_num)]

        # Mock ObsToTensorWrapper (still needed for constructor, but its transform methods are not directly called by MasterAgent anymore)
        self.mock_state_processor = MagicMock(spec=ObsToTensorWrapper)
        self.mock_state_processor.num_channels = 3 # MasterAgentのnum_channelsプロパティがIAgentNetworkから取得されるため、ObsToTensorWrapperのnum_channelsは直接関係しないが、インスタンスとしては必要。
        self.mock_state_processor._flatten_global_state_dict.return_value = torch.randn((self.goals_num + self.n_agents) * 2, device=self.device)

        # AgentNetwork setup (using actual class, not mock)
        self.agent_network = AgentNetwork(
            grid_size=self.grid_size,
            output_size=self.action_size, # Use the newly defined action_size
            total_agents=self.n_agents
        ).to(self.device)

        # Create IQLMasterAgent instance
        self.iql_master_agent = IQLMasterAgent(
            n_agents=self.n_agents,
            action_size=self.action_size, # Use the newly defined action_size
            grid_size=self.grid_size,
            goals_number=self.goals_num,
            device=self.device,
            state_processor=self.mock_state_processor,
            agent_network=self.agent_network,
            gamma=self.gamma,
            agent_ids=self.agent_ids,
            goal_ids=self.goal_ids,
            agent_reward_processing_mode='individual' # Added new argument
        )

    def test_get_actions_greedy(
        self,
    ):
        # Test greedy action selection (epsilon = 0)
        # get_actions now expects agent_obs_tensor
        agent_obs_tensor = torch.randn(self.n_agents, self.agent_network.num_channels, self.grid_size, self.grid_size, device=self.device)
        epsilon = 0.0

        # Mock agent_network's forward pass to return predictable Q-values
        # (n_agents, action_size)
        with patch.object(self.iql_master_agent.agent_network, 'forward', return_value=torch.tensor(
            [[1.0, 2.0, 3.0, 0.5, 1.5], [5.0, 4.0, 3.0, 2.0, 1.0]], dtype=torch.float32, device=self.device
        )) as mock_forward:
            actions: Dict[str, int] = self.iql_master_agent.get_actions(agent_obs_tensor, epsilon)

            self.assertEqual(len(actions), self.n_agents)
            self.assertIsInstance(actions, dict)

            # Verify actions are chosen greedily (argmax)
            self.assertEqual(actions[self.agent_ids[0]], 2)
            self.assertEqual(actions[self.agent_ids[1]], 0)

            # Ensure forward was called correctly
            self.assertTrue(mock_forward.called)
            # Ensure eval/train modes are set correctly (it should be reset to train mode)
            self.assertTrue(self.iql_master_agent.agent_network.training)

    def test_get_actions_exploratory(
        self,
    ):
        # Test exploratory action selection (epsilon = 1)
        # get_actions now expects agent_obs_tensor
        agent_obs_tensor = torch.randn(self.n_agents, self.agent_network.num_channels, self.grid_size, self.grid_size, device=self.device)
        epsilon = 1.0

        with patch.object(self.iql_master_agent.agent_network, 'forward') as mock_forward:
            actions: Dict[str, int] = self.iql_master_agent.get_actions(agent_obs_tensor, epsilon)

            self.assertEqual(len(actions), self.n_agents)
            self.assertIsInstance(actions, dict)
            # Verify actions are random
            for aid in self.agent_ids:
                self.assertTrue(0 <= actions[aid] < self.action_size)

            # When epsilon is 1, forward pass still happens to determine action space, but actual choice is random.
            self.assertTrue(mock_forward.called)
            self.assertTrue(self.iql_master_agent.agent_network.training)

    def test_evaluate_q(
        self,
    ):
        batch_size = 4
        # Dummy tensor batches for new evaluate_q signature
        agent_obs_tensor_batch = torch.randn(batch_size, self.n_agents, self.agent_network.num_channels, self.grid_size, self.grid_size, device=self.device)
        global_state_tensor_batch = torch.randn(batch_size, (self.goals_num + self.n_agents) * 2, device=self.device)
        actions_tensor_batch = torch.tensor([[0, 1], [2, 3], [4, 0], [1, 2]], dtype=torch.long, device=self.device)
        rewards_tensor_batch = torch.randn(batch_size, self.n_agents, device=self.device)
        dones_tensor_batch = torch.tensor([[0, 0], [0, 1], [1, 0], [1, 1]], dtype=torch.float32, device=self.device)
        all_agents_done_batch = torch.tensor([[0], [0], [0], [1]], dtype=torch.float32, device=self.device) # NEW
        next_agent_obs_tensor_batch = torch.randn(batch_size, self.n_agents, self.agent_network.num_channels, self.grid_size, self.grid_size, device=self.device)
        next_global_state_tensor_batch = torch.randn(batch_size, (self.goals_num + self.n_agents) * 2, device=self.device)

        is_weights_batch = torch.ones(batch_size, device=self.device) # No IS weights for simplicity

        # Mock _get_agent_q_values (from BaseMasterAgent) to return predictable Q-values
        # It will be called twice: once for main_network, once for target_network
        # Expected return shape: (batch_size, n_agents, action_size)
        mock_main_q_values = torch.randn(batch_size, self.n_agents, self.action_size, device=self.device)
        mock_target_q_values = torch.randn(batch_size, self.n_agents, self.action_size, device=self.device)

        with patch.object(self.iql_master_agent, '_get_agent_q_values') as mock_get_q_values:
            mock_get_q_values.side_effect = [mock_main_q_values, mock_target_q_values]

            loss, abs_td_errors = self.iql_master_agent.evaluate_q(
                agent_obs_tensor_batch, global_state_tensor_batch, actions_tensor_batch, rewards_tensor_batch,
                dones_tensor_batch, all_agents_done_batch, next_agent_obs_tensor_batch, next_global_state_tensor_batch, is_weights_batch
            )

            # Verify output types and shapes
            self.assertIsInstance(loss, torch.Tensor)
            self.assertIsInstance(abs_td_errors, torch.Tensor)
            self.assertEqual(loss.shape, torch.Size(())) # Scalar loss
            self.assertEqual(abs_td_errors.shape, (batch_size,)) # type: ignore # Mean TD error per sample

            # Verify mocks were called correctly
            self.assertEqual(mock_get_q_values.call_count, 2)

            # Verify network modes
            self.assertTrue(self.iql_master_agent.agent_network.training)
            self.assertFalse(self.iql_master_agent.agent_network_target.training) # target network should be in eval mode

            # Check basic loss calculation (this is hard to assert precisely without replicating Bellman eq)
            # Just ensure it's not NaN or inf
            self.assertFalse(torch.isnan(loss))
            self.assertFalse(torch.isinf(loss))

            # Test with PER weights (should apply to loss)
            is_weights_batch_per = torch.rand(batch_size, device=self.device)
            mock_get_q_values.side_effect = [mock_main_q_values, mock_target_q_values] # Reset mock for second call

            loss_per, abs_td_errors_per = self.iql_master_agent.evaluate_q(
                agent_obs_tensor_batch, global_state_tensor_batch, actions_tensor_batch, rewards_tensor_batch,
                dones_tensor_batch, all_agents_done_batch, next_agent_obs_tensor_batch, next_global_state_tensor_batch, is_weights_batch_per
            )
            self.assertFalse(torch.equal(loss, loss_per)) # Loss should be different if IS weights are applied

    def test_sync_target_network(self):
        # Modify main network weights
        self.iql_master_agent.agent_network.fc1.weight.data.fill_(0.5) # type: ignore # Corrected reference

        # Before sync, target weights should be different (initial state)
        self.assertFalse(torch.equal(
            self.iql_master_agent.agent_network.fc1.weight.data, # type: ignore
            self.iql_master_agent.agent_network_target.fc1.weight.data # type: ignore
        ))

        self.iql_master_agent.sync_target_network()

        # After sync, target weights should be identical to main weights
        self.assertTrue(torch.equal(
            self.iql_master_agent.agent_network.fc1.weight.data, # type: ignore
            self.iql_master_agent.agent_network_target.fc1.weight.data # type: ignore
        ))
