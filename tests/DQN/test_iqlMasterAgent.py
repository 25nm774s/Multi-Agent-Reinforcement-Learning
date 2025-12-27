import torch
import unittest
from unittest.mock import MagicMock, patch

from src.DQN.IQLMasterAgent import IQLMasterAgent
from src.DQN.dqn import AgentNetwork
from src.utils.StateProcesser import StateProcessor

class TestIQLMasterAgent(unittest.TestCase):
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
        # Mock transform_state_batch to return dummy grid states
        # (batch_size, num_channels, grid_size, grid_size)
        self.mock_state_processor.transform_state_batch.side_effect = \
            lambda agent_id, raw_state_batch: \
                torch.randn(raw_state_batch.shape[0], self.mock_state_processor.num_channels, self.grid_size, self.grid_size, device=self.device)

        # AgentNetwork setup (using actual class, not mock)
        self.agent_network = AgentNetwork(
            grid_size=self.grid_size,
            output_size=self.action_size,
            total_agents=self.n_agents
        ).to(self.device)

        # Create IQLMasterAgent instance
        self.iql_master_agent = IQLMasterAgent(
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
        # Test greedy action selection (epsilon = 0)
        global_state = tuple([(0,0) for _ in range(self.goals_num + self.n_agents)])
        epsilon = 0.0

        # Mock agent_network's forward pass to return predictable Q-values
        # (n_agents, action_size)
        with patch.object(self.iql_master_agent.agent_network, 'forward', return_value=torch.tensor(
            [[1.0, 2.0, 3.0, 0.5, 1.5], [5.0, 4.0, 3.0, 2.0, 1.0]], dtype=torch.float32, device=self.device
        )) as mock_forward:
            actions = self.iql_master_agent.get_actions(global_state, epsilon)

            # Verify actions are chosen greedily (argmax)
            self.assertEqual(actions[0], 2) # Argmax of [1.0, 2.0, 3.0, 0.5, 1.5] is 2
            self.assertEqual(actions[1], 0) # Argmax of [5.0, 4.0, 3.0, 2.0, 1.0] is 0
            # Ensure forward was called correctly
            self.assertTrue(mock_forward.called)
            # Ensure eval/train modes are set correctly
            self.assertTrue(self.iql_master_agent.agent_network.training)

    def test_get_actions_exploratory(self):
        # Test exploratory action selection (epsilon = 1)
        global_state = tuple([(0,0) for _ in range(self.goals_num + self.n_agents)])
        epsilon = 1.0

        with patch.object(self.iql_master_agent.agent_network, 'forward') as mock_forward:
            actions = self.iql_master_agent.get_actions(global_state, epsilon)

            # Verify actions are random
            for action in actions:
                self.assertTrue(0 <= action < self.action_size)
            # When epsilon is 1, forward pass still happens to determine action space, but actual choice is random.
            self.assertTrue(mock_forward.called)
            self.assertTrue(self.iql_master_agent.agent_network.training)

    def test_evaluate_q(self):
        batch_size = 4
        # Dummy raw global states (B, flattened_state_dim)
        flattened_state_dim = (self.goals_num + self.n_agents) * 2
        global_states_batch_raw = torch.randn(batch_size, flattened_state_dim, device=self.device)
        next_global_states_batch_raw = torch.randn(batch_size, flattened_state_dim, device=self.device)

        # Dummy actions (B, n_agents), e.g., agent0 took action 0, agent1 took action 1
        actions_batch = torch.tensor([[0, 1], [2, 3], [4, 0], [1, 2]], dtype=torch.long, device=self.device)
        rewards_batch = torch.randn(batch_size, device=self.device) # Team reward

        # Individual done flags (B, n_agents)
        dones_batch = torch.tensor([[0, 0], [0, 1], [1, 0], [1, 1]], dtype=torch.float32, device=self.device)

        is_weights_batch = torch.ones(batch_size, device=self.device) # No IS weights for simplicity

        # Mock agent_network forward to return predictable Q-values
        # _get_agent_q_values expects (batch_size, n_agents, action_size)
        # It will be called twice: once for main_network, once for target_network

        # Mock for main agent_network
        mock_main_q_values = torch.randn(batch_size, self.n_agents, self.action_size, device=self.device)
        # Mock for target agent_network_target
        mock_target_q_values = torch.randn(batch_size, self.n_agents, self.action_size, device=self.device)

        with patch.object(self.iql_master_agent, '_get_agent_q_values') as mock_get_q_values:
            mock_get_q_values.side_effect = [mock_main_q_values, mock_target_q_values]

            loss, abs_td_errors = self.iql_master_agent.evaluate_q(
                global_states_batch_raw, actions_batch, rewards_batch,
                next_global_states_batch_raw, dones_batch, is_weights_batch
            )

            # Verify output types and shapes
            self.assertIsInstance(loss, torch.Tensor)
            self.assertIsInstance(abs_td_errors, torch.Tensor)
            self.assertEqual(loss.shape, torch.Size(())) # Scalar loss
            self.assertEqual(abs_td_errors.shape, (batch_size,)) #type:ignore 直前でNoneチェック Mean TD error per sample

            # Verify _get_agent_q_values was called twice correctly
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
            mock_get_q_values.side_effect = [mock_main_q_values, mock_target_q_values]
            loss_per, abs_td_errors_per = self.iql_master_agent.evaluate_q(
                global_states_batch_raw, actions_batch, rewards_batch,
                next_global_states_batch_raw, dones_batch, is_weights_batch_per
            )
            self.assertFalse(torch.equal(loss, loss_per)) # Loss should be different if IS weights are applied
