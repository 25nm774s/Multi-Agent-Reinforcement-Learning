import torch
import torch.nn as nn
import torch.nn.functional as F
import unittest

from DQN.network import AgentNetwork

class TestAgentNetwork(unittest.TestCase):
    def setUp(self):
        self.grid_size = 5
        self.output_size = 5 # 5 actions
        self.total_agents = 3
        self.device = torch.device('cpu')

        self.agent_network = AgentNetwork(
            grid_size=self.grid_size,
            output_size=self.output_size,
            total_agents=self.total_agents
        ).to(self.device)

    def test_initialization(self):
        # Check if the network layers are correctly initialized
        self.assertIsInstance(self.agent_network.fc1, nn.Linear)
        self.assertIsInstance(self.agent_network.fc2, nn.Linear)
        self.assertIsInstance(self.agent_network.fc3, nn.Linear)

        # Check input/output dimensions
        expected_input_size = (self.agent_network.num_channels * self.grid_size**2) + self.total_agents
        self.assertEqual(self.agent_network.fc1.in_features, expected_input_size)
        self.assertEqual(self.agent_network.fc3.out_features, self.output_size)

    def test_forward_pass(self):
        batch_size = 2

        # Dummy state input (batch_size, num_channels, grid_size, grid_size)
        dummy_state_input = torch.randn(
            batch_size,
            self.agent_network.num_channels,
            self.grid_size,
            self.grid_size
        ).to(self.device)

        # Dummy agent IDs (batch_size,)
        dummy_agent_ids = torch.tensor([0, 2], dtype=torch.long, device=self.device)

        # Perform forward pass
        output = self.agent_network(dummy_state_input, dummy_agent_ids)

        # Check output shape
        expected_output_shape = (batch_size, self.output_size)
        self.assertEqual(output.shape, expected_output_shape)

        # Check if output values are floats
        self.assertEqual(output.dtype, torch.float32)

    def test_one_hot_encoding(self):
        batch_size = 1
        dummy_state_input = torch.randn(
            batch_size,
            self.agent_network.num_channels,
            self.grid_size,
            self.grid_size
        ).to(self.device)
        agent_id = torch.tensor([1], dtype=torch.long, device=self.device)

        # Directly check the one-hot encoding part by isolating the input to fc1
        x_flattened = dummy_state_input.flatten(start_dim=1)
        agent_id_one_hot = F.one_hot(agent_id, num_classes=self.total_agents).float()

        # Verify one-hot vector
        expected_one_hot = torch.tensor([0., 1., 0.], device=self.device)
        self.assertTrue(torch.equal(agent_id_one_hot.squeeze(0), expected_one_hot))

        # Verify combined input size
        combined_input = torch.cat((x_flattened, agent_id_one_hot), dim=1)
        expected_combined_input_size = (self.agent_network.num_channels * self.grid_size**2) + self.total_agents
        self.assertEqual(combined_input.size(1), expected_combined_input_size)

import torch
import torch.nn as nn
import torch.nn.functional as F
import unittest

from DQN.network import MixingNetwork

class TestMixingNetwork(unittest.TestCase):
    def setUp(self):
        self.n_agents = 2
        self.goals_number = 4 # For state_dim calculation
        self.state_dim = (self.goals_number + self.n_agents) * 2 # (4 + 2) * 2 = 12
        self.hidden_dim = 32
        self.batch_size = 4
        self.device = torch.device('cpu')

        self.mixing_network = MixingNetwork(
            n_agents=self.n_agents,
            state_dim=self.state_dim,
            hidden_dim=self.hidden_dim
        ).to(self.device)

    def test_forward_shape(self):
        agent_q_values = torch.randn(self.batch_size, self.n_agents, 1, device=self.device)
        global_state = torch.randn(self.batch_size, self.state_dim, device=self.device)

        Q_tot = self.mixing_network(agent_q_values, global_state)

        self.assertEqual(Q_tot.shape, (self.batch_size, 1))

    def test_monotonicity(self):
        # Generate random Q-values and global state
        agent_q_values = torch.randn(self.batch_size, self.n_agents, 1, device=self.device)
        global_state = torch.randn(self.batch_size, self.state_dim, device=self.device)

        # Calculate base Q_tot
        Q_tot_base = self.mixing_network(agent_q_values, global_state)

        # Perturb one agent's Q-value by a positive amount
        perturbed_agent_q_values = agent_q_values.clone()
        # Increase Q-value of agent 0 in all batches by a small positive amount
        # Ensure the increase is noticeable but not too large to avoid saturation effects too early
        increase_amount = 0.1
        perturbed_agent_q_values[:, 0, 0] += increase_amount

        # Calculate Q_tot with perturbed Q-values
        Q_tot_perturbed = self.mixing_network(perturbed_agent_q_values, global_state)

        # Assert that Q_tot_perturbed is greater than or equal to Q_tot_base
        # Use assertGreaterEqual for element-wise comparison if the output is not a scalar
        # Since Q_tot is (batch_size, 1), we compare element-wise
        self.assertTrue(torch.all(Q_tot_perturbed >= Q_tot_base))
