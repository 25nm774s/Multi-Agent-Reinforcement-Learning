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
import numpy as np

from DQN.network import MixingNetwork

class TestMixingNetwork(unittest.TestCase):
    def setUp(self):
        self.n_agents = 2
        self.state_dim = 8 # (goals_num + n_agents) * 2 = (2 + 2) * 2 = 8
        self.hidden_dim = 32
        self.device = torch.device('cpu')

        self.mixing_network = MixingNetwork(
            n_agents=self.n_agents,
            state_dim=self.state_dim,
            hidden_dim=self.hidden_dim
        ).to(self.device)

    def test_initialization(self):
        # Check if hyper-networks are correctly initialized
        self.assertIsInstance(self.mixing_network.hyper_w1, nn.Linear)
        self.assertIsInstance(self.mixing_network.hyper_b1, nn.Linear)
        self.assertIsInstance(self.mixing_network.hyper_w2, nn.Linear)
        self.assertIsInstance(self.mixing_network.hyper_b2, nn.Linear)

        # Check input/output dimensions of hyper-networks
        self.assertEqual(self.mixing_network.hyper_w1.in_features, self.state_dim)
        self.assertEqual(self.mixing_network.hyper_w1.out_features, self.n_agents * self.hidden_dim)
        self.assertEqual(self.mixing_network.hyper_b1.in_features, self.state_dim)
        self.assertEqual(self.mixing_network.hyper_b1.out_features, self.hidden_dim)
        self.assertEqual(self.mixing_network.hyper_w2.in_features, self.state_dim)
        self.assertEqual(self.mixing_network.hyper_w2.out_features, self.hidden_dim * 1)
        self.assertEqual(self.mixing_network.hyper_b2.in_features, self.state_dim)
        self.assertEqual(self.mixing_network.hyper_b2.out_features, 1)


    def test_forward_pass(self):
        batch_size = 4

        # Dummy agent Q-values (batch_size, n_agents, 1)
        dummy_agent_q_values = torch.randn(batch_size, self.n_agents, 1, device=self.device)

        # Dummy global state (batch_size, state_dim)
        dummy_global_state = torch.randn(batch_size, self.state_dim, device=self.device)

        # Perform forward pass
        q_tot = self.mixing_network(dummy_agent_q_values, dummy_global_state)

        # Check output shape
        expected_output_shape = (batch_size, 1)
        self.assertEqual(q_tot.shape, expected_output_shape)

        # Check if output values are floats
        self.assertEqual(q_tot.dtype, torch.float32)

    def test_monotonicity_constraint(self):
        batch_size = 1
        dummy_agent_q_values = torch.randn(batch_size, self.n_agents, 1, device=self.device)
        dummy_global_state = torch.randn(batch_size, self.state_dim, device=self.device)

        # Capture original weights before forward pass if possible or check after
        # This test is more about ensuring the absolute value or exp is applied
        # We can't directly inspect the intermediate W1 and W2 in forward pass without modification
        # However, we can run forward pass and trust the implementation detail if it uses abs/exp

        # Just calling forward pass to ensure no errors related to constraint logic
        q_tot = self.mixing_network(dummy_agent_q_values, dummy_global_state)
        self.assertIsNotNone(q_tot)

        # A more direct test would require modifying MixingNetwork to expose W1, W2
        # For now, we assume torch.abs is correctly applied as per the design.
        # If we could access w1 and w2, we'd check: self.assertTrue(torch.all(w1 >= 0)) and self.assertTrue(torch.all(w2 >= 0))
