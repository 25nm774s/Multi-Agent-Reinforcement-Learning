import unittest
import torch
import numpy as np

from src.utils.replay_buffer import ReplayBuffer

class TestReplayBuffer(unittest.TestCase):
    def setUp(self):
        self.buffer_size = 100
        self.batch_size = 32
        self.device = torch.device('cpu')
        self.alpha = 0.6
        self.n_agents = 2
        self.goals_number = 2
        self.grid_size = 6
        self.num_channels = 3 # From ObsToTensorWrapper

        # self.global_state_dim = (self.goals_num + self.n_agents) * 2 # Reinstated

    def _create_sample_experience(self, action_value: int = 0, reward_value: float = 1.0, done_value: bool = False):
        # Create dummy tensors for the experience
        agent_obs_tensor = torch.randn(self.n_agents, self.num_channels, self.grid_size, self.grid_size, device=self.device)
        global_state_tensor = torch.randn((self.goals_number + self.n_agents) * 2, device=self.device)
        actions_tensor = torch.full((self.n_agents,), fill_value=action_value, dtype=torch.long, device=self.device)
        rewards_tensor = torch.full((self.n_agents,), fill_value=reward_value, dtype=torch.float32, device=self.device)
        dones_tensor = torch.full((self.n_agents,), fill_value=float(done_value), dtype=torch.float32, device=self.device)
        all_agents_done_scalar = torch.full((self.n_agents,), fill_value=float(done_value), dtype=torch.float32, device=self.device)
        next_agent_obs_tensor = torch.randn(self.n_agents, self.num_channels, self.grid_size, self.grid_size, device=self.device)
        next_global_state_tensor = torch.randn((self.goals_number + self.n_agents) * 2, device=self.device)

        return (
            agent_obs_tensor,
            global_state_tensor,
            actions_tensor,
            rewards_tensor,
            dones_tensor,
            all_agents_done_scalar,
            next_agent_obs_tensor,
            next_global_state_tensor
        )

    def test_add_and_len_uniform(self):
        rb = ReplayBuffer(
            buffer_size=self.buffer_size,
            batch_size=self.batch_size,
            device=self.device,
            n_agents=self.n_agents,
            num_channels=self.num_channels,
            grid_size=self.grid_size,
            goals_number=self.goals_number,
            use_per=False
        )
        self.assertEqual(len(rb), 0)
        for i in range(50):
            rb.add(*self._create_sample_experience()) # Call with tensor arguments
        self.assertEqual(len(rb), 50);
        for i in range(100):
            rb.add(*self._create_sample_experience()) # Call with tensor arguments
        self.assertEqual(len(rb), self.buffer_size) # Should cap at buffer_size
        self.assertEqual(rb.n_agents, self.n_agents)

    def test_sample_uniform(self):
        rb = ReplayBuffer(
            buffer_size=self.buffer_size,
            batch_size=self.batch_size,
            device=self.device,
            n_agents=self.n_agents,
            num_channels=self.num_channels,
            grid_size=self.grid_size,
            goals_number=self.goals_number,
            use_per=False
        )
        for i in range(self.buffer_size):
            rb.add(*self._create_sample_experience(action_value=i%5, reward_value=float(i)))

        sample_output = rb.sample(beta=0.0) # beta is not used in uniform
        self.assertIsNotNone(sample_output)

        (
            agent_obs_tensor_batch,
            global_state_tensor_batch,
            actions_tensor_batch,
            rewards_tensor_batch,
            dones_tensor_batch,
            all_agents_done_batch,
            next_agent_obs_tensor_batch,
            next_global_state_tensor_batch,
            is_weights_batch,
            sampled_indices
        ) = sample_output # type: ignore

        self.assertEqual(agent_obs_tensor_batch.shape, (self.batch_size, self.n_agents, self.num_channels, self.grid_size, self.grid_size))
        self.assertEqual(global_state_tensor_batch.shape, (self.batch_size, (self.goals_number + self.n_agents) * 2))
        self.assertEqual(actions_tensor_batch.shape, (self.batch_size, self.n_agents))
        self.assertEqual(rewards_tensor_batch.shape, (self.batch_size, self.n_agents))
        self.assertEqual(dones_tensor_batch.shape, (self.batch_size, self.n_agents))
        self.assertEqual(all_agents_done_batch.shape, (self.batch_size, self.n_agents))
        self.assertEqual(next_agent_obs_tensor_batch.shape, (self.batch_size, self.n_agents, self.num_channels, self.grid_size, self.grid_size))
        self.assertEqual(next_global_state_tensor_batch.shape, (self.batch_size, (self.goals_number + self.n_agents) * 2))

        self.assertIsNone(is_weights_batch)
        self.assertIsNone(sampled_indices)
        self.assertEqual(actions_tensor_batch.dtype, torch.long)
        self.assertEqual(dones_tensor_batch.dtype, torch.float32)

    def test_add_and_len_per(self):
        rb = ReplayBuffer(
            buffer_size=self.buffer_size,
            batch_size=self.batch_size,
            device=self.device,
            n_agents=self.n_agents,
            num_channels=self.num_channels,
            grid_size=self.grid_size,
            goals_number=self.goals_number,
            alpha=self.alpha,
            use_per=True
        )
        self.assertEqual(len(rb), 0)
        for i in range(50):
            rb.add(*self._create_sample_experience()) # Call with tensor arguments
        self.assertEqual(len(rb), 50)
        for i in range(100):
            rb.add(*self._create_sample_experience()) # Call with tensor arguments
        self.assertEqual(len(rb), self.buffer_size) # Should cap at buffer_size
        self.assertEqual(rb.n_agents, self.n_agents)

    def test_sample_per(self):
        rb = ReplayBuffer(
            buffer_size=self.buffer_size,
            batch_size=self.batch_size,
            device=self.device,
            n_agents=self.n_agents,
            num_channels=self.num_channels,
            grid_size=self.grid_size,
            goals_number=self.goals_number,
            alpha=self.alpha,
            use_per=True
        )
        for i in range(self.buffer_size):
            # Add experiences with varying rewards to influence priority indirectly
            rb.add(*self._create_sample_experience(reward_value=float(i % 10)))

        sample_output = rb.sample(beta=0.4) # beta is used in PER
        self.assertIsNotNone(sample_output)

        (
            agent_obs_tensor_batch,
            global_state_tensor_batch,
            actions_tensor_batch,
            rewards_tensor_batch,
            dones_tensor_batch,
            all_agents_done_batch,
            next_agent_obs_tensor_batch,
            next_global_state_tensor_batch,
            is_weights_batch,
            sampled_indices
        ) = sample_output # type: ignore

        self.assertEqual(agent_obs_tensor_batch.shape, (self.batch_size, self.n_agents, self.num_channels, self.grid_size, self.grid_size))
        self.assertEqual(global_state_tensor_batch.shape, (self.batch_size, (self.goals_number + self.n_agents) * 2))
        self.assertEqual(actions_tensor_batch.shape, (self.batch_size, self.n_agents))
        self.assertEqual(rewards_tensor_batch.shape, (self.batch_size, self.n_agents))
        self.assertEqual(dones_tensor_batch.shape, (self.batch_size, self.n_agents))
        self.assertEqual(all_agents_done_batch.shape, (self.batch_size, self.n_agents))
        self.assertEqual(next_agent_obs_tensor_batch.shape, (self.batch_size, self.n_agents, self.num_channels, self.grid_size, self.grid_size))
        self.assertEqual(next_global_state_tensor_batch.shape, (self.batch_size, (self.goals_number + self.n_agents) * 2))

        self.assertIsNotNone(is_weights_batch) # Should have IS weights
        self.assertEqual(is_weights_batch.shape, (self.batch_size,)) # type: ignore
        self.assertIsNotNone(sampled_indices) # Should have sampled indices
        self.assertEqual(len(sampled_indices), self.batch_size) # type: ignore

    def test_update_priorities(self):
        rb = ReplayBuffer(
            buffer_size=self.buffer_size,
            batch_size=self.batch_size,
            device=self.device,
            n_agents=self.n_agents,
            num_channels=self.num_channels,
            grid_size=self.grid_size,
            goals_number=self.goals_number,
            alpha=self.alpha,
            use_per=True
        )
        for i in range(self.buffer_size):
            rb.add(*self._create_sample_experience(reward_value=float(i)))

        # Sample some experiences
        sample_output = rb.sample(beta=0.4)
        self.assertIsNotNone(sample_output)
        (
            agent_obs_tensor_batch,
            global_state_tensor_batch,
            actions_tensor_batch,
            rewards_tensor_batch,
            dones_tensor_batch,
            all_agents_done_batch,
            next_agent_obs_tensor_batch,
            next_global_state_tensor_batch,
            is_weights_batch,
            sampled_indices
        ) = sample_output # type: ignore

        # Create dummy TD errors for updating priorities
        dummy_td_errors = np.abs(np.random.randn(self.batch_size)) + 0.1 # Ensure positive

        # The `sampled_indices` returned by `rb.sample` are the `tree_idx` values from SumTree.
        # These are what `update_priorities` expects.
        self.assertIsNotNone(sampled_indices)
        rb.update_priorities(sampled_indices, dummy_td_errors) # type: ignore

        # Verify that priorities in SumTree have changed
        # A simple check: total_priority should reflect the sum of updated priorities, roughly
        # This is hard to precisely assert without knowing the exact initial priorities, but it should be positive
        self.assertGreater(rb.tree.total_priority, 0)
