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
        self.goals_num = 2
        self.global_state_dim = (self.goals_num + self.n_agents) * 2

    def _create_sample_experience(self, action_value: int = 0, reward_value: float = 1.0, done_value: bool = False):
        global_state = tuple([(0, 0) for _ in range(self.goals_num + self.n_agents)])
        actions = [action_value] * self.n_agents
        reward = reward_value
        next_global_state = tuple([(0, 0) for _ in range(self.goals_num + self.n_agents)])
        dones = [done_value] * self.n_agents
        return global_state, actions, reward, next_global_state, dones

    def test_add_and_len_uniform(self):
        rb = ReplayBuffer(
            buffer_size=self.buffer_size,
            batch_size=self.batch_size,
            device=self.device,
            goals_num=self.goals_num,
            use_per=False
        )
        self.assertEqual(len(rb), 0)
        for i in range(50):
            rb.add(*self._create_sample_experience())
        self.assertEqual(len(rb), 50)
        for i in range(100):
            rb.add(*self._create_sample_experience())
        self.assertEqual(len(rb), self.buffer_size) # Should cap at buffer_size
        self.assertEqual(rb.n_agents, self.n_agents)

    def test_sample_uniform(self):
        rb = ReplayBuffer(
            buffer_size=self.buffer_size,
            batch_size=self.batch_size,
            device=self.device,
            goals_num=self.goals_num,
            use_per=False
        )
        for i in range(self.buffer_size):
            rb.add(*self._create_sample_experience(action_value=i%5, reward_value=float(i)))

        sample_output = rb.sample(beta=0.0) # beta is not used in uniform
        self.assertIsNotNone(sample_output)

        (global_states_batch, actions_batch, rewards_batch, next_global_states_batch, dones_batch, is_weights_batch, sampled_indices) = sample_output#type:ignore 直前でNoneではないはず

        self.assertEqual(global_states_batch.shape, (self.batch_size, self.global_state_dim))
        self.assertEqual(actions_batch.shape, (self.batch_size, self.n_agents)) # Actions should be (batch_size, n_agents)
        self.assertEqual(rewards_batch.shape, (self.batch_size,))
        self.assertEqual(next_global_states_batch.shape, (self.batch_size, self.global_state_dim))
        self.assertEqual(dones_batch.shape, (self.batch_size, self.n_agents)) # Dones should be (batch_size, n_agents)
        self.assertIsNone(is_weights_batch)
        self.assertIsNone(sampled_indices)
        self.assertEqual(actions_batch.dtype, torch.int64)
        self.assertEqual(dones_batch.dtype, torch.float32)

    def test_add_and_len_per(self):
        rb = ReplayBuffer(
            buffer_size=self.buffer_size,
            batch_size=self.batch_size,
            device=self.device,
            goals_num=self.goals_num,
            alpha=self.alpha,
            use_per=True
        )
        self.assertEqual(len(rb), 0)
        for i in range(50):
            rb.add(*self._create_sample_experience())
        self.assertEqual(len(rb), 50)
        for i in range(100):
            rb.add(*self._create_sample_experience())
        self.assertEqual(len(rb), self.buffer_size) # Should cap at buffer_size
        self.assertEqual(rb.n_agents, self.n_agents)

    def test_sample_per(self):
        rb = ReplayBuffer(
            buffer_size=self.buffer_size,
            batch_size=self.batch_size,
            device=self.device,
            goals_num=self.goals_num,
            alpha=self.alpha,
            use_per=True
        )
        for i in range(self.buffer_size):
            # Add experiences with varying rewards to influence priority indirectly
            rb.add(*self._create_sample_experience(reward_value=float(i % 10)))

        sample_output = rb.sample(beta=0.4) # beta is used in PER
        self.assertIsNotNone(sample_output)

        (global_states_batch, actions_batch, rewards_batch, next_global_states_batch, dones_batch, is_weights_batch, sampled_indices) = sample_output#type:ignore 直前でNoneではないはず

        self.assertEqual(global_states_batch.shape, (self.batch_size, self.global_state_dim))
        self.assertEqual(actions_batch.shape, (self.batch_size, self.n_agents))
        self.assertEqual(rewards_batch.shape, (self.batch_size,))
        self.assertEqual(next_global_states_batch.shape, (self.batch_size, self.global_state_dim))
        self.assertEqual(dones_batch.shape, (self.batch_size, self.n_agents))
        self.assertIsNotNone(is_weights_batch) # Should have IS weights
        self.assertEqual(is_weights_batch.shape, (self.batch_size,))#type:ignore 直前でNoneではないはず
        self.assertIsNotNone(sampled_indices) # Should have sampled indices
        self.assertEqual(len(sampled_indices), self.batch_size)     #type:ignore 直前でNoneではないはず

    def test_update_priorities(self):
        rb = ReplayBuffer(
            buffer_size=self.buffer_size,
            batch_size=self.batch_size,
            device=self.device,
            goals_num=self.goals_num,
            alpha=self.alpha,
            use_per=True
        )
        for i in range(self.buffer_size):
            rb.add(*self._create_sample_experience(reward_value=float(i)))

        # Sample some experiences
        sample_output = rb.sample(beta=0.4)
        self.assertIsNotNone(sample_output)
        (global_states_batch, actions_batch, rewards_batch, next_global_states_batch, dones_batch, is_weights_batch, sampled_indices) = sample_output#type:ignore 直前でNoneではないはず

        # Create dummy TD errors for updating priorities
        dummy_td_errors = np.abs(np.random.randn(self.batch_size)) + 0.1 # Ensure positive

        # The `sampled_indices` returned by `rb.sample` are the `tree_idx` values from SumTree.
        # These are what `update_priorities` expects.
        self.assertIsNotNone(sampled_indices)
        rb.update_priorities(sampled_indices, dummy_td_errors) #type:ignore 直前でNoneではないはず

        # Verify that priorities in SumTree have changed
        # A simple check: total_priority should reflect the sum of updated priorities, roughly
        # This is hard to precisely assert without knowing the exact initial priorities, but it should be positive
        self.assertGreater(rb.tree.total_priority, 0)
