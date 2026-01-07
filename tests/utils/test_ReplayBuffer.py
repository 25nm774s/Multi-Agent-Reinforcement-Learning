import unittest
import torch
import numpy as np
from typing import Dict,List,Any

from src.utils.replay_buffer import ReplayBuffer

class TestReplayBuffer(unittest.TestCase):
    def setUp(self):
        self.buffer_size = 100
        self.batch_size = 32
        self.device = torch.device('cpu')
        self.alpha = 0.6
        self.n_agents = 2
        self.goals_num = 2
        self._goal_ids: List[str] = [f'goal_{i}' for i in range(self.goals_num)]
        self._agent_ids: List[str] = [f'agent_{i}' for i in range(self.n_agents)]
        # self.global_state_dim = (self.goals_num + self.n_agents) * 2 # Reinstated

    def _create_sample_experience(self, action_value: int = 0, reward_value: float = 1.0, done_value: bool = False):
        # global_state and next_global_state will now be GlobalState dicts
        global_state: Dict[str, Dict[str, Any]] = {
            'agent_0': {'self': (0,0), 'all_goals': [(1,1)], 'others': {'agent_1': (0,1)}},
            'agent_1': {'self': (0,1), 'all_goals': [(1,1)], 'others': {'agent_0': (0,0)}}}
        actions = [action_value] * self.n_agents
        # rewards should be a Dict[str, float]
        rewards_dict = {agent_id: reward_value for agent_id in self._agent_ids}
        next_global_state: Dict[str, Dict[str, Any]] = {
            'agent_0': {'self': (0,1), 'all_goals': [(1,1)], 'others': {'agent_1': (0,2)}},
            'agent_1': {'self': (0,2), 'all_goals': [(1,1)], 'others': {'agent_0': (0,1)}}}
        # dones should be a Dict[str, bool]
        dones_dict = {agent_id: done_value for agent_id in self._agent_ids}
        return global_state, actions, rewards_dict, next_global_state, dones_dict

    def test_add_and_len_uniform(self):
        rb = ReplayBuffer(
            buffer_size=self.buffer_size,
            batch_size=self.batch_size,
            device=self.device,
            goals_number=self.goals_num,
            goal_ids=self._goal_ids,
            agent_ids=self._agent_ids,
            use_per=False
        )
        self.assertEqual(len(rb), 0)
        for i in range(50):
            rb.add(*self._create_sample_experience()) # Call with Dict[str, Dict[str, Any]]
        self.assertEqual(len(rb), 50)
        for i in range(100):
            rb.add(*self._create_sample_experience()) # Call with Dict[str, Dict[str, Any]]
        self.assertEqual(len(rb), self.buffer_size) # Should cap at buffer_size
        self.assertEqual(rb.n_agents, self.n_agents)

    def test_sample_uniform(self):
        rb = ReplayBuffer(
            buffer_size=self.buffer_size,
            batch_size=self.batch_size,
            device=self.device,
            goals_number=self.goals_num,
            goal_ids=self._goal_ids,
            agent_ids=self._agent_ids,
            use_per=False
        )
        for i in range(self.buffer_size):
            # Pass individual reward dicts
            rewards = {agent_id: float(i) for agent_id in self._agent_ids}
            rb.add(*self._create_sample_experience(action_value=i%5, reward_value=float(i)))

        sample_output = rb.sample(beta=0.0) # beta is not used in uniform
        self.assertIsNotNone(sample_output)

        (global_states_batch, actions_batch, rewards_batch, next_global_states_batch, dones_batch, is_weights_batch, sampled_indices) = sample_output#type:ignore

        # Now expecting a list of observation dictionaries
        self.assertIsInstance(global_states_batch, list)
        self.assertIsInstance(global_states_batch[0], dict)
        self.assertEqual(len(global_states_batch), self.batch_size)

        self.assertEqual(actions_batch.shape, (self.batch_size, self.n_agents)) # Actions should be (batch_size, n_agents)
        # rewards_batch should now be (batch_size, n_agents)
        self.assertEqual(rewards_batch.shape, (self.batch_size, self.n_agents))

        self.assertIsInstance(next_global_states_batch, list)
        self.assertIsInstance(next_global_states_batch[0], dict)
        self.assertEqual(len(next_global_states_batch), self.batch_size)

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
            goals_number=self.goals_num,
            goal_ids=self._goal_ids,
            agent_ids=self._agent_ids,
            alpha=self.alpha,
            use_per=True
        )
        self.assertEqual(len(rb), 0)
        for i in range(50):
            rb.add(*self._create_sample_experience()) # Call with Dict[str, Dict[str, Any]]
        self.assertEqual(len(rb), 50)
        for i in range(100):
            rb.add(*self._create_sample_experience()) # Call with Dict[str, Dict[str, Any]]
        self.assertEqual(len(rb), self.buffer_size) # Should cap at buffer_size
        self.assertEqual(rb.n_agents, self.n_agents)

    def test_sample_per(self):
        rb = ReplayBuffer(
            buffer_size=self.buffer_size,
            batch_size=self.batch_size,
            device=self.device,
            goals_number=self.goals_num,
            goal_ids=self._goal_ids,
            agent_ids=self._agent_ids,
            alpha=self.alpha,
            use_per=True
        )
        for i in range(self.buffer_size):
            # Add experiences with varying rewards to influence priority indirectly
            rb.add(*self._create_sample_experience(reward_value=float(i % 10)))

        sample_output = rb.sample(beta=0.4) # beta is used in PER
        self.assertIsNotNone(sample_output)

        (global_states_batch, actions_batch, rewards_batch, next_global_states_batch, dones_batch, is_weights_batch, sampled_indices) = sample_output#type:ignore

        # Now expecting a list of observation dictionaries
        self.assertIsInstance(global_states_batch, list)
        self.assertIsInstance(global_states_batch[0], dict)
        self.assertEqual(len(global_states_batch), self.batch_size)

        self.assertEqual(actions_batch.shape, (self.batch_size, self.n_agents))
        # rewards_batch should now be (batch_size, n_agents)
        self.assertEqual(rewards_batch.shape, (self.batch_size, self.n_agents))

        self.assertIsInstance(next_global_states_batch, list)
        self.assertIsInstance(next_global_states_batch[0], dict)
        self.assertEqual(len(next_global_states_batch), self.batch_size)

        self.assertEqual(dones_batch.shape, (self.batch_size, self.n_agents))
        self.assertIsNotNone(is_weights_batch) # Should have IS weights
        self.assertEqual(is_weights_batch.shape, (self.batch_size,))#type:ignore
        self.assertIsNotNone(sampled_indices) # Should have sampled indices
        self.assertEqual(len(sampled_indices), self.batch_size)#type:ignore

    def test_update_priorities(self):
        rb = ReplayBuffer(
            buffer_size=self.buffer_size,
            batch_size=self.batch_size,
            device=self.device,
            goals_number=self.goals_num,
            goal_ids=self._goal_ids,
            agent_ids=self._agent_ids,
            alpha=self.alpha,
            use_per=True
        )
        for i in range(self.buffer_size):
            rb.add(*self._create_sample_experience(reward_value=float(i)))

        # Sample some experiences
        sample_output = rb.sample(beta=0.4)
        self.assertIsNotNone(sample_output)
        (global_states_batch, actions_batch, rewards_batch, next_global_states_batch, dones_batch, is_weights_batch, sampled_indices) = sample_output#type:ignore

        # Create dummy TD errors for updating priorities
        dummy_td_errors = np.abs(np.random.randn(self.batch_size)) + 0.1 # Ensure positive

        # The `sampled_indices` returned by `rb.sample` are the `tree_idx` values from SumTree.
        # These are what `update_priorities` expects.
        self.assertIsNotNone(sampled_indices)
        rb.update_priorities(sampled_indices, dummy_td_errors)#type:ignore

        # Verify that priorities in SumTree have changed
        # A simple check: total_priority should reflect the sum of updated priorities, roughly
        # This is hard to precisely assert without knowing the exact initial priorities, but it should be positive
        self.assertGreater(rb.tree.total_priority, 0)
