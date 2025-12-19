'''
### `ReplayBuffer` ユニットテストの概要

このユニットテストは、`ReplayBuffer` クラスの主要な機能が正しく動作するかどうかを確認するために作成されました。テストは、通常の（一様サンプリング）リプレイバッファと、優先順位付き経験再生（PER）を使用するリプレイバッファの両方に対して行われます。

**テストケース一覧:**

1.  **`test_add_uniform_buffer`**:
    *   一様サンプリングのリプレイバッファに経験を追加する機能をテストします。
    *   バッファサイズ制限内での追加と、バッファが満杯になったときに古い経験が上書きされることを確認します。

2.  **`test_add_per_buffer`**:
    *   PERバッファに経験を追加する機能をテストします。
    *   追加された経験の数と、`SumTree`内の合計優先度が正しく更新されていることを確認します。
    *   バッファサイズを超えて経験が追加された場合の動作も確認します。

3.  **`test_sample_uniform_buffer`**:
    *   一様サンプリングのリプレイバッファからの経験サンプリングをテストします。
    *   バッファが満杯のときに正しくバッチがサンプリングされること、返されるテンソルの形状が正しいこと、およびPER固有の`is_weights`と`original_indices`が`None`であることを確認します。
    *   バッチサイズ分の経験がまだ蓄積されていない場合にサンプリングが`None`を返すことを確認します。

4.  **`test_sample_per_buffer`**:
    *   PERバッファからの経験サンプリングをテストします。
    *   バッファが満杯のときに正しくバッチがサンプリングされること、返されるテンソルの形状が正しいこと、および`is_weights`と`original_indices`が正しく返されることを確認します。
    *   バッチサイズ分の経験がまだ蓄積されていない場合にサンプリングが`None`を返すことを確認します。

5.  **`test_update_priorities_per_buffer`**:
    *   PERバッファにおける経験の優先度更新機能をテストします。
    *   サンプリングされた経験のTD誤差に基づいて`SumTree`内の優先度が正しく更新されることを確認します。
    *   `_max_priority`が、新規経験の初期優先度設定のために、観測された最高の優先度を保持し続けることを確認します。

6.  **`test_sample_empty_buffer`**:
    *   空のリプレイバッファ（一様サンプリングおよびPERの両方）からサンプリングしようとした場合に`None`が返されることを確認します。

7.  **`test_sample_not_enough_for_batch`**:
    *   バッチサイズに必要な数より少ない経験しか含まれていないリプレイバッファ（一様サンプリングおよびPERの両方）からサンプリングしようとした場合に`None`が返されることを確認します。

これらのテストを通じて、`ReplayBuffer`クラスが指定された機能要件を満たし、エッジケースも適切に処理できることを検証しています。
'''
import unittest
import torch
import numpy as np
import random
from collections import deque

from src.utils.replay_buffer import ReplayBuffer
# Assuming SumTree and ReplayBuffer classes are defined in the current notebook scope
# (as they are in the provided notebook state)

class TestReplayBuffer(unittest.TestCase):
    def setUp(self):
        self.device = torch.device('cpu') # Use CPU for testing
        self.buffer_size = 10
        self.batch_size = 4
        self.alpha = 0.6 # PER parameter
        self.beta_init = 0.4 # PER parameter

        # Initialize ReplayBuffer for uniform sampling
        self.uniform_buffer = ReplayBuffer(
            learning_mode='dqn',
            buffer_size=self.buffer_size,
            batch_size=self.batch_size,
            device=self.device,
            use_per=False
        )

        # Initialize ReplayBuffer for prioritized experience replay (PER)
        self.per_buffer = ReplayBuffer(
            learning_mode='dqn',
            buffer_size=self.buffer_size,
            batch_size=self.batch_size,
            device=self.device,
            alpha=self.alpha,
            use_per=True
        )

        # Dummy experience data
        self.dummy_experience = (
            ((0.0, 0.0), (1.0, 1.0)), # global_state (tuple of tuples)
            0, # action
            1.0, # reward
            ((0.1, 0.1), (1.1, 1.1)), # next_global_state
            False # done
        )

    def _add_dummy_experiences(self, buffer, num_experiences):
        for i in range(num_experiences):
            global_state = ((i * 0.1, i * 0.1), (i * 0.1 + 1.0, i * 0.1 + 1.0))
            action = i % 2
            reward = float(i)
            next_global_state = ((i * 0.1 + 0.1, i * 0.1 + 0.1), (i * 0.1 + 1.1, i * 0.1 + 1.1))
            done = (i % 3 == 0)
            buffer.add(global_state, action, reward, next_global_state, done)

    def test_add_uniform_buffer(self):
        # Test adding experiences to uniform buffer
        self._add_dummy_experiences(self.uniform_buffer, 5)
        self.assertEqual(len(self.uniform_buffer), 5)
        self._add_dummy_experiences(self.uniform_buffer, 10) # Overfill buffer
        self.assertEqual(len(self.uniform_buffer), self.buffer_size)

    def test_add_per_buffer(self):
        # Test adding experiences to PER buffer
        self._add_dummy_experiences(self.per_buffer, 5)
        self.assertEqual(len(self.per_buffer), 5)
        self.assertEqual(self.per_buffer.size, 5)
        self.assertAlmostEqual(self.per_buffer.tree.total_priority, 5.0) # All added with max_priority=1.0

        self._add_dummy_experiences(self.per_buffer, 10) # Overfill buffer
        self.assertEqual(len(self.per_buffer), self.buffer_size)
        self.assertEqual(self.per_buffer.size, self.buffer_size)
        self.assertAlmostEqual(self.per_buffer.tree.total_priority, float(self.buffer_size))

    def test_sample_uniform_buffer(self):
        self._add_dummy_experiences(self.uniform_buffer, self.buffer_size)
        self.assertEqual(len(self.uniform_buffer), self.buffer_size)

        # Test sampling when buffer is full
        global_states, actions, rewards, next_global_states, dones, is_weights, original_indices = self.uniform_buffer.sample(self.beta_init)#type:ignore Noneが返されてもいいかどうかをチェックするため。

        self.assertIsNotNone(global_states)
        self.assertEqual(global_states.shape, (self.batch_size, 4))
        self.assertEqual(actions.shape, (self.batch_size,))
        self.assertEqual(rewards.shape, (self.batch_size,))
        self.assertEqual(next_global_states.shape, (self.batch_size, 4))
        self.assertEqual(dones.shape, (self.batch_size,))
        self.assertIsNone(is_weights)
        self.assertIsNone(original_indices)

        # Test sampling when buffer is not yet full enough
        empty_buffer = ReplayBuffer('dqn', self.buffer_size, self.batch_size, self.device, use_per=False)
        self._add_dummy_experiences(empty_buffer, self.batch_size - 1)
        self.assertIsNone(empty_buffer.sample(self.beta_init))

    def test_sample_per_buffer(self):
        self._add_dummy_experiences(self.per_buffer, self.buffer_size)
        self.assertEqual(len(self.per_buffer), self.buffer_size)

        # Test sampling when buffer is full
        global_states, actions, rewards, next_global_states, dones, is_weights, original_indices = self.per_buffer.sample(self.beta_init)#type:ignore Noneが返されてもいいかどうかをチェックするため。

        self.assertIsNotNone(global_states)
        self.assertEqual(global_states.shape, (self.batch_size, 4))
        self.assertEqual(actions.shape, (self.batch_size,))
        self.assertEqual(rewards.shape, (self.batch_size,))
        self.assertEqual(next_global_states.shape, (self.batch_size, 4))
        self.assertEqual(dones.shape, (self.batch_size,))
        self.assertIsNotNone(is_weights)
        self.assertEqual(is_weights.shape, (self.batch_size,))
        self.assertIsNotNone(original_indices)
        self.assertEqual(len(original_indices), self.batch_size)

        # Test sampling when buffer is not yet full enough
        empty_per_buffer = ReplayBuffer('dqn', self.buffer_size, self.batch_size, self.device, use_per=True)
        self._add_dummy_experiences(empty_per_buffer, self.batch_size - 1)
        self.assertIsNone(empty_per_buffer.sample(self.beta_init))

    def test_update_priorities_per_buffer(self):
        self._add_dummy_experiences(self.per_buffer, self.buffer_size)
        # Sample some experiences to get tree_indices
        _, _, _, _, _, _, sampled_original_indices = self.per_buffer.sample(self.beta_init)#type:ignore Noneが返されてもいいかどうかをチェックするため。

        # For PER, the sampled_original_indices are the indices in self.experiences
        # We need to map them back to tree_indices by adding (capacity - 1)
        tree_indices_to_update = [idx + (self.per_buffer.tree.capacity - 1) for idx in sampled_original_indices]
        td_errors = np.array([0.5, 0.1, 0.9, 0.3]) # Example TD errors

        self.per_buffer.update_priorities(tree_indices_to_update, td_errors)

        # Verify priorities are updated
        min_error = 1e-6 # Matches ReplayBuffer's internal min_error
        for i, tree_idx in enumerate(tree_indices_to_update):
            expected_priority_in_tree = max(np.abs(td_errors[i]), min_error)
            self.assertAlmostEqual(self.per_buffer.tree.tree[tree_idx], expected_priority_in_tree)
            # Ensure total priority is updated correctly
            self.assertTrue(self.per_buffer.tree.total_priority > 0)

        # Test _max_priority update.
        # The _max_priority should be max(initial_max_priority_of_buffer, max_td_error_in_this_batch).
        # Since initial _max_priority was 1.0 (from add calls) and max TD error is 0.9,
        # _max_priority should remain 1.0.
        self.assertAlmostEqual(self.per_buffer._max_priority, 1.0)

    def test_sample_empty_buffer(self):
        # Test uniform buffer
        self.assertIsNone(self.uniform_buffer.sample(self.beta_init))
        # Test PER buffer
        self.assertIsNone(self.per_buffer.sample(self.beta_init))

    def test_sample_not_enough_for_batch(self):
        # Uniform buffer
        self._add_dummy_experiences(self.uniform_buffer, self.batch_size - 1)
        self.assertIsNone(self.uniform_buffer.sample(self.beta_init))

        # PER buffer
        self._add_dummy_experiences(self.per_buffer, self.batch_size - 1)
        self.assertIsNone(self.per_buffer.sample(self.beta_init))


# This block allows running the tests directly from the notebook
# if __name__ == '__main__':
#     # To run tests in a notebook, you might need to use a TextTestRunner
#     # or load them into a test suite. A simpler way for a quick check:
#     suite = unittest.TestSuite()
#     # Updated to use TestLoader().loadTestsFromTestCase to avoid DeprecationWarning
#     suite.addTest(unittest.TestLoader().loadTestsFromTestCase(TestReplayBuffer))
#     runner = unittest.TextTestRunner()
#     runner.run(suite)