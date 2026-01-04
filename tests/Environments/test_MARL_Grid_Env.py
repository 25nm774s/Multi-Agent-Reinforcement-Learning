import unittest
from unittest.mock import MagicMock

from src.Environments.MultiAgentGridEnv import MultiAgentGridEnv

class TestMultiAgentGridEnvPartialObs(unittest.TestCase):
    def setUp(self):
        # 最小限の設定
        self.args = MagicMock()
        self.args.grid_size = 10
        self.args.agents_number = 3
        self.args.goals_number = 3
        self.args.reward_mode = 1

        # テスト対象のインスタンス化
        # neighbor_distance を 3 に設定したと想定
        def manhattan_distance(p:tuple[int,int],q:tuple[int,int]): return abs(p[0] - q[0]) + abs(p[1] - q[1])

        self.env = MultiAgentGridEnv(self.args)
        self.env.neighbor_distance = 3
        self.env.distance_fn = manhattan_distance # 明示的にセット

    def test_observation_visibility_with_manhattan(self):
        """マンハッタン距離に基づき、視界内のエージェントのみ座標が表示されるか"""

        # 状況設定 (agent_0 は (3, 3) に位置)
        # agent_1: (3, 5) -> 距離 2 (視界内: 2 <= 3)
        # agent_2: (6, 4) -> 距離 3 + 1 = 4 (視界外: 4 > 3)
        positions = {
            'agent_0': (3, 3),
            'agent_1': (3, 5),  # 距離 2 (視界内)
            'agent_2': (6, 4),  # 距離 4 (視界外)
            'goal_0': (3, 0),
            'goal_1': (0, 0),
            'goal_2': (9, 9)
        }
        self.env._grid._object_positions = positions
        obs = self.env._get_observation()

        # agent_0 の視点
        a0_view = obs['agent_0']

        # --- エージェントの可視性チェック ---
        # 自分の位置は常に見える
        self.assertEqual(a0_view['self'], (3, 3))
        # agent_1 は距離2なので見える
        self.assertEqual(a0_view['others']['agent_1'], (3, 5))
        # agent_2 は距離4なので見えない (-1, -1)
        self.assertEqual(a0_view['others']['agent_2'], (-1, -1))

        # --- ゴールの可視性チェック ---
        self.assertEqual(a0_view['all_goals'], [(3, 0),(0, 0),(9, 9)])

import unittest
from unittest.mock import MagicMock

# マンハッタン距離の定義（テスト用）
def manhattan_distance(pos1, pos2):
    return abs(pos1[0] - pos2[0]) + abs(pos1[1] - pos2[1])

class TestMultiAgentGridEnvObservationUpdate(unittest.TestCase):
    def setUp(self):
        # 設定オブジェクト
        self.args = MagicMock()
        self.args.grid_size = 10
        self.args.agents_number = 2
        self.args.goals_number = 2
        self.args.reward_mode = 1
        self.args.neighbor_distance = 3  # 観測半径を3に設定

        # 座標のセットアップ
        # agent_0: (0,0), agent_1: (2,2) -> 距離 4 (視界外)
        # goal_0:  (1,1) -> (agent_0のゴール)
        self.positions = {
            'agent_0': (0, 0),
            'agent_1': (2, 2),
            'goal_0': (1, 1),
            'goal_1': (9, 9)
        }

        # 2. 距離計算関数を注入して初期化
        self.env = MultiAgentGridEnv(
            self.args,
            distance_fn=manhattan_distance
        )
        # 設定が正しく反映されているか確認
        self.env.neighbor_distance = 3

        self.env._grid._object_positions = self.positions

    def test_observation_is_dict_not_tuple(self):
        """戻り値がタプルではなく、エージェントIDをキーとした辞書であることを確認"""
        obs = self.env._get_observation()

        self.assertIsInstance(obs, dict, "観測データは辞書形式である必要があります")
        self.assertEqual(len(obs), 2)
        self.assertIn('agent_0', obs)
        self.assertIn('agent_1', obs)

    def test_partial_visibility_logic(self):
        """距離に基づいて他者の位置が正しく隠蔽されているか"""
        obs = self.env._get_observation()

        # agent_0 から見た視界
        a0_obs = obs['agent_0']

        # 自分の位置と自分のゴールは常に見える
        self.assertEqual(a0_obs['self'], (0, 0))
        self.assertEqual(a0_obs['all_goals'], [(1, 1),(9,9)])

        # agent_1 (2,2) までの距離は |0-2| + |0-2| = 4
        # neighbor_distance=3 なので、(-1, -1) になっているべき
        self.assertEqual(a0_obs['others']['agent_1'], (-1, -1),
                         "視界外のエージェントは (-1, -1) である必要があります")

    def test_full_visibility_when_near(self):
        """距離が近い場合は他者の位置が見えるか"""
        # neighbor_distance を 5 に引き上げれば、距離4の agent_1 が見えるはず
        self.env.neighbor_distance = 5
        obs = self.env._get_observation()

        self.assertEqual(obs['agent_0']['others']['agent_1'], (2, 2),
                         "視界内のエージェントの座標が正しく取得できていません")


class TestMultiAgentGridEnvCycle(unittest.TestCase):
    def setUp(self):
        # 距離計算関数の定義
        def manhattan_distance(p1, p2):
            return abs(p1[0]-p2[0]) + abs(p1[1]-p2[1])

        self.args = MagicMock()
        self.args.grid_size = 5
        self.args.agents_number = 2
        self.args.goals_number = 2
        self.args.reward_mode = 1
        self.args.neighbor_distance = 3

        # Fixed goals to ensure goals are always present in the grid
        self.fixed_test_goals = [(0, 0), (self.args.grid_size - 1, self.args.grid_size - 1)]

        self.env = MultiAgentGridEnv(self.args, fixrd_goals=self.fixed_test_goals, distance_fn=manhattan_distance)

        # 内部メソッドのモック化（計算ロジック自体は別でテスト済みのため）
        self.env._calculate_reward = MagicMock(return_value={"agent_0":0.2,"agent_1":1.9})
        self.env._check_done_condition = MagicMock(return_value=False)

    def test_reset_returns_correct_obs_format(self):
        """reset() がタプルではなく辞書形式の観測を返すか"""
        obs = self.env.reset() # Call reset to initialize agents and get observation

        # 検証：戻り値が辞書であり、キーにエージェントIDが含まれていること
        self.assertIsInstance(obs, dict)
        self.assertIn('agent_0', obs)
        self.assertIsInstance(obs['agent_0'], dict)
        self.assertIn('others', obs['agent_0'])

    def test_step_returns_correct_obs_format(self):
        """step() の第1戻り値が辞書形式の観測を返すか"""

        _ = self.env.reset(initial_agent_positions=[(1,1),(2,2)]) # Reset with agents for a valid state

        actions = [0, 0] # 全エージェントが「静止」または「上」などのダミーアクション
        obs, rewards, done, info = self.env.step(actions)

        # 検証：第1戻り値(obs)が新仕様の辞書であること
        self.assertIsInstance(obs, dict)
        self.assertEqual(len(obs), self.args.agents_number)

        # 検証：その他の戻り値の型が変わっていないこと
        self.assertIsInstance(rewards, dict)
        self.assertIsInstance(done, dict)
        self.assertIsInstance(info, dict)

    def test_step_error_on_mismatched_actions(self):
        """エージェント数とアクション数が合わない場合に正しくエラーが出るか"""
        self.env.reset()
        with self.assertRaises(ValueError):
            self.env.step([0]) # エージェントは2人なのにアクションが1つ

    def test_step_returns_individual_rewards(self):
        """step() がエージェントごとの報酬辞書を返すかテスト"""
        self.env.reset()

        # 期待される個別報酬をモックで設定
        self.env._calculate_reward = MagicMock(return_value={'agent_0': 3.3, 'agent_1': -1.3})

        actions = [1, 1]
        obs, rewards, done, info = self.env.step(actions)

        # 1. 報酬が辞書であること
        self.assertIsInstance(rewards, dict)
        self.assertEqual(rewards['agent_0'], 3.3)
        self.assertEqual(rewards['agent_1'], -1.3)

        # 2. QMIX用に全体報酬（和）がinfoに含まれているか（オプション）
        self.assertIn('total_reward', info)
        self.assertAlmostEqual(info['total_reward'], 2.0)

    def test_iql_weighting_scenario(self):
        """IQLで報酬に重みを付けるシミュレーション"""
        self.env.reset(initial_agent_positions=[(1,1), (2,2)]) # Ensure agents are placed

        _, rewards, _, _ = self.env.step([0, 0])

        # エージェントごとに異なる重み（重要度）をかけるケース
        weights = {'agent_0': 1.0, 'agent_1': 0.5}
        weighted_rewards = {k: rewards[k] * weights[k] for k in rewards}

        self.assertEqual(weighted_rewards['agent_1'], rewards['agent_1'] * 0.5)

    def test_observation_consistency(self):
        """観測内のゴールリストの順序が不変であることをテスト"""
        # 環境をリセットしてエージェントを配置
        self.env.reset(initial_agent_positions=[(1,1), (self.args.grid_size - 2, self.args.grid_size - 2)])

        obs = self.env._get_observation()

        # すべてのエージェントが「同じ順序」でゴールのリストを受け取っているか
        a0_goals = obs['agent_0']['all_goals']
        a1_goals = obs['agent_1']['all_goals']

        self.assertEqual(a0_goals, a1_goals)
        self.assertEqual(a0_goals[0], self.fixed_test_goals[0]) # Use fixed goals from setUp
        self.assertEqual(a0_goals[1], self.fixed_test_goals[1]) # Use fixed goals from setUp
