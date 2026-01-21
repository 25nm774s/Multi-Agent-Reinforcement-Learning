import unittest
from unittest.mock import MagicMock, patch

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
        self.args.reward_mode = 2
        self.args.neighbor_distance = 3

        # Fixed goals to ensure goals are always present in the grid
        self.fixed_test_goals = [(0, 0), (self.args.grid_size - 1, self.args.grid_size - 1)]

        self.env = MultiAgentGridEnv(self.args, fixrd_goals=self.fixed_test_goals, distance_fn=manhattan_distance)

        # 内部メソッドのモック化（計算ロジック自体は別でテスト済みのため）
        self.env._calculate_reward = MagicMock(return_value={"agent_0":0.2,"agent_1":1.9})
        mock_dones = {"agent_0": True, "agent_1": True, "__all__": True}
        self.env._check_done_condition = MagicMock(return_value=mock_dones)

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

        actions: dict[str, int] = {"agent_0": 0, "agent_1": 0} # 全エージェントが「静止」または「上」などのダミーアクション
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
            self.env.step({"agent_0": 0}) # エージェントは2人なのにアクションが1つ

    def test_step_returns_individual_rewards(self):
        """step() がエージェントごとの報酬辞書を返すかテスト"""
        self.env.reset()

        # 期待される個別報酬をモックで設定
        self.env._calculate_reward = MagicMock(return_value={'agent_0': 3.3, 'agent_1': -1.3})

        actions: dict[str, int] = {"agent_0": 1, "agent_1": 1}
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

        _, rewards, _, _ = self.env.step({"agent_0": 0, "agent_1": 0})

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

class TestMultiAgentGridEnv(unittest.TestCase):
    def setUp(self):
        """テストごとの前準備"""
        self.args = MagicMock()
        self.args.grid_size = 5
        self.args.agents_number = 2
        self.args.goals_number = 2
        self.args.reward_mode = 1
        self.args.neighbor_distance = 1

        with patch('src.Environments.Grid'), patch('src.Environments.CollisionResolver'):
            self.env = MultiAgentGridEnv(self.args)

    def test_calculate_distance_basic(self):
        """キーがエージェントID（文字列）になっているか確認"""
        self.env.get_agent_positions = MagicMock(return_value={"agent_0": (0, 0)})
        self.env.get_goal_positions = MagicMock(return_value={"goal_0": (2, 3)})
        
        distances = self.env._calculate_total_distance_to_goals()
        
        # キーが "agent_0" であることを確認
        self.assertIn("agent_0", distances)
        self.assertEqual(distances["agent_0"], 5)

    def test_greedy_pairing_logic(self):
        """IDをキーとしたペアリングの検証"""
        self.env.get_agent_positions = MagicMock(return_value={
            "agent_0": (0, 0), "agent_1": (1, 1)
        })
        self.env.get_goal_positions = MagicMock(return_value={
            "goal_0": (0, 1), "goal_1": (10, 10)
        })
        
        distances = self.env._calculate_total_distance_to_goals()
        
        # インデックスではなくIDでアクセス
        self.assertEqual(distances["agent_0"], 1)
        self.assertEqual(distances["agent_1"], 18)

    def test_more_agents_than_goals(self):
        """エージェント数がゴール数より多い場合"""
        self.env.get_agent_positions = MagicMock(return_value={
            "agent_0": (0, 0), 
            "agent_1": (5, 5)
        })
        self.env.get_goal_positions = MagicMock(return_value={
            "goal_0": (0, 1)
        })
        
        distances = self.env._calculate_total_distance_to_goals()
        
        # 「ペアなし」も inf で含まれるため、長さはエージェント数と同じ 2 になる
        self.assertEqual(len(distances), 2)
        
        # agent_0 は近いので 1.0
        self.assertIn("agent_0", distances)
        self.assertEqual(distances["agent_0"], 1.0)
        
        # agent_1 はペアなしなので inf
        self.assertIn("agent_1", distances)
        self.assertEqual(distances["agent_1"], float('inf'))

    def test_calculation_overhead_warning(self):
        """計算量上限を超えた場合に例外が発生するか"""
        # 21 * 20 = 420 > 400
        self.env.agents_number = 21
        self.env.goals_number = 20
        
        with self.assertRaises(Exception) as cm:
            self.env._calculate_total_distance_to_goals()
        
        self.assertIn("計算量のオーバーヘッド", str(cm.exception))

    def test_no_agents(self):
        """
        エージェントの位置情報が取得できない場合でも、
        定義されている全エージェントIDに対して inf を返すか
        """
        # get_agent_positions が空を返す状況をシミュレート
        self.env.get_agent_positions = MagicMock(return_value={})
        self.env.get_goal_positions = MagicMock(return_value={"goal_0": (1, 1)})
        
        distances = self.env._calculate_total_distance_to_goals()
        
        # 期待値: 空の辞書 {} ではなく、全IDが含まれ、値が inf であること
        expected = {aid: float('inf') for aid in self.env._agent_ids}
        self.assertEqual(distances, expected)

    def test_more_agents_than_goals_all_keys_exist(self):
        """
        エージェント数がゴール数より多い場合でも、
        すべてのエージェントIDが辞書のキーに含まれているか
        """
        # エージェント2人、ゴール1つの設定
        self.env.get_agent_positions = MagicMock(return_value={
            "agent_0": (0, 0), 
            "agent_1": (5, 5)
        })
        self.env.get_goal_positions = MagicMock(return_value={
            "goal_0": (0, 1)
        })
        
        distances = self.env._calculate_total_distance_to_goals()
        
        # 1. すべてのエージェントIDがキーに存在するか確認
        self.assertEqual(len(distances), 2)
        self.assertIn("agent_0", distances)
        self.assertIn("agent_1", distances)
        
        # 2. 近い方のエージェントには実際の距離、ペアなしには inf が入っているか
        self.assertEqual(distances["agent_0"], 1.0)
        self.assertEqual(distances["agent_1"], float('inf'))

    def test_no_goals_all_inf(self):
        """
        ゴールが1つもない場合、すべてのエージェントの距離が inf になるか
        """
        self.env.get_agent_positions = MagicMock(return_value={
            "agent_0": (0, 0), 
            "agent_1": (1, 1)
        })
        self.env.get_goal_positions = MagicMock(return_value={}) # ゴールなし
        
        distances = self.env._calculate_total_distance_to_goals()
        
        # 全員分が inf で返ってくることを期待
        self.assertEqual(distances["agent_0"], float('inf'))
        self.assertEqual(distances["agent_1"], float('inf'))

class TestRewardCalculation(unittest.TestCase):
    def setUp(self):
        # 環境の簡易セットアップ
        self.args = MagicMock()
        self.args.grid_size = 5
        self.args.agents_number = 2
        self.args.goals_number = 2
        self.args.reward_mode = 1
        
        # テスト対象クラスをインスタンス化
        with patch('src.Environments.Grid'), patch('src.Environments.CollisionResolver'):
            # MultiAgentGridEnv の __init__ が動作する前提
            self.env = MultiAgentGridEnv(self.args)
            self.env._agent_ids = ["agent_0", "agent_1"]
            # Reward Shaping用の初期化
            self.env.prev_distances = {"agent_0": 5.0, "agent_1": 5.0}

    def test_reward_mode_0_done(self):
        """モード0: 全員ゴール時に +100 + 同時ボーナス(5*2=10) = 110"""
        self.env.reward_mode = 0
        # 修正された _check_done_condition の戻り値を模倣
        mock_dones = {"agent_0": True, "agent_1": True, "__all__": True}
        self.env._check_done_condition = MagicMock(return_value=mock_dones)
        
        rewards = self.env._calculate_reward(done_mode=0)
        
        # 100(完了) + 10(2人同時ボーナス) = 110.0
        self.assertEqual(rewards["agent_0"], 110.0)
        self.assertEqual(rewards["agent_1"], 110.0)

    def test_reward_mode_1_not_done(self):
        """モード1: 未完了時は全員 -0.1 (ステップペナルティのみ)"""
        self.env.reward_mode = 1
        mock_dones = {"agent_0": False, "agent_1": False, "__all__": False}
        self.env._check_done_condition = MagicMock(return_value=mock_dones)
        
        rewards = self.env._calculate_reward(done_mode=1)
        
        self.assertEqual(rewards["agent_0"], -0.1)
        self.assertEqual(rewards["agent_1"], -0.1)

    def test_reward_mode_2_distance_based(self):
        """モード2: 距離に基づく負の報酬 + 到着状況に応じたボーナス"""
        self.env.reward_mode = 2
        # agent_0だけゴールにいる状況
        mock_dones = {"agent_0": True, "agent_1": False, "__all__": False}
        self.env._check_done_condition = MagicMock(return_value=mock_dones)
        
        mock_distances = {"agent_0": 0.0, "agent_1": float('inf')}
        self.env._calculate_total_distance_to_goals = MagicMock(return_value=mock_distances)
        
        rewards = self.env._calculate_reward(done_mode=2)
        
        # agent_0: -0.0(距離) + 5.0(1人ゴールボーナス) = 5.0
        self.assertEqual(rewards["agent_0"], 5.0)
        # agent_1: -10.0(クリップされた距離) + 5.0(1人ゴールボーナス) = -5.0
        self.assertEqual(rewards["agent_1"], -5.0)

    def test_reward_mode_3_potential_shaping(self):
        """モード3: ポテンシャル報酬の計算テスト"""
        self.env.reward_mode = 3
        # 前回の距離は setUp で 5.0 に設定済み
        # 今回 agent_0 は接近(3.0)、agent_1 は離脱(6.0)
        current_distances = {"agent_0": 3.0, "agent_1": 6.0}
        self.env._calculate_total_distance_to_goals = MagicMock(return_value=current_distances)
        
        mock_dones = {"agent_0": False, "agent_1": False, "__all__": False}
        self.env._check_done_condition = MagicMock(return_value=mock_dones)
        
        rewards = self.env._calculate_reward(done_mode=2)
        
        # agent_0: (5.0-3.0) - 0.01(penalty) + 0(bonus) = 1.99
        self.assertAlmostEqual(rewards["agent_0"], 1.99)
        # agent_1: (5.0-6.0) - 0.01(penalty) + 0(bonus) = -1.01
        self.assertAlmostEqual(rewards["agent_1"], -1.01)
        
        # prev_distances が更新されているか確認
        self.assertEqual(self.env.prev_distances["agent_0"], 3.0)

    def test_reward_mode_3_moving_away(self):
        """モード3: ゴールから遠ざかった場合に負の報酬が出るか"""
        self.env.reward_mode = 3
        self.env.prev_distances = {"agent_0": 2.0}
        current_distances = {"agent_0": 5.0} # 遠ざかった
        self.env._calculate_total_distance_to_goals = MagicMock(return_value=current_distances)
        self.env._check_done_condition = MagicMock(return_value={"agent_0": False, "__all__": False})

        rewards = self.env._calculate_reward(done_mode=2)
        # (2.0 - 5.0) - 0.01 = -3.01
        self.assertAlmostEqual(rewards["agent_0"], -3.01)

    def test_reward_mode_2_clipping(self):
        """モード2: 距離報酬が grid_size * 2 でクリップされるか"""
        self.env.reward_mode = 2
        self.env.grid_size = 5 # max_penalty = 10
        mock_distances = {"agent_0": 99.0} # 異常に遠い距離
        self.env._calculate_total_distance_to_goals = MagicMock(return_value=mock_distances)
        self.env._check_done_condition = MagicMock(return_value={"agent_0": False, "__all__": False})

        rewards = self.env._calculate_reward(done_mode=2)
        # -min(99.0, 10.0) + bonus(0) = -10.0
        self.assertEqual(rewards["agent_0"], -10.0)

class TestDoneCondition(unittest.TestCase):
    def setUp(self):
        self.args = MagicMock()
        self.args.grid_size = 5
        self.args.agents_number = 2
        self.args.goals_number = 2
        
        with patch('src.Environments.Grid'), patch('src.Environments.CollisionResolver'):
            self.env = MultiAgentGridEnv(self.args)
            self.env._agent_ids = ["agent_0", "agent_1"]

    def set_positions(self, agent_pos: dict, goal_pos: dict):
        """テスト用に位置情報をセットするヘルパーメソッド"""
        self.env.get_agent_positions = MagicMock(return_value=agent_pos)
        self.env.get_goal_positions = MagicMock(return_value=goal_pos)

    def test_mode_0_all_goals_covered(self):
        """Mode 0: 全てのゴールが占有されている必要がある"""
        # ケース1: 2つのゴールに対して2人が別々に乗っている (True)
        self.set_positions(
            {"agent_0": (1, 1), "agent_1": (2, 2)},
            {"goal_0": (1, 1), "goal_1": (2, 2)}
        )
        dones = self.env._check_done_condition(done_mode=0)
        self.assertTrue(dones["__all__"])

        # ケース2: 2人が同じゴールに乗っており、もう1つのゴールが空いている (False)
        self.set_positions(
            {"agent_0": (1, 1), "agent_1": (1, 1)},
            {"goal_0": (1, 1), "goal_1": (2, 2)}
        )
        dones = self.env._check_done_condition(done_mode=0)
        self.assertFalse(dones["__all__"])

    def test_mode_1_any_goal_reached(self):
        """Mode 1: 誰か一人がどこかのゴールに到達すれば終了"""
        # ケース1: agent_0だけがゴール (True)
        self.set_positions(
            {"agent_0": (1, 1), "agent_1": (4, 4)},
            {"goal_0": (1, 1), "goal_1": (2, 2)}
        )
        dones = self.env._check_done_condition(done_mode=1)
        self.assertTrue(dones["__all__"])
        self.assertTrue(dones["agent_0"])
        self.assertFalse(dones["agent_1"])

    def test_mode_2_all_agents_in_goals(self):
        """Mode 2: 全てのエージェントがいずれかのゴールにいる必要がある"""
        # ケース1: agent_0はゴール、agent_1は道半ば (False)
        self.set_positions(
            {"agent_0": (1, 1), "agent_1": (3, 3)},
            {"goal_0": (1, 1), "goal_1": (2, 2)}
        )
        dones = self.env._check_done_condition(done_mode=2)
        self.assertFalse(dones["__all__"])

        # ケース2: 全員がゴール（同じゴールでも可） (True)
        self.set_positions(
            {"agent_0": (2, 2), "agent_1": (2, 2)},
            {"goal_0": (1, 1), "goal_1": (2, 2)}
        )
        dones = self.env._check_done_condition(done_mode=2)
        self.assertTrue(dones["__all__"])

    def test_individual_dones_format(self):
        """各エージェントの個別doneが正しく含まれているか"""
        self.set_positions(
            {"agent_0": (1, 1), "agent_1": (4, 4)},
            {"goal_0": (1, 1), "goal_1": (2, 2)}
        )
        dones = self.env._check_done_condition(done_mode=1)
        
        # 必要なキーが揃っているか
        self.assertIn("agent_0", dones)
        self.assertIn("agent_1", dones)
        self.assertIn("__all__", dones)
        
        # 個別判定の正確性
        self.assertEqual(dones["agent_0"], True)
        self.assertEqual(dones["agent_1"], False)