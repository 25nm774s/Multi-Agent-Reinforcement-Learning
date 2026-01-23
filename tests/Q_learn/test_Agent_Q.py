import unittest
import numpy as np
import os
import csv
from typing import Tuple, Dict, List, Any

# QTableクラスとSimpleArgsクラスが定義されている必要があります。
# ここでは、ノートブックの以前のセルで定義されたQTableとSimpleArgsクラスを使用することを想定しています。
# Constants
RED = '\033[91m'
GREEN = '\033[92m'
RESET = '\033[0m'

QState = Tuple[int, ...]
PosType = Tuple[int, int]
GlobalState = Dict[str, Any]
QTableType = Dict[QState, List[float]]

from src.Q_learn.Agent_Q import Agent
from src.Q_learn.QTable import QTable

# MultiAgent_Q クラスを実行するための引数を定義する簡単なArgsクラス
class SimpleArgs:
    def __init__(self, grid_size=5, agents_number=2, goals_number=1, reward_mode=2,
                 render_mode=False, window_width=500, window_height=500,
                 episode_number=10, max_timestep=100, buffer_size=10000,
                 batch_size=32, decay_epsilon=1000, load_model=0, mask=False, device="cpu",
                 learning_rate=0.1, discount_factor=0.99, epsilon=1.0, min_epsilon=0.01, max_epsilon=1.0, neighbor_distance=256):
        self.grid_size = grid_size
        self.agents_number = agents_number
        self.goals_number = goals_number
        self.reward_mode = reward_mode
        self.render_mode = render_mode
        self.window_width = window_width
        self.window_height = window_height
        self.episode_number = episode_number
        self.max_timestep = max_timestep
        self.buffer_size = buffer_size
        self.batch_size = batch_size
        self.decay_epsilon = decay_epsilon
        self.load_model = load_model
        self.mask = mask
        self.device = device
        self.learning_rate = learning_rate
        self.discount_factor = discount_factor
        self.epsilon = epsilon
        self.min_epsilon = min_epsilon
        self.max_epsilon = max_epsilon
        self.neighbor_distance = neighbor_distance

# Mock Saver class (used for context, not directly for Agent tests)
class MockSaver:
    def __init__(self, save_dir, grid_size):
        print(f"MockSaver initialized with save_dir: {save_dir}")
        self.save_dir = save_dir
        self.grid_size = grid_size
        os.makedirs(save_dir, exist_ok=True)
        self.episode_batch_summary_path = os.path.join(self.save_dir, "aggregated_episode_metrics_100.csv")
        self.visited_coordinates_path = os.path.join(self.save_dir, "visited_coordinates.npy")

        if not os.path.exists(self.episode_batch_summary_path):
            with open(self.episode_batch_summary_path, "w", newline='') as f:
                csv.writer(f).writerow(["episode_group_start", "episode_group_end", "avg_time_step_100", "avg_reward_100", "avg_loss_100", "done_rate"])

        self.visited_count_grid = np.zeros((self.grid_size, self.grid_size), dtype=int)
        self.episode_data_buffer = []
        self.episode_data_counter = 0
        self.visited_updates_counter = 0

    def log_agent_states(self, agent_id, x, y):
        # print(f"MockSaver: log_agent_states called for agent {agent_id} at ({x}, {y})")
        self.visited_count_grid[y, x] += 1
        self.visited_updates_counter += 1

    def log_episode_data(self, episode: int, time_step: int, reward: float, loss: float, done: bool):
        self.episode_data_buffer.append({'episode': episode, 'time_step': time_step, 'reward': reward, 'loss': loss, 'done':int(done)})
        self.episode_data_counter += 1

    def save_visited_coordinates(self):
        if self.visited_count_grid is not None:
            np.save(self.visited_coordinates_path, self.visited_count_grid)
            # print(f"MockSaver: Visited count grid saved to {self.visited_coordinates_path}")

# Mock PlotResults class (used for context, not directly for Agent tests)
class MockPlotResults:
    def __init__(self, save_dir):
        print(f"MockPlotResults initialized with save_dir: {save_dir}")
        self.save_dir = save_dir

    def draw(self):
        print("MockPlotResults: draw called (plotting scores)")

    def draw_heatmap(self):
        print("MockPlotResults: draw_heatmap called (plotting agent states heatmap)")

# Mock GridWorld class (used for context, not directly for Agent tests)
class MockGridWorld:
    def __init__(self, args):
        print("MockGridWorld initialized")
        self.grid_size = args.grid_size
        self.agents_num = args.agents_number
        self.goals_num = args.goals_number
        self.reward_mode = args.reward_mode
        self.render_mode = args.render_mode

        self.goals = [(0, 0)] * self.goals_num
        self.agents = [(gs-1, gs-1) for gs in [args.grid_size] * self.agents_num]

    def reset(self):
        print("MockGridWorld: reset called")
        return ({'agent_0': {'self': (3,3), 'all_goals': [(1,1)], 'others': {}}})

    def step(self, actions):
        print("MockGridWorld: step called")
        next_global_state = ({'agent_0': {'self': (3,2), 'all_goals': [(1,1)], 'others': {}}})
        reward = {'agent_0': 1.0}
        done = {'agent_0': False, '__all__': False}
        info = {}
        return next_global_state, reward, done, info

    def get_goal_positions(self):
        return {f'goal_{i}': pos for i, pos in enumerate(self.goals)}

    def get_agent_positions(self):
        return {f'agent_{i}': pos for i, pos in enumerate(self.agents)}


class TestAgent(unittest.TestCase):

    def setUp(self):
        """
        各テストメソッドの実行前に呼ばれるセットアップメソッド.
        テスト用のAgentインスタンスとArgsオブジェクトを作成します.
        """
        class TestArgs:
            def __init__(self, grid_size=5, agents_number=2, goals_number=1, reward_mode=2,
                         learning_rate=0.1, discount_factor=0.99, epsilon=1.0, min_epsilon=0.01, max_epsilon = 1.0, neighbor_distance=256):
                self.grid_size = grid_size
                self.agents_number = agents_number
                self.goals_number = goals_number
                self.reward_mode = reward_mode
                self.learning_rate = learning_rate
                self.discount_factor = discount_factor
                self.epsilon = epsilon
                self.min_epsilon = min_epsilon
                self.max_epsilon = max_epsilon
                self.neighbor_distance = neighbor_distance

        self.args = TestArgs()
        self.agent_id = 0 # テスト対象のエージェントID
        self.agent_instance = Agent(self.args, self.agent_id)


    def test_agent_initialization(self):
        """
        Agentクラスの初期化が正しく行われるかテスト.
        """
        self.assertEqual(self.agent_instance.id, self.agent_id)
        self.assertEqual(self.agent_instance.action_size, 5) # 固定値
        self.assertAlmostEqual(self.agent_instance.epsilon, self.args.epsilon)
        self.assertAlmostEqual(self.agent_instance.min_epsilon, self.args.min_epsilon)
        self.assertAlmostEqual(self.agent_instance.max_epsilon, self.args.max_epsilon)

        # QTableインスタンスが正しく生成され、保持されているか確認
        self.assertIsInstance(self.agent_instance.q_table, QTable)
        self.assertEqual(self.agent_instance.q_table.action_size, self.agent_instance.action_size)
        self.assertAlmostEqual(self.agent_instance.q_table.lr, self.args.learning_rate)
        self.assertAlmostEqual(self.agent_instance.q_table.gamma, self.args.discount_factor)


    def test_agent_get_q_state(self):
        """
        Agentの_get_q_stateメソッドが正しい状態タプルを生成するかテスト (様々なケース).
        """
        # ケース1: ゴール1個、エージェント2個、対象エージェントID=0
        args1 = SimpleArgs(goals_number=1, agents_number=2, neighbor_distance=0) # neighbor_distance=0で自分以外は-1,-1
        agent_id1 = 0
        agent_instance1 = Agent(args1, agent_id1)
        global_state1: GlobalState = {
            'agent_0': {'self': (3, 3), 'all_goals': [(1, 1)], 'others': {'agent_1': (-1, -1)}},
            'agent_1': {'self': (4, 4), 'all_goals': [(1, 1)], 'others': {'agent_0': (-1, -1)}}
        }
        expected_state1: QState = (1, 1, 3, 3, -1, -1) # goals, self, other(masked)
        q_state1 = agent_instance1._get_q_state(global_state1)
        self.assertEqual(q_state1, expected_state1, f"Case 1 Failed: Expected {expected_state1}, Got {q_state1}")

        # ケース2: ゴール1個、エージェント2個、対象エージェントID=1
        args2 = SimpleArgs(goals_number=1, agents_number=2, neighbor_distance=0)
        agent_id2 = 1
        agent_instance2 = Agent(args2, agent_id2)
        global_state2: GlobalState = {
            'agent_0': {'self': (3, 3), 'all_goals': [(1, 1)], 'others': {'agent_1': (-1, -1)}},
            'agent_1': {'self': (4, 4), 'all_goals': [(1, 1)], 'others': {'agent_0': (-1, -1)}}
        }
        expected_state2: QState = (1, 1, 4, 4, -1, -1) # goals, self, other(masked)
        q_state2 = agent_instance2._get_q_state(global_state2)
        self.assertEqual(q_state2, expected_state2, f"Case 2 Failed: Expected {expected_state2}, Got {q_state2}")

        # ケース3: ゴール2個、エージェント1個、対象エージェントID=0
        args3 = SimpleArgs(goals_number=2, agents_number=1, neighbor_distance=256)
        agent_id3 = 0
        agent_instance3 = Agent(args3, agent_id3)
        global_state3: GlobalState = {
            'agent_0': {'self': (2, 2), 'all_goals': [(5, 5), (6, 6)], 'others': {}} # No other agents
        }
        expected_state3: QState = (5, 5, 6, 6, 2, 2) # goals, self
        q_state3 = agent_instance3._get_q_state(global_state3)
        self.assertEqual(q_state3, expected_state3, f"Case 3 Failed: Expected {expected_state3}, Got {q_state3}")

        # ケース4: ゴール2個、エージェント3個、対象エージェントID=1 (neighbors within range)
        args4 = SimpleArgs(goals_number=2, agents_number=3, neighbor_distance=256) # All visible
        agent_id4 = 1
        agent_instance4 = Agent(args4, agent_id4)
        global_state4: GlobalState = {
            'agent_0': {'self': (1, 1), 'all_goals': [(10, 10), (11, 11)], 'others': {'agent_1': (2, 2), 'agent_2': (3, 3)}},
            'agent_1': {'self': (2, 2), 'all_goals': [(10, 10), (11, 11)], 'others': {'agent_0': (1, 1), 'agent_2': (3, 3)}},
            'agent_2': {'self': (3, 3), 'all_goals': [(10, 10), (11, 11)], 'others': {'agent_0': (1, 1), 'agent_1': (2, 2)}}
        }
        # For agent_1, observation is goals, self, other_0, other_2
        expected_state4: QState = (10, 10, 11, 11, 2, 2, 1, 1, 3, 3)
        q_state4 = agent_instance4._get_q_state(global_state4)
        self.assertEqual(q_state4, expected_state4, f"Case 4 Failed: Expected {expected_state4}, Got {q_state4}")

        # ケース5: ゴール0個, エージェント1個, 対象エージェントID=0
        args5 = SimpleArgs(goals_number=0, agents_number=1, neighbor_distance=256)
        agent_id5 = 0
        agent_instance5 = Agent(args5, agent_id5)
        global_state5: GlobalState = {
            'agent_0': {'self': (7, 7), 'all_goals': [], 'others': {}}
        }
        expected_state5: QState = (7, 7)
        q_state5 = agent_instance5._get_q_state(global_state5)
        self.assertEqual(q_state5, expected_state5, f"Case 5 Failed: Expected {expected_state5}, Got {q_state5}")

        # ケース6: global_stateにagent_idが存在しない場合 (KeyError)
        args6 = SimpleArgs(goals_number=1, agents_number=1)
        agent_id6 = 0
        agent_instance6 = Agent(args6, agent_id6)
        global_state6: GlobalState = {
            'agent_X': {'self': (1, 1), 'all_goals': [(0, 0)], 'others': {}}
        }
        with self.assertRaises(KeyError, msg="Case 6 Failed: Should raise KeyError for invalid agent_id in global_state"):
             agent_instance6._get_q_state(global_state6)

    def test_agent_get_action_exploration(self):
        """
        Agentのget_actionメソッドでepsilon=1.0 の場合、ランダムな行動が選択されるかテスト.
        """
        self.agent_instance.epsilon = 1.0
        global_state: GlobalState = {'agent_0': {'self': (3, 3), 'all_goals': [(1, 1)], 'others': {}}}
        actions = [self.agent_instance.get_action(global_state) for _ in range(100)] # 100回試行
        # 少なくともいくつかの異なる行動が出現することを確認 (完全にランダムなので厳密なテストは難しい)
        self.assertTrue(len(set(actions)) > 1)
        # 選択された行動が有効な行動範囲内であるか確認
        for action in actions:
            self.assertTrue(0 <= action < self.agent_instance.action_size)


    def test_agent_get_action_exploitation(self):
        """
        Agentのget_actionメソッドでepsilon=0.0 の場合、Q値に基づいて最適な行動が選択されるかテスト.
        """
        self.agent_instance.epsilon = 0.0
        global_state: GlobalState = {'agent_0': {'self': (3, 3), 'all_goals': [(1, 1)], 'others': {}}}
        q_state = self.agent_instance._get_q_state(global_state)

        # Agentが持つQTableインスタンスの内部辞書に直接アクセスしてQ値を設定
        # 例: 行動2(LEFT)のQ値を最も高く設定
        self.agent_instance.q_table.q_table[q_state] = [0.0, 0.0, 1.0, 0.0, 0.0]

        # epsilon=0なので、行動2(LEFT)が選択されるはず
        selected_action = self.agent_instance.get_action(global_state)
        self.assertEqual(selected_action, 2, " exploitation failed when single max Q-value")

        # 最大Q値が複数ある場合、ランダムに選択されるかテスト
        self.agent_instance.q_table.q_table[q_state] = [0.0, 1.0, 1.0, 0.0, 0.0] # 行動1と2が最大

        actions = [self.agent_instance.get_action(global_state) for _ in range(100)] # 100回試行
        # 選択された行動が1または2のみであることを確認
        for action in actions:
            self.assertTrue(action in [1, 2], f"exploitation failed with multiple max Q-values: {action}")
        # 少なくとも行動1と2の両方が出現することを確認
        self.assertTrue(1 in actions and 2 in actions, "exploitation did not explore all max Q-values")


    def test_agent_decay_epsilon_power(self):
        """
        Agentのdecay_epsilon_powerメソッドによるε減衰が正しく行われるかテスト.
        """
        initial_epsilon = self.agent_instance.epsilon # デフォルトは1.0
        min_epsilon = self.agent_instance.min_epsilon # デフォルトは0.01

        self.agent_instance.epsilon = 1.0 # 明示的に初期化

        # 1ステップ減衰
        self.agent_instance.decay_epsilon_power()
        expected_epsilon = 1.0 * 0.9999
        self.assertAlmostEqual(self.agent_instance.epsilon, max(expected_epsilon, min_epsilon))

        # 複数ステップ減衰
        self.agent_instance.epsilon = 1.0 # リセット
        for _ in range(100):
            self.agent_instance.decay_epsilon_power()
        expected_epsilon = 1.0 * (0.9999 ** 100)
        self.assertAlmostEqual(self.agent_instance.epsilon, max(expected_epsilon, min_epsilon))

        # min_epsilonを下回らないことを確認
        self.agent_instance.epsilon = 0.005 # min_epsilon (0.01)より小さい値に設定
        self.agent_instance.decay_epsilon_power()
        self.assertAlmostEqual(self.agent_instance.epsilon, min_epsilon)

        self.agent_instance.epsilon = min_epsilon # min_epsilonに設定
        self.agent_instance.decay_epsilon_power()
        self.assertAlmostEqual(self.agent_instance.epsilon, min_epsilon)

    def test_agent_learn(self):
        """
        AgentのlearnメソッドがQTableのlearnメソッドを正しく呼び出すかテスト.
        """
        # global_state (observation) は Dict[str, dict] の形式を想定
        global_state: GlobalState = {'agent_0': {'self': (3, 3), 'all_goals': [(1, 1)], 'others': {}}}
        action = 2 # 例: LEFT
        reward = 1.0
        next_global_state: GlobalState = {'agent_0': {'self': (3, 2), 'all_goals': [(1, 1)], 'others': {}}}
        done = False

        # AgentがQTableに渡すはずのQStateを事前に計算
        current_q_state = self.agent_instance._get_q_state(global_state)
        next_q_state = self.agent_instance._get_q_state(next_global_state)

        # Mock QTableを作成して、Agentが持つQTableインスタンスを置き換える
        class MockQTableForAgentLearnTest:
            def __init__(self, action_size, learning_rate, discount_factor):
                self.action_size = action_size
                self.lr = learning_rate
                self.gamma = discount_factor
                self._learn_called_with = None
                self._initial_q_value = 0.0
                self.q_table = {}

            def learn(self, state, action, reward, next_state, done) -> float:
                self._learn_called_with = (state, action, reward, next_state, done)
                return 0.123 # ダミーのTD誤差を返す

            def get_q_values(self, state) -> List[float]:
                 if state not in self.q_table:
                      self.q_table[state] = [self._initial_q_value] * self.action_size
                 return self.q_table[state]

        mock_q_table = MockQTableForAgentLearnTest(
            action_size=self.agent_instance.action_size,
            learning_rate=self.agent_instance.q_table.lr,
            discount_factor=self.agent_instance.q_table.gamma
        )

        # AgentインスタンスのQTableをモックに置き換え
        self.agent_instance.q_table = mock_q_table # type: ignore

        # Agentのobserveメソッドで経験をストア
        self.agent_instance.observe(global_state, action, reward, next_global_state, done)

        # Agentのlearnメソッドを実行
        returned_td_delta = self.agent_instance.learn()

        # QTableのlearnメソッドが正しい引数で呼ばれたか確認
        self.assertIsNotNone(mock_q_table._learn_called_with)
        called_state, called_action, called_reward, called_next_state, called_done = mock_q_table._learn_called_with # type: ignore

        self.assertEqual(called_state, current_q_state)
        self.assertEqual(called_action, action)
        self.assertEqual(called_reward, reward)
        self.assertEqual(called_next_state, next_q_state)
        self.assertEqual(called_done, done)

        # AgentのlearnメソッドがQTableから返されたTD誤差をそのまま返しているか確認
        self.assertAlmostEqual(returned_td_delta, 0.123)


    def test_agent_get_set_weights(self):
        """
        Agentのget_weightsとset_weightsメソッドがQTableのデータを正しく取得・設定するかテスト.
        """
        initial_q_table: QTableType = {(1, 2, 3, 4): [0.1, 0.2, 0.3, 0.4, 0.5]}

        # set_weightsでQTableを設定
        self.agent_instance.set_weights(initial_q_table)

        # get_weightsでQTableを取得
        retrieved_q_table = self.agent_instance.get_weights()

        # 設定したQTableと取得したQTableが一致することを確認
        self.assertEqual(retrieved_q_table, initial_q_table)

        # 別のQTableを設定して再度確認
        new_q_table: QTableType = {(5, 6, 7, 8): [1.0, 2.0, 3.0, 4.0, 5.0], (1,1,2,2):[0.1,0.1,0.1,0.1,0.1]}
        self.agent_instance.set_weights(new_q_table)
        retrieved_new_q_table = self.agent_instance.get_weights()
        self.assertEqual(retrieved_new_q_table, new_q_table)


    def test_agent_get_q_table_size(self):
        """
        Agentのget_q_table_sizeメソッドがQTableのget_q_table_sizeメソッドを正しく呼び出すかテスト.
        """
        class MockQTableForAgentSizeTest:
            def __init__(self):
                 self._get_size_called = False
                 self._dummy_size = 123
                 self.action_size = 5 # Required by Agent initialization logic
                 self.lr = 0.1
                 self.gamma = 0.99

            def get_q_table_size(self) -> int:
                self._get_size_called = True
                return self._dummy_size

            def get_q_values(self, state) -> List[float]:
                return [0.0] * self.action_size

            def learn(self, state, action, reward, next_state, done) -> float:
                return 0.0

            def get_Qtable(self) -> QTableType:
                return {}
            def set_Qtable(self, q_table: QTableType) -> None:
                pass

        mock_q_table = MockQTableForAgentSizeTest()

        self.agent_instance.q_table = mock_q_table # type: ignore

        size = self.agent_instance.get_q_table_size()

        self.assertTrue(mock_q_table._get_size_called)
        self.assertEqual(size, mock_q_table._dummy_size)

    def test_agent_get_all_q_values(self):
        """
        Agentのget_all_q_valuesメソッドがQTableのget_q_valuesメソッドを正しく呼び出すかテスト.
        """
        global_state: GlobalState = {'agent_0': {'self': (3, 3), 'all_goals': [(1, 1)], 'others': {}}}
        q_state = self.agent_instance._get_q_state(global_state)

        expected_q_values = [0.1, 0.2, 0.3, 0.4, 0.5]
        self.agent_instance.q_table.q_table[q_state] = expected_q_values

        retrieved_q_values = self.agent_instance.get_all_q_values(global_state)
        self.assertEqual(retrieved_q_values, expected_q_values)

        # 存在しない状態の場合、初期化されたQ値が返されることを確認
        non_existent_global_state: GlobalState = {'agent_0': {'self': (9, 9), 'all_goals': [(8, 8)], 'others': {}}}
        retrieved_empty_q_values = self.agent_instance.get_all_q_values(non_existent_global_state)
        self.assertEqual(retrieved_empty_q_values, [0.0] * self.agent_instance.action_size)
