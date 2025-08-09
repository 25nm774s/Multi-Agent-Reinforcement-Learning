# Assume QTable class, Agent class, SimpleArgs class, and Mock classes are already defined
# Assume TestQTable class is already defined in the previous cell

import unittest
import numpy as np
import os
import csv
from typing import Tuple

from src.Q_learn.QTable import QTable
from src.Q_learn.Agent_Q import Agent

# QTableクラスとSimpleArgsクラスが定義されている必要があります。
# ここでは、ノートブックの以前のセルで定義されたQTableとSimpleArgsクラスを使用することを想定しています。
# Constants
RED = '\033[91m'
GREEN = '\033[92m'
RESET = '\033[0m'

QState = Tuple[int, ...]

# MultiAgent_Q クラスを実行するための引数を定義する簡単なArgsクラス
class SimpleArgs:
    def __init__(self, grid_size=5, agents_number=2, goals_number=1, reward_mode=2,
                 render_mode=False, window_width=500, window_height=500,
                 episode_number=10, max_timestep=100, buffer_size=10000,
                 batch_size=32, decay_epsilon=1000, load_model=0, mask=False, device="cpu",
                 learning_rate=0.1, discount_factor=0.99, epsilon=1.0, min_epsilon=0.01, max_epsilon=1.0):
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
        # dir_path は MultiAgent_Q の中で設定されるためここでは不要

# Mock ReplayBuffer class
class MockReplayBuffer:
    def __init__(self, buffer_size, batch_size, device):
        print("MockReplayBuffer initialized")
        self.buffer = []
        self.buffer_size = buffer_size
        self.batch_size = batch_size
        self.device = device

    def add(self, state, action, reward, next_state, done):
        self.buffer.append((state, action, reward, next_state, done))
        if len(self.buffer) > self.buffer_size:
            self.buffer.pop(0) # Simple FIFO

    def get_batch(self):
        if len(self.buffer) < self.batch_size:
            return [], [], [], [], []
        indices = np.random.choice(len(self.buffer), self.batch_size, replace=False)
        batch = [self.buffer[i] for i in range(indices)]
        states, actions, rewards, next_states, dones = zip(*batch)
        return states, actions, rewards, next_states, dones

    def __len__(self):
        return len(self.buffer)


# Mock Linear class (for Q-table approximation)
class MockLinear:
    def __init__(self, args, action_size):
        print("MockLinear initialized")
        self.action_size = action_size
        self.q_table = {} # Using a dictionary to mock Q-table

    def _get_q_state_key(self, agents_pos, agent_pos, action=None):
        key_list = []
        for pos in agents_pos:
            key_list.extend(pos)
        key_list.extend(agent_pos)
        if action is not None:
             key_list.append(action)
        return tuple(key_list)


    def getQ(self, agents_pos, agent_pos, action):
        state_key = self._get_q_state_key(agents_pos, agent_pos)
        if state_key not in self.q_table:
            self.q_table[state_key] = [0.0] * self.action_size # Initialize Q-values for new state
        return self.q_table[state_key][action]


    def update(self, i, states, action, reward, next_state, done, step):
        goals_num = len(states) - (len(states) // 2)
        agents_pos_list = [list(pos) for pos in states[goals_num:]]
        agent_pos = agents_pos_list[i]
        other_agents_pos = [pos for j, pos in enumerate(agents_pos_list) if j != i]

        current_state_key = self._get_q_state_key(other_agents_pos, agent_pos)

        if current_state_key not in self.q_table:
             self.q_table[current_state_key] = [0.0] * self.action_size

        td_error = 0.1
        return abs(td_error)


# Mock Saver class
class MockSaver:
    def __init__(self, save_dir):
        print(f"MockSaver initialized with save_dir: {save_dir}")
        self.save_dir = save_dir
        os.makedirs(save_dir, exist_ok=True)
        self.scores_path = os.path.join(save_dir, "scores.csv")
        self.agents_states_path = os.path.join(save_dir, "agents_states.csv")
        if not os.path.exists(self.scores_path):
            with open(self.scores_path, 'w', newline='') as f:
                csv.writer(f).writerow(["episode", "time_step", "reward", "loss"])
        # log_agent_statesメソッド内でファイルを作成するため、ここではヘッダーの書き込みは不要
        # if not os.path.exists(self.agents_states_path):
        #     with open(self.agents_states_path, 'a', newline='') as f:
        #         csv.writer(f).writerow(["episode", "time_step", "agent_id", "agent_state"])

    def log_scores(self, episode, time_step, reward, loss):
        with open(self.scores_path, 'a', newline='') as f:
            csv.writer(f).writerow([episode, time_step, reward, loss])

    def log_agent_states(self, episode, time_step, agent_id, agent_state):
        # ファイルが存在しない場合にヘッダーを書き込む
        if not os.path.exists(self.agents_states_path):
             with open(self.agents_states_path, 'w', newline='') as f:
                 csv.writer(f).writerow(["episode", "time_step", "agent_id", "agent_state"])

        if isinstance(agent_state, (list, tuple, np.ndarray)):
            state_str = '_'.join(map(str, agent_state))
        else:
            state_str = str(agent_state)
        with open(self.agents_states_path, 'a', newline='') as f:
            csv.writer(f).writerow([episode, time_step, agent_id, state_str])

    def save_q_table(self, agents, mask):
        print("MockSaver: save_q_table called")
        dummy_save_path = os.path.join(self.save_dir, "mock_q_table_save.txt")
        with open(dummy_save_path, 'w') as f:
            f.write("This is a mock Q-table save file.\n")
            f.write(f"Mask setting: {mask}\n")
            for i, agent in enumerate(agents):
                 agent.save_q_table(os.path.join(self.save_dir, f"agent_{i}_q_table.pkl"))


# Mock PlotResults class
class MockPlotResults:
    def __init__(self, save_dir):
        print(f"MockPlotResults initialized with save_dir: {save_dir}")
        self.save_dir = save_dir
        self.scores_path = os.path.join(save_dir, "scores.csv")
        self.agents_states_path = os.path.join(save_dir, "agents_states.csv")

    def draw(self):
        print("MockPlotResults: draw called (plotting scores)")
        print(f"Mock plotting scores from {self.scores_path}")

    def draw_heatmap(self, grid_size):
        print(f"MockPlotResults: draw_heatmap called with grid_size {grid_size} (plotting agent states heatmap)")
        print(f"Mock plotting heatmap from {self.agents_states_path} for grid size {grid_size}")

# Mock GridWorld class
class MockGridWorld:
    def __init__(self, args):
        print("MockGridWorld initialized")
        self.grid_size = args.grid_size
        self.agents_num = args.agents_number
        self.goals_num = args.goals_number
        self.reward_mode = args.reward_mode
        self.render_mode = args.render_mode

        # モック用のゴールとエージェントの位置を適当に設定
        self.goals = [(0, 0)] * self.goals_num
        self.agents = [(gs-1, gs-1) for gs in [args.grid_size] * self.agents_num]

        if self.render_mode:
            # MockGridRendererが存在すると仮定 (ここでは定義しない)
            print("Mock renderer would be initialized here")
        else:
            self.renderer = None

    def reset(self):
        """ モック用のリセットメソッド """
        print("MockGridWorld: reset called")
        # ダミーの初期状態を返す (ゴール + エージェントの位置タプル)
        return tuple(self.goals + self.agents)

    def step(self, global_state, actions):
        """ モック用のステップメソッド """
        print("MockGridWorld: step called")
        # ダミーの次の状態、報酬、doneフラグを返す
        next_global_state = global_state # 状態は変化しないとする
        reward = 1.0 # ダミー報酬
        done = False # 常に未完了とする
        return next_global_state, reward, done

    def update_positions(self, global_state, actions):
         """ モック用の位置更新メソッド """
         print("MockGridWorld: update_positions called")
         return global_state # 位置は変化しないとする

    def generate_unique_positions(self, num_positions, object_positions, grid_size):
         """ モック用の位置生成メソッド """
         print("MockGridWorld: generate_unique_positions called")
         # ダミーの位置を返す
         return [(i, i) for i in range(num_positions)]

    def _generate_fixed_goals(self):
         """ モック用のゴール生成メソッド """
         print("MockGridWorld: _generate_fixed_goals called")
         # ダミーのゴールを生成
         self.goals = [(i, 0) for i in range(self.goals_num)]

    def get_goal_positions(self,global_state):
        """ モック用のゴール位置取得メソッド """
        print("MockGridWorld: get_goal_positions called")
        return list(global_state[:self.goals_num])

    def get_agent_positions(self,global_state):
        """ モック用のエージェント位置取得メソッド """
        print("MockGridWorld: get_agent_positions called")
        return list(global_state[self.goals_num:])


class TestAgent(unittest.TestCase):

    def setUp(self):
        """
        各テストメソッドの実行前に呼ばれるセットアップメソッド.
        テスト用のAgentインスタンスとArgsオブジェクトを作成します.
        """
        class TestArgs: # テスト用の簡易Argsクラス
            def __init__(self, grid_size=5, agents_number=2, goals_number=1, reward_mode=2,
                         render_mode=False, window_width=500, window_height=500,
                         episode_number=10, max_timestep=100, buffer_size=10000,
                         batch_size=32, load_model=0, mask=False, device="cpu", # Removed decay_epsilon
                         learning_rate=0.1, discount_factor=0.99, epsilon=1.0, min_epsilon=0.01, max_epsilon = 1.0):
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
                # self.decay_epsilon = decay_epsilon # Removed
                self.load_model = load_model
                self.mask = mask
                self.device = device
                self.learning_rate = learning_rate
                self.discount_factor = discount_factor
                self.epsilon = epsilon
                self.min_epsilon = min_epsilon
                self.max_epsilon = max_epsilon
                self.dir_path = "test_agent_models" # テスト用に一時的なディレクトリを指定

        self.args = TestArgs()
        self.agent_id = 0 # テスト対象のエージェントID
        self.agent_instance = Agent(self.args, self.agent_id)

        # テスト用のディレクトリを作成
        os.makedirs(self.args.dir_path, exist_ok=True)
        self.test_q_table_path = os.path.join(self.args.dir_path, f"agent_{self.agent_id}_q_table.pkl")


    def tearDown(self):
        """
        各テストメソッドの実行後に呼ばれるクリーンアップメソッド.
        テストで作成したファイルやディレクトリを削除します.
        """
        # Agentが保存するQテーブルファイルを削除
        if os.path.exists(self.test_q_table_path):
            os.remove(self.test_q_table_path)
        # ディレクトリが空であれば削除
        if os.path.exists(self.args.dir_path) and not os.listdir(self.args.dir_path):
             os.rmdir(self.args.dir_path)


    def test_agent_initialization(self):
        """
        Agentクラスの初期化が正しく行われるかテスト.
        """
        self.assertEqual(self.agent_instance.agent_id, self.agent_id)
        self.assertEqual(self.agent_instance.grid_size, self.args.grid_size)
        self.assertEqual(self.agent_instance.goals_num, self.args.goals_number)
        self.assertEqual(self.agent_instance.action_size, 5) # 固定値
        self.assertAlmostEqual(self.agent_instance.epsilon, self.args.epsilon)
        self.assertAlmostEqual(self.agent_instance.min_epsilon, self.args.min_epsilon)
        self.assertAlmostEqual(self.agent_instance.max_epsilon, self.args.max_epsilon)
        self.assertEqual(self.agent_instance.model_path, self.test_q_table_path)

        # QTableインスタンスが正しく生成され、保持されているか確認
        self.assertIsInstance(self.agent_instance.q_table, QTable)
        self.assertEqual(self.agent_instance.q_table.action_size, self.agent_instance.action_size)
        self.assertAlmostEqual(self.agent_instance.q_table.lr, self.args.learning_rate)
        self.assertAlmostEqual(self.agent_instance.q_table.gamma, self.args.discount_factor)
        # Removed assertion for q_table.load_model as it's no longer an attribute of QTable
        # self.assertFalse(self.agent_instance.q_table.load_model) # デフォルトはload_model=0なのでFalse
        self.assertEqual(self.agent_instance.q_table.model_path, self.test_q_table_path)


    def test_agent_get_q_state(self):
        """
        Agentの_get_q_stateメソッドが正しい状態タプルを生成するかテスト (様々なケース).
        """
        # ケース1: ゴール1個、エージェント2個、対象エージェントID=0
        args1 = SimpleArgs(goals_number=1, agents_number=2) # Agentテスト用のSimpleArgsを使用
        global_state1 = ((1, 1), (3, 3), (4, 4)) # (goal1_pos, agent0_pos, agent1_pos)
        agent_id1 = 0
        expected_state1: QState = (1, 1, 3, 3)
        agent_instance1 = Agent(args1, agent_id1)
        q_state1 = agent_instance1._get_q_state(global_state1)
        self.assertEqual(q_state1, expected_state1, "Case 1 Failed")

        # ケース2: ゴール1個、エージェント2個、対象エージェントID=1
        args2 = SimpleArgs(goals_number=1, agents_number=2)
        global_state2 = ((1, 1), (3, 3), (4, 4)) # (goal1_pos, agent0_pos, agent1_pos)
        agent_id2 = 1
        expected_state2: QState = (1, 1, 4, 4)
        agent_instance2 = Agent(args2, agent_id2)
        q_state2 = agent_instance2._get_q_state(global_state2)
        self.assertEqual(q_state2, expected_state2, "Case 2 Failed")

        # ケース3: ゴール2個、エージェント1個、対象エージェントID=0
        args3 = SimpleArgs(goals_number=2, agents_number=1)
        global_state3 = ((5, 5), (6, 6), (2, 2)) # (goal1_pos, goal2_pos, agent0_pos)
        agent_id3 = 0
        expected_state3: QState = (5, 5, 6, 6, 2, 2)
        agent_instance3 = Agent(args3, agent_id3)
        q_state3 = agent_instance3._get_q_state(global_state3)
        self.assertEqual(q_state3, expected_state3, "Case 3 Failed")

        # ケース4: ゴール2個、エージェント3個、対象エージェントID=1
        args4 = SimpleArgs(goals_number=2, agents_number=3)
        global_state4 = ((10, 10), (11, 11), (1, 1), (2, 2), (3, 3)) # (g1, g2, a0, a1, a2)
        agent_id4 = 1
        expected_state4: QState = (10, 10, 11, 11, 2, 2)
        agent_instance4 = Agent(args4, agent_id4)
        q_state4 = agent_instance4._get_q_state(global_state4)
        self.assertEqual(q_state4, expected_state4, "Case 4 Failed")

        # ケース5: ゴール0個, エージェント1個, 対象エージェントID=0
        args5 = SimpleArgs(goals_number=0, agents_number=1)
        global_state5 = ((7, 7),) # (agent0_pos,)
        agent_id5 = 0
        expected_state5: QState = (7, 7)
        agent_instance5 = Agent(args5, agent_id5)
        q_state5 = agent_instance5._get_q_state(global_state5)
        self.assertEqual(q_state5, expected_state5, "Case 5 Failed")

        # エージェントIDがagents_number以上の場合のテスト (IndexErrorになるはず)
        args7 = SimpleArgs(goals_number=1, agents_number=2)
        global_state7 = ((1, 1), (3, 3), (4, 4))
        agent_id7 = 2 # 存在しないエージェントID
        agent_instance7 = Agent(args7, agent_id7)
        with self.assertRaises(IndexError, msg="Case 7 Failed: Should raise IndexError for invalid agent_id"):
             agent_instance7._get_q_state(global_state7)


    def test_agent_get_action_exploration(self):
        """
        Agentのget_actionメソッドでepsilon=1.0 の場合、ランダムな行動が選択されるかテスト.
        """
        self.agent_instance.epsilon = 1.0
        global_state = ((1, 1), (3, 3), (4, 4)) # goals=1, agents=2
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
        global_state = ((1, 1), (3, 3), (4, 4)) # goals=1, agents=2
        q_state = self.agent_instance._get_q_state(global_state)

        # Agentが持つQTableインスタンスの内部辞書に直接アクセスしてQ値を設定
        # 例: 行動2(LEFT)のQ値を最も高く設定
        self.agent_instance.q_table.q_table[q_state] = [0.0, 0.0, 1.0, 0.0, 0.0] # Q値リスト

        # epsilon=0なので、行動2(LEFT)が選択されるはず
        selected_action = self.agent_instance.get_action(global_state)
        self.assertEqual(selected_action, 2)

        # 最大Q値が複数ある場合、ランダムに選択されるかテスト
        self.agent_instance.q_table.q_table[q_state] = [0.0, 1.0, 1.0, 0.0, 0.0] # 行動1と2が最大

        actions = [self.agent_instance.get_action(global_state) for _ in range(100)] # 100回試行
        # 選択された行動が1または2のみであることを確認
        for action in actions:
            self.assertTrue(action in [1, 2])
        # 少なくとも行動1と2の両方が出現することを確認
        self.assertTrue(1 in actions and 2 in actions)


    def test_agent_decay_epsilon_pow(self):
        """
        Agentのdecay_epsilon_powメソッドによるε減衰が正しく行われるかテスト.
        """
        initial_epsilon = self.agent_instance.epsilon # デフォルトは1.0
        min_epsilon = self.agent_instance.min_epsilon # デフォルトは0.01
        max_epsilon = self.agent_instance.max_epsilon # デフォルトは1.0
        alpha = 0.5

        # ステップ0ではepsilonはmax_epsilonのままか、それに近い値
        self.agent_instance.decay_epsilon_pow(0, alpha)
        # べき乗減衰の式 (1/step**alpha) はstep=0で無限大になるため、step=1として扱う
        expected_epsilon_step0 = max_epsilon * (1.0 / max(1, 0)**alpha) # max(1,0)=1
        self.assertAlmostEqual(self.agent_instance.epsilon, max(expected_epsilon_step0, min_epsilon)) # 1.0 * 1.0 = 1.0

        # ステップ100でのepsilonを確認
        self.agent_instance.epsilon = max_epsilon # epsilonをリセット
        step1 = 100
        self.agent_instance.decay_epsilon_pow(step1, alpha)
        expected_epsilon_step1 = max_epsilon * (1.0 / step1**alpha) # 1.0 * (1/100**0.5) = 1.0 * 0.1 = 0.1
        self.assertAlmostEqual(self.agent_instance.epsilon, max(expected_epsilon_step1, min_epsilon)) # max(0.1, 0.01) = 0.1

        # ステップ10000でのepsilonを確認 (min_epsilonに近づく)
        self.agent_instance.epsilon = max_epsilon # epsilonをリセット
        step2 = 10000
        self.agent_instance.decay_epsilon_pow(step2, alpha)
        expected_epsilon_step2 = max_epsilon * (1.0 / step2**alpha) # 1.0 * (1/10000**0.5) = 1.0 * 0.01 = 0.01
        self.assertAlmostEqual(self.agent_instance.epsilon, max(expected_epsilon_step2, min_epsilon)) # max(0.01, 0.01) = 0.01

        # 非常に大きなステップ数でmin_epsilonに張り付くか確認
        self.agent_instance.epsilon = max_epsilon # epsilonをリセット
        step3 = 1000000
        self.agent_instance.decay_epsilon_pow(step3, alpha)
        expected_epsilon_step3 = max_epsilon * (1.0 / step3**alpha) # 1.0 * (1/1000000**0.5) = 1.0 * 0.001 = 0.001
        self.assertAlmostEqual(self.agent_instance.epsilon, max(expected_epsilon_step3, min_epsilon), places=3) # max(0.001, 0.01) = 0.01

        # alpha=0.1の場合の減衰を確認 (緩やか)
        self.agent_instance.epsilon = max_epsilon # epsilonをリセット
        step4 = 100
        alpha2 = 0.1
        self.agent_instance.decay_epsilon_pow(step4, alpha2)
        expected_epsilon_step4 = max_epsilon * (1.0 / step4**alpha2) # 1.0 * (1/100**0.1) = 1.0 * (100**(-0.1)) approx 1.0 * 0.63
        self.assertAlmostEqual(self.agent_instance.epsilon, max(expected_epsilon_step4, min_epsilon), places=2) # max(0.63.., 0.01) = 0.63..

        # alpha=0.9の場合の減衰を確認 (急峻)
        self.agent_instance.epsilon = max_epsilon # epsilonをリセット
        step5 = 100
        alpha3 = 0.9
        self.agent_instance.decay_epsilon_pow(step5, alpha3)
        expected_epsilon_step5 = max_epsilon * (1.0 / step5**alpha3) # 1.0 * (1/100**0.9) approx 1.0 * 0.0125
        self.assertAlmostEqual(self.agent_instance.epsilon, max(expected_epsilon_step5, min_epsilon), places=3) # max(0.0125.., 0.01) = 0.0125..


    def test_agent_learn(self):
        """
        AgentのlearnメソッドがQTableのlearnメソッドを正しく呼び出すかテスト.
        """
        global_state = ((1, 1), (3, 3), (4, 4)) # goals=1, agents=2
        action = 2 # 例: LEFT
        reward = 1.0
        next_global_state = ((1, 1), (3, 2), (4, 4)) # エージェント0が左に移動した状態を想定
        done = False

        # AgentがQTableに渡すはずのQStateを事前に計算
        current_q_state = self.agent_instance._get_q_state(global_state)
        next_q_state = self.agent_instance._get_q_state(next_global_state)

        # Mock QTableを作成して、Agentが持つQTableインスタンスを置き換える
        # QTableのlearnメソッドが呼ばれたことを確認するためのモック
        class MockQTableForAgentLearnTest:
            def __init__(self, action_size, learning_rate, discount_factor, load_model, model_path):
                self.action_size = action_size
                self.lr = learning_rate
                self.gamma = discount_factor
                # self.load_model = load_model # Removed as QTable doesn't have this attribute
                self.model_path = model_path
                self._learn_called_with = None # learnが呼ばれた引数を記録する変数
                self._initial_q_value = 0.0 # learnメソッド内で必要になる可能性があるため定義
                self.q_table = {} # learnメソッド内でアクセスされる可能性があるため定義

            def learn(self, state, action, reward, next_state, done):
                self._learn_called_with = (state, action, reward, next_state, done)
                # ダミーのTD誤差を返す
                return 0.123 # 適当なfloat値を返す

            # learnメソッド内でget_max_q_valueが呼ばれる可能性があるのでモックしておく
            def get_max_q_value(self, state):
                 return 0.0 # ダミー値を返す

            # learnメソッド内でget_q_valuesが呼ばれる可能性があるのでモックしておく
            def get_q_values(self, state):
                 if state not in self.q_table:
                      self.q_table[state] = [self._initial_q_value] * self.action_size
                 return self.q_table[state]


            # Agentクラスで呼ばれる可能性のある他のメソッドもモック
            def save_q_table(self, file_path):
                 pass # 何もしない

            def get_q_table_size(self):
                 return 0 # ダミー値を返す


        mock_q_table = MockQTableForAgentLearnTest(
            action_size=self.agent_instance.action_size,
            learning_rate=self.agent_instance.q_table.lr,
            discount_factor=self.agent_instance.q_table.gamma,
            load_model=False, # Pass the argument even if not stored as attribute
            model_path="dummy_path.pkl"
        )

        # AgentインスタンスのQTableをモックに置き換え
        self.agent_instance.q_table = mock_q_table# type: ignore

        # Agentのlearnメソッドを実行
        returned_td_delta = self.agent_instance.learn(global_state, action, reward, next_global_state, done)

        # QTableのlearnメソッドが正しい引数で呼ばれたか確認
        self.assertIsNotNone(mock_q_table._learn_called_with)
        called_state, called_action, called_reward, called_next_state, called_done = mock_q_table._learn_called_with# type: ignore

        self.assertEqual(called_state, current_q_state)
        self.assertEqual(called_action, action)
        self.assertEqual(called_reward, reward)
        self.assertEqual(called_next_state, next_q_state)
        self.assertEqual(called_done, done)

        # AgentのlearnメソッドがQTableから返されたTD誤差をそのまま返しているか確認
        self.assertAlmostEqual(returned_td_delta, 0.123)


    def test_agent_save_q_table(self):
        """
        Agentのsave_q_tableメソッドがQTableのsave_q_tableメソッドを正しいパスで呼び出すかテスト.
        """
        # Mock QTableを作成して、Agentが持つQTableインスタンスを置き換える
        class MockQTableForAgentSaveTest:
            def __init__(self, action_size, learning_rate, discount_factor, load_model, model_path):
                 self.model_path = model_path
                 self._save_called_with_path = None # save_q_tableが呼ばれた引数を記録する変数

            def save_q_table(self, file_path: str):
                self._save_called_with_path = file_path

            # Agentクラスで呼ばれる可能性のある他のメソッドもモック
            def learn(self, state, action, reward, next_state, done):
                 return 0.0

            def get_q_values(self, state):
                 return [0.0] * 5

            def get_max_q_value(self, state):
                 return 0.0

            def get_q_table_size(self):
                 return 0

        mock_q_table = MockQTableForAgentSaveTest(5, 0.1, 0.99, False, self.test_q_table_path)

        # AgentインスタンスのQTableをモックに置き換え
        self.agent_instance.q_table = mock_q_table# type: ignore

        # Agentのsave_q_tableメソッドを実行
        self.agent_instance.save_q_table()

        # QTableのsave_q_tableメソッドがAgentのmodel_pathで呼ばれたか確認
        self.assertIsNotNone(mock_q_table._save_called_with_path)
        self.assertEqual(mock_q_table._save_called_with_path, self.test_q_table_path)


    def test_agent_get_q_table_size(self):
        """
        Agentのget_q_table_sizeメソッドがQTableのget_q_table_sizeメソッドを正しく呼び出すかテスト.
        """
        # Mock QTableを作成して、Agentが持つQTableインスタンスを置き換える
        class MockQTableForAgentSizeTest:
            def __init__(self, action_size, learning_rate, discount_factor, load_model, model_path):
                 self._get_size_called = False # get_q_table_sizeが呼ばれたか記録
                 self._dummy_size = 123 # 返すダミーサイズ

            def get_q_table_size(self) -> int:
                self._get_size_called = True
                return self._dummy_size

            # Agentクラスで呼ばれる可能性のある他のメソッドもモック
            def learn(self, state, action, reward, next_state, done):
                 return 0.0

            def get_q_values(self, state):
                 return [0.0] * 5

            def get_max_q_value(self, state):
                 return 0.0

            def save_q_table(self, file_path):
                 pass

        mock_q_table = MockQTableForAgentSizeTest(5, 0.1, 0.99, False, "dummy_path.pkl")

        # AgentインスタンスのQTableをモックに置き換え
        self.agent_instance.q_table = mock_q_table# type: ignore

        # Agentのget_q_table_sizeメソッドを実行
        size = self.agent_instance.get_q_table_size()

        # QTableのget_q_table_sizeメソッドが呼ばれたか確認
        self.assertTrue(mock_q_table._get_size_called)
        # AgentがQTableから返されたサイズをそのまま返しているか確認
        self.assertEqual(size, mock_q_table._dummy_size)


# This allows running the tests directly in a notebook cell
# if __name__ == '__main__':
#     # TestQTableとTestAgentの両方を実行
#     unittest.main(argv=['first-arg-is-ignored'], exit=False)