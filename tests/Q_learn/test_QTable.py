import unittest
import numpy as np
import os
import csv
import pickle
from typing import Tuple, Dict, List, Any


#from src.Q_learn.QTable import QTable
# QTableクラスとSimpleArgsクラスが定義されている必要があります。
# ここでは、ノートブックの以前のセルで定義されたQTableとSimpleArgsクラスを使用することを想定しています。
# Constants
RED = '\033[91m'
GREEN = '\033[92m'
RESET = '\033[0m'

QState = Tuple[int, ...]
QValues = List[float]
QTableType = Dict[QState, QValues]


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


class TestQTable(unittest.TestCase):

    def setUp(self):
        """
        各テストメソッドの実行前に呼ばれるセットアップメソッド.
        テスト用のQTableインスタンスとArgsオブジェクトを作成します.
        """
        class TestArgs: # テスト用の簡易Argsクラス (QTableには不要だが、Agentテストで使うため残す)
            def __init__(self, grid_size=5, agents_number=2, goals_number=1, reward_mode=2,
                         render_mode=False, window_width=500, window_height=500,
                         episode_number=10, max_timestep=100, buffer_size=10000,
                         batch_size=32, decay_epsilon=1000, load_model=0, mask=False, device="cpu",
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
                self.decay_epsilon = decay_epsilon
                self.load_model = load_model
                self.mask = mask
                self.device = device
                self.learning_rate = learning_rate
                self.discount_factor = discount_factor
                self.epsilon = epsilon
                self.min_epsilon = min_epsilon
                self.max_epsilon = max_epsilon
                self.dir_path = "test_models" # テスト用に一時的なディレクトリを指定


        self.args = TestArgs() # Agentテスト用
        self.action_size = 5 # QTableコンストラクタに必要なサイズ
        self.lr = self.args.learning_rate
        self.gamma = self.args.discount_factor
        self.load_model = False # QTableテストでは基本的にロードしないモードで開始
        self.test_dir = "test_qtable_saves" # QTable保存テスト用のディレクトリ
        self.test_q_table_path = os.path.join(self.test_dir, "q_table_test.pkl")

        # テスト前に既存のファイルとディレクトリがあれば削除
        if os.path.exists(self.test_dir):
            # ディレクトリ内のファイルを全て削除
            for file_name in os.listdir(self.test_dir):
                file_path_to_remove = os.path.join(self.test_dir, file_name)
                if os.path.isfile(file_path_to_remove):
                    os.remove(file_path_to_remove)
            # ディレクトリが空になったら削除
            if not os.listdir(self.test_dir):
                os.rmdir(self.test_dir)


        # 新しいQTableインスタンスを作成
        self.q_table_instance = QTable(
            action_size=self.action_size,
            learning_rate=self.lr,
            discount_factor=self.gamma,
            load_model=self.load_model,
            model_path=self.test_q_table_path # ダミーパスだがload/saveテストで使う
        )

        # テスト用のディレクトリを作成
        os.makedirs(self.test_dir, exist_ok=True)


    def tearDown(self):
        """
        各テストメソッドの実行後に呼ばれるクリーンアップメソッド.
        テストで作成したファイルやディレクトリを削除します.
        """
        # テストで作成したディレクトリとファイルを削除
        if os.path.exists(self.test_dir):
            for file_name in os.listdir(self.test_dir):
                file_path_to_remove = os.path.join(self.test_dir, file_name)
                if os.path.isfile(file_path_to_remove):
                    os.remove(file_path_to_remove)
            if not os.listdir(self.test_dir):
                os.rmdir(self.test_dir)


    def test_qtable_initialization(self):
        """
        QTableクラスの初期化が正しく行われるかテスト.
        """
        self.assertIsNotNone(self.q_table_instance.q_table)
        self.assertEqual(len(self.q_table_instance.q_table), 0) # 初期状態ではQテーブルは空
        self.assertEqual(self.q_table_instance.action_size, self.action_size)
        self.assertAlmostEqual(self.q_table_instance.lr, self.lr)
        self.assertAlmostEqual(self.q_table_instance.gamma, self.gamma)
        # epsilonはAgentクラスに移動したので、QTableには存在しないことを確認 (またはチェックしない)
        # self.assertFalse(hasattr(self.q_table_instance, 'epsilon'))


    def test_qtable_learn(self):
        """
        QTableのlearnメソッドによるQテーブルの更新が正しく行われるかテスト (シングルエージェント想定).
        Q学習の更新式に基づいた更新が行われるか確認します.
        """
        # QTableのlearnメソッドはフラット化されたQStateを直接受け取る
        current_q_state: QState = (1, 1, 3, 3) # 例: goal=(1,1), agent=(3,3)
        action = 2 # 例: LEFT
        reward = 1.0
        next_q_state: QState = (1, 1, 3, 2) # 例: goal=(1,1), agent=(3,2) (agentが左に移動)
        done = False

        # 初期Q値を設定 (テストのため)
        initial_current_q = 0.5
        # QTableの内部辞書に直接アクセスして初期値を設定
        self.q_table_instance.q_table[current_q_state] = [initial_current_q] * self.action_size

        initial_next_q_values = [0.1, 0.2, 0.3, 0.4, 0.5] # 次の状態でのQ値リスト
        # QTableの内部辞書に直接アクセスして初期値を設定
        self.q_table_instance.q_table[next_q_state] = initial_next_q_values


        # learnメソッドを実行
        td_delta = self.q_table_instance.learn(current_q_state, action, reward, next_q_state, done)

        # Q学習の更新式: Q(s, a) = Q(s, a) + α * [r + γ * max(Q(s', a')) - Q(s, a)]
        # ここでは action=2 のQ値が更新される
        max_next_q = max(initial_next_q_values) # 0.5
        expected_td_target = reward + self.gamma * max_next_q
        expected_td_delta = expected_td_target - initial_current_q # td_deltaは更新前のQ値で計算される
        expected_updated_q_value = initial_current_q + self.lr * expected_td_delta

        # 更新後のQ値を確認
        # QTableの内部辞書から更新後のQ値リストを取得
        updated_q_values = self.q_table_instance.q_table[current_q_state]
        self.assertAlmostEqual(updated_q_values[action], expected_updated_q_value)

        # TD誤差の絶対値が返されるか確認
        self.assertAlmostEqual(td_delta, abs(expected_td_delta))

        # done=Trueの場合のテスト
        done = True
        reward_on_done = 10.0
        # Q値をリセットして再テスト
        self.q_table_instance.q_table[current_q_state] = [initial_current_q] * self.action_size
        # learnメソッドを実行
        td_delta_done = self.q_table_instance.learn(current_q_state, action, reward_on_done, next_q_state, done)

        # done=Trueの場合、max_next_qは0.0になる
        expected_td_target_done = reward_on_done + self.gamma * 0.0
        expected_td_delta_done = expected_td_target_done - initial_current_q
        expected_updated_q_value_done = initial_current_q + self.lr * expected_td_delta_done

        # QTableの内部辞書から更新後のQ値リストを取得
        updated_q_values_done = self.q_table_instance.q_table[current_q_state]
        self.assertAlmostEqual(updated_q_values_done[action], expected_updated_q_value_done)
        self.assertAlmostEqual(td_delta_done, abs(expected_td_delta_done))


    def test_qtable_learn_multi_agent(self):
        """
        QTableのlearnメソッドのテスト (マルチエージェント).
        マルチエージェントの状態表現でTD学習の更新が正しく行われるかを確認.
        """
        # マルチエージェントの状態表現: (goal1_x, goal1_y, ..., agent1_x, agent1_y, agent2_x, agent2_y, ...)
        # テスト用にゴール1つ、エージェント2つを想定 (grid_size=5)
        # 状態タプルの形式は (goal1_x, goal1_y, agent1_x, agent1_y, agent2_x, agent2_y)
        # action_sizeはsetUpで5に設定されているので、行動数は5を想定

        # 初期状態と行動
        state = (0, 0, 1, 1, 3, 3) # 例: ゴール(0,0), エージェント1(1,1), エージェント2(3,3)
        # agent_idx = 0 # learnメソッドはエージェントIDを引数にとらない
        action = 1 # エージェント1の行動1 (右に移動)を想定 (実際はグローバルな行動空間のインデックス)
        reward = -1.0 # 報酬 (エージェント1に対するもの)
        next_state = (0, 0, 1, 2, 3, 3) # 例: 次の状態 ゴール(0,0), エージェント1(1,2), エージェント2(3,3)
        done = False # エピソードは完了していない

        # learnメソッドはエージェントIDを引数にとらないため、状態タプル自体に全てのエージェントの状態が含まれている必要がある.
        # このテストでは、learnメソッドが受け取る state と next_state がマルチエージェントの状態表現であることを確認する.
        # learnメソッド内のQテーブルへのアクセスは、このタプルをキーとして行われる.

        # learnメソッド実行前のQ値を取得 (初期値は0.0)
        initial_q = self.q_table_instance.get_q_values(state)[action]
        self.assertEqual(initial_q, 0.0)

        # 次の状態のQ値を事前に設定 (テストのために任意の値を入れる)
        # next_state (0, 0, 1, 2, 3, 3) の行動0,1,2,3,4 のQ値を例えば [0.6, 0.3, 0.9, 0.2, 0.5] とする
        if next_state not in self.q_table_instance.q_table:
            self.q_table_instance.q_table[next_state] = [self.q_table_instance._initial_q_value] * self.action_size
        self.q_table_instance.q_table[next_state] = [0.6, 0.3, 0.9, 0.2, 0.5]
        max_next_q = max(self.q_table_instance.q_table[next_state]) # 0.9

        # learnメソッドを実行
        td_delta = self.q_table_instance.learn(state, action, reward, next_state, done)

        # learnメソッド実行後のQ値を取得
        updated_q = self.q_table_instance.q_table.get(state, [0.0]*self.action_size)[action]

        # TDターゲットの計算: r + gamma * max_next_q
        expected_td_target = reward + self.gamma * max_next_q # -1.0 + 0.99 * 0.9 = -1.0 + 0.891 = -0.109
        # setUpでgammaは0.99に設定されていることに注意

        # TD誤差の計算: td_target - initial_q
        expected_td_delta = expected_td_target - initial_q # -0.109 - 0.0 = -0.109

        # Q値の更新計算: initial_q + lr * td_delta
        expected_updated_q = initial_q + self.lr * expected_td_delta # 0.0 + 0.1 * (-0.109) = -0.0109
        # setUpでlrは0.1に設定されていることに注意

        self.assertAlmostEqual(updated_q, expected_updated_q, places=5)

        # 返されたTD誤差の絶対値が期待値と一致するか確認
        self.assertAlmostEqual(abs(td_delta), abs(expected_td_delta), places=5)

        # doneがTrueの場合のテスト (次の状態の価値が0になること)
        state_done = (0, 0, 2, 2, 4, 4)
        action_done = 0
        reward_done = 10.0 # ゴール報酬
        next_state_done = (0, 0, 2, 2, 4, 4) # ゴール後の状態 (環境によるが、ここでは同じ状態とする)
        done_flag = True

        initial_q_done = self.q_table_instance.get_q_values(state_done)[action_done] # 初期値0.0

        td_delta_done = self.q_table_instance.learn(state_done, action_done, reward_done, next_state_done, done_flag)

        updated_q_done = self.q_table_instance.q_table.get(state_done, [0.0]*self.action_size)[action_done]

        # TDターゲット (done=True): r + gamma * 0
        expected_td_target_done = reward_done + self.gamma * 0.0 # 10.0 + 0.99 * 0.0 = 10.0

        # TD誤差 (done=True): td_target - initial_q
        expected_td_delta_done = expected_td_target_done - initial_q_done # 10.0 - 0.0 = 10.0

        # Q値の更新 (done=True): initial_q + lr * td_delta
        expected_updated_q_done = initial_q_done + self.lr * expected_td_delta_done # 0.0 + 0.1 * 10.0 = 1.0

        self.assertAlmostEqual(updated_q_done, expected_updated_q_done, places=5)
        self.assertAlmostEqual(abs(td_delta_done), abs(expected_td_delta_done), places=5)


    def test_qtable_save_and_load(self):
        """
        QTableの保存とロードが正しく行われるかテスト.
        """
        # Qテーブルにダミーデータを設定
        dummy_state: QState = (0, 0, 1, 1)
        dummy_q_values = [0.1, 0.2, 0.3, 0.4, 0.5]
        self.q_table_instance.q_table[dummy_state] = dummy_q_values

        # 保存メソッドをテスト (setupで設定したパスを使用)
        self.q_table_instance.save_q_table(self.test_q_table_path)
        self.assertTrue(os.path.exists(self.test_q_table_path))

        # ロード用の新しいQTableインスタンスを作成 (load_model=Trueで初期化)
        loaded_q_table_instance = QTable(
            action_size=self.action_size,
            learning_rate=self.lr,
            discount_factor=self.gamma,
            load_model=True, # ロードモードを有効に
            model_path=self.test_q_table_path
        )

        # ロードされたQテーブルの内容を確認
        self.assertIn(dummy_state, loaded_q_table_instance.q_table)
        loaded_q_values = loaded_q_table_instance.q_table[dummy_state]
        self.assertEqual(loaded_q_values, dummy_q_values)
        self.assertEqual(loaded_q_table_instance.get_q_table_size(), 1) # ダミーデータ1つなのでサイズは1


    def test_qtable_get_q_values(self):
        """
        QTableのget_q_valuesメソッドが正しくQ値を取得できるかテスト (シングルエージェント想定).
        """
        existing_state: QState = (10, 10, 2, 2)
        existing_q_values = [5.0, -1.0, 0.0, 1.0, 2.5]
        self.q_table_instance.q_table[existing_state] = existing_q_values

        # 既存の状態のQ値を取得
        retrieved_q_values = self.q_table_instance.get_q_values(existing_state)
        self.assertEqual(retrieved_q_values, existing_q_values)

        # 存在しない状態のQ値を取得 - 新しい状態が初期化されて返されるか確認
        new_state: QState = (10, 10, 3, 3)
        initial_q_values = [self.q_table_instance._initial_q_value] * self.action_size
        retrieved_new_q_values = self.q_table_instance.get_q_values(new_state)
        self.assertEqual(retrieved_new_q_values, initial_q_values)
        self.assertIn(new_state, self.q_table_instance.q_table) # 新しい状態がQテーブルに追加されたか確認

    def test_qtable_get_q_values_multi_agent(self):
        """
        QTableのget_q_valuesメソッドのテスト (マルチエージェント).
        マルチエージェントの状態表現でQ値リストが正しく取得できるかを確認.
        """
        # マルチエージェントの状態表現: (goal1_x, goal1_y, ..., agent1_x, agent1_y, agent2_x, agent2_y, ...)
        # テスト用にゴール1つ、エージェント2つを想定 (grid_size=5)
        # 状態タプルの形式は (goal1_x, goal1_y, agent1_x, agent1_y, agent2_x, agent2_y)

        # テスト用の状態と期待されるQ値
        state_existing = (0, 0, 1, 1, 3, 3) # 例: ゴール(0,0), エージェント1(1,1), エージェント2(3,3)
        q_values_existing = [0.1, -0.5, 1.2, 0.0, 0.5] # この状態に対するQ値リスト (action_size=5)

        state_new = (0, 0, 2, 2, 4, 4) # 例: ゴール(0,0), エージェント1(2,2), エージェント2(4,4)

        # 既存の状態をQテーブルに設定
        self.q_table_instance.q_table[state_existing] = q_values_existing

        # 既存の状態に対するget_q_valuesのテスト
        retrieved_q_existing = self.q_table_instance.get_q_values(state_existing)
        self.assertEqual(retrieved_q_existing, q_values_existing)

        # 新しい状態に対するget_q_valuesのテスト (初期化されることを確認)
        retrieved_q_new = self.q_table_instance.get_q_values(state_new)
        expected_q_new = [self.q_table_instance._initial_q_value] * self.action_size
        self.assertEqual(retrieved_q_new, expected_q_new)
        # 新しい状態がQテーブルに追加されたことを確認
        self.assertIn(state_new, self.q_table_instance.q_table)


    def test_qtable_get_max_q_value(self):
        """
        QTableのget_max_q_valueメソッドが正しく最大Q値を取得できるかテスト (シングルエージェント想定).
        """
        existing_state: QState = (20, 20, 5, 5)
        existing_q_values = [5.0, -1.0, 0.0, 1.0, 2.5]
        self.q_table_instance.q_table[existing_state] = existing_q_values

        # 既存の状態の最大Q値を取得
        max_q = self.q_table_instance.get_max_q_value(existing_state)
        self.assertAlmostEqual(max_q, 5.0)

        # 存在しない状態の最大Q値を取得 - 初期化されたQ値リストの最大値(通常0.0)が返されるか確認
        new_state: QState = (20, 20, 6, 6)
        # QTableのget_q_valuesが内部で状態を初期化するので、直接get_max_q_valueを呼び出す
        max_q_new_state = self.q_table_instance.get_max_q_value(new_state)
        self.assertAlmostEqual(max_q_new_state, self.q_table_instance._initial_q_value)
        self.assertIn(new_state, self.q_table_instance.q_table) # 新しい状態がQテーブルに追加されたか確認

    def test_qtable_get_max_q_value_multi_agent(self):
        """
        QTableのget_max_q_valueメソッドが正しく最大Q値を取得できるかテスト (マルチエージェント).
        マルチエージェントの状態表現で最大Q値が取得できるかを確認.
        """
         # マルチエージェントの状態表現: (goal1_x, goal1_y, ..., agent1_x, agent1_y, agent2_x, agent2_y, ...)
        # テスト用にゴール1つ、エージェント2つを想定 (grid_size=5)
        # 状態タプルの形式は (goal1_x, goal1_y, agent1_x, agent1_y, agent2_x, agent2_y)

        existing_state: QState = (0, 0, 1, 1, 3, 3) # 例: ゴール(0,0), エージェント1(1,1), エージェント2(3,3)
        existing_q_values = [0.1, -0.5, 1.2, 0.0, 0.5] # この状態に対するQ値リスト (action_size=5)
        self.q_table_instance.q_table[existing_state] = existing_q_values

        # 既存の状態の最大Q値を取得
        max_q = self.q_table_instance.get_max_q_value(existing_state)
        self.assertAlmostEqual(max_q, 1.2)

        # 存在しない状態の最大Q値を取得 - 初期化されたQ値リストの最大値(通常0.0)が返されるか確認
        new_state: QState = (0, 0, 2, 2, 4, 4)
        # QTableのget_q_valuesが内部で状態を初期化するので、直接get_max_q_valueを呼び出す
        max_q_new_state = self.q_table_instance.get_max_q_value(new_state)
        self.assertAlmostEqual(max_q_new_state, self.q_table_instance._initial_q_value)
        self.assertIn(new_state, self.q_table_instance.q_table) # 新しい状態がQテーブルに追加されたか確認


# QTableのテストを実行
# unittest.main(argv=['first-arg-is-ignored'], exit=False)