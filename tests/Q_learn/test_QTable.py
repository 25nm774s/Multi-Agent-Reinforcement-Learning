import unittest

from src.Q_learn.QTable import QTable, QTableType
from src.Q_learn.QTable import QValues, QState, QTableType

class QTableTests(unittest.TestCase):

    def setUp(self):
        """
        各テストメソッド実行前に呼び出されるセットアップメソッド.
        """
        class ArgsManager:
            def __init__(self):
                self.action_size = 5
                self.lr = 0.5
                self.gamma = 0.9

        self.args = ArgsManager()
        self.qtable = QTable(self.args.action_size, self.args.lr, self.args.gamma)

    def _set_dummy_qtable_data(self) -> QTableType:
        """
        テスト用のダミーQテーブルデータを生成するヘルパーメソッド.
        """
        return {
            (2, 5, 2, 14, 15, 2, 8, 7): [0.95, 0.10, 0.00, 25.0, -1.25],
            (2, 5, 2, 14, 15, 2, 8, 8): [0.85, 0.10, 1.00, 24.0, -1.25],
            (2, 5, 2, 14, 15, 2, 8, 9): [0.75, 0.10, 2.00, 23.0, -1.25],
        }

    def test_01_initialization(self):
        """
        QTable の初期化が正しく行われるかテスト.
        """
        self.assertEqual(self.qtable.action_size, self.args.action_size)
        self.assertEqual(self.qtable.lr, self.args.lr)
        self.assertEqual(self.qtable.gamma, self.args.gamma)
        self.assertEqual(self.qtable.get_q_table_size(), 0) # 初期状態ではQテーブルは空

    def test_02_get_q_values(self):
        """
        QTable の get_q_values メソッドが正しく Q 値を取得できるかテスト.
        存在する状態と存在しない状態の両方を確認.
        """
        dammy_qtable = QTable(5, 0.5, 0.9)
        dammy_qtable.set_Qtable(self._set_dummy_qtable_data())

        # 存在するケースを確認
        self.assertEqual(dammy_qtable.get_q_values((2, 5, 2, 14, 15, 2, 8, 7)), [0.95, 0.10, 0.00, 25.0, -1.25])
        self.assertEqual(dammy_qtable.get_q_values((2, 5, 2, 14, 15, 2, 8, 8)), [0.85, 0.10, 1.00, 24.0, -1.25])

        # 存在しないケースを確認 (初期化されて返ってくるはず)
        self.assertEqual(dammy_qtable.get_q_values((1, 0, 1, 0, 1, 0, 0, 0)), [0.0, 0.0, 0.0, 0.0, 0.0])
        # get_q_values を呼び出したことで新しい状態が追加されたか確認
        self.assertEqual(dammy_qtable.get_q_table_size(), len(self._set_dummy_qtable_data()) + 1)


    def test_03_get_max_q_value(self):
        """
        QTable の get_max_q_value メソッドが正しく最大 Q 値を取得できるかテスト.
        存在する状態と存在しない状態の両方を確認.
        """
        dammy_qtable = QTable(5, 0.5, 0.9)
        dammy_qtable.set_Qtable(self._set_dummy_qtable_data())

        # 存在するケースを確認
        self.assertEqual(dammy_qtable.get_max_q_value((2, 5, 2, 14, 15, 2, 8, 7)), 25.0)
        self.assertEqual(dammy_qtable.get_max_q_value((2, 5, 2, 14, 15, 2, 8, 8)), 24.0)

        # 存在しないケースを確認 (初期値の最大値である 0.0 が返ってくるはず)
        self.assertEqual(dammy_qtable.get_max_q_value((1, 0, 1, 0, 1, 0, 0, 0)), 0.0)
        # get_max_q_value を呼び出したことで新しい状態が追加されたか確認
        self.assertEqual(dammy_qtable.get_q_table_size(), len(self._set_dummy_qtable_data()) + 1)

    def test_04_learn_method(self):
        """
        QTable の learn メソッドによる Q テーブルの更新が正しく行われるかテスト.
        Q 学習の更新式に基づいた更新が行われるか確認. done=True のケースも確認.
        """
        q_table_instance = QTable(self.args.action_size, self.args.lr, self.args.gamma)

        current_q_state: QState = (1, 1, 3, 3)
        action = 1 # 例: DOWN
        reward = 10.0
        next_q_state: QState = (1, 1, 3, 4)
        done = False

        # 初期Q値を設定 (テストのため)
        initial_current_q = 1.0
        q_table_instance.q_table[current_q_state] = [initial_current_q] * self.args.action_size

        initial_next_q_values = [0.1, 0.2, 0.3, 0.4, 0.5] # 次の状態でのQ値リスト
        q_table_instance.q_table[next_q_state] = initial_next_q_values

        # learnメソッドを実行
        td_delta = q_table_instance.learn(current_q_state, action, reward, next_q_state, done)

        # Q学習の更新式: Q(s, a) = Q(s, a) + α * [r + γ * max(Q(s', a')) - Q(s, a)]
        max_next_q = max(initial_next_q_values)
        expected_td_target = reward + self.args.gamma * max_next_q
        expected_td_delta = expected_td_target - initial_current_q
        expected_updated_q_value = initial_current_q + self.args.lr * expected_td_delta

        # 更新後のQ値を確認
        updated_q_values = q_table_instance.q_table[current_q_state]
        self.assertEqual(updated_q_values[action], expected_updated_q_value)

        # TD誤差の絶対値が返されるか確認
        self.assertEqual(td_delta, abs(expected_td_delta))

        # done=Trueの場合のテスト
        done = True
        reward_on_done = 10.0
        # Q値をリセットして再テスト
        q_table_instance.q_table[current_q_state] = [initial_current_q] * self.args.action_size
        # learnメソッドを実行
        td_delta_done = q_table_instance.learn(current_q_state, action, reward_on_done, next_q_state, done)

        # done=Trueの場合、max_next_qは0.0になる
        expected_td_target_done = reward_on_done + self.args.gamma * 0.0
        expected_td_delta_done = expected_td_target_done - initial_current_q
        expected_updated_q_value_done = initial_current_q + self.args.lr * expected_td_delta_done

        # 更新後のQ値リストを取得
        updated_q_values_done = q_table_instance.q_table[current_q_state]
        self.assertEqual(updated_q_values_done[action], expected_updated_q_value_done)
        self.assertEqual(td_delta_done, abs(expected_td_delta_done))

    def test_05_empty_qtable_behavior(self):
        """
        QTable が空の場合の各種メソッドの挙動をテスト.
        """
        empty_qtable = QTable(5, 0.1, 0.9)
        empty_state: QState = (1, 2, 3, 4)

        # get_q_values が正しく初期化されたリストを返すか
        initial_q_values = empty_qtable.get_q_values(empty_state)
        self.assertEqual(initial_q_values, [0.0] * empty_qtable.action_size)
        # get_q_values を呼び出したことで状態が追加されたか確認
        self.assertIn(empty_state, empty_qtable.get_Qtable())

        # get_max_q_value が正しく初期値 (0.0) を返すか
        self.assertEqual(empty_qtable.get_max_q_value(empty_state), 0.0)

        # get_q_table_size が 1 を返すか (get_q_values で追加されたため)
        self.assertEqual(empty_qtable.get_q_table_size(), 1)

        # 新しい空のQTableでget_q_table_sizeを確認
        another_empty_qtable = QTable(5, 0.1, 0.9)
        self.assertEqual(another_empty_qtable.get_q_table_size(), 0)

    def test_06_learn_with_extreme_params(self):
        """
        学習率や割引率が 0 や 1 の場合の learn メソッドの挙動をテスト.
        """
        # 学習率 α = 0 の場合のテスト
        qtable_lr_zero = QTable(5, 0.0, 0.9)
        state_lr_zero: QState = (1, 1, 1, 1)
        next_state_lr_zero: QState = (1, 1, 1, 2)
        action_lr_zero = 1
        reward_lr_zero = 5.0
        done_lr_zero = False

        qtable_lr_zero.q_table[state_lr_zero] = [1.0] * 5
        qtable_lr_zero.q_table[next_state_lr_zero] = [0.1, 0.2, 0.3, 0.4, 0.5]

        initial_q_value_lr_zero = qtable_lr_zero.q_table[state_lr_zero][action_lr_zero]
        qtable_lr_zero.learn(state_lr_zero, action_lr_zero, reward_lr_zero, next_state_lr_zero, done_lr_zero)
        updated_q_value_lr_zero = qtable_lr_zero.q_table[state_lr_zero][action_lr_zero]

        self.assertEqual(updated_q_value_lr_zero, initial_q_value_lr_zero) # 学習率0なのでQ値は更新されない

        # 割引率 γ = 0 の場合のテスト
        qtable_gamma_zero = QTable(5, 0.5, 0.0)
        state_gamma_zero: QState = (2, 2, 2, 2)
        next_state_gamma_zero: QState = (2, 2, 2, 3)
        action_gamma_zero = 2
        reward_gamma_zero = 10.0
        done_gamma_zero = False

        qtable_gamma_zero.q_table[state_gamma_zero] = [2.0] * 5
        qtable_gamma_zero.q_table[next_state_gamma_zero] = [0.6, 0.7, 0.8, 0.9, 1.0]

        initial_q_value_gamma_zero = qtable_gamma_zero.q_table[state_gamma_zero][action_gamma_zero]
        qtable_gamma_zero.learn(state_gamma_zero, action_gamma_zero, reward_gamma_zero, next_state_gamma_zero, done_gamma_zero)
        updated_q_value_gamma_zero = qtable_gamma_zero.q_table[state_gamma_zero][action_gamma_zero]

        # TDターゲット = 報酬 + γ * max(next_q) = 10.0 + 0.0 * max([0.6..1.0]) = 10.0
        expected_td_target_gamma_zero = reward_gamma_zero + 0.0 * max(qtable_gamma_zero.q_table[next_state_gamma_zero])
        expected_td_delta_gamma_zero = expected_td_target_gamma_zero - initial_q_value_gamma_zero
        expected_updated_q_gamma_zero = initial_q_value_gamma_zero + 0.5 * expected_td_delta_gamma_zero
        self.assertEqual(updated_q_value_gamma_zero, expected_updated_q_gamma_zero)


        # 学習率 α = 1 の場合のテスト
        qtable_lr_one = QTable(5, 1.0, 0.9)
        state_lr_one: QState = (3, 3, 3, 3)
        next_state_lr_one: QState = (3, 3, 3, 4)
        action_lr_one = 3
        reward_lr_one = 15.0
        done_lr_one = False

        qtable_lr_one.q_table[state_lr_one] = [3.0] * 5
        qtable_lr_one.q_table[next_state_lr_one] = [1.1, 1.2, 1.3, 1.4, 1.5]

        initial_q_value_lr_one = qtable_lr_one.q_table[state_lr_one][action_lr_one]
        qtable_lr_one.learn(state_lr_one, action_lr_one, reward_lr_one, next_state_lr_one, done_lr_one)
        updated_q_value_lr_one = qtable_lr_one.q_table[state_lr_one][action_lr_one]

        # TDターゲット = 報酬 + γ * max(next_q) = 15.0 + 0.9 * max([1.1..1.5]) = 15.0 + 0.9 * 1.5 = 16.35
        # Q(s,a) = Q(s,a) + 1.0 * [TDターゲット - Q(s,a)] = TDターゲット
        expected_td_target_lr_one = reward_lr_one + 0.9 * max(qtable_lr_one.q_table[next_state_lr_one])
        self.assertEqual(updated_q_value_lr_one, expected_td_target_lr_one)

        # 割引率 γ = 1 の場合のテスト
        qtable_gamma_one = QTable(5, 0.5, 1.0)
        state_gamma_one: QState = (4, 4, 4, 4)
        next_state_gamma_one: QState = (4, 4, 4, 5)
        action_gamma_one = 4
        reward_gamma_one = 20.0
        done_gamma_one = False

        qtable_gamma_one.q_table[state_gamma_one] = [4.0] * 5
        qtable_gamma_one.q_table[next_state_gamma_one] = [2.1, 2.2, 2.3, 2.4, 2.5]

        initial_q_value_gamma_one = qtable_gamma_one.q_table[state_gamma_one][action_gamma_one]
        qtable_gamma_one.learn(state_gamma_one, action_gamma_one, reward_gamma_one, next_state_gamma_one, done_gamma_one)
        updated_q_value_gamma_one = qtable_gamma_one.q_table[state_gamma_one][action_gamma_one]

        # TDターゲット = 報酬 + γ * max(next_q) = 20.0 + 1.0 * max([2.1..2.5]) = 20.0 + 1.0 * 2.5 = 22.5
        expected_td_target_gamma_one = reward_gamma_one + 1.0 * max(qtable_gamma_one.q_table[next_state_gamma_one])
        expected_td_delta_gamma_one = expected_td_target_gamma_one - initial_q_value_gamma_one
        expected_updated_q_gamma_one = initial_q_value_gamma_one + 0.5 * expected_td_delta_gamma_one
        self.assertEqual(updated_q_value_gamma_one, expected_updated_q_gamma_one)

    def test_07_set_Qtable_method(self):
        """
        QTable の set_Qtable メソッドが正しく Q テーブルを設定できるかテスト.
        """
        qtable_for_set = QTable(5, 0.1, 0.9)
        initial_qtable_data = qtable_for_set.get_Qtable() # 初期状態は空辞書

        new_qtable_data: QTableType = {
            (10, 20): [1.0, 2.0, 3.0, 4.0, 5.0],
            (30, 40): [5.0, 4.0, 3.0, 2.0, 1.0]
        }

        qtable_for_set.set_Qtable(new_qtable_data)

        # set_Qtable で設定したデータが正しく取得できるか確認
        retrieved_qtable_data = qtable_for_set.get_Qtable()
        self.assertEqual(retrieved_qtable_data, new_qtable_data)

        # サイズも正しく更新されているか確認
        self.assertEqual(qtable_for_set.get_q_table_size(), len(new_qtable_data))

        # set_Qtable で設定後、learn メソッドが正しく動作するか簡単なテスト
        state_after_set: QState = (10, 20)
        next_state_after_set: QState = (30, 40)
        action_after_set = 0
        reward_after_set = 100.0
        done_after_set = False

        initial_q_value_after_set = qtable_for_set.get_q_values(state_after_set)[action_after_set] # 1.0

        qtable_for_set.learn(state_after_set, action_after_set, reward_after_set, next_state_after_set, done_after_set)

        updated_q_value_after_set = qtable_for_set.get_q_values(state_after_set)[action_after_set]
        max_next_q_after_set = qtable_for_set.get_max_q_value(next_state_after_set) # max([5.0..1.0]) == 5.0

        # 期待される更新値の計算
        expected_td_target_after_set = reward_after_set + qtable_for_set.gamma * max_next_q_after_set
        expected_td_delta_after_set = expected_td_target_after_set - initial_q_value_after_set
        expected_updated_q_value_after_set = initial_q_value_after_set + qtable_for_set.lr * expected_td_delta_after_set

        self.assertEqual(updated_q_value_after_set, expected_updated_q_value_after_set)

    def test_08_invalid_input_behavior(self):
        """
        不正な入力に対する QTable の挙動をテスト (特に learn メソッドの action).
        """
        qtable_invalid_action = QTable(5, 0.1, 0.9)
        state_invalid_action: QState = (5, 5)
        next_state_invalid_action: QState = (5, 6)
        reward_invalid_action = 1.0
        done_invalid_action = False

        qtable_invalid_action.q_table[state_invalid_action] = [0.1] * 5
        qtable_invalid_action.q_table[next_state_invalid_action] = [0.2] * 5

        # action_size (5) 以上の action を指定して ValueError が発生することを確認
        invalid_action = 5 # action_size は 5 なのでインデックスは 0-4
        with self.assertRaises(ValueError):
            qtable_invalid_action.learn(state_invalid_action, invalid_action, reward_invalid_action, next_state_invalid_action, done_invalid_action)

        # action が負の値の場合も同様に ValueError が発生することを確認
        invalid_action_negative = -1
        with self.assertRaises(ValueError):
            qtable_invalid_action.learn(state_invalid_action, invalid_action_negative, reward_invalid_action, next_state_invalid_action, done_invalid_action)

        # 存在しない state を get_q_values に渡すテストは test_02 でカバー済み
        # 存在しない state を get_max_q_value に渡すテストは test_03 でカバー済み
