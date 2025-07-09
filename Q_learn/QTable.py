import numpy as np
import csv
import os

class QTable:
    """
    離散的な状態と行動を扱う汎用的なQテーブルクラス.
    """

    def __init__(self, state_size, action_size, init_type='zero', random_seed=None):
        """
        Qテーブルを初期化する.

        Args:
            state_size: 状態空間のサイズ (離散的な状態の総数).
            action_size: 行動空間のサイズ (離散的な行動の総数).
            init_type: Q値の初期化方法 ('zero' or 'random').
            random_seed: ランダム初期化の場合のシード値.
        """
        self.state_size = state_size
        self.action_size = action_size

        if random_seed is not None:
            np.random.seed(random_seed)

        if init_type == 'zero':
            self.q_table = np.zeros((state_size, action_size))
        elif init_type == 'random':
            # 小さな乱数で初期化するのが一般的
            self.q_table = np.random.randn(state_size, action_size) * 0.01
        else:
            raise ValueError("init_type must be 'zero' or 'random'")

        #print(f"QTable initialized with state_size={state_size}, action_size={action_size}, init_type='{init_type}'")


    def get_q_value(self, state, action):
        """
        特定の状態と行動のペアに対応するQ値を取得する.

        Args:
            state: 取得したい状態 (int).
            action: 取得したい行動 (int).

        Returns:
            対応するQ値 (float).
        """
        if not (0 <= state < self.state_size and 0 <= action < self.action_size):
            raise ValueError("Invalid state or action index")
        return self.q_table[state, action]

    def update_q_value(self, state, action, reward, next_state, learning_rate, discount_factor):
        """
        Q学習の更新式に基づいてQ値を更新する.
        TDターゲット = 報酬 + 割引率 * max(Q(next_state, a'))

        Args:
            state: 現在の状態 (int).
            action: 現在の行動 (int).
            reward: 即時報酬 (float).
            next_state: 次の状態 (int).
            learning_rate: 学習率 (float).
            discount_factor: 割引率 (float).
        """
        if not (0 <= state < self.state_size and 0 <= action < self.action_size):
             raise ValueError("Invalid state or action index for update")
        if not (0 <= next_state < self.state_size):
             raise ValueError("Invalid next_state index for update")


        # 次の状態での最大Q値を取得
        max_next_q = np.max(self.q_table[next_state, :])

        # TDターゲットを計算
        td_target = reward + discount_factor * max_next_q

        # 現在のQ値
        current_q = self.q_table[state, action]

        # TD誤差
        td_delta = td_target - current_q

        # Q値の更新
        self.q_table[state, action] += learning_rate * td_delta

    # SARSA更新用のメソッドも考慮できるが、今回はQ学習に絞る
    # def update_sarsa(self, state, action, reward, next_state, next_action, learning_rate, discount_factor):
    #     """
    #     SARSAの更新式に基づいてQ値を更新する.
    #     TDターゲット = 報酬 + 割引率 * Q(next_state, next_action)
    #     """
    #     if not (0 <= state < self.state_size and 0 <= action < self.action_size):
    #          raise ValueError("Invalid state or action index for update")
    #     if not (0 <= next_state < self.state_size and 0 <= next_action < self.action_size):
    #          raise ValueError("Invalid next_state or next_action index for update")

    #     # TDターゲットを計算
    #     td_target = reward + discount_factor * self.q_table[next_state, next_action]

    #     # 現在のQ値
    #     current_q = self.q_table[state, action]

    #     # TD誤差
    #     td_delta = td_target - current_q

    #     # Q値の更新
    #     self.q_table[state, action] += learning_rate * td_delta


    def save_q_table(self, file_path, file_format='csv'):
        """
        Qテーブルをファイルに保存する.

        Args:
            file_path: 保存先のファイルパス.
            file_format: 保存形式 ('csv').
        """
        if file_format == 'csv':
            try:
                with open(file_path, 'w', newline='') as f:
                    writer = csv.writer(f)
                    writer.writerows(self.q_table)
                print(f"QTable saved to {file_path} in CSV format.")
            except IOError as e:
                print(f"Error saving QTable to CSV: {e}")
                raise
        # Removed pickle save option
        else:
            raise ValueError("file_format must be 'csv'")

    def load_q_table(self, file_path, file_format='csv'):
        """
        ファイルからQテーブルを読み込む.

        Args:
            file_path: 読み込み元のファイルパス.
            file_format: 読み込み形式 ('csv').

        Returns:
            読み込まれたQテーブルのnumpy配列.
        """
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"QTable file not found: {file_path}")

        if file_format == 'csv':
            try:
                with open(file_path, 'r', newline='') as f:
                    reader = csv.reader(f)
                    loaded_q_table = np.array(list(reader), dtype=float)
                # 読み込んだテーブルの形状が現在のQテーブルと一致するか確認
                if loaded_q_table.shape != self.q_table.shape:
                    raise ValueError(f"Loaded QTable shape {loaded_q_table.shape} does not match current QTable shape {self.q_table.shape}")
                self.q_table = loaded_q_table
                print(f"QTable loaded from {file_path} in CSV format.")
            except IOError as e:
                print(f"Error loading QTable from CSV: {e}")
                raise
            except ValueError as e:
                print(f"Error processing CSV data: {e}")
                raise

        # Removed pickle load option
        else:
            raise ValueError("file_format must be 'csv'")

        return self.q_table