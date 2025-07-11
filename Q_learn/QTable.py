# Constants (already defined in a previous cell)
# RED = '\033[91m'
# GREEN = '\033[92m'
# RESET = '\033[0m'

import numpy as np
from typing import Tuple, Dict, List, Any
import pickle
import os

# Define the type aliases again for clarity within this cell
QState = Tuple[int, ...]
QValues = List[float]
QTableType = Dict[QState, QValues]

class QTable:
    """
    Qテーブルのデータ構造と学習ロジックを管理するクラス.
    エージェントIDや行動選択ロジックは含まない.
    """
    def __init__(self, action_size: int, learning_rate: float, discount_factor: float,
                 load_model: bool, model_path: str):
        """
        QTable コンストラクタ.

        Args:
            action_size (int): 環境の行動空間サイズ.
            learning_rate (float): 学習率 α.
            discount_factor (float): 割引率 γ.
            load_model (bool): 既存のQテーブルをロードするかどうか.
            model_path (str): Qテーブルファイルのパス.
        """
        self.action_size = action_size
        self.lr = learning_rate       # 学習率 α
        self.gamma = discount_factor  # 割引率 γ

        # Qテーブルを辞書として初期化
        self.q_table: QTableType = {}

        # Q値の初期化に使用する値
        self._initial_q_value = 0.0

        self.model_path = model_path

        # ロードフラグが立っている場合はQテーブルをロード
        if load_model:
            self.load_q_table(self.model_path)


    def learn(self, state: QState, action: int, reward: float, next_state: QState, done: bool) -> float:
        """
        単一の経験 (状態, 行動, 報酬, 次の状態, 完了フラグ) に基づいてQテーブルを更新する (Q学習).

        Args:
            state (QState): 現在の状態のQテーブル用表現 (フラット化タプル).
            action (int): エージェントが取った行動.
            reward (float): 行動によって得られた報酬.
            next_state (QState): 次の状態のQテーブル用表現 (フラット化タプル).
            done (bool): エピソードが完了したかどうかのフラグ.

        Returns:
            float: 更新に使用されたTD誤差の絶対値.
        """
        # Qテーブルに状態が存在しない場合は初期化
        if state not in self.q_table:
             self.q_table[state] = [self._initial_q_value] * self.action_size
        if next_state not in self.q_table:
             self.q_table[next_state] = [self._initial_q_value] * self.action_size

        # 現在の状態・行動に対するQ値
        current_q_value = self.q_table[state][action]

        # 次の状態での最大Q値 (Q学習)
        # エピソードが完了した場合は次の状態の価値は0
        max_next_q_value = 0.0
        if not done:
            max_next_q_value = max(self.q_table[next_state])

        # Q学習の更新式: Q(s, a) = Q(s, a) + α * [r + γ * max(Q(s', a')) - Q(s, a)]
        td_target = reward + self.gamma * max_next_q_value
        td_delta = td_target - current_q_value

        # Q値の更新
        self.q_table[state][action] += self.lr * td_delta

        return abs(td_delta) # TD誤差の絶対値を返す (損失の目安として)

    def save_q_table(self, file_path: str) -> None:
        """
        Qテーブルをファイルに保存する (pickle形式).
        """
        os.makedirs(os.path.dirname(file_path), exist_ok=True)
        try:
            with open(file_path, 'wb') as f:
                pickle.dump(self.q_table, f)
            print(f"Qテーブルを {file_path} に保存しました.")
        except Exception as e:
            print(f"Qテーブルの保存中にエラーが発生しました: {e}")

    def load_q_table(self, file_path: str) -> None:
        """
        ファイルからQテーブルをロードする (pickle形式).
        """
        if not os.path.exists(file_path):
            print(f"指定されたQテーブルファイルが見つかりません: {file_path}")
            return

        try:
            with open(file_path, 'rb') as f:
                self.q_table = pickle.load(f)
            print(f"Qテーブルを {file_path} からロードしました.")
        except Exception as e:
            print(f"Qテーブルのロード中にエラーが発生しました: {e}")

    def get_q_table_size(self) -> int:
        """
        Qテーブルに登録されている状態の数を返す.
        """
        return len(self.q_table)

    def get_q_values(self, state: QState) -> List[float]:
        """
        指定された状態のQ値リストを取得する.
        状態がQテーブルに存在しない場合は、初期化されたQ値リストを返す.

        Args:
            state (QState): 状態のQテーブル用表現 (フラット化タプル).

        Returns:
            List[float]: その状態における各行動のQ値リスト.
        """
        if state not in self.q_table:
            # 状態が存在しない場合は初期化して返す
            self.q_table[state] = [self._initial_q_value] * self.action_size
        return self.q_table[state]

    def get_max_q_value(self, state: QState) -> float:
        """
        指定された状態の最大Q値を取得する.
        状態がQテーブルに存在しない場合は、初期化されたQ値リストから最大値(通常0.0)を返す.

        Args:
            state (QState): 状態のQテーブル用表現 (フラット化タプル).

        Returns:
            float: その状態における最大Q値.
        """
        return max(self.get_q_values(state))
