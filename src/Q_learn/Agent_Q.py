from typing import Tuple
import numpy as np

# from Base.Agent_Base import BaseMasterAgent

from .QTable import QTable
from Base.Constant import GlobalState, QTableType, QState

# 例: (goal1_x, goal1_y, ..., goalG_x, goalG_y, agent_i_x, agent_i_y, ..., agent_N_x, agent_N_y)

MAX_EPSILON = 1.0
MIN_EPSILON = 0.05

class Agent:
    """
    エージェント個別のロジックを管理するクラス.
    QTableインスタンスを持ち、行動選択、ε-greedy、ε減衰、学習プロセスを担う.
    ストラテジーパターンを使用して行動選択と学習ロジックをカプセル化する.
    状態表現の生成はActionSelectionStrategyに委譲される.
    """
    def __init__(self, args, id):
        """
        Agent コンストラクタ.

        Args:
            args: 環境設定を含むオブジェクト (mask属性を含む).
            agent_id (int): このエージェントのID.
        """
        self.action_size = 5
        self.goals_number = args.goals_number
        self.agents_number = args.agents_number
        self.id = id

        self.epsilon = MAX_EPSILON

        self.observe_store:dict = {}

        # QTable Instance
        lr = getattr(args, 'learning_rate', 0.1)
        discount_factor = getattr(args, 'discount_factor', 0.99)

        self.q_table = QTable(
            action_size=self.action_size,
            learning_rate=lr,
            discount_factor=discount_factor
        )

        # ε-greedyのためのパラメータを保持
        self.min_epsilon = getattr(args, 'min_epsilon', 0.01)
        self.max_epsilon = getattr(args, 'max_epsilon', 1.0)

    def _get_q_state(self, global_state, debug=False) -> QState:
        """
        環境の全体状態から、このエージェントにとってのQテーブル用の状態表現を抽出・生成する.
        現在の config (mask, observation_mode) に応じて、異なる状態表現を生成する.
        """
        self_id = f"agent_{self.id}"
        self_state:dict = global_state[self_id]
        r = []

        for pos in self_state["all_goals"]:
            r.append(pos)

        r.append(self_state['self'])

        others:dict[str, tuple[int,int]] = self_state['others']
        for k in others:
            r.append(others[k])

        if debug:
            print(f"--- agent_{self.id} ---")
            print("\tgs-raw: ", global_state)
            print("\tothers: ", self_state['others'])
            print("\tr:",r)

        return self._flatten_state(tuple(r))

    def _flatten_state(self, state: Tuple[Tuple[int,int],...]) -> QState:
        """
        ((x, y), (x, y)) -> (x, y, x, y) にフラット化する
        """
        # 座標タプルを一つずつ展開して平坦なリストにし、最後にタプル化
        return tuple(val for pos in state for val in pos)

    def select_action_greedy(self, q_table: QTable, q_state: QState, action_size: int, epsilon: float) -> int:
        if np.random.rand() < epsilon:
            return np.random.choice(action_size)
        else:
            q_values = q_table.get_q_values(q_state)
            if not q_values:
                return np.random.choice(action_size)
            max_q = max(q_values)
            best_actions = [a for a, q in enumerate(q_values) if q == max_q]
            return np.random.choice(best_actions)


    def get_action(self, global_state: GlobalState) -> int:
        """
        現在の全体状態に基づいて、エージェントの行動を決定する.
        行動選択ロジックは ActionSelectionStrategy オブジェクトに委譲される.

        Args:
            global_state (Tuple[Tuple[int, int], ...]): 環境の現在の全体状態タプル.

        Returns:
            int: 選択された行動 (0:UP, 1:DOWN, 2:LEFT, 3:RIGHT, 4:STAY).
        """
        # Qテーブル用の状態表現を取得 (ストラテジーに委譲)
        q_state = self._get_q_state(global_state)

        # 行動選択ロジックをストラテジーオブジェクトに委譲
        return self.select_action_greedy(self.q_table, q_state, self.action_size, self.epsilon)

    def observe(self, global_state: GlobalState, action: int, reward: float, next_global_state: GlobalState, done: bool) -> None:
        self.observe_store = {
            "global_state": global_state,
            "action": action,
            "reward": reward,
            "next_global_state": next_global_state,
            "done": done
        }

    def learn(self, total_step=None) -> float:
        """
        単一の経験に基づいてQテーブルを更新するプロセスをAgentが管理する.
        学習ロジックは LearningStrategy オブジェクトに委譲される.

        Args:

        Returns:
            float: 更新に使用されたTD誤差の絶対値 (LearningStrategyから返される).
        """
        # 現在および次のエージェント固有の状態表現を取得 (ストラテジーに委譲)
        global_state: GlobalState       = self.observe_store['global_state']
        next_global_state: GlobalState  = self.observe_store['next_global_state']
        action: int                     = self.observe_store['action']
        reward: float                   = self.observe_store['reward']
        done: bool                      = self.observe_store['done']

        current_q_state = self._get_q_state(global_state)
        next_q_state = self._get_q_state(next_global_state)

        if not isinstance(action, int):
            raise TypeError(f'actionはint型を期待しますが、実際は{type(action)}でした。')

        # 学習ロジックをストラテジーオブジェクトに委譲
        td_delta = self.q_table.learn(current_q_state, action, reward, next_q_state, done)
        return td_delta

    def decay_epsilon_power(self):
        self.epsilon *= 0.9999
        self.epsilon = max(self.min_epsilon, self.epsilon)

    def get_weights(self) -> QTableType:
        """
        このエージェントのQテーブルをファイルに保存する.
        Agentに紐づけられたmodel_pathを使用する.
        (Same as before - Delegate to QTable object)
        """
        return self.q_table.get_Qtable()

    def set_weights(self, q_table:QTableType):
        self.q_table.set_Qtable(q_table)

    def get_q_table_size(self) -> int:
        """
        このエージェントのQテーブルに登録されている状態の数を返す.
        (Same as before - Delegate to QTable object)
        """
        return self.q_table.get_q_table_size()

    def get_all_q_values(self, global_state: GlobalState) -> list[float]:
        qs = self._get_q_state(global_state)
        return self.q_table.get_q_values(qs)
