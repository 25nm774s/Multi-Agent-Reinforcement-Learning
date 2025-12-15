# Define the MaskedActionSelection and MaskedQLearning strategy classes

# Assuming QState is defined (it is in previous cells)
import numpy as np
from typing import List, Tuple
#QState = Tuple[int, ...]
from Q_learn.QTable import QTable, QState

from Q_learn.strategys.action_selection import ActionSelectionStrategy
from Q_learn.strategys.learning import LearningStrategy

class CooperativeActionSelection(ActionSelectionStrategy):
    """
    協調行動の選択に関する具体的な戦略（他のエージェントの位置を考慮）
    """

    UNOBSERVED_POSITION = (-1, -1)

    def __init__(self, grid_size: int, goals_num: int, agent_id: int, total_agents: int):
        """
        Initializes the CooperativeActionSelection strategy.
        """
        self.grid_size = grid_size
        self.goals_num = goals_num
        self.agent_id = agent_id
        self.total_agents = total_agents

    def select_action(self, q_table: QTable, q_state: QState, action_size: int, epsilon: float) -> int:
        """
        epsilon-greedyを使用してアクションを選択
        """
        if np.random.rand() < epsilon:
            return np.random.choice(action_size)
        else:
            q_values = q_table.get_q_values(q_state)
            if not q_values:
                return np.random.choice(action_size)
            max_q = max(q_values)
            best_actions = [a for a, q in enumerate(q_values) if q == max_q]
            return np.random.choice(best_actions)

    def get_q_state_representation(self, global_state: Tuple[Tuple[int, int], ...], neighbor_distance:int) -> QState:
        """
        協調戦略の状態表現（すべての目標位置 + すべてのエージェント位置）を生成
        """

        goal_positions = global_state[:self.goals_num]
        agent_positions = global_state[self.goals_num:]
        agent_position = agent_positions[self.agent_id]

        # CooperativeActionSelection では、__init__ で受け取った neighbor_distance を使う可能性があるため、
        # Agent クラスからこの情報をストラテジーに渡す必要があります。
        # もしくは、CooperativeActionSelection が Agent インスタンスへの参照を持つように設計します。
        # 部分観測ロジックは、このメソッド内に移動させます。
        flat_state_list: List[int] = []

        # ゴールはそのまま
        for p in goal_positions:
            flat_state_list.extend(p)

        for i,pos in enumerate(agent_positions):
            if i==self.agent_id:
                flat_state_list.extend(pos)
                continue    #自身を除く

            d = max(abs(pos[0] - agent_position[0]), abs(pos[1] - agent_position[1]))

            if d <= neighbor_distance:
                flat_state_list.extend(pos)
            else:
                flat_state_list.extend(self.UNOBSERVED_POSITION)

        return tuple(flat_state_list)

#print("Strategy classes (SelfishQLearning with __init__ fix) redefined.")

class CooperativeQLearning(LearningStrategy):
    """
    協調的 Q 学習の更新に関する具体的な戦略 (mask=0)。

    この戦略は、将来的な協調学習の更新のためのプレースホルダーです。
    これは、マスク=0 のときに利用可能な完全な状態情報 (他のエージェントの位置を含む)
    を利用して、協調的な更新や他のエージェントのQ値の考慮(CTDE 設定など)を行うものになるかもしれません。
    現時点では、標準の QTable.learn に委任していますが、必要に応じて協調ロジックを組み込む準備は整っています。
    """
    def __init__(self, grid_size: int, goals_num: int, agent_id: int, total_agents: int):
        """
        Initializes the CooperativeQLearning strategy.

        Args:
            grid_size (int): The size of the grid.
            goals_num (int): The number of goals.
            agent_id (int): The ID of the agent using this strategy.
            total_agents (int): The total number of agents in the environment.
        """
        self.grid_size = grid_size
        self.goals_num = goals_num
        self.agent_id = agent_id
        self.total_agents = total_agents
        # Add other necessary parameters for future cooperative logic if needed
        # e.g., a reference to a shared model or a way to access other agents' Q-values

    def update_q_value(self, q_table: QTable, state: QState, action: int, reward: float, next_state: QState, done: bool) -> float:
        """
        Q 学習の更新を実行し、協力的なロジックを組み込む可能性もあります。
        マスク=0 のときに利用可能な完全な状態情報（他のエージェントの位置を含む）
        を利用します。

        これはプレースホルダーの実装です。実際の協力的なロジック（例えば、衝突による禁止されたアクションに基づく更新の防止、
        他のエージェントとの更新の調整など）は、ここで実装されます。
        """
        # --- 協調学習ロジックプレースホルダー ---
        # 実際の協調戦略では、更新ルールが変更される場合があります。
        # 現時点では、SelfishQLearning と同じですが、協調ロジックに対応しています。
        return q_table.learn(state, action, reward, next_state, done)

#print("Cooperative action selection and learning strategies defined as placeholders (renamed).")