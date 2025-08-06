# Update the Agent class __init__ method to select strategies based on 'mask'

import os
from typing import Tuple, List

from Q_learn.QTable import QState, QTable
from Q_learn.strategys.action_selection import StandardActionSelection
from Q_learn.strategys.learning import StandardQLearning
from Q_learn.strategys.masked_strategies import MaskedQLearning, MaskedActionSelection

class Agent:
    """
    エージェント個別のロジックを管理するクラス.
    QTableインスタンスを持ち、行動選択、ε-greedy、ε減衰、学習プロセスを担う.
    ストラテジーパターンを使用して行動選択と学習ロジックをカプセル化する.
    """
    def __init__(self, args, agent_id: int): # Added mask argument with a default
        """
        Agent コンストラクタ.

        Args:
            args: 環境設定を含むオブジェクト (mask属性を含む).
            agent_id (int): このエージェントのID.
        """
        self.agent_id = agent_id
        self.grid_size = args.grid_size
        self.goals_num = args.goals_number
        self.action_size = 5 # UP, DOWN, LEFT, RIGHT, STAY
        self.total_agents = args.agents_number # Store total agents for masked strategy

        # Determine strategy based on args.mask
        mask = getattr(args, 'mask', 0) # Get mask value, default to 0 if not present

        if mask == 1:
            # Use Masked Strategies
            self._action_selection_strategy = MaskedActionSelection(
                grid_size=self.grid_size,
                goals_num=self.goals_num,
                agent_id=self.agent_id,
                total_agents=self.total_agents
            )
            self._learning_strategy = MaskedQLearning(
                grid_size=self.grid_size,
                goals_num=self.goals_num,
                agent_id=self.agent_id,
                total_agents=self.total_agents
            )
            print(f"Agent {self.agent_id}: Using Masked Strategies")
        else:
            # Use Standard Strategies
            self._action_selection_strategy = StandardActionSelection()
            self._learning_strategy = StandardQLearning()
            print(f"Agent {self.agent_id}: Using Standard Strategies")


        # Model path for QTable remains the same
        save_dir = getattr(args, 'dir_path', 'models')
        self.model_path = os.path.join(save_dir, f'agent_{self.agent_id}_q_table.pkl')

        # QTable Instance (shared state managed by the agent)
        self.q_table = QTable(
            action_size=self.action_size,
            learning_rate=getattr(args, 'learning_rate', 0.1),
            discount_factor=getattr(args, 'discount_factor', 0.99),
            load_model=args.load_model,
            model_path=self.model_path
        )

        # ε-greedyのためのパラメータを保持
        self.epsilon = getattr(args, 'epsilon', 1.0)
        self.min_epsilon = getattr(args, 'min_epsilon', 0.01)
        self.max_epsilon = getattr(args, 'max_epsilon', 1.0)


    def _get_q_state(self, global_state: Tuple[Tuple[int, int], ...]) -> QState:
        """
        環境の全体状態から、このエージェントにとってのQテーブル用の状態表現を抽出・生成する.
        (Same as before)
        """
        # global_state の構造: ((g1_x, g1_y), ..., (a1_x, a1_y), ...)
        goal_positions = global_state[:self.goals_num]

        if self.goals_num + self.agent_id >= len(global_state):
            raise IndexError(f"Invalid agent_id {self.agent_id} or global_state structure.")

        agent_position = global_state[self.goals_num + self.agent_id]

        flat_state_list: List[int] = []
        for pos in goal_positions:
            if not isinstance(pos, tuple) or len(pos) != 2:
                raise ValueError(f"Unexpected goal position format: {pos}")
            flat_state_list.extend(pos)

        if not isinstance(agent_position, tuple) or len(agent_position) != 2:
            raise ValueError(f"Unexpected agent position format: {agent_position}")
        flat_state_list.extend(agent_position)

        return tuple(flat_state_list)


    def get_action(self, global_state: Tuple[Tuple[int, int], ...]) -> int:
        """
        現在の全体状態に基づいて、エージェントの行動を決定する.
        行動選択ロジックは ActionSelectionStrategy オブジェクトに委譲される.

        Args:
            global_state (Tuple[Tuple[int, int], ...]): 環境の現在の全体状態タプル.

        Returns:
            int: 選択された行動 (0:UP, 1:DOWN, 2:LEFT, 3:RIGHT, 4:STAY).
        """
        # Qテーブル用の状態表現を取得
        q_state = self._get_q_state(global_state)

        # 行動選択ロジックをストラテジーオブジェクトに委譲
        # ストラテジーにQTableインスタンス自体を渡すことで、ストラテジーはQTableのメソッドを使用できる
        return self._action_selection_strategy.select_action(
            self.q_table,      # QTableインスタンス
            q_state,           # 現在の状態
            self.action_size,  # 行動空間サイズ
            self.epsilon       # ε値
        )


    def decay_epsilon_pow(self, step:int, alpha=0.90):
        """
        ステップ数に基づいてεをべき乗減衰させる.
        (Same as before)
        """
        effect_step = max(1,step)
        if alpha >= 1.0 or alpha <= 0.0:
            pass

        self.epsilon = self.max_epsilon * (1.0 / effect_step**alpha)
        self.epsilon = max(self.epsilon, self.min_epsilon)


    def learn(self, global_state: Tuple[Tuple[int, int], ...], action: int, reward: float, next_global_state: Tuple[Tuple[int, int], ...], done: bool) -> float:
        """
        単一の経験に基づいてQテーブルを更新するプロセスをAgentが管理する.
        学習ロジックは LearningStrategy オブジェクトに委譲される.

        Args:
            global_state (Tuple[Tuple[int, int], ...]): 経験の現在の全体状態.
            action (int): エージェントが取った行動.
            reward (float): 行動によって得られた報酬.
            next_global_state (Tuple[Tuple[int, int], ...]): 経験の次の全体状態.
            done (bool): エピソードが完了したかどうかのフラグ.

        Returns:
            float: 更新に使用されたTD誤差の絶対値 (LearningStrategyから返される).
        """
        # エージェント固有の状態表現を取得
        current_q_state = self._get_q_state(global_state)
        next_q_state = self._get_q_state(next_global_state)

        # 学習ロジックをストラテジーオブジェクトに委譲
        # ストラテジーにQTableインスタンス、状態、行動、報酬、次の状態、doneフラグを渡す
        td_delta = self._learning_strategy.update_q_value(
            self.q_table,         # QTableインスタンス
            current_q_state,      # 現在の状態
            action,               # 取られた行動
            reward,               # 報酬
            next_q_state,         # 次の状態
            done                  # 完了フラグ
        )

        return td_delta


    def save_q_table(self) -> None:
        """
        このエージェントのQテーブルをファイルに保存する.
        Agentに紐づけられたmodel_pathを使用する.
        (Same as before - QTable object handles saving)
        """
        self.q_table.save_q_table(self.model_path)

    def get_q_table_size(self) -> int:
        """
        このエージェントのQテーブルに登録されている状態の数を返す.
        (Same as before - Delegate to QTable object)
        """
        return self.q_table.get_q_table_size()
