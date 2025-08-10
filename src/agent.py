# Update the Agent class __init__ method to select strategies based on 'mask'

from typing import Tuple, List

# 例: (goal1_x, goal1_y, ..., goalG_x, goalG_y, agent_i_x, agent_i_y, ..., agent_N_x, agent_N_y)
from Q_learn.QTable import QState, QTableType

from Q_learn.QTable import QTable
from Q_learn.strategys.action_selection import SelfishActionSelection
from Q_learn.strategys.learning import SelfishQLearning
from Q_learn.strategys.masked_strategies import CooperativeActionSelection, CooperativeQLearning

class Agent:
    """
    エージェント個別のロジックを管理するクラス.
    QTableインスタンスを持ち、行動選択、ε-greedy、ε減衰、学習プロセスを担う.
    ストラテジーパターンを使用して行動選択と学習ロジックをカプセル化する.
    """
    def __init__(self, args, agent_id: int):
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

        # mask = 0 (協調的): 他のエージェントを考慮 -> Cooperative Strategies
        # mask = 1 (自己中心的): 他のエージェントを無視 -> Selfish Strategies
        if mask == 0:
            # mask=0: 他のエージェントを考慮するモード (協調的)
            # Cooperative Strategies クラスが、他のエージェントの位置を含む状態を扱うように設計されています。
            # 実際の協調ロジックは CooperativeActionSelection/CooperativeQLearning に実装されます。
            self._action_selection_strategy = CooperativeActionSelection(
                grid_size=self.grid_size,
                goals_num=self.goals_num,
                agent_id=self.agent_id,
                total_agents=self.total_agents
            )
            self._learning_strategy = CooperativeQLearning(
                grid_size=self.grid_size,
                goals_num=self.goals_num,
                agent_id=self.agent_id,
                total_agents=self.total_agents
            )
            print(f"Agent {self.agent_id}: Using Cooperative Strategies (mask=0)")
        else:
            # mask=1: 他のエージェントを考慮しないモード (自己中心的)
            # Selfish Strategies クラスが、自身の位置とゴールのみを状態として扱うように設計されています。
            self._action_selection_strategy = SelfishActionSelection()
            self._learning_strategy = SelfishQLearning()
            print(f"Agent {self.agent_id}: Using Selfish Strategies (mask=1)")

        # QTable Instance (shared state managed by the agent)
        self.q_table = QTable(
            action_size=self.action_size,
            learning_rate=getattr(args, 'learning_rate', 0.1),
            discount_factor=getattr(args, 'discount_factor', 0.99)
        )

        # ε-greedyのためのパラメータを保持
        self.epsilon = getattr(args, 'epsilon', 1.0)
        self.min_epsilon = getattr(args, 'min_epsilon', 0.01)
        self.max_epsilon = getattr(args, 'max_epsilon', 1.0)


    def _get_q_state(self, global_state: Tuple[Tuple[int, int], ...]) -> QState:
        """
        環境の全体状態から、このエージェントにとってのQテーブル用の状態表現を抽出・生成する.
        行動選択ストラテジーに応じて、他のエージェントの位置を含むか含まないかを決定する.
        """
        # global_state の構造: ((g1_x, g1_y), ..., (a1_x, a1_y), ...)
        goal_positions = global_state[:self.goals_num]
        agent_positions = global_state[self.goals_num:] # 全てのエージェント位置


        # 現在の行動選択ストラテジーが CooperativeActionSelection (mask=0) か SelfishActionSelection (mask=1) かによって状態表現を切り替える
        if isinstance(self._action_selection_strategy, CooperativeActionSelection):
            # Cooperative モード (mask=0): 全エージェント位置を状態に含める
            flat_state_list: List[int] = []
            for pos in goal_positions:
                 if not isinstance(pos, tuple) or len(pos) != 2:
                     raise ValueError(f"Unexpected goal position format: {pos}")
                 flat_state_list.extend(pos)

            for pos in agent_positions:
                 if not isinstance(pos, tuple) or len(pos) != 2:
                     raise ValueError(f"Unexpected agent position format: {pos}")
                 flat_state_list.extend(pos) # 全てのエージェント位置を追加

            return tuple(flat_state_list)

        elif isinstance(self._action_selection_strategy, SelfishActionSelection):
            # Selfish モード (mask=1): 自身の位置のみを状態に含める
            if self.goals_num + self.agent_id >= len(global_state):
                raise IndexError(f"Invalid agent_id {self.agent_id} or global_state structure.")

            agent_position = global_state[self.goals_num + self.agent_id] # そのエージェント自身の位置

            flat_state_list: List[int] = []
            for pos in goal_positions:
                 if not isinstance(pos, tuple) or len(pos) != 2:
                     raise ValueError(f"Unexpected goal position format: {pos}")
                 flat_state_list.extend(pos)

            if not isinstance(agent_position, tuple) or len(agent_position) != 2:
                 raise ValueError(f"Unexpected agent position format: {agent_position}")
            flat_state_list.extend(agent_position) # そのエージェント自身の位置を追加

            return tuple(flat_state_list)

        else:
            # 未知のストラテジーが設定されている場合
            raise TypeError(f"Unknown action selection strategy type: {type(self._action_selection_strategy)}")


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
        # _get_q_state メソッドが、現在設定されているストラテジーに基づいて適切な状態を生成する
        q_state = self._get_q_state(global_state)

        # 行動選択ロジックをストラテジーオブジェクトに委譲
        # ストラテジーにQTableインスタンス自体を渡すことで、ストラテジーはQTableのメソッドを使用できる
        # CooperativeActionSelection (mask=0時) は、必要に応じて global_state 全体や他のエージェント位置を
        # 内部ロジックで使用するために、それらの情報にアクセスする方法を持つか、引数として受け取る必要があるかもしれません。
        # 現状の select_action シグネチャでは global_state 全体は渡されていませんが、
        # _get_q_state で状態表現自体は切り替わっています。
        # より高度なマスクロジックでは、select_action メソッドのシグネチャ変更が必要になるかもしれません。
        return self._action_selection_strategy.select_action(
            self.q_table,      # QTableインスタンス
            q_state,           # 現在の状態 (ストラテジーによって内容が異なる)
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
        # _get_q_state メソッドが、現在設定されているストラテジーに基づいて適切な状態を生成する
        current_q_state = self._get_q_state(global_state)
        next_q_state = self._get_q_state(next_global_state)


        # 学習ロジックをストラテジーオブジェクトに委譲
        # ストラテジーにQTableインスタンス、状態、行動、報酬、次の状態、doneフラグを渡す
        # CooperativeQLearning (mask=0時) は、必要に応じて global_state 全体や他のエージェント位置を
        # 内部ロジックで使用するために、それらの情報にアクセスする方法を持つか、引数として受け取る必要があるかもしれません。
        # 現状の update_q_value シグネチャでは global_state 全体は渡されていませんが、
        # _get_q_state で状態表現自体は切り替わっています。
        # より高度なマスクロジックでは、update_q_value メソッドのシグネチャ変更が必要になるかもしれません。
        td_delta = self._learning_strategy.update_q_value(
            self.q_table,         # QTableインスタンス
            current_q_state,      # 現在の状態 (ストラテジーによって内容が異なる)
            action,               # 取られた行動
            reward,               # 報酬
            next_q_state,         # 次の状態 (ストラテジーによって内容が異なる)
            done                  # 完了フラグ
        )

        return td_delta


    def get_Qtable(self) -> QTableType:
        """
        このエージェントのQテーブルをファイルに保存する.
        Agentに紐づけられたmodel_pathを使用する.
        (Same as before - Delegate to QTable object)
        """
        return self.q_table.get_Qtable()
    
    def set_Qtable(self, q_table):
        self.q_table.set_Qtable(q_table)

    def get_q_table_size(self) -> int:
        """
        このエージェントのQテーブルに登録されている状態の数を返す.
        (Same as before - Delegate to QTable object)
        """
        return self.q_table.get_q_table_size()

