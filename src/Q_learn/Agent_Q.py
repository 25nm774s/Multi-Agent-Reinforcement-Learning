from typing import Tuple, List

# 例: (goal1_x, goal1_y, ..., goalG_x, goalG_y, agent_i_x, agent_i_y, ..., agent_N_x, agent_N_y)
from Q_learn.QTable import QState, QTableType

from Q_learn.QTable import QTable
from Q_learn.strategys.action_selection import SelfishActionSelection, ActionSelectionStrategy
from Q_learn.strategys.learning import SelfishQLearning, LearningStrategy
from Q_learn.strategys.masked_strategies import CooperativeActionSelection, CooperativeQLearning

from Enviroments.Grid import PositionType

MAX_EPSILON = 1.0
MIN_EPSILON = 0.05

class Agent:
    """
    エージェント個別のロジックを管理するクラス.
    QTableインスタンスを持ち、行動選択、ε-greedy、ε減衰、学習プロセスを担う.
    ストラテジーパターンを使用して行動選択と学習ロジックをカプセル化する.
    状態表現の生成はActionSelectionStrategyに委譲される.
    """
    def __init__(self, args, agent_id: int):
        """
        Agent コンストラクタ.

        Args:
            args: 環境設定を含むオブジェクト (mask属性を含む).
            agent_id (int): このエージェントのID.
        """
        self.agent_id:int           = agent_id
        self.grid_size:int          = args.grid_size
        self.goals_num:int          = args.goals_number
        self.action_size:int        = 5 # UP, DOWN, LEFT, RIGHT, STAY
        self.total_agents:int       = args.agents_number
        self.mask:bool              = args.mask
        # self.observation_mode:str   = args.observation_mode
        self.neighbor_distance:int  = args.neighbor_distance

        # マスク値に基づいて戦略をインスタンス化
        # 戦略に必要な初期化引数を渡す
        if args.mask == 0:
            # mask==0: 協調モード (他のエージェントを考慮する)
            self._action_selection_strategy: ActionSelectionStrategy = CooperativeActionSelection(
                grid_size=self.grid_size,
                goals_num=self.goals_num,
                agent_id=self.agent_id,
                total_agents=self.total_agents
            )
            self._learning_strategy: LearningStrategy = CooperativeQLearning(
                grid_size=self.grid_size,
                goals_num=self.goals_num,
                agent_id=self.agent_id,
                total_agents=self.total_agents
            )
            print(f"Agent {self.agent_id}: Using Cooperative Strategies (mask=0)")
        else:
            # mask==1: 利己的モード（他のエージェントを無視する）
            self._action_selection_strategy: ActionSelectionStrategy = SelfishActionSelection(
                 grid_size=self.grid_size,
                 goals_num=self.goals_num,
                 agent_id=self.agent_id,
                 total_agents=self.total_agents
            )
            self._learning_strategy: LearningStrategy = SelfishQLearning(
                 grid_size=self.grid_size,
                 goals_num=self.goals_num,
                 agent_id=self.agent_id,
                 total_agents=self.total_agents
            )
            print(f"Agent {self.agent_id}: Using Selfish Strategies (mask=1)")

        # QTable Instance
        self.q_table = QTable(
            action_size=self.action_size,
            learning_rate=getattr(args, 'learning_rate', 0.1),
            discount_factor=getattr(args, 'discount_factor', 0.99)
        )

        # ε-greedyのためのパラメータを保持
        self.epsilon = getattr(args, 'epsilon', 1.0)
        self.min_epsilon = getattr(args, 'min_epsilon', 0.01)
        self.max_epsilon = getattr(args, 'max_epsilon', 1.0)
        self.epsilon_decay = args.epsilon_decay #getattr(args, 'epsilon_decay_alpha', 0.70)


    #def _get_q_state(self, global_state: Tuple[Tuple[int, int], ...]) -> QState:
    def _get_q_state(self, global_state: Tuple[PositionType, ...]) -> QState:
        """
        環境の全体状態から、このエージェントにとってのQテーブル用の状態表現を抽出・生成する.
        現在の config (mask, observation_mode) に応じて、異なる状態表現を生成する.
        """
        # global_state の構造: ((g1_x, g1_y), ..., (a1_x, a1_y), ..., (aN_x, aN_y))
        return self._action_selection_strategy.get_q_state_representation(
            global_state,
            self.neighbor_distance
        )


    #def get_action(self, global_state: Tuple[Tuple[int, int], ...]) -> int:
    def get_action(self, global_state: Tuple[PositionType, ...]) -> int:
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
        return self._action_selection_strategy.select_action(
            self.q_table,      # QTableインスタンス
            q_state,           # 現在の状態 (ストラテジーによって内容が異なる)
            self.action_size,  # 行動空間サイズ
            self.epsilon       # ε値
        )


    def decay_epsilon_power(self, step: int):
        """
        ステップ数に基づき、探索率εを指数的に減衰させる関数。
        Args:
            step (int): 現在のステップ数（またはエピソード数）。
        """
        lambda_ = 0.00001
        # 指数減衰式: ε_t = ε_start * (decay_rate)^t
        # self.epsilon = MAX_EPSILON * (self.epsilon_decay ** (step*lambda_))
        self.epsilon *= MAX_EPSILON * (self.epsilon_decay ** (lambda_))

        # 最小値（例: 0.01）を下回らないようにすることが多いが、ここではシンプルな式のみを返します。
        self.epsilon = max(MIN_EPSILON, self.epsilon)


    #def learn(self, global_state: Tuple[Tuple[int, int], ...], action: int, reward: float, next_global_state: Tuple[Tuple[int, int], ...], done: bool) -> float:
    def learn(self, global_state: Tuple[PositionType, ...], action: int, reward: float, next_global_state: Tuple[PositionType, ...], done: bool) -> float:
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
        # 現在および次のエージェント固有の状態表現を取得 (ストラテジーに委譲)
        current_q_state = self._get_q_state(global_state)
        next_q_state = self._get_q_state(next_global_state)

        if not isinstance(action, int): 
            raise TypeError(f'actionはint型を期待しますが、実際は{type(action)}でした。')
        
        # 学習ロジックをストラテジーオブジェクトに委譲
        td_delta = self._learning_strategy.update_q_value(
            self.q_table,
            current_q_state,
            action,
            reward,
            next_q_state,
            done
        )

        return td_delta


    def get_Qtable(self) -> QTableType:
        """
        このエージェントのQテーブルをファイルに保存する.
        Agentに紐づけられたmodel_pathを使用する.
        (Same as before - Delegate to QTable object)
        """
        return self.q_table.get_Qtable()
    
    def set_Qtable(self, q_table:QTableType):
        self.q_table.set_Qtable(q_table)

    def get_q_table_size(self) -> int:
        """
        このエージェントのQテーブルに登録されている状態の数を返す.
        (Same as before - Delegate to QTable object)
        """
        return self.q_table.get_q_table_size()

