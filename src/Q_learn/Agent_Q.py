from typing import Any, Tuple, List

from Strategy.StateRepresentationStrategy import StateRepresentationStrategy

from Strategy.CooperativeStateRepresentation import CooperativeStateRepresentation
from Strategy.SelfishStateRepresentation import SelfishStateRepresentation
from Base.Agent_Base import AgentBase

from .strategys.SelfishStrategy import SelfishStrategy
from .strategys.CooperativeStrategy import CooperativeStrategy
from .QTable import QTable
from Base.Constant import GlobalState, QTableType, QState

# 例: (goal1_x, goal1_y, ..., goalG_x, goalG_y, agent_i_x, agent_i_y, ..., agent_N_x, agent_N_y)
QState = Tuple[int,...]
PositionType = Tuple[int,int]

MAX_EPSILON = 1.0
MIN_EPSILON = 0.05

class Agent(AgentBase):
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
        super().__init__(agent_id, args)

        self.observe_store:dict = {}

        self._state_representation_strategy = self._get_strategy(args.mask)

        # 現環境全く同じなので代表してSelfishStrategyを使用
        self._use_strategy = SelfishStrategy(self.grid_size, self.goals_num, self.agent_id, self.total_agents)
        
        # QTable Instance
        self.q_table = QTable(
            action_size=self.action_size,
            learning_rate=getattr(args, 'learning_rate', 0.1),
            discount_factor=getattr(args, 'discount_factor', 0.99)
        )

        # ε-greedyのためのパラメータを保持
        self.min_epsilon = getattr(args, 'min_epsilon', 0.01)
        self.max_epsilon = getattr(args, 'max_epsilon', 1.0)

    def _get_strategy(self, mask) -> StateRepresentationStrategy:
        if mask == 0:
            # mask==0: 協調モード (他のエージェントを考慮する)
            action_selection_strategy: StateRepresentationStrategy = CooperativeStateRepresentation(
                grid_size=self.grid_size,
                goals_num=self.goals_num,
                agent_id=self.agent_id,
                total_agents=self.total_agents
            )
            print(f"Agent {self.agent_id}: Using Cooperative Strategies (mask=0)")
        else:
            # mask==1: 利己的モード（他のエージェントを無視する）
            action_selection_strategy: StateRepresentationStrategy = SelfishStateRepresentation(
                grid_size=self.grid_size,
                goals_num=self.goals_num,
                agent_id=self.agent_id,
                total_agents=self.total_agents
            )
            print(f"Agent {self.agent_id}: Using Selfish Strategies (mask=1)")
        return action_selection_strategy

    #def _get_q_state(self, global_state: Tuple[Tuple[int, int], ...]) -> QState:
    def _get_q_state(self, global_state: GlobalState) -> QState:
        """
        環境の全体状態から、このエージェントにとってのQテーブル用の状態表現を抽出・生成する.
        現在の config (mask, observation_mode) に応じて、異なる状態表現を生成する.
        """
        # global_state の構造: ((g1_x, g1_y), ..., (a1_x, a1_y), ..., (aN_x, aN_y))
        r = self._state_representation_strategy.get_q_state_representation(
            global_state,
            self.neighbor_distance
        )
        return self._flatten_state(r)

    def _flatten_state(self, state: GlobalState) -> QState:
        """
        ((x, y), (x, y)) -> (x, y, x, y) にフラット化する
        """
        # 座標タプルを一つずつ展開して平坦なリストにし、最後にタプル化
        return tuple(val for pos in state for val in pos)


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
        return self._use_strategy.select_action(
            self.q_table,      # QTableインスタンス
            q_state,           # 現在の状態 (ストラテジーによって内容が異なる)
            self.action_size,  # 行動空間サイズ
            self.epsilon       # ε値
        )

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
        td_delta = self._use_strategy.update_q_value(
            self.q_table,
            current_q_state,
            action,
            reward,
            next_q_state,
            done
        )

        return td_delta


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

    def get_all_q_values(self, global_state: Tuple[PositionType,...]) -> list[float]:
        qs = self._get_q_state(global_state)
        return self.q_table.get_q_values(qs)
