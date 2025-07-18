import numpy as np
import os
from typing import Tuple, List

from Q_learn.QTable import QTable
# QState 定義 (エージェントクラスのメソッドの型ヒントに必要)
# 例: (goal1_x, goal1_y, ..., goalG_x, goalG_y, agent_i_x, agent_i_y)
QState = Tuple[int, ...]

class Agent:
    """
    エージェント個別のロジックを管理するクラス.
    QTableインスタンスを持ち、行動選択、ε-greedy、ε減衰、学習プロセスを担う.
    """
    def __init__(self, args, agent_id: int):
        """
        Agent コンストラクタ.

        Args:
            args: 環境設定を含むオブジェクト.
            agent_id (int): このエージェントのID.
        """
        self.agent_id = agent_id
        self.grid_size = args.grid_size
        self.goals_num = args.goals_number
        # 行動空間サイズはグリッドワールドの仕様に依存 (UP, DOWN, LEFT, RIGHT, STAY)
        self.action_size = 5

        # モデルパスはエージェントIDごとに異なる可能性を考慮し、ディレクトリパスとIDから生成
        save_dir = getattr(args, 'dir_path', 'models') # argsにdir_pathがない場合のデフォルト
        self.model_path = os.path.join(save_dir, f'agent_{self.agent_id}_q_table.pkl')

        # QTable インスタンスを生成し、保持
        # QTable コンストラクタに必要な引数を渡す
        self.q_table = QTable(
            action_size=self.action_size,
            learning_rate=getattr(args, 'learning_rate', 0.1),
            discount_factor=getattr(args, 'discount_factor', 0.99),
            load_model=args.load_model, # argsからload_modelフラグを渡す
            model_path=self.model_path  # エージェント固有のモデルパスを渡す
        )

        # ε-greedyのためのパラメータを保持
        self.epsilon = getattr(args, 'epsilon', 1.0)
        self.min_epsilon = getattr(args, 'min_epsilon', 0.01)
        self.max_epsilon = getattr(args, 'max_epsilon', 1.0)


    def _get_q_state(self, global_state: Tuple[Tuple[int, int], ...]) -> QState:
        """
        環境の全体状態から、このエージェントにとってのQテーブル用の状態表現を抽出・生成する.
        仕様: (goal1_x, goal1_y, ..., goalG_x, goalG_y, agent_id_x, agent_id_y) のフラット化タプル.

        Args:
            global_state (Tuple[Tuple[int, int], ...]): 環境の現在の全体状態タプル
                                                        (ゴール位置タプル..., エージェント位置タプル...).

        Returns:
            QState: Qテーブルのキーとして使用するフラット化タプル形式の状態.
        """
        # global_state の構造: ((g1_x, g1_y), ..., (a1_x, a1_y), ...)
        # ゴール位置タプルのリスト
        goal_positions = global_state[:self.goals_num]
        # このエージェントの位置タプル
        # global_stateはゴール位置 + エージェント位置の順に並んでいると仮定
        # 自身のエージェント位置は goals_num + agent_id のインデックスにある
        if self.goals_num + self.agent_id >= len(global_state):
            # エージェントIDが不正な場合や global_state の構造が想定外の場合
            raise IndexError(f"Invalid agent_id {self.agent_id} or global_state structure.")

        agent_position = global_state[self.goals_num + self.agent_id]

        # フラット化タプルを作成
        flat_state_list: List[int] = []
        for pos in goal_positions:
            if not isinstance(pos, tuple) or len(pos) != 2:
                raise ValueError(f"Unexpected goal position format: {pos}")
            flat_state_list.extend(pos) # ゴール座標を追加

        if not isinstance(agent_position, tuple) or len(agent_position) != 2:
            raise ValueError(f"Unexpected agent position format: {agent_position}")
        flat_state_list.extend(agent_position) # 自身の座標を追加

        return tuple(flat_state_list) # タプルに変換して返す


    def get_action(self, global_state: Tuple[Tuple[int, int], ...]) -> int:
        """
        現在の全体状態に基づいて、エージェントの行動を決定する (ε-greedy).

        Args:
            global_state (Tuple[Tuple[int, int], ...]): 環境の現在の全体状態タプル.

        Returns:
            int: 選択された行動 (0:UP, 1:DOWN, 2:LEFT, 3:RIGHT, 4:STAY).
        """
        # Qテーブル用の状態表現を取得
        q_state = self._get_q_state(global_state)

        # ε-greedyに基づいて行動を選択
        if np.random.rand() < self.epsilon:
            # 探索: ランダムに行動を選択
            return np.random.choice(self.action_size)
        else:
            # 活用: Q値に基づいて最適な行動を選択
            # AgentはQTableインスタンスのget_q_valuesメソッドを使用
            q_values = self.q_table.get_q_values(q_state)

            # 最大Q値を持つ行動を選択 (複数の場合ランダムに1つ)
            max_q = max(q_values)
            best_actions = [a for a, q in enumerate(q_values) if q == max_q]
            return np.random.choice(best_actions)

    def decay_epsilon_pow(self, step:int, alpha=0.90):
        """
        ステップ数に基づいてεをべき乗減衰させる.

        Args:
            step (int): 現在の総ステップ数.
            alpha (float): 減衰の度合いを制御するパラメータ (0 < alpha < 1).
        """
        effect_step = max(1,step)
        if alpha >= 1.0 or alpha <= 0.0:
            # エラーではなく警告またはログ出力にする方が運用しやすいかもしれない
            # print(f"Warning: alpha({alpha}) should be between 0 and 1.")
            pass # 一旦エラーチェックは残すが、実運用では調整が必要

        # べき乗減衰式を使用してイプシロンを更新
        # self.max_epsilon, self.min_epsilon は __init__ で args から取得済み
        self.epsilon = self.max_epsilon * (1.0 / effect_step**alpha)
        # イプシロンがmin_epsilonを下回らないように保証
        self.epsilon = max(self.epsilon, self.min_epsilon)


    def learn(self, global_state: Tuple[Tuple[int, int], ...], action: int, reward: float, next_global_state: Tuple[Tuple[int, int], ...], done: bool) -> float:
        """
        単一の経験に基づいてQテーブルを更新するプロセスをAgentが管理する.

        Args:
            global_state (Tuple[Tuple[int, int], ...]): 経験の現在の全体状態.
            action (int): エージェントが取った行動.
            reward (float): 行動によって得られた報酬.
            next_global_state (Tuple[Tuple[int, int], ...]): 経験の次の全体状態.
            done (bool): エピソードが完了したかどうかのフラグ.

        Returns:
            float: 更新に使用されたTD誤差の絶対値 (QTableから返される).
        """
        # エージェント固有の状態表現を取得
        current_q_state = self._get_q_state(global_state)
        next_q_state = self._get_q_state(next_global_state)

        # QTableインスタンスのlearnメソッドを呼び出して学習を実行
        # Agentは自分自身の状態表現と行動、報酬、次の状態表現、doneフラグをQTableに渡す
        td_delta = self.q_table.learn(current_q_state, action, reward, next_q_state, done)

        return td_delta # QTableから返されたTD誤差をそのまま返す


    def save_q_table(self) -> None:
        """
        このエージェントのQテーブルをファイルに保存する.
        Agentに紐づけられたmodel_pathを使用する.
        """
        # Agentインスタンスは自身のQTableインスタンスと保存パスを知っている
        self.q_table.save_q_table(self.model_path)

    def get_q_table_size(self) -> int:
        """
        このエージェントのQテーブルに登録されている状態の数を返す.
        """
        # Agentインスタンスは自身のQTableインスタンスのメソッドを呼び出す
        return self.q_table.get_q_table_size()