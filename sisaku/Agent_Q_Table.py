import numpy as np
from abc import ABC, abstractmethod
from typing import Any, List, Tuple, Optional, Dict
from qtable import QTable # QTableクラスをインポート

# 環境の状態を表す型（ここでは柔軟にAnyとするか、具体的な型を定義する）
# グリッドワールドの座標リストなら List[Tuple[int, int]] など
StateType = Any

# 行動を表す型（離散行動の場合はint）
ActionType = int

# 報酬を表す型
RewardType = float

# 経験を表す型（状態インデックス, 行動, 報酬, 次状態インデックス, 終了フラグ）
ExperienceType = Tuple[int, ActionType, RewardType, int, bool]

class Agent(ABC):
    """
    強化学習エージェントの抽象基底クラス.
    """

    def __init__(self, agent_id: int, args: Any, **kwargs):
         """
         エージェントを初期化する.

         Args:
             agent_id: エージェントの識別子.
             args: 設定引数を保持するオブジェクト（環境やエージェント共通の設定など）.
             **kwargs: その他の初期化パラメータ（QTableのサイズ、ハイパーパラメータなど）.
         """
         self.agent_id = agent_id
         self.args = args


    @abstractmethod
    def get_action(self, agent_id: int, full_states: StateType) -> ActionType:
        """
        現在の環境全体の状態に基づき、エージェントの行動を選択する.

        Args:
            agent_id: 行動を選択するエージェントのID.
            full_states: 環境全体の状態.

        Returns:
            選択された行動.
        """
        pass

    @abstractmethod
    def observe_and_store_experience(self, full_states: StateType, action: ActionType, reward: RewardType, next_full_states: StateType, done: bool) -> None:
        """
        環境とのインタラクションによる経験を観測し、ストアする.

        Args:
            full_states: 環境の現在の状態.
            action: エージェントが取った行動.
            reward: 環境から得られた報酬.
            next_full_states: 環境の次の状態.
            done: エピソードが終了したかどうかのフラグ.
        """
        pass

    @abstractmethod
    def learn_from_experience(self, agent_id: int, episode_num: int) -> Optional[float]:
        """
        ストアされた経験を用いて学習を行う.

        Args:
            agent_id: 学習を行うエージェントのID.
            episode_num: 現在のエピソード番号.

        Returns:
            学習によって得られた損失など（任意）.
        """
        pass

    @abstractmethod
    def decay_epsilon_pow(self, episode_num: int) -> None:
        """
        ε-greedy法のε値を減衰させる.

        Args:
            episode_num: 現在のエピソード番号.
        """
        pass

    ## 必要に応じて、Qテーブルの保存/読み込みメソッドなども抽象メソッドとして追加可能
    ## @abstractmethod
    ## def save_model(self, file_path: str) -> None:
    ##     pass

    ## @abstractmethod
    ## def load_model(self, file_path: str) -> None:
    ##     pass


class Agent_Q(Agent):
    """
    Q学習に基づくグリッドワールド用エージェント.
    状態は自身の位置と最も近いゴールの位置で定義される.
    """

    def __init__(self, agent_id: int, args: Any, grid_width: int, grid_height: int, num_goals: int, action_size: int, learning_rate: float, discount_factor: float, epsilon_start: float, epsilon_end: float, epsilon_decay: float):
        """
        Agent_Qエージェントを初期化する.

        Args:
            agent_id: エージェントの識別子.
            args: 設定引数を保持するオブジェクト.
            grid_width: グリッドの幅.
            grid_height: グリッドの高さ.
            num_goals: 環境内のゴールの数.
            action_size: 行動空間のサイズ.
            learning_rate: 学習率.
            discount_factor: 割引率.
            epsilon_start: ε-greedy法の初期ε値.
            epsilon_end: ε-greedy法の最終ε値.
            epsilon_decay: ε-greedy法の減衰率.
        """
        super().__init__(agent_id, args)

        self.grid_width = grid_width
        self.grid_height = grid_height
        self.num_goals = num_goals
        self.action_size = action_size
        self.learning_rate = learning_rate
        self.discount_factor = discount_factor
        self.epsilon = epsilon_start
        self.epsilon_start = epsilon_start
        self.epsilon_end = epsilon_end
        self.epsilon_decay = epsilon_decay

        # 状態空間サイズ: 自身のX * 自身のY * ゴールX * ゴールY
        # 各要素の最大値はグリッドサイズ
        # 自身の位置 (grid_width * grid_height)
        # 最も近いゴールの位置 (grid_width * grid_height)
        state_size = grid_width * grid_height * grid_width * grid_height

        self.q_table = QTable(state_size, self.action_size)

        # 経験をストアする変数 (逐次学習用)
        self.stored_experience: Optional[ExperienceType] = None


    def coordinate_pair_to_index(self, agent_coord: Tuple[int, int], goal_coord: Tuple[int, int]) -> int:
        """
        エージェント座標と最も近いゴール座標のペアをQテーブルのインデックスに変換する.
        """
        ax, ay = agent_coord
        gx, gy = goal_coord

        # 座標がグリッド範囲内かチェック (オプション、環境側で保証される場合不要)
        if not (0 <= ax < self.grid_width and 0 <= ay < self.grid_height and
                0 <= gx < self.grid_width and 0 <= gy < self.grid_height):
            # 範囲外の場合は無効なインデックスなどを返す
            print(f"Warning: Invalid coordinates received: Agent {agent_coord}, Goal {goal_coord}")
            return -1 # 例: 無効なインデックスとして-1を返す

        # インデックスへの変換ロジック
        # agent_x (0 to W-1)
        # agent_y (0 to H-1) -> + agent_y * W
        # goal_x (0 to W-1) -> + goal_x * (W*H)
        # goal_y (0 to H-1) -> + goal_y * (W*H*W)
        index = ax + ay * self.grid_width + gx * (self.grid_width * self.grid_height) + gy * (self.grid_width * self.grid_height * self.grid_width)

        return index

    def find_nearest_goal(self, agent_coord: Tuple[int, int], goal_coords: List[Tuple[int, int]]) -> Tuple[int, int]:
        """
        エージェントの座標から最も近いゴールの座標を見つける.
        ユークリッド距離を使用.
        """
        if not goal_coords:
            # ゴールがない場合の処理 (例: 特殊な座標を返す、エラーを発生させる)
            # ここでは仮にエージェント自身の座標を返す
            print(f"Warning: No goals provided for agent {self.agent_id}. Returning agent's own position as goal.")
            return agent_coord

        min_dist = float('inf')
        nearest_goal = goal_coords[0] # デフォルトとして最初のゴールを設定

        ax, ay = agent_coord

        for gx, gy in goal_coords:
            dist = np.sqrt((ax - gx)**2 + (ay - gy)**2)
            if dist < min_dist:
                min_dist = dist
                nearest_goal = (gx, gy)

        return nearest_goal


    def extract_agent_state_info(self, full_states: StateType) -> Tuple[Tuple[int, int], Tuple[int, int]]:
        """
        full_states から自身の座標と最も近いゴールの座標を抽出する.
        full_states がタプルのリストで、最初の num_agents 個がエージェント座標、
        残りがゴール座標であると仮定する.
        例: ((ax1, ay1), ..., (g1x, g1y), ...)
        """
        # argsオブジェクトからエージェントの総数を取得する必要がある
        # self.args に num_agents が含まれていると仮定
        if not hasattr(self.args, 'num_agents'):
             raise AttributeError("args object must have 'num_agents' attribute.")

        num_agents = self.args.num_agents

        if not isinstance(full_states, (list, tuple)) or len(full_states) != (num_agents + self.num_goals):
            raise ValueError(f"full_states has unexpected structure or length. Expected list/tuple of length {num_agents + self.num_goals}")

        # 自身の座標を抽出
        if 0 <= self.agent_id < num_agents:
            agent_coord = full_states[self.agent_id]
        else:
            raise ValueError(f"Invalid agent_id: {self.agent_id} for num_agents {num_agents}")

        # ゴール座標リストを抽出
        goal_coords = full_states[num_agents:]

        # 最も近いゴールを見つける
        nearest_goal_coord = self.find_nearest_goal(agent_coord, goal_coords)

        return agent_coord, nearest_goal_coord


    def get_action(self, agent_id: int, full_states: StateType) -> ActionType:
        # エージェントIDの確認 (自身のIDと一致するか)
        if agent_id != self.agent_id:
             print(f"Warning: get_action called for agent_id {agent_id} on agent {self.agent_id}. Processing for self.")

        # full_states から自身の状態情報 (自身の座標と最も近いゴール座標) を抽出
        try:
            agent_coord, nearest_goal_coord = self.extract_agent_state_info(full_states)
        except (ValueError, AttributeError) as e:
            print(f"Error extracting state info for agent {self.agent_id}: {e}")
            # エラー発生時はランダムに行動を選択するなど、フォールバック処理が必要
            return np.random.randint(self.action_size)


        # 自身の状態情報ペアをQテーブルのインデックスに変換
        discrete_agent_state_index = self.coordinate_pair_to_index(agent_coord, nearest_goal_coord)

        # 無効な状態インデックスの場合は、ランダムに行動を選択
        if discrete_agent_state_index == -1:
             print(f"Warning: Agent {self.agent_id} is in an invalid state representation, choosing random action.")
             return np.random.randint(self.action_size)


        # ε-greedy法による行動選択
        if np.random.rand() < self.epsilon:
            # ランダムに行動を選択
            action = np.random.randint(self.action_size)
        else:
            # Q値に基づいて最適な行動を選択
            # QTableのget_q_valueを使うか、直接q_table配列にアクセス
            action = np.argmax(self.q_table.q_table[discrete_agent_state_index, :])

        return action


    def observe_and_store_experience(self, full_states: StateType, action: ActionType, reward: RewardType, next_full_states: StateType, done: bool) -> None:
        # full_states から現在の自身の状態情報ペアを抽出
        try:
            current_agent_coord, current_nearest_goal_coord = self.extract_agent_state_info(full_states)
             # 自身の現在の状態情報ペアをインデックスに変換
            discrete_current_state_index = self.coordinate_pair_to_index(current_agent_coord, current_nearest_goal_coord)
        except (ValueError, AttributeError) as e:
             print(f"Error extracting current state info for agent {self.agent_id}: {e}. Skipping experience storage.")
             self.stored_experience = None
             return


        # next_full_states から次の自身の状態情報ペアを抽出
        try:
            next_agent_coord, next_nearest_goal_coord = self.extract_agent_state_info(next_full_states)
             # 自身の次の状態情報ペアをインデックスに変換
            discrete_next_state_index = self.coordinate_pair_to_index(next_agent_coord, next_nearest_goal_coord)
        except (ValueError, AttributeError) as e:
             print(f"Error extracting next state info for agent {self.agent_id}: {e}. Skipping experience storage.")
             self.stored_experience = None
             return


        # 無効な状態インデックスの場合は経験としてストアしない
        if discrete_current_state_index == -1 or discrete_next_state_index == -1:
             print(f"Warning: Agent {self.agent_id} received experience with invalid state indices. Skipping storage/learning.")
             self.stored_experience = None # ストアしない
             return

        # 経験をストア（逐次学習用）
        self.stored_experience = (discrete_current_state_index, action, reward, discrete_next_state_index, done)

        # 経験再生を使う場合はバッファに追加
        # self.replay_buffer.add(...)


    def learn_from_experience(self, agent_id: int, episode_num: int) -> Optional[float]:
        # エージェントIDの確認
        if agent_id != self.agent_id:
             #print(f"Info: learn_from_experience called for agent_id {agent_id} on agent {self.agent_id}. Processing for self.")
             pass # 警告がうるさい場合はコメントアウト

        # ストアされた経験がない場合は学習しない
        if self.stored_experience is None:
            return None

        # ストアされた経験を取得
        state, action, reward, next_state, done = self.stored_experience

        # Q学習の更新
        # done の扱い: 終了状態からの遷移のQ値は0とする必要がある
        # QTable.update_q_value は done を直接受け取らないため、ここで調整
        if done:
            # 終了状態の場合、次の状態の最大Q値は0とみなす
            # Q(s, a) = Q(s, a) + alpha * (r + gamma * 0 - Q(s, a))
            current_q = self.q_table.get_q_value(state, action)
            td_target = reward
            td_delta = td_target - current_q
            self.q_table.q_table[state, action] += self.learning_rate * td_delta
            #print(f"Agent {self.agent_id} learned from terminal state: {state} -> {action}, reward {reward}")
        else:
            # 通常のQ学習更新
            self.q_table.update_q_value(state, action, reward, next_state, self.learning_rate, self.discount_factor)
            #print(f"Agent {self.agent_id} learned: {state} -> {action} -> {next_state}, reward {reward}")


        self.stored_experience = None # 経験をクリア
        return None # 必要に応じて損失値を返す


    def decay_epsilon_pow(self, episode_num: int) -> None:
        """
        エピソード番号に基づいてε-greedy法のε値を減衰させる (冪乗減衰).
        """
        self.epsilon = max(self.epsilon_end, self.epsilon_start * (self.epsilon_decay ** (episode_num - 1))) # エピソード番号は1から始まる想定

    # Qテーブルの保存/読み込みメソッド (必要に応じて追加)
    def save_q_table(self, file_path: str) -> None:
        """自身のQテーブルをファイルに保存する."""
        try:
            self.q_table.save_q_table(file_path)
        except Exception as e:
            print(f"Error saving Q table for agent {self.agent_id}: {e}")

    def load_q_table(self, file_path: str) -> None:
        """ファイルから自身のQテーブルを読み込む."""
        try:
            self.q_table.load_q_table(file_path)
        except FileNotFoundError:
             print(f"Q table file not found for agent {self.agent_id} at {file_path}. Starting with a new table.")
        except Exception as e:
            print(f"Error loading Q table for agent {self.agent_id}: {e}")