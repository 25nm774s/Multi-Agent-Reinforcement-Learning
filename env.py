"""
グリッドワールドの環境を定義するクラス．
状態の初期化やレンダリング，報酬計算などを行う．

- レンダリング処理は GridRenderer クラスに分離済み．
- 環境のロジック（状態管理，状態遷移，報酬計算など）も，
将来的に必要に応じて別のクラスに分離する可能性あり．
"""

import numpy as np
from utils.grid_renderer import GridRenderer
import random

# 乱数の初期化（初期位置固定用）（あとでGW.initに配置しても良い）
# TODO: 乱数シードの設定場所について検討 (現状はクラス外で一度だけ固定)
# 各GridWorldインスタンスで独立した乱数が必要な場合は__init__内に移動
np.random.seed(0)

class GridWorld:
    """
    マルチエージェントグリッドワールド環境クラス.
    エージェントとゴールの配置、状態遷移、報酬計算を行います.
    """
    def __init__(self, args):
        """
        GridWorld コンストラクタ.

        Args:
            args: 環境設定を含むオブジェクト.
                (grid_size, agents_number, goals_number, reward_mode,
                render_mode, window_width, window_height 属性を持つことを想定)
        """
        self.grid_size = args.grid_size
        self.agents_num = args.agents_number
        self.goals_num = args.goals_number
        self.reward_mode = args.reward_mode

        # ゴールとエージェントの位置を保持
        self.goals = []
        self.agents = []

        # レンダラーのインスタンスを生成 (args.render_modeに基づいて)
        if args.render_mode:
            #self.renderer = GridRenderer(args.window_width, args.window_height, self.grid_size)
            self.renderer = None
        else:
            self.renderer = None

        # ゴール位置は一度生成したら固定
        self._generate_fixed_goals()

        # TODO: statesからゴール・エージェント位置を取り出す方法について検討
        # 現在はメソッドの引数として渡されるstatesに対してスライスを使用
        # Alternatives:
        # 1. get_goal_positions(states), get_agent_positions(states) メソッドを作成 (現在の設計に合う)
        # 2. GridWorldが内部で現在の状態を保持し、goal_positions, agent_positions プロパティを作成 (設計変更が必要)
        # メソッド vs プロパティの使い分けについては、states変数をどこで管理するかに依存する。
        # - main (または MultiAgent_Q)で管理 -> メソッドが自然
        # - GridWorld内部で管理 -> プロパティが自然

    def _generate_fixed_goals(self):
        """
        最初の一度だけゴール位置を生成して固定する内部メソッド.
        既存の位置と重複しないようにランダムに生成します.
        """
        object_positions = []
        self.goals = self.generate_unique_positions(
            self.goals_num, object_positions, self.grid_size
        )

    def get_goal_positions(self,global_state):
        """
        全体状態タプルからゴール位置のリストを取得.

        Args:
            global_state (tuple): 環境の全体状態タプル.

        Returns:
            list: ゴール位置のリスト.
        """
        # TODO: global_stateの構造が変わった場合にここだけ修正すれば済むように、このメソッドを使用することを検討
        return list(global_state[:self.goals_num])

    def get_agent_positions(self,global_state):
        """
        全体状態タプルからエージェント位置のリストを取得.

        Args:
            global_state (tuple): 環境の全体状態タプル.

        Returns:
            list: エージェント位置のリスト.
        """
        # TODO: global_stateの構造が変わった場合にここだけ修正すれば済むように、このメソッドを使用することを検討
        return list(global_state[self.goals_num:])


    def generate_unique_positions(self, num_positions, existing_positions, grid_size):
        """
        渡された既存座標(existing_positions)と重ならないように，
        指定された個数(num_positions)のランダム座標を生成して返す．

        Args:
            num_positions (int): 生成する座標の個数.
            existing_positions (list): 既存のオブジェクト位置のリスト.
            grid_size (int): グリッドのサイズ.

        Returns:
            list: 生成された一意な位置のリスト.
        """
        positions = []
        # 既存のオブジェクト位置をセットに変換して高速化
        existing_positions_set = set(existing_positions)
        while len(positions) < num_positions:
            pos = (np.random.randint(0, grid_size), np.random.randint(0, grid_size))
            if pos not in existing_positions_set:
                positions.append(pos)
                existing_positions_set.add(pos) # 生成した位置も既存に追加
        # 生成した位置をexisting_positionsリストにも追加（呼び出し元で利用するため）
        # existing_positions.extend(positions) # この行は不要かもしれないが、元のロジックを踏襲
        return positions

    def reset(self):
        """
        エピソード開始時に環境をリセットし、初期の全体状態を返す．
        エージェントの位置をゴール近傍、またはランダムに再配置する．

        Returns:
            tuple: 環境の初期全体状態 (ゴール位置 + エージェント位置のタプル).
        """
        # ゴール位置は_generate_fixed_goalsで生成済みのものを利用
        object_positions = list(self.goals) # ゴール位置を初期の既存位置としてコピー
        self.agents = []
        existing_positions_set = set(object_positions) # 既存の位置セット

        # エージェントの初期位置をゴール近傍に配置するロジック
        # TODO: init_near_goal フラグでこの動作を切り替えられるようにする
        # if self.init_near_goal:
        for _ in range(self.agents_num):
            placed = False
            max_attempts = 100 # ゴール近傍での配置試行回数
            attempt = 0
            while not placed and attempt < max_attempts:
                # いずれかのゴールをランダムに選択
                target_goal = random.choice(self.goals)
                gx, gy = target_goal

                # ゴール近傍の座標を生成 (マンハッタン距離1または2以内)
                # dx, dy は -2 から 2 の範囲でランダムに選択し、マンハッタン距離 <= 2 を満たすもの
                dx = np.random.randint(-2, 3)
                dy = np.random.randint(-2, 3)
                if abs(dx) + abs(dy) > 2 or (dx == 0 and dy == 0): # ゴール位置自体は避ける
                     continue

                new_ax = gx + dx
                new_ay = gy + dy

                # グリッド範囲内か確認
                if 0 <= new_ax < self.grid_size and 0 <= new_ay < self.grid_size:
                    new_pos = (new_ax, new_ay)

                    # 生成された位置が既存のオブジェクト(ゴールや他のエージェント)と重複しないか確認
                    if new_pos not in existing_positions_set:
                        self.agents.append(new_pos)
                        existing_positions_set.add(new_pos) # 配置したエージェント位置も既存に追加
                        placed = True
                        break # 配置成功
                attempt += 1

            # もしゴール近傍に配置できなかった場合 (一定回数試行しても重複する場合)
            if not placed:
                 # ランダムな位置に配置するフォールバック
                 while True:
                     pos = (np.random.randint(0, self.grid_size), np.random.randint(0, self.grid_size))
                     if pos not in existing_positions_set:
                         self.agents.append(pos)
                         existing_positions_set.add(pos)
                         break
        # else: # もし init_near_goal が False の場合、元のランダム配置ロジックを使用
        #     object_positions = list(self.goals) # ゴール位置を初期の既存位置としてコピー
        #     self.agents = self.generate_unique_positions(
        #         self.agents_num, object_positions, self.grid_size
        #     )


        # global_state にはゴール + エージェントが一続きに入る
        global_state = tuple(self.goals + self.agents)

        # レンダラーがあれば初期状態を描画
        if self.renderer:
            self.renderer.render(self.goals, self.agents)

        # 密な報酬計算のために、前ステップのゴールまでの距離を保持
        # reset時に初期化
        self._prev_total_distance_to_goals = self._calculate_total_distance_to_goals(global_state)
        # 各ゴールに到達したかどうかのフラグを保持 (段階報酬用)
        self._goals_reached_status = [False] * self.goals_num


        return global_state


    def update_positions(self, global_state, actions):
        """
        行動(actions)に基づいてエージェント位置を更新．
        他エージェントと衝突しそうな場合は更新しない．

        Args:
            global_state (tuple): 現在の環境全体状態.
            actions (list): 各エージェntの行動リスト (int のリスト).

        Returns:
            tuple: 更新されたエージェント位置を含む次の全体状態タプル.
        """
        # global_state からゴールとエージェントの位置を分離
        # TODO: get_goal_positions, get_agent_positions メソッドの使用を検討
        goals_pos = list(global_state[:self.goals_num])
        agents_pos = [list(pos) for pos in global_state[self.goals_num:]]
        new_positions = [pos[:] for pos in agents_pos]

        for idx, action in enumerate(actions):
            if action == 0 and agents_pos[idx][1] > 0:                # UP
                new_positions[idx][1] -= 1
            elif action == 1 and agents_pos[idx][1] < self.grid_size - 1:  # DOWN
                new_positions[idx][1] += 1
            elif action == 2 and agents_pos[idx][0] > 0:                # LEFT
                new_positions[idx][0] -= 1
            elif action == 3 and agents_pos[idx][0] < self.grid_size - 1:  # RIGHT
                new_positions[idx][0] += 1
            elif action == 4:
                pass  # STAY

        # 他のエージェントと位置が重なる場合は更新前に戻す
        for idx in range(len(new_positions)):
            # self以外のagent_idxで自分と同じ位置をnew_positionsの中で探す
            other_agents_at_new_pos = [
                i for i, pos in enumerate(new_positions)
                if i != idx and pos == new_positions[idx]
            ]
            if other_agents_at_new_pos:
                new_positions[idx] = agents_pos[idx] # 元の位置に戻す

        # 更新されたエージェントの位置をインスタンス変数に反映
        self.agents = [tuple(pos) for pos in new_positions]
        # 新しい全体状態を返す (ゴール位置 + 更新されたエージェント位置)
        return tuple(goals_pos + self.agents)

    def _calculate_total_distance_to_goals(self, global_state):
        """
        環境全体の状態から、全ゴールに対して最近傍のエージェントまでのマンハッタン距離の合計を計算する。

        Args:
            global_state (tuple): 環境の全体状態タプル.

        Returns:
            float: 全ゴールに対する最近傍エージェントまでのマンハッタン距離の合計.
        """
        goals_pos  = [tuple(pos) for pos in global_state[:self.goals_num]]
        agents_pos = [tuple(pos) for pos in global_state[self.goals_num:]]

        total_distance = 0
        for goal in goals_pos:
            # 各ゴールに対して、全てのエージェントとのマンハッタン距離を計算し、最小値を取る
            min_dist = min(abs(goal[0] - ag[0]) + abs(goal[1] - ag[1]) for ag in agents_pos)
            total_distance += min_dist
        return float(total_distance)

    def step(self, global_state, actions):
        """
        行動を受け取り状態を更新．報酬と終了条件を返す．

        Args:
            global_state (tuple): 現在の環境全体状態.
            actions (list): 各エージェントの行動リスト (int のリスト).

        Returns:
            tuple: (next_global_state, reward, done) のタプル.
                next_global_state (tuple): 次の環境全体状態.
                reward (float): 得られた報酬.
                done (bool): エピソード完了フラグ (全エージェントがゴールに到達したか).
        """
        # global_stateを受け取り、update_positionsで次の状態を計算
        next_global_state = self.update_positions(global_state, actions)

        # 報酬と終了条件をモード別に計算
        goals_pos  = [tuple(pos) for pos in global_state[:self.goals_num]]
        agents_pos = [tuple(pos) for pos in next_global_state[self.goals_num:]]

        # 完了条件の判定 (全エージェントがゴールに乗ったか)
        # goals_pos と agents_pos はタプルのリストになっているはずなので、比較可能
        done = all(goal in agents_pos for goal in goals_pos)

        reward = 0.0 # 報酬を初期化

        if self.reward_mode == 0:
            # モード 0: エージェントが全ゴールに乗ったら +100, それ以外は 0
            reward = 100.0 if done else 0.0

        elif self.reward_mode == 1:
            # モード 1: 未完了時は -5, 完了時は 500
            reward = 500.0 if done else -5.0

        elif self.reward_mode == 2:
            # モード 2: ゴールから最近傍エージェントまでのマンハッタン距離の合計を負報酬
            # total_distance = self._calculate_total_distance_to_goals(next_global_state) # これはステップごとの距離自体を報酬にする場合
            # reward = - total_distance
            pass # このモードは距離の差分を使う密な報酬で置き換えるか、区別する

        elif self.reward_mode == 3:
             # モード 3: 密な報酬 (距離の差分) + 段階報酬 + 完了報酬
             # 1. 前ステップからの合計距離の変化に基づく報酬
             current_total_distance = self._calculate_total_distance_to_goals(next_global_state)
             distance_change = self._prev_total_distance_to_goals - current_total_distance # 距離が減れば正、増えれば負
             reward += distance_change * 1.0 # 距離の差分に比例した報酬 (スケールは調整可能)

             # 2. 各ゴールに到達した際の段階報酬
             current_agents_pos_set = set(agents_pos)
             for goal_idx, goal in enumerate(goals_pos):
                 # そのゴールにエージェントがいて、かつまだそのゴールを達成済みとして記録していない場合
                 if goal in current_agents_pos_set and not self._goals_reached_status[goal_idx]:
                     reward += 50.0 # ゴール一つ達成につき報酬 (調整可能)
                     self._goals_reached_status[goal_idx] = True # このゴールは達成済みとして記録

             # 3. 全ゴール達成時の完了報酬 (既存の密な報酬や段階報酬に加えて)
             if done:
                 reward += 200.0 # 全完了時の追加報酬 (調整可能)

             # 次のステップのために現在の合計距離を保存
             self._prev_total_distance_to_goals = current_total_distance

        else: # 未知のreward_modeの場合
            print(f"Warning: Unknown reward_mode: {self.reward_mode}. Reward is 0.")
            reward = 0.0


        # レンダラーがあれば描画
        if self.renderer:
            # レンダラーにはタプルのリストを渡す必要があるかもしれないので、変換
            self.renderer.render(goals_pos, agents_pos) # リストのリストからタプルのリストに変更したので引数も修正


        return next_global_state, reward, done