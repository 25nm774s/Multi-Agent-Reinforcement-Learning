"""
グリッドワールドの環境を定義するクラス．
状態の初期化やレンダリング，報酬計算などを行う．

- レンダリング処理は GridRenderer クラスに分離済み．
- 環境のロジック（状態管理，状態遷移，報酬計算など）も，
将来的に必要に応じて別のクラスに分離する可能性あり．
"""

import numpy as np
from utils.grid_renderer import GridRenderer

# 乱数の初期化（初期位置固定用）（あとでGW.initに配置しても良い）
# TODO: 乱数シードの設定場所について検討 (現状はクラス外で一度だけ固定)
# 各GridWorldインスタンスで独立した乱数が必要な場合は__init__内に移動
np.random.seed(0)

# カラー定義 (レンダラーに移動したので不要ですが、もし他の場所で使うなら残しても良いです)
# BLACK  = (0, 0, 0)
# WHITE  = (255, 255, 255)
# GREEN  = (0, 255, 0)
# BLUE   = (0, 0, 255)
# GRAY   = (128, 128, 128)

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
            # GridRenderer クラスは別途定義されていると仮定し、ここではモックや実際のインポートは行わない
            # from utils.renderer import GridRenderer # 実際のコードでは必要
            class MockGridRenderer: # Mock Renderer for demonstration
                def __init__(self, width, height, grid_size):
                    print(f"MockGridRenderer initialized with width={width}, height={height}, grid_size={grid_size}")
                def render(self, goals, agents):
                    #print(f"MockGridRenderer: render called with goals={goals}, agents={agents}")
                    pass # モックなので描画はしない
            self.renderer = MockGridRenderer(args.window_width, args.window_height, self.grid_size)
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

    def generate_unique_positions(self, num_positions, object_positions, grid_size):
        """
        渡された既存座標(object_positions)と重ならないように，
        指定された個数(num_positions)のランダム座標を生成して返す．

        Args:
            num_positions (int): 生成する座標の個数.
            object_positions (list): 既存のオブジェクト位置のリスト.
            grid_size (int): グリッドのサイズ.

        Returns:
            list: 生成された一意な位置のリスト.
        """
        positions = []
        # 既存のオブジェクト位置をセットに変換して高速化
        existing_positions_set = set(object_positions)
        while len(positions) < num_positions:
            pos = (np.random.randint(0, grid_size), np.random.randint(0, grid_size))
            if pos not in existing_positions_set:
                positions.append(pos)
                existing_positions_set.add(pos) # 生成した位置も既存に追加
        # 生成した位置をobject_positionsリストにも追加（呼び出し元で利用するため）
        # object_positions.extend(positions) # この行は不要かもしれないが、元のロジックを踏襲
        return positions

    def reset(self):
        """
        エピソード開始時に環境をリセットし、初期の全体状態を返す．
        エージェントの位置をランダムに再配置する．

        Returns:
            tuple: 環境の初期全体状態 (ゴール位置 + エージェント位置のタプル).
        """
        # ゴール位置は_generate_fixed_goalsで生成済みのものを利用
        # エージェントの位置をランダム生成（ゴール座標との重複回避）
        object_positions = list(self.goals) # ゴール位置を初期の既存位置としてコピー
        self.agents = self.generate_unique_positions(
            self.agents_num, object_positions, self.grid_size
        )
        # global_state にはゴール + エージェントが一続きに入る
        global_state = tuple(self.goals + self.agents)

        # レンダラーがあれば初期状態を描画
        if self.renderer:
            self.renderer.render(self.goals, self.agents)

        return global_state


    def update_positions(self, global_state, actions):
        """
        行動(actions)に基づいてエージェント位置を更新．
        他エージェントと衝突しそうな場合は更新しない．

        Args:
            global_state (tuple): 現在の環境全体状態.
            actions (list): 各エージェントの行動リスト (int のリスト).

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
        # ゴール位置はglobal_stateから取得（固定なのでnext_global_stateと同じだが、step開始時のglobal_stateを使うのがロジックとして自然）
        # TODO: get_goal_positions メソッドの使用を検討
        goals_pos  = [tuple(pos) for pos in global_state[:self.goals_num]] # リストのリストではなくタプルのリストに変換
        # エージェント位置は更新後のnext_global_stateから取得
        # TODO: get_agent_positions メソッドの使用を検討
        agents_pos = [tuple(pos) for pos in next_global_state[self.goals_num:]] # リストのリストではなくタプルのリストに変換

        # 完了条件の判定 (全エージェントがゴールに乗ったか)
        # goals_pos と agents_pos はタプルのリストになっているはずなので、比較可能
        done = all(goal in agents_pos for goal in goals_pos)

        if self.reward_mode == 0:
            # エージェントが全ゴールに乗ったら +10, それ以外は 0
            # 計算された done フラグに基づいて報酬と完了フラグを設定・返却
            reward = 100.0 if done else 0.0 # 報酬をfloatにする
            return next_global_state, reward, done # 計算された done を返す

        elif self.reward_mode == 1:
            # 未完了時は -1, 完了時は 0
            # 計算された done フラグに基づいて報酬と完了フラグを設定・返却
            reward = 1000.0 if done else -1.0 # 報酬をfloatにする
            return next_global_state, reward, done # 計算された done を返す

        else:  # reward_mode == 2
            # ゴールから最近傍エージェントまでのマンハッタン距離の合計を負報酬
            total_distance = 0
            # goals_pos はタプルのリスト
            # agents_pos はタプルのリスト
            for goal in goals_pos:
                # 各ゴールに対して、全てのエージェントとのマンハッタン距離を計算し、最小値を取る
                min_dist = min(abs(goal[0] - ag[0]) + abs(goal[1] - ag[1]) for ag in agents_pos)
                total_distance += min_dist
            # 合計距離の負の値を報酬とする
            reward = - float(total_distance) # 報酬をfloatにする

            # レンダラーがあれば描画
            if self.renderer:
                # レンダラーにはタプルのリストを渡す必要があるかもしれないので、変換
                self.renderer.render(goals_pos, agents_pos) # リストのリストからタプルのリストに変更したので引数も修正

            # reward_mode 2 でも完了条件は同じ
            return next_global_state, reward, done