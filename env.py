"""
グリッドワールドの環境を定義するクラス．
状態の初期化やレンダリング，報酬計算などを行う．

- レンダリング処理は GridRenderer クラスに分離済み．
- 環境のロジック（状態管理，状態遷移，報酬計算など）も，
  将来的に必要に応じて別のクラスに分離する可能性あり．
"""

import numpy as np
from utils.grid_renderer import GridRenderer

# 乱数の初期化（初期位置固定用）
np.random.seed(0)

# カラー定義 (レンダラーに移動したので不要ですが、もし他の場所で使うなら残しても良いです)
# BLACK  = (0, 0, 0)
# WHITE  = (255, 255, 255)
# GREEN  = (0, 255, 0)
# BLUE   = (0, 0, 255)
# GRAY   = (128, 128, 128)

class GridWorld:
    def __init__(self, args):
        """コンストラクタ：ウィンドウサイズや環境パラメータの初期設定"""
        self.grid_size = args.grid_size
        self.agents_num = args.agents_number
        self.goals_num = args.goals_number
        self.reward_mode = args.reward_mode

        # ゴールとエージェントの位置を保持
        self.goals = []
        self.agents = []

        # レンダラーのインスタンスを生成 (args.render_modeに基づいて)
        if args.render_mode:
            self.renderer = GridRenderer(args.window_width, args.window_height, self.grid_size)
        else:
            self.renderer = None

        # ゴール位置は一度生成したら固定
        self._generate_fixed_goals()


    def _generate_fixed_goals(self):
        """
        最初の一度だけゴール位置を生成して固定する内部メソッド
        """
        object_positions = []
        self.goals = self.generate_unique_positions(
            self.goals_num, object_positions, self.grid_size
        )


    def generate_unique_positions(self, num_positions, object_positions, grid_size):
        """
        渡された既存座標と重ならないように，
        (num_positions)個のランダム座標を生成して返す
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
        エピソード開始時に環境をリセットし、初期状態を返す．
        エージェントの位置をランダムに再配置する．
        """
        # ゴール位置は_generate_fixed_goalsで生成済みのものを利用
        # エージェントの位置をランダム生成（ゴール座標との重複回避）
        object_positions = list(self.goals) # ゴール位置を初期の既存位置としてコピー
        self.agents = self.generate_unique_positions(
            self.agents_num, object_positions, self.grid_size
        )
        # states にはゴール + エージェントが一続きに入る
        states = tuple(self.goals + self.agents)
        return states


    def update_positions(self, states, actions):
        """
        行動(actions)に基づいてエージェント位置を更新．
        他エージェントと衝突しそうな場合は更新しない．
        """
        # states からゴールとエージェントの位置を分離
        goals_pos = list(states[:self.goals_num])
        agents_pos = [list(pos) for pos in states[self.goals_num:]]
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
        # 新しい状態を返す (ゴール位置 + 更新されたエージェント位置)
        return tuple(goals_pos + self.agents)


    def step(self, states, actions):
        """
        行動を受け取り状態を更新．報酬と終了条件を返す
        """
        # statesを受け取り、update_positionsで次の状態を計算
        next_state = self.update_positions(states, actions)

        # 報酬と終了条件をモード別に計算
        # ゴール位置はstatesから取得（固定なのでnext_stateと同じだが、step開始時のstatesを使うのがロジックとして自然）
        goals_pos  = [list(pos) for pos in states[:self.goals_num]]
        # エージェント位置は更新後のnext_stateから取得
        agents_pos = [list(pos) for pos in next_state[self.goals_num:]]


        if self.reward_mode == 0:
            # エージェントが全ゴールに乗ったら +10
            # agents_pos はリストのリストなので、goals_posのタプルと比較するために変換
            done = all(goal in [tuple(ag) for ag in agents_pos] for goal in goals_pos)
            reward = 10 if done else 0
            return next_state, reward, done

        elif self.reward_mode == 1:
            # 未完了時は -1, 完了時は 0
            done = all(goal in [tuple(ag) for ag in agents_pos] for goal in goals_pos)
            reward = 0 if done else -1
            return next_state, reward, done

        else:  # reward_mode == 2
            # ゴールから最近傍エージェントまでのマンハッタン距離の合計を負報酬
            total_distance = 0
            for goal in goals_pos:
                min_dist = min(abs(goal[0] - ag[0]) + abs(goal[1] - ag[1]) for ag in agents_pos)
                total_distance += min_dist
            reward = - total_distance
            done = all(goal in [tuple(ag) for ag in agents_pos] for goal in goals_pos)
            return next_state, reward, done

    def render(self, states, episode_num=0, time_step=0):
        """pygame を用いて環境を描画 (レンダラーに処理を委譲)"""
        # レンダラーが存在する場合のみ描画処理を行う
        if self.renderer:
            # states からゴールとエージェントの位置を分離してレンダラーに渡す
            goals_pos = list(states[:self.goals_num])
            agents_pos = list(states[self.goals_num:])
            self.renderer.render(goals_pos, agents_pos, episode_num, time_step)

    # レンダリング関連のメソッドは削除
    # def _draw_grid(self):
    #     """マス目を描画"""
    #     pass # 削除

    # def _draw_goals_and_agents(self):
    #     """ゴールとエージェントを描画"""
    #     pass # 削除

    # def _draw_episode_and_step_info(self, episode_num, time_step):
    #     """エピソード数とステップ数を画面に描画"""
    #     pass # 削除