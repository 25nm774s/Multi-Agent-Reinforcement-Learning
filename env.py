"""
グリッドワールドの環境を定義するクラス．
状態の初期化やレンダリング，報酬計算などを行う．

- レンダリング処理は GridRenderer クラスに分離済み．
- 環境のロジック（状態管理，状態遷移，報酬計算など）も，
  将来的に必要に応じて別のクラスに分離する可能性あり．
"""

import numpy as np
import pygame
from grid_renderer import GridRenderer

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
        self.window_width = args.window_width
        self.window_height = args.window_height
        self.cell_num = args.grid_size
        self.agents_num = args.agents_number
        self.goals_num = args.goals_number
        self.reward_mode = args.reward_mode

        # pygame.init() # レンダラーに移動
        # self.screen = pygame.display.set_mode((self.window_width, self.window_height)) # レンダラーに移動
        # self.font = pygame.font.SysFont(None, 36) # レンダラーに移動

        # ゴールとエージェントの位置を保持
        self.goals = []
        self.agents = []

        # レンダラーのインスタンスを生成
        self.renderer = GridRenderer(self.window_width, self.window_height, self.cell_num)


    def generate_unique_positions(self, num_positions, object_positions, cell_num):
        """
        渡された既存座標と重ならないように，
        (num_positions)個のランダム座標を生成して返す
        """
        positions = []
        while len(positions) < num_positions:
            pos = (np.random.randint(0, cell_num), np.random.randint(0, cell_num))
            if pos not in object_positions:
                positions.append(pos)
                object_positions.append(pos)
        return positions

    def update_positions(self, states, actions):
        """
        行動(actions)に基づいてエージェント位置を更新．
        他エージェントと衝突しそうな場合は更新しない．
        """
        goals_pos = list(states[:self.goals_num])
        agents_pos = [list(pos) for pos in states[self.goals_num:]]
        new_positions = [pos[:] for pos in agents_pos]

        for idx, action in enumerate(actions):
            if action == 0 and agents_pos[idx][1] > 0:                # UP
                new_positions[idx][1] -= 1
            elif action == 1 and agents_pos[idx][1] < self.cell_num - 1:  # DOWN
                new_positions[idx][1] += 1
            elif action == 2 and agents_pos[idx][0] > 0:                # LEFT
                new_positions[idx][0] -= 1
            elif action == 3 and agents_pos[idx][0] < self.cell_num - 1:  # RIGHT
                new_positions[idx][0] += 1
            elif action == 4:
                pass  # STAY

        # 他のエージェントと位置が重なる場合は更新前に戻す
        for idx in range(len(new_positions)):
            if new_positions.count(new_positions[idx]) > 1:
                new_positions[idx] = agents_pos[idx]

        self.agents = [tuple(pos) for pos in new_positions]
        return tuple(goals_pos + self.agents)

    def step(self, states, actions, time_step):
        """
        行動を受け取り状態を更新．報酬と終了条件を返す
        """
        next_state = self.update_positions(states, actions)
        goals_pos  = [list(pos) for pos in states[:self.goals_num]]
        agents_pos = [list(pos) for pos in next_state[self.goals_num:]]

        # 報酬と終了条件をモード別に計算
        if self.reward_mode == 0:
            # エージェントが全ゴールに乗ったら +10
            done = all(goal in agents_pos for goal in goals_pos)
            reward = 10 if done else 0
            return next_state, reward, done

        elif self.reward_mode == 1:
            # 未完了時は -1, 完了時は 0
            done = all(goal in agents_pos for goal in goals_pos)
            reward = 0 if done else -1
            return next_state, reward, done

        else:  # reward_mode == 2
            # ゴールから最近傍エージェントまでのマンハッタン距離の合計を負報酬
            total_distance = 0
            for goal in goals_pos:
                min_dist = min(abs(goal[0] - ag[0]) + abs(goal[1] - ag[1]) for ag in agents_pos)
                total_distance += min_dist
            reward = - total_distance
            done = all(goal in agents_pos for goal in goals_pos)
            return next_state, reward, done

    def render(self, states, episode_num=0, time_step=0):
        """pygame を用いて環境を描画 (レンダラーに処理を委譲)"""
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