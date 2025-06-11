"""
グリッドワールドの環境を定義するクラス．
状態の初期化やレンダリング，報酬計算などを行う．
"""

import numpy as np
import pygame

# 乱数の初期化（初期位置固定用）
np.random.seed(0)

# カラー定義
BLACK  = (0, 0, 0)
WHITE  = (255, 255, 255)
GREEN  = (0, 255, 0)
BLUE   = (0, 0, 255)
GRAY   = (128, 128, 128)

class GridWorld:
    def __init__(self, args):
        """コンストラクタ：ウィンドウサイズや環境パラメータの初期設定"""
        self.window_width = args.window_width
        self.window_height = args.window_height
        self.cell_num = args.cell_number
        self.agents_num = args.agents_number
        self.goals_num = args.goals_number
        self.reward_mode = args.reward_mode

        pygame.init()
        self.screen = pygame.display.set_mode((self.window_width, self.window_height))
        self.font = pygame.font.SysFont(None, 36)

        # ゴールとエージェントの位置を保持
        self.goals = []
        self.agents = []

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

    def render(self, episode_num=0, time_step=0):
        """pygame を用いて環境を描画"""
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit()
                exit()

        self._draw_grid()
        self._draw_goals_and_agents()
        self._draw_episode_and_step_info(episode_num, time_step)

        pygame.display.flip()

    def _draw_grid(self):
        """マス目を描画"""
        cell_w = self.window_width / self.cell_num
        cell_h = self.window_height / self.cell_num
        self.screen.fill(GRAY)

        for x in range(self.cell_num):
            for y in range(self.cell_num):
                rect = pygame.Rect(x * cell_w, y * cell_h, cell_w, cell_h)
                pygame.draw.rect(self.screen, BLACK, rect, 1)

    def _draw_goals_and_agents(self):
        """ゴールとエージェントを描画"""
        cell_w = self.window_width / self.cell_num
        cell_h = self.window_height / self.cell_num

        # ゴールの描画
        for goal in self.goals:
            rect = pygame.Rect(goal[0] * cell_w, goal[1] * cell_h, cell_w, cell_h)
            pygame.draw.rect(self.screen, GREEN, rect)

        # エージェントの描画
        agent_w = cell_w * 0.8
        agent_h = cell_h * 0.8
        font = pygame.font.Font(None, int(cell_h * 0.5))

        for idx, agent in enumerate(self.agents):
            rect = pygame.Rect(
                agent[0] * cell_w + (cell_w - agent_w) / 2,
                agent[1] * cell_h + (cell_h - agent_h) / 2,
                agent_w,
                agent_h
            )
            pygame.draw.rect(self.screen, BLUE, rect)

            # エージェント番号
            text_surf = font.render(str(idx), True, WHITE)
            text_rect = text_surf.get_rect(center=(
                agent[0] * cell_w + cell_w / 2,
                agent[1] * cell_h + cell_h / 2
            ))
            self.screen.blit(text_surf, text_rect)

    def _draw_episode_and_step_info(self, episode_num, time_step):
        """エピソード数とステップ数を画面に描画"""
        ep_text = self.font.render(f'Episode: {episode_num}', True, BLACK)
        st_text = self.font.render(f'Step: {time_step}', True, BLACK)
        self.screen.blit(ep_text, (10, 10))
        self.screen.blit(st_text, (10, 50))
