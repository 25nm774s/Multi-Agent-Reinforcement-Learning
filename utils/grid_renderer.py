"""
グリッドワールドのレンダリングを行うクラス．
pygameを用いて環境を描画する．
"""

#import pygame

# カラー定義
BLACK  = (0, 0, 0)
WHITE  = (255, 255, 255)
GREEN  = (0, 255, 0)
BLUE   = (0, 0, 255)
GRAY   = (128, 128, 128)

class GridRenderer:
    pass
'''
class GridRenderer:
    def __init__(self, window_width, window_height, grid_size):
        """コンストラクタ：ウィンドウサイズや環境パラメータの初期設定"""
        self.window_width = window_width
        self.window_height = window_height
        self.grid_size = grid_size

        pygame.init()
        self.screen = pygame.display.set_mode((self.window_width, self.window_height))
        self.font = pygame.font.SysFont(None, 36)

    def render(self, goals, agents, episode_num=0, time_step=0):
        """pygame を用いて環境を描画"""
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit()
                exit()

        self._draw_grid()
        self._draw_goals_and_agents(goals, agents)
        self._draw_episode_and_step_info(episode_num, time_step)

        pygame.display.flip()

    def _draw_grid(self):
        """マス目を描画"""
        cell_w = self.window_width / self.grid_size
        cell_h = self.window_height / self.grid_size
        self.screen.fill(GRAY)

        for x in range(self.grid_size):
            for y in range(self.grid_size):
                rect = pygame.Rect(x * cell_w, y * cell_h, cell_w, cell_h)
                pygame.draw.rect(self.screen, BLACK, rect, 1)

    def _draw_goals_and_agents(self, goals, agents):
        """ゴールとエージェントを描画"""
        cell_w = self.window_width / self.grid_size
        cell_h = self.window_height / self.grid_size

        # ゴールの描画
        for goal in goals:
            rect = pygame.Rect(goal[0] * cell_w, goal[1] * cell_h, cell_w, cell_h)
            pygame.draw.rect(self.screen, GREEN, rect)

        # エージェントの描画
        agent_w = cell_w * 0.8
        agent_h = cell_h * 0.8
        font = pygame.font.Font(None, int(cell_h * 0.5))

        for idx, agent in enumerate(agents):
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
'''