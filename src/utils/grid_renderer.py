import matplotlib.pyplot as plt
import matplotlib.patches as patches

# Use the 'inline' backend for displaying plots in Colab notebooks
# %matplotlib inline
'''
class GridRenderer:
    """
    GridWorld 環境を matplotlib を使用してレンダリングするクラス.
    """
    def __init__(self, window_width, window_height, grid_size, pause_duration=0.01):
        """
        GridRenderer コンストラクタ.

        Args:
            window_width (int): レンダリングウィンドウの幅 (ピクセル).
            window_height (int): レンダeringウィンドウの高さ (ピクセル).
            grid_size (int): グリッドのサイズ.
            pause_duration (float): 描画後の一時停止時間 (秒).
        """
        self.grid_size = grid_size
        self.pause_duration = pause_duration # ポーズ時間をインスタンス変数として保持
        # matplotlib の描画領域を grid_size に合わせて調整
        # figsize はインチ単位なので、dpiを考慮する必要があるが、ここでは単純化し、grid_size に比例させる
        fig_size_inches = 5 # 適当なサイズ。必要に応じて調整
        self.fig, self.ax = plt.subplots(1, 1, figsize=(fig_size_inches, fig_size_inches))
        # self.ax の設定は render メソッド内で行うように変更

        # エージェントとゴールのパッチを保持するリスト
        self.agent_patches = []
        self.goal_patches = []


    def render(self, goals, agents):
        """
        現在の環境の状態 (ゴールとエージェントの位置) を描画する.

        Args:
            goals (list): ゴール位置のリスト [(x, y), ...].
            agents (list): エージェント位置のリスト [(x, y), ...].
        """
        # 既存の軸をクリア
        self.ax.clear()

        # 軸の設定を再度行う
        self.ax.set_xlim(0, self.grid_size)
        self.ax.set_ylim(0, self.grid_size)
        self.ax.set_aspect('equal', adjustable='box') # アスペクト比を固定
        self.ax.set_xticks(range(self.grid_size + 1))
        self.ax.set_yticks(range(self.grid_size + 1))
        self.ax.grid(True)
        self.ax.set_title("GridWorld") # タイトルを追加
        # self.ax.invert_yaxis() # Y軸を反転させて、(0,0)を左上にする（オプション）


        # 前回の描画要素をクリア (ax.clear() でクリアされるはずだが、念のためリストもクリア)
        self.agent_patches = []
        self.goal_patches = []

        # ゴールを描画 (正方形)
        # 凡例のために最初のゴールにラベルを付ける
        for i, goal_pos in enumerate(goals):
            label = "Goal" if i == 0 else None
            rect = patches.Rectangle(
                (goal_pos[0], goal_pos[1]), 1, 1,
                linewidth=1, edgecolor='black', facecolor='green', alpha=0.7, label=label
            )
            self.ax.add_patch(rect)
            self.goal_patches.append(rect)

        # エージェントを描画 (円)
        # 凡例のために最初のエージェントにラベルを付ける
        for i, agent_pos in enumerate(agents):
            label = "Agent" if i == 0 else None
            circle = patches.Circle(
                (agent_pos[0] + 0.5, agent_pos[1] + 0.5), 0.4, # 中心座標と半径
                linewidth=1, edgecolor='black', facecolor='blue', alpha=0.7, label=label
            )
            self.ax.add_patch(circle)
            self.agent_patches.append(circle)

        # 凡例を表示
        self.ax.legend()

        # 描画を更新
        self.fig.canvas.draw()
        #self.fig.canvas.flush_events() # インラインバックエンドでは不要な場合があります
        plt.pause(self.pause_duration) # 設定されたポーズ時間を使用

    def close(self):
        """
        描画ウィンドウを閉じる.
        """
        plt.close(self.fig)
'''