import matplotlib.pyplot as plt
import os
import pandas as pd
import numpy as np

class PlotResults:
    """
    学習結果のプロットを管理するクラス.

    エピソードごとの生データ一括処理を避け、100 エピソードごとの集計を逐次ファイルに保存する形式に改善。
    エージェントの訪問状態を記録し、ヒートマップとして可視化する機能も提供します。
    保存処理は Saver クラスに委譲されます。
    """
    def __init__(self, scores_summary_path: str, visited_coordinates_path: str):
 
        # save_dir を使用して PlotResults 内でファイルパスを定義
        # scores_summary100.csv:集計済みファイル
        # self.scores_summary_100_path = os.path.join(self.save_dir, "aggregated_episode_metrics_100.csv")
        self.scores_summary_100_path = scores_summary_path

        # visited_coordinates.npy をヒートマップデータファイルとする
        # self.visited_coordinates_path = os.path.join(self.save_dir, "visited_coordinates.npy")
        self.visited_coordinates_path = visited_coordinates_path

    def road_csv(self):
        # scores_summary100.csv から読み込むように更新
        if not os.path.exists(self.scores_summary_100_path):
            print(f"警告: 要約スコアファイルが見つかりません: {self.scores_summary_100_path}")
            return pd.DataFrame()
        # 集計データの正しい列名を確認
        data = pd.read_csv(self.scores_summary_100_path)
        return data

    def draw(self):
        # scores_summary100.csv パスを使用するように更新
        if not os.path.exists(self.scores_summary_100_path):
            print(f"エラー: 要約スコアファイルが見つかりません: {self.scores_summary_100_path}")
            return
        # 集計データの正しい列名を確認
        grouped_data = pd.read_csv(self.scores_summary_100_path)
        if grouped_data.empty:
            print("プロットする要約スコアデータがありません。")
            return
        plt.figure(figsize=(18, 5))

        # エピソードごとの平均ステップ数をプロット
        plt.subplot(1, 3, 1)
        # scores_summary100.csv に合うように列名を更新
        plt.plot(grouped_data['episode_group_end'], grouped_data['avg_time_step_100'], marker='o', markersize=4)
        plt.xlabel('Episode Group End')
        plt.ylabel('Average Time Steps')
        plt.title('Average Time Steps per 100 Episodes')
        plt.grid(True)

        # エピソードごとの平均報酬をプロット
        plt.subplot(1, 3, 2)
        # scores_summary100.csv に合うように列名を更新
        plt.plot(grouped_data['episode_group_end'], grouped_data['avg_reward_100'], marker='o', markersize=4)
        plt.xlabel('Episode Group End')
        plt.ylabel('Average Reward')
        plt.title('Average Reward per 100 Episodes')
        plt.grid(True)

        # エピソードごとの平均損失をプロット
        plt.subplot(1, 3, 3)
        # scores_summary100.csv に合うように列名を更新
        plt.plot(grouped_data['episode_group_end'], grouped_data['avg_loss_100'], marker='o', markersize=4)
        plt.xlabel('Episode Group End')
        plt.ylabel('Average Loss')
        plt.title('Average Loss per 100 Episodes')
        plt.grid(True)

        # 出力ファイル名を更新
        plot_path_pdf = self.scores_summary_100_path.replace('.csv', '.pdf')
        plt.tight_layout()
        plt.savefig(plot_path_pdf, format='pdf', dpi=300)
        plt.close()

    def draw_heatmap(self):
        # visited_coordinates.npy パスを使用するように更新
        if not os.path.exists(self.visited_coordinates_path):
            print(f"エラー: エージェント状態ファイルが見つかりません: {self.visited_coordinates_path}")
            return
        if os.path.exists(self.visited_coordinates_path) and os.stat(self.visited_coordinates_path).st_size == 0:
            print(f"警告: エージェント状態ファイルが空です: {self.visited_coordinates_path}")
            return
        try:
            # .npy ファイルからデータをロード
            visited_count_grid = np.load(self.visited_coordinates_path)
        except FileNotFoundError:
            print(f"エラー: エージェント状態ファイルが見つかりません: {self.visited_coordinates_path}")
            return
        except Exception as e:
            print(f"エージェント状態ファイルの読み込みエラー: {e}")
            return

        if visited_count_grid.size == 0:
            print("ヒートマップを生成するエージェント状態データがありません。")
            return

        # visited_count_grid が NxN 配列であると仮定
        cell_num_grid = visited_count_grid.shape[0] # ロードされた配列からグリッドサイズを取得
        if cell_num_grid != visited_count_grid.shape[1]:
            print(f"エラー: ロードされた訪問回数グリッドの形状が無効です: {visited_count_grid.shape}")
            return

        label_fontsize = 14
        title_fontsize = 16
        # 全体の訪問回数を表す単一のヒートマップの場合、ここでは agent_id ロジックは不要
        # 集計された訪問回数に対して単一のヒートマップを生成

        # 単一ヒートマップのプロットロジック
        fig, ax = plt.subplots(1, 1, figsize=(6, 5))

        ax.set_title('Total Agent Visit Frequency Heatmap', fontsize=title_fontsize)
        # Use extent to align grid lines with cell boundaries
        im = ax.imshow(visited_count_grid, origin='lower', cmap='plasma', extent = (-0.5, float(cell_num_grid - 0.5), -0.5, float(cell_num_grid - 0.5)))
        cbar = fig.colorbar(im, ax=ax, label='Visit Count')
        ax.set_xlabel('X', fontsize=label_fontsize)
        ax.set_ylabel('Y', fontsize=label_fontsize)

        # Set major ticks at integer positions (0, 1, 2, ...) for labels
        ax.set_xticks(np.arange(0, cell_num_grid, 1))
        ax.set_yticks(np.arange(0, cell_num_grid, 1))

        # Set tick labels to be integer positions (0, 1, 2, ...)
        ax.set_xticklabels(np.arange(0, cell_num_grid, 1))
        ax.set_yticklabels(np.arange(0, cell_num_grid, 1))

        # Set minor ticks at half-integer positions for grid lines
        ax.set_xticks(np.arange(-0.5, cell_num_grid, 1), minor=True)
        ax.set_yticks(np.arange(-0.5, cell_num_grid, 1), minor=True)

        # Enable grid lines for minor ticks (boundaries)
        ax.grid(which='minor', color='white', linestyle='-', linewidth=0.5)

        # Optionally, disable grid lines for major ticks if they are distracting
        ax.grid(which='major', color='none', linestyle='None')


        output_path = self.visited_coordinates_path.replace('.npy', '_heatmap.pdf') # 出力ファイル名を更新
        plt.tight_layout()
        plt.savefig(output_path, format='pdf', dpi=300)
        plt.close()
