"""
行動の軌跡を保存したcsvを元に, 
・100 episode 毎の報酬の平均値
・100 episode 毎のゴールに要するステップ数の平均値
・100 episode 毎の損失の平均
を導出し, その推移をプロットする.
また，ヒートマップの出力も担当.
"""

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

class PlotResults:
    def __init__(self, scores_path, agents_states_path):
        self.scores_path = scores_path
        self.agents_states_path = agents_states_path

    # CSVファイルの読み込み
    def road_csv(self):
        data = pd.read_csv(self.scores_path)

        # 100エピソードごとのstep, reward, lossの平均値
        data['episode_group'] = (data['episode'] - 1) // 100
        grouped_data = data.groupby('episode_group').mean()
        
        plt.figure(figsize=(12, 4))
        
        return grouped_data

    def draw(self):
        grouped_data = self.road_csv()

        # ステップ数の平均値の推移
        plt.subplot(1, 3, 1)
        plt.plot(grouped_data.index * 100 + 50, grouped_data['time_step'], marker='o', markersize=4)
        plt.xlabel('Episode')
        plt.ylabel('Average Steps')
        plt.title('Average Steps per 100 Episodes')
        plt.grid(True)

        # 報酬の平均値の推移
        plt.subplot(1, 3, 2)
        plt.plot(grouped_data.index * 100 + 50, grouped_data['reward'], marker='o', markersize=4)
        plt.xlabel('Episode')
        plt.ylabel('Average Reward')
        plt.title('Average Reward per 100 Episodes')
        plt.grid(True)

        # 損失の平均値の推移
        plt.subplot(1, 3, 3)
        plt.plot(grouped_data.index * 100 + 50, grouped_data['loss'], marker='o', markersize=4)
        plt.xlabel('Episode')
        plt.ylabel('Average Loss')
        plt.title('Average Loss per 100 Episodes')
        plt.grid(True)
        
        # プロット結果を保存するためにpathの拡張子を変更
        plot_path_pdf = self.scores_path.replace('.csv', '.pdf')
        
        plt.tight_layout()
        plt.savefig(plot_path_pdf, format='pdf', dpi=300)
        plt.show()

    def draw_heatmap(self, cell_num):
        data = pd.read_csv(self.agents_states_path)

        label_fontsize = 18  # ラベルのフォントサイズ
        title_fontsize = 18  # タイトルのフォントサイズ

        if 'agnet_state' in data.columns:
            data.rename(columns={'agnet_state': 'agent_state'}, inplace=True)

        # 最大のエージェントインデックスを確認
        max_agent_id = data['agent_id'].max()

        # エージェントごとのヒートマップデータを格納するディクショナリ
        heatmaps = {agent_id: [[0] * cell_num for _ in range(cell_num)] for agent_id in range(max_agent_id + 1)}

        # 各エージェントの状態をカウントしてヒートマップデータに追加
        for _, row in data.iterrows():
            agent_id = row['agent_id']
            try:
                # 状態を解析
                pos_str = row['agent_state'].split('_')
                x = int(pos_str[0])
                y = int(pos_str[1])
                if 0 <= x < cell_num and 0 <= y < cell_num:
                    heatmaps[agent_id][y][x] += 1
            except ValueError:
                continue  # 無効な状態は無視
        
        # ヒートマップをエージェントごとに生成・表示
        fig, axes = plt.subplots(1, max_agent_id + 1, figsize=(6 * (max_agent_id + 1), 5))
        for i, (agent_id, heatmap_data) in enumerate(heatmaps.items()):
            ax = axes[i]
            ax.set_title(f'Agent {agent_id} Visit Frequency Heatmap', fontsize=title_fontsize)
            im = ax.imshow(heatmap_data, origin='lower', cmap='plasma', aspect='auto')
            cbar = fig.colorbar(im, ax=ax, label='Visit Count')
            #ax.set_xlabel('X', fontsize=label_fontsize)
            #ax.set_ylabel('Y', fontsize=label_fontsize)

        output_path = self.agents_states_path.replace('.csv', '_agents_heatmap.pdf')
        plt.tight_layout()
        plt.savefig(output_path, format='pdf', dpi=300)
        plt.show()
        plt.close()