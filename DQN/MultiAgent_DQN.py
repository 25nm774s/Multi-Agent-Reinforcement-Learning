"""
このファイルを実行して学習する．
python main.py --grid_size 10 のようにして各種設定を変更可能.
主要なハイパーパラメータは parse_args() で管理.

ゴール位置は初期に一度だけランダム生成し，エージェント位置のみエピソード毎に再生成するように修正済み。

--- リファクタリングに関する議論のまとめ (2024/05/20) ---
現在の Main クラスの処理を、よりオブジェクト指向的に構造化する方向性を議論。
主要な提案は以下の通り：
1.  マルチエージェントシステムの中核ロジックを MultiAgent クラスにカプセル化。
2.  MultiAgent クラス内に train() と evaluate() メソッドを分離。
3.  さらに進んで、学習アルゴリズム (Q, DQNなど) を AgentBase を継承したサブクラスとして実装し、MultiAgent がこれらを管理する（ポリモーフィズム活用）。

メリット：責務明確化、コード整理、再利用性/拡張性向上、テスト容易性、状態管理改善。
考慮点：学習/評価の処理分離、状態管理、引数渡し、モデル保存/読み込み、ログ/可視化、学習モードの扱い、共通インターフェース設計（サブクラス化の場合）。
目的：コードの保守性・拡張性を高め、将来的な機能追加（例: 新しいアルゴリズム）を容易にする。
---------------------------------------------------------

TODO: 学習モードごとの処理分岐(if self.learning_mode == ...)がMainクラスのrunメソッド内に混在しており、
      保守性・拡張性の観点から、学習モードごとに別のクラス(例: Q_Main, DQN_Main)に分割する設計を検討する。
      全体を制御するクラスで、どの学習モードのクラスを使うかを選択する形にする。

"""

# --- プログレスバー（進捗表示）に関する注意点と解決策 ---
# Colab環境ではリアルタイムに表示されるプログレスバー（例: '■'）が、
# ローカルPython環境で実行すると、処理が完了した後に一気に表示されてしまう場合があります。
# これは、Pythonの標準出力がパフォーマンス向上のために「バッファリング」されるためです。

# この問題を解決し、ローカル環境でもリアルタイムにプログレスバーを表示するための方法は以下の通りです。

# 1. print()関数の 'flush=True' 引数を使用する (最もシンプル)
#    - print()関数に 'flush=True' を追加すると、出力が即座に画面に書き出されます。
#    - 例: print('■', end='', flush=True)

# 2. sys.stdout.flush() を使用する (より柔軟な制御が必要な場合)
#    - print()以外の方法で出力している場合や、特定のタイミングでまとめてフラッシュしたい場合に有効です。
#    - import sys をファイルの先頭に追加し、出力後に sys.stdout.flush() を呼び出します。
#    - 例:
#      import sys
#      sys.stdout.write('■')
#      sys.stdout.flush()

# 3. tqdm ライブラリを使用する (推奨: より高機能で美しいプログレスバー)
#    - プログレスバーの表示に特化した外部ライブラリです。
#    - 内部で適切なフラッシュ処理が行われるため、Colabでもローカルでも期待通りに動作します。
#    - 残り時間推定などの追加機能も提供されます。
#    - インストール: pip install tqdm
#    - 使用例:
#      from tqdm import tqdm
#      for item in tqdm(iterable_object):
#          # 処理内容
#          pass
# --------------------------------------------------------

import sys
import argparse
import torch
import os
import csv
import numpy as np

from env import GridWorld
from DQN.Agent_DQN import Agent_DQN
from utils.plot_results import PlotResults

RED = '\033[91m'
GREEN = '\033[92m'
RESET = '\033[0m'

class MultiAgent_DQN:
    def __init__(self, args):
        self.env = GridWorld(args)
        self.reward_mode = args.reward_mode
        self.render_mode = args.render_mode
        self.episode_num = args.episode_number
        self.max_ts = args.max_timestep
        self.agents_num = args.agents_number
        self.goals_num = args.goals_number
        self.grid_size = args.grid_size
        self.load_model = args.load_model
        self.mask = args.mask

        self.OUT_FOLDER_NAME = "output"

        # OutputFile
        self.save_dir = (
            f"DQN_mask[{self.mask}]_RewardType[{self.reward_mode}]"
            f"_env[{self.grid_size}x{self.grid_size}]_max_ts[{self.max_ts}]_agents[{self.agents_num}]"
        )

        # モデル保存先のパス生成(あとでクラス分けしてここはなかったことになる)
        self.model_path = []
        for b_idx in range(self.agents_num):
            self.model_path.append(
                os.path.join(self.OUT_FOLDER_NAME, self.save_dir, 'model_weights', f"{b_idx}.pth")
            )

        # 結果保存先のパス生成
        self.scores_path = os.path.join(self.OUT_FOLDER_NAME, self.save_dir, "scores.csv")
        self.agents_states_path = os.path.join(self.OUT_FOLDER_NAME, self.save_dir, "agents_states.csv")

        # ディレクトリがなければ作成
        dir_for_agents_states = os.path.dirname(self.agents_states_path)
        if not os.path.exists(dir_for_agents_states):
            os.makedirs(dir_for_agents_states)

        self.plot_results = PlotResults(self.scores_path, self.agents_states_path)
        #self.clock = pygame.time.Clock()

        # エージェントの状態をcsvに保存するかどうか
        self.save_agent_states = args.save_agent_states
        if self.save_agent_states:
            with open(self.agents_states_path, 'w', newline='') as f:
                csv.writer(f).writerow(['episode', 'time_step', 'agent_id', 'agent_state'])

        # スコアファイルを初期化（ヘッダ書き込み）
        with open(self.scores_path, 'w', newline='') as f:
            csv.writer(f).writerow(['episode', 'time_step', 'reward', 'loss'])

    def run(self, agents):
        # 事前条件チェック
        if self.agents_num < self.goals_num:
            print('goals_num <= agents_num に設定してください.\n')
            sys.exit()

        # 学習開始メッセージ
        print(f"{GREEN}DQN{RESET} で学習中...\n")

        # ------------------------------------------------------------------
        # ゴールの位置は最初の一度だけ生成して固定 (object_positions_goals に保持)
        # ------------------------------------------------------------------
        object_positions_goals = []
        self.env.goals = self.env.generate_unique_positions(
            self.goals_num, object_positions_goals, self.grid_size
        )

        total_step = 0
        avg_reward_temp, avg_step_temp = 0, 0

        # ----------------------------------
        # メインループ（各エピソード）
        # ----------------------------------
        for episode_num in range(1, self.episode_num + 1):
            print('■', end='',flush=True)  # 進捗表示

            # 100エピソードごとに平均を出力
            if episode_num % 100 == 0:
                print()
                avg_reward = avg_reward_temp / 100
                avg_step = avg_step_temp / 100
                print(f"==== エピソード {episode_num - 99} ~ {episode_num} の平均 step  : {GREEN}{avg_step}{RESET}")
                print(f"==== エピソード {episode_num - 99} ~ {episode_num} の平均 reward: {GREEN}{avg_reward}{RESET}\n")
                avg_reward_temp, avg_step_temp = 0, 0

            # --------------------------------------------
            # ここで各エピソードごとにエージェントを再配置
            # ゴール位置は固定のobject_positions_goalsをコピー
            # --------------------------------------------
            object_positions = object_positions_goals.copy()

            # エージェントの位置をランダム生成(ゴール座標との重複回避)
            self.env.agents = self.env.generate_unique_positions(
                self.agents_num, object_positions, self.grid_size
            )

            # states にはゴール + エージェントが一続きに入る
            # (ゴールの位置1(x,y),ゴールの位置2(x,y)...ゴールの位置N(x,y),エージェント自身の位置1(x,y),エージェント自身の位置2(x,y)...エージェント自身の位置N(x,y))
            #環境の現在の状態を表す大きなテンソル
            states = tuple(object_positions)

            #print(f"states:{states}")

            done = False
            step_count = 0
            ep_reward = 0

            # ---------------------------
            # 1エピソードのステップループ
            # ---------------------------
            while not done and step_count < self.max_ts:
                actions = []
                for i, agent in enumerate(agents):
                    agent.decay_epsilon(total_step)

                    actions.append(agent.get_action(i, states))

                # エージェントの状態を保存（オプション）
                if self.save_agent_states:
                    for i, pos in enumerate(states[self.goals_num:]):
                        self.log_agent_states(episode_num, step_count, i, pos)

                # 環境にステップを与えて状態を更新
                next_state, reward, done = self.env.step(states, actions)

                # DQNは逐次更新
                losses = []
                for i, agent in enumerate(agents):
                    losses.append(agent.update_brain(
                        i, states, actions[i], reward,
                        next_state, done, episode_num
                    ))

                states = next_state
                ep_reward += reward

                step_count += 1
                total_step += 1

            # エピソード終了
            valid_losses = [l for l in losses if l is not None]
            avg_loss = sum(valid_losses) / len(valid_losses) if valid_losses else 0

            # ログにスコアを記録
            self.log_scores(episode_num, step_count, ep_reward, avg_loss)

            avg_reward_temp += ep_reward
            avg_step_temp += step_count

        print()  # 終了時に改行

        # モデル保存やプロット
        if self.load_model == 0:
            self.save_model(agents)
            self.plot_results.draw()
        elif self.load_model == 2:
            self.plot_results.draw()

        if self.save_agent_states:
            self.plot_results.draw_heatmap(self.grid_size)

    def log_scores(self, episode, time_step, reward, loss):
        with open(self.scores_path, 'a', newline='') as f:
            csv.writer(f).writerow([episode, time_step, reward, loss])

    def log_agent_states(self, episode, time_step, agent_id, agent_state):
        if isinstance(agent_state, (list, tuple, np.ndarray)):
            state_str = '_'.join(map(str, agent_state))
        else:
            state_str = str(agent_state)
        with open(self.agents_states_path, 'a', newline='') as f:
            csv.writer(f).writerow([episode, time_step, agent_id, state_str])

    def save_model(self, agents):
        model_dir_path = os.path.join(self.OUT_FOLDER_NAME, self.save_dir,'model_weights')
        if not os.path.exists(model_dir_path):
            os.makedirs(model_dir_path)

        print('モデル保存中...')
        for i, agent in enumerate(agents):
            torch.save(agent.model.qnet.state_dict(), self.model_path[i])
        print(f"保存先: {GREEN}{model_dir_path}{RESET}\n")


if __name__ == '__main__':
    def parse_args():
        parser = argparse.ArgumentParser()
        #parser.add_argument('--dir_path', default='/Users/ryohei_nakano/Desktop/研究コード/orig_rl_ver4.3')
        parser.add_argument('--grid_size', default=8, type=int)
        parser.add_argument('--agents_number', default=2, type=int)
        parser.add_argument('--goals_number', default=2, type=int)
        #parser.add_argument('--learning_mode', choices=['V', 'Q', 'DQN'], default='DQN')
        parser.add_argument('--optimizer', choices=['Adam', 'RMSProp'], default='Adam')
        parser.add_argument('--mask', choices=[0, 1], default=0, type=int)
        parser.add_argument('--load_model', choices=[0, 1, 2], default=1, type=int)
        parser.add_argument('--reward_mode', choices=[0, 1, 2], default=0, type=int)
        parser.add_argument('--device', choices=['auto', 'cpu', 'cuda', 'mps'], default='auto') # 'auto'を追加し、デフォルトを'auto'に変更
        parser.add_argument('--episode_number', default=5000, type=int)
        parser.add_argument('--max_timestep', default=100, type=int)
        parser.add_argument('--decay_epsilon', default=500000, type=int)
        parser.add_argument('--learning_rate', default=0.000005, type=float)
        parser.add_argument('--gamma', default=0.95, type=float)
        parser.add_argument('--buffer_size', default=10000, type=int)
        parser.add_argument('--batch_size', default=2, type=int)
        parser.add_argument('--save_agent_states', choices=[0, 1], default=1, type=int)
        parser.add_argument('--window_width', default=500, type=int)
        parser.add_argument('--window_height', default=500, type=int)
        parser.add_argument('--render_mode', choices=[0, 1], default=0, type=int)
        return parser.parse_args()

    args = parse_args()

    # auto選択時のデバイス決定ロジックを追加
    if args.device == 'auto':
        if torch.cuda.is_available():
            args.device = 'cuda'
        elif torch.backends.mps.is_available():
            args.device = 'mps'
        else:
            args.device = 'cpu'
        print(f"自動選択されたデバイス: {GREEN}{args.device}{RESET}\n")


    ma = MultiAgent_DQN(args)
    agents = [Agent_DQN(args, ma.model_path[b_idx]) for b_idx in range(args.agents_number)]
    ma.run(agents)