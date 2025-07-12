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
import torch
import os
import csv
import numpy as np

from env import GridWorld
from DQN.Agent_DQN import Agent_DQN
from utils.plot_results import PlotResults
from utils.Model_Saver import Saver


RED = '\033[91m'
GREEN = '\033[92m'
RESET = '\033[0m'

class MultiAgent_DQN:
    """
    複数のDQNエージェントを用いた強化学習の実行を管理するクラス.
    環境とのインタラクション、エピソードの進行、学習ループ、結果の保存・表示を統括します。
    """
    def __init__(self, args, agents:list[Agent_DQN]):
        """
        MultiAgent_DQN クラスのコンストラクタ.

        Args:
            args: 実行設定を含むオブジェクト.
                  (reward_mode, render_mode, episode_number, max_timestep,
                   agents_number, goals_num, grid_size, load_model, mask,
                   save_agent_states 属性を持つことを想定)
            agents (list[Agent_DQN]): 使用するエージェントオブジェクトのリスト.
        """
        self.env = GridWorld(args)
        self.agents = agents

        self.reward_mode = args.reward_mode
        self.render_mode = args.render_mode
        self.episode_num = args.episode_number
        self.max_ts = args.max_timestep
        self.agents_num = args.agents_number
        self.goals_num = args.goals_number
        self.grid_size = args.grid_size
        self.load_model = args.load_model
        self.mask = args.mask

        self.save_agent_states = args.save_agent_states

        # 結果保存ディレクトリの設定と作成
        save_dir = os.path.join(
            "output",
            f"DQN_mask[{args.mask}]_Reward[{args.reward_mode}]_env[{args.grid_size}x{args.grid_size}]_max_ts[{args.max_timestep}]_agents[{args.agents_number}]"
        )
        if not os.path.exists(save_dir): os.makedirs(save_dir)

        # 結果保存およびプロット関連クラスの初期化
        self.saver = Saver(save_dir)
        self.plot_results = PlotResults(save_dir)


    def run(self):
        """
        強化学習のメイン実行ループ.
        指定されたエピソード数だけ環境とのインタラクションと学習を行います。
        """
        # 事前条件チェック: ゴール数はエージェント数以下である必要がある
        if self.agents_num < self.goals_num:
            print('goals_num <= agents_num に設定してください.\n')
            sys.exit()

        # 学習開始メッセージ
        print(f"{GREEN}DQN{RESET} で学習中...\n")

        total_step = 0 # 環境との全インタラクションステップ数の累積
        avg_reward_temp, avg_step_temp = 0, 0 # 期間内の平均計算用一時変数
        achieved_episodes_temp = 0 # 集計期間内に目標を達成したエピソード数のカウント
        achieved_episodes_step = 0 # 集計期間内に目標を達成したステップ数

        # ----------------------------------
        # メインループ（各エピソード）
        # ----------------------------------
        for episode_num in range(1, self.episode_num + 1):
            print('■', end='',flush=True)  # 進捗表示 (エピソード100回ごとに改行)

            # 100エピソードごとに集計結果を出力
            if episode_num % 100 == 0:
                print() # 改行
                avg_reward = avg_reward_temp / 100 # 期間内の平均報酬
                avg_step = avg_step_temp / 100     # 期間内の平均ステップ数
                # 達成率を計算 (達成したエピソード数 / 集計エピソード数)
                achievement_rate = achieved_episodes_temp / 100
                
                if achieved_episodes_temp: 
                    avg_step_success=achieved_episodes_step/achieved_episodes_temp
                else:avg_step_success =-1
                avg_step = avg_step_temp / 100     # 期間内の平均ステップ数
                avg_step = avg_step_temp / 100     # 期間内の平均ステップ数
                print(f"==== エピソード {episode_num - 99} ~ {episode_num} の平均step(成功) : {GREEN}{avg_step_success}{RESET}")#/{GREEN}{avg_step}{RESET}")
                #print(f"==== エピソード {episode_num - 99} ~ {episode_num} の平均 reward: {GREEN}{avg_reward}{RESET}")
                print(f"==== エピソード {episode_num - 99} ~ {episode_num} の達成率: {GREEN}{achievement_rate:.2f}{RESET}\n") # 達成率も出力 .2f で小数点以下2桁表示
                # 集計変数をリセット
                avg_reward_temp, avg_step_temp = 0, 0
                achieved_episodes_temp = 0

            # --------------------------------------------
            # 各エピソード開始時に環境をリセット
            # これによりエージェントが再配置される
            # --------------------------------------------
            # 環境をリセットし、初期全体状態を取得
            current_global_state = self.env.reset()

            done = False # エピソード完了フラグ
            step_count = 0 # 現在のエピソードのステップ数
            ep_reward = 0.0 # 現在のエピソードの累積報酬

            # 各ステップで発生した学習損失を収集するためのリスト (オプション)
            # 逐次更新の場合、各ステップで学習が発生するわけではないため、
            # learn_from_experience が学習を実行した場合にのみ損失がリストに追加される。
            # このリストは、そのエピソード中に発生した学習ステップでの損失を記録する。
            losses_this_episode = []

            # ---------------------------
            # 1エピソードのステップループ
            # ---------------------------
            while not done and step_count < self.max_ts:
                # 各エージェントの行動を選択
                actions = []
                for i, agent in enumerate(self.agents):
                    # エージェントにε減衰を適用 (全ステップ数に基づき減衰)
                    agent.decay_epsilon_power(total_step,0.3)

                    # エージェントに行動を選択させる
                    # エージェント内部で自身の観測(masking)を行うため、全体状態を渡す
                    actions.append(agent.get_action(i, current_global_state))

                # エージェントの状態を保存（オプション）
                # 全体状態からエージェント部分を抽出し、Saverでログ記録
                if self.save_agent_states:
                     # current_global_state の構造に依存してエージェント部分を抽出
                     # モックの構造に合わせて、agent_pos は goals_num 以降とする
                    agent_positions_in_global_state = current_global_state[self.goals_num:]
                    for i, agent_pos in enumerate(agent_positions_in_global_state):
                        self.saver.log_agent_states(episode_num, step_count, i, agent_pos)

                # 環境にステップを与えて状態を更新し、結果を取得
                # 入力に現在の全体状態と全エージェントの行動を使用
                next_global_state, reward, done = self.env.step(current_global_state, actions)

                # 各ステップで獲得した報酬をエピソード報酬に加算
                ep_reward += reward

                # 各エージェントの経験をリプレイバッファにストアし、学習を試行
                for i, agent in enumerate(self.agents):
                    # エージェントは自身の経験 (状態s, 行動a, 報酬r, 次状態s', 終了フラグdone) をストア
                    # 状態sと次状態s'は環境全体の全体状態を渡す
                    agent.observe_and_store_experience(current_global_state, actions[i], reward, next_global_state, done)

                    # エージェントに学習を試行させる
                    # learn_from_experience はバッファサイズが満たされているなど、学習可能な場合に損失を返す
                    loss = agent.learn_from_experience(i, episode_num)
                    if loss is not None:
                        # 学習が発生した場合、その損失をリストに追加
                        losses_this_episode.append(loss)

                # 全体状態を次の状態に更新
                current_global_state = next_global_state

                step_count += 1 # エピソード内のステップ数をインクリメント
                total_step += 1 # 全体のステップ数をインクリメント

            # ---------------------------
            # エピソード終了後の処理
            # ---------------------------

            # エピソードが完了 (done == True) した場合、達成エピソード数カウンタをインクリメント
            if done:
                achieved_episodes_temp += 1
                achieved_episodes_step += step_count

            # エピソード中に発生した学習ステップでの平均損失を計算
            # losses_this_episode リストに収集された損失の平均
            valid_losses = [l for l in losses_this_episode if l is not None] # Noneでない損失のみをフィルタリング (learn_from_experienceがNoneを返す場合があるため)
            avg_loss_this_episode = sum(valid_losses) / len(valid_losses) if valid_losses else 0 # losses_this_episodeが空の場合は0

            # Saverでエピソードごとのスコアをログに記録
            # エピソード番号、最終ステップ数、累積報酬、エピソード中の平均損失を記録
            self.saver.log_scores(episode_num, step_count, ep_reward, avg_loss_this_episode)

            # 集計期間内の平均計算のための累積
            avg_reward_temp += ep_reward
            avg_step_temp += step_count

        print()  # 全エピソード終了後に改行

    def save_model_weights(self):
        """学習済みモデルの重みを保存する."""
        # モデル保存を Saver に依頼
        self.saver.save_dqn_weights(self.agents)

    def save_Qtable(self):
        """
        Qテーブルを保存する (NNベースの場合は近似計算が必要な場合がある).
        現在の実装はNNベースのため、このメソッドは必要ないか、
        Q関数からQ値を計算して保存するなどの実装が必要。
        """
        # エージェントがNNベースなのでQテーブルは保存しない (または、Q関数から近似的に計算して保存するなど)
        # 現在の実装では NN ベースなので save_Qtable は不要かもしれない。
        # モックとしては存在させる。
        print("MultiAgent_DQN: save_Qtable called (for NN based agents).")
        # self.saver.save_q_table(self.agents,self.mask) # Saverのモックを呼ぶ
        pass

    def result_show(self):
        """学習結果をプロットして表示する."""
        # PlotResults にプロットを依頼
        self.plot_results.draw()
        self.plot_results.draw_heatmap(self.grid_size)


# モックに必要な args オブジェクトを定義
# 実際の実行時には、適切な引数を持つ args オブジェクトが渡される必要があります。
class MockArgs:
    def __init__(self):
        self.grid_size = 5
        self.agents_number = 2
        self.goals_number = 4
        self.reward_mode = 0
        self.render_mode = False
        self.episode_number = 10
        self.max_timestep = 100
        self.load_model = 0
        self.mask = True
        self.save_agent_states = True
        self.buffer_size = 10000
        self.batch_size = 32
        self.optimizer = 'Adam'
        self.gamma = 0.99
        self.learning_rate = 0.001
        self.decay_epsilon = 10000 # ε減衰ステップ数のモック
        self.device = 'cpu' # または 'cuda' if torch.cuda.is_available() else 'cpu'
        self.target_update_frequency = 100
