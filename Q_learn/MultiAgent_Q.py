import sys
import argparse
import torch
import os
import csv
import numpy as np

from env import GridWorld
from Q_learn.Agent_Q import Agent_Q
from utils.Model_Saver import Saver
from utils.plot_results import PlotResults

RED = '\033[91m'
GREEN = '\033[92m'
RESET = '\033[0m'

class MultiAgent_Q:
    def __init__(self, args, agents:Agent_Q, saver:Saver):
        self.env = GridWorld(args) # GridWorldインスタンス生成時にゴール位置は固定生成される
        self.agents = agents

        #self.learning_mode = args.learning_mode
        self.reward_mode = args.reward_mode
        self.render_mode = args.render_mode
        self.episode_num = args.episode_number
        self.max_ts = args.max_timestep
        self.agents_num = args.agents_number
        self.goals_num = args.goals_number
        self.grid_size = args.grid_size
        #self.dir_path = args.dir_path
        self.load_model = args.load_model
        self.mask = args.mask

        # 結果保存先のパス生成
        self.save_dir = os.path.join(
            "output",
            f"DQN_mask[{self.mask}]_Reward[{self.reward_mode}]_env[{self.grid_size}x{self.grid_size}]_max_ts[{self.max_ts}]_agents[{self.agents_num}]"
        )

        self.scores_path = os.path.join(self.save_dir, "scores.csv")
        self.agents_states_path = os.path.join(self.save_dir, "agents_states.csv")

        self.saver = saver
        self.plot_results = PlotResults(self.scores_path, self.agents_states_path)

        # エージェントの状態をcsvに保存するかどうか(簡単のため常に1)
        #self.save_agent_states = args.save_agent_states

        # TODO: Saverクラスの導入を検討
        # 現在、ファイルパス生成、ディレクトリ作成、csv書き込みロジックがMultiAgent_Qに混在している。
        # これらのデータ永続化に関する責務をSaverクラスに切り出すことで、MultiAgent_Qクラスを学習のメインループ制御に専念させ、コードの見通しと保守性を向上させる。
        # Saverクラスは、モデルパラメータ、学習ログ、エージェント状態の保存メソッドを持つ。
        # MultiAgent_QはSaverインスタンスを持ち、適切なタイミングでSaverのメソッドを呼び出す。

    def run(self):

        # 事前条件チェック
        if self.agents_num < self.goals_num:
            print('goals_num <= agents_num に設定してください.\n')
            sys.exit()

        # 学習開始メッセージ
        if self.mask:
            print(f"{GREEN}IQLで学習中{RESET}\n")
        else:
            print(f"{GREEN}CQLで学習中{RESET}\n")

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
            # 各エピソード開始時に環境をリセット
            # これによりエージェントが再配置される
            # --------------------------------------------
            states = self.env.reset()

            done = False
            step_count = 0
            ep_reward = 0

            # ---------------------------
            # 1エピソードのステップループ
            # ---------------------------
            while not done and step_count < self.max_ts:
                actions = []
                for i, agent in enumerate(self.agents):
                    agent.decay_epsilon(total_step)

                    # エージェントに行動を選択させる際に、現在の状態(states)全体を渡す
                    # エージェント内部で自身の観測(masking)を行う
                    actions.append(agent.get_action(i, states))

                # エージェントの状態を保存（オプション）
                #if self.save_agent_states:
                # states はゴール位置 + エージェント位置のタプルになっている
                # エージェントの位置は states の self.goals_num 以降
                # TODO: GridWorldのget_agent_positionsメソッドの使用を検討
                for i, pos in enumerate(states[self.goals_num:]):
                    #self.log_agent_states(episode_num, step_count, i, pos)
                    self.saver.log_agent_states(episode_num, step_count, i, pos)

                # 環境にステップを与えて状態を更新
                next_state, reward, done = self.env.step(states, actions)

                # 状態価値関数学習以外(Q, DQN)は逐次更新
                losses = []
                for i, agent in enumerate(self.agents):
                    # エージェントは自身の経験(状態s, 行動a, 報酬r, 次状態s', 終了フラグdone)をストア
                    # ここでも状態sと次状態s'は環境全体の状態を渡す
                    agent.observe_and_store_experience(states, actions[i], reward, next_state, done)

                    # 学習は別のタイミングでトリガー
                    # 学習時にも環境全体の状態を渡す必要があるか、エージェントが自身の観測範囲で学習するかは
                    # Agent_Qクラスの実装に依存
                    loss = agent.learn_from_experience(i, episode_num)
                    if loss is not None:
                        losses.append(loss)

                states = next_state # 状態を更新
                ep_reward += reward

                step_count += 1
                total_step += 1

            # エピソード終了
            # lossesがNoneでないものだけを抽出して平均を計算
            valid_losses = [l for l in losses if l is not None]
            avg_loss = sum(valid_losses) / len(valid_losses) if valid_losses else 0

            # ログにスコアを記録
            # TODO: Saverクラスのlog_scoresメソッドに置き換え
            #self.log_scores(episode_num, step_count, ep_reward, avg_loss)
            self.saver.log_scores(episode_num, step_count, ep_reward, avg_loss)

            avg_reward_temp += ep_reward
            avg_step_temp += step_count

        print()  # 終了時に改行

    
        # モデル保存やプロット
        if self.load_model == 0:
            # TODO: Saverクラスのsave_modelメソッドに置き換え
            self.save_model(self.agents)
            self.plot_results.draw()
            #self.saver.save_model(self.agents)
        elif self.load_model == 1:
            self.plot_results.draw()

        #if self.save_agent_states:
        self.plot_results.draw_heatmap(self.grid_size)
        #self.plot_results.draw_heatmap(self.grid_size)
    
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

    def save_model(self,agents):
        print("あとで実装")

"""
    ma = MultiAgent_Q(args)
    # Agent_Qの初期化時に、モデルパスは各エージェント固有 or 共通で渡す
    # Agent_Qクラス内でモデルのロード処理を行う必要がある
    agents = [Agent_Q(args, ma.model_path[b_idx]) for b_idx in range(args.agents_number)]
    ma.run(agents)
"""