import sys
import argparse
import torch
import os
import csv
import numpy as np

from env import GridWorld
from Q_learn.Agent_Q import Agent_Q
from utils.plot_results import PlotResults

RED = '\033[91m'
GREEN = '\033[92m'
RESET = '\033[0m'

class MultiAgent_Q:
    def __init__(self, args):
        self.env = GridWorld(args) # GridWorldインスタンス生成時にゴール位置は固定生成される
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

        self.OUT_FOLDER_NAME = "output"

        # 保存ファイル名
        #self.f_name = (
        #    f"{self.learning_mode}_mask[{self.mask}]_RewardType[{self.reward_mode}]"
        #    f"_env[{self.grid_size}x{self.grid_size}]_agents[{self.agents_num}]_goals[{self.goals_num}]"
        #)
        # OutputFile
        self.save_dir = (
            f"Q_mask[{self.mask}]_RewardType[{self.reward_mode}]"
            f"_env[{self.grid_size}x{self.grid_size}]_agents[{self.agents_num}]_goals[{self.goals_num}]"
        )

        # モデル保存先のパス生成(あとでクラス分けしてここはなかったことになる)
        self.model_path = []
        for b_idx in range(self.agents_num):
            if self.mask:
                self.model_path.append(
                    os.path.join(self.OUT_FOLDER_NAME, self.save_dir, 'model_weights', f"{b_idx}.csv")
                )
            else:
                self.model_path.append(
                    os.path.join(self.OUT_FOLDER_NAME, self.save_dir, 'model_weights', "common.csv")
                )

        # 結果保存先のパス生成
        self.scores_path = os.path.join(self.OUT_FOLDER_NAME, self.save_dir, "scores.csv")
        self.agents_states_path = os.path.join(self.OUT_FOLDER_NAME, self.save_dir, "agents_states.csv")

        # ディレクトリがなければ作成
        dir_for_agents_states = os.path.dirname(self.agents_states_path)
        if not os.path.exists(dir_for_agents_states):
            os.makedirs(dir_for_agents_states)

        self.plot_results = PlotResults(self.scores_path, self.agents_states_path)

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
                for i, agent in enumerate(agents):
                    agent.decay_epsilon(total_step)

                    # エージェントに行動を選択させる際に、現在の状態(states)全体を渡す
                    # エージェント内部で自身の観測(masking)を行う
                    actions.append(agent.get_action(i, states))

                # エージェントの状態を保存（オプション）
                if self.save_agent_states:
                    # states はゴール位置 + エージェント位置のタプルになっている
                    # エージェントの位置は states の self.goals_num 以降
                    for i, pos in enumerate(states[self.goals_num:]):
                        self.log_agent_states(episode_num, step_count, i, pos)

                # 環境にステップを与えて状態を更新
                next_state, reward, done = self.env.step(states, actions)

                # 状態価値関数学習以外(Q, DQN)は逐次更新
                losses = []
                for i, agent in enumerate(agents):
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

        print('パラメータ保存中...')
        for i, agent in enumerate(agents):
            path = (os.path.join(model_dir_path, f'{i}.csv')
                    if self.mask else
                    os.path.join(model_dir_path, 'common.csv'))
            with open(path, 'w', newline='') as f:
                writer = csv.writer(f)
                # Agent_Qクラスが持つ学習パラメータを取得する必要がある
                # 例: agent.get_model_params() のようなメソッドをAgent_Qに追加する必要があるかもしれない
                # 現在のコードではagent.linear.theta_listやagent.linear.common_theta_listを参照しているが、
                # Agent_Qクラスの内部実装に依存しすぎている
                # Agent_Qクラスにパラメータを公開するメソッドを追加するか、
                # ここで直接アクセス可能な構造になっているか確認が必要
                try:
                    if self.mask:
                        # IQLの場合、各エージェントが独自のthetaを持つと仮定
                        # Agent_Qクラスに self.theta_list を持たせる必要がある
                        data = agent.theta_list
                    else:
                        # CQLの場合、共通のthetaを持つと仮定
                        # Agent_Qクラスに self.common_theta_list を持たせる必要がある
                        # そして、それを2次元にリシェイプして保存
                        arr = np.array(agent.common_theta_list)
                        data = arr.reshape(-1, arr.shape[2]) if arr.ndim > 2 else arr
                except AttributeError as e:
                     print(f"エラー: Agent_Qクラスに学習パラメータを保持する変数がないか、名前が異なります: {e}")
                     print("Agent_Qクラスの実装を確認し、学習パラメータが self.theta_list または self.common_theta_list として保持されているか確認してください。")
                     return # 保存処理を中断

                for row in data:
                    writer.writerow(row)
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
        parser.add_argument('--learning_rate', default=0.000005, type=float) # ここを修正: add_number -> add_argument
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

    ma = MultiAgent_Q(args)
    # Agent_Qの初期化時に、モデルパスは各エージェント固有 or 共通で渡す
    # Agent_Qクラス内でモデルのロード処理を行う必要がある
    agents = [Agent_Q(args, ma.model_path[b_idx]) for b_idx in range(args.agents_number)]
    ma.run(agents)