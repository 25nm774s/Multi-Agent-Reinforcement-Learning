"""
このファイルを実行して学習する．
python main.py --cell_num 10 のようにして各種設定を変更可能.
主要なハイパーパラメータは parse_args() で管理.

ゴール位置は初期に一度だけランダム生成し，エージェント位置のみエピソード毎に再生成するように修正済み。
"""

import sys
import argparse
import pygame
import torch
import os
import csv
import numpy as np

from env import GridWorld
from agent import Agent
from plot_results import PlotResults

RED = '\033[91m'
GREEN = '\033[92m'
RESET = '\033[0m'

class Main:
    def __init__(self, args):
        self.env = GridWorld(args)
        self.learning_mode = args.learning_mode
        self.reward_mode = args.reward_mode
        self.render_mode = args.render_mode
        self.episode_num = args.episode_number
        self.max_ts = args.max_timestep
        self.agents_num = args.agents_number
        self.goals_num = args.goals_number
        self.cell_num = args.cell_number
        self.dir_path = args.dir_path
        self.load_model = args.load_model
        self.mask = args.mask

        # 保存ファイル名
        self.f_name = (
            f"{self.learning_mode}_mask[{self.mask}]_RewardType[{self.reward_mode}]"
            f"_env[{self.cell_num}*{self.cell_num}]_agents[{self.agents_num}]_goals[{self.goals_num}]"
        )

        # モデル保存先のパス生成
        self.model_path = []
        for b_idx in range(self.agents_num):
            if self.learning_mode in ['V', 'Q']:
                if self.mask:
                    self.model_path.append(
                        os.path.join(self.dir_path, 'model_weights', self.f_name, f"{b_idx}.csv")
                    )
                else:
                    self.model_path.append(
                        os.path.join(self.dir_path, 'model_weights', self.f_name, "common.csv")
                    )
            else:
                self.model_path.append(
                    os.path.join(self.dir_path, 'model_weights', self.f_name, f"{b_idx}.pth")
                )

        # 結果保存先のパス生成
        self.scores_path = os.path.join(self.dir_path, 'results', f"{self.f_name}_scores.csv")
        self.agents_states_path = os.path.join(self.dir_path, 'results', 'positions', f"{self.f_name}_agents_states.csv")

        # ディレクトリがなければ作成
        dir_for_agents_states = os.path.dirname(self.agents_states_path)
        if not os.path.exists(dir_for_agents_states):
            os.makedirs(dir_for_agents_states)

        self.plot_results = PlotResults(self.scores_path, self.agents_states_path)
        self.clock = pygame.time.Clock()

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
        if self.load_model == 0 and self.learning_mode == 'V':
            print('load_model == 0 と learning_mode == V の組み合わせは未実装です.\n')
            sys.exit()
        if self.load_model == 2 and self.learning_mode == 'V' and self.mask:
            print('mask == 0 に設定してください.\n')
            sys.exit()

        # 学習開始メッセージ
        if self.load_model in [0, 2]:
            if self.learning_mode == 'Q':
                print(f"{GREEN}{'IQL' if self.mask else 'CQL'} で学習中{RESET}\n")
            elif self.learning_mode == 'V' and self.load_model == 2:
                print('状態価値関数を価値共有で学習中\n')
            else:
                print(f"{GREEN}{self.learning_mode}{RESET} で学習中...\n")

        # ------------------------------------------------------------------
        # ゴールの位置は最初の一度だけ生成して固定 (object_positions_goals に保持)
        # ------------------------------------------------------------------
        object_positions_goals = []
        self.env.goals = self.env.generate_unique_positions(
            self.goals_num, object_positions_goals, self.cell_num
        )

        total_step = 0
        avg_reward_temp, avg_step_temp = 0, 0

        # ----------------------------------
        # メインループ（各エピソード）
        # ----------------------------------
        for episode_num in range(1, self.episode_num + 1):
            print('■', end='')  # 進捗表示

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
                self.agents_num, object_positions, self.cell_num
            )

            # states にはゴール + エージェントが一続きに入る
            states = tuple(object_positions)

            if self.render_mode:
                self.env.render(episode_num)
                self.clock.tick(50)

            done = False
            step_count = 0
            ep_reward = 0

            # ---------------------------
            # 1エピソードのステップループ
            # ---------------------------
            while not done and step_count < self.max_ts:
                actions = []
                for i, agent in enumerate(agents):
                    # load_model == 1 → 学習済みモデル (epsilon=0.1)
                    if self.load_model == 1:
                        agent.epsilon = 0.1
                    elif self.learning_mode == 'V':
                        agent.epsilon = 1.0
                    else:
                        agent.decay_epsilon(total_step)

                    actions.append(agent.get_action(i, states))

                # エージェントの状態を保存（オプション）
                if self.save_agent_states:
                    for i, pos in enumerate(states[self.goals_num:]):
                        self.log_agent_states(episode_num, step_count, i, pos)

                # 環境にステップを与えて状態を更新
                next_state, reward, done = self.env.step(states, actions, step_count)

                # 状態価値関数学習以外(Q, DQN)は逐次更新
                losses = []
                if self.learning_mode != 'V':
                    for i, agent in enumerate(agents):
                        losses.append(agent.update_brain(
                            i, states, actions[i], reward,
                            next_state, done, episode_num, step_count
                        ))

                states = next_state
                ep_reward += reward

                step_count += 1
                total_step += 1

                if self.render_mode:
                    self.env.render(episode_num, step_count)
                    self.clock.tick(50)

            # エピソード終了後，状態価値の更新 (V 学習時のみ)
            if self.learning_mode == 'V':
                avg_loss = agents[0].update_brain(0, states, None, reward, next_state,
                                                  done, episode_num, step_count)
            else:
                valid_losses = [l for l in losses if l is not None]
                avg_loss = sum(valid_losses) / len(valid_losses) if valid_losses else 0

            # ログにスコアを記録
            self.log_scores(episode_num, step_count, ep_reward, avg_loss)

            avg_reward_temp += ep_reward
            avg_step_temp += step_count

        print()  # 終了時に改行
        pygame.quit()

        # モデル保存やプロット
        if self.load_model == 0:
            self.save_model(agents)
            self.plot_results.draw()
        elif self.load_model == 2:
            self.plot_results.draw()

        if self.save_agent_states:
            self.plot_results.draw_heatmap(self.cell_num)

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
        model_dir_path = os.path.join(self.dir_path, 'model_weights', self.f_name)
        if not os.path.exists(model_dir_path):
            os.makedirs(model_dir_path)

        if self.learning_mode in ['V', 'Q']:
            print('パラメータ保存中...')
            for i, agent in enumerate(agents):
                path = (os.path.join(model_dir_path, f'{i}.csv')
                        if self.mask else
                        os.path.join(model_dir_path, 'common.csv'))
                with open(path, 'w', newline='') as f:
                    writer = csv.writer(f)
                    if self.mask:
                        data = agent.linear.theta_list
                    else:
                        arr = np.array(agent.linear.common_theta_list)
                        data = arr.reshape(-1, arr.shape[2])  # 2次元にリシェイプ
                    for row in data:
                        writer.writerow(row)
            print(f"保存先: {GREEN}{model_dir_path}{RESET}\n")
        else:
            print('モデル保存中...')
            for i, agent in enumerate(agents):
                torch.save(agent.model.qnet.state_dict(), self.model_path[i])
            print(f"保存先: {GREEN}{model_dir_path}{RESET}\n")


if __name__ == '__main__':
    def parse_args():
        parser = argparse.ArgumentParser()
        parser.add_argument('--dir_path', default='/Users/ryohei_nakano/Desktop/研究コード/orig_rl_ver4.3')
        parser.add_argument('--cell_number', default=8, type=int)
        parser.add_argument('--agents_number', default=2, type=int)
        parser.add_argument('--goals_number', default=2, type=int)
        parser.add_argument('--learning_mode', choices=['V', 'Q', 'DQN'], default='DQN')
        parser.add_argument('--optimizer', choices=['Adam', 'RMSProp'], default='Adam')
        parser.add_argument('--mask', choices=[0, 1], default=0, type=int)
        parser.add_argument('--load_model', choices=[0, 1, 2], default=1, type=int)
        parser.add_argument('--reward_mode', choices=[0, 1, 2], default=0, type=int)
        parser.add_argument('--device', choices=['cpu', 'cuda', 'mps'], default='cpu')
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
    main = Main(args)
    agents = [Agent(args, main.model_path[b_idx]) for b_idx in range(args.agents_number)]
    main.run(agents)