"""
このファイルを実行して学習する．
python main.py --grid_size 10 のようにして各種設定を変更可能.
主要なハイパーパラメータは parse_args() で管理.

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

import argparse
import torch

RED = '\033[91m'
GREEN = '\033[92m'
RESET = '\033[0m'

class Main:
    def __init__(self, args, agents):
        self.agents = agents
        self.args = args

    def run(self):
        self.agents.run()

    """
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

    def save_model(self):
        self.agents.save_model(agents)
    """

if __name__ == '__main__':
    def parse_args():
        parser = argparse.ArgumentParser()
        parser.add_argument('--dir_path', default='./')
        parser.add_argument('--grid_size', default=4, type=int)
        parser.add_argument('--agents_number', default=2, type=int)
        parser.add_argument('--goals_number', default=2, type=int)
        parser.add_argument('--learning_mode', choices=['V', 'Q', 'DQN'], default='DQN')
        parser.add_argument('--optimizer', choices=['Adam', 'RMSProp'], default='Adam')
        parser.add_argument('--mask', choices=[0, 1], default=0, type=int)
        parser.add_argument('--load_model', choices=[0, 1, 2], default=0, type=int)
        parser.add_argument('--reward_mode', choices=[0, 1, 2, 3], default=3, type=int)
        parser.add_argument('--device', choices=['auto', 'cpu', 'cuda', 'mps'], default='auto')
        parser.add_argument('--episode_number', default=1000, type=int)
        parser.add_argument('--max_timestep', default=25, type=int)
        parser.add_argument('--decay_epsilon', default=500000, type=int)
        parser.add_argument('--learning_rate', default=0.001, type=float)
        parser.add_argument('--gamma', default=0.99, type=float)
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

    #from utils.Model_Saver import Saver
    #from utils.plot_results import PlotResults

    if args.learning_mode == "Q":
        from Q_learn.Agent_Q import Agent
        from Q_learn.MultiAgent_Q import MultiAgent_Q

        #agents = [Agent_Q(args, multiagent_q.model_path[b_idx]) for b_idx in range(args.agents_number)]
        agents = [Agent(args,i) for i in range(args.agents_number)]

        """
        save_dir = os.path.join(
            "output",
            f"Q_mask[{args.mask}]_Reward[{args.reward_mode}]_env[{args.grid_size}x{args.grid_size}]_max_ts[{args.max_timestep}]_agents[{args.agents_number}]"
        )
        scores_path = os.path.join(save_dir, "scores.csv")
        agents_states_path = os.path.join(save_dir, "agents_states.csv")

        plot_results = PlotResults(scores_path, agents_states_path)

        saver = Saver(save_dir)
        """

        maq = MultiAgent_Q(args,agents)
        maq.run()

        maq.save_Qtable()
        maq.result_show()

        #saver.save_model(agents)
        #plot_results.draw()

    elif args.learning_mode == "DQN":
        from DQN.Agent_DQN import Agent_DQN
        from DQN.MultiAgent_DQN import MultiAgent_DQN

        """
        save_dir = os.path.join(
            "output",
            f"DQN_mask[{args.mask}]_Reward[{args.reward_mode}]_env[{args.grid_size}x{args.grid_size}]_max_ts[{args.max_timestep}]_agents[{args.agents_number}]"
        )
        scores_path = os.path.join(save_dir, "scores.csv")
        agents_states_path = os.path.join(save_dir, "agents_states.csv")

        #plot_results = PlotResults(scores_path, agents_states_path)
        saver = Saver(save_dir)
        """

        #agents = [Agent_DQN(args, i) for i in range(args.agents_number)]
        agents = [Agent_DQN(args) for _ in range(args.agents_number)]
        ma_dqn = MultiAgent_DQN(args, agents)

        ma_dqn.run()

        ma_dqn.save_model_weights()
        ma_dqn.result_show()
        #saver.save_model_weights(agents)
        #plot_results.draw_heatmap(args.grid_size)

    else:
        print(f"{args.learning_mode}は未実装")