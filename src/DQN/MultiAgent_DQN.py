import sys
import os
import torch.optim as optim
from typing import List, Tuple, Optional, Any

from Environments.MultiAgentGridEnv import MultiAgentGridEnv

from utils.plot_results import PlotResults
from utils.Saver import Saver

from .IQLMasterAgent import IQLMasterAgent
from .QMIXMasterAgent import QMIXMasterAgent
from .dqn import AgentNetwork 
from .IO_Handler import Model_IO

from Base.Constant import GlobalState
from utils.replay_buffer import ReplayBuffer
from utils.StateProcesser import StateProcessor

RED = '\033[91m'
GREEN = '\033[92m'
RESET = '\033[0m'

MAX_EPSILON = 1.0
MIN_EPSILON = 0.05

class MARLTrainer:
    """
    複数のDQNエージェントを用いた強化学習の実行を管理するクラス.
    環境とのインタラクション、エピソードの進行、学習ループ、結果の保存・表示を統括します。
    """
    def __init__(self, args, mode: str, shared_agent_network: AgentNetwork, shared_state_processor: StateProcessor, shared_replay_buffer: ReplayBuffer):
        """
        MARLTrainer クラスのコンストラクタ.

        Args:
            args: 実行設定を含むオブジェクト.
                  (reward_mode, render_mode, episode_number, max_timestep,
                   agents_number, goals_num, grid_size, load_model, mask,
                   save_agent_states, alpha, beta, beta_anneal_steps, use_per 属性を持つことを想定)
            mode (str): 学習モード ('IQL' または 'QMIX').
            shared_agent_network (AgentNetwork): 共有AgentNetworkのインスタンス.
            shared_state_processor (StateProcessor): 共有StateProcessorのインスタンス.
            shared_replay_buffer (ReplayBuffer): 共有ReplayBufferのインスタンス.
        """
        self.args = args # Store args for later use

        # 3つまでゴールを手動で設定できるように変更。
        fix_goal_pool = [(args.grid_size-1,args.grid_size-1),(args.grid_size//4,args.grid_size//3),(args.grid_size-1,0),(args.grid_size//4,args.grid_size//6)]
        self._fix_goal_from_goal_number = fix_goal_pool[:min(args.goals_number, len(fix_goal_pool))]

        self.env = MultiAgentGridEnv(args, fixrd_goals=self._fix_goal_from_goal_number)

        self.reward_mode = args.reward_mode
        self.render_mode = args.render_mode
        self.episode_num = args.episode_number
        self.max_ts = args.max_timestep
        self.agents_number = args.agents_number
        self.goals_number = args.goals_number
        self.grid_size = args.grid_size
        self.update_frequency = 4 # 学習の頻度
        self.target_update_frequency = args.target_update_frequency # From args

        self.save_agent_states = args.save_agent_states

        self.start_episode = 1

        # Epsilon decay parameters
        self.epsilon = 1.0
        self.epsilon_decay = args.epsilon_decay

        # PER beta annealing parameters
        self.beta = args.beta # Initial beta
        self.beta_anneal_steps = args.beta_anneal_steps
        self.use_per = args.use_per

        # MasterAgentのインスタンス化
        if mode == 'IQL':
            self.master_agent = IQLMasterAgent(
                n_agents=args.agents_number,
                action_size=self.env.action_space_size,
                grid_size=args.grid_size,
                goals_num=args.goals_number,
                device=args.device,
                state_processor=shared_state_processor,
                agent_network=shared_agent_network,
                gamma=args.gamma
            )
        elif mode == 'QMIX':
            self.master_agent = QMIXMasterAgent(
                n_agents=args.agents_number,
                action_size=self.env.action_space_size,
                grid_size=args.grid_size,
                goals_num=args.goals_number,
                device=args.device,
                state_processor=shared_state_processor,
                agent_network=shared_agent_network,
                gamma=args.gamma
            )
        else:
            raise ValueError(f"Unknown mode: {mode}. Must be 'IQL' or 'QMIX'.")

        # ReplayBufferの割り当て
        self.replay_buffer = shared_replay_buffer

        # Optimizer initialization
        optim_params = self.master_agent.get_optimizer_params()

        if args.optimizer == 'Adam':
            self.optimizer: optim.Optimizer = optim.Adam(optim_params, lr=args.learning_rate)
        elif args.optimizer == 'RMSProp':
            self.optimizer: optim.Optimizer = optim.RMSprop(optim_params, lr=args.learning_rate)
        else:
            print(f"Warning: Optimizer type '{args.optimizer}' not recognized. Using Adam as default.")
            self.optimizer: optim.Optimizer = optim.Adam(optim_params, lr=args.learning_rate)

        # 結果保存ディレクトリの設定と作成
        folder_name = f"{mode}"

        # 観測範囲に基づいた識別子を追加
        if args.neighbor_distance == 0:
            folder_name += "_Selfish"
        elif args.neighbor_distance >= args.grid_size:
            folder_name += "_FullObs"
        else:
            folder_name += f"_PartialObs[{args.neighbor_distance}]"

        folder_name += f"_報酬[{self.reward_mode}]_[{self.grid_size}x{self.grid_size}]_T[{self.max_ts}]_A-G[{self.agents_number}-{self.goals_number}]"

        # PERを使用している場合、PERパラメータを追加
        if args.use_per:
            folder_name += f"_PER_alpha[{args.alpha}]_beta_anneal[{self.beta_anneal_steps}]"

        self.save_dir = os.path.join(
            "output",
            folder_name
        )

        # 結果保存およびプロット関連クラスの初期化
        self.saver = Saver(self.save_dir,self.grid_size)
        self.saver.CALCULATION_PERIOD = 50
        self.plot_results = PlotResults(self.save_dir)

        self.load_checkpoint(None)

    def decay_epsilon_power(self, step: int):
        """
        ステップ数に基づき、探索率εを指数的に減衰させる関数。
        Args:
            step (int): 現在のステップ数（またはエピソード数）。
        """
        lambda_ = 0.0001
        self.epsilon *= MAX_EPSILON * (self.epsilon_decay ** (lambda_))
        self.epsilon = max(MIN_EPSILON, self.epsilon)

    def result_save(self):
        self.plot_results.draw_heatmap()
        self.plot_results.draw()

    def train(self):
        """
        強化学習のメイン実行ループ.
        指定されたエピソード数だけ環境とのインタラクションと学習を行います。
        """
        # 事前条件チェック: エージェント数はゴール数以下である必要がある
        if self.agents_number > self.goals_number:
            print('goals_num >= agents_num に設定してください.\n')
            sys.exit()

        # 学習開始メッセージ
        print(f"{GREEN}MARLTrainer{RESET} で学習中..." + (f" ({GREEN}PER enabled{RESET})") + "\n")
        print(f"goals: {self.env.get_goal_positions().values()}")

        total_step = 0 # 環境との全インタラクションステップ数の累積
        # 集計用一時変数の初期化
        episode_rewards: list[float] = []   # エピソードごとの報酬を格納
        episode_steps  : list[int]   = []   # エピソードごとのステップ数を格納
        episode_losses : list[float] = []   # エピソードごとの損失を格納
        done_counts    : list[int]   = []   # エピソードごとで成功/失敗を記録

        # ----------------------------------
        # メインループ（各エピソード）
        # ----------------------------------
        for episode in range(self.start_episode, self.episode_num + 1):

            print(f'{GREEN if done_counts and done_counts[-1] else ""}■{RESET}', end='',flush=True)  # 進捗表示 (エピソード100回ごとに改行)

            # 50エピソードごとに集計結果を出力
            CONSOLE_LOG_FREQ = 50

            # 各エピソード開始時に環境をリセット
            iap = [(0,i) for i in range(self.agents_number)]
            current_global_state:GlobalState = self.env.reset(initial_agent_positions=iap)

            # individual_dones will track if an agent has completed its task or dropped out
            # This is the `dones` List[bool] that gets passed to ReplayBuffer.add
            individual_dones: List[bool] = [False] * self.agents_number # Initialize individual done flags for all agents
            episode_done = False # Overall episode done flag, returned by env.step
            step_count:int = 0 # 現在のエピソードのステップ数
            episode_reward:float = 0.0 # 現在のエピソードの累積報酬

            step_losses:list[float] = []

            # ---------------------------
            # 1エピソードのステップループ
            # ---------------------------
            while not episode_done and step_count < self.max_ts:
                # Epsilon decay
                self.decay_epsilon_power(total_step)

                # 各エージェントの行動を選択
                actions: List[int] = self.master_agent.get_actions(current_global_state, self.epsilon)

                # エージェントの状態を保存（オプション）
                if self.save_agent_states:
                    agent_positions_in_global_state = current_global_state[self.goals_number:]
                    for i, agent_pos in enumerate(agent_positions_in_global_state):
                        self.saver.log_agent_states(i, agent_pos[0], agent_pos[1])

                # 環境にステップを与えて状態を更新し、結果を取得
                next_global_state, reward, episode_done, info = self.env.step(actions)

                # `dones_for_experience` should be the individual done flags for each agent at this step.
                # For now, if the episode is done, all agents are considered done for this experience.
                # In a more complex environment, info could contain individual agent done status.
                dones_for_experience = [episode_done] * self.agents_number # Assuming all agents done if episode is done

                # Store experience in replay buffer
                self.replay_buffer.add(current_global_state, actions, reward, next_global_state, dones_for_experience)

                # Each step accumulates total reward
                episode_reward += reward

                # Perform learning at update_frequency
                if total_step % self.update_frequency == 0 and len(self.replay_buffer) >= self.replay_buffer.batch_size:
                    # Sample from replay buffer
                    sample_output = self.replay_buffer.sample(beta=self.beta)

                    if sample_output is not None:
                        global_states_batch_raw, actions_batch, rewards_batch, next_global_states_batch_raw, dones_batch, is_weights_batch, sampled_indices = sample_output

                        # Calculate loss using MasterAgent
                        loss, abs_td_errors = self.master_agent.evaluate_q(
                            global_states_batch_raw, actions_batch, rewards_batch,
                            next_global_states_batch_raw, dones_batch, is_weights_batch
                        )

                        # Backward pass and optimize
                        self.optimizer.zero_grad()
                        loss.backward()
                        self.optimizer.step()

                        # Update priorities in ReplayBuffer if PER is used
                        if self.use_per and sampled_indices is not None and abs_td_errors is not None:
                            # abs_td_errors from evaluate_q is (batch_size,) for the sample's total TD error
                            self.replay_buffer.update_priorities(sampled_indices, abs_td_errors.detach().cpu().numpy())

                        step_losses.append(loss.item())

                    # Sync target network periodically
                    if total_step > 0 and total_step % self.target_update_frequency == 0: # Use self.target_update_frequency
                        self.master_agent.sync_target_network()

                    # PER: Beta annealing
                    if self.use_per: # PER: Beta annealing only if use_per is True
                        # Ensure beta doesn't exceed 1.0
                        beta_increment_per_learning_step = (1.0 - self.args.beta) / self.beta_anneal_steps # Use initial beta from args to calculate increment
                        self.beta = min(1.0, self.beta + beta_increment_per_learning_step)

                current_global_state = next_global_state
                step_count += 1
                total_step += 1

            # ---------------------------
            # エピソード終了後の処理
            # ---------------------------

            if (episode % CONSOLE_LOG_FREQ == 0) and (episode!=self.start_episode):
                print() # 改行して進捗表示をクリア

                # エピソードごとの平均損失、平均ステップ、平均報酬を計算し、表示に追加
                avg_loss   = sum(episode_losses) / len(episode_losses)      # 期間内の平均損失
                avg_reward = sum(episode_rewards) / len(episode_rewards)    # 期間内の平均報酬
                avg_step   = sum(episode_steps) / len(episode_steps)        # 期間内の平均ステップ数
                done_rate  = sum(done_counts) / len(done_counts)            # 達成率

                print(f"     エピソード {episode - CONSOLE_LOG_FREQ+1} ~ {episode} の平均 step  : {GREEN}{avg_step:.3f}{RESET}")
                print(f"     エピソード {episode - CONSOLE_LOG_FREQ+1} ~ {episode} の平均 reward: {GREEN}{avg_reward:.3f}{RESET}")
                print(f"     エピソード {episode - CONSOLE_LOG_FREQ+1} ~ {episode} の達成率     : {GREEN}{done_rate:.3f}{RESET}") # 達成率も出力 .3f で小数点以下2桁表示
                print(f"     エピソード {episode - CONSOLE_LOG_FREQ+1} ~ {episode} の平均 loss  : {GREEN}{avg_loss:.5f}{RESET}") # 平均損失も出力
                if self.use_per:
                    print(f"     (Step: {total_step}), 探索率 : {GREEN}{self.epsilon:.3f}{RESET}, beta: {GREEN}{self.beta:.3f}{RESET}")
                else:
                    print(f"     (Step: {total_step}), 探索率 : {GREEN}{self.epsilon:.3f}{RESET}")

                episode_losses = [] # 100エピソードごとに損失リストもリセット
                episode_rewards = []
                episode_steps  = []
                done_counts = []

            if episode % 50 == 0:
                self.save_checkpoint(episode)

            # エピソードが完了 (episode_done == True) した場合、達成エピソード数カウンタをインクリメント
            if episode_done:
                episode_steps.append(step_count)

            # エピソードの平均損失を計算
            episode_loss:float = sum(step_losses)/len(step_losses) if step_losses else 0.0
            episode_step:int = step_count

            # Saverでエピソードごとのスコアをログに記録
            self.saver.log_episode_data(episode, step_count, episode_reward, episode_loss, episode_done)

            # 集計期間内の平均計算のための累積 (avg_reward_temp accumulation)
            episode_losses.append(episode_loss)
            episode_rewards.append(episode_reward)
            episode_steps.append(episode_step)
            done_counts.append(episode_done)

        self.saver.save_remaining_episode_data()
        self.saver.save_visited_coordinates()
        print()  # 全エピソード終了後に改行

    def save_model_weights(self):
        """学習済みモデルの重みを保存する."""
        model_io = Model_IO()
        model_dir = os.path.join(self.save_dir, "models")
        if not os.path.exists(model_dir):
            os.makedirs(model_dir)

        # Save AgentNetwork weights
        agent_net_path = os.path.join(model_dir, "agent_network.pth")
        model_io.save(agent_net_path, self.master_agent.agent_network.state_dict())

        # If QMIX, save MixingNetwork weights
        if isinstance(self.master_agent, QMIXMasterAgent):
            mixing_net_path = os.path.join(model_dir, "mixing_network.pth")
            model_io.save(mixing_net_path, self.master_agent.mixing_network.state_dict())

    def load_model_weights(self):
        model_io = Model_IO()
        model_dir = os.path.join(self.save_dir, "models")

        # Load AgentNetwork weights
        agent_net_path = os.path.join(model_dir, "agent_network.pth")
        loaded_agent_net_state_dict = model_io.load(agent_net_path)
        self.master_agent.agent_network.load_state_dict(loaded_agent_net_state_dict)
        self.master_agent.agent_network.eval()

        # If QMIX, load MixingNetwork weights
        if isinstance(self.master_agent, QMIXMasterAgent):
            mixing_net_path = os.path.join(model_dir, "mixing_network.pth")
            loaded_mixing_net_state_dict = model_io.load(mixing_net_path)
            self.master_agent.mixing_network.load_state_dict(loaded_mixing_net_state_dict)
            self.master_agent.mixing_network.eval()

        # Sync target networks after loading
        self.master_agent.sync_target_network()

    def simulate_agent_behavior(self, num_simulation_episodes: int = 1, max_simulation_timestep:int =-1):
        """
        学習済みモデルを使ってエージェントの行動をシミュレーションします。

        Args:
            num_simulation_episodes (int): シミュレーションを実行するエピソード数。
            max_simulation_timestep (Optional[int]): 各シミュレーションエピソードの最大ステップ数。
                                                     Noneの場合、MultiAgent_DQNのmax_tsが使用されます。
        """
        print(f"{GREEN}--- シミュレーション開始 (学習済みモデル使用) ---" + f"{RESET}\n")
        print(f"シミュレーションエピソード数: {num_simulation_episodes}\n")

        if max_simulation_timestep ==-1:
            max_simulation_timestep = self.max_ts

        self.load_model_weights()

        # Set all networks to eval mode for simulation
        self.master_agent.agent_network.eval()
        if isinstance(self.master_agent, QMIXMasterAgent):
            self.master_agent.mixing_network.eval()

        for episode_idx in range(1, num_simulation_episodes + 1):
            print(f"--- シミュレーションエピソード {episode_idx} / {num_simulation_episodes} ---")

            # エージェントの初期位置を設定
            current_global_state = self.env.reset(initial_agent_positions=[(0,i) for i in range(self.agents_number)])
            episode_done = False
            step_count = 0
            ep_reward = 0.0

            while not episode_done and step_count < max_simulation_timestep:
                print(f"\nステップ {step_count}:")
                print(f"  現在の全体状態: {current_global_state}")

                # Get actions from master_agent (epsilon=0 for greedy actions)
                actions = self.master_agent.get_actions(current_global_state, epsilon=0.0)

                # Display Q-values (This part requires MasterAgent to expose a way to get all Q-values, or replicate logic)
                # For now, let's just display the chosen actions
                print(f"  選択された行動: {actions}")

                # 環境にステップを与えて状態を更新し、結果を取得
                next_global_state, reward, episode_done, _ = self.env.step(actions)
                ep_reward += reward

                print(f"  報酬: {reward:.2f}, 完了: {episode_done}")

                current_global_state = next_global_state
                step_count += 1

                if episode_done:
                    print(f"エピソード {episode_idx} 完了. 最終ステップ: {step_count}, 累積報酬: {ep_reward:.2f}")
                elif step_count == max_simulation_timestep:
                    print(f"エピソード {episode_idx} タイムアウト. 最終ステップ: {step_count}, 累積報酬: {ep_reward:.2f}")

            print("\n" + "-" * 50 + "\n")

        print(f"{GREEN}--- シミュレーション終了 ---" + f"{RESET}")

    def load_checkpoint(self, episode:int|None=None):
        model_io = Model_IO()
        model_dir = os.path.join(self.save_dir, "checkpoints")

        if not os.path.exists(model_dir):
            print(f"{model_dir}は存在しなかったため、新規学習とする")
            return

        try:
            # Define checkpoint file path(s)
            agent_net_checkpoint_path = os.path.join(model_dir, "checkpoint_agent_network.pth")
            mixing_net_checkpoint_path = os.path.join(model_dir, "checkpoint_mixing_network.pth")

            # Load latest checkpoint if episode is None, otherwise load specific episode checkpoint
            if episode is not None:
                agent_net_checkpoint_path_ep = os.path.join(model_dir, f"checkpoint_episode[{episode}]_agent_network.pth")
                if os.path.exists(agent_net_checkpoint_path_ep):
                    agent_net_checkpoint_path = agent_net_checkpoint_path_ep

                if isinstance(self.master_agent, QMIXMasterAgent):
                    mixing_net_checkpoint_path_ep = os.path.join(model_dir, f"checkpoint_episode[{episode}]_mixing_network.pth")
                    if os.path.exists(mixing_net_checkpoint_path_ep):
                        mixing_net_checkpoint_path = mixing_net_checkpoint_path_ep

            # Load AgentNetwork checkpoint
            agent_net_checkpoint_data = model_io.load_checkpoint(agent_net_checkpoint_path)
            self.master_agent.agent_network.load_state_dict(agent_net_checkpoint_data['model_state'])
            self.master_agent.agent_network_target.load_state_dict(agent_net_checkpoint_data['target_state'])

            # Load MixingNetwork checkpoint if in QMIX mode
            if isinstance(self.master_agent, QMIXMasterAgent):
                mixing_net_checkpoint_data = model_io.load_checkpoint(mixing_net_checkpoint_path)
                self.master_agent.mixing_network.load_state_dict(mixing_net_checkpoint_data['model_state'])
                self.master_agent.mixing_network_target.load_state_dict(mixing_net_checkpoint_data['target_state'])

            # Load optimizer state (assuming one optimizer for all networks, saved with agent_network)
            self.optimizer.load_state_dict(agent_net_checkpoint_data['optimizer_state'])

            # Update start_episode and epsilon/beta
            self.start_episode = agent_net_checkpoint_data['epoch'] + 1
            self.epsilon = agent_net_checkpoint_data['epsilon']
            # The beta value for PER annealing might need to be explicitly saved/loaded if it's dynamic.
            # For simplicity, we are not loading beta here. It will reset to args.beta and anneal from there.

            # Set networks to train mode
            self.master_agent.agent_network.train()
            if isinstance(self.master_agent, QMIXMasterAgent):
                self.master_agent.mixing_network.train()

        except Exception as e:
            print(f"エラー発生: {e}")
            print("チェックポイントのロードに失敗しました。新規学習を開始します。")

        print(f"エピソード{self.start_episode}から再開")

    def save_checkpoint(self, episode:int):
        model_io = Model_IO()
        model_dir = os.path.join(self.save_dir, "checkpoints")
        if not os.path.exists(model_dir): os.makedirs(model_dir)

        # Save AgentNetwork checkpoint (latest and episode-specific)
        agent_net_file_path = os.path.join(model_dir, "checkpoint_agent_network.pth")
        agent_net_file_path_episode = os.path.join(model_dir, f"checkpoint_episode[{episode}]_agent_network.pth")
        model_io.save_checkpoint(
            agent_net_file_path,
            self.master_agent.agent_network.state_dict(),
            self.master_agent.agent_network_target.state_dict(),
            self.optimizer.state_dict(),
            episode,
            epsilon=self.epsilon
            # beta=self.beta # Optionally save beta as well
        )
        model_io.save_checkpoint(
            agent_net_file_path_episode,
            self.master_agent.agent_network.state_dict(),
            self.master_agent.agent_network_target.state_dict(),
            self.optimizer.state_dict(),
            episode,
            epsilon=self.epsilon
            # beta=self.beta
        )

        # If QMIX, save MixingNetwork checkpoint (latest and episode-specific)
        if isinstance(self.master_agent, QMIXMasterAgent):
            mixing_net_file_path = os.path.join(model_dir, "checkpoint_mixing_network.pth")
            mixing_net_file_path_episode = os.path.join(model_dir, f"checkpoint_episode[{episode}]_mixing_network.pth")
            model_io.save_checkpoint(
                mixing_net_file_path,
                self.master_agent.mixing_network.state_dict(),
                self.master_agent.mixing_network_target.state_dict(),
                None, # Optimizer state already saved with agent_network, or needs separate handling for mixing network optimizer
                episode,
                epsilon=self.epsilon
                # beta=self.beta
            )
            model_io.save_checkpoint(
                mixing_net_file_path_episode,
                self.master_agent.mixing_network.state_dict(),
                self.master_agent.mixing_network_target.state_dict(),
                None,
                episode,
                epsilon=self.epsilon
                # beta=self.beta
            )

