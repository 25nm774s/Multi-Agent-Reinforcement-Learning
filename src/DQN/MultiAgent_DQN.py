import sys
import os
import torch
import torch.optim as optim
from typing import Dict

from Environments.MultiAgentGridEnv import IEnvWrapper

from utils.plot_results import PlotResults
from utils.Saver import Saver

from .IQLMasterAgent import IQLMasterAgent
from .QMIXMasterAgent import QMIXMasterAgent
from .VDNMasterAgent import VDNMasterAgent
from .DICGMasterAgent import DICGMasterAgent
from .network import AgentNetwork 
from .IO_Handler import Model_IO

from Base.Constant import GlobalState
from utils.replay_buffer import ReplayBuffer
from Environments.StateProcesser import ObsToTensorWrapper

RED = '\033[91m'
GREEN = '\033[92m'
RESET = '\033[0m'

# MAX_EPSILON = 1.0
# MIN_EPSILON = 0.01

class MARLTrainer:
    """
    複数のDQNエージェントを用いた強化学習の実行を管理するクラス.
    環境とのインタラクション、エピソードの進行、学習ループ、結果の保存・表示を統括します。
    """
    def __init__(self, args, mode: str, env_wrapper: IEnvWrapper, shared_agent_network: AgentNetwork, shared_state_processor: ObsToTensorWrapper, run_id=None):
        """
        MARLTrainer クラスのコンストラクタ.

        Args:
            args: 実行設定を含むオブジェクト.
                  (reward_mode, render_mode, episode_number, max_timestep,
                   agents_number, goals_number, grid_size, load_model, mask,
                   save_agent_states, alpha, beta, beta_anneal_steps, use_per, agent_reward_processing_mode 属性を持つことを想定)
            mode (str): 学習モード ('IQL' または 'QMIX' または 'VDN').
            env_wrapper (IEnvWrapper): 環境とのインタラクションを標準化するラッパーインスタンス.
            shared_agent_network (AgentNetwork): 共有AgentNetworkのインスタンス.
            shared_state_processor (ObsToTensorWrapper): 共有ObsToTensorWrapperのインスタンス.
        """
        self.args = args # Store args for later use

        self.env_wrapper = env_wrapper # Store the IEnvWrapper instance

        self.reward_mode = args.reward_mode
        self.render_mode = args.render_mode
        self.episode_num = args.episode_number
        self.max_ts = args.max_timestep
        self.MAX_EPSILON = args.max_epsilon
        self.MIN_EPSILON = args.min_epsilon

        # 環境ラッパーからプロパティを取得
        self.action_space_size = self.env_wrapper.action_space_size
        self.agents_number = self.env_wrapper.n_agents
        self.goals_number = self.env_wrapper.goals_number
        self.grid_size = self.env_wrapper.grid_size
        self.num_channels = self.env_wrapper.num_channels

        self.update_frequency = 4 # 学習の頻度
        self.target_update_frequency = args.target_update_frequency

        self.save_agent_states = args.save_agent_states

        self.start_episode = 1

        # Epsilon decay parameters
        self.epsilon = self.MAX_EPSILON
        self.epsilon_decay = args.epsilon_decay

        # PER beta annealing parameters
        self.beta = args.beta # Initial beta
        self.beta_anneal_steps = args.beta_anneal_steps
        self.use_per = args.use_per

        # These are already defined in self.env_wrapper
        self._agent_ids: list[str] = self.env_wrapper.agent_ids
        self._goal_ids: list[str] = self.env_wrapper.goal_ids

        # MasterAgentのインスタンス化
        if mode == 'IQL':
            self.master_agent = IQLMasterAgent(
                n_agents=self.agents_number,
                action_size=self.action_space_size,
                grid_size=self.grid_size,
                goals_number=self.goals_number,
                device=args.device,
                state_processor=shared_state_processor,
                agent_network_instance=shared_agent_network,
                gamma=args.gamma,
                agent_ids=self._agent_ids,
                goal_ids=self._goal_ids,
                agent_reward_processing_mode=args.agent_reward_processing_mode # Pass new argument
            )
        elif mode == 'QMIX':
            self.master_agent = QMIXMasterAgent(
                n_agents=self.agents_number,
                action_size=self.action_space_size,
                grid_size=self.grid_size,
                goals_number=self.goals_number,
                device=args.device,
                state_processor=shared_state_processor,
                agent_network_instance=shared_agent_network,
                gamma=args.gamma,
                agent_ids=self._agent_ids,
                goal_ids=self._goal_ids,
                agent_reward_processing_mode=args.agent_reward_processing_mode # Pass new argument
            )
        # Added VDN mode instantiation as per the plan
        elif mode == 'VDN':
            self.master_agent = VDNMasterAgent(
                n_agents=self.agents_number,
                action_size=self.action_space_size,
                grid_size=self.grid_size,
                goals_number=self.goals_number,
                device=args.device,
                state_processor=shared_state_processor,
                agent_network_instance=shared_agent_network,
                gamma=args.gamma,
                agent_ids=self._agent_ids,
                goal_ids=self._goal_ids,
                agent_reward_processing_mode=args.agent_reward_processing_mode
            )
        # Add DICG mode instantiation
        elif mode == 'DICG':
            self.master_agent = DICGMasterAgent(
                n_agents=self.agents_number,
                action_size=self.action_space_size,
                grid_size=self.grid_size,
                goals_number=self.goals_number,
                device=args.device,
                state_processor=shared_state_processor,
                agent_network_instance=shared_agent_network,
                gamma=args.gamma,
                agent_ids=self._agent_ids,
                goal_ids=self._goal_ids,
                agent_reward_processing_mode=args.agent_reward_processing_mode
            )
        else:
            raise ValueError(f"Unknown mode: {mode}. Must be 'IQL', 'QMIX', 'VDN', or 'DICG'.")

        # ReplayBufferの割り当て - MARLTrainer内で初期化
        self.replay_buffer = ReplayBuffer(
            buffer_size=args.buffer_size,
            batch_size=args.batch_size,
            device=args.device,
            alpha=args.alpha,
            use_per=bool(args.use_per),
            n_agents=self.agents_number, # env_wrapperから取得
            num_channels=self.num_channels, # env_wrapperから取得
            grid_size=self.grid_size, # env_wrapperから取得
            goals_number=self.goals_number # env_wrapperから取得
        )

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
        if args.neighbor_distance >= args.grid_size:
            folder_name += "_全観測"
        else:
            folder_name += f"_観測径[{args.neighbor_distance}]"

        folder_name += f"_報酬[{self.reward_mode}]_ARPM[{args.agent_reward_processing_mode}]_[{self.grid_size}x{self.grid_size}]_T[{self.max_ts}]_A-G[{self.agents_number}-{self.goals_number}]"

        # PERを使用している場合、PERパラメータを追加
        if args.use_per:
            folder_name += f"_PER_alpha[{args.alpha}]_beta_anneal[{self.beta_anneal_steps}]"

        self.save_dir = os.path.join(
            "output",
            folder_name
        )

        if run_id is not None:
            self.save_dir = os.path.join(self.save_dir, f"run_{run_id}")
            os.makedirs(self.save_dir, exist_ok=True)

        # 結果保存およびプロット関連クラスの初期化
        os.makedirs(self.save_dir,exist_ok=True)
        score_sammary_path = os.path.join(self.save_dir, "aggregated_episode_metrics_10.csv")
        visited_coordinates_path = os.path.join(self.save_dir, "visited_coordinates.npy")

        self.saver = Saver(score_sammary_path,visited_coordinates_path,self.grid_size)
        self.saver.CALCULATION_PERIOD = 10

        # saverクラスで保存されたデータを使ってグラフを作るクラス
        self.plot_results = PlotResults(score_sammary_path,visited_coordinates_path)

        self.load_checkpoint(None)

    def decay_epsilon_power(self):
        """
        ステップ数に基づき、探索率εを指数的に減衰させる関数。
        Args:
            step (int): 現在のステップ数（またはエピソード数）。
        """
        lambda_ = 0.0001
        self.epsilon *= self.MAX_EPSILON * (self.epsilon_decay ** (lambda_))
        self.epsilon = max(self.MIN_EPSILON, self.epsilon)

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
            print('goals_number >= agents_number に設定してください。\n')
            sys.exit()

        # 学習開始メッセージ
        print(f"{GREEN}MARLTrainer{RESET}" + " で学習中..." + (f" ({GREEN}PER enabled{RESET})") + "\n")
        # goals_number の直接参照からenv_wrapper経由に変更
        # これはMultiAgentGridEnvの内部実装に依存しない方法を模索すべき
        # 現在はenv_wrapper._env.get_goal_positions()に依存するが、IEnvWrapperにはゴール位置取得メソッドがない
        # 一旦コメントアウトするか、info dict経由でゴール位置を取得するように変更する
        # print(f"goals: {self.env_wrapper._env.get_goal_positions().values()}")

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
            # reset() は部分観測テンソル、グローバル状態テンソル、および情報辞書を返します。
            agent_obs_tensor, global_state_tensor, info = self.env_wrapper.reset(initial_agent_positions=iap)
            true_current_global_state: GlobalState = info['raw_global_state'] # For logging and potential future use where full global state is needed

            episode_done = False # Overall episode done flag, returned by env_wrapper.step
            step_count:int = 0 # 現在のエピソードのステップ数
            episode_reward:float = 0.0 # 現在のエピソードの累積報酬

            step_losses:list[float] = []

            # ---------------------------
            # 1エピソードのステップループ
            # ---------------------------
            while not episode_done and step_count < self.max_ts:
                # Epsilon decay
                self.decay_epsilon_power()

                # 各エージェントの行動を選択 - master_agentは観測テンソルを期待します
                actions_dict: Dict[str, int] = self.master_agent.get_actions(agent_obs_tensor, self.epsilon)

                # エージェントの状態を保存（オプション）
                if self.save_agent_states:
                    # true_current_global_state (Dict[str, PosType]) は 'agent_X' キーを含みます
                    for agent_id_str in self._agent_ids:
                        if agent_id_str in true_current_global_state:
                            agent_pos = true_current_global_state[agent_id_str]
                            self.saver.log_agent_states(agent_id_str, agent_pos[0], agent_pos[1])

                # 環境にステップを与えて状態を更新し、結果を取得
                # step() は次の観測テンソル、グローバル状態テンソル、報酬テンソル、完了テンソル、および情報辞書を返します。
                next_agent_obs_tensor, next_global_state_tensor, rewards_tensor, dones_tensor, step_info = self.env_wrapper.step(actions_dict)

                # アクション辞書をテンソルに変換（ReplayBufferに渡すため）
                actions_tensor = torch.tensor([actions_dict[agent_id] for agent_id in self._agent_ids], dtype=torch.long, device=self.args.device)

                # Store experience in replay buffer - use the observation dictionaries directly
                # Pass all_agents_done_scalar from step_info
                all_agents_done_scalar = torch.tensor([float(step_info['all_agents_done'])], dtype=torch.float32, device=self.args.device)
                self.replay_buffer.add(agent_obs_tensor, global_state_tensor, actions_tensor, rewards_tensor, dones_tensor, all_agents_done_scalar, next_agent_obs_tensor, next_global_state_tensor)

                # Each step accumulates total reward
                episode_reward += rewards_tensor.sum().item() # rewards_tensorは(n_agents,)です。

                # Perform learning at update_frequency
                if total_step % self.update_frequency == 0 and len(self.replay_buffer) >= self.replay_buffer.batch_size:
                    # Sample from replay buffer
                    sample_output = self.replay_buffer.sample(beta=self.beta)

                    if sample_output is not None:
                        # replay_buffer.sample now returns lists of observation dictionaries
                        (current_agent_obs_batch, current_global_state_batch, actions_batch, rewards_batch, dones_batch, all_agents_done_batch, next_agent_obs_batch, next_global_state_batch, is_weights_batch, sampled_indices) = sample_output

                        # Calculate loss using MasterAgent
                        # master_agent.evaluate_q expects lists of observation dictionaries
                        loss, abs_td_errors = self.master_agent.evaluate_q(
                            current_agent_obs_batch, current_global_state_batch, actions_batch, rewards_batch,
                            dones_batch, all_agents_done_batch, next_agent_obs_batch, next_global_state_batch, is_weights_batch
                        )

                        # Backward pass and optimize
                        self.optimizer.zero_grad()
                        loss.backward()

                        # Gradient clipping
                        torch.nn.utils.clip_grad_norm_(self.master_agent.get_optimizer_params(), self.args.grad_norm_clip)

                        self.optimizer.step()

                        # Update priorities in ReplayBuffer if PER is used
                        if self.use_per and sampled_indices is not None and abs_td_errors is not None:
                            # evaluate_qからのabs_td_errorsは、サンプルの合計TD誤差に対する(batch_size,)です。
                            self.replay_buffer.update_priorities(sampled_indices, abs_td_errors.detach().cpu().numpy())

                        step_losses.append(loss.item())

                    # Sync target network periodically
                    if total_step > 0 and total_step % self.target_update_frequency == 0: # self.target_update_frequencyを使用
                        self.master_agent.sync_target_network()

                    # PER: Beta annealing
                    if self.use_per: # PERがTrueの場合のみベータアニーリング
                        # ベータが1.0を超えないようにします
                        beta_increment_per_learning_step = (1.0 - self.args.beta) / self.beta_anneal_steps # 初期ベータ値から増分を計算
                        self.beta = min(1.0, self.beta + beta_increment_per_learning_step)

                # 次のステップのために状態を更新
                agent_obs_tensor = next_agent_obs_tensor
                global_state_tensor = next_global_state_tensor
                true_current_global_state = step_info['raw_global_state'] # ログのために真のグローバル状態も更新
                step_count += 1
                total_step += 1

                if step_info['all_agents_done']: # もしエピソードが成功したら
                    episode_done = True
                    # print(f"DEBUG: Episode {episode} finished successfully at step: {step_count}")
                elif step_count == self.max_ts: # もし最大ステップ数に達したら
                    # print(f"DEBUG: Episode {episode} timed out at step: {step_count}")
                    pass

            # ---------------------------
            # エピソード終了後の処理
            # ---------------------------

            # ログ表示
            if (episode % CONSOLE_LOG_FREQ == 0) and (episode!=self.start_episode):
                print() # 改行して進捗表示をクリア

                # エピソードごとの平均損失、平均ステップ、平均報酬を計算し、表示に追加
                avg_loss   = sum(episode_losses) / len(episode_losses)      # 期間内の平均損失
                avg_reward = sum(episode_rewards) / len(episode_rewards)    # 期間内の平均報酬
                avg_step   = sum(episode_steps) / len(episode_steps)        # 期間内の平均ステップ数
                done_rate  = sum(done_counts) / len(done_counts)            # 達成率

                print(f"     エピソード {episode - CONSOLE_LOG_FREQ+1} ~ {episode} の平均 step  : {GREEN}{avg_step:.3f}{RESET},\t 最小/最大:{GREEN}{min(episode_steps)}{RESET}/{RED}{max(episode_steps)}{RESET}")
                print(f"     エピソード {episode - CONSOLE_LOG_FREQ+1} ~ {episode} の平均 reward: {GREEN}{avg_reward:.3f}{RESET},\t 最大/最小:{GREEN}{max(episode_rewards):.2f}{RESET}/{RED}{min(episode_rewards):.2f}{RESET}")
                print(f"     エピソード {episode - CONSOLE_LOG_FREQ+1} ~ {episode} の平均 loss  : {GREEN}{avg_loss:.5f}{RESET},\t 最小/最大:{GREEN}{min(episode_losses):.3f}{RESET}/{RED}{max(episode_losses):.3f}{RESET}")
                print(f"     エピソード {episode - CONSOLE_LOG_FREQ+1} ~ {episode} の達成率     : {GREEN}{done_rate:.2f}{RESET}") # 達成率も出力 .2f で小数点以下2桁表示
                if self.use_per:
                    print(f"     (Step: {total_step}), 探索率 : {GREEN}{self.epsilon:.3f}{RESET}, beta: {GREEN}{self.beta:.3f}{RESET}")
                else:
                    print(f"     (Step: {total_step}), 探索率 : {GREEN}{self.epsilon:.3f}{RESET}")

                episode_losses = [] # 100エピソードごとに損失リストもリセット
                episode_rewards = []
                episode_steps  = []
                done_counts = []

            if episode % 200 == 0:
                self.save_checkpoint(episode)

            # エピソードが完了 (episode_done == True) した場合、達成エピソード数カウンタをインクリメント
            if step_info['all_agents_done']:
                episode_steps.append(step_count)

            # エピソードの平均損失を計算
            episode_loss:float = sum(step_losses)/len(step_losses) if step_losses else 0.0
            episode_step:int = step_count

            # Saverでエピソードごとのスコアをログに記録
            self.saver.log_episode_data(episode, step_count, episode_reward, episode_loss, step_info['all_agents_done'])

            # 集計期間内の平均計算のための累積 (avg_reward_temp accumulation)
            episode_losses.append(episode_loss)
            episode_rewards.append(episode_reward)
            episode_steps.append(episode_step)
            done_counts.append(1 if step_info['all_agents_done'] else 0)

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
            model_io.save(mixing_net_path, self.master_agent.mixer_network.state_dict())
        # If VDN, no mixing network to save
        elif isinstance(self.master_agent, VDNMasterAgent):
            pass # No mixing network to save for VDN
        # If DICG, save MixingNetwork weights
        elif isinstance(self.master_agent, DICGMasterAgent):
            mixing_net_path = os.path.join(model_dir, "mixing_network.pth")
            model_io.save(mixing_net_path, self.master_agent.mixer_network.state_dict())

    def load_model_weights(self):
        model_io = Model_IO()
        model_dir = os.path.join(self.save_dir, "models")

        # Load AgentNetwork weights
        agent_net_path = os.path.join(model_dir, "agent_network.pth")
        loaded_agent_net_state_dict = model_io.load(agent_net_path)
        self.master_agent.agent_network.load_state_dict(loaded_agent_net_state_dict)
        self.master_agent.agent_network.eval()

        # Sync target networks after loading
        self.master_agent.sync_target_network()

        # If QMIX, load MixingNetwork weights
        if isinstance(self.master_agent, QMIXMasterAgent):
            mixing_net_path = os.path.join(model_dir, "mixing_network.pth")
            loaded_mixing_net_state_dict = model_io.load(mixing_net_path)
            self.master_agent.mixer_network.load_state_dict(loaded_mixing_net_state_dict)
            self.master_agent.mixer_network.eval()
        # If VDN, no mixing network to load
        elif isinstance(self.master_agent, VDNMasterAgent):
            pass # No mixing network to load for VDN
        # If DICG, load MixingNetwork weights
        elif isinstance(self.master_agent, DICGMasterAgent):
            mixing_net_path = os.path.join(model_dir, "mixing_network.pth")
            loaded_mixing_net_state_dict = model_io.load(mixing_net_path)
            self.master_agent.mixer_network.load_state_dict(loaded_mixing_net_state_dict)
            self.master_agent.mixer_network.eval()


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
            self.master_agent.mixer_network.eval()
        elif isinstance(self.master_agent, VDNMasterAgent):
            pass # No mixing network for VDN
        elif isinstance(self.master_agent, DICGMasterAgent):
            self.master_agent.mixer_network.eval()

        for episode_idx in range(1, num_simulation_episodes + 1):
            print(f"--- シミュレーションエピソード {episode_idx} / {num_simulation_episodes} ---")

            # エージェントの初期位置を設定
            # current_partial_observations は agent_obs_tensor, global_state_tensor, info を返します。
            agent_obs_tensor, global_state_tensor, info = self.env_wrapper.reset(initial_agent_positions=[(0,i) for i in range(self.agents_number)])
            current_global_state_for_logging = info['raw_global_state'] # Reset後の真のグローバル状態を取得

            episode_done = False
            step_count = 0
            ep_reward = 0.0

            while not episode_done and step_count < max_simulation_timestep:
                print(f"\nステップ {step_count}:")
                print(f"  現在の全体状態: {current_global_state_for_logging}")

                # Get actions from master_agent (epsilon=0 for greedy actions)
                actions_dict = self.master_agent.get_actions(agent_obs_tensor, epsilon=0.0)

                # Display Q-values (This part requires MasterAgent to expose a way to get all Q-values, or replicate logic)
                # For now, let's just display the chosen actions
                print(f"  選択された行動: {actions_dict}")

                # 環境にステップを与えて状態を更新し、結果を取得
                next_agent_obs_tensor, next_global_state_tensor, rewards_tensor, dones_tensor, step_info = self.env_wrapper.step(actions_dict)

                # 次のステップの観測を current にコピー
                agent_obs_tensor = next_agent_obs_tensor
                global_state_tensor = next_global_state_tensor
                current_global_state_for_logging = step_info['raw_global_state'] # ログのためにグローバル状態も更新

                ep_reward += rewards_tensor.sum().item()
                episode_done = step_info['all_agents_done']

                print(f"  報酬: {rewards_tensor.sum().item():.2f}, 完了: {episode_done}")

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

                if isinstance(self.master_agent, QMIXMasterAgent) or isinstance(self.master_agent, DICGMasterAgent):
                    mixing_net_checkpoint_path_ep = os.path.join(model_dir, f"checkpoint_episode[{episode}]_mixing_network.pth")
                    if os.path.exists(mixing_net_checkpoint_path_ep):
                        mixing_net_checkpoint_path = mixing_net_checkpoint_path_ep
                # For VDN, no mixing network checkpoint

            # Load AgentNetwork checkpoint
            agent_net_checkpoint_data = model_io.load_checkpoint(agent_net_checkpoint_path)
            self.master_agent.agent_network.load_state_dict(agent_net_checkpoint_data['model_state'])
            self.master_agent.agent_network_target.load_state_dict(agent_net_checkpoint_data['target_state'])

            # Load optimizer state (assuming one optimizer for all networks, saved with agent_network)
            self.optimizer.load_state_dict(agent_net_checkpoint_data['optimizer_state'])

            # Update start_episode and epsilon/beta
            self.start_episode = agent_net_checkpoint_data['epoch'] + 1 # Fixed: should be + 1 to resume from next episode
            self.epsilon = agent_net_checkpoint_data['epsilon']
            if 'beta' in agent_net_checkpoint_data: # betaが保存されている場合のみロード
                self.beta = agent_net_checkpoint_data['beta']

            # Set networks to train mode
            self.master_agent.agent_network.train()
            self.master_agent.agent_network_target.eval() # Ensure target network is in eval mode

            if isinstance(self.master_agent, QMIXMasterAgent) or isinstance(self.master_agent, DICGMasterAgent):
                self.master_agent.mixer_network.train()
                self.master_agent.mixer_network_target.eval() # Ensure target network is in eval mode
                mixing_net_checkpoint_data = model_io.load_checkpoint(mixing_net_checkpoint_path)
                self.master_agent.mixer_network.load_state_dict(mixing_net_checkpoint_data['model_state'])
                self.master_agent.mixer_network_target.load_state_dict(mixing_net_checkpoint_data['target_state'])
            # For VDN, no mixing network to load
            elif isinstance(self.master_agent, VDNMasterAgent):
                pass # No mixing network for VDN

        except Exception as e:
            print(f"エラー発生: {e}")
            print("チェックポイントのロードに失敗しました。新規学習を開始します。")
            # If loading fails, ensure networks are in a consistent state (e.g., train mode)
            self.master_agent.agent_network.train()
            self.master_agent.agent_network_target.eval()
            if isinstance(self.master_agent, QMIXMasterAgent) or isinstance(self.master_agent, DICGMasterAgent):
                self.master_agent.mixer_network.train()
                self.master_agent.mixer_network_target.eval()
            elif isinstance(self.master_agent, VDNMasterAgent):
                pass # No mixing network for VDN

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
            epsilon=self.epsilon,
            beta=self.beta # Optionally save beta as well
        )
        model_io.save_checkpoint(
            agent_net_file_path_episode,
            self.master_agent.agent_network.state_dict(),
            self.master_agent.agent_network_target.state_dict(),
            self.optimizer.state_dict(),
            episode,
            epsilon=self.epsilon,
            beta=self.beta
        )

        # If QMIX, save MixingNetwork checkpoint (latest and episode-specific)
        if isinstance(self.master_agent, QMIXMasterAgent) or isinstance(self.master_agent, DICGMasterAgent):
            mixing_net_file_path = os.path.join(model_dir, "checkpoint_mixing_network.pth")
            mixing_net_file_path_episode = os.path.join(model_dir, f"checkpoint_episode[{episode}]_mixing_network.pth")
            model_io.save_checkpoint(
                mixing_net_file_path,
                self.master_agent.mixer_network.state_dict(),
                self.master_agent.mixer_network_target.state_dict(),
                None, # Optimizer state already saved with agent_network, or needs separate handling for mixing network optimizer
                episode,
                epsilon=self.epsilon,
                beta=self.beta
            )
            model_io.save_checkpoint(
                mixing_net_file_path_episode,
                self.master_agent.mixer_network.state_dict(),
                self.master_agent.mixer_network_target.state_dict(),
                None,
                episode,
                epsilon=self.epsilon,
                beta=self.beta
            )
        # If VDN, no mixing network to save
        elif isinstance(self.master_agent, VDNMasterAgent):
            pass # No mixing network to save for VDN
