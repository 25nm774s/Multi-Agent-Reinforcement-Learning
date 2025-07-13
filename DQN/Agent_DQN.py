"""
エージェントの行動を出力.
学習器の更新など.
"""
import torch
import numpy as np
#import os
#import sys


from utils.replay_buffer import ReplayBuffer
from DQN.dqn import DQNModel
#from DQN.dqn import QNet

import torch

MAX_EPSILON = 1.0
MIN_EPSILON = 0.01

# Assuming these modules are available in the environment or will be provided
# from env import GridWorld
# from DQN.Agent_DQN import Agent_DQN # Modified below
# from DQN.MultiAgent_DQN import MultiAgent_DQN # Modified below
# from utils.Model_Saver import Saver # Modified below
# from utils.plot_results import PlotResults
# from utils.grid_renderer import GridRenderer
from DQN.dqn import DQNModel # Modified below
from utils.replay_buffer import ReplayBuffer # Modified below


# Add beta and beta_anneal_steps to Agent_DQN.__init__
# Modify Agent_DQN.learn_from_experience to use sample(self.beta), pass indices/weights to model.update, and update beta
class Agent_DQN:
    """
    DQN エージェントクラス.

    環境とのインタラクション、経験の収集、リプレイバッファへの保存、
    および DQN モデルの学習ロジックを管理します。
    """
    def __init__(self, args):
        """
        Agent_DQN クラスのコンストラクタ.

        Args:
            args: エージェントの設定を含む属性を持つオブジェクト (例: argparse.Namespace).
                  必要な属性: agents_number, batch_size, decay_epsilon,
                  load_model, goals_num, mask, device, buffer_size,
                  optimizer, gamma, learning_rate, target_update_frequency,
                  alpha, beta, beta_anneal_steps.
        """
        self.agents_num = args.agents_number
        self.batch_size = args.batch_size
        self.decay_epsilon_step = args.decay_epsilon
        self.action_size = 5
        self.epsilon = MAX_EPSILON
        self.load_model = args.load_model
        self.goals_num = args.goals_number
        self.mask = args.mask
        self.device = torch.device(args.device)

        # PER パラメータの追加 (Step 1)
        self.alpha = args.alpha if hasattr(args, 'alpha') else 0.6
        self.beta = args.beta if hasattr(args, 'beta') else 0.4 # Betaの初期値
        self.beta_anneal_steps = args.beta_anneal_steps if hasattr(args, 'beta_anneal_steps') else args.episode_number # ベータを1.0まで増加させるエピソード数

        # ReplayBuffer の初期化 (PER パラメータ alpha を渡す) (Step 1 & 10)
        # learning_mode は ReplayBuffer の get_batch の挙動に影響するため、args から渡すか適切に設定する必要がある
        # 現在の ReplayBuffer 実装は 'V', 'Q', 'else' で分岐しており、DQNでは 'else' が使われる想定
        # 'else' ブランチに学習モードを指定する必要はないが、ReplayBuffer の __init__ に learning_mode があるため仮に渡す
        # args に learning_mode がない場合はデフォルト値を設定するか、args に追加が必要
        """　将来learning_modeはなくす　"""
        # Pass alpha to ReplayBuffer (Step 10)
        self.replay_buffer = ReplayBuffer("DQN", args.buffer_size, self.batch_size, self.device, alpha=self.alpha)


        # Agent_DQN の内部で DQNModel を初期化 (use_per フラグを追加) (Step 5)
        self.model = DQNModel(
            args.optimizer,
            args.gamma,
            args.batch_size,
            self.agents_num,
            self.goals_num,
            args.load_model,
            args.learning_rate,
            args.mask,
            args.device,
            100,#args.target_update_frequency if hasattr(args, 'target_update_frequency') else 100,
            use_per=True # PERを使用することをモデルに伝える (Step 5)
        )

    def get_action(self, i: int, global_state: tuple) -> int:
        """
        現在の全体状態に基づいて、エージェントの行動を決定する (ε-greedy).

        Args:
            i (int): 行動を決定するエージェントのインデックス.
            global_state (tuple): 環境の現在の全体状態 (ゴール位置と全エージェント位置のタプル).

        Returns:
            int: 選択された行動 (0:UP, 1:DOWN, 2:LEFT, 3:RIGHT, 4:STAY).
        """
        #全体状態をNNの入力形式に変換
        # 現在のGridWorldの状態表現はタプルなので、フラット化してPyTorchテンソルに変換
        flat_global_state = np.array(global_state).flatten()
        global_state_tensor = torch.tensor(flat_global_state, dtype=torch.float32) # 1次元のテンソルに変換

        # ε-greedyに基づいて行動を選択
        return self.nn_greedy_actor(i, global_state_tensor)

    # Q系列のNNモデル使用時のε-greedy
    def nn_greedy_actor(self, i: int, global_state_tensor: torch.Tensor) -> int:
        """
        ε-greedy法を用いて、現在の状態から行動を選択する。
        maskがTrueの場合、エージェント自身の状態のみをQNetへの入力とする。
        maskがFalseの場合、全体状態をQNetへの入力とする。

        Args:
            i (int): 行動を選択するエージェントのインデックス.
            global_state_tensor (torch.Tensor): 環境の現在の全体状態を表すテンソル.

        Returns:
            int: 選択された行動 (0-4).
        """
        # ε-greedyによる探索か活用かの決定
        if np.random.rand() < self.epsilon:
            # 探索: ランダムに行動を選択
            return np.random.choice(self.action_size)
        else:
            # 活用: Q値に基づいて最適な行動を選択
            # QNetへの入力となる状態を準備 (masking)
            if self.mask:
                # マスクがTrueの場合、全体状態テンソルからエージェントi自身の位置情報のみを抽出
                # 全体状態テンソルの構造: [goal1_x, g1_y, ..., agent1_x, a1_y, ..., agent_i_x, a_i_y, ...]
                # goals_num * 2 がゴール部分の次元数
                # i * 2 がエージェントiの開始インデックス (0-indexed)
                agent_state_tensor = global_state_tensor[self.goals_num * 2 + i * 2 : self.goals_num * 2 + i * 2 + 2] # (x, y)の2次元を抽出
            else:
                # マスクがFalseの場合、全体状態テンソルをそのまま使用
                agent_state_tensor = global_state_tensor # 形状: ((goals+agents)*2,)

            # QNetはバッチ入力を想定しているため、単一の状態テンソルをバッチ次元を追加して渡す
            # unsqueeze(0) で形状を (1, state_dim) にする
            agent_state_tensor = agent_state_tensor.unsqueeze(0).to(self.device)

            # QNetを使って各行動のQ値を計算
            with torch.no_grad(): # 推論時は勾配計算を無効化
                qs = self.model.qnet(agent_state_tensor) # 形状: (1, action_size)

            # 最大Q値に対応する行動のインデックスを取得
            return qs.argmax().item()

    # epsilonの線形アニーリング (現在のコードでは power アニーリングが使われているが、参考として残す)
    def decay_epsilon(self, step):
        if step < self.decay_epsilon_step:
            self.epsilon = MIN_EPSILON + (MAX_EPSILON - MIN_EPSILON) * (self.decay_epsilon_step - step) / self.decay_epsilon_step
        else:
            self.epsilon = MIN_EPSILON

    def decay_epsilon_power(self,step:int,alpha=0.5):
        """
        εをステップ数に基づいてべき乗で減衰させる。
        探索率εは step^(-alpha) に比例して減少する。

        Args:
            step (int): 現在のステップ数.
            alpha (float, optional): 減衰率を調整するパラメータ. Defaults to 0.9.
        """
        # ゼロ除算対策
        effective_step = max(1, step)
        self.epsilon = MAX_EPSILON * (1.0 / (effective_step ** alpha))
        # 必要に応じて MIN_EPSILON で下限を設ける
        self.epsilon = max(MIN_EPSILON, self.epsilon)

    def observe_and_store_experience(self, global_state: tuple, action: int, reward: float, next_global_state: tuple, done: bool) -> None:
        """
        環境からの単一ステップの経験 (全体状態, 行動, 報酬, 次の全体状態, 完了フラグ) をリプレイバッファに追加する。

        Args:
            global_state (tuple): 現在の全体状態.
            action (int): エージェントが取った行動.
            reward (float): 行動によって得られた報酬.
            next_global_state (tuple): 環境の次の全体状態.
            done (bool): エピソードが完了したかどうかのフラグ.
        """
        self.replay_buffer.add(global_state, action, reward, next_global_state, done)

    def learn_from_experience(self, i: int, episode_num: int, total_episode_num: int) -> float | None:
        """
        リプレイバッファからバッチを取得し、モデルを学習させる。
        バッチサイズに満たない場合は学習を行わない。PERを使用する場合は、
        サンプリングされた経験の優先度をTD誤差に基づいて更新する。

        Args:
            i (int): 学習を行うエージェントのインデックス.
            episode_num (int): 現在のエピソード番号 (ターゲットネットワーク更新タイミングに使用).
            total_episode_num (int): 全体のエピソード数 (betaアニーリング用).

        Returns:
            float | None: 計算された損失の平均値 (学習が行われた場合)、または None (学習が行われなかった場合).
        """
        # モデルロード設定が1 (学習済みモデル使用) の場合は学習しない
        if self.load_model == 1:
            return None

        # リプレイバッファにバッチサイズ分の経験が溜まっていない場合は学習しない
        if len(self.replay_buffer) < self.batch_size:
            return None

        # 1. バッチデータの取得 (PER対応 sample メソッドを使用) (Step 2)
        # ReplayBuffer の sample メソッドは経験データに加え、IS重みとサンプリングされたインデックスを返す
        batch_data = self.replay_buffer.sample(self.beta) # betaを渡してサンプリング (Step 2)

        if batch_data is None:
             return None

        # sample メソッドは Tuple[..., is_weights_tensor, sampled_indices] を返すことを期待する
        global_states_batch, actions_batch, rewards_batch, next_global_states_batch, dones_batch, is_weights_batch, sampled_indices = batch_data

        # 2. モデルの更新
        # DQNModel の update メソッドにバッチデータ、IS重み、サンプリングされたインデックスを渡す (Step 3)
        # update メソッドは float | None (学習損失) と TD誤差 (PER用) を返すように修正されている
        loss, td_errors = self.model.update(
            i,
            global_states_batch,
            actions_batch,
            rewards_batch,
            next_global_states_batch,
            dones_batch,
            episode_num,
            is_weights_batch, # IS重みを渡す (Step 3)
            sampled_indices # サンプリングされたインデックスを渡す (PERの優先度更新はReplayBufferで行うが、TD誤差計算はModelで行うためここで渡す必要はないかもしれない -> update内でTD誤差を計算して返すようにする)
        )

        # 3. PER: 優先度の更新 (Step 9)
        # モデルの update メソッドから計算されたTD誤差の絶対値を取得し、リプレイバッファの優先度を更新
        if td_errors is not None and sampled_indices is not None:
             self.replay_buffer.update_priorities(sampled_indices, td_errors.detach().cpu().numpy()) # TD誤差をCPUに移動しNumPyに変換

        # 4. PER: Betaの線形アニーリング (Step 4)
        # エピソードの進行に応じて beta を線形的に 1.0 まで増加させる
        # beta_increment_per_episode = (1.0 - args.beta) / args.beta_anneal_steps # args.beta_anneal_steps は総エピソード数か、βを1にするステップ数
        beta_increment_per_episode = (1.0 - (self.beta)) / (self.beta_anneal_steps)
        self.beta = min(1.0, self.beta + beta_increment_per_episode)

        return loss
