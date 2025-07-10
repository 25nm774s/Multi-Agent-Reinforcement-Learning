"""
エージェントの行動を出力.
学習器の更新など.
"""
import torch
import numpy as np
import os
import sys


from utils.replay_buffer import ReplayBuffer # ReplayBuffer クラスは既に上のセルで定義されているため不要
from DQN.dqn import DQNModel # DQNModel クラスは既に上のセルで定義されているため不要
from DQN.dqn import QNet # QNet クラスは既に上のセルで定義されているため不要
from utils.replay_buffer import ReplayBuffer # 重複

np.random.seed(0)
torch.manual_seed(0)

# ターミナルの表示関連
RED = '\033[91m'
GREEN = '\033[92m'
RESET = '\033[0m'

MAX_EPSILON = 1
MIN_EPSILON = 0.01

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
                  optimizer, gamma, learning_rate, target_update_frequency.
        """
        self.agents_num = args.agents_number
        self.batch_size = args.batch_size
        self.decay_epsilon_step = args.decay_epsilon
        self.action_size = 5
        self.epsilon = MAX_EPSILON
        #self.learning_mode = args.learning_mode # learn_from_experience で ReplayBuffer に渡す際に使用される想定？
        self.load_model = args.load_model
        self.goals_num = args.goals_number
        self.mask = args.mask
        #self.model_path = model_path
        self.device = torch.device(args.device)
        # ReplayBuffer の初期化
        # learning_mode は ReplayBuffer の get_batch の挙動に影響するため、args から渡すか適切に設定する必要がある
        # 現在の ReplayBuffer 実装は 'V', 'Q', 'else' で分岐しており、DQNでは 'else' が使われる想定
        # 'else' ブランチに学習モードを指定する必要はないが、ReplayBuffer の __init__ に learning_mode があるため仮に渡す
        # args に learning_mode がない場合はデフォルト値を設定するか、args に追加が必要
        self.replay_buffer = ReplayBuffer("DQN_else", args.buffer_size, self.batch_size, self.device) # learning_mode を仮設定


        # Agent_DQN の内部で DQNModel を初期化
        self.model = DQNModel(
            args.optimizer,
            args.gamma,
            args.batch_size,
            self.agents_num,
            self.goals_num,
            args.load_model, # argsからload_modelを渡す
            args.learning_rate,
            args.mask,
            args.target_update_frequency if hasattr(args, 'target_update_frequency') else 100 # デフォルト値を追加
        )

    """
        # モデルのロード
        if self.load_model == 1 or self.load_model == 2:
            self.model_path = model_path
            self.loading_model(self.model_path)

        self.model = DQNModel(args.optimizer,args.gamma,args.batch_size,args.agents_number,self.goals_num,self.load_model,args.learning_rate,self.mask)

    # 学習済みモデルの存在の確認
    def loading_model(self, model_path):
        if os.path.exists(model_path):
            print('モデルを読み込みました.')
            print(f"from {GREEN}{model_path}{RESET}\n")
        else:
            print(f"学習済みモデル {RED}{model_path}{RESET} が見つかりません.")
            print('学習する場合, load_model=0 に変更してください.\n')
            sys.exit()

    """

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
            alpha (float, optional): 減衰率を調整するパラメータ. Defaults to 0.5.
        """
        # ゼロ除算対策
        effective_step = max(1, step)
        self.epsilon = MAX_EPSILON * (1.0 / (effective_step ** alpha))
        # 必要に応じて MIN_EPSILON で下限を設ける
        self.epsilon = max(MIN_EPSILON, self.epsilon)

    # 価値更新(非推奨)
    def update_brain(self, i, states, action, reward, next_state, done, episode_num):
        """
        非推奨メソッド: 環境とのインタラクション、経験の保存、学習をまとめて行う (副作用が強い).
        代わりに observe_and_store_experience と learn_from_experience を組み合わせて使用することを推奨します。

        Args:
            i (int): 更新を行うエージェントのインデックス.
            states (Any): 環境の現在の全体状態.
            action (int): エージェントが取った行動.
            reward (float): 行動によって得られた報酬.
            next_state (Any): 環境の次の全体状態.
            done (bool): エピソードが完了したかどうかのフラグ.
            episode_num (int): 現在のエピソード番号.

        Returns:
            float | None: 計算された損失の平均値 (学習が行われた場合)、または None (学習が行われなかった場合).
        """
        """
        self.replay_buffer.add(states, action, reward, next_state, done)

        if self.load_model == 1:
            return None # 学習済みモデル使用時は更新なし

        if len(self.replay_buffer) < self.batch_size:
            return None #<-こっちでバグったら単にreturnにする?

        states, action, reward, next_state, done = self.replay_buffer.get_batch()

        # Qネットワークの重み更新
        scalar_loss = self.model.update(i, states, action, reward, next_state, done, episode_num)

        return scalar_loss
        """
        print("Warning: update_brainは非推奨。代わりにobserve_and_store_experienceとlearn_from_experienceを推奨。")
        # 必要であれば、互換性のために内部で呼び出しをラップすることも可能だが、非推奨であることを明確にする
        #self.observe_and_store_experience(states, action, reward, next_state, done)
        #return self.learn_from_experience(i, episode_num)


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

    def learn_from_experience(self, i: int, episode_num: int) -> float | None:
        """
        リプレイバッファからバッチを取得し、モデルを学習させる。
        バッチサイズに満たない場合は学習を行わない。

        Args:
            i (int): 学習を行うエージェントのインデックス.
            episode_num (int): 現在のエピソード番号 (ターゲットネットワーク更新タイミングに使用).

        Returns:
            float | None: 計算された損失の平均値 (学習が行われた場合)、または None (学習が行われなかった場合).
        """
        # モデルロード設定が1 (学習済みモデル使用) の場合は学習しない
        if self.load_model == 1:
            return None # 学習済みモデル使用時は更新なし

        # リプレイバッファにバッチサイズ分の経験が溜まっていない場合は学習しない
        if len(self.replay_buffer) < self.batch_size: # batch_sizeはselfにある
            return None # バッチサイズに満たない場合は学習しない

        # 1. バッチデータの取得
        # ReplayBuffer からは全体状態のバッチが返される (Tensorのタプル)
        batch = self.replay_buffer.get_dqn_batch()

        # get_batchがNoneを返す可能性は、上記のlenチェックで防がれているはずだが、念のため型チェック
        if batch is None:
             return None # ここには到達しない想定だが、安全策として

        # get_batch は Tuple[torch.Tensor, ...] を返すことを期待する
        # 型ヒントに従い、unpack する
        global_states_batch, actions_batch, rewards_batch, next_global_states_batch, dones_batch = batch

        # 2. モデルの更新
        # DQNModel の update メソッドにバッチデータを渡す
        # episode_num はターゲットネットワーク更新タイミングのため必要
        # update メソッドは float | None を返す
        loss = self.model.update(i, global_states_batch, actions_batch, rewards_batch, next_global_states_batch, dones_batch, episode_num)

        return loss