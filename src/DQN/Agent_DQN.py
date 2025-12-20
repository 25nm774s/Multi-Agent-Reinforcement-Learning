import torch
import numpy as np

from utils.replay_buffer import ReplayBuffer
from utils.StateProcesser import StateProcessor
from DQN.dqn import DQNModel, QNet
from Q_learn.strategys.masked_strategies import CooperativeActionSelection

from Base.Agent_Base import AgentBase

MAX_EPSILON = 1.0
MIN_EPSILON = 0.05

class Agent(AgentBase):
    """
    DQN エージェントクラス.

    環境とのインタラクション、経験の収集、リプレイバッファへの保存、
    および DQN モデルの学習ロジックを管理します。
    """
    # Add use_per parameter to __init__ (Step 1)
    def __init__(self, agent_id, args, use_per: bool = False):
        """
        Agent_DQN クラスのコンストラクタ.

        Args:
            args: エージェントの設定を含む属性を持つオブジェクト (例: argparse.Namespace).
                  必要な属性: agents_number, batch_size, epsilon_decay,
                  goals_num, mask, device, buffer_size,
                  optimizer, gamma, learning_rate, target_update_frequency,
                  alpha, beta, beta_anneal_steps.
            use_per (bool, optional): Prioritized Experience Replay を使用するかどうか. Defaults to False. (Step 1)
        """
        super().__init__(agent_id, args)

        self.alpha = args.alpha if hasattr(args, 'alpha') else 0.6
        self.beta = args.beta if hasattr(args, 'beta') else 0.4 # Betaの初期値
        self.beta_anneal_steps = int(args.episode_number * args.max_timestep/4) # ベータを1.0まで増加させるエピソード数
        self.use_per = use_per

        # ReplayBuffer の初期化 (PER パラメータ alpha と use_per を渡す) (Step 2)
        self.replay_buffer = ReplayBuffer(args.buffer_size, self.batch_size, self.device, alpha=self.alpha, use_per=self.use_per)

        # StateProcessor のインスタンスを初期化
        self.state_processor = StateProcessor(args.grid_size, args.goals_number, args.agents_number, self.device)

        # Agent_DQN の内部で DQNModel を初期化 (use_per フラグを追加) (Step 3)
        self.model = DQNModel(
            args.optimizer,
            args.grid_size,
            args.gamma,
            args.batch_size,
            args.agents_number,
            self.goals_num,
            args.learning_rate,
            args.mask,
            args.device,
            args.target_update_frequency,
            use_per=self.use_per, # Pass use_per to DQNModel
            state_processor=self.state_processor
        )

        # self.grid_size = args.grid_size # action selectionのため
        self.state_representation = CooperativeActionSelection(args.grid_size, self.goals_num, self.agent_id, args.agents_number)

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
        # pre_gs = global_state
        global_state = self._get_observation(global_state) # 部分観測に変換(簡易的に)
        # print(pre_gs,"->\n",global_state)
        flat_global_state = np.array(global_state).flatten()
        global_state_tensor = torch.tensor(flat_global_state, dtype=torch.float32) # 1次元のテンソルに変換

        # ε-greedyに基づいて行動を選択
        return self.nn_greedy_actor(i, global_state_tensor)

    def get_all_q_values(self, i: int, global_state: tuple) -> torch.Tensor:
        """
        現在の全体状態における、指定されたエージェントの各行動に対するQ値を取得する。
        `nn_greedy_actor` と同様に状態の前処理を行い、QNetからQ値を取得する。

        Args:
            i (int): Q値を計算するエージェントのインデックス.
            global_state (tuple): 環境の現在の全体状態 (ゴール位置と全エージェント位置のタプル).

        Returns:
            torch.Tensor: 各行動に対するQ値のテンソル (形状: (output_size,)).
        """
        # 全体状態をNNの入力形式に変換
        flat_global_state = np.array(global_state).flatten()
        global_state_tensor = torch.tensor(flat_global_state, dtype=torch.float32) # 1次元のテンソルに変換

        # StateProcessor を使用して QNet への入力状態を準備
        # unsqueeze(0) はバッチ次元を追加するため、個別の状態に対しては事前にStateProcessorに渡す前に適用
        agent_state_tensor = self.state_processor.transform_state_batch(i, global_state_tensor.unsqueeze(0)).to(self.device)

        # QNetを使って各行動のQ値を計算
        with torch.no_grad(): # 推論時は勾配計算を無効化
            qs = self.model.qnet(agent_state_tensor) # 形状: (1, action_size)

        return qs.squeeze(0) # バッチ次元を削除して (action_size,) のテンソルを返す

    # Q系列のNNモデル使用時のε-greedy
    def nn_greedy_actor(self, i: int, global_state_tensor: torch.Tensor) -> int:
        """
        ε-greedy法を用いて、現在の状態から行動を選択する。
        maskがTrueの場合、エージェント自身の状態のみをQNetへの入力とする。
        maskがFalseの場合、全体状態をQNetへの入力とする。

        Args:
            i (int): 行動を選択するエージェントのインデックス.
            global_state_tensor (torch.Tensor): 環境の現在の全体状態を表す1次元テンソル.

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
            # StateProcessor を使用して QNet への入力状態を準備
            # unsqueeze(0) はバッチ次元を追加するため、個別の状態に対しては事前にStateProcessorに渡す前に適用
            agent_state_tensor = self.state_processor.transform_state_batch(i, global_state_tensor.unsqueeze(0)).to(self.device)

            # QNetを使って各行動のQ値を計算
            with torch.no_grad(): # 推論時は勾配計算を無効化
                qs = self.model.qnet(agent_state_tensor) # 形状: (1, action_size)

            # 最大Q値に対応する行動のインデックスを取得
            return qs.argmax().item()

    def _get_observation(self, global_state:tuple[tuple[int,int]]):
        """
        部分観測に対応するために、ストラテジーを流用した。
        
        :param global_state: 環境のタプル表現(例: `((Gx1,Gy1),(Gx2,Gy2),...(GxN,GyN),(Ax1,Ay1),...(AxN,AyN))`)
        """
        return self.state_representation.get_q_state_representation(global_state, self.neighbor_distance)   

    # epsilonの線形アニーリング (現在のコードでは power アニーリングが使われているが、参考として残す)
    def decay_epsilon_linear(self, step):
        if step < self.epsilon_decay:
            self.epsilon = MIN_EPSILON + (MAX_EPSILON - MIN_EPSILON) * (self.epsilon_decay - step) / self.epsilon_decay
        else:
            self.epsilon = MIN_EPSILON

    def decay_epsilon_power(self, step: int):
        """
        ステップ数に基づき、探索率εを指数的に減衰させる関数。
        Args:
            step (int): 現在のステップ数（またはエピソード数）。
        """
        lambda_ = 0.0001
        # 指数減衰式: ε_t = ε_start * (decay_rate)^t
        # self.epsilon = MAX_EPSILON * (self.epsilon_decay ** (step*lambda_))
        self.epsilon *= MAX_EPSILON * (self.epsilon_decay ** (lambda_))

        # 最小値（例: 0.01）を下回らないようにすることが多いが、ここではシンプルな式のみを返します。
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

    def learn_from_experience(self, i: int, total_step: int) -> float | None:
        """
        リプレイバッファからバッチを取得し、モデルを学習させる。
        バッチサイズに満たない場合は学習を行わない。PERを使用する場合は、
        サンプリングされた経験の優先度をTD誤差に基づいて更新する。

        Args:
            i (int): 学習を行うエージェントのインデックス.
            total_step (int): 全ステップ数 (ターゲットネットワーク更新タイミングに使用).

        Returns:
            float | None: 計算された損失の平均値 (学習が行われた場合)、または None (学習が行われなかった場合).
        """
        # モデルロード設定が1 (学習済みモデル使用) の場合は学習しない
        # if self.load_model == 1:
        #     return None

        # リプレイバッファにバッチサイズ分の経験が溜まっていない場合は学習しない
        if len(self.replay_buffer) < self.batch_size:
            return None

        # 1. バッチデータの取得 (PER対応 sample メソッドを使用) (Step 4)
        # ReplayBuffer の sample メソッドは経験データに加え、IS重みとサンプリングされたインデックスを返す
        # Pass beta only if use_per is True (Step 4)
        # 修正: self.use_per が True の場合に self.beta を渡すように修正
        batch_data = self.replay_buffer.sample(self.beta if self.use_per else 0.0) # Pass beta conditionally

        if batch_data is None:
            return None

        # sample メソッドは Tuple[..., is_weights_tensor, sampled_indices] を返すことを期待する
        global_states_batch, actions_batch, rewards_batch, next_global_states_batch, dones_batch, is_weights_batch, sampled_indices = batch_data

        # 2. モデルの更新
        # DQNModel の update メソッドにバッチデータ、IS重み、サンプリングされたインデックスを渡す (Step 5)
        loss, td_errors = self.model.update(
            i,
            global_states_batch,
            actions_batch,
            rewards_batch,
            next_global_states_batch,
            dones_batch,
            total_step,
            is_weights_batch if self.use_per else None, # Pass IS weights conditionally
            sampled_indices if self.use_per else None # Pass sampled indices conditionally
        )

        # 3. PER: 優先度の更新 (Step 6)
        # モデルの update メソッドから計算されたTD誤差の絶対値を取得し、リプレイバッファの優先度を更新
        if self.use_per and td_errors is not None and sampled_indices is not None:
            self.replay_buffer.update_priorities(sampled_indices, td_errors.detach().cpu().numpy()) # TD誤差をCPUに移動しNumPyに変換

        # 4. PER: Betaの線形アニーリング (Step 7)
        # エピソードの進行に応じて beta を線形的に 1.0 まで増加させる
        # Perform beta annealing only if use_per is True (Step 7)
        if self.use_per:
            beta_increment_per_learning_step = (1.0 - (self.beta)) / (self.beta_anneal_steps)
            self.beta = min(1.0, self.beta + beta_increment_per_learning_step)

        return loss

    def get_weights(self):
        return self.model.get_weights()

    def set_weights_for_training(self, qnet_dict, target_dict, optim_dict, epsilon):
        self.model.set_qnet_state(qnet_dict)
        self.model.set_target_state(target_dict)
        self.model.set_optimizer_state(optim_dict)
        
        self.epsilon = epsilon
        
        self.model.qnet.train() # 学習モード

    # 推論用：最小限のデータで実行準備をする
    def set_weights_for_inference(self, q_dict):
        self.model.set_qnet_state(q_dict)
        self.epsilon = 0.0
        self.model.qnet.eval()  # 推論モード
