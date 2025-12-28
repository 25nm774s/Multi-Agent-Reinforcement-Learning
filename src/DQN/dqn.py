import torch
import torch.optim as optim
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Tuple, List

class AgentNetwork(nn.Module):
    """
    DQNで使用されるQネットワークモデル.
    状態を入力として受け取り、各行動に対するQ値を出力する.
    """
    def __init__(self, grid_size: int, output_size: int, total_agents: int = 1):
        """
        AgentNetwork クラスのコンストラクタ.

        Args:
            input_size (int): Qネットワークへの入力サイズ (状態空間の次元).
            output_size (int): Qネットワークの出力サイズ (行動空間の次元).
            total_agents (int): 全エージェント数 (パラメータ共有のためのAgent IDワンホットエンコーディングに使用)
        """
        super().__init__()

        self.grid_size = grid_size
        self.num_channels = 3
        self.total_agents = total_agents # Add total_agents to instance variables

        # 状態入力の次元 (グリッドマップのフラット化されたサイズ)
        state_input_size = self.num_channels * self.grid_size**2

        # Agent ID埋め込み層は削除し、ワンホットエンコーディングを使用するため不要

        # 状態入力とエージェントワンホットベクトルを結合した後の最終的な入力サイズ
        # Agent IDはワンホットベクトルとして扱われるため、そのサイズは total_agents となる
        combined_input_size = state_input_size + self.total_agents

        self.fc1 = nn.Linear(combined_input_size, 128)
        self.fc2 = nn.Linear(128, 128)
        self.fc3 = nn.Linear(128, output_size)

    def forward(self, x: torch.Tensor, agent_ids: torch.Tensor) -> torch.Tensor:
        """
        順伝播処理.

        Args:
            x (torch.Tensor): 入力状態のテンソル (形状: (batch_size, num_channels, grid_size, grid_size)).
            agent_ids (torch.Tensor): エージェントIDのテンソル (形状: (batch_size,)).

        Returns:
            torch.Tensor: 各行動に対するQ値のテンソル (形状: (batch_size, output_size)).
        """
        # 状態をフラット化
        x = x.flatten(start_dim=1)

        # Agent IDをワンホットベクトルに変換
        # agent_ids は (batch_size,) 形状の整数IDを想定
        agent_id_one_hot = F.one_hot(agent_ids, num_classes=self.total_agents).float() # (batch_size, total_agents)

        # 状態とエージェントワンホットベクトルを結合
        x = torch.cat((x, agent_id_one_hot), dim=1)

        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x

class MixingNetwork(nn.Module):
    """
    QMIXで使用されるMixing Network.
    各エージェントのQ値とグローバル状態を入力として受け取り、チーム全体のQ_totを出力する.
    """
    def __init__(self, n_agents: int, state_dim: int, hidden_dim: int = 32):
        """
        MixingNetwork クラスのコンストラクタ.

        Args:
            n_agents (int): エージェントの数.
            state_dim (int): グローバル状態の次元.
            hidden_dim (int): 隠れ層の次元 (HyperNetwork内部で使用).
        """
        super().__init__()
        self.n_agents = n_agents
        self.state_dim = state_dim
        self.hidden_dim = hidden_dim

        # HyperNetwork for weights W1 (state_dim -> n_agents * hidden_dim)
        self.hyper_w1 = nn.Linear(state_dim, self.n_agents * hidden_dim)
        # HyperNetwork for biases B1 (state_dim -> hidden_dim)
        self.hyper_b1 = nn.Linear(state_dim, hidden_dim)

        # HyperNetwork for weights W2 (state_dim -> hidden_dim * 1)
        self.hyper_w2 = nn.Linear(state_dim, hidden_dim * 1)
        # HyperNetwork for biases B2 (state_dim -> 1)
        self.hyper_b2 = nn.Linear(state_dim, 1)

        # Hidden layer for the mixing network itself (shared for all batches)
        self.mix_hidden = nn.Linear(self.n_agents, hidden_dim)

    def forward(self, agent_q_values: torch.Tensor, global_state: torch.Tensor) -> torch.Tensor:
        """
        順伝播処理.

        Args:
            agent_q_values (torch.Tensor): 各エージェントのQ値のテンソル (形状: (batch_size, n_agents, 1)).
            global_state (torch.Tensor): グローバル状態のテンソル (形状: (batch_size, state_dim)).

        Returns:
            torch.Tensor: チーム全体のQ値 (形状: (batch_size, 1)).
        """
        batch_size = agent_q_values.size(0)

        # hyper_w1とhyper_b1からW1とB1を生成
        # W1: (batch_size, state_dim) -> (batch_size, n_agents * hidden_dim)
        # W1を(batch_size, n_agents, hidden_dim)に再成形し、agent_q_valuesとの行列乗算を行う
        w1 = self.hyper_w1(global_state)
        w1 = w1.view(batch_size, self.n_agents, self.hidden_dim)

        # 単調性制約を適用: 重みが非負であることを保証
        w1 = torch.abs(w1) # torch.exp(w1)も使用可能

        # B1: (batch_size, state_dim) -> (batch_size, hidden_dim)
        b1 = self.hyper_b1(global_state)
        b1 = b1.view(batch_size, 1, self.hidden_dim) # Reshape for broadcasting

        # 混合ネットワークの最初の層の出力を計算する
        # (batch_size, n_agents, 1) @ (batch_size, n_agents, hidden_dim) -> (batch_size, 1, hidden_dim)
        hidden = F.elu(torch.bmm(agent_q_values.transpose(1,2), w1) + b1)

        # hyper_w2 および hyper_b2 から W2 および B2 を生成
        # W2: (batch_size, state_dim) -> (batch_size, hidden_dim * 1)
        # Reshape W2 to (batch_size, hidden_dim, 1)
        w2 = self.hyper_w2(global_state)
        w2 = w2.view(batch_size, self.hidden_dim, 1)

        # 単調性制約を適用
        w2 = torch.abs(w2) # Or torch.exp(w2)

        # B2: (batch_size, state_dim) -> (batch_size, 1)
        b2 = self.hyper_b2(global_state)
        b2 = b2.view(batch_size, 1, 1) # Reshape for broadcasting

        # 第2層の出力を計算 (Q_tot)
        # (batch_size, 1, hidden_dim) @ (batch_size, hidden_dim, 1) -> (batch_size, 1, 1)
        Q_tot = torch.bmm(hidden, w2) + b2

        return Q_tot.view(batch_size, 1) # Reshape to (batch_size, 1)

class DQNModel:
    """
    DQN (Deep Q-Network) の学習ロジックを管理するクラス.
    Qネットワーク、ターゲットネットワーク、最適化、損失計算などを担当する.
    Prioritized Experience Replay (PER) に対応.
    """

    # Add use_per parameter to __init__ (Step 1)
    def __init__(self, optimizer_type: str, grid_size: int,gamma: float, batch_size: int, agent_num: int,
                 goals_num: int, learning_rate: float, mask: bool, device:str, target_update_frequency: int = 100, use_per: bool = False, state_processor=None):
        """
        DQNModel クラスのコンストラクタ.

        Args:
            optimizer_type (str): 使用するオプティマイザの種類 ('Adam' または 'RMSProp').
            gamma (float): 割引率.
            batch_size (int): 学習に使用するバッチサイズ.
            agent_num (int): 環境中のエージェント数.
            goals_num (int): 環境中のゴール数.
            learning_rate (float): 学習率.
            mask (bool): 状態にマスキングを適用するかどうか (True: 自身の位置のみ, False: 全体状態).
            device (str): device名
            target_update_frequency (int, optional): ターゲットネットワークを更新する頻度 (エピソード数). Defaults to 100.
            use_per (bool, optional): Prioritized Experience Replay を使用するかどうか. Defaults to False. 
            state_processor (StateProcessor, optional): StateProcessor のインスタンス. None の場合はエラー.
        """
        print("DQNModelが持っていたロジックは、MARLTrainerとMasterAgentサブクラスの責務として再構築されます。")
        print("現在このクラスは不使用。")

        self.grid_size = grid_size
        self.gamma: float = gamma
        self.batch_size: int = batch_size
        self.agents_num: int = agent_num
        self.goals_num: int = goals_num
        self.mask: bool = mask
        self.lr: float = learning_rate
        self.action_size: int = 5
        self.target_update_frequency: int = target_update_frequency
        self.device: torch.device = torch.device(device) # Use passed device string directly

        # PERを使用するかどうかのフラグ
        self.use_per = use_per

        # StateProcessor のインスタンスを保持
        if state_processor is None:
            raise ValueError("StateProcessor instance must be provided to DQNModel.")
        self.state_processor = state_processor

        self.qnet_target: AgentNetwork = AgentNetwork(grid_size,self.action_size).to(self.device)
        self.qnet: AgentNetwork = AgentNetwork(grid_size,self.action_size).to(self.device)

        # オプティマイザの初期化
        if optimizer_type == 'Adam':
            self.optimizer: optim.Optimizer = optim.Adam(self.qnet.parameters(), lr=self.lr)
        elif optimizer_type == 'RMSProp':
            self.optimizer: optim.Optimizer = optim.RMSprop(self.qnet.parameters(), lr=self.lr)
        else:
            print(f"Warning: Optimizer type '{optimizer_type}' not recognized. Using Adam as default.")
            self.optimizer: optim.Optimizer = optim.Adam(self.qnet.parameters(), lr=self.lr)


    # Modify huber_loss to apply IS weights only if self.use_per is True and is_weights is not None (Step 2)
    def huber_loss(self, q: torch.Tensor, target: torch.Tensor, is_weights: Optional[torch.Tensor] = None) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Huber Loss を計算する. PERを使用する場合は重要度サンプリング重みを適用する.
        TD誤差も計算して返す.

        Args:
            q (torch.Tensor): 予測されたQ値のテンソル (形状: (batch_size,)).
            target (torch.Tensor): ターゲットQ値のテンソル (形状: (batch_size,)).
            is_weights (Optional[torch.Tensor]): 重要度サンプリング重みのテンソル (形状: (batch_size,)).

        Returns:
            Tuple[torch.Tensor, torch.Tensor]:
                - 計算されたHuber Loss (スカラー).
                - 計算されたTD誤差 (絶対値) (形状: (batch_size,)).
        """
        # TD誤差の計算
        td_errors: torch.Tensor = target - q
        abs_td_errors: torch.Tensor = torch.abs(td_errors)

        # Huber lossの実装 (元のloss計算)
        HUBER_LOSS_DELTA = 1.0 # 固定値を使用
        cond: torch.Tensor = abs_td_errors < HUBER_LOSS_DELTA
        L2: torch.Tensor = 0.5 * torch.square(td_errors)
        L1: torch.Tensor = HUBER_LOSS_DELTA * (abs_td_errors - 0.5 * HUBER_LOSS_DELTA)
        loss_per_sample: torch.Tensor = torch.where(cond, L2, L1) # 形状: (batch_size,)

        # PERを使用する場合、損失に重要度サンプリング重みを適用 (Step 2)
        if self.use_per and is_weights is not None:
            # 損失 (batch_size,) に IS重み (batch_size,) を要素ごとに乗算
            weighted_loss = loss_per_sample * is_weights # 形状: (batch_size,)
            # 最終的な損失は重み付き損失の平均
            final_loss = torch.mean(weighted_loss)
        else:
            # PERを使用しない場合、通常の平均損失
            final_loss = torch.mean(loss_per_sample)

        # 計算された損失とTD誤差の絶対値を返す
        return final_loss, abs_td_errors.detach() # TD誤差は勾配計算から切り離して返す

    def _calculate_q_values(self, agent_states_batch: torch.Tensor, action_batch: torch.Tensor) -> torch.Tensor:
        """
        現在のエージェントの状態バッチと行動バッチに対応するQ値をメインQネットで計算する。

        Args:
            agent_states_batch (torch.Tensor): 現在のエージェントの状態バッチ (形状: (batch_size, agent_state_dim)).
            action_batch (torch.Tensor): バッチ内の各サンプルでエージェントが取った行動のバッチ (形状: (batch_size,)).

        Returns:
            torch.Tensor: 各サンプルで取られた行動に対応する予測Q値 (形状: (batch_size,)).
        """
        # agent_states_batch は抽出されたエージェントの状態バッチ (batch_size, agent_state_dim)
        predicted_qs_batch: torch.Tensor = self.qnet(agent_states_batch) # 形状: (batch_size, action_size)

        # 各バッチサンプルで取られた行動に対応するQ値を取得
        # action_batch は形状 (batch_size,)
        # torch.arange(self.batch_size) はバッチ内のインデックス [0, 1, ..., batch_size-1]
        batch_indices: torch.Tensor = torch.arange(self.batch_size, device=agent_states_batch.device)
        q_values_taken_action: torch.Tensor = predicted_qs_batch[batch_indices, action_batch] # 形状: (batch_size,)

        return q_values_taken_action

    def _calculate_next_max_q_values(self, next_agent_states_batch: torch.Tensor) -> torch.Tensor:
        """
        次のエージェントの状態バッチにおけるターゲットQネットの最大Q値を計算する。

        Args:
            next_agent_states_batch (torch.Tensor): 次のエージェントの状態バッチ (形状: (batch_size, agent_state_dim)).

        Returns:
            torch.Tensor: 各サンプルで、次の状態でのターゲットQネットによる最大Q値 (形状: (batch_size,)).
        """
        # next_agent_states_batch は抽出された次のエージェントの状態バッチ (batch_size, agent_state_dim)
        next_predicted_qs_target: torch.Tensor = self.qnet_target(next_agent_states_batch) # 形状: (batch_size, action_size)

        # 各バッチサンプルで、次の状態での最大Q値を取得
        next_max_q_values_target: torch.Tensor = next_predicted_qs_target.max(1)[0].detach() # .max(1) は各行（サンプル）ごとに最大値とインデックスを返す
        # [0] で最大値のテンソルを取得。detach() は勾配計算から切り離す
        return next_max_q_values_target

    def _calculate_target_q_values(self, reward_batch: torch.Tensor, done_batch: torch.Tensor, next_max_q_values: torch.Tensor) -> torch.Tensor:
        """
        報酬バッチ、完了フラグバッチ、および次の状態の最大Q値バッチに基づいてターゲットQ値を計算する。
        ベルマン方程式: Target Q = Reward + Gamma * MaxQ(next_state) * (1 - done)

        Args:
            reward_batch (torch.Tensor): 報酬のバッチ (形状: (batch_size,)).
            done_batch (torch.Tensor): 完了フラグのバッチ (形状: (batch_size,)). True/False は 1/0 に変換される.
            next_max_q_values (torch.Tensor): 次の状態における最大Q値のバッチ (形状: (batch_size,)).

        Returns:
            torch.Tensor: ターゲットQ値のバッチ (形状: (batch_size,)).
        """
        # reward_batch, done_batch は形状 (batch_size,)
        # next_max_q_values は形状 (batch_size,)
        target_q_values: torch.Tensor = reward_batch + (1 - done_batch.float()) * self.gamma * next_max_q_values
        return target_q_values

    def _optimize_network(self, loss: torch.Tensor) -> None:
        """
        バックプロパゲーションとパラメータの更新を行う。

        Args:
            loss (torch.Tensor): 計算された損失テンソル.
        """
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

    def sync_qnet(self) -> None:
        """
        ターゲットネットワークをメインネットワークと同期するメソッド。
        メインネットワークの重みをターゲットネットワークにコピーする。
        """
        self.qnet_target.load_state_dict(self.qnet.state_dict())


    # Modify _perform_standard_dqn_update to accept IS weights and return TD errors conditionally (Step 3)
    def _perform_standard_dqn_update(self, agent_states_batch: torch.Tensor, action_batch: torch.Tensor, reward_batch: torch.Tensor, next_agent_states_batch: torch.Tensor, done_batch: torch.Tensor, total_step: int, is_weights_batch: Optional[torch.Tensor] = None) -> Tuple[float, torch.Tensor | None]:
        """
        通常のDQN学習ロジックを実行する。
        (入力は特定エージェントの状態バッチ) PER対応済み。

        Args:
            agent_states_batch (torch.Tensor): 現在のエージェントの状態バッチ (形状: (batch_size, agent_state_dim)).
            action_batch (torch.Tensor): バッチ内の各サンプルでエージェントが取った行動のバッチ (形状: (batch_size,)).
            reward_batch (torch.Tensor): 報酬のバッチ (形状: (batch_size,)).
            next_agent_states_batch (torch.Tensor): 次のエージェントの状態バッチ (形状: (batch_size, agent_state_dim)).
            done_batch (torch.Tensor): 完了フラグのバッチ (形状: (batch_size,)).
            total_step (int): 総ステップ数 (ターゲットネットワーク更新タイミングに使用).
            is_weights_batch (Optional[torch.Tensor]): PERの重要度サンプリング重みバッチ (形状: (batch_size,)).

        Returns:
            Tuple[float, torch.Tensor | None]:
                - 計算された損失の平均値 (スカラー).
                - 計算されたTD誤差 (絶対値) (形状: (batch_size,)) (PER有効時のみ)、または None (PER無効時).
        """
        # Q値の計算 (特定エージェントの状態バッチを使用)
        predicted_q: torch.Tensor = self._calculate_q_values(agent_states_batch, action_batch)
        next_max_q: torch.Tensor = self._calculate_next_max_q_values(next_agent_states_batch)

        # ターゲットQ値の計算
        target_q: torch.Tensor = self._calculate_target_q_values(reward_batch, done_batch, next_max_q)

        # 損失の計算と最適化 (IS重みを渡す) (Step 3)
        loss, abs_td_errors = self.huber_loss(predicted_q, target_q, is_weights_batch) # TD誤差もここで計算されて返される

        self._optimize_network(loss)

        # スカラー損失の計算 (学習の進捗確認用)
        scalar_loss: float = loss.item() # 平均損失を返す

        # ターゲットネットワークの同期
        if total_step > 0 and total_step % self.target_update_frequency == 0:
            self.sync_qnet()
            # print(f"Step {total_step}: Target network synced.") # デバッグ用

        # Return TD errors only if use_per is True (Step 3)
        return scalar_loss, abs_td_errors if self.use_per else None

    # Modify update to accept IS weights and sampled indices, and return TD errors conditionally (Step 4)
    def update(self, i: int, global_states_batch: torch.Tensor, actions_batch: torch.Tensor, rewards_batch: torch.Tensor, next_global_states_batch: torch.Tensor, dones_batch: torch.Tensor, total_step: int, is_weights_batch: Optional[torch.Tensor] = None, sampled_indices: Optional[List[int]] = None) -> Tuple[float | None, torch.Tensor | None]:
        """
        Qネットワークのメインの更新ロジック。学習モードによって処理を分岐する。
        リプレイバッファから取得したバッチデータ (全体の状態のバッチ) を使用する。PER対応済み。

        Args:
            i (int): 更新を行うエージェントのインデックス.
            global_states_batch (torch.Tensor): リプレイバッファからサンプリングされた現在の全体状態のバッチ.
            actions_batch (torch.Tensor): リプレイバッファからサンプリングされた行動のバッチ (エージェント i の行動に対応).
            rewards_batch (torch.Tensor): リプレイバッファからサンプリングされた報酬のバッチ.
            next_global_states_batch (torch.Tensor): リプレイバッバからサンプリングされた次の全体状態のバッチ.
            dones_batch (torch.Tensor): リプレイバッファからサンプリングされた完了フラグのバッチ.
            total_step (int): 全ステップ数 (ターゲットネットワーク更新タイミングに使用).
            is_weights_batch (Optional[torch.Tensor]): PERの重要度サンプリング重みバッチ (形状: (batch_size,)).
            sampled_indices (Optional[List[int]]): PERでサンプリングされた経験の元のバッファ内インデックスリスト.

        Returns:
            Tuple[float | None, torch.Tensor | None]:
                - 計算された損失の平均値 (学習が行われた場合)、または None.
                - 計算されたTD誤差 (絶対値) (形状: (batch_size,)) (PER有効時のみ)、または None.
        """
        # データの準備とフィルタリング (全体の状態バッチから特定エージェントの状態バッチを抽出)
        # StateProcessor を使用してデータを変換
        agent_states_batch_for_NN = self.state_processor.transform_state_batch(i, global_states_batch)
        next_agent_states_batch_for_NN = self.state_processor.transform_state_batch(i, next_global_states_batch)

        loss, td_errors = self._perform_standard_dqn_update(agent_states_batch_for_NN, actions_batch, rewards_batch, next_agent_states_batch_for_NN, dones_batch, total_step, is_weights_batch if self.use_per else None)

        return loss, td_errors if self.use_per else None

    # get_weightsは「状態(dict)」だけを返すようにすると管理が楽です
    def get_weights(self) -> tuple[dict, dict, dict]:
        """
        モデルの重みを返す

        :return: 学習モデル, ターゲットネットワーク重み, オプティマイザーの状態
        :rtype: tuple[dict[Any, Any], dict[Any, Any], dict[Any, Any]]
        """
        return self.qnet.state_dict(), self.qnet_target.state_dict(), self.optimizer.state_dict()

    # インスタンスを差し替えるのではなく、中身(state_dict)を流し込む
    def set_qnet_state(self, state_dict: dict):
        self.qnet.load_state_dict(state_dict)

    # 推論時には呼ばないが、将来の拡張のために残す
    def set_target_state(self, state_dict: dict):
        self.qnet_target.load_state_dict(state_dict)

    def set_optimizer_state(self, state_dict: dict):
        self.optimizer.load_state_dict(state_dict)
