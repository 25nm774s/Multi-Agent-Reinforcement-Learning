import torch
import torch.optim as optim
import torch.nn as nn
import torch.nn.functional as F

class QNet(nn.Module):
    """
    DQNで使用されるQネットワークモデル.
    状態を入力として受け取り、各行動に対するQ値を出力する.
    """
    def __init__(self, input_size: int, output_size: int):
        """
        QNet クラスのコンストラクタ.

        Args:
            input_size (int): Qネットワークへの入力サイズ (状態空間の次元).
            output_size (int): Qネットワークの出力サイズ (行動空間の次元).
        """
        super().__init__()

        self.fc1 = nn.Linear(input_size, 128)
        self.fc2 = nn.Linear(128, 128)
        self.fc3 = nn.Linear(128, output_size)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        順伝播処理.

        Args:
            x (torch.Tensor): 入力状態のテンソル (形状: (batch_size, input_size)).

        Returns:
            torch.Tensor: 各行動に対するQ値のテンソル (形状: (batch_size, output_size)).
        """
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x

class DQNModel:
    """
    DQN (Deep Q-Network) の学習ロジックを管理するクラス.
    Qネットワーク、ターゲットネットワーク、最適化、損失計算などを担当する.
    """

    def __init__(self, optimizer_type: str, gamma: float, batch_size: int, agent_num: int,
                 goals_num: int, load_model: int, learning_rate: float, mask: bool, target_update_frequency: int = 100):
        """
        DQNModel クラスのコンストラクタ.

        Args:
            optimizer_type (str): 使用するオプティマイザの種類 ('Adam' または 'RMSProp').
            gamma (float): 割引率.
            batch_size (int): 学習に使用するバッチサイズ.
            agent_num (int): 環境中のエージェント数.
            goals_num (int): 環境中のゴール数.
            load_model (int): モデルのロード設定 (0: 学習, 1: 学習済みモデル使用(推論), 2: 知識蒸留/模倣学習).
            learning_rate (float): 学習率.
            mask (bool): 状態にマスキングを適用するかどうか (True: 自身の位置のみ, False: 全体状態).
            target_update_frequency (int, optional): ターゲットネットワークを更新する頻度 (エピソード数). Defaults to 100.
        """
        self.gamma: float = gamma
        self.batch_size: int = batch_size
        self.agents_num: int = agent_num
        self.goals_num: int = goals_num
        self.load_model: int = load_model
        self.mask: bool = mask
        self.lr: float = learning_rate
        self.action_size: int = 5
        self.target_update_frequency: int = target_update_frequency

        # mask設定に応じた入力サイズ計算
        # マスクモード時は自身の位置(x,y)で2次元、非マスク時は全体状態の次元 ((goals+agents)*2)
        input_size: int = 2 if self.mask else (self.agents_num + self.goals_num) * 2

        self.qnet_target: QNet = QNet(input_size, self.action_size)
        self.qnet: QNet = QNet(input_size, self.action_size)

        # オプティマイザの初期化
        if optimizer_type == 'Adam':
            self.optimizer: optim.Optimizer = optim.Adam(self.qnet.parameters(), lr=self.lr)
        elif optimizer_type == 'RMSProp':
            self.optimizer: optim.Optimizer = optim.RMSprop(self.qnet.parameters(), lr=self.lr)
        else:
            print(f"Warning: Optimizer type '{optimizer_type}' not recognized. Using Adam as default.")
            self.optimizer: optim.Optimizer = optim.Adam(self.qnet.parameters(), lr=self.lr)


    def huber_loss(self, q: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        """
        Huber Loss を計算する.

        Args:
            q (torch.Tensor): 予測されたQ値のテンソル.
            target (torch.Tensor): ターゲットQ値のテンソル.

        Returns:
            torch.Tensor: 計算されたHuber Loss.
        """
        err: torch.Tensor = target - q
        abs_err: torch.Tensor = torch.abs(err)

        # 標準偏差を用いた動的なHUBER_LOSS_DELTAの設定
        # バッチサイズが1より大きい場合のみ標準偏差を計算
        huber_loss_delta: float
        if len(err) > 1:
            huber_loss_delta = torch.mean(abs_err).item() + torch.std(err).item()
        else:
            huber_loss_delta = torch.mean(abs_err).item()

        # HUBER_LOSS_DELTA がゼロに近すぎる場合（全ての誤差がゼロに近いなど）の対策
        if huber_loss_delta < 1e-6:
             huber_loss_delta = 1.0

        # Huber loss implementation
        # If |err| < delta, use 0.5 * err^2 (L2)
        # If |err| >= delta, use delta * (|err| - 0.5 * delta) (L1)
        cond: torch.Tensor = abs_err < huber_loss_delta
        L2: torch.Tensor = 0.5 * torch.square(err)
        L1: torch.Tensor = huber_loss_delta * (abs_err - 0.5 * huber_loss_delta)
        loss: torch.Tensor = torch.where(cond, L2, L1)

        return torch.mean(loss)

    def _extract_agent_state(self, i: int, global_states_batch: torch.Tensor, next_global_states_batch: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """
        複数エージェントの全体の状態バッチから、特定エージェントの状態バッチを抽出する。
        mask が True の場合は、エージェント i の位置情報 (x, y) のみを抽出する。
        mask が False の場合は、全体状態バッチをそのまま返す。

        Args:
            i (int): 状態を抽出するエージェントのインデックス.
            global_states_batch (torch.Tensor): 現在の全体状態のバッチ.
            next_global_states_batch (torch.Tensor): 次の全体状態のバッチ.

        Returns:
            tuple[torch.Tensor, torch.Tensor]: 抽出された (または全体の)
                                               現在のエージェントの状態バッチと次のエージェントの状態バッチ.
                                               形状は (batch_size, agent_state_dim).
        """
        if not self.mask:
            # mask が False の場合は、全体の状態バッチをそのまま返す
            return global_states_batch, next_global_states_batch

        # 複数エージェントの全体の状態バッチから、現在のエージェントの状態バッチを抽出
        # global_states_batch の形状は (batch_size, 全体の状態次元) を想定
        # 例: (batch_size, (goals_num + agents_num) * 2)
        # 抽出したいのは各バッチサンプルの、特定エージェントの状態に対応する部分
        # idx_offset はゴール座標の合計次元数
        idx_offset: int = self.goals_num * 2
        # agent_idx_start は現在のエージェントのX座標の開始インデックス
        agent_idx_start: int = i * 2 + idx_offset
        # idx_lst は (X, Y)座標のインデックスリスト
        idx_lst: list[int] = [agent_idx_start, agent_idx_start + 1]

        # バッチ内のすべてのサンプルについて、指定された特徴量のみを抽出
        # .clone() は、元のテンソルから独立したコピーを作成し、元のテンソルへの意図しない変更を防ぐ
        # global_states_batch[:, idx_lst] は形状 (batch_size, len(idx_lst)) -> (batch_size, 2)
        agent_states_batch: torch.Tensor = global_states_batch[:, idx_lst].clone()
        next_agent_states_batch: torch.Tensor = next_global_states_batch[:, idx_lst].clone()

        # 抽出したエージェントの状態バッチを返す (形状: (batch_size, 2))
        return agent_states_batch, next_agent_states_batch


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

    def _calculate_next_max_q_values(self, next_agent_states_batch: torch.Tensor) -> torch.Tensor: # 引数名を変更
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

    def _perform_standard_dqn_update(self, agent_states_batch: torch.Tensor, action_batch: torch.Tensor, reward_batch: torch.Tensor, next_agent_states_batch: torch.Tensor, done_batch: torch.Tensor, episode_num: int) -> float:
        """
        通常のDQN学習ロジックを実行する。
        (入力は特定エージェントの状態バッチ)

        Args:
            agent_states_batch (torch.Tensor): 現在のエージェントの状態バッチ (形状: (batch_size, agent_state_dim)).
            action_batch (torch.Tensor): バッチ内の各サンプルでエージェントが取った行動のバッチ (形状: (batch_size,)).
            reward_batch (torch.Tensor): 報酬のバッチ (形状: (batch_size,)).
            next_agent_states_batch (torch.Tensor): 次のエージェントの状態バッチ (形状: (batch_size, agent_state_dim)).
            done_batch (torch.Tensor): 完了フラグのバッチ (形状: (batch_size,)).
            episode_num (int): 現在のエピソード番号 (ターゲットネットワーク更新タイミングに使用).

        Returns:
            float: 計算された損失の平均値 (学習が行われた場合).
        """
        # Q値の計算 (特定エージェントの状態バッチを使用)
        predicted_q: torch.Tensor = self._calculate_q_values(agent_states_batch, action_batch)
        next_max_q: torch.Tensor = self._calculate_next_max_q_values(next_agent_states_batch)

        # ターゲットQ値の計算
        target_q: torch.Tensor = self._calculate_target_q_values(reward_batch, done_batch, next_max_q)

        # 損失の計算と最適化
        loss: torch.Tensor = self.huber_loss(predicted_q, target_q)
        self._optimize_network(loss)

        # スカラー損失の計算 (学習の進捗確認用)
        scalar_loss: float = loss.item() # 平均損失を返す

        # ターゲットネットワークの同期
        # episode_num が target_update_frequency の倍数の場合に同期
        if episode_num > 0 and episode_num % self.target_update_frequency == 0: # episode_num > 0 を追加して最初のエピソードでは同期しないようにするなど
             self.sync_qnet()
             # print(f"Episode {episode_num}: Target network synced.") # デバッグ用

        return scalar_loss

    def _perform_knowledge_distillation_update(self, agent_states_batch: torch.Tensor, action_batch: torch.Tensor) -> float:
        """
        学習済みモデルを真の価値関数として改めて学習する特殊な学習ロジックを実行する。
        （知識蒸留や模倣学習に相当）
        (入力は特定エージェントの状態バッチ)

        Args:
            agent_states_batch (torch.Tensor): 現在のエージェントの状態バッチ (形状: (batch_size, agent_state_dim)).
            action_batch (torch.Tensor): バッチ内の各サンプルでエージェントが取った行動のバッチ (形状: (batch_size,)).

        Returns:
            float: 計算された損失の平均値 (学習が行われた場合).
        """
        # メインネットワークの予測Q値 (特定エージェントの状態バッチを使用)
        predicted_q: torch.Tensor = self.qnet(agent_states_batch) # 形状: (batch_size, action_size)

        # ターゲットネットワークを「真の価値関数」として利用
        # 現在のエージェントの状態バッチ `agent_states_batch` をターゲットQネットに入力し、実際に取られた行動のQ値を取得
        # agent_states_batch は形状 (batch_size, agent_state_dim)
        true_qs_batch: torch.Tensor = self.qnet_target(agent_states_batch) # 形状: (batch_size, action_size)
        batch_indices: torch.Tensor = torch.arange(self.batch_size, device=agent_states_batch.device)
        true_q: torch.Tensor = true_qs_batch[batch_indices, action_batch].detach() # detach() はここで

        # 損失の計算と最適化
        loss: torch.Tensor = self.huber_loss(predicted_q, true_q)
        self._optimize_network(loss)

        # スカラー損失の計算 (学習の進捗確認用)
        scalar_loss: float = loss.item() # 平均損失を返す

        # 知識蒸留モードでは通常ターゲットネットワークの同期は行わない？
        # if episode_num % self.target_update_frequency == 0:
        #    self.sync_qnet() # 知識蒸留の目的に応じて同期するか検討が必要

        return scalar_loss


    def update(self, i: int, global_states_batch: torch.Tensor, actions_batch: torch.Tensor, rewards_batch: torch.Tensor, next_global_states_batch: torch.Tensor, dones_batch: torch.Tensor, episode_num: int) -> float | None:
        """
        Qネットワークのメインの更新ロジック。学習モードによって処理を分岐する。
        リプレイバッファから取得したバッチデータ (全体の状態のバッチ) を使用する。

        Args:
            i (int): 更新を行うエージェントのインデックス.
            global_states_batch (torch.Tensor): リプレイバッファからサンプリングされた現在の全体状態のバッチ.
            actions_batch (torch.Tensor): リプレイバッファからサンプリングされた行動のバッチ (エージェント i の行動に対応).
            rewards_batch (torch.Tensor): リプレイバッファからサンプリングされた報酬のバッチ.
            next_global_states_batch (torch.Tensor): リプレイバッファからサンプリングされた次の全体状態のバッチ.
            dones_batch (torch.Tensor): リプレイバッファからサンプリングされた完了フラグのバッチ.
            episode_num (int): 現在のエピソード番号 (ターゲットネットワーク更新タイミングに使用).

        Returns:
            float | None: 計算された損失の平均値 (学習が行われた場合)、または None (学習が行われなかった場合).
        """
        # load_model == 1 の場合は学習は行わない
        if self.load_model == 1:
            return None

        # 1. データの準備とフィルタリング (全体の状態バッチから特定エージェントの状態バッチを抽出)
        agent_states_batch, next_agent_states_batch = self._extract_agent_state(i, global_states_batch, next_global_states_batch)

        # 2. 学習モードの分岐と実行 (特定エージェントの状態バッチを使用)
        # load_model = 0: 通常学習 (DQN)
        # load_model = 1: 学習済みモデル使用 (推論のみ、学習なし) -> 上でハンドリング済み
        # load_model = 2: 知識蒸留/模倣学習など (ターゲットQネットを真の価値関数として利用)
        if self.load_model == 0:
            # 通常のDQN学習モード
            return self._perform_standard_dqn_update(agent_states_batch, actions_batch, rewards_batch, next_agent_states_batch, dones_batch, episode_num)
        elif self.load_model == 2:
             # 特殊な学習モード (知識蒸留/模倣学習など)
             # このモードの場合、rewards_batch, dones_batch は使用しない設計になっている可能性
             # perform_knowledge_distillation_update には actions_batch のみが渡される
             return self._perform_knowledge_distillation_update(agent_states_batch, actions_batch)
        else: # load_model が 0, 1, 2 以外の値の場合 (予期しない値)
            print(f"Warning: Unexpected load_model value: {self.load_model}. No learning performed.")
            return None