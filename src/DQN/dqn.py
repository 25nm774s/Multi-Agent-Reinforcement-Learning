import torch
import torch.optim as optim
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Tuple, List

class QNet(nn.Module):
    """
    DQNで使用されるQネットワークモデル.
    状態を入力として受け取り、各行動に対するQ値を出力する.
    """
    def __init__(self, grid_size: int, output_size: int):
        """
        QNet クラスのコンストラクタ.

        Args:
            input_size (int): Qネットワークへの入力サイズ (状態空間の次元).
            output_size (int): Qネットワークの出力サイズ (行動空間の次元).
        """
        super().__init__()

        self.grid_size = grid_size
        self.num_channels = 3
        input_size = self.num_channels * self.grid_size**2

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
        x = x.flatten(start_dim=1)

        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x

class DQNModel:
    """
    DQN (Deep Q-Network) の学習ロジックを管理するクラス.
    Qネットワーク、ターゲットネットワーク、最適化、損失計算などを担当する.
    Prioritized Experience Replay (PER) に対応.
    """

    # Add use_per parameter to __init__ (Step 1)
    def __init__(self, optimizer_type: str, grid_size: int,gamma: float, batch_size: int, agent_num: int,
                 goals_num: int, load_model: int, learning_rate: float, mask: bool, device:str, target_update_frequency: int = 100, use_per: bool = False):
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
            device (str): device名
            target_update_frequency (int, optional): ターゲットネットワークを更新する頻度 (エピソード数). Defaults to 100.
            use_per (bool, optional): Prioritized Experience Replay を使用するかどうか. Defaults to False. (Step 1)
        """
        self.grid_size = grid_size
        self.gamma: float = gamma
        self.batch_size: int = batch_size
        self.agents_num: int = agent_num
        self.goals_num: int = goals_num
        self.load_model: int = load_model
        self.mask: bool = mask
        self.lr: float = learning_rate
        self.action_size: int = 5
        self.target_update_frequency: int = target_update_frequency
        self.device: torch.device = torch.device(device) # Use passed device string directly

        # PERを使用するかどうかのフラグ (Step 1)
        self.use_per = use_per

        # mask設定に応じた入力サイズ計算
        # マスクモード時は自身の位置(x,y)で2次元、非マスク時は全体状態の次元 ((goals+agents)*2)
        # chanel: int = 2 if self.mask else (self.agents_num + self.goals_num) * 2

        self.qnet_target: QNet = QNet(grid_size,self.action_size).to(self.device)
        self.qnet: QNet = QNet(grid_size,self.action_size).to(self.device)

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


    def _perform_knowledge_distillation_update(self, agent_states_batch: torch.Tensor, action_batch: torch.Tensor) -> Tuple[float, None]:
        """
        学習済みモデルを真の価値関数として改めて学習する特殊な学習ロジックを実行する。
        （知識蒸留や模倣学習に相当）PERには対応しない。

        Args:
            agent_states_batch (torch.Tensor): 現在のエージェントの状態バッチ (形状: (batch_size, agent_state_dim)).
            action_batch (torch.Tensor): バッチ内の各サンプルでエージェントが取った行動のバッチ (形状: (batch_size,)).

        Returns:
            Tuple[float, None]:
                - 計算された損失の平均値 (学習が行われた場合).
                - TD誤差は計算されないため None.
        """
        # メインネットワークの予測Q値 (特定エージェントの状態バッチを使用)
        predicted_q: torch.Tensor = self.qnet(agent_states_batch) # 形状: (batch_size, action_size)

        # ターゲットネットワークを「真の価値関数」として利用
        true_qs_batch: torch.Tensor = self.qnet_target(agent_states_batch) # 形状: (batch_size, action_size)
        batch_indices: torch.Tensor = torch.arange(self.batch_size, device=agent_states_batch.device)
        true_q: torch.Tensor = true_qs_batch[batch_indices, action_batch].detach()

        # 損失の計算と最適化 (IS重みは使用しない)
        loss, _ = self.huber_loss(predicted_q, true_q, is_weights=None) # TD誤差は不要なので無視

        self._optimize_network(loss)

        # スカラー損失の計算 (学習の進捗確認用)
        scalar_loss: float = loss.item() # 平均損失を返す

        # 知識蒸留モードでは通常ターゲットネットワークの同期は行わない？
        # if episode_num % self.target_update_frequency == 0:
        #    self.sync_qnet() # 知識蒸留の目的に応じて同期するか検討が必要

        # TD誤差は計算されないため None を返す
        return scalar_loss, None


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
        # load_model == 1 の場合は学習は行わない
        if self.load_model == 1:
            return None, None # 損失とTD誤差の両方をNoneで返す

        # 1. データの準備とフィルタリング (全体の状態バッチから特定エージェントの状態バッチを抽出)
        agent_states_batch_for_NN = self.bat_data_transform_for_NN_model_for_batch(i, global_states_batch)
        next_agent_states_batch_for_NN = self.bat_data_transform_for_NN_model_for_batch(i, next_global_states_batch)

        # 2. 学習モードの分岐と実行 (特定エージェントの状態バッチを使用)
        if self.load_model == 0:
            # 通常のDQN学習モード (PER対応)
            # Pass IS weights conditionally, _perform_standard_dqn_update returns TD errors conditionally (Step 4)
            # loss, td_errors = self._perform_standard_dqn_update(agent_states_batch, actions_batch, rewards_batch, next_agent_states_batch, dones_batch, total_step, is_weights_batch if self.use_per else None)
            loss, td_errors = self._perform_standard_dqn_update(agent_states_batch_for_NN, actions_batch, rewards_batch, next_agent_states_batch_for_NN, dones_batch, total_step, is_weights_batch if self.use_per else None)
            # Return TD errors conditionally (Step 4)
            return loss, td_errors if self.use_per else None
        # elif self.load_model == 2:
        #     # 特殊な学習モード (知識蒸留/模倣学習など) (PER非対応)
        #     # _perform_knowledge_distillation_updateはTD誤差を返さない
        #     loss, td_errors = self._perform_knowledge_distillation_update(agent_states_batch, actions_batch)
        #     return loss, None # TD誤差は常にNone
        else: # load_model が 0, 1, 2 以外の値の場合 (予期しない値)
            print(f"Warning: Unexpected load_model value: {self.load_model}. No learning performed.")
            return None, None # 損失とTD誤差の両方をNoneで返す


    def get_weights(self) ->tuple[QNet,QNet,dict]:
        return self.qnet,self.qnet_target,self.optimizer.state_dict()

    def set_model_weights(self, qnet: QNet):
        self.qnet = qnet

    def set_target_weights(self, qnet_target: QNet):
        self.qnet_target = qnet_target

    def bat_data_transform_for_NN_model_for_batch(self, i: int, batch_raw_data: torch.Tensor) -> torch.Tensor:
        """
        一次元の全体状態テンソル (B, feature_dim) を受け取り、
        エージェント i の視点に基づいた3チャネルのグリッド表現テンソル (B, 3, G, G) に変換する。

        Args:
            i (int): 観測を生成するエージェントのインデックス (0-indexed)。
            batch_raw_data (torch.Tensor): リプレイバッファから取り出したバッチテンソル (B, feature_dim)。
                                           全体状態 (ゴールと全エージェントの座標) を含む。

        Returns:
            torch.Tensor: 形状 (batch_size, 3, grid_size, grid_size) のテンソル。
        """

        batch_size = batch_raw_data.size(0)
        G = self.grid_size

        # 座標はインデックスとして使うため、整数型に変換 (Long型推奨)
        coords = batch_raw_data.long().to(self.device)

        # --- 座標の抽出とリシェイプ ---
        goal_coords_end = self.goals_num * 2

        # 1. ゴール座標: (B, goals_num, 2)
        goal_coords = coords[:, :goal_coords_end].reshape(batch_size, self.goals_num, 2)

        # 2. 全エージェント座標: (B, agents_num, 2)
        all_agent_coords = coords[:, goal_coords_end:].reshape(batch_size, self.agents_num, 2)

        # --- グリッドマップの作成 (3チャネル: ゴール, 自身, 他者) ---

        # 最終的な出力テンソルを初期化 (B, 3, G, G)
        state_map = torch.zeros((batch_size, 3, G, G), dtype=torch.float32, device=self.device)

        # 全バッチ、全エンティティに対応するインデックスを準備
        # B * N の数のインデックスが必要
        batch_indices_base = torch.arange(batch_size, device=self.device).repeat_interleave(self.goals_num)

        # --- チャネル 0: ゴールマップの設定 (全バッチ共通) ---

        # x座標とy座標を平坦化
        goal_x = goal_coords[:, :, 0].flatten()
        goal_y = goal_coords[:, :, 1].flatten()

        # state_map[バッチインデックス, チャネル0, x座標, y座標] = 1.0
        state_map[batch_indices_base, 0, goal_x, goal_y] = 1.0

        # --- チャネル 1: 自身のエージェント i のマップの設定 ---

        # 自身のエージェント i の座標 (B, 1, 2) を抽出
        # agents_num が 1 の場合は all_agent_coords がそのまま自身になる
        self_coords = all_agent_coords[:, i, :] # (B, 2)

        # 自身のエージェント i の座標を平坦化 (B, 2 -> B)
        self_x = self_coords[:, 0] # (B,)
        self_y = self_coords[:, 1] # (B,)

        # バッチインデックス (B,)
        batch_indices_self = torch.arange(batch_size, device=self.device)

        # state_map[バッチインデックス, チャネル1, iのx座標, iのy座標] = 1.0
        state_map[batch_indices_self, 1, self_x, self_y] = 1.0

        # --- チャネル 2: 他のエージェントのマップの設定 ---

        if self.agents_num > 1:
            # 他のエージェントの座標を一時的に保持
            other_agents_coords_list = []

            # i 以外のエージェントの座標をリストに追加
            # (B, 2) のテンソルを agents_num - 1 個集める
            for j in range(self.agents_num):
                if j != i:
                    other_agents_coords_list.append(all_agent_coords[:, j, :])

            # (B * (agents_num - 1), 2) に結合
            other_coords = torch.cat(other_agents_coords_list, dim=0)

            # x座標とy座標を抽出
            other_x = other_coords[:, 0] # (B * (agents_num - 1),)
            other_y = other_coords[:, 1] # (B * (agents_num - 1),)

            # 他のエージェントの数に対応したバッチインデックスを準備
            batch_indices_other = batch_indices_self.repeat_interleave(self.agents_num - 1)

            # state_map[バッチインデックス, チャネル2, 他のx座標, 他のy座標] = 1.0
            state_map[batch_indices_other, 2, other_x, other_y] = 1.0

        # グリッドの座標値チェック（デバッグ用: 座標がG-1を超えていないかなど）
        # assert (state_map.sum(dim=(0,1,2,3)) > 0), "state_map is all zero!"

        return state_map
