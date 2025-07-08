import torch
import torch.optim as optim
import torch.nn as nn
import torch.nn.functional as F

class QNet(nn.Module):
    #def __init__(self, mask, agents_num, goals_num, action_size):
    def __init__(self, input_size, output_size):
        super().__init__()
        
        #if mask:
        #    input_size = 2
        #else:
        #    input_size = agents_num*2 + goals_num*2

        self.fc1 = nn.Linear(input_size, 128)
        self.fc2 = nn.Linear(128, 128)
        self.fc3 = nn.Linear(128, output_size)
    
    # 順伝播
    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x
        
class DQNModel: # 仮のクラス名

    def __init__(self, optimizer_type, gamma, batch_size, agent_num,
                 goals_num, load_model, learning_rate,mask,target_update_frequency=100):

        #optimizer_type = optimizer_type # optimizer と変数名が衝突しないように変更
        self.gamma = gamma
        self.batch_size = batch_size
        self.agents_num = agent_num
        self.goals_num = goals_num
        self.load_model = load_model
        self.mask = mask
        self.lr = learning_rate
        self.action_size = 5
        self.target_update_frequency = target_update_frequency
        #self.qnet = qnet # 依存性注入でもよい
        #self.qnet_target = qnet_target
        #self.qnet_target = QNet(mask,self.agents_num,self.goals_num,self.action_size)

        # maskがTrueの場合、入力サイズは次元数 * (エージェント数 + ゴール数)
        # maskがFalseの場合、入力サイズは次元数のみ
        input_size = (self.agents_num + self.goals_num) * 2 if self.mask else 2

        self.qnet_target = QNet(input_size, self.action_size)
        self.qnet = QNet(input_size, self.action_size)

        # オプティマイザの初期化
        if optimizer_type == 'Adam': # 変更した変数名を使用
            self.optimizer = optim.Adam(self.qnet.parameters(), lr=self.lr)
        elif optimizer_type == 'RMSProp':
            self.optimizer = optim.RMSprop(self.qnet.parameters(), lr=self.lr)
        else:
            print(f"Warning: Optimizer type '{optimizer_type}' not recognized. Using Adam as default.")
            self.optimizer = optim.Adam(self.qnet.parameters(), lr=self.lr)


    def huber_loss(self, q, target):
        err = target - q
        abs_err = torch.abs(err)

        # 標準偏差を用いた動的なHUBER_LOSS_DELTAの設定
        # バッチサイズが1より大きい場合のみ標準偏差を計算
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
        cond = abs_err < huber_loss_delta # Use abs_err for the condition
        L2 = 0.5 * torch.square(err)
        L1 = huber_loss_delta * (abs_err - 0.5 * huber_loss_delta) # Use abs_err here as well
        loss = torch.where(cond, L2, L1)

        return torch.mean(loss)

    def _extract_agent_state(self, i, global_states_batch, next_global_states_batch): # 引数名を変更
        """
        複数エージェントの全体の状態バッチから、特定エージェントの状態バッチを抽出する。
        """
        if not self.mask:
            # mask が False の場合は、全体の状態バッチをそのまま返す
            return global_states_batch, next_global_states_batch

        # 複数エージェントの全体の状態バッチから、現在のエージェントの状態バッチを抽出
        # global_states_batch の形状は (batch_size, 全体の状態次元) を想定
        # 例: (batch_size, (goals_num + agents_num) * 2)
        # 抽出したいのは各バッチサンプルの、特定エージェントの状態に対応する部分
        # idx_offset と agent_idx_start は、バッチ内の各サンプルにおけるエージェントの状態の開始インデックス
        idx_offset = self.goals_num * 2 # ゴール座標の合計次元数
        agent_idx_start = i * 2 + idx_offset # 現在のエージェントのX座標の開始インデックス
        idx_lst = [agent_idx_start, agent_idx_start + 1] # (X, Y)座標のインデックスリスト

        # バッチ内のすべてのサンプルについて、指定された特徴量のみを抽出
        # .clone() は、元のテンソルから独立したコピーを作成し、元のテンソルへの意図しない変更を防ぐ
        # global_states_batch[:, idx_lst] は形状 (batch_size, len(idx_lst))
        agent_states_batch = global_states_batch[:, idx_lst].clone()
        next_agent_states_batch = next_global_states_batch[:, idx_lst].clone()

        return agent_states_batch, next_agent_states_batch # 抽出したエージェントの状態バッチを返す


    def _calculate_q_values(self, agent_states_batch, action_batch): # 引数名を変更
        """
        現在のエージェントの状態バッチと行動バッチに対応するQ値をメインQネットで計算する。
        """
        # agent_states_batch は抽出されたエージェントの状態バッチ (batch_size, agent_state_dim)
        predicted_qs_batch = self.qnet(agent_states_batch) # 形状: (batch_size, action_size)

        # 各バッチサンプルで取られた行動に対応するQ値を取得
        # action_batch は形状 (batch_size,)
        # torch.arange(self.batch_size) はバッチ内のインデックス [0, 1, ..., batch_size-1]
        batch_indices = torch.arange(self.batch_size, device=agent_states_batch.device)
        q_values_taken_action = predicted_qs_batch[batch_indices, action_batch] # 形状: (batch_size,)

        return q_values_taken_action

    def _calculate_next_max_q_values(self, next_agent_states_batch): # 引数名を変更
        """
        次のエージェントの状態バッチにおけるターゲットQネットの最大Q値を計算する。
        """
        # next_agent_states_batch は抽出された次のエージェントの状態バッチ (batch_size, agent_state_dim)
        next_predicted_qs_target = self.qnet_target(next_agent_states_batch) # 形状: (batch_size, action_size)

        # 各バッチサンプルで、次の状態での最大Q値を取得
        next_max_q_values_target = next_predicted_qs_target.max(1)[0].detach() # .max(1) は各行（サンプル）ごとに最大値とインデックスを返す
        # [0] で最大値のテンソルを取得。detach() は勾配計算から切り離す
        return next_max_q_values_target

    def _calculate_target_q_values(self, reward_batch, done_batch, next_max_q_values): # 引数名を変更
        """
        報酬バッチ、完了フラグバッチ、および次の状態の最大Q値バッチに基づいてターゲットQ値を計算する。
        """
        # reward_batch, done_batch は形状 (batch_size,)
        # next_max_q_values は形状 (batch_size,)
        # ベルマン方程式: Target Q = Reward + Gamma * MaxQ(next_state) * (1 - done)
        target_q_values = reward_batch + (1 - done_batch.float()) * self.gamma * next_max_q_values
        return target_q_values

    def _optimize_network(self, loss:torch.Tensor):
        """
        バックプロパゲーションとパラメータの更新を行う。
        """
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

    def sync_qnet(self):
        """
        ターゲットネットワークをメインネットワークと同期するメソッド。
        """
        self.qnet_target.load_state_dict(self.qnet.state_dict())

    def _perform_standard_dqn_update(self, agent_states_batch, action_batch, reward_batch, next_agent_states_batch, done_batch, episode_num): # 引数名を変更
        """
        通常のDQN学習ロジックを実行する。
        (入力は特定エージェントの状態バッチ)
        """
        # Q値の計算 (特定エージェントの状態バッチを使用)
        predicted_q = self._calculate_q_values(agent_states_batch, action_batch)
        next_max_q = self._calculate_next_max_q_values(next_agent_states_batch)

        # ターゲットQ値の計算
        target_q = self._calculate_target_q_values(reward_batch, done_batch, next_max_q)

        # 損失の計算と最適化
        loss = self.huber_loss(predicted_q, target_q)
        self._optimize_network(loss)

        # スカラー損失の計算
        scalar_loss = (target_q - predicted_q).mean().item()

        # ターゲットネットワークの同期
        # episode_num が target_update_frequency の倍数の場合に同期
        if episode_num > 0 and episode_num % self.target_update_frequency == 0: # episode_num > 0 を追加して最初のエピソードでは同期しないようにするなど
             self.sync_qnet()
             # print(f"Episode {episode_num}: Target network synced.") # デバッグ用

        return scalar_loss

    def _perform_knowledge_distillation_update(self, agent_states_batch, action_batch): # 引数名を変更
        """
        学習済みモデルを真の価値関数として改めて学習する特殊な学習ロジックを実行する。
        （知識蒸留や模倣学習に相当）
        (入力は特定エージェントの状態バッチ)
        """
        # メインネットワークの予測Q値 (特定エージェントの状態バッチを使用)
        predicted_q = self.qnet(agent_states_batch) # 形状: (batch_size, action_size)

        # ターゲットネットワークを「真の価値関数」として利用
        # 現在のエージェントの状態バッチ `agent_states_batch` をターゲットQネットに入力し、実際に取られた行動のQ値を取得
        # agent_states_batch は形状 (batch_size, agent_state_dim)
        true_qs_batch = self.qnet_target(agent_states_batch) # 形状: (batch_size, action_size)
        batch_indices = torch.arange(self.batch_size, device=agent_states_batch.device)
        true_q = true_qs_batch[batch_indices, action_batch].detach() # detach() はここで

        # 損失の計算と最適化
        loss = self.huber_loss(predicted_q, true_q)
        self._optimize_network(loss)

        # スカラー損失の計算
        scalar_loss = (true_q - predicted_q).mean().item()

        # 知識蒸留モードでは通常ターゲットネットワークの同期は行わない？
        # if episode_num % self.target_update_frequency == 0:
        #    self.sync_qnet() # 知識蒸留の目的に応じて同期するか検討が必要

        return scalar_loss


    def update(self, i, global_states_batch, action_batch, reward_batch, next_global_states_batch, done_batch, episode_num)->float: # 引数名を変更
        """
        Qネットワークのメインの更新ロジック。学習モードによって処理を分岐する。
        (入力は全体の状態のバッチ)
        """
        # 1. データの準備とフィルタリング (全体の状態バッチから特定エージェントの状態バッチを抽出)
        agent_states_batch, next_agent_states_batch = self._extract_agent_state(i, global_states_batch, next_global_states_batch)

        # 2. 学習モードの分岐と実行 (特定エージェントの状態バッチを使用)
        if self.load_model == 0 or self.load_model == 1:
            # 通常のDQN学習モード
            return self._perform_standard_dqn_update(agent_states_batch, action_batch, reward_batch, next_agent_states_batch, done_batch, episode_num)
        else:
            # 特殊な学習モード (知識蒸留/模倣学習など)
            # このモードの場合、reward_batch, done_batch は使用しない設計になっている可能性
            return self._perform_knowledge_distillation_update(agent_states_batch, action_batch)