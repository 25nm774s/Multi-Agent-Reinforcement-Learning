from abc import ABC, abstractmethod
import torch
import torch.nn as nn
import torch.nn.functional as F

def state_to_map(
    global_state: torch.Tensor,
    grid_size: int,
    goals_num: int,
    agents_num: int
) -> torch.Tensor:
    """
    flat state を CNN用マップへ変換

    Input:
        global_state: (B, state_dim)

    Output:
        state_map: (B, 2, G, G)

    channel 0 : goals
    channel 1 : agents
    """

    B = global_state.size(0)
    device = global_state.device

    state_map = torch.zeros(
        B,
        2,
        grid_size,
        grid_size,
        device=device,
        dtype=torch.float32
    )

    # -------------------------
    # goals
    # -------------------------
    for i in range(goals_num):

        x = global_state[:, 2 * i].long()
        y = global_state[:, 2 * i + 1].long()

        valid = (
            (x >= 0)
            & (x < grid_size)
            & (y >= 0)
            & (y < grid_size)
        )

        batch_idx = torch.arange(B, device=device)

        state_map[
            batch_idx[valid],
            0,
            x[valid],
            y[valid]
        ] += 1.0

    # -------------------------
    # agents
    # -------------------------

    offset = goals_num * 2
    batch_idx = torch.arange(B, device=device)

    for i in range(agents_num):

        x = global_state[:, offset + 2 * i].long()
        y = global_state[:, offset + 2 * i + 1].long()

        valid = (
            (x >= 0)
            & (x < grid_size)
            & (y >= 0)
            & (y < grid_size)
        )

        state_map[
            batch_idx[valid],
            1,
            x[valid],
            y[valid]
        ] += 1.0

    return state_map

class StateEncoder(nn.Module):

    def __init__(
        self,
        state_feature_dim=32
    ):
        super().__init__()

        self.conv1 = nn.Conv2d(
            2,
            16,
            kernel_size=3,
            padding=1
        )

        self.conv2 = nn.Conv2d(
            16,
            16,
            kernel_size=3,
            padding=1
        )

        self.pool = nn.AdaptiveAvgPool2d(1)

        self.fc = nn.Linear(
            16,
            state_feature_dim
        )

    def forward(self, state_map):

        x = F.relu(self.conv1(state_map))
        x = F.relu(self.conv2(x))

        x = self.pool(x)

        x = x.flatten(1)

        return F.relu(self.fc(x))

class IAgentNetwork(nn.Module, ABC):
    """
    Abstract Base Class for AgentNetwork.
    Defines the interface for all agent Q-value networks.
    """
    def __init__(self, grid_size: int, output_size: int, total_agents: int = 1):
        super().__init__()
        # Abstract properties are defined below, concrete classes will implement them.
        pass

    @property
    @abstractmethod
    def grid_size(self) -> int:
        """Returns the grid size the network is configured for."""
        pass

    @property
    @abstractmethod
    def num_channels(self) -> int:
        """Returns the number of input channels for the network."""
        pass

    @property
    @abstractmethod
    def total_agents(self) -> int:
        """Returns the total number of agents the network is configured for."""
        pass

    @property
    @abstractmethod
    def action_size(self) -> int:
        """Returns the size of the action space the network outputs for."""
        pass

    @abstractmethod
    def forward(self, x: torch.Tensor, agent_ids: torch.Tensor) -> torch.Tensor:
        """
        Computes Q-values for given observations and agent IDs.
        Args:
            x (torch.Tensor): Transformed observations (B*N, C, G, G).
            agent_ids (torch.Tensor): Agent IDs (B*N,).
        Returns:
            torch.Tensor: Q-values for each action (B*N, action_size).
        """
        pass

class AgentNetwork(IAgentNetwork):
    """
    DQNで使用されるQネットワークモデル.
    状態を入力として受け取り、各行動に対するQ値を出力する.
    """
    def __init__(self, grid_size: int, output_size: int, total_agents: int = 1):
        super().__init__(grid_size, output_size, total_agents)

        self._grid_size = grid_size
        # num_channelsは環境から取得するか、確定した値を使用
        self._num_channels = 3 
        self._total_agents = total_agents
        self._action_size = output_size

        # --- CNN層の定義 (プーリングなし) ---
        # Conv層1: in_channelsは環境の観測チャネル数, out_channels=32, kernel=3x3, stride=1, padding='same'
        self.conv1 = nn.Conv2d(in_channels=self.num_channels, out_channels=32, kernel_size=3, stride=1, padding=1)
        # Conv層2: in_channelsは前層のout_channels, out_channels=64, kernel=3x3, stride=1, padding='same'
        self.conv2 = nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3, stride=1, padding=1)

        # CNN出力のサイズ計算
        # ダミーデータを使ってCNN層の出力サイズを計算するのが最も確実です。
        with torch.no_grad():
            dummy_input = torch.zeros(1, self.num_channels, self.grid_size, self.grid_size)
            x = F.relu(self.conv1(dummy_input))
            x = F.relu(self.conv2(x))
            flattened_cnn_output_size = x.view(1, -1).size(1)
            
        # --- 全結合層の定義 ---
        # 結合入力サイズ = フラット化されたCNN出力サイズ + ワンホットエンコードされたエージェントIDのサイズ
        combined_input_size = flattened_cnn_output_size + self.total_agents

        self.fc1 = nn.Linear(combined_input_size, 128)
        self.fc2 = nn.Linear(128, 128)
        self.fc3 = nn.Linear(128, output_size)

    @property
    def grid_size(self) -> int:
        return self._grid_size

    @property
    def num_channels(self) -> int:
        return self._num_channels

    @property
    def total_agents(self) -> int:
        return self._total_agents

    @property
    def action_size(self) -> int:
        return self._action_size

    def forward(self, x: torch.Tensor, agent_ids: torch.Tensor) -> torch.Tensor:
        batch_size = x.size(0)

        # --- CNNによる特徴抽出 ---
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        
        # CNNの出力をフラット化
        x = x.view(batch_size, -1) 

        # Agent IDをワンホットエンコーディング
        agent_one_hot = F.one_hot(agent_ids, num_classes=self.total_agents).float()

        # CNN特徴とワンホットエンコーディングされたAgent IDを結合
        combined_input = torch.cat([x, agent_one_hot], dim=1)

        # 全結合層の順伝播
        x = F.relu(self.fc1(combined_input))
        x = F.relu(self.fc2(x))
        q_values = self.fc3(x)

        return q_values

class AbstractMixer(nn.Module, ABC):
    """
    ミキサーネットワークの抽象基底クラス.
    チーム全体のQ値を計算するインターフェースを定義します。
    """
    @abstractmethod
    def forward(self, agent_q_values: torch.Tensor, global_state: torch.Tensor) -> torch.Tensor:
        """
        チーム全体のQ値を計算します。
        Args:
            agent_q_values (torch.Tensor): 各エージェントが選択した行動のQ値 (形状: (batch_size, n_agents, 1)).
            global_state (torch.Tensor): グローバル状態のテンソル (形状: (batch_size, state_dim)).

        Returns:
            torch.Tensor: チーム全体のQ値 (形状: (batch_size, 1)).
        """
        pass

class SumMixer(AbstractMixer):
    """
    Value Decomposition Networks (VDN) で使用されるミキサー.
    各エージェントのQ値を単純に合計します。
    """
    def __init__(self):
        super().__init__()

    def forward(self, agent_q_values: torch.Tensor, global_state: torch.Tensor) -> torch.Tensor:
        """
        各エージェントのQ値を合計してチーム全体のQ値を計算します。
        global_stateは引数として受け取りますが、ここでは使用しません。

        Args:
            agent_q_values (torch.Tensor): 各エージェントが選択した行動のQ値 (形状: (batch_size, n_agents, 1)).
            global_state (torch.Tensor): グローバル状態のテンソル (ここでは使用されない). (形状: (batch_size, state_dim)).

        Returns:
            torch.Tensor: チーム全体のQ値 (形状: (batch_size, 1)).
        """
        Q_tot = agent_q_values.sum(dim=1) # (batch_size, 1)
        return Q_tot


class MixingNetwork(AbstractMixer):
    """
    QMIX Mixing Network (corrected version)

    - monotonic mixing enforced via softplus weights
    - proper per-agent contribution mixing
    - stable tensor shapes
    """

    def __init__(
        self,
        n_agents: int,
        state_dim: int,
        grid_size: int,
        n_goals: int,
        hidden_dim: int = 32
    ):
        super().__init__()

        self.n_agents = n_agents
        self.state_dim = state_dim
        self.hidden_dim = hidden_dim
        self.grid_size = grid_size
        self.n_goals = n_goals

        # =========================================================
        # State encoder (CNN)
        # =========================================================
        self.state_encoder = nn.Sequential(
            nn.Conv2d(2, 16, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(16, 16, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.AdaptiveAvgPool2d(1),
        )

        self.state_fc = nn.Linear(16, hidden_dim)

        # =========================================================
        # Hyper networks
        # =========================================================
        # first layer weights: (hidden_dim * n_agents)
        self.hyper_w1 = nn.Linear(hidden_dim, n_agents * hidden_dim)
        self.hyper_b1 = nn.Linear(hidden_dim, hidden_dim)

        # second layer weights
        self.hyper_w2 = nn.Linear(hidden_dim, hidden_dim)
        self.hyper_b2 = nn.Linear(hidden_dim, 1)

    # =============================================================
    # forward
    # =============================================================
    def forward(self, agent_q_values: torch.Tensor, global_state: torch.Tensor):
        """
        Args:
            agent_q_values: (B, n_agents, 1)
            global_state:   (B, state_dim)

        Returns:
            Q_tot: (B, 1)
        """

        B = agent_q_values.size(0)

        # ---------------------------------------------------------
        # 1. state → map → embedding
        # ---------------------------------------------------------
        state_map = state_to_map(
            global_state,
            self.grid_size,
            self.n_goals,
            self.n_agents
        )  # (B, 2, G, G)

        x = self.state_encoder(state_map)     # (B, 16, 1, 1)
        x = x.flatten(1)                      # (B, 16)
        state_emb = F.relu(self.state_fc(x))  # (B, H)

        # ---------------------------------------------------------
        # 2. hypernetwork weights
        # ---------------------------------------------------------
        w1 = torch.abs(self.hyper_w1(state_emb))  # monotonic constraint
        w1 = w1.view(B, self.n_agents, self.hidden_dim)

        b1 = self.hyper_b1(state_emb).view(B, 1, self.hidden_dim)

        # ---------------------------------------------------------
        # 3. first mixing layer (IMPORTANT FIX)
        # ---------------------------------------------------------
        # element-wise mixing per agent (NO bmm compression)
        # (B, N, 1) * (B, N, H) → (B, N, H)
        hidden = F.elu(agent_q_values * w1 + b1)

        # sum over agents → (B, H)
        hidden = hidden.sum(dim=1)

        # ---------------------------------------------------------
        # 4. second layer hypernetwork
        # ---------------------------------------------------------
        w2 = torch.abs(self.hyper_w2(state_emb)).view(B, self.hidden_dim, 1)
        b2 = self.hyper_b2(state_emb).view(B, 1, 1)

        # ---------------------------------------------------------
        # 5. final Q_tot
        # ---------------------------------------------------------
        hidden = hidden.unsqueeze(1)  # (B, 1, H)

        Q_tot = torch.bmm(hidden, w2) + b2  # (B, 1, 1)

        return Q_tot.view(B, 1)
    
class DICGMixer(AbstractMixer):
    """
    DICG (Attention-based QMIX-style Mixer)

    - global state → CNN encoder → query
    - agent Q (or features) → key/value
    - attention-based mixing
    """

    def __init__(
        self,
        n_agents: int,
        grid_size: int,
        n_goals: int,
        state_feature_dim: int = 32,
        hidden_dim: int = 32,
        agent_feat_dim: int = 1
    ):
        super().__init__()

        self.n_agents = n_agents
        self.grid_size = grid_size
        self.n_goals = n_goals
        self.hidden_dim = hidden_dim

        # -------------------------
        # State encoder (CNN)
        # -------------------------
        self.state_encoder = nn.Sequential(
            nn.Conv2d(2, 16, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(16, 16, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.AdaptiveAvgPool2d(1),
        )

        self.state_fc = nn.Linear(16, state_feature_dim)

        # -------------------------
        # Query (global context)
        # -------------------------
        self.query_net = nn.Linear(state_feature_dim, hidden_dim)

        # -------------------------
        # Key (agent-dependent)
        # ※ここを改善：Q値だけでなく拡張可能
        # -------------------------
        self.key_net = nn.Linear(agent_feat_dim, hidden_dim)

        # -------------------------
        # Value projection (optional but stabilizes mixing)
        # -------------------------
        self.value_net = nn.Linear(1, hidden_dim)

        # -------------------------
        # Bias from global state
        # -------------------------
        self.bias_net = nn.Linear(state_feature_dim, 1)

    def forward(self, agent_q_values: torch.Tensor, global_state: torch.Tensor):
        """
        Args:
            agent_q_values: (B, n_agents, 1)
            global_state:   (B, state_dim)

        Returns:
            Q_tot: (B, 1)
        """

        B = agent_q_values.size(0)

        # =========================================================
        # 1. global state → map → CNN
        # =========================================================
        state_map = state_to_map(
            global_state,
            self.grid_size,
            self.n_goals,
            self.n_agents
        )  # (B, 2, G, G)

        x = self.state_encoder(state_map)      # (B, 16, 1, 1)
        x = x.view(B, 16)                      # flatten
        state_feat = F.relu(self.state_fc(x)) # (B, state_feature_dim)

        # =========================================================
        # 2. query / bias
        # =========================================================
        query = self.query_net(state_feat)    # (B, H)
        bias = self.bias_net(state_feat)      # (B, 1)

        # =========================================================
        # 3. agent embeddings
        # =========================================================

        # key: (B, N, H)
        keys = self.key_net(agent_q_values)

        # value: (B, N, H)
        values = self.value_net(agent_q_values)

        # =========================================================
        # 4. attention scores
        # =========================================================

        # (B, N, H) @ (B, H, 1) → (B, N, 1)
        scores = torch.bmm(keys, query.unsqueeze(-1))

        # scale
        scores = scores / (self.hidden_dim ** 0.5)

        # softmax over agents
        attn = F.softmax(scores, dim=1)  # (B, N, 1)

        # =========================================================
        # 5. weighted aggregation
        # =========================================================

        # value-based mixing (more stable than raw Q)
        mixed = (attn * values).sum(dim=1)  # (B, H)

        # project to scalar Q
        q_tot = mixed.mean(dim=-1, keepdim=True)  # (B, 1)

        # =========================================================
        # 6. bias
        # =========================================================
        q_tot = q_tot + bias

        return q_tot