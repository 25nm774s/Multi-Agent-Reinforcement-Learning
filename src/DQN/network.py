from abc import ABC, abstractmethod
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional

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
    def total_agents(self) -> int:
        """Returns the total number of agents the network is configured for."""
        pass

    @property
    @abstractmethod
    def action_size(self) -> int:
        """Returns the size of the action space the network outputs for."""
        pass
    
    @property
    @abstractmethod
    def obs_channels(self) -> int:
        """
        Returns the raw observation channels before preprocessing.
        """
        pass

    @property
    @abstractmethod
    def input_channels(self) -> int:
        """
        Returns the channels actually fed into the neural network layers.
        """
        pass
    
    @abstractmethod
    def forward(self, x: torch.Tensor, agent_ids: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Computes Q-values for given observations and agent IDs.
        Args:
            x (torch.Tensor):
                Raw observations before agent-specific preprocessing.
                Shape: (B*N, obs_channels, G, G)
            agent_ids (torch.Tensor): 
                Agent IDs (B*N,).
        Returns:
            torch.Tensor: Q-values for each action (B*N, action_size).
        """
        pass

class CNNAgentNetworkBase(IAgentNetwork):
    """
    DQNで使用されるQネットワークモデル.
    状態を入力として受け取り、各行動に対するQ値を出力する.
    """
    def __init__(self, grid_size: int, output_size: int, total_agents: int = 1, pool_layer:nn.Module | None=None):
        """
        AgentNetwork クラスのコンストラクタ.

        Args:
            input_size (int): Qネットワークへの入力サイズ (状態空間の次元).
            output_size (int): Qネットワークの出力サイズ (行動空間の次元).
            total_agents (int): 全エージェント数 (パラメータ共有のためのAgent IDワンホットエンコーディングに使用)
        """
        super().__init__(grid_size, output_size, total_agents)

        self._grid_size = grid_size
        # num_channelsは環境から取得するか、確定した値を使用
        self._obs_channels = 3

        self._total_agents = total_agents
        self._action_size = output_size

        # --- CNN層の定義 ---
        # Conv層1: in_channelsは環境の観測チャネル数, out_channels=32, kernel=3x3, stride=1, padding='same'
        self.conv1 = nn.Conv2d(in_channels=self.input_channels, out_channels=32, kernel_size=3, stride=1, padding=1)
        # Conv層2: in_channelsは前層のout_channels, out_channels=64, kernel=3x3, stride=1, padding='same'
        self.conv2 = nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3, stride=1, padding=1)

        self.pool = pool_layer
        # CNN出力のサイズ計算
        with torch.no_grad():
            dummy_input = torch.zeros(
                1,
                self._obs_channels,
                self.grid_size,
                self.grid_size
            )

            x = self._append_agent_id_channel(
                dummy_input,
                torch.zeros(1, dtype=torch.long)
            )

            x = F.relu(self.conv1(x))
            x = F.relu(self.conv2(x))

            if self.pool is not None:
                x = self.pool(x)

            flattened_cnn_output_size = torch.flatten(x, 1).size(1)
            
        # --- 全結合層の定義 ---
        # Agent IDはCNN入力チャネルとして付与済み
        combined_input_size = flattened_cnn_output_size

        self.fc1 = nn.Linear(combined_input_size, 128)
        self.fc2 = nn.Linear(128, 128)
        self.fc3 = nn.Linear(128, output_size)

    def _append_agent_id_channel(
        self,
        x: torch.Tensor,
        agent_ids: torch.Tensor
    )->torch.Tensor:
        """
        x:
            (batch, 3, H, W)

        agent_ids:
            (batch,)
        """

        batch_size, _, h, w = x.shape

        # 正規化したAgent ID
        agent_id_value = (
            agent_ids.float()
            / max(self.total_agents - 1, 1)
        )

        # (batch,1) -> (batch,1,H,W)
        agent_channel = (
            agent_id_value
            .view(batch_size, 1, 1, 1)
            .expand(batch_size, 1, h, w)
        )

        x = torch.cat(
            [x, agent_channel],
            dim=1
        )

        return x

    @property
    def grid_size(self) -> int:
        return self._grid_size

    @property
    def input_channels(self) -> int:
        return self._obs_channels + self.agent_id_channels

    @property
    def agent_id_channels(self):
        return 1
    
    @property
    def total_agents(self) -> int:
        return self._total_agents

    @property
    def hidden_dim(self) -> int:
        return 128
    
    @property
    def action_size(self) -> int:
        return self._action_size

    @property
    def obs_channels(self) -> int:
        return self._obs_channels

    @property
    def pooling_name(self):
        if self.pool is None:
            return "None"
        return self.pool.__class__.__name__

    def forward(self, x: torch.Tensor, agent_ids: torch.Tensor) -> tuple[torch.Tensor,torch.Tensor]:

        x = self._append_agent_id_channel(
            x,
            agent_ids
        )
        # --- CNNによる特徴抽出 ---
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))

        if self.pool is not None:
            x = self.pool(x)

        # x = x.view(batch_size, -1)
        x = torch.flatten(x, start_dim=1)

        combined_input = x

        # 全結合層の順伝播
        x = F.relu(self.fc1(combined_input))

        hidden_features = F.relu(
            self.fc2(x)
        )

        q_values = self.fc3(
            hidden_features
        )

        return q_values, hidden_features

class AgentNetwork(CNNAgentNetworkBase):
    def __init__(self, grid_size: int, output_size: int, total_agents: int = 1):
        super().__init__(grid_size, output_size, total_agents, pool_layer=None)

class AgentNetworkMaxPool(CNNAgentNetworkBase):
    def __init__(self, grid_size: int, output_size: int, total_agents: int = 1):
        super().__init__(grid_size, output_size, total_agents, pool_layer=nn.MaxPool2d(2))

class AgentNetworkAdaptivePool(CNNAgentNetworkBase):
    def __init__(self, grid_size: int, output_size: int, total_agents: int = 1):
        super().__init__(grid_size, output_size, total_agents, pool_layer=nn.AdaptiveAvgPool2d((3,3)))

class CNNAgentIdChannelNetworkBase(CNNAgentNetworkBase):

    @property
    def agent_id_channels(self):
        return self.total_agents

    def _append_agent_id_channel(
        self,
        x,
        agent_ids
    ):
        batch_size, _, h, w = x.shape

        one_hot = F.one_hot(
            agent_ids,
            num_classes=self.total_agents
        ).float()

        agent_channels = (
            one_hot
            .view(
                batch_size,
                self.total_agents,
                1,
                1
            )
            .expand(
                batch_size,
                self.total_agents,
                h,
                w
            )
        )

        return torch.cat(
            [x, agent_channels],
            dim=1
        )

class AbstractMixer(nn.Module, ABC):
    """
    ミキサーネットワークの抽象基底クラス.
    チーム全体のQ値を計算するインターフェースを定義します。
    """
    @abstractmethod
    def forward(
        self,
        agent_q_values: torch.Tensor,
        agent_hidden: Optional[torch.Tensor] = None,
        global_state: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
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

    def forward(
        self,
        agent_q_values,
        agent_hidden=None,
        global_state=None
    ):
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

class MixingNetwork(AbstractMixer): # AbstractMixerを継承するように変更
    """
    QMIXで使用されるMixing Network.
    各エージェントのQ値とグローバル状態を入力として受け取り、チーム全体のQ_totを出力する.
    """
    def __init__(self, n_agents: int, n_goals: int, grid_size: int, state_dim: int, hidden_dim: int = 32):
        """
        MixingNetwork クラスのコンストラクタ.

        Args:
            n_agents (int): エージェントの数.
            state_dim (int): グローバル状態の次元.
            hidden_dim (int): 隠れ層の次元 (HyperNetwork内部で使用).
        """
        super().__init__()
        self.n_agents = n_agents
        self.n_goals = n_goals      # まだ不使用
        self.grid_size = grid_size  # まだ不使用
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
        # Removed as it's not part of the standard QMIX Mixing Network, and its purpose is unclear.
        # self.mix_hidden = nn.Linear(self.n_agents, hidden_dim)

    def forward(self, agent_q_values: torch.Tensor, global_state: torch.Tensor) -> torch.Tensor:
        """
        順伝播処理.

        Args:
            agent_q_values (torch.Tensor): 各エージェントが選択した行動のQ値 (形状: (batch_size, n_agents, 1)).
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
        # w1 = torch.exp(w1)
        w1 = F.softplus(w1)

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
        # w2 = torch.exp(w2)
        w2 = F.softplus(w2)

        # B2: (batch_size, state_dim) -> (batch_size, 1)
        b2 = self.hyper_b2(global_state)
        b2 = b2.view(batch_size, 1, 1) # Reshape for broadcasting

        # 第2層の出力を計算 (Q_tot)
        # (batch_size, 1, hidden_dim) @ (batch_size, hidden_dim, 1) -> (batch_size, 1, 1)
        Q_tot = torch.bmm(hidden, w2) + b2

        return Q_tot.view(batch_size, 1) # Reshape to (batch_size, 1)

class DICGMixer(AbstractMixer):
    """
    DICG (Difference Inidividual Contribution Global) で使用されるミキサー。
    グローバル状態からクエリを生成し、各エージェントのQ値からキーを生成するアテンションメカニズムを導入します。
    アテンションスコアを計算し、softmaxで正規化されたアテンション重みを各エージェントのQ値に適用することで、
    より動的で状況に応じたQ値の混合を実現します。
    """
    def __init__(
            self, 
            n_agents: int, 
            n_goals: int, 
            grid_size: int, 
            state_dim: int, 
            agent_hidden_dim=128, 
            hidden_dim: int = 32, 
            num_heads=4
        ):
        """
        DICGMixer クラスのコンストラクタ。

        Args:
            n_agents (int): エージェントの数。
            state_dim (int): グローバル状態の次元。
            hidden_dim (int): アテンション機構の隠れ層の次元（クエリとキーの次元）。
        """
        super().__init__()
        self.n_goals = n_goals      # まだ不使用
        self.grid_size = grid_size  # まだ不使用
        self.n_agents = n_agents
        self.state_dim = state_dim
        self.hidden_dim = hidden_dim

        self.num_heads = num_heads
        self.head_dim = hidden_dim // num_heads
        assert hidden_dim % num_heads == 0

        # グローバル状態からクエリを生成するネットワーク
        # クエリは各エージェントの寄与度を評価するためのグローバルな「質問」
        self.query_network = nn.Linear(state_dim, hidden_dim)

        # 各エージェントの特徴表現(hidden state)からキーを生成
        # キーは各エージェントのQ値が持つ「情報」
        # agent_q_valuesの最後の次元が1なので、入力次元は1
        self.key_network = nn.Sequential(
            nn.Linear(
                agent_hidden_dim,
                hidden_dim
            ),
            nn.ReLU(),
            nn.Linear(
                hidden_dim,
                hidden_dim
            )
        )
        self.head_projection = nn.Sequential(
            nn.Linear(
                self.num_heads,
                hidden_dim
            ),
            nn.ReLU(),
            nn.Linear(
                hidden_dim,
                1
            )
        )

        # 全体のバイアスを生成するネットワーク (以前のhyper_bに相当)
        self.bias_network = nn.Linear(state_dim, 1)

    def forward(
        self,
        agent_q_values: torch.Tensor,
        agent_hidden  : torch.Tensor,
        global_state  : torch.Tensor
    ):
        """
        順伝播処理。

        Args:
            agent_q_values (torch.Tensor): 各エージェントが選択した行動のQ値 (形状: (batch_size, n_agents, 1)).
            global_state (torch.Tensor): グローバル状態のテンソル (形状: (batch_size, state_dim)).

        Returns:
            torch.Tensor: チーム全体のQ値 (形状: (batch_size, 1)).
        """
        batch_size = agent_q_values.size(0)

        # query
        query = self.query_network(global_state)

        query = query.view(
            batch_size,
            self.num_heads,
            self.head_dim
        )

        query = query.unsqueeze(1)
        # (B,1,H,D)


        # key
        keys = self.key_network(agent_hidden)

        keys = keys.view(
            batch_size,
            self.n_agents,
            self.num_heads,
            self.head_dim
        )
        # (B,N,H,D)


        # attention score
        attention_scores = (
            keys * query
        ).sum(dim=-1)
        # (B,N,H)


        attention_weights = F.softmax(
            attention_scores,
            dim=1
        )
        # (B,N,H)


        attention_weights = attention_weights.unsqueeze(-1)
        # (B,N,H,1)


        q = agent_q_values.unsqueeze(2)
        # (B,N,1,1)


        weighted_q = q * attention_weights
        # (B,N,H,1)


        weighted_q = weighted_q.sum(dim=1)
        # (B,H,1)


        weighted_q = weighted_q.squeeze(-1)
        # (B,H)


        mixed_q = self.head_projection(weighted_q)
        # (B,1)


        bias = self.bias_network(global_state)

        Q_tot = mixed_q + bias

        return Q_tot
