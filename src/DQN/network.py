from abc import ABC, abstractmethod
import torch
import torch.nn as nn
import torch.nn.functional as F

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
        """
        AgentNetwork クラスのコンストラクタ.

        Args:
            input_size (int): Qネットワークへの入力サイズ (状態空間の次元).
            output_size (int): Qネットワークの出力サイズ (行動空間の次元).
            total_agents (int): 全エージェント数 (パラメータ共有のためのAgent IDワンホットエンコーディングに使用)
        """
        super().__init__(grid_size, output_size, total_agents)

        self._grid_size = grid_size
        self._num_channels = 3 # Concrete value
        self._total_agents = total_agents
        self._action_size = output_size

        # 状態入力の次元 (グリッドマップのフラット化されたサイズ)
        state_input_size = self.num_channels * self.grid_size**2

        # Agent ID埋め込み層は削除し、ワンホットエンコーディングを使用するため不要

        # 状態入力とエージェントワンホットベクトルを結合した後の最終的な入力サイズ
        # Agent IDはワンホットベクトルとして扱われるため、そのサイズは total_agents となる
        combined_input_size = state_input_size + self.total_agents

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

class MixingNetwork(AbstractMixer): # AbstractMixerを継承するように変更
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

class DICGMixer(AbstractMixer):
    """
    DICG (Difference Inidividual Contribution Global) で使用されるミキサー。
    グローバル状態からクエリを生成し、各エージェントのQ値からキーを生成するアテンションメカニズムを導入します。
    アテンションスコアを計算し、softmaxで正規化されたアテンション重みを各エージェントのQ値に適用することで、
    より動的で状況に応じたQ値の混合を実現します。
    """
    def __init__(self, n_agents: int, state_dim: int, hidden_dim: int = 32):
        """
        DICGMixer クラスのコンストラクタ。

        Args:
            n_agents (int): エージェントの数。
            state_dim (int): グローバル状態の次元。
            hidden_dim (int): アテンション機構の隠れ層の次元（クエリとキーの次元）。
        """
        super().__init__()
        self.n_agents = n_agents
        self.state_dim = state_dim
        self.hidden_dim = hidden_dim

        # グローバル状態からクエリを生成するネットワーク
        # クエリは各エージェントの寄与度を評価するためのグローバルな「質問」
        self.query_network = nn.Linear(state_dim, hidden_dim)

        # 各エージェントのQ値からキーを生成するネットワーク
        # キーは各エージェントのQ値が持つ「情報」
        # agent_q_valuesの最後の次元が1なので、入力次元は1
        self.key_network = nn.Linear(1, hidden_dim)

        # 全体のバイアスを生成するネットワーク (以前のhyper_bに相当)
        self.bias_network = nn.Linear(state_dim, 1)

    def forward(self, agent_q_values: torch.Tensor, global_state: torch.Tensor) -> torch.Tensor:
        """
        順伝播処理。

        Args:
            agent_q_values (torch.Tensor): 各エージェントが選択した行動のQ値 (形状: (batch_size, n_agents, 1)).
            global_state (torch.Tensor): グローバル状態のテンソル (形状: (batch_size, state_dim)).

        Returns:
            torch.Tensor: チーム全体のQ値 (形状: (batch_size, 1)).
        """
        batch_size = agent_q_values.size(0)

        # 1. グローバル状態からクエリを生成
        # (batch_size, state_dim) -> (batch_size, hidden_dim)
        query = F.relu(self.query_network(global_state))

        # 2. 各エージェントのQ値からキーを生成
        # agent_q_valuesの形状は (batch_size, n_agents, 1)
        # key_networkはnn.Linear(1, hidden_dim)なので、各エージェントのQ値(1次元)をhidden_dimに変換
        # (batch_size, n_agents, 1) -> (batch_size, n_agents, hidden_dim)
        keys = F.relu(self.key_network(agent_q_values))

        # 3. クエリとキーのドット積によりアテンションスコアを計算
        # query: (batch_size, hidden_dim) -> unsqueezeで (batch_size, hidden_dim, 1)
        # keys: (batch_size, n_agents, hidden_dim)
        # bmm((batch_size, n_agents, hidden_dim), (batch_size, hidden_dim, 1)) -> (batch_size, n_agents, 1)
        attention_scores = torch.bmm(keys, query.unsqueeze(-1))

        # 4. softmaxで正規化されたアテンション重みを計算
        # dim=1でエージェント軸に沿ってsoftmaxを適用
        # (batch_size, n_agents, 1)
        attention_weights = F.softmax(attention_scores, dim=1)

        # 5. 各エージェントのQ値にアテンション重みを適用し、合計
        # (batch_size, n_agents, 1) * (batch_size, n_agents, 1) -> (batch_size, n_agents, 1)
        # .sum(dim=1) -> (batch_size, 1)
        weighted_q_values_sum = (agent_q_values * attention_weights).sum(dim=1)

        # 6. グローバルなバイアスを加算
        # (batch_size, state_dim) -> (batch_size, 1)
        bias = self.bias_network(global_state)

        # 最終的なチームの総Q値
        Q_tot = weighted_q_values_sum + bias

        return Q_tot
