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

# class MixingNetwork(nn.Module):
#     """
#     QMIXで使用されるMixing Network.
#     各エージェントのQ値とグローバル状態を入力として受け取り、チーム全体のQ_totを出力する.
#     """
#     def __init__(self, n_agents: int, state_dim: int, hidden_dim: int = 32):
#         """
#         MixingNetwork クラスのコンストラクタ.

#         Args:
#             n_agents (int): エージェントの数.
#             state_dim (int): グローバル状態の次元.
#             hidden_dim (int): 隠れ層の次元 (HyperNetwork内部で使用).
#         """
#         super().__init__()
#         self.n_agents = n_agents
#         self.state_dim = state_dim
#         self.hidden_dim = hidden_dim

#         # HyperNetwork for weights W1 (state_dim -> n_agents * hidden_dim)
#         self.hyper_w1 = nn.Linear(state_dim, self.n_agents * hidden_dim)
#         # HyperNetwork for biases B1 (state_dim -> hidden_dim)
#         self.hyper_b1 = nn.Linear(state_dim, hidden_dim)

#         # HyperNetwork for weights W2 (state_dim -> hidden_dim * 1)
#         self.hyper_w2 = nn.Linear(state_dim, hidden_dim * 1)
#         # HyperNetwork for biases B2 (state_dim -> 1)
#         self.hyper_b2 = nn.Linear(state_dim, 1)

#         # Hidden layer for the mixing network itself (shared for all batches)
#         self.mix_hidden = nn.Linear(self.n_agents, hidden_dim)

#     def forward(self, agent_q_values: torch.Tensor, global_state: torch.Tensor) -> torch.Tensor:
#         """
#         順伝播処理.

#         Args:
#             agent_q_values (torch.Tensor): 各エージェントのQ値のテンソル (形状: (batch_size, n_agents, 1)).
#             global_state (torch.Tensor): グローバル状態のテンソル (形状: (batch_size, state_dim)).

#         Returns:
#             torch.Tensor: チーム全体のQ値 (形状: (batch_size, 1)).
#         """
#         batch_size = agent_q_values.size(0)

#         # hyper_w1とhyper_b1からW1とB1を生成
#         # W1: (batch_size, state_dim) -> (batch_size, n_agents * hidden_dim)
#         # W1を(batch_size, n_agents, hidden_dim)に再成形し、agent_q_valuesとの行列乗算を行う
#         w1 = self.hyper_w1(global_state)
#         w1 = w1.view(batch_size, self.n_agents, self.hidden_dim)

#         # 単調性制約を適用: 重みが非負であることを保証
#         w1 = torch.abs(w1) # torch.exp(w1)も使用可能

#         # B1: (batch_size, state_dim) -> (batch_size, hidden_dim)
#         b1 = self.hyper_b1(global_state)
#         b1 = b1.view(batch_size, 1, self.hidden_dim) # Reshape for broadcasting

#         # 混合ネットワークの最初の層の出力を計算する
#         # (batch_size, n_agents, 1) @ (batch_size, n_agents, hidden_dim) -> (batch_size, 1, hidden_dim)
#         hidden = F.elu(torch.bmm(agent_q_values.transpose(1,2), w1) + b1)

#         # hyper_w2 および hyper_b2 から W2 および B2 を生成
#         # W2: (batch_size, state_dim) -> (batch_size, hidden_dim * 1)
#         # Reshape W2 to (batch_size, hidden_dim, 1)
#         w2 = self.hyper_w2(global_state)
#         w2 = w2.view(batch_size, self.hidden_dim, 1)

#         # 単調性制約を適用
#         w2 = torch.abs(w2) # Or torch.exp(w2)

#         # B2: (batch_size, state_dim) -> (batch_size, 1)
#         b2 = self.hyper_b2(global_state)
#         b2 = b2.view(batch_size, 1, 1) # Reshape for broadcasting

#         # 第2層の出力を計算 (Q_tot)
#         # (batch_size, 1, hidden_dim) @ (batch_size, hidden_dim, 1) -> (batch_size, 1, 1)
#         Q_tot = torch.bmm(hidden, w2) + b2

#         return Q_tot.view(batch_size, 1) # Reshape to (batch_size, 1)

class MixingNetwork(nn.Module):
    """
    QMIX ミキシングネットワーク.
    各エージェントのQ値を入力とし、グローバル状態に基づいた動的な重みを用いて
    チーム全体のQ値を算出する。単調性制約(重み >= 0)を保証する。
    """
    def __init__(self, n_agents: int, state_dim: int, hidden_dim: int = 32):
        """
        Args:
            n_agents (int): エージェントの数.
            state_dim (int): グローバル状態の次元.
            hidden_dim (int): 内部的な隠れ層の次元.
        """
        super().__init__()
        self.n_agents = n_agents
        self.state_dim = state_dim
        self.hidden_dim = hidden_dim

        # --- HyperNetwork W1: state -> (n_agents x hidden_dim) ---
        # 2層にすることで、状態からの非線形な重み生成を可能にする
        self.hyper_w1 = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, n_agents * hidden_dim)
        )
        
        # HyperNetwork B1: state -> hidden_dim
        self.hyper_b1 = nn.Linear(state_dim, hidden_dim)

        # --- HyperNetwork W2: state -> (hidden_dim x 1) ---
        self.hyper_w2 = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim)  # 出力は(hidden_dim * 1)
        )
        
        # HyperNetwork B2: state -> 1
        # V(s)のような役割を果たし、状態に基づいたベースラインQ値を生成する
        self.hyper_b2 = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1)
        )

    def forward(self, agent_q_values: torch.Tensor, global_state: torch.Tensor) -> torch.Tensor:
        """
        Args:
            agent_q_values (torch.Tensor): (batch_size, n_agents)
            global_state (torch.Tensor): (batch_size, state_dim)
        Returns:
            torch.Tensor: (batch_size, 1)
        """
        batch_size = agent_q_values.size(0)
        
        # agent_q_values を (batch_size, 1, n_agents) に変形
        # 行列演算(bmm)のために1次元追加
        q_vals = agent_q_values.view(batch_size, 1, self.n_agents)

        # --- 第1層の重みとバイアス ---
        # W1生成: 非負制約のために torch.exp を使用
        w1 = torch.exp(self.hyper_w1(global_state))
        w1 = w1.view(batch_size, self.n_agents, self.hidden_dim)
        
        # B1生成
        b1 = self.hyper_b1(global_state)
        b1 = b1.view(batch_size, 1, self.hidden_dim)

        # 1層目計算: (1, n_agents) @ (n_agents, hidden_dim) + (1, hidden_dim)
        hidden = F.elu(torch.bmm(q_vals, w1) + b1)

        # --- 第2層の重みとバイアス ---
        # W2生成: 同様に torch.exp
        w2 = torch.exp(self.hyper_w2(global_state))
        w2 = w2.view(batch_size, self.hidden_dim, 1)
        
        # B2生成: 二層にすることで表現力を向上
        b2 = self.hyper_b2(global_state)
        b2 = b2.view(batch_size, 1, 1)

        # 2層目計算: (1, hidden_dim) @ (hidden_dim, 1) + (1, 1)
        q_tot = torch.bmm(hidden, w2) + b2

        return q_tot.view(batch_size, 1)
