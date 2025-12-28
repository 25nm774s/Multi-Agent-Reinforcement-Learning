from abc import ABC, abstractmethod
import torch
import torch.nn as nn # オプティマイザーを設定するため。
from typing import List, Optional, Tuple

from utils.StateProcesser import StateProcessor
from DQN.dqn import AgentNetwork
from .Constant import GlobalState

class BaseMasterAgent(ABC):
    """
    抽象基底クラス BaseMasterAgent.
    状態処理、Q値計算、生存マスク適用などの共通ロジックをカプセル化し、
    IQLMasterAgent および QMIXMasterAgent の基盤を提供します。
    """
    def __init__(self,
                 n_agents: int,
                 action_size: int,
                 grid_size: int,
                 goals_number: int,
                 device: torch.device,
                 state_processor: StateProcessor,
                 agent_network: AgentNetwork):
        """
        BaseMasterAgent のコンストラクタ.

        Args:
            n_agents (int): 環境内のエージェント数.
            action_size (int): エージェントのアクション空間のサイズ.
            grid_size (int): グリッド環境のサイズ.
            goals_number (int): 環境内のゴール数.
            device (torch.device): テンソル操作に使用するデバイス (CPU/GPU).
            state_processor (StateProcessor): 状態変換に使用するStateProcessorのインスタンス.
            agent_network (AgentNetwork): 共有AgentNetworkのインスタンス.
        """
        self.n_agents = n_agents
        self.action_size = action_size
        self.grid_size = grid_size
        self.goals_num = goals_number
        self.device = device
        self.state_processor = state_processor
        self.agent_network = agent_network

        # AgentNetworkクラスでnum_channelsを定義済みの前提
        self.num_channels = self.agent_network.num_channels

        # ターゲットAgentNetworkを初期化し、メインネットワークの重みをコピー
        # AgentNetworkのコンストラクタは (grid_size, output_size, total_agents) を期待
        self.agent_network_target = AgentNetwork(grid_size, action_size, n_agents).to(device)
        self.agent_network_target.load_state_dict(self.agent_network.state_dict())
        self.agent_network_target.eval() # ターゲットネットワークは推論モードに設定

    def _get_agent_q_values(
        self,
        agent_network_instance: AgentNetwork,
        raw_global_state_batch: torch.Tensor, # This is the raw flattened global state batch (B, total_features)
        agent_ids_batch_for_q_values: torch.Tensor # (B, N) agent_ids for batch processing
    ) -> torch.Tensor:
        """
        与えられたAgentNetworkインスタンス（メインまたはターゲット）から、
        バッチ内の全エージェントの全アクションに対するQ値を計算します。
        内部でStateProcessorを使用して生の状態をグリッド表現に変換します。

        Args:
            agent_network_instance (AgentNetwork): Q値を計算するAgentNetworkのインスタンス.
            raw_global_state_batch (torch.Tensor): リプレイバッファから取得した生の状態のバッチ (形状: (batch_size, total_flat_features)).
            agent_ids_batch_for_q_values (torch.Tensor): バッチ内の各エージェントのID (形状: (batch_size, n_agents)).

        Returns:
            torch.Tensor: 各エージェントの各アクションに対するQ値 (形状: (batch_size, n_agents, action_size)).
        """
        batch_size = raw_global_state_batch.size(0)
        n_agents = self.n_agents # Use self.n_agents

        # StateProcessorを通して各エージェントの観測を生成 (B*N, C, G, G)
        transformed_obs_list = []
        for agent_idx in range(n_agents):
            # Each call to transform_state_batch processes the entire raw_global_state_batch
            # for a specific agent_idx
            transformed_obs_for_agent = self.state_processor.transform_state_batch(
                agent_idx, raw_global_state_batch
            ) # (batch_size, num_channels, grid_size, grid_size)
            transformed_obs_list.append(transformed_obs_for_agent)

        # Stack them to get (n_agents, batch_size, num_channels, grid_size, grid_size)
        stacked_transformed_obs = torch.stack(transformed_obs_list, dim=0)
        # Reshape to (batch_size * n_agents, num_channels, grid_size, grid_size) for AgentNetwork
        reshaped_transformed_obs = stacked_transformed_obs.permute(1, 0, 2, 3, 4).reshape(
            batch_size * n_agents, self.num_channels, self.grid_size, self.grid_size
        )

        # Agent IDs for the network (B*N,)
        flat_agent_ids = agent_ids_batch_for_q_values.flatten()

        # AgentNetworkからQ値を計算
        # agent_network_instance((B*N, C, G, G), (B*N,)) -> (B*N, A)
        q_values_flat = agent_network_instance(reshaped_transformed_obs, flat_agent_ids)

        # 結果を元の形状 (B, N, A) に戻す
        q_values_reshaped = q_values_flat.view(batch_size, n_agents, self.action_size)

        return q_values_reshaped

    @abstractmethod
    def get_actions(self, global_state: GlobalState, epsilon: float) -> List[int]:
        """
        与えられたグローバル状態とイプシロンに基づいて、各エージェントのアクションを選択します。
        具体的な実装はサブクラスで提供されます。
        """
        pass

    @abstractmethod
    def evaluate_q(
        self,
        global_states_batch_raw: torch.Tensor, # Raw flattened global state (B, total_features)
        actions_batch: torch.Tensor, # (batch_size, n_agents)
        rewards_batch: torch.Tensor, # (batch_size,)
        next_global_states_batch_raw: torch.Tensor, # Raw flattened next global state (B, total_features)
        dones_batch: torch.Tensor, # (batch_size, n_agents) - Individual done flags
        is_weights_batch: Optional[torch.Tensor] = None # (batch_size,)
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        """
        リプレイバッファからサンプリングされたバッチデータを使用して、Q値を評価し、
        損失とTD誤差を計算します。具体的な実装はサブクラスで提供されます。
        """
        pass

    def sync_target_network(self) -> None:
        """
        ターゲットAgentNetworkをメインAgentNetworkと同期します。
        メインネットワークの重みをターゲットネットワークにコピーします。
        """
        self.agent_network_target.load_state_dict(self.agent_network.state_dict())
        self.agent_network_target.eval() # ターゲットネットワークは常に推論モード

    def _huber_loss(
        self,
        q: torch.Tensor,
        target: torch.Tensor,
        is_weights: Optional[torch.Tensor] = None
    ) -> Tuple[torch.Tensor, torch.Tensor]:
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

        # Huber lossの実装
        HUBER_LOSS_DELTA = 1.0
        cond: torch.Tensor = abs_td_errors < HUBER_LOSS_DELTA
        L2: torch.Tensor = 0.5 * torch.square(td_errors)
        L1: torch.Tensor = HUBER_LOSS_DELTA * (abs_td_errors - 0.5 * HUBER_LOSS_DELTA)
        loss_per_sample: torch.Tensor = torch.where(cond, L2, L1) # 形状: (batch_size,)

        # PERを使用する場合、損失に重要度サンプリング重みを適用
        if is_weights is not None:
            # is_weightsの形状が(batch_size, 1)または(batch_size,)であることを確認
            # 必要であればbroadcastする
            weighted_loss = loss_per_sample * is_weights.squeeze(-1) if is_weights.ndim > 1 else loss_per_sample * is_weights
            final_loss = torch.mean(weighted_loss)
        else:
            final_loss = torch.mean(loss_per_sample)

        # 計算された損失とTD誤差の絶対値を返す
        return final_loss, abs_td_errors.detach()

    @abstractmethod
    def get_optimizer_params(self) -> List[nn.Parameter]:
        """
        オプティマイザが更新すべきネットワークパラメータのリストを返します。
        """
        pass
