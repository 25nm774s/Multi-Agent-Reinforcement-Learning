from abc import ABC, abstractmethod
import copy
import torch
import torch.nn as nn # オプティマイザーを設定するため。
from typing import List, Optional, Tuple, Dict

from Environments.StateProcesser import ObsToTensorWrapper
from DQN.network import IAgentNetwork, IAgentNetwork

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
                 state_processor: ObsToTensorWrapper,
                 agent_network_instance: IAgentNetwork,
                 agent_ids: List[str],
                 goal_ids: List[str],
                 agent_reward_processing_mode: str): # Add new argument
        """
        BaseMasterAgent のコンストラクタ.

        Args:
            n_agents (int): 環境内のエージェント数.
            action_size (int): エージェントのアクション空間のサイズ.
            grid_size (int): グリッド環境のサイズ.
            goals_number (int): 環境内のゴール数.
            device (torch.device): テンソル操作に使用するデバイス (CPU/GPU).
            state_processor (StateProcessor): 状態変換に使用するStateProcessorのインスタンス.
            agent_network_instance (IAgentNetwork): 共有AgentNetworkのインスタンス (IAgentNetwork型).
            agent_ids (List[str]): エージェントのIDリスト.
            goal_ids (List[str]): ゴールのIDリスト.
            agent_reward_processing_mode (str): 報酬処理モード ('individual', 'sum_and_distribute', 'mean_and_distribute').
        """
        super().__init__()
        self.n_agents = n_agents
        self.action_size = action_size
        self.grid_size = grid_size
        self.goals_num = goals_number
        self.device = device
        self.state_processor = state_processor
        self.agent_network = agent_network_instance # Store the provided instance
        self._agent_ids = agent_ids
        self._goal_ids = goal_ids
        self.agent_reward_processing_mode = agent_reward_processing_mode # Store the new argument

        # num_channels は IAgentNetwork のプロパティから取得
        self.num_channels = self.agent_network.num_channels

        # ターゲットネットワークはメインネットワークのコピーとして作成
        self.agent_network_target = copy.deepcopy(self.agent_network).to(device)
        self.agent_network_target.eval()

    def _process_rewards_and_dones(
        self,
        rewards_tensor_batch: torch.Tensor, # (B, N)
        dones_tensor_batch: torch.Tensor,   # (B, N) - individual dones for each agent
        all_agents_done_batch: torch.Tensor # (B, 1) - overall episode done flag
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        指定されたモードに基づいて報酬と完了フラグを処理し、IQLとQMIXの両方で使用できる形式で返します。

        Args:
            rewards_tensor_batch (torch.Tensor): バッチ内の各エージェントの報酬 (B, N)。
            dones_tensor_batch (torch.Tensor): バッチ内の各エージェントの完了フラグ (B, N)。
            all_agents_done_batch (torch.Tensor): 各バッチサンプルにおけるエピソード全体の完了フラグ (B, 1)。

        Returns:
            Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
                - processed_rewards_for_iql (torch.Tensor): IQLで使用する (B, N) 形状の報酬。
                - processed_dones_for_iql (torch.Tensor): IQLで使用する (B, N) 形状の完了フラグ。
                - team_reward_scalar (torch.Tensor): QMIXで使用する (B, 1) 形状のチーム報酬。
                - team_done_scalar (torch.Tensor): QMIXで使用する (B, 1) 形状のチーム完了フラグ (all_agents_done_batchをそのまま使用)。
        """
        batch_size = rewards_tensor_batch.size(0)

        # IQL用報酬: agent_reward_processing_modeに従って処理 (B, N)
        processed_rewards_for_iql = rewards_tensor_batch.clone()
        if self.agent_reward_processing_mode == 'sum_and_distribute':
            sum_rewards = rewards_tensor_batch.sum(dim=1, keepdim=True) # (B, 1)
            processed_rewards_for_iql = sum_rewards.expand_as(rewards_tensor_batch) # (B, N)
        elif self.agent_reward_processing_mode == 'mean_and_distribute':
            mean_rewards = rewards_tensor_batch.mean(dim=1, keepdim=True) # (B, 1)
            processed_rewards_for_iql = mean_rewards.expand_as(rewards_tensor_batch) # (B, N)
        # 'individual'の場合はprocessed_rewards_for_iqlはrewards_tensor_batchのクローンがそのまま使われる

        # IQL用完了フラグ: 各エージェントの完了フラグをそのまま使用 (B, N)
        processed_dones_for_iql = dones_tensor_batch.clone()

        # QMIX用チーム報酬: agent_reward_processing_modeに従って処理 (B, 1)
        # ここでのagent_reward_processing_modeは、QMIXの「チーム全体報酬」の定義を制御すると解釈する
        team_reward_scalar: torch.Tensor
        if self.agent_reward_processing_mode == 'individual' or self.agent_reward_processing_mode == 'sum_and_distribute':
            # 'individual'モードでは、QMIXはデフォルトで個々の報酬の合計をチーム報酬とする
            # 'sum_and_distribute'モードでも、チーム報酬は個々の報酬の合計が自然な解釈
            team_reward_scalar = rewards_tensor_batch.sum(dim=1, keepdim=True) # (B, 1)
        elif self.agent_reward_processing_mode == 'mean_and_distribute':
            # 'mean_and_distribute'モードでは、チーム報酬は個々の報酬の平均
            team_reward_scalar = rewards_tensor_batch.mean(dim=1, keepdim=True) # (B, 1)
        else:
            raise ValueError(f"Unknown agent_reward_processing_mode for QMIX: {self.agent_reward_processing_mode}")

        # QMIX用チーム完了フラグ: エピソード全体の完了フラグを使用 (B, 1)
        team_done_scalar = all_agents_done_batch.clone() # (B, 1)

        return processed_rewards_for_iql, processed_dones_for_iql, team_reward_scalar, team_done_scalar

    # _process_raw_observations_batch は ReplayBuffer が既にテンソルを扱うようになったため、不要となり削除されます。
    # _flatten_global_state_dict は StateProcessor に移動したため削除されます。

    def _get_agent_q_values(
        self,
        agent_network_instance: "IAgentNetwork",
        agent_obs_batch: torch.Tensor, # (B * N, C, G, G)
        agent_ids_batch: torch.Tensor # (B * N,)
    ) -> torch.Tensor:
        """
        与えられたAgentNetworkインスタンス（メインまたはターゲット）から、
        バッチ内の全エージェントの全アクションに対するQ値を計算します。

        Args:
            agent_network_instance (IAgentNetwork): Q値を計算するAgentNetworkのインスタンス.
            agent_obs_batch (torch.Tensor): AgentNetworkへの入力 (形状: (batch_size * n_agents, num_channels, grid_size, grid_size)).
            agent_ids_batch (torch.Tensor): バッチ内の各エージェントのID (形状: (batch_size * n_agents,)).

        Returns:
            torch.Tensor: 各エージェントの各アクションに対するQ値 (形状: (batch_size, n_agents, action_size)).
        """
        # AgentNetworkからQ値を計算
        # agent_network_instance((B*N, C, G, G), (B*N,)) -> (B*N, A)
        q_values_flat = agent_network_instance(agent_obs_batch, agent_ids_batch)

        # 結果を元の形状 (B, N, A) に戻す
        batch_size = agent_obs_batch.size(0) // self.n_agents
        q_values_reshaped = q_values_flat.view(batch_size, self.n_agents, self.action_size)

        return q_values_reshaped

    @abstractmethod
    def get_actions(self, agent_obs_tensor: torch.Tensor, epsilon: float) -> Dict[str, int]:
        """
        与えられた現在のステップの観測とイプシロンに基づいて、各エージェントのアクションを選択します。

        Args:
            agent_obs_tensor (torch.Tensor): 各エージェントのグリッド観測 (n_agents, num_channels, grid_size, grid_size)。
            epsilon (float): 探索率。

        Returns:
            Dict[str, int]: 各エージェントIDをキー、選択された行動を値とする辞書。
        """
        pass

    @abstractmethod
    def evaluate_q(
        self,
        agent_obs_tensor_batch: torch.Tensor, # (B, N, C, G, G)
        global_state_tensor_batch: torch.Tensor, # (B, state_dim)
        actions_tensor_batch: torch.Tensor, # (B, N)
        rewards_tensor_batch: torch.Tensor, # (B, N)
        dones_tensor_batch: torch.Tensor, # (B, N)
        all_agents_done_batch: torch.Tensor, # (B, 1) - NEW
        next_agent_obs_tensor_batch: torch.Tensor, # (B, N, C, G, G)
        next_global_state_tensor_batch: torch.Tensor, # (B, state_dim)
        is_weights_batch: Optional[torch.Tensor] = None
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        """
        リプレイバッファからサンプリングされたバッチデータを使用して、Q値を評価し、
        損失とTD誤差を計算します。具体的な実装はサブクラスで提供されます。

        Args:
            agent_obs_tensor_batch (torch.Tensor): 現在のエージェント観測バッチ (B, N, C, G, G)。
            global_state_tensor_batch (torch.Tensor): 現在のグローバル状態バッチ (B, state_dim)。
            actions_tensor_batch (torch.Tensor): エージェントが取った行動のバッチ (B, N)。
            rewards_tensor_batch (torch.Tensor): 環境からの生の報酬のバッチ (B, N)。
            dones_tensor_batch (torch.Tensor): 環境からの個々の完了フラグのバッチ (B, N)。
            all_agents_done_batch (torch.Tensor): エピソード全体の完了フラグのバッチ (B, 1)。
            next_agent_obs_tensor_batch (torch.Tensor): 次のエージェント観測バッチ (B, N, C, G, G)。
            next_global_state_tensor_batch (torch.Tensor): 次のグローバル状態バッチ (B, state_dim)。
            is_weights_batch (Optional[torch.Tensor]): PERの重要度サンプリング重み (B,)。

        Returns:
            Tuple[torch.Tensor, Optional[torch.Tensor]]:
                - loss (torch.Tensor): 計算された損失。
                - abs_td_errors (Optional[torch.Tensor]): TD誤差の絶対値 (PERの場合)。
        """
        pass

    def sync_target_network(self) -> None:
        """
        ターゲットAgentNetworkをメインAgentNetworkと同期します。
        メインネットワークの重みをターゲットネットワークにコピーします。
        """
        self.agent_network_target.load_state_dict(self.agent_network.state_dict())
        self.agent_network_target.eval()

    def _huber_loss(
        self,
        q: torch.Tensor,
        target: torch.Tensor,
        is_weights: Optional[torch.Tensor] = None
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Huber Loss を計算する. PERを使用する場合は重要度サンプリング重みを適用する.
        TD誤差も計算して返す.
        """
        td_errors: torch.Tensor = target - q
        abs_td_errors: torch.Tensor = torch.abs(td_errors)

        HUBER_LOSS_DELTA = 1.0
        cond: torch.Tensor = abs_td_errors < HUBER_LOSS_DELTA
        L2: torch.Tensor = 0.5 * torch.square(td_errors)
        L1: torch.Tensor = HUBER_LOSS_DELTA * (abs_td_errors - 0.5 * HUBER_LOSS_DELTA)
        loss_per_sample: torch.Tensor = torch.where(cond, L2, L1)

        if is_weights is not None:
            # `q` が `(B*N,)` で `is_weights` が `(B,)` の場合、`is_weights` を `(B*N,)` に拡張します。
            # evaluate_q メソッド内で既に `expanded_is_weights` が生成されているため、
            # ここでは `is_weights` が適切な形状であることを期待します。
            final_loss = torch.mean(loss_per_sample * is_weights)
        else:
            final_loss = torch.mean(loss_per_sample)

        return final_loss, abs_td_errors.detach()

    @abstractmethod
    def get_optimizer_params(self) -> List[nn.Parameter]:
        """
        オプティマイザが更新すべきネットワークパラメータのリストを返します。
        """
        pass
