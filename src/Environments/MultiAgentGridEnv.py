import random
from abc import ABC, abstractmethod
import torch
from typing import Tuple, Dict, List, Optional, Any
from itertools import product

from .Grid import Grid
from .CollisionResolver import CollisionResolver
from .RewardCalculator import RewardCalculator

from Base.Constant import PosType, GlobalState

class IEnvWrapper(ABC):
    """
    環境ラッパー抽象クラス。
    強化学習エージェントとのインターフェースを標準化し、
    環境の観測、行動、報酬、完了状態を統一された形式で提供します。
    """

    @abstractmethod
    def reset(self, initial_agent_positions: Optional[List[PosType]] = None) -> Tuple[torch.Tensor, torch.Tensor, Dict[str, Any]]:
        """
        環境をリセットし、初期観測を返します。

        Args:
            initial_agent_positions (Optional[List[PosType]]): 各エージェントの初期位置のリスト。
                                                            Noneの場合、ランダムに配置されます。

        Returns:
            Tuple[torch.Tensor, torch.Tensor, Dict[str, Any]]:
                - agent_obs_tensor (torch.Tensor): 各エージェントのグリッド観測 (n_agents, num_channels, grid_size, grid_size)。
                - global_state_tensor (torch.Tensor): フラット化されたグローバル状態 ((goals_number + n_agents) * 2,)。
                - info (Dict[str, Any]): 環境に関する追加情報。少なくとも 'raw_global_state': Dict[str, PosType] を含む。
        """
        pass

    @abstractmethod
    def step(self, actions: Dict[str, int]) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, Dict[str, Any]]:
        """
        指定されたエージェントの行動を実行し、次の状態、報酬、完了フラグを返します。

        Args:
            actions (Dict[str, int]): 各エージェントのIDをキーとする行動の辞書。

        Returns:
            Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, Dict[str, Any]]:
                - next_agent_obs_tensor (torch.Tensor): 各エージェントの次のグリッド観測 (n_agents, num_channels, grid_size, grid_size)。
                - next_global_state_tensor (torch.Tensor): 次のフラット化されたグローバル状態 ((goals_number + n_agents) * 2,)。
                - rewards_tensor (torch.Tensor): 各エージェントの報酬 (n_agents,)。
                - dones_tensor (torch.Tensor): 各エージェントの完了フラグ (0.0: not done, 1.0: done) (n_agents,)。
                - info (Dict[str, Any]): 環境に関する追加情報。少なくとも 'raw_global_state': Dict[str, PosType] および 'all_agents_done': bool を含む。
        """
        pass

    @property
    @abstractmethod
    def action_space_size(self) -> int:
        """エージェントのアクション空間のサイズを返します。"""
        pass

    @property
    @abstractmethod
    def n_agents(self) -> int:
        """環境内のエージェントの数を返します。"""
        pass

    @property
    @abstractmethod
    def grid_size(self) -> int:
        """グリッド環境のサイズを返します。"""
        pass

    @property
    @abstractmethod
    def goals_number(self) -> int:
        """環境内のゴールの数を返します。"""
        pass

    @property
    @abstractmethod
    def agent_ids(self) -> List[str]:
        """エージェントのIDのリストを返します。"""
        pass

    @property
    @abstractmethod
    def goal_ids(self) -> List[str]:
        """ゴールのIDのリストを返します。"""
        pass

    @property
    @abstractmethod
    def num_channels(self) -> int:
        """観測グリッドのチャネル数を返します。"""
        pass

class MultiAgentGridEnv:
    """
    マルチエージェントグリッドワールドのための強化学習環境。
    """
    def __init__(self,
                 args,
                 fixrd_goals=[],
                 distance_fn=(lambda p,q: int(abs(p[0] - q[0]) + abs(p[1] - q[1])))):
        """
        MultiAgentGridEnv を初期化します。

        Args:
            args: 以下の属性を持つ設定オブジェクト：
                grid_size (int)
                agents_number (int)
                goals_number (int)
                reward_mode (int)
        """
        self.grid_size: int = args.grid_size
        self.agents_number: int = args.agents_number
        self.goals_number: int = args.goals_number
        # self.reward_mode: int = args.reward_mode
        self.neighbor_distance: float = args.neighbor_distance
        self.done_mode = 2

        # 新しい Grid クラスを使用した内部状態管理
        self._grid: Grid = Grid(self.grid_size)
        self.collision_resolver = CollisionResolver(self._grid)
        self.action_space_size = self.collision_resolver.action_space_size

        self._agent_ids: list[str] = [f'agent_{i}' for i in range(self.agents_number)]
        self._goal_ids: list[str] = [f'goal_{i}' for i in range(self.goals_number)]

        self.reward_calculator = RewardCalculator(self.grid_size, args.reward_mode, self._agent_ids, self.agents_number, self.goals_number)

        self.distance_fn = distance_fn
        self.prev_distances: Dict[str, float] = {}

        fixrd_goals_list=list(fixrd_goals)
        if len(fixrd_goals) < self.goals_number:
            sub = self.goals_number - len(fixrd_goals)# 不足するゴール数
            zahyo = self._grid.sample(sub)
            for goal in zahyo:
                fixrd_goals_list.append(goal)
        # 環境初期化時に一度だけ固定ゴールを設定します
        for i, goal in enumerate(fixrd_goals_list):
            self._grid.add_object(self._goal_ids[i], goal)


    def reset(self, initial_agent_positions: list[PosType] = []) -> Dict:
        """
        新しいエピソードのために環境をリセットします。

        Args:
            initial_agent_positions (list[PositionType], optional): エージェントの明示的な開始位置のリスト。
                None の場合、ランダム配置。

        Returns:
            object: 環境の初期観測。
        """
        # グリッドからエージェントのみをクリアし、ゴールは保持します
        agent_ids_to_remove = [obj_id for obj_id in self._grid._object_positions if obj_id.startswith('agent_')]
        for agent_id in agent_ids_to_remove:
            del self._grid._object_positions[agent_id]


        # エージェントをゴールに配置しないように、現在のゴール位置を取得します
        # existing_positions をここで定義し、常に利用可能にします
        existing_positions = list(self.get_goal_positions().values())


        # モードまたは明示的な位置に基づいてエージェントを配置します
        if initial_agent_positions:
            if len(initial_agent_positions) != self.agents_number:
                raise ValueError(f"初期エージェント位置数 ({len(initial_agent_positions)}) が設定数 ({self.agents_number}) と異なります。")
            agent_positions = initial_agent_positions
            # 明示的な位置がゴールと重複していないか検証します
            for pos in agent_positions:
                if tuple(pos) in existing_positions:
                    raise ValueError(f"エージェント初期位置 {pos} がゴール位置と重複しています。")

        else:
            # ゴールに占有されていない一意なランダム位置を生成します
            agent_positions = self._grid.sample(self.agents_number)
            #print(agent_positions,"\tMAGE.reset():agent_positions")

        # グリッドにエージェントを追加します
        for i, pos in enumerate(agent_positions):
            # エージェント位置がゴールと重複していないか、追加前にチェックします (明示的な場合は上で既に実施)
            self._grid.add_object(self._agent_ids[i], pos)

        # 初回の距離を計算して保持（報酬計算の基準点）
        # self.prev_distances = self._calculate_total_distance_to_goals()
        # self.reward_calculator.prev_distances(self.get_agent_positions(), self.get_goal_positions())

        # 観測 (例: グローバル状態タプル) を返します
        return self._get_observation()

    def step(self, actions: Dict[str, int])->Tuple[Dict,Dict,Dict,Dict]:
        """
        各エージェントのアクションを受け取り、環境を更新します。

        Args:
            actions (Dict[str, int]): 各エージェントに対するアクションの辞書

        Returns:
            tuple: (observation, reward, done, info)
                observation (object): ステップ後の環境の観測。
                reward (dict): ステップ後に受け取った報酬。
                done (dict): エピソードが終了したかどうか。
                info (dict): 追加情報 (例: デバッグ情報)。
        """
        if len(actions) != self.agents_number:
            raise ValueError(f"アクション数 ({len(actions)}) がエージェント数 ({self.agents_number}) と異なります。")

        # Grid クラスの衝突解決メソッドを呼び出し、位置を更新してもらう
        # このメソッド内で Grid インスタンスの_object_positionsが更新されます
        self.collision_resolver.resolve_agent_movements(actions)

        # 4. 終了条件をチェックします
        dones = self._check_done_condition(self.done_mode)

        # 5. 報酬を計算します
        # 報酬計算は Grid の更新後の位置に基づいて行います
        reward = self.reward_calculator._calculate_reward(dones['__all__'], self.get_agent_positions(), self.get_goal_positions())

        # 6. 次の観測を取得します
        # Grid が既に更新されているので、_get_observation は現在のグリッド状態を反映します
        next_observation = self._get_observation()

        # 7. 情報辞書 (オプション)
        info:Dict = {"global_state":self._get_global_state(), "total_reward":sum(reward.values())}

        return next_observation, reward, dones, info

    # --- ヘルパーメソッド (内部ロジック) ---

    def _setup_fixed_goals(self):
        """
        グリッド内に固定ゴール位置を設定します。
        これは環境初期化時に一度だけ呼び出されます。
        実装例: ランダムな固定ゴール。
        """
        # 予期せず複数回呼び出された場合のために、既存のゴールをクリアします
        for goal_id in self._goal_ids:
            if goal_id in self._grid._object_positions:
                del self._grid._object_positions[goal_id]

        # ゴールのランダムな一意な位置を生成します
        # 環境インスタンスから grid_size 属性を渡します
        goal_positions = self._generate_unique_positions(self.goals_number, [], self.grid_size)

        # グリッドにゴールを追加します
        for i, pos in enumerate(goal_positions):
            self._grid.add_object(self._goal_ids[i], pos)


    def _generate_unique_positions(self, num_positions: int, existing_positions: list[PosType], grid_size: int) -> list[PosType]:
        """
        既存の位置を避けながら、グリッド内に指定された数の一意なランダム位置のリストを生成します。

        Args:
            num_positions (int): 生成する一意な位置の数。
            existing_positions (list[PositionType]): 避ける位置。
            grid_size (int): グリッドのサイズ。

        Returns:
            list[PositionType]: 生成された一意な位置のリスト。

        Raises:
            ValueError: 必要な数の一意な位置を生成できない場合。
        """
        positions = []
        existing_positions_set = set(existing_positions)
        all_possible_positions = [(x, y) for x in range(grid_size) for y in range(grid_size)]
        available_positions = [pos for pos in all_possible_positions if pos not in existing_positions_set]

        if num_positions > len(available_positions):
            raise ValueError(f"{num_positions} 個の一意な位置を生成できません。利用可能なのは {len(available_positions)} 個のみです。")

        positions = random.sample(available_positions, num_positions)
        return positions

    def _get_observation(self) -> dict:
        """
        各エージェントごとの部分観測を生成します。
        - 自分の位置
        - 全ゴールの位置（ID順に固定）
        - 他エージェントの位置（近傍以外は -1, -1）
        """
        observations = {}

        # 1. 全ゴールの位置を事前にリスト化（ID順を保証）
        all_goal_positions = [
            self._grid.get_object_position(goal_id) for goal_id in self._goal_ids
        ]

        for id in (self._agent_ids):
            my_pos = self._grid.get_object_position(id)

            # 2. 他のエージェントの情報を収集（自分以外）
            others_info:Dict[str,PosType] = {}
            for other_id in self._agent_ids:
                if id == other_id: continue

                other_pos = self._grid.get_object_position(other_id)
                # 距離関数で判定（マンハッタン距離など）
                if self.distance_fn(my_pos, other_pos) <= self.neighbor_distance:
                    others_info[other_id] = other_pos
                else:
                    others_info[other_id] = (-1, -1)

            # 3. エージェントごとの観測辞書を構築
            observations[id] = {
                'self': my_pos,                  # エージェントの位置
                'all_goals': all_goal_positions, # 全エージェントで共通のリスト
                'others': others_info            # 他のエージェントの位置
            }

        return observations

    def _get_global_state(self)->GlobalState: return self._grid.get_all_object_positions()

    def get_goal_positions(self) -> dict[str, PosType]:
        """現在のゴール位置を取得するヘルパーメソッド。"""
        # グリッドから goal_ids の位置を取得します
        goal_pos_dict = {}
        for goal_id in self._goal_ids:
            try:
                goal_pos_dict[goal_id] = self._grid.get_object_position(goal_id)
            except KeyError:
                # グリッドにゴールが見つかりません。_setup_fixed_goals が実行されていればこれは起こりません。
                print(f"Warning: ゴール '{goal_id}' がグリッドに見つかりませんでした。")
                pass # または適切にエラー処理

        return goal_pos_dict

    def get_agent_positions(self) -> dict[str, PosType]:
        """現在のエージェント位置を取得するヘルパーメソッド。"""
        # グリッドから agent_ids の位置を取得します
        agent_pos_dict = {}
        for agent_id in self._agent_ids:
            try:
                agent_pos_dict[agent_id] = self._grid.get_object_position(agent_id)
            except KeyError:
                # グリッドにエージェントが見つかりません。リセット後であればこれは起こりません。
                print(f"Warning: エージェント '{agent_id}' がグリッドに見つかりませんでした。")
                pass # または適切にエラー処理
        return agent_pos_dict

    def _check_done_condition(self,done_mode) -> Dict[str, bool]:
        """
        エピソードが終了したかチェックします。完了条件は self.done_mode に依存します。
        - done_mode 0: 全てのゴールが少なくとも1つのエージェントによって占有されているか。
        - done_mode 1: 誰か一人でもゴール。
        - done_mode 2: 全員がゴール。
        """
        agent_positions = self.get_agent_positions()
        goal_positions_set = set(self.get_goal_positions().values())

        # 各エージェントが現在ゴールにいるか
        dones = {aid: pos in goal_positions_set for aid, pos in agent_positions.items()}
        overall_done = False

        if done_mode == 0:
            # Rule 1: 全てのゴールがエージェントによって占有されているか
            overall_done = all(goal in set(agent_positions.values()) for goal in self.get_goal_positions().values())

        elif done_mode == 1:
            # Rule 2: 誰か一人でもゴールしたらエピソード終了
            overall_done = any(dones.values())        

        elif done_mode == 2:
            # Rule 3: 全員がゴールしたらエピソード終了
            overall_done = all(dones.values())

        else:
            print(f"Warning: 未知の完了条件モード: {done_mode}。done_mode=2にフォールバック。")
            overall_done = all(dones.values())
        
        dones["__all__"] = overall_done
        return dones

    # def get_all_object(self):
    #     r = self._grid.get_all_object_positions()
    #     return r

from Environments.StateProcesser import ObsToTensorWrapper
class GridEnvWrapper(IEnvWrapper):
    """
    MultiAgentGridEnv を IEnvWrapper インターフェースに適合させるラッパークラス。
    環境の生の出力を標準化されたテンソル形式に変換します。
    """
    def __init__(self, env_instance: MultiAgentGridEnv, state_processor_instance: ObsToTensorWrapper):
        self._env = env_instance
        self._state_processor = state_processor_instance
        self._agent_ids = self._env._agent_ids
        self._goal_ids = self._env._goal_ids

    @property
    def action_space_size(self) -> int:
        return self._env.action_space_size

    @property
    def n_agents(self) -> int:
        return self._env.agents_number

    @property
    def grid_size(self) -> int:
        return self._env.grid_size

    @property
    def goals_number(self) -> int:
        return self._env.goals_number

    @property
    def agent_ids(self) -> List[str]:
        return self._agent_ids

    @property
    def goal_ids(self) -> List[str]:
        return self._goal_ids

    @property
    def num_channels(self) -> int:
        return self._state_processor.num_channels

    def reset(self, initial_agent_positions:List[PosType]=[]) -> Tuple[torch.Tensor, torch.Tensor, Dict[str, Any]]:
        current_partial_observations_dict = self._env.reset(initial_agent_positions)
        true_current_global_state: GlobalState = self._env._get_global_state()

        # (goals_number + n_agents) * 2,)
        global_state_tensor = self._state_processor._flatten_global_state_dict(true_current_global_state)

        # 各エージェントの観測を変換し、スタックして (n_agents, num_channels, grid_size, grid_size) テンソルを作成
        transformed_agent_obs_list = []
        for agent_id in self._agent_ids:
            single_agent_obs = current_partial_observations_dict[agent_id]
            transformed_obs = self._state_processor.transform_state_batch(single_agent_obs)
            transformed_agent_obs_list.append(transformed_obs)
        agent_obs_tensor = torch.stack(transformed_agent_obs_list, dim=0)

        info: Dict[str, Any] = {
            'raw_global_state': true_current_global_state
        }

        return agent_obs_tensor, global_state_tensor, info

    def step(self, actions: Dict[str, int]) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, Dict[str, Any]]:
        next_partial_observations_dict, reward_dict_per_agent, done_dict, env_info = self._env.step(actions)
        true_next_global_state: GlobalState = self._env._get_global_state()

        # (goals_number + n_agents) * 2,)
        next_global_state_tensor = self._state_processor._flatten_global_state_dict(true_next_global_state)

        # 各エージェントの次の観測を変換し、スタックして (n_agents, num_channels, grid_size, grid_size) テンソルを作成
        transformed_next_agent_obs_list = []
        for agent_id in self._agent_ids:
            single_agent_obs = next_partial_observations_dict[agent_id]
            transformed_obs = self._state_processor.transform_state_batch(single_agent_obs)
            transformed_next_agent_obs_list.append(transformed_obs)
        next_agent_obs_tensor = torch.stack(transformed_next_agent_obs_list, dim=0)

        # 報酬を (n_agents,) テンソルに変換
        rewards_list = [reward_dict_per_agent[agent_id] for agent_id in self._agent_ids]
        rewards_tensor = torch.tensor(rewards_list, dtype=torch.float32, device=self._state_processor.device)

        # Doneフラグを (n_agents,) テンソルに変換
        dones_list = [float(done_dict[agent_id]) for agent_id in self._agent_ids]
        dones_tensor = torch.tensor(dones_list, dtype=torch.float32, device=self._state_processor.device)

        info: Dict[str, Any] = {
            'raw_global_state': true_next_global_state,
            'all_agents_done': done_dict['__all__']
        }

        return next_agent_obs_tensor, next_global_state_tensor, rewards_tensor, dones_tensor, info