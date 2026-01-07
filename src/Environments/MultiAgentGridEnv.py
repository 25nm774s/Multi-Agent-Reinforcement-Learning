import random
from typing import Tuple, List, Dict

from .Grid import Grid
from .CollisionResolver import CollisionResolver

from Base.Constant import PosType#, GlobalState

GlobalState = Dict[str,PosType]

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
        self.reward_mode: int = args.reward_mode
        self.neighbor_distance: float = args.neighbor_distance

        # 新しい Grid クラスを使用した内部状態管理
        self._grid: Grid = Grid(self.grid_size)
        self.collision_resolver = CollisionResolver(self._grid)
        self.action_space_size = self.collision_resolver.action_space_size

        self._agent_ids: list[str] = [f'agent_{i}' for i in range(self.agents_number)]
        self._goal_ids: list[str] = [f'goal_{i}' for i in range(self.goals_number)]

        self.distance_fn = distance_fn

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

        # 4. 報酬を計算します
        # 報酬計算は Grid の更新後の位置に基づいて行います
        done_mode = 2
        reward = self._calculate_reward(done_mode=done_mode)

        # 5. 終了条件をチェックします
        done = self._check_done_condition(done_mode=done_mode)

        # 6. 次の観測を取得します
        # Grid が既に更新されているので、_get_observation は現在のグリッド状態を反映します
        next_observation = self._get_observation()

        # 7. 情報辞書 (オプション)
        info:Dict = {"global_state":self._get_global_state(), "total_reward":sum(reward.values())}
        done_dict = {}

        for aid in self._agent_ids:
            done_dict[aid] = done # 全エージェントに同じ全体完了状態を割り当て

        done_dict['__all__'] = done # 全体の完了状態を直接設定

        return next_observation, reward, done_dict, info

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


    def _calculate_total_distance_to_goals_LEGACY(self) -> float:
        """
        各ゴールから最も近いエージェントまでのマンハッタン距離の合計を計算します。
        """
        # print("このメソッドの代わりに_calculate_total_distance_to_goalsを使うことを検討。")

        goal_positions = list(self.get_goal_positions().values())
        agent_positions = list(self.get_agent_positions().values())

        if not agent_positions:
            # エージェントが配置されていない場合 (リセット後には起こらないはず)、距離は無限大です。
            # これは密な報酬で処理されない場合、問題を引き起こす可能性があります。
            # agents_num > 0 で、エージェントがリセットで配置されることを想定しています。
            # print("Warning: _calculate_total_distance_to_goals がエージェントなしで呼び出されました。") # デバッグが必要な場合は残す
            return float('inf') # または非常に大きな数

        total_distance = 0
        for goal in goal_positions:
            # min() を呼び出す前に agent_positions が空でないことを確認します
            if agent_positions: # リセット時に agents_pos が空でないことを確認済みですが、念のため
                min_dist = min(abs(goal[0] - ag[0]) + abs(goal[1] - ag[1]) for ag in agent_positions)
                total_distance += min_dist
            else:
                # ゴールは存在するがエージェントがいない場合の処理 (到達不可能)
                total_distance += self.grid_size * 2 # 最大距離などを加算 (適当な大きな値)

        return float(total_distance)

    def _calculate_total_distance_to_goals(self) -> float:
        """
        エージェントとゴールの最適なペアリング（近い順）を行い、
        そのマンハッタン距離の合計を計算します。
        """
        # 計算量のチェック (100はかなり安全圏。必要に応じて400等に調整)
        if self.agents_number * self.goals_number > 400:
            raise Exception(f"\x1b[41m[警告] 計算量のオーバーヘッドが懸念されます。\x1b[49mペア数: {self.agents_number * self.goals_number}")

        goal_positions = list(self.get_goal_positions().values())
        agent_positions = list(self.get_agent_positions().values())

        if not agent_positions or not goal_positions:
            return float(self.grid_size * 2 * self.goals_number)

        # 1. 全ペアの距離を計算
        all_pairs = []
        for a_idx, a_pos in enumerate(agent_positions):
            for g_idx, g_pos in enumerate(goal_positions):
                dist = abs(a_pos[0] - g_pos[0]) + abs(a_pos[1] - g_pos[1])
                all_pairs.append((dist, a_idx, g_idx))

        # 2. 距離が近い順にソート
        all_pairs.sort(key=lambda x: x[0])

        # 3. 近いペアから順に確定
        total_distance = 0.0
        used_agents = set()
        used_goals = set()
        matched_count = 0

        for dist, a_idx, g_idx in all_pairs:
            if a_idx not in used_agents and g_idx not in used_goals:
                used_agents.add(a_idx)
                used_goals.add(g_idx)
                total_distance += dist
                matched_count += 1

                if matched_count == min(len(agent_positions), len(goal_positions)):
                    break

        return float(total_distance)

    def _calculate_reward(self,done_mode) -> Dict[str,float]:
        """
        現在の状態と報酬モードに基づいて報酬を計算します。
        """
        reward = 0.0
        done = self._check_done_condition(done_mode) # 報酬計算のために完了ステータスをチェック

        if self.reward_mode == 0:
            # モード 0: 完了条件を満たしていれば +100、それ以外 0
            reward = 100.0 if done else 0.0

        elif self.reward_mode == 1:
            # モード 2: ゴールまでの合計距離の負の値
            current_total_distance = self._calculate_total_distance_to_goals_LEGACY()
            reward = - current_total_distance

        elif self.reward_mode == 2:
            # モード 2: 近いペアから順に確定させていく（貪欲法）」ロジック
            current_total_distance = self._calculate_total_distance_to_goals()
            reward = - current_total_distance

        else:
            print(f"Warning: 未知の報酬モード: {self.reward_mode}。報酬は 0 です。")
            reward = 0.0

        res:Dict[str,float] = {}
        for id in self._agent_ids:
            res[id] = reward
        return res

    def _check_done_condition(self,done_mode) -> bool:
        """
        エピソードが終了したかチェックします。完了条件は self.done_mode に依存します。
        - done_mode 0: 全てのゴールが少なくとも1つのエージェントによって占有されているか。
        - done_mode 1: いずれかのゴールが少なくとも1つのエージェントによって占有されているか。
        - done_mode 2: 全てのエージェントがいずれかのゴール位置にいるか。
        """
        agent_positions = list(self.get_agent_positions().values())
        current_agents_pos_set = set(agent_positions)
        goal_positions = list(self.get_goal_positions().values())
        goal_positions_set = set(goal_positions)


        if done_mode == 0:
            # Rule 1: 全てのゴールがエージェントによって占有されているか
            return all(goal in current_agents_pos_set for goal in goal_positions)
        elif done_mode == 1:
            # Rule 2: いずれかのゴールが少なくとも1つのエージェントによって占有されているか
            return any(goal in current_agents_pos_set for goal in goal_positions)
        elif done_mode == 2:
            # Rule 3: 全てのエージェントがいずれかのゴール位置にいるか
            return all(agent_pos in goal_positions_set for agent_pos in agent_positions)
        else:
            print(f"Warning: 未知の完了条件モード: {done_mode}。常に False を返します。")
            return False

    def get_all_object(self):
        r = self._grid.get_all_object_positions()
        return r
