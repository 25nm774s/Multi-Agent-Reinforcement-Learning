# Implementation of Grid and MultiAgentGridEnv classes based on the design

import random
#import matplotlib.pyplot as plt

from .Grid import Grid, PositionType
from .CollisionResolver import CollisionResolver
#from utils.grid_renderer import GridRenderer

# GridRenderer クラスは以前のセルで定義され、利用可能であることを想定しています。
# もし利用できない場合は、MultiAgentGridEnv.__init__ 内の警告で通知されます。

class MultiAgentGridEnv:
    """
    マルチエージェントグリッドワールドのための強化学習環境。
    """
    def __init__(self, args, fixrd_goals:tuple[PositionType,...]|list[PositionType]=[]):
        """
        MultiAgentGridEnv を初期化します。

        Args:
            args: 以下の属性を持つ設定オブジェクト：
                grid_size (int)
                agents_number (int)
                goals_num (int)
                reward_mode (int)
        """
        self.grid_size: int = args.grid_size
        self.agents_number: int = args.agents_number
        self.goals_number: int = args.goals_number
        self.reward_mode: int = args.reward_mode
        #self.render_mode: int = args.render_mode
        #self.pause_duration: float = args.pause_duration

        # 新しい Grid クラスを使用した内部状態管理
        self._grid: Grid = Grid(self.grid_size)
        self.collision_resolver = CollisionResolver(self._grid)

        self._agent_ids: list[str] = [f'agent_{i}' for i in range(self.agents_number)]
        self._goal_ids: list[str] = [f'goal_{i}' for i in range(self.goals_number)]

        # 報酬計算のための内部状態
        self._goals_reached_status: list[bool] = [False] * self.goals_number # 密な報酬モード3用
        self._prev_total_distance_to_goals: float = 0.0 # 密な報酬モード3用

        fixrd_goals_list=list(fixrd_goals)
        if len(fixrd_goals) < self.goals_number:
            sub = self.goals_number - len(fixrd_goals)# 不足するゴール数
            zahyo = self._grid.sample(sub)
            for goal in zahyo: 
                fixrd_goals_list.append(goal)
        # 環境初期化時に一度だけ固定ゴールを設定します
        for i, goal in enumerate(fixrd_goals_list):
            #print("Goal.type: ", type(goal))
            self._grid.add_object(self._goal_ids[i], goal)


    def reset(self, initial_agent_positions: list[PositionType] = None, placement_mode: str = 'random') -> tuple[PositionType, ...]:#type:ignore
        """
        新しいエピソードのために環境をリセットします。

        Args:
            initial_agent_positions (list[PositionType], optional): エージェントの明示的な開始位置のリスト。
                None の場合、placement_mode が使用されます。デフォルトは 'random'。
            placement_mode (str): initial_agent_positions が None の場合のエージェント配置モード ('random' または 'near_goals')。デフォルトは 'random'。

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
        if initial_agent_positions is not None:
            if len(initial_agent_positions) != self.agents_number:
                raise ValueError(f"初期エージェント位置数 ({len(initial_agent_positions)}) が設定数 ({self.agents_number}) と異なります。")
            agent_positions = initial_agent_positions
            # 明示的な位置がゴールと重複していないか検証します
            for pos in agent_positions:
                if tuple(pos) in existing_positions:
                    raise ValueError(f"エージェント初期位置 {pos} がゴール位置と重複しています。")

        else:
            if placement_mode == 'random':
                # ゴールに占有されていない一意なランダム位置を生成します
                #agent_positions = self._generate_unique_positions(
                #    self.agents_number, existing_positions, self.grid_size
                #) <<-Gridにランダム座標の生成を移行したため
                agent_positions = self._grid.sample(self.agents_number)
                #print(agent_positions,"\tMAGE.reset():agent_positions")
            elif placement_mode == 'near_goals':
                # ゴール近傍にエージェントを配置するロジックを実装します
                try:
                    agent_positions = self._place_agents_near_goals_logic() # 内部ロジック
                    # 生成された near_goals 位置がゴールと重複していないかチェックします
                    if any(pos in existing_positions for pos in agent_positions):
                        raise ValueError("生成された near_goals 位置がゴール位置と重複しています。")
                except (RuntimeError, ValueError) as e: # ロジックからの特定のエラーを捕捉
                    print(f"ゴール近傍にエージェントを配置できませんでした: {e}")
                    print("このエピソードではランダム配置にフォールバックします。")
                    agent_positions:list[PositionType] = self._generate_unique_positions(
                        self.agents_number, existing_positions, self.grid_size
                    )
            else:
                raise ValueError(f"未知の配置モード: {placement_mode}")

        # グリッドにエージェントを追加します
        for i, pos in enumerate(agent_positions):
            # エージェント位置がゴールと重複していないか、追加前にチェックします (明示的な場合は上で既に実施)
            self._grid.add_object(self._agent_ids[i], pos)


        # 報酬計算のための内部状態変数をリセットします
        self._goals_reached_status = [False] * self.goals_number
        # エージェントが配置された**後に**初期距離を計算します
        self._prev_total_distance_to_goals = self._calculate_total_distance_to_goals()


        # 観測 (例: グローバル状態タプル) を返します
        return self._get_observation()

    def step(self, actions: list[int]):
        """
        各エージェントのアクションを受け取り、環境を更新します。

        Args:
            actions (list[int]): 各エージェントに対するアクションのリスト。

        Returns:
            tuple: (observation, reward, done, info)
                observation (object): ステップ後の環境の観測。
                reward (float): ステップ後に受け取った報酬。
                done (bool): エピソードが終了したかどうか。
                info (dict): 追加情報 (例: デバッグ情報)。
        """
        if len(actions) != self.agents_number:
            raise ValueError(f"アクション数 ({len(actions)}) がエージェント数 ({self.agents_number}) と異なります。")

        # エージェントIDとアクションの辞書を作成
        agent_actions = {self._agent_ids[i]: actions[i] for i in range(self.agents_number)}

        # Grid クラスの衝突解決メソッドを呼び出し、位置を更新してもらう
        # このメソッド内で Grid インスタンスの_object_positionsが更新されます
        self.collision_resolver.resolve_agent_movements(agent_actions)

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
        info = {}

        return next_observation, reward, done, info


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


    def _generate_unique_positions(self, num_positions: int, existing_positions: list[PositionType], grid_size: int) -> list[PositionType]:
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


    def _place_agents_near_goals_logic(self) -> list[PositionType]:
        """
        ゴール近傍にエージェントを配置する内部ロジック。

        Returns:
            list[PositionType]: 生成されたエージェント位置のリスト。

        Raises:
            RuntimeError: ゴール位置がグリッドに設定されていない場合。
            ValueError: ゴール近傍に十分な一意な位置がない場合。
        """
        goal_positions = list(self.get_goal_positions().values())
        if not goal_positions or len(goal_positions) != self.goals_number:
            # _setup_fixed_goals が正しく呼び出されていれば、これは理想的には起こりません。
            raise RuntimeError("ゴール位置がグリッドに正しく設定されていません。")

        near_positions = set()
        radius = 2 # ゴール周辺の探索半径

        for goal_pos in goal_positions:
            for dx in range(-radius, radius + 1):
                for dy in range(-radius, radius + 1):
                    if dx == 0 and dy == 0: continue
                    nx, ny = goal_pos[0] + dx, goal_pos[1] + dy
                    pos = (nx, ny)
                    # グリッド境界チェックと、位置がゴールでないことを確認します
                    if self._grid.is_valid_position(pos) and pos not in goal_positions:
                        near_positions.add(pos)

        near_positions_list = list(near_positions)

        if len(near_positions_list) < self.agents_number:
            raise ValueError(f"{self.agents_number} 個のエージェントに対して、ゴール近傍に十分な一意な位置 ({len(near_positions_list)}) がありません。")

        return random.sample(near_positions_list, self.agents_number)


    def _get_observation(self) -> tuple[PositionType, ...]:
        """
        エージェントの観測を生成します。
        ここではグローバル状態 (全位置のタプル: ゴール、次にエージェント) を返します。
        エージェントごとの部分観測に拡張することも可能です。

        Returns:
            object: 観測。
        """
        # 全てのオブジェクト位置のタプルを、オブジェクトIDタイプ (ゴール、次にエージェント) でソートして返します
        goal_positions = [self._grid.get_object_position(goal_id) for goal_id in self._goal_ids]
        agent_positions = [self._grid.get_object_position(agent_id) for agent_id in self._agent_ids]
        return tuple(tuple(goal_positions) + tuple(agent_positions))


    def get_goal_positions(self) -> dict[str, PositionType]:
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


    def get_agent_positions(self) -> dict[str, PositionType]:
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

    def _calculate_reward(self,done_mode) -> float:
        """
        現在の状態と報酬モードに基づいて報酬を計算します。
        """
        reward = 0.0
        done = self._check_done_condition(done_mode) # 報酬計算のために完了ステータスをチェック

        agent_positions = list(self.get_agent_positions().values())
        current_agents_pos_set = set(agent_positions)
        goal_positions = list(self.get_goal_positions().values())


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

        elif self.reward_mode == 3: # うまくいくケースが見つからなかった。よって不採用
            # モード 3: 密な報酬 (距離変化) + 段階報酬 + 完了報酬
            current_total_distance = self._calculate_total_distance_to_goals()

            # 1. 距離変化報酬
            # _prev_total_distance_to_goals が初期化されているか (つまり、エージェントが配置されているか) チェックします
            if self._prev_total_distance_to_goals != float('inf'):
                distance_change = self._prev_total_distance_to_goals - current_total_distance # 距離が減ると正、増えると負
                if distance_change > 0:
                    reward +=  1.0 # 距離が減った報酬
                else:
                    reward += -1.0 # "止"または距離が増えた
                # reward += distance_change * 1.0 # スケールは調整可能
            # else: 初期状態またはエージェントが配置されていない場合、距離変化報酬なし

            # 2. 個別ゴール到達に対する段階報酬
            # goals_num がグリッド内の実際のゴール数と一致しているか確認します
            if len(goal_positions) != self.goals_number:
                print("Warning: goals_num とグリッド内の実際のゴール数に不一致があります。")
                # ゴールが変更された場合、_goals_reached_status のリセットが必要になる可能性がありますが、
                # 設計では固定ゴールとされています。

            for goal_idx, goal in enumerate(goal_positions):
                # goal_idx が _goals_reached_status の範囲内にあることを確認します
                if goal_idx < len(self._goals_reached_status):
                    if goal in current_agents_pos_set and not self._goals_reached_status[goal_idx]:
                        reward += 30.0 # 新しいゴール到達に対する報酬 (調整可能)
                        self._goals_reached_status[goal_idx] = True
                else:
                    print(f"Warning: ゴールインデックス {goal_idx} が _goals_reached_status の範囲外です。")


            # 3. 完了報酬
            if done:
                reward += 100.0 # 終了時の追加報酬 (調整可能)
            
            # 時間経過の罰則
            reward += -1.0

            # 次のステップのために現在の合計距離を更新します
            self._prev_total_distance_to_goals = current_total_distance

        else:
            print(f"Warning: 未知の報酬モード: {self.reward_mode}。報酬は 0 です。")
            reward = 0.0

        return reward

    '''
    def _check_done_condition(self) -> bool:
        """
        エピソードが終了したかチェックします (全ゴールが少なくとも1つのエージェントによって占有されているか)。
        """
        goal_positions = list(self.get_goal_positions().values())
        agent_positions = list(self.get_agent_positions().values())
        current_agents_pos_set = set(agent_positions)

        # 全てのゴール位置が現在のエージェント位置のセットに存在する場合、エピソードは終了です
        return all(goal in current_agents_pos_set for goal in goal_positions)
    '''

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
            # Rule 2: いずれかのゴールがエージェントによって占有されているか
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
        

##import random
class GoalSetter:
    def __init__(self, grid:Grid, fixed_goal=None, seed=None) -> None:
        self._grid:Grid = grid
        # 得られた結果が(1,2),(3,4)だとする。
        if not fixed_goal and seed:
            goals:tuple[PositionType, ...] = self._goal_setter_from_seed(seed)
        elif not seed and fixed_goal:
            goals:tuple[PositionType, ...] = fixed_goal

        self._goal_setter(goals)
        self._goals = goals

    #def _goal_setter(self, *goals:tuple[tuple[int,int]]):
    def _goal_setter(self, goals:tuple[PositionType, ...]):
        for id, goal in enumerate(goals):
            self._grid.add_object(f'goal_{id}',goal)

    def _goal_setter_from_seed(self,seed:int)-> tuple[PositionType, ...]:
        # ランダムに得られた結果が(1,2),(3,4)だと仮定する。
        return ((1,2),(3,4))
    
    def get_goal_pos(self):
        return self._goals


if __name__=='__main__':
    class arg_object:
        def __init__(self) -> None:
            self.grid_size=10# (int) 
            self.agents_number=2# (int) 
            self.goals_number=2# (int) 
            self.reward_mode=9# (int) 
            self.render_mode=False# (int) 
            self.window_width=400# (int) 
            self.window_height=600# (int) 
            self.pause_duration=1.0# (float)

    arg = arg_object()