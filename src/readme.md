# 使用方法：
- orig_rl直下のmain.pyを実行することで使用
- 実行前にorig_rlディレクトリまでのpathを指定する必要あり
- ほとんどのハイパーパラメータはmain.pyのparse_args()で管理

# 主要導入パッケージ一覧：
|Package                 |Version|
|------------------------|-------|
|matplotlib|                3.8.3|
|numpy     |                1.26.4|
|pandas    |                2.2.1|
|pygame    |                2.5.2|
|Python    |                3.10.0|
|torch     |                2.2.1|


# クラス詳細
## `Grid` クラスドキュメント

`Grid` クラスは、マルチエージェント環境のグリッド空間を管理するためのクラスです。グリッドサイズ、グリッド上のオブジェクト（エージェントやゴールなど）の位置、およびオブジェクトの移動や衝突に関する基本的なロジックを扱います。

### 属性

- **`grid_size`**: グリッドの一辺のサイズ（整数）。グリッドは `grid_size` x `grid_size` の正方形です。
- **`_object_positions`**: グリッド上のオブジェクトの位置を保持する辞書。キーはオブジェクトのID（文字列）、値はグリッド上の座標タプル `(x, y)` です。

### メソッド

- **`__init__(self, grid_size)`**: `Grid` クラスの新しいインスタンスを初期化します。指定されたグリッドサイズで空のグリッドを作成します。
- **`add_object(self, obj_id, position)`**: 指定されたIDと位置にオブジェクトをグリッドに追加します。位置が有効でない場合や、同じIDのオブジェクトが既に存在する場合はエラーが発生します。
- **`get_object_position(self, obj_id)`**: 指定されたIDのオブジェクトの現在位置を返します。オブジェクトが存在しない場合は `KeyError` が発生します。
- **`set_object_position(self, obj_id, new_position)`**: 指定されたIDのオブジェクトの位置を新しい位置に更新します。オブジェクトが存在しない場合や、新しい位置が有効でない場合はエラーが発生します。
- **`is_valid_position(self, position)`**: 指定された位置 `(x, y)` がグリッドの有効な範囲内にあるか（0 <= x < grid_size かつ 0 <= y < grid_size）を判定し、ブール値を返します。
- **`is_position_occupied(self, position, exclude_obj_id=None)`**: 指定された位置が他のオブジェクトによって占有されているかを判定し、ブール値を返します。`exclude_obj_id` が指定された場合、そのIDのオブジェクト自身は占有チェックから除外されます。
- **`calculate_next_position(self, current_position, action)`**: 現在位置と行動（0: UP, 1: DOWN, 2: LEFT, 3: RIGHT, 4: STAY）に基づいて、境界チェックを行う前の潜在的な次の位置を計算します。
- **`resolve_agent_movements(self, agent_actions)`**: 複数のエージェントの行動を受け取り、衝突解決ルール（現在の位置または移動先の位置が他のエージェントによって占有されている場合は移動しない）に基づいて、各エージェントの最終的な移動先を決定し、グリッド上のエージェントの位置を更新します。最終的な位置の辞書を返します。
- **`get_all_object_positions(self)`**: グリッド上の全てのオブジェクトの現在位置を、オブジェクトIDをキーとする辞書で返します。
- **`remove_object(self, obj_id)`**: 指定されたIDのオブジェクトをグリッドから削除します。

`Grid` クラスは、環境の物理的な空間とオブジェクトの基本的な相互作用をモデル化する役割を担います。

## `MultiAgentGridEnv` クラスドキュメント (詳細)

`MultiAgentGridEnv` クラスは、マルチエージェント強化学習タスクのための OpenAI Gym/Gymnasium スタイルの環境です。エージェントとゴールの配置、行動の受け付け、状態遷移、報酬計算、およびエピソードの終了管理を行います。

### 属性

- **`grid_size`**: グリッドの一辺のサイズ。
- **`agents_num`**: 環境内のエージェントの数。
- **`goals_num`**: 環境内のゴールの数。
- **`reward_mode`**: 報酬計算モード（0, 1, 2, 3）。
- **`render_mode`**: レンダリングモード（0: 無効, 1: 有効）。
- **`window_width`, `window_height`**: レンダリングウィンドウのサイズ。
- **`pause_duration`**: レンダリング時のポーズ時間。
- **`_grid`**: `Grid` クラスのインスタンス。
- **`_agent_ids`**: エージェントのIDのリスト。
- **`_goal_ids`**: ゴールのIDのリスト。
- **`action_space`**: エージェントの行動空間（離散値）。
- **`observation_space`**: 環境の観測空間（各オブジェクトの位置のボックス）。
- **`_goals_reached_status`**: 各ゴールが到達されたかを示すブール値のリスト。
- **`_prev_total_distance_to_goals`**: 前ステップにおける、到達済みでないゴールまでの合計距離。

### メソッド

- **`__init__(self, config: GridWorldConfig)`**: 環境を初期化します。グリッド、エージェント、ゴールを生成し、行動空間と観測空間を設定します。
- **`reset(self, seed=None, options=None, placement_mode='random', initial_agent_positions=None)`**: 環境を初期状態にリセットし、初期観測を返します。エージェントとゴールは指定されたモードまたは位置に基づいて再配置されます。
- **`step(self, actions)`**: 環境を1ステップ進めます。各エージェントの行動を受け取り、グリッド上での移動を解決し、新しい状態、報酬、完了フラグ、および情報辞書を返します。
- **`render(self)`**: 現在の環境状態を視覚化します。
- **`close(self)`**: レンダリングウィンドウやその他のリソースを閉じます。
- **`_get_observation(self)`**: 環境の現在の観測（ゴール位置とエージェント位置のタプル）を生成します。
- **`_calculate_reward(self)`**: 現在の環境状態に基づいて報酬を計算します。`reward_mode` に応じた計算ロジックが含まれます。
- **`_check_done_condition(self)`**: 全てのゴールがエージェントによって到達されたかを確認し、エピソードの完了条件を判定します。
- **`_generate_unique_positions(self, num_positions, occupied_positions, grid_size)`**: 指定された数のユニークなグリッド位置を生成する内部ヘルパーメソッド。
- **`get_agent_positions(self)`**: 現在のエージェントのIDと位置の辞書を返します。
- **`get_goal_positions(self)`**: 現在のゴールのIDと位置の辞書を返します。
- **`_calculate_total_distance_to_goals(self)`**: 全てのエージェントから、まだ到達されていない対応するゴールまでの合計マンハッタン距離を計算します。

このクラスは、強化学習アルゴリズムが相互作用するためのインターフェースを提供し、グリッドワールド環境の動的な挙動をカプセル化します。


# テストクラス実装詳細ノート

## `TestGrid` クラスドキュメント

`TestGrid` クラスは、グリッドベースの環境におけるオブジェクトの配置、移動、衝突解決を管理する `Grid` クラスの単体テストスイートです。`unittest` フレームワークを使用して記述されており、`Grid` クラスの様々な機能が期待通りに動作するか検証します。

### テストメソッド

- **`setUp(self)`**: 各テストメソッドの実行前に呼び出され、テスト用の `Grid` インスタンスを初期化します。
- **`test_initialization(self)`**: `Grid` クラスが正しく初期化されるか（グリッドサイズ、オブジェクト位置リストの初期状態など）をテストします。
- **`test_add_and_get_object(self)`**: オブジェクトをグリッドに追加し、その位置を正しく取得できるかをテストします。
- **`test_get_nonexistent_object_raises_keyerror(self)`**: 存在しないオブジェクトの位置を取得しようとしたときに `KeyError` が発生することをテストします。
- **`test_add_object_out_of_bounds_raises_valueerror(self)`**: グリッドの範囲外にオブジェクトを追加しようとしたときに `ValueError` が発生することをテストします。
- **`test_add_duplicate_object_id_raises_valueerror(self)`**: 同じIDを持つオブジェクトを複数追加しようとしたときに `ValueError` が発生することをテストします。
- **`test_is_valid_position(self)`**: 指定された位置がグリッドの有効な範囲内であるかを判定する `is_valid_position` メソッドをテストします。
- **`test_is_position_occupied(self)`**: 指定された位置が他のオブジェクトによって占有されているかを判定する `is_position_occupied` メソッドをテストします。除外IDの機能もテストします。
- **`test_calculate_next_position(self)`**: 現在位置と行動に基づいて、次のグリッド位置を計算する `calculate_next_position` メソッドをテストします。境界チェックは行われないことを確認します。
- **`test_set_object_position(self)`**: 既存のオブジェクトの位置を更新する `set_object_position` メソッドをテストします。存在しないオブジェクトや範囲外の位置への設定に対するエラーハンドリングもテストします。
- **`test_resolve_agent_movements_single_agent(self)`**: 単一エージェントの移動と、グリッド境界での移動解決をテストします。
- **`test_resolve_agent_movements_multiple_agents_no_collision(self)`**: 複数エージェントが衝突せずに移動するシナリオをテストします。
- **`test_resolve_agent_movements_head_on_collision(self)`**: エージェントが正面衝突するシナリオで、両エージェントが移動しないことをテストします。
- **`test_resolve_agent_movements_move_to_occupied_spot(self)`**: エージェントが別のエージェントの現在位置に移動しようとするシナリオで、移動が阻止されることをテストします。
- **`test_resolve_agent_movements_nonexistent_agent_raises_keyerror(self)`**: 存在しないエージェントの行動を指定した場合に `KeyError` が発生することをテストします。

これらのテストは、`Grid` クラスがマルチエージェント環境シミュレーションの基盤として堅牢であることを保証するために役立ちます。

## `MultiAgentGridEnv` クラスドキュメント

`MultiAgentGridEnv` クラスは、Reinforcement Learning 環境としてのマルチエージェントグリッドワールドを実装します。Gymnasium（旧 Gym）ライブラリの環境インターフェースに準拠することを目指しており、エージェントとゴールの配置、ステップ実行、報酬計算、およびエピソード完了条件のチェックを扱います。

### クラス構造と設定

コンストラクタは `GridWorldConfig` オブジェクトを受け取り、環境のパラメータ（グリッドサイズ、エージェント数、ゴール数、報酬モード、レンダリングモードなど）を設定します。

内部的には、`Grid` クラスのインスタンスを保持し、オブジェクト（エージェントとゴール）の位置を管理します。

### メソッド

- **`__init__(self, config: GridWorldConfig)`**: 環境を初期化します。指定された設定に基づいてグリッド、エージェント、ゴールを生成し、初期状態を設定します。
- **`reset(self, seed=None, options=None, placement_mode='random', initial_agent_positions=None)`**: 環境を初期状態にリセットします。
    - `seed`: 乱数シードを設定します。
    - `options`: 追加のオプション（現在の実装では未使用）。
    - `placement_mode`: エージェントの初期配置モードを指定します（'random' または 'near_goals'）。
    - `initial_agent_positions`: 明示的にエージェントの初期位置を指定する場合に使用します。
    - 返り値: 初期観測と情報辞書（現在の実装では空の辞書）。
- **`step(self, actions)`**: 環境を1ステップ進めます。
    - `actions`: 各エージェントの行動のリスト（例: `[0, 3]` はエージェント0がUP、エージェント1がRIGHTに移動）。行動は Grid クラスで定義されたものを使用します (0: UP, 1: DOWN, 2: LEFT, 3: RIGHT, 4: STAY)。
    - 返り値: `(observation, reward, done, info)` のタプル。
        - `observation`: 環境の現在の観測（ゴール位置とエージェント位置のタプル）。
        - `reward`: このステップで得られた報酬。
        - `done`: エピソードが完了したかを示すブール値。
        - `info`: 追加情報を含む辞書（現在の実装では空の辞書）。
- **`render(self)`**: 環境を視覚化します（レンダリングモードが有効な場合）。Pygame を使用してグリッドとオブジェクトを描画します。
- **`close(self)`**: 環境のリソース（Pygame ウィンドウなど）を解放します。
- **`_get_observation(self)`**: 現在の環境状態から観測を生成します。ゴール位置とエージェント位置のタプルを返します。
- **`_calculate_reward(self)`**: 現在の環境状態に基づいて報酬を計算します。計算方法は `reward_mode` によって異なります。
    - モード 0: 全てのゴールに到達していれば +100、そうでなければ 0。
    - モード 1: 全てのゴールに到達していれば +500、そうでなければ -5。
    - モード 2: 全てのエージェントから全ての到達済みゴールまでの合計距離の負の値。
    - モード 3 (Dense Reward): 前ステップからの合計距離の変化（近づけばプラス、遠ざかればマイナス）+ 新たにゴールに到達したエージェントごとの報酬 + 全てのゴールに到達した場合の完了報酬。
- **`_check_done_condition(self)`**: 全てのエージェントがそれぞれのゴールに到達したかを確認し、エピソードの完了条件をチェックします。
- **`_generate_unique_positions(self, num_positions, occupied_positions, grid_size)`**: 指定された数のユニークなグリッド位置を生成します。既存の占有位置と重複しないようにします。
- **`get_agent_positions(self)`**: 現在のエージェントの位置をエージェントIDをキーとする辞書で返します。
- **`get_goal_positions(self)`**: 現在のゴールの位置をゴールIDをキーとする辞書で返します。
- **`_calculate_total_distance_to_goals(self)`**: 全てのエージェントからそれぞれの対応するゴールまでの合計マンハッタン距離を計算します。ゴールに到達済みのエージェントは距離0として扱います。

### 使用例

典型的な使用例は以下のようになります。

```
# MultiAgentGridEnvの使用例

# 環境設定クラスの定義 (以前のセルで定義されていると仮定)
# class GridWorldConfig:
#     def __init__(self):
#         self.grid_size = 10
#         self.agents_number = 2
#         self.goals_num = 2
#         self.reward_mode = 3
#         self.render_mode = 0 # テストのためレンダリングを無効に
#         self.window_width = 800
#         self.window_height = 800
#         self.pause_duration = 0.1

# 環境の初期化
config = GridWorldConfig()
env = MultiAgentGridEnv(config)

# 環境のリセット
observation = env.reset()
print("初期観測:", observation)

# 適当な行動を生成 (例: 全てのエージェントが右に移動)
actions = [3] * env.agents_num
print("実行する行動:", actions)

# 環境を1ステップ進める
next_observation, reward, done, info = env.step(actions)
print("次観測:", next_observation)
print("報酬:", reward)
print("完了:", done)
print("情報:", info)

# エピソードが完了するまでステップを繰り返す例
# done = False
# while not done:
#     # ランダムな行動を選択する例
#     random_actions = [env.action_space.sample() for _ in range(env.agents_num)]
#     next_observation, reward, done, info = env.step(random_actions)
#     print("ステップ実行 - 報酬:", reward, "完了:", done)

# 環境のリソースを解放
env.close()

print("使用例の実行が完了しました。")
```

