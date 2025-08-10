import unittest
import os
import numpy as np
import csv

from src.Q_learn.linear import Linear
# MockArgsクラスはテストファイル内で定義を修正します

class MockArgs:
    def __init__(self):
        self.max_timestep = 100
        self.goals_number = 1 # テスト用にゴール数を調整
        self.agents_number = 2 # テスト用にエージェント数を調整
        self.gamma = 0.9
        self.learning_rate = 0.1
        self.grid_size = 5
        self.batch_size = 32
        self.load_model = 0
        self.mask = 0
        self.model_path = "dummy_model.csv"


class TestLinear(unittest.TestCase):

    def setUp(self):
        """
        各テストメソッドの実行前に呼ばれるセットアップメソッド.
        テスト用のLinearインスタンスを作成.
        """
        self.args = MockArgs()
        self.action_size = 4

        # Linearクラスのcommon_theta_listをリセット（他のテストとの干渉を防ぐため）
        # common_theta_listはクラス変数なので、各テストで独立させるためにリセットが必要
        Linear.common_theta_list = None


        # モデルパスが空になるのを防ぐため、setUpで設定
        # save/load テストで具体的なパスを使用するため、ダミーパスを設定しておく
        self.args.model_path = "test_initial_model.csv"


    def tearDown(self):
        """
        各テストメソッドの実行後に呼ばれるティアダウンメソッド.
        テストで作成したファイルを削除.
        """
        # save_model_params テストで作成したディレクトリとファイルを削除
        save_dir = "test_models" # テスト用のディレクトリ
        if os.path.exists(save_dir):
            for file_name in os.listdir(save_dir):
                file_path_to_remove = os.path.join(save_dir, file_name)
                if os.path.isfile(file_path_to_remove):
                    os.remove(file_path_to_remove)
            # ディレクトリが空になったら削除
            if not os.listdir(save_dir):
                os.rmdir(save_dir)

        # learn テストなどで作成された可能性のあるデフォルトのモデルファイルも削除
        default_model_path = self.args.model_path # MockArgsのデフォルトパス
        if os.path.exists(default_model_path):
            os.remove(default_model_path)


        # テスト後にcommon_theta_listをリセット
        Linear.common_theta_list = None


    def test_linear_learn_single_agent(self):
        """
        Linearのlearnメソッドのテスト (単一エージェント).
        経験の蓄積とエピソード終了時の一括更新が正しく行われるかを確認.
        (TD誤差の計算とパラメータ更新の基本的なロジックを確認)
        """
        args = MockArgs()
        args.agents_number = 1 # learnテスト用にエージェント数を1に設定
        args.goals_number = 1 # learnテスト用にゴール数を1に設定
        args.mask = 0 # マスクなしの場合をテスト
        # setUpでargs.model_pathは設定されているが、ここでは新しいMockArgsを作成しているので再設定
        args.model_path = "test_learn_model_single.csv"
        linear_instance = Linear(args, self.action_size)

        # テスト用の経験データを作成
        # learnメソッドは (i, states, action, reward, next_state, done, step) を引数にとる
        # states, next_state は (goals + agents) のタプル形式
        # このテストでは単一エージェント(agents_num=1)を想定

        # エージェントi = 0
        # goals_pos = [(0,0)]
        # agents_pos = [(1,1)] -> states = ((0,0), (1,1))
        state = ((0,0), (1,1))
        action = 1 # 行動1 (右)
        reward = -1.0 # 報酬
        # next_agents_pos = [(1,2)] -> next_state = ((0,0), (1,2))
        next_state = ((0,0), (1,2))
        done = False
        step = 1

        # 初回のlearn呼び出し (経験を蓄積)
        delta1 = linear_instance.learn(0, state, action, reward, next_state, done, step)
        self.assertEqual(delta1, 0.0) # エピソード終了時ではないため、deltaは0.0が返る想定

        # 経験データが蓄積されていることを確認
        self.assertEqual(len(linear_instance.episode_data), 1)

        # episode_data の各要素の構造を理解し、比較を行う
        # 経験データは (st_idx, action, agent_pos, reward, next_st_idx, next_agent_pos, done)
        expected_st_idx = linear_instance.get_index_from_states([]) # 単一エージェントなので他エージェントなし
        expected_action = action
        # Corrected: Access agent_pos from the state tuple correctly
        expected_agent_pos = np.array(state[linear_instance.goals_number + 0])
        expected_reward = reward
        expected_next_st_idx = linear_instance.get_index_from_states([]) # 単一エージェントなので他エージェントなし
        # Corrected: Access next_agent_pos from the next_state tuple correctly
        expected_next_agent_pos = np.array(next_state[linear_instance.goals_number + 0])
        expected_done = done

        exp1 = linear_instance.episode_data[0]

        self.assertEqual(expected_st_idx, exp1[0])
        np.testing.assert_array_equal(exp1[1], expected_action) # Modified to use np.testing.assert_array_equal for action
        np.testing.assert_array_equal(expected_agent_pos, exp1[2])
        np.testing.assert_array_equal(exp1[3], expected_reward) # Modified to use np.testing.assert_array_equal for reward
        self.assertEqual(expected_next_st_idx, exp1[4])
        np.testing.assert_array_equal(expected_next_agent_pos, exp1[5])
        np.testing.assert_array_equal(exp1[6], expected_done) # Modified to use np.testing.assert_array_equal for done


        # 2回目のlearn呼び出し (エピソード終了)
        state2 = next_state # 現在の状態は前回の次の状態
        action2 = 1 # 行動1 (右) -> (1,3)へ
        reward2 = -1.0 # 報酬
        next_state2 = ((0,0), (1,3)) # 次の状態
        done2 = True # エピソード終了
        step2 = 2 # 例えば2ステップで終了

        # learnメソッド実行 (エピソード終了のため更新処理が走る)
        delta_avg = linear_instance.learn(0, state2, action2, reward2, next_state2, done2, step2)

        # 経験データがクリアされていることを確認
        self.assertEqual(len(linear_instance.episode_data), 0)
        self.assertEqual(len(linear_instance.loss), 0) # lossもクリアされる

        # パラメータ(theta_list)が更新されていることを確認
        # 初期値はゼロベクトルなので、更新されていればゼロベクトルではないはず
        # (単一エージェント、マスクなしの場合 common_theta_list[0][action] が更新される)
        b = linear_instance.grid_size ** 2
        if args.mask == 0:
            # common_theta_list が更新されていることを確認
            updated_theta = Linear.common_theta_list[0][action]#type:ignore
            updated_theta2 = Linear.common_theta_list[0][action2]#type:ignore
            # 少なくともゼロベクトルではないことを確認 (厳密な値のテストは複雑なので省略)
            # 更新が行われたアクションに対応するthetaが非ゼロであることを確認
            self.assertTrue(np.any(updated_theta != 0) or np.any(updated_theta2 != 0))
        else:
            # self.theta_list が更新されていることを確認 (mask=1のテストケースを別途追加する必要あり)
            pass # TODO: mask=1 のテストケースを追加

        # 返されたdelta_avgが0より大きいことを確認 (更新が行われたことを示唆)
        self.assertGreater(delta_avg, 0.0)


    def test_linear_learn_multi_agent(self):
        """
        Linearのlearnメソッドのテスト (複数エージェント).
        複数エージェントの場合の経験の蓄積と一括更新が正しく行われるかを確認.
        """
        args = MockArgs()
        args.agents_number = 2 # エージェント数を2に設定
        args.goals_number = 1 # ゴール数を1に設定
        args.mask = 0 # マスクなしの場合をテスト
        args.model_path = "test_learn_model_multi.csv"
        linear_instance = Linear(args, self.action_size)

        # テスト用の経験データを作成 (エージェント0に焦点を当てる)
        i = 0 # 自エージェントのインデックス (エージェント0)
        # goals_pos = [(0,0)]
        # agents_pos = [(1,1), (2,2)] -> states = ((0,0), (1,1), (2,2))
        state = ((0,0), (1,1), (2,2))
        action = 1 # エージェント0の行動1 (右)
        reward = -1.0 # 報酬
        # next_agents_pos = [(1,2), (2,2)] # エージェント0が右に移動、エージェント1は静止と仮定
        next_state = ((0,0), (1,2), (2,2))
        done = False
        step = 1

        # 初回のlearn呼び出し (経験を蓄積)
        delta1 = linear_instance.learn(i, state, action, reward, next_state, done, step)
        self.assertEqual(delta1, 0.0) # エピソード終了時ではないため、deltaは0.0が返る想定

        # 経験データが蓄積されていることを確認
        self.assertEqual(len(linear_instance.episode_data), 1)

        # episode_data の各要素の構造を理解し、比較を行う
        # 経験データは (st_idx, action, agent_pos, reward, next_st_idx, next_agent_pos, done)
        # 他エージェントの状態リスト (エージェント1のみ)
        agents_pos = list(state[linear_instance.goals_number:]) # [(1,1), (2,2)]
        agents_pos_others = [agents_pos[j] for j in range(args.agents_number) if j != i] # [(2,2)]
        expected_st_idx = linear_instance.get_index_from_states(agents_pos_others)

        expected_action = action
        expected_agent_pos = np.array(agents_pos[i]) # エージェント0の位置 (1,1)
        expected_reward = reward

        next_agents_pos = list(next_state[linear_instance.goals_number:]) # [(1,2), (2,2)]
        next_agents_pos_others = [next_agents_pos[j] for j in range(args.agents_number) if j != i] # [(2,2)]
        expected_next_st_idx = linear_instance.get_index_from_states(next_agents_pos_others)

        expected_next_agent_pos = np.array(next_agents_pos[i]) # エージェント0の次の位置 (1,2)
        expected_done = done

        exp1 = linear_instance.episode_data[0]

        self.assertEqual(expected_st_idx, exp1[0])
        np.testing.assert_array_equal(exp1[1], expected_action) # Modified to use np.testing.assert_array_equal for action
        np.testing.assert_array_equal(expected_agent_pos, exp1[2])
        np.testing.assert_array_equal(exp1[3], expected_reward) # Modified to use np.testing.assert_array_equal for reward
        self.assertEqual(expected_next_st_idx, exp1[4])
        np.testing.assert_array_equal(expected_next_agent_pos, exp1[5])
        np.testing.assert_array_equal(exp1[6], expected_done) # Modified to use np.testing.assert_array_equal for done


        # 2回目のlearn呼び出し (エピソード終了)
        state2 = next_state # 現在の状態は前回の次の状態
        action2 = 1 # エージェント0の行動1 (右) -> (1,3)へ
        reward2 = -1.0 # 報酬
        next_state2 = ((0,0), (1,3), (2,2)) # 次の状態
        done2 = True # エピソード終了
        step2 = 2 # 例えば2ステップで終了

        # learnメソッド実行 (エピソード終了のため更新処理が走る)
        delta_avg = linear_instance.learn(i, state2, action2, reward2, next_state2, done2, step2)

        # 経験データがクリアされていることを確認
        self.assertEqual(len(linear_instance.episode_data), 0)
        self.assertEqual(len(linear_instance.loss), 0) # lossもクリアされる

        # パラメータ(common_theta_list)が更新されていることを確認
        # 初期値はゼロベクトルなので、更新されていればゼロベクトルではないはず
        # マスクなしの場合 common_theta_list[st_idx][action] が更新される
        # st_idx はエージェントの状態の組み合わせに依存する
        b = linear_instance.grid_size ** 2
        # エージェント1の位置 (2,2) に対応する st_idx を計算
        agents_pos2 = list(state2[linear_instance.goals_number:]) # [(1,2), (2,2)]
        agents_pos_others2 = [agents_pos2[j] for j in range(args.agents_number) if j != i] # [(2,2)]
        st_idx2 = linear_instance.get_index_from_states(agents_pos_others2) # エージェント1の状態 (2,2) のインデックス

        updated_theta_exp1 = Linear.common_theta_list[expected_st_idx][expected_action]#type:ignore
        updated_theta_exp2 = Linear.common_theta_list[st_idx2][action2]#type:ignore

        # 更新が行われたアクションに対応するthetaが非ゼロであることを確認
        self.assertTrue(np.any(updated_theta_exp1 != 0) or np.any(updated_theta_exp2 != 0))

        # 返されたdelta_avgが0より大きいことを確認 (更新が行われたことを示唆)
        self.assertGreater(delta_avg, 0.0)


    def test_linear_get_q_values_single_agent(self):
        """
        Linearのget_q_valuesメソッドのテスト (単一エージェント).
        指定された状態に対するQ値リストが正しく計算されるかを確認.
        """
        args = MockArgs()
        args.agents_number = 1 # get_q_values テスト用にエージェント数を1に設定
        args.goals_number = 1 # get_q_values テスト用にゴール数を1に設定
        args.mask = 0 # マスクなしの場合をテスト
        args.model_path = "test_get_q_values_model_single.csv"
        linear_instance = Linear(args, self.action_size)

        # テスト用の状態を作成 (自エージェントインデックス, goals_pos, agents_pos)
        # 単一エージェント (agents_num=1), 単一ゴール (goals_num=1) を想定
        i = 0 # 自エージェントのインデックス
        goals_pos = [(0,0)]
        agents_pos = [(2,2)] # 自エージェントの位置
        state_tuple = (i, goals_pos, agents_pos)

        # Linearインスタンスのtheta_list (または common_theta_list) を事前に設定
        # 初期化時は全てゼロなので、テストのために適当な値を設定する
        # agents_num=1, mask=0 の場合、common_theta_list は shape (1, action_size, b) の NumPy配列
        b = linear_instance.grid_size ** 2
        # dummy_theta は (other_agents_states_size, action_size, b) の NumPy配列
        other_agents_states_size = b**(max(0, args.agents_number - 1)) # max(0, 1-1) = 0, b^0 = 1
        dummy_theta = np.random.rand(other_agents_states_size, self.action_size, b) * 0.1

        if args.mask == 0:
            # common_theta_list に設定 (NumPy配列として設定)
            Linear.common_theta_list = dummy_theta
        else:
            # theta_list に設定 (リストのリストとして) -> NumPy配列のリストとして設定
            # マスクありの場合、theta_list の shape は (action_size, b)
            linear_instance.theta_list = [np.random.rand(b) * 0.1 for _ in range(self.action_size)]


        # get_q_values メソッドを実行
        q_values = linear_instance.get_q_values(state_tuple)

        # 期待されるQ値を手動で計算
        # get_q_values 内部では getQ を呼び出している
        # getQ(agents_pos_others, agent_pos, action)
        # agents_pos_others は単一エージェントの場合 []
        agents_pos_others = []
        agent_pos = np.array(agents_pos[i])
        phi_s = linear_instance.rbfs(agent_pos) # 自エージェント位置に対応する基底関数値

        expected_q_values = []
        for action in range(self.action_size):
            if args.mask == 0:
                # common_theta_list[0][action] (NumPy配列) と phi_s (NumPy配列) の内積
                expected_q = phi_s.dot(Linear.common_theta_list[0][action])#type:ignore
            else:
                # self.theta_list[action] (NumPy配列) と phi_s (NumPy配列) の内積
                expected_q = phi_s.dot(linear_instance.theta_list[action])
            expected_q_values.append(expected_q)

        # 計算されたQ値リストが期待値リストと一致するか確認
        np.testing.assert_array_almost_equal(q_values, expected_q_values, decimal=5)

    def test_linear_get_q_values_multi_agent(self):
        """
        Linearのget_q_valuesメソッドのテスト (複数エージェント).
        指定された状態に対するQ値リストが正しく計算されるかを確認.
        """
        args = MockArgs()
        # get_q_values テスト用にエージェント数とゴール数を設定
        args.agents_number = 2
        args.goals_number = 1
        args.mask = 0 # マスクなしの場合をテスト
        args.model_path = "test_get_q_values_model_multi.csv"
        linear_instance = Linear(args, self.action_size)

        # テスト用の状態を作成 (自エージェントインデックス, goals_pos, agents_pos)
        # エージェント数2, ゴール数1 を想定
        i = 0 # 自エージェントのインデックス
        goals_pos = [(0,0)]
        agents_pos = [(2,2), (3,3)] # 自エージェント(0)の位置(2,2), 他エージェント(1)の位置(3,3)
        state_tuple = (i, goals_pos, agents_pos)

        # Linearインスタンスのtheta_list (または common_theta_list) を事前に設定
        # 初期化時は全てゼロなので、テストのために適当な値を設定する
        # agents_num=2, mask=0 の場合、common_theta_list は shape (b**(2-1), action_size, b) = (b, action_size, b)
        b = linear_instance.grid_size ** 2 # 25
        other_agents_states_size = b**(max(0, args.agents_number - 1)) # 25^1 = 25
        # dummy_theta_to_set は (other_agents_states_size, action_size, b) の NumPy配列
        dummy_theta_to_set = np.random.rand(other_agents_states_size, self.action_size, b) * 0.1

        if args.mask == 0:
            # common_theta_list に設定 (NumPy配列として設定)
            Linear.common_theta_list = dummy_theta_to_set
        else:
            # theta_list に設定 (リストのリストとして) -> NumPy配列のリストとして設定
            # マスクありの場合、theta_list の shape は (action_size, b)
            linear_instance.theta_list = [np.random.rand(b) * 0.1 for _ in range(self.action_size)]


        # get_q_values メソッドを実行
        q_values = linear_instance.get_q_values(state_tuple)

        # 期待されるQ値を手動で計算
        # get_q_values 内部では getQ を呼び出している
        # getQ(agents_pos_others, agent_pos, action)
        # agents_pos は [(2,2), (3,3)]
        agents_pos_others = [agents_pos[j] for j in range(args.agents_number) if j != i] # [(3,3)]
        agent_pos = np.array(agents_pos[i]) # エージェント0の位置 (2,2)
        phi_s = linear_instance.rbfs(agent_pos) # 自エージェント位置に対応する基底関数値

        expected_q_values = []
        for action in range(self.action_size):
            if args.mask == 0:
                # 他エージェントの状態 [(3,3)] に対応する st_idx を計算
                st_idx = linear_instance.get_index_from_states(agents_pos_others)
                # common_theta_list[st_idx][action] (NumPy配列) と phi_s (NumPy配列) の内積
                expected_q = phi_s.dot(Linear.common_theta_list[st_idx][action])#type:ignore
            else:
                # self.theta_list[action] (NumPy配列) と phi_s (NumPy配列) の内積
                expected_q = phi_s.dot(linear_instance.theta_list[action])
            expected_q_values.append(expected_q)

        # 計算されたQ値リストが期待値リストと一致するか確認
        np.testing.assert_array_almost_equal(q_values, expected_q_values, decimal=5)

    def test_linear_save_model_params_single_agent(self):
        """
        Linearのsave_model_paramsメソッドのテスト (単一エージェント).
        モデルパラメータが正しくCSVファイルに保存されるかを確認.
        """
        args = MockArgs()
        args.agents_number = 1 # save_model_params テスト用にエージェント数を1に設定
        args.goals_number = 1 # save_model_params テスト用にゴール数を1に設定
        args.mask = 0 # マスクなしの場合をテスト
        # 保存先のファイルパス
        save_dir = "test_models" # テスト用のディレクトリ
        save_filename = "test_linear_save_single.csv"
        save_path = os.path.join(save_dir, save_filename)
        args.model_path = save_path # argsのmodel_pathも更新

        # Linearインスタンスを作成
        linear_instance = Linear(args, self.action_size)

        # 保存対象となるtheta_list (または common_theta_list) に適当な値を設定
        # 単一エージェント (agents_num=1), マスクなし (mask=0) を想定
        b = linear_instance.grid_size ** 2
        # dummy_theta_to_save は (other_agents_states_size, action_size, b) の形状になるように作成
        # 単一エージェントの場合、other_agents_states_size = b**(1-1) = b^0 = 1
        other_agents_states_size = b**(max(0, args.agents_number - 1)) # max(0, 1-1) = 0, b^0 = 1
        dummy_theta_to_save = np.random.rand(other_agents_states_size, self.action_size, b) * 10.0

        if args.mask == 0:
            # common_theta_list に設定 (NumPy配列として設定)
            Linear.common_theta_list = dummy_theta_to_save
        else:
            # theta_list に設定 (リストのリストとして) -> NumPy配列のリストとして設定
            # マスクありの場合、theta_list の shape は (action_size, b)
            linear_instance.theta_list = [np.random.rand(b) * 10.0 for _ in range(self.action_size)]


        # save_model_params メソッドを実行
        linear_instance.save_model_params(save_path)

        # ファイルが作成されたことを確認
        self.assertTrue(os.path.exists(save_path))

        # 保存されたファイルを読み込み、内容が元のデータと一致するか確認
        loaded_data = []
        with open(save_path, 'r') as f:
            reader = csv.reader(f)
            for row in reader:
                loaded_data.append(np.array(row, dtype=float))

        # ロードしたデータが元のデータ (dummy_theta_to_save) と一致するか確認
        loaded_theta_array = np.array(loaded_data)

        # common_theta_list または theta_list を比較用の単一のNumpy配列に変換
        if args.mask == 0:
            # common_theta_list を flattern して比較
            # common_theta_list の shape は (other_agents_states_size, action_size, b)
            # save_model_params はこれを (other_agents_states_size * action_size, b) の形状で保存
            # dummy_theta_to_save は shape (other_agents_states_size, action_size, b) なので
            # これを (other_agents_states_size * action_size, b) に reshape して比較
            expected_theta_flat = dummy_theta_to_save.reshape(-1, b)
        else:
            # theta_list を Numpy 配列に変換して比較
            expected_theta_flat = np.array(linear_instance.theta_list)

        # 比較対象の shape を表示
        # print(f"Shape of loaded_theta_array: {loaded_theta_array.shape}")
        # print(f"Shape of expected_theta_flat: {expected_theta_flat.shape}")

        np.testing.assert_array_almost_equal(loaded_theta_array, expected_theta_flat, decimal=5)

        # TODO: mask=1 の場合のテストケースを追加

    def test_linear_save_model_params_multi_agent(self):
        """
        Linearのsave_model_paramsメソッドのテスト (複数エージェント).
        複数エージェントの場合のモデルパラメータが正しくCSVファイルに保存されるかを確認.
        """
        args = MockArgs()
        args.agents_number = 2 # save_model_params テスト用にエージェント数を2に設定
        args.goals_number = 1 # save_model_params テスト用にゴール数を1に設定
        args.mask = 0 # マスクなしの場合をテスト
        # 保存先のファイルパス
        save_dir = "test_models" # テスト用のディレクトリ
        save_filename = "test_linear_save_multi.csv"
        save_path = os.path.join(save_dir, save_filename)
        args.model_path = save_path # argsのmodel_pathも更新

        # Linearインスタンスを作成
        linear_instance = Linear(args, self.action_size)

        # 保存対象となるtheta_list (または common_theta_list) に適当な値を設定
        # エージェント数2 (agents_num=2), マスクなし (mask=0) を想定
        b = linear_instance.grid_size ** 2 # 25
        other_agents_states_size = b**(max(0, args.agents_number - 1)) # max(0, 2-1) = 1, b^1 = 25
        # dummy_theta_to_save は (other_agents_states_size, action_size, b) の NumPy配列になるように作成
        dummy_theta_to_save = np.random.rand(other_agents_states_size, self.action_size, b) * 10.0

        if args.mask == 0:
            # common_theta_list に設定 (NumPy配列として設定)
            Linear.common_theta_list = dummy_theta_to_save # shape (25, 4, 25)
            # save_model_params は mask==0 の場合 common_theta_list を保存するため、theta_list は直接使用されない.
            pass # theta_list は設定しない
        else:
            # theta_list に設定 (リストのリストとして) -> NumPy配列のリストとして設定
            # マスクありの場合、theta_list の shape は (action_size, b)
            linear_instance.theta_list = [np.random.rand(b) * 10.0 for _ in range(self.action_size)]


        # save_model_params メソッドを実行
        linear_instance.save_model_params(save_path)

        # ファイルが作成されたことを確認
        self.assertTrue(os.path.exists(save_path))

        # 保存されたファイルを読み込み、内容が元のデータと一致するか確認
        loaded_data = []
        with open(save_path, 'r') as f:
            reader = csv.reader(f)
            for row in reader:
                loaded_data.append(np.array(row, dtype=float))

        # ロードしたデータが元のデータ (dummy_theta_to_save) と一致するか確認
        loaded_theta_array = np.array(loaded_data)

        # common_theta_list または theta_list を比較用の単一のNumpy配列に変換
        if args.mask == 0:
            # common_theta_list を flattern して比較
            # common_theta_list の shape は (other_agents_states_size, action_size, b)
            # save_model_params はこれを (other_agents_states_size * action_size, b) の形状で保存
            # dummy_theta_to_save は shape (other_agents_states_size, action_size, b) なので
            # これを (other_agents_states_size * action_size, b) に reshape して比較
            expected_theta_flat = dummy_theta_to_save.reshape(-1, b)
        else:
            # theta_list を Numpy 配列に変換して比較
            expected_theta_flat = np.array(linear_instance.theta_list)

        # 比較対象の shape を表示
        # print(f"Shape of loaded_theta_array: {loaded_theta_array.shape}")
        # print(f"Shape of expected_theta_flat: {expected_theta_flat.shape}")

        np.testing.assert_array_almost_equal(loaded_theta_array, expected_theta_flat, decimal=5)

        # TODO: mask=1 の場合のテストケースを追加


    def test_linear_get_index_from_states(self):
        """
        Linearのget_index_from_statesメソッドのテスト.
        複数エージェントの状態リストから一意なインデックスが正しく計算されるかを確認.
        """
        args = MockArgs()
        args.agents_number = 3 # エージェント数を3に設定
        args.grid_size = 5 # グリッドサイズを5に設定
        # setUpでargs.model_pathは設定されているが、ここでは新しいMockArgsを作成しているので再設定
        args.model_path = "test_get_index_model.csv"
        linear_instance = Linear(args, self.action_size)

        # テスト用の他エージェントの状態リスト
        # エージェント0に焦点を当てると、他エージェントはエージェント1とエージェント2
        # states は [[x1, y1], [x2, y2], ...] の形式
        #states_others_1 = [[1, 2], [3, 4]] # エージェント1が(1,2), エージェント2が(3,4) にいる状態
        #states_others_2 = [[3, 4], [1, 2]] # エージェント1が(3,4), エージェント2が(1,2) にいる状態 (順序が異なる)
        #states_others_3 = [[0, 0], [0, 0]] # 全て原点にいる状態
        states_others_1 = [(1, 2), (3, 4)] # エージェント1が(1,2), エージェント2が(3,4) にいる状態
        states_others_2 = [(3, 4), (1, 2)] # エージェント1が(3,4), エージェント2が(1,2) にいる状態 (順序が異なる)
        states_others_3 = [(0, 0), (0, 0)] # 全て原点にいる状態

        # 期待されるインデックスを手動で計算
        # grid_size = 5, agents_number = 3, other_agents_number = 2
        # インデックス計算は (x*grid_size + y) を基数 b = grid_size^2 = 25 として行う
        # states_others_1: [[1, 2], [3, 4]]
        # エージェント1の状態 (1,2) -> 1*5 + 2 = 7
        # エージェント2の状態 (3,4) -> 3*5 + 4 = 19
        # インデックス = (7 * b^1) + (19 * b^0) = 7 * 25 + 19 * 1 = 175 + 19 = 194
        expected_index_1_calc = 0
        b = args.grid_size**2
        for k, (x, y) in enumerate(states_others_1):
             power = b**(len(states_others_1) - k - 1)
             expected_index_1_calc += (x * args.grid_size + y) * power
        self.assertEqual(expected_index_1_calc, 194)


        # states_others_2: [[3, 4], [1, 2]]
        # エージェント1の状態 (3,4) -> 19
        # エージェント2の状態 (1,2) -> 7
        # インデックス = (19 * b^1) + (7 * b^0) = 19 * 25 + 7 * 1 = 475 + 7 = 482
        expected_index_2_calc = 0
        for k, (x, y) in enumerate(states_others_2):
             power = b**(len(states_others_2) - k - 1)
             expected_index_2_calc += (x * args.grid_size + y) * power
        self.assertEqual(expected_index_2_calc, 482)


        # states_others_3: [[0, 0], [0, 0]]
        # エージェント1の状態 (0,0) -> 0
        # エージェント2の状態 (0,0) -> 0
        # インデックス = (0 * b^1) + (0 * b^0) = 0
        expected_index_3_calc = 0
        for k, (x, y) in enumerate(states_others_3):
             power = b**(len(states_others_3) - k - 1)
             expected_index_3_calc += (x * args.grid_size + y) * power
        self.assertEqual(expected_index_3_calc, 0)


        # get_index_from_states メソッドを実行し、期待される値と比較
        calculated_index_1 = linear_instance.get_index_from_states(states_others_1)
        self.assertEqual(calculated_index_1, expected_index_1_calc)

        calculated_index_2 = linear_instance.get_index_from_states(states_others_2)
        self.assertEqual(calculated_index_2, expected_index_2_calc)

        calculated_index_3 = linear_instance.get_index_from_states(states_others_3)
        self.assertEqual(calculated_index_3, expected_index_3_calc)

        # キャッシュ機能のテスト
        # 一度計算した状態のインデックスがキャッシュされていることを確認
        self.assertIn(tuple(tuple(pos) for pos in states_others_1), linear_instance.index_cache)
        self.assertEqual(linear_instance.index_cache[tuple(tuple(pos) for pos in states_others_1)], expected_index_1_calc)#type:ignore

        # 別の状態でキャッシュが使われないことを確認 (最初の呼び出し時)
        # test_linear_learn_multi_agent でキャッシュがクリアされているため、ここではキャッシュは空になっているはず
        # または、このテストケース内でキャッシュを有効にするためにインスタンスを別途作成するか、setUp/tearDownのキャッシュクリアを調整
        # Simple test: Call a state again and check if it's in cache
        calculated_index_1_again = linear_instance.get_index_from_states(states_others_1)
        self.assertEqual(calculated_index_1_again, expected_index_1_calc)
        self.assertIn(tuple(tuple(pos) for pos in states_others_1), linear_instance.index_cache)

# Linearのテストを実行
#unittest.main(argv=['first-arg-is-ignored'], exit=False)