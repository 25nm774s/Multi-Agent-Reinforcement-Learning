import unittest
import torch

from src.DQN.dqn import DQNModel

class TestDQNModel(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        # mask=True の DQNModel インスタンスを生成
        params_mask_true = {
            'optimizer_type': 'Adam',
            'gamma': 0.99,
            'batch_size': 4,
            'agent_num': 2,
            'goals_num': 3,
            'load_model': 0, # 標準DQNモードをテスト
            'learning_rate': 0.001,
            'device':'cpu',
            'mask': True # mask有効でテスト
        }
        cls.dqn_model_mask_true = DQNModel(**params_mask_true)

        # mask=False の DQNModel インスタンスを生成
        params_mask_false = {
            'optimizer_type': 'Adam',
            'gamma': 0.99,
            'batch_size': 4,
            'agent_num': 2, # maskがFalseの場合、agent_numとgoals_numは状態サイズに関係ない
            'goals_num': 3,
            'load_model': 0,
            'learning_rate': 0.001,
            'device':'cpu',
            'mask': False # mask無効でテスト
        }
        cls.dqn_model_mask_false = DQNModel(**params_mask_false)


    def test_huber_loss(self):
        # このテストには dqn_model_mask_true インスタンスを使用
        dqn_model = self.dqn_model_mask_true

        # テストケース 1: 誤差が delta より小さい場合 (L2損失を期待)
        q1 = torch.tensor([1.0, 2.0, 3.0])
        target1 = torch.tensor([1.1, 2.2, 3.3])
        err1 = target1 - q1
        abs_err1 = torch.abs(err1)
        # このバッチに特化した delta を計算
        if len(err1) > 1 and not torch.isnan(torch.std(err1)):
            huber_loss_delta1 = torch.mean(abs_err1).item() + torch.std(err1).item()
        else:
            huber_loss_delta1 = torch.mean(abs_err1).item()

        # delta が小さすぎないことを確認
        if huber_loss_delta1 < 1e-6:
            huber_loss_delta1 = 1.0
        # 期待される Huber 損失を手動で計算 (誤差 < delta の場合は L2 損失)
        expected_loss1 = torch.mean(0.5 * torch.square(err1)).item()
        calculated_loss1 = dqn_model.huber_loss(q1, target1).item()
        self.assertAlmostEqual(calculated_loss1, expected_loss1, places=6)

        # テストケース 2: 誤差が delta より大きい場合 (L1損失を期待)
        q2 = torch.tensor([1.0, 2.0, 3.0])
        target2 = torch.tensor([5.0, 7.0, 9.0])
        err2 = target2 - q2
        abs_err2 = torch.abs(err2)
        # このバッチに特化した delta を計算
        if len(err2) > 1 and not torch.isnan(torch.std(err2)):
            huber_loss_delta2 = torch.mean(abs_err2).item() + torch.std(err2).item()
        else:
            huber_loss_delta2 = torch.mean(abs_err2).item()
        # delta が小さすぎないことを確認
        if huber_loss_delta2 < 1e-6:
            huber_loss_delta2 = 1.0

        # デバッグ用: 中間値を出力
        #print("\n--- Debugging test_huber_loss Case 2 ---")
        #print(f"Errors (err2): {err2}")
        #print(f"Absolute Errors (abs_err2): {abs_err2}")
        #print(f"Calculated Delta (huber_loss_delta2): {huber_loss_delta2}")

        # 期待される Huber 損失を手動で計算 (誤差 >= delta の場合は L1損失、そうでなければL2損失)
        cond2 = torch.abs(err2) < huber_loss_delta2
        L2_2 = 0.5 * torch.square(err2)
        L1_2 = huber_loss_delta2 * (torch.abs(err2) - 0.5 * huber_loss_delta2)
        expected_loss2 = torch.mean(torch.where(cond2, L2_2, L1_2)).item()

        calculated_loss2 = dqn_model.huber_loss(q2, target2).item()

        # デバッグ用
        #print(f"Expected Loss (expected_loss2): {expected_loss2}")
        #print(f"Calculated Loss (calculated_loss2): {calculated_loss2}")
        #print("--- End Debugging ---")

        self.assertAlmostEqual(calculated_loss2, expected_loss2, places=6)

        # テストケース 3: 誤差が混在する場合 (delta より小さいものと大きいもの)
        q3 = torch.tensor([1.0, 2.0, 3.0, 4.0])
        target3 = torch.tensor([1.1, 2.2, 8.0, 9.0])
        err3 = target3 - q3
        abs_err3 = torch.abs(err3)
        # このバッチに特化した delta を計算
        if len(err3) > 1 and not torch.isnan(torch.std(err3)):
            huber_loss_delta3 = torch.mean(abs_err3).item() + torch.std(err3).item()
        else:
            huber_loss_delta3 = torch.mean(abs_err3).item()
        # delta が小さすぎないことを確認
        if huber_loss_delta3 < 1e-6:
            huber_loss_delta3 = 1.0
        # 期待される Huber 損失を手動で計算
        cond3 = torch.abs(err3) < huber_loss_delta3
        L2_3 = 0.5 * torch.square(err3)
        L1_3 = huber_loss_delta3 * (torch.abs(err3) - 0.5 * huber_loss_delta3)
        expected_loss3 = torch.mean(torch.where(cond3, L2_3, L1_3)).item()
        calculated_loss3 = dqn_model.huber_loss(q3, target3).item()
        self.assertAlmostEqual(calculated_loss3, expected_loss3, places=6)

        # テストケース 4: バッチサイズが 1 の場合
        q4 = torch.tensor([5.0])
        target4 = torch.tensor([5.5])
        err4 = target4 - q4
        abs_err4 = torch.abs(err4)
        # バッチサイズ 1 の場合の delta を計算
        if len(err4) > 1 and not torch.isnan(torch.std(err4)):
            huber_loss_delta4 = torch.mean(abs_err4).item() + torch.std(err4).item()
        else:
            huber_loss_delta4 = torch.mean(abs_err4).item()
        # delta が小さすぎないことを確認
        if huber_loss_delta4 < 1e-6:
            huber_loss_delta4 = 1.0
        # 期待される Huber 損失を手動で計算 (誤差 < delta の場合は L2 損失)
        expected_loss4 = torch.mean(0.5 * torch.square(err4)).item()
        calculated_loss4 = dqn_model.huber_loss(q4, target4).item()
        self.assertAlmostEqual(calculated_loss4, expected_loss4, places=6)

        # テストケース 5: バッチサイズが 1 で、誤差が delta より大きい場合 (delta は |誤差| となる)
        q5 = torch.tensor([5.0])
        target5 = torch.tensor([10.0])
        err5 = target5 - q5
        abs_err5 = torch.abs(err5)
        # バッチサイズ 1 の場合の delta を計算
        if len(err5) > 1 and not torch.isnan(torch.std(err5)):
            huber_loss_delta5 = torch.mean(abs_err5).item() + torch.std(err5).item()
        else:
            huber_loss_delta5 = torch.mean(abs_err5).item()
        # delta が小さすぎないことを確認
        if huber_loss_delta5 < 1e-6:
            huber_loss_delta5 = 1.0
        # 期待される Huber 損失を手動で計算 (誤差 >= delta の場合は L1 損失)
        # delta が |誤差| の場合、|err| - 0.5 * delta は |err| - 0.5 * |err| = 0.5 * |err| となる
        # L1 損失は delta * (0.5 * |err|) = |err| * 0.5 * |err| = 0.5 * |err|^2
        # これは delta が |誤差| の場合のバッチサイズ 1 の L2 損失と同じになる
        expected_loss5 = torch.mean(0.5 * torch.square(err5)).item()
        calculated_loss5 = dqn_model.huber_loss(q5, target5).item()
        self.assertAlmostEqual(calculated_loss5, expected_loss5, places=6)

    def test_extract_agent_state(self):
        batch_size = self.dqn_model_mask_true.batch_size
        agents_num = self.dqn_model_mask_true.agents_num
        goals_num = self.dqn_model_mask_true.goals_num
        global_state_dim_masked = (agents_num + goals_num) * 2
        global_state_dim_unmasked = 2 # mask が False の場合、入力サイズはエージェント自身の状態 (x, y) のみ

        # テストケース 1: mask = True
        # 抽出を検証するために異なる値を持つモックのグローバル状態テンソルを作成
        # 構造: [goal1_x, goal1_y, ..., goalN_x, goalN_y, agent1_x, agent1_y, ..., agentM_x, agentM_y]
        mock_global_states_masked = torch.arange(batch_size * global_state_dim_masked, dtype=torch.float32).reshape(batch_size, global_state_dim_masked)
        mock_next_global_states_masked = torch.arange(batch_size * global_state_dim_masked, dtype=torch.float32).reshape(batch_size, global_state_dim_masked) + 1000 # 次の状態には異なる値を使用

        # エージェント 0 (最初のエージェント) の抽出をテスト
        agent_index_0 = 0
        agent_states_batch_0, next_agent_states_batch_0 = self.dqn_model_mask_true._extract_agent_state(
            agent_index_0, mock_global_states_masked, mock_next_global_states_masked
        )

        # 形状をアサート
        self.assertEqual(agent_states_batch_0.shape, (batch_size, 2))
        self.assertEqual(next_agent_states_batch_0.shape, (batch_size, 2))

        # エージェント 0 の値をアサート
        # エージェント 0 の状態は全てのゴール状態の後に始まる
        agent0_start_idx = goals_num * 2 + agent_index_0 * 2
        expected_agent_states_0 = mock_global_states_masked[:, agent0_start_idx : agent0_start_idx + 2]
        expected_next_agent_states_0 = mock_next_global_states_masked[:, agent0_start_idx : agent0_start_idx + 2]

        torch.testing.assert_close(agent_states_batch_0, expected_agent_states_0)
        torch.testing.assert_close(next_agent_states_batch_0, expected_next_agent_states_0)

        # エージェント 1 (2番目のエージェント) の抽出をテスト
        agent_index_1 = 1
        agent_states_batch_1, next_agent_states_batch_1 = self.dqn_model_mask_true._extract_agent_state(
            agent_index_1, mock_global_states_masked, mock_next_global_states_masked
        )

        # 形状をアサート
        self.assertEqual(agent_states_batch_1.shape, (batch_size, 2))
        self.assertEqual(next_agent_states_batch_1.shape, (batch_size, 2))

        # エージェント 1 の値をアサート
        agent1_start_idx = goals_num * 2 + agent_index_1 * 2
        expected_agent_states_1 = mock_global_states_masked[:, agent1_start_idx : agent1_start_idx + 2]
        expected_next_agent_states_1 = mock_next_global_states_masked[:, agent1_start_idx : agent1_start_idx + 2]

        torch.testing.assert_close(agent_states_batch_1, expected_agent_states_1)
        torch.testing.assert_close(next_agent_states_batch_1, expected_next_agent_states_1)


        # テストケース 2: mask = False
        # mask が False の場合のモックのグローバル状態テンソル。期待される入力サイズは 2 (エージェント自身の x, y)。
        # メソッドシグネチャは 'global_states_batch' を引数にとるが、mask=False の場合は、入力テンソルが既にエージェント自身の状態を表していると想定される。
        mock_global_states_unmasked = torch.arange(batch_size * global_state_dim_unmasked, dtype=torch.float32).reshape(batch_size, global_state_dim_unmasked) + 2000
        mock_next_global_states_unmasked = torch.arange(batch_size * global_state_dim_unmasked, dtype=torch.float32).reshape(batch_size, global_state_dim_unmasked) + 3000

        # mask が False の場合、エージェントインデックスは無視されるが、引数として渡す必要はある。
        agent_index_ignored = 0

        agent_states_batch_unmasked, next_agent_states_batch_unmasked = self.dqn_model_mask_false._extract_agent_state(
            agent_index_ignored, mock_global_states_unmasked, mock_next_global_states_unmasked
        )

        # 形状をアサート - 入力形状と同じであるはず
        self.assertEqual(agent_states_batch_unmasked.shape, (batch_size, global_state_dim_unmasked))
        self.assertEqual(next_agent_states_batch_unmasked.shape, (batch_size, global_state_dim_unmasked))

        # 値をアサート - 入力テンソルと同じであるはず
        torch.testing.assert_close(agent_states_batch_unmasked, mock_global_states_unmasked)
        torch.testing.assert_close(next_agent_states_batch_unmasked, mock_next_global_states_unmasked)

    def test_calculate_q_values(self):
        # このテストには dqn_model_mask_true インスタンスを使用
        dqn_model = self.dqn_model_mask_true
        batch_size = dqn_model.batch_size
        # mask=True の場合の入力サイズを使用
        input_size = (dqn_model.agents_num + dqn_model.goals_num) * 2

        # 1. モックのエージェント状態バッチテンソルを作成
        # 形状: (batch_size, input_size)
        mock_agent_states_batch = torch.randn(batch_size, input_size)

        # 2. モックの行動バッチテンソルを作成
        # 行動は 0 から action_size - 1 の整数値である必要がある
        mock_action_batch = torch.randint(0, dqn_model.action_size, (batch_size,))

        # 3. QNet の forward パスから返されるモックの予測 Q 値テンソルを作成
        # 形状: (batch_size, action_size)
        mock_predicted_qs_batch = torch.randn(batch_size, dqn_model.action_size, requires_grad=True)

        # 4. 一時的に qnet の forward パスを置き換える
        # 後で元に戻すために元の forward メソッドを保存
        original_qnet_forward = dqn_model.qnet.forward

        # qnet の forward メソッドをモック関数に置き換える
        def mock_forward(x):
            # オプション: 入力が mock_agent_states_batch であることをアサート
            # torch.testing.assert_close(x, mock_agent_states_batch)
            return mock_predicted_qs_batch

        dqn_model.qnet.forward = mock_forward

        # 5. _calculate_q_values メソッドを呼び出す
        calculated_q_values = dqn_model._calculate_q_values(mock_agent_states_batch, mock_action_batch)

        # 6. 期待される Q 値を手動で計算する
        # バッチ内の各サンプルで取られた行動に対応する Q 値を選択
        batch_indices = torch.arange(batch_size)
        expected_q_values = mock_predicted_qs_batch[batch_indices, mock_action_batch]

        # 7. 計算された Q 値と期待される Q 値が一致することをアサート
        torch.testing.assert_close(calculated_q_values, expected_q_values)

        # 8. 元の qnet の forward メソッドを復元
        dqn_model.qnet.forward = original_qnet_forward

    def test_calculate_next_max_q_values(self):
        # このテストには dqn_model_mask_true インスタンスを使用
        dqn_model = self.dqn_model_mask_true
        batch_size = dqn_model.batch_size
        # mask=True の場合の入力サイズを使用
        input_size = (dqn_model.agents_num + dqn_model.goals_num) * 2
        action_size = dqn_model.action_size

        # 1. モックの次のエージェント状態バッチテンソルを作成
        mock_next_agent_states_batch = torch.randn(batch_size, input_size)

        # 2. ターゲット Q ネットワークのモックの Q 値を定義済みテンソルとして作成
        # 形状は (batch_size, action_size) である必要がある
        mock_next_predicted_qs_target = torch.randn(batch_size, action_size, requires_grad=True)

        # 3. 一時的に qnet_target の forward パスを置き換える
        # 後で元に戻すために元の forward メソッドを保存
        original_qnet_target_forward = dqn_model.qnet_target.forward

        # qnet_target の forward メソッドをモック関数に置き換える
        def mock_target_forward(x):
            # オプション: 入力が mock_next_agent_states_batch であることをアサート
            # torch.testing.assert_close(x, mock_next_agent_states_batch)
            return mock_next_predicted_qs_target

        dqn_model.qnet_target.forward = mock_target_forward

        # 4. _calculate_next_max_q_values メソッドを呼び出す
        calculated_next_max_q_values = dqn_model._calculate_next_max_q_values(mock_next_agent_states_batch)

        # 5. 期待される最大 Q 値を手動で計算する
        # バッチ内の各アイテムについて、行動次元 (dim=1) に沿った最大値を見つける
        # テスト対象のメソッドが detach() するように、結果を detach() することを忘れない
        expected_next_max_q_values = mock_next_predicted_qs_target.max(1)[0].detach()

        # 6. 計算された最大 Q 値が期待される最大 Q 値と一致することをアサート
        torch.testing.assert_close(calculated_next_max_q_values, expected_next_max_q_values)

        # 7. 元の qnet_target の forward メソッドを復元
        dqn_model.qnet_target.forward = original_qnet_target_forward

    def test_calculate_target_q_values(self):
        # このテストには dqn_model_mask_true インスタンスを使用
        dqn_model = self.dqn_model_mask_true
        batch_size = dqn_model.batch_size

        # 2. reward_batch, done_batch, next_max_q_values のモックテンソルを定義
        # テンソルが同じバッチサイズであることを確認
        reward_batch = torch.randn(batch_size)
        # done_batch はブール値または float にキャスト可能なテンソルを含む必要がある
        # 例: not done (0) と done (1) の混合
        done_batch_bool = torch.randint(0, 2, (batch_size,), dtype=torch.bool)
        # あるいは float として: done_batch_float = done_batch_bool.float()
        next_max_q_values = torch.randn(batch_size) # _calculate_next_max_q_values からの値

        # 4. ベルマン方程式を使用して期待されるターゲット Q 値を手動で計算する
        # ターゲット Q = 報酬 + ガンマ * MaxQ(次の状態) * (1 - done)
        # モデルのガンマを使用
        expected_target_q_values = reward_batch + (1 - done_batch_bool.float()) * dqn_model.gamma * next_max_q_values

        # 5. モック入力テンソルを使用して _calculate_target_q_values メソッドを呼び出す
        calculated_target_q_values = dqn_model._calculate_target_q_values(reward_batch, done_batch_bool, next_max_q_values)

        # 6. torch.testing.assert_close を使用して計算値と期待値を比較
        torch.testing.assert_close(calculated_target_q_values, expected_target_q_values)

    def test_optimize_network(self):
        # このテストには dqn_model_mask_true インスタンスを使用
        dqn_model = self.dqn_model_mask_true

        # 1. モックの損失テンソルを作成
        # requires_grad=True が必要
        mock_loss = torch.tensor(123.45, requires_grad=True)

        # 2. オプティマイザのメソッドをモックする
        # side_effect として何も指定しないか、Pass (Lambda関数など) を使用できる
        # unittest.mock をインポートする必要がある
        from unittest.mock import MagicMock

        original_optimizer = dqn_model.optimizer
        mock_optimizer = MagicMock()
        dqn_model.optimizer = mock_optimizer

        # 3. 損失テンソルの backward メソッドをモックする
        # 後で元に戻すために元のメソッドを保存
        original_loss_backward = mock_loss.backward
        mock_loss.backward = MagicMock()

        # 4. _optimize_network メソッドを呼び出す
        dqn_model._optimize_network(mock_loss)

        # 5. 期待されるメソッドが呼ばれたことをアサート
        mock_optimizer.zero_grad.assert_called_once()
        mock_loss.backward.assert_called_once()
        mock_optimizer.step.assert_called_once()

        # 6. 元のメソッドを復元
        dqn_model.optimizer = original_optimizer
        mock_loss.backward = original_loss_backward

    def test_sync_qnet(self):
        # このテストには dqn_model_mask_true インスタンスを使用
        dqn_model = self.dqn_model_mask_true

        # 1. メイン Q ネットワークとターゲット Q ネットワークの状態辞書が異なることを確認
        # 初期化後、通常は異なる (ランダムに初期化されるため)
        # 確認のために、一方の状態を少し変更することも可能
        initial_qnet_state = dqn_model.qnet.state_dict()
        initial_qnet_target_state = dqn_model.qnet_target.state_dict()

        # 最初は状態辞書が同じでないことを確認 (ランダム初期化に依存するが、通常は異なる)
        # より確実なテストのために、qnet の状態を任意の値で更新することもできる
        # 例: for param in dqn_model.qnet.parameters(): param.data.fill_(1.0)
        # initial_qnet_state_modified = dqn_model.qnet.state_dict()
        # states_are_different_initially = any(not torch.equal(initial_qnet_state_modified[key], initial_qnet_target_state[key]) for key in initial_qnet_state_modified)
        # self.assertTrue(states_are_different_initially, "Initial state dictionaries should be different for a meaningful test")


        # 2. sync_qnet メソッドを呼び出す
        dqn_model.sync_qnet()

        # 3. ターゲット Q ネットワークの状態辞書がメイン Q ネットワークの状態辞書と同じであることをアサート
        synced_qnet_target_state = dqn_model.qnet_target.state_dict()
        qnet_state_after_sync = dqn_model.qnet.state_dict() # 同期後も qnet の状態は変わらないはず

        # 辞書のキーが一致することを確認
        self.assertEqual(synced_qnet_target_state.keys(), qnet_state_after_sync.keys())

        # 各パラメータのテンソルが一致することを確認
        for key in qnet_state_after_sync:
            torch.testing.assert_close(synced_qnet_target_state[key], qnet_state_after_sync[key])

    def test_perform_standard_dqn_update(self):
        # このテストには dqn_model_mask_true インスタンスを使用
        dqn_model = self.dqn_model_mask_true
        batch_size = dqn_model.batch_size
        # mask=True の場合の入力サイズを使用
        input_size = (dqn_model.agents_num + dqn_model.goals_num) * 2
        action_size = dqn_model.action_size

        # 1. モックデータをセットアップ
        mock_agent_states_batch = torch.randn(batch_size, input_size)
        mock_action_batch = torch.randint(0, action_size, (batch_size,))
        mock_reward_batch = torch.randn(batch_size)
        mock_next_agent_states_batch = torch.randn(batch_size, input_size)
        mock_done_batch = torch.randint(0, 2, (batch_size,), dtype=torch.bool)
        mock_episode_num = 100 # ターゲットネットワークが同期されるエピソード番号

        # 2. 内部メソッドをモックする
        from unittest.mock import patch, MagicMock

        # _calculate_q_values, _calculate_next_max_q_values, _calculate_target_q_values,
        # huber_loss, _optimize_network, sync_qnet をモック
        mock_loss_value = 0.123 # huber_loss の戻り値として期待するスカラー値
        with patch.object(dqn_model, '_calculate_q_values', return_value=torch.randn(batch_size)) as mock_calculate_q_values, \
            patch.object(dqn_model, '_calculate_next_max_q_values', return_value=torch.randn(batch_size)) as mock_calculate_next_max_q_values, \
            patch.object(dqn_model, '_calculate_target_q_values', return_value=torch.randn(batch_size)) as mock_calculate_target_q_values, \
            patch.object(dqn_model, 'huber_loss', return_value=torch.tensor(mock_loss_value, requires_grad=True)) as mock_huber_loss, \
            patch.object(dqn_model, '_optimize_network') as mock_optimize_network, \
            patch.object(dqn_model, 'sync_qnet') as mock_sync_qnet:

            # 3. _perform_standard_dqn_update メソッドを呼び出す
            calculated_scalar_loss = dqn_model._perform_standard_dqn_update(
                mock_agent_states_batch,
                mock_action_batch,
                mock_reward_batch,
                mock_next_agent_states_batch,
                mock_done_batch,
                mock_episode_num
            )

            # 4. 期待されるメソッドが適切な引数で呼ばれたことをアサート

            # _calculate_q_values が正しい引数で呼ばれたことをアサート
            mock_calculate_q_values.assert_called_once_with(mock_agent_states_batch, mock_action_batch)

            # _calculate_next_max_q_values が正しい引数で呼ばれたことをアサート
            mock_calculate_next_max_q_values.assert_called_once_with(mock_next_agent_states_batch)

            # _calculate_target_q_values が正しい引数で呼ばれたことをアサート
            # モックされた _calculate_next_max_q_values の戻り値が渡されることを確認
            mock_calculate_target_q_values.assert_called_once_with(
                mock_reward_batch, mock_done_batch, mock_calculate_next_max_q_values.return_value
            )

            # huber_loss が正しい引数で呼ばれたことをアサート
            # モックされた _calculate_q_values と _calculate_target_q_values の戻り値が渡されることを確認
            mock_huber_loss.assert_called_once_with(
                mock_calculate_q_values.return_value, mock_calculate_target_q_values.return_value
            )

            # _optimize_network が正しい引数で呼ばれたことをアサート
            # モックされた huber_loss の戻り値 (損失テンソル) が渡されることを確認
            mock_optimize_network.assert_called_once_with(mock_huber_loss.return_value)

            # episode_num が target_update_frequency の倍数なので、sync_qnet が呼ばれたことをアサート
            dqn_model.target_update_frequency = 100 # テストパラメータに合わせる
            if mock_episode_num > 0 and mock_episode_num % dqn_model.target_update_frequency == 0:
                mock_sync_qnet.assert_called_once()
            else:
                mock_sync_qnet.assert_not_called() # 同期されないケースもテストする場合はこちらを使用

            # スカラー損失の計算が正しいことをアサート
            # _perform_standard_dqn_update メソッドは huber_loss の戻り値に対して .item() を呼び出しているため、
            # 期待値はモックされた損失テンソルの item() となります。
            expected_scalar_loss = mock_huber_loss.return_value.item()
            self.assertAlmostEqual(calculated_scalar_loss, expected_scalar_loss, places=6)


    def test_update(self):
        # このテストには dqn_model_mask_true インスタンスを使用 (load_model=0)
        dqn_model_standard = self.dqn_model_mask_true
        batch_size = dqn_model_standard.batch_size
        # mask=True の場合の全体の状態のサイズを使用
        global_state_dim_masked = (dqn_model_standard.agents_num + dqn_model_standard.goals_num) * 2

        # 1. モックデータをセットアップ
        mock_global_states_batch = torch.randn(batch_size, global_state_dim_masked)
        mock_action_batch = torch.randint(0, dqn_model_standard.action_size, (batch_size,))
        mock_reward_batch = torch.randn(batch_size)
        mock_next_global_states_batch = torch.randn(batch_size, global_state_dim_masked)
        mock_done_batch = torch.randint(0, 2, (batch_size,), dtype=torch.bool)
        mock_episode_num = 50

        # 2. 内部メソッドをモックする
        from unittest.mock import patch, MagicMock

        # _extract_agent_state, _perform_standard_dqn_update, _perform_knowledge_distillation_update をモック
        # _extract_agent_state は抽出された状態と次の状態のペアを返すようにモック
        mock_agent_states = torch.randn(batch_size, 2) # 抽出されたエージェント状態のモック
        mock_next_agent_states = torch.randn(batch_size, 2) # 抽出された次のエージェント状態のモック

        with patch.object(dqn_model_standard, '_extract_agent_state', return_value=(mock_agent_states, mock_next_agent_states)) as mock_extract_agent_state, \
            patch.object(dqn_model_standard, '_perform_standard_dqn_update', return_value=0.5) as mock_perform_standard_dqn_update, \
            patch.object(dqn_model_standard, '_perform_knowledge_distillation_update', return_value=0.7) as mock_perform_knowledge_distillation_update:

            # 3. update メソッドを呼び出す (標準 DQN モード)
            # load_model が 0 または 1 の場合
            dqn_model_standard.load_model = 0
            calculated_loss_standard = dqn_model_standard.update(
                0, # agent index
                mock_global_states_batch,
                mock_action_batch,
                mock_reward_batch,
                mock_next_global_states_batch,
                mock_done_batch,
                mock_episode_num
            )

            # 4. 期待されるメソッドが呼ばれたことをアサート (標準 DQN モード)

            # _extract_agent_state が正しい引数で呼ばれたことをアサート
            mock_extract_agent_state.assert_called_once_with(
                0, mock_global_states_batch, mock_next_global_states_batch
            )

            # _perform_standard_dqn_update が正しい引数で呼ばれたことをアサート
            # 抽出されたエージェント状態が渡されることを確認
            mock_perform_standard_dqn_update.assert_called_once_with(
                mock_agent_states,
                mock_action_batch,
                mock_reward_batch,
                mock_next_agent_states,
                mock_done_batch,
                mock_episode_num
            )

            # _perform_knowledge_distillation_update が呼ばれなかったことをアサート
            mock_perform_knowledge_distillation_update.assert_not_called()

            # 返された損失が _perform_standard_dqn_update の戻り値と一致することをアサート
            self.assertEqual(calculated_loss_standard, 0.5)

            # --- 特殊な学習モードのテスト (load_model が 0 または 1 以外の場合) ---
            # モックの呼び出しカウントをリセット
            mock_extract_agent_state.reset_mock()
            mock_perform_standard_dqn_update.reset_mock()
            mock_perform_knowledge_distillation_update.reset_mock()

            # load_model を 2 に設定して特殊な学習モードを有効にする
            dqn_model_standard.load_model = 2 # 特殊な学習モードをトリガーする値

            calculated_loss_special = dqn_model_standard.update(
                0, # agent index
                mock_global_states_batch,
                mock_action_batch,
                mock_reward_batch, # このモードでは使用されない可能性あり
                mock_next_global_states_batch, # このモードでは使用されない可能性あり
                mock_done_batch, # このモードでは使用されない可能性あり
                mock_episode_num
            )

            # 5. 期待されるメソッドが呼ばれたことをアサート (特殊な学習モード)

            # _extract_agent_state が正しい引数で呼ばれたことをアサート
            mock_extract_agent_state.assert_called_once_with(
                0, mock_global_states_batch, mock_next_global_states_batch
            )

            # _perform_standard_dqn_update が呼ばれなかったことをアサート
            mock_perform_standard_dqn_update.assert_not_called()

            # _perform_knowledge_distillation_update が正しい引数で呼ばれたことをアサート
            # 抽出されたエージェント状態が渡されることを確認
            # このメソッドは通常 reward, done, next_state を使用しないため、それらが渡されないことをアサート（あるいは渡されるが無視されることを確認）
            mock_perform_knowledge_distillation_update.assert_called_once_with(
                mock_agent_states,
                mock_action_batch
                # reward_batch, next_global_states_batch, done_batch は渡されないと想定
            )

            # 返された損失が _perform_knowledge_distillation_update の戻り値と一致することをアサート
            self.assertEqual(calculated_loss_special, 0.7)

"""
if __name__ == '__main__':
    # Jupyter Notebook/Colab で unittest を実行するための設定
    # argv=['first-arg-is-ignored'] は、sys.argv からスクリプト名を削除するために必要
    # exit=False は、テスト完了後にシステムが終了するのを防ぐ
    unittest.main(argv=['first-arg-is-ignored'], exit=False)
"""