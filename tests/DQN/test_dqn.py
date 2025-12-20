# -*- coding: utf-8 -*-

import torch
import torch.optim as optim
import unittest

from src.DQN.dqn import QNet, DQNModel
from src.utils.StateProcesser import StateProcessor
# NOTE: QNet and DQNModel classes are expected to be defined in a previous cell.
# If not, please ensure they are available in the current execution environment.


class DQNModelTest(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        # GPUが利用可能であればGPUを使用し、そうでなければCPUを使用
        # cuda:0 と明示的に指定することで、cuda と cuda:0 の比較問題を回避
        if torch.cuda.is_available():
            cls.device = torch.device("cuda:0")
        else:
            cls.device = torch.device("cpu")
        print(f"\nUsing device for tests: {cls.device}")

    def setUp(self):
        self.grid_size = 5
        self.batch_size = 4
        self.agent_num = 2
        self.goals_num = 1
        self.action_size = 5 # 上下左右 + 停止
        self.state_processor = StateProcessor(self.grid_size, self.goals_num, self.agent_num, self.device)

        # PERなしモデルの初期化
        self.dqn_model_no_per = DQNModel(
            optimizer_type='Adam',
            grid_size=self.grid_size,
            gamma=0.99,
            batch_size=self.batch_size,
            agent_num=self.agent_num,
            goals_num=self.goals_num,
            learning_rate=0.001,
            mask=False,
            device=str(self.device),
            use_per=False,
            state_processor=self.state_processor
        )

        # PERありモデルの初期化
        self.dqn_model_with_per = DQNModel(
            optimizer_type='Adam',
            grid_size=self.grid_size,
            gamma=0.99,
            batch_size=self.batch_size,
            agent_num=self.agent_num,
            goals_num=self.goals_num,
            learning_rate=0.001,
            mask=False,
            device=str(self.device),
            use_per=True,
            state_processor=self.state_processor
        )

        # ダミーデータの生成 (NNへの入力は (batch_size, num_channels, grid_size, grid_size) の形式)
        self.agent_state_shape = (self.batch_size, 3, self.grid_size, self.grid_size)
        self.agent_states_batch = torch.randn(self.agent_state_shape, device=self.device)
        self.next_agent_states_batch = torch.randn(self.agent_state_shape, device=self.device)

        self.action_batch = torch.randint(0, self.action_size, (self.batch_size,), device=self.device)
        self.reward_batch = torch.randn((self.batch_size,), device=self.device)
        self.done_batch = torch.randint(0, 2, (self.batch_size,), dtype=torch.bool, device=self.device)

        # PER用の重要度サンプリング重み (use_perがTrueの場合にのみ使用)
        self.is_weights_batch = torch.rand((self.batch_size,), device=self.device) * 10 # 適当な重み

        # _calculate_next_max_q_values のためのダミーデータ
        self.next_max_q_values_dummy = self.dqn_model_no_per._calculate_next_max_q_values(self.next_agent_states_batch)

        # update メソッド用のダミーグローバル状態バッチ (feature_dimを正しく計算)
        self.feature_dim = (self.goals_num + self.agent_num) * 2 # (1 + 2) * 2 = 6
        self.dummy_global_states_batch = torch.randint(0, self.grid_size, (self.batch_size, self.feature_dim), device=self.device)
        self.dummy_next_global_states_batch = torch.randint(0, self.grid_size, (self.batch_size, self.feature_dim), device=self.device)


    def test_initialization(self):
        """DQNModelの初期化とデバイス、use_perフラグの確認"""
        # モデルが正しいデバイスにいることを確認
        self.assertEqual(next(iter(self.dqn_model_no_per.qnet.parameters())).device, self.device)
        self.assertEqual(next(iter(self.dqn_model_no_per.qnet_target.parameters())).device, self.device)
        self.assertEqual(self.dqn_model_no_per.use_per, False)
        self.assertEqual(self.dqn_model_with_per.use_per, True)
        print("\nInitialization test passed: Networks are on the correct device and use_per flag is correct.")

    def test_sync_qnet(self):
        """ターゲットネットワークの同期テスト"""
        # QNetとQNetTargetの初期重みが異なることを確認 (初期化直後)
        initial_qnet_params = {n: p.clone() for n, p in self.dqn_model_no_per.qnet.named_parameters()}
        initial_qnet_target_params = {n: p.clone() for n, p in self.dqn_model_no_per.qnet_target.named_parameters()}

        # 少なくとも一つのパラメータが異なることを確認
        diff_found = False
        for name in initial_qnet_params:
            if not torch.equal(initial_qnet_params[name], initial_qnet_target_params[name]):
                diff_found = True
                break
        self.assertTrue(diff_found, "Initial QNet and QNetTarget parameters should be different.")
        # print("Initial QNet and QNetTarget parameters are different (as expected).")

        # ターゲットネットワークを同期
        self.dqn_model_no_per.sync_qnet()

        # 同期後、QNetとQNetTargetの重みが同じであることを確認
        for name, param in self.dqn_model_no_per.qnet.named_parameters():
            self.assertTrue(torch.equal(param, self.dqn_model_no_per.qnet_target.state_dict()[name]), \
                            f"Parameter {name} did not sync correctly.")
        print("Sync QNet test passed: QNet and QNetTarget are successfully synced.")

    def test_calculate_q_values(self):
        """_calculate_q_values および _calculate_next_max_q_values のテスト"""
        predicted_q_values = self.dqn_model_no_per._calculate_q_values(self.agent_states_batch, self.action_batch)
        self.assertEqual(predicted_q_values.shape, (self.batch_size,))

        next_max_q_values = self.dqn_model_no_per._calculate_next_max_q_values(self.next_agent_states_batch)
        self.assertEqual(next_max_q_values.shape, (self.batch_size,))
        self.assertFalse(next_max_q_values.requires_grad) # detachされていることを確認
        print("Calculate Q values test passed: Output shapes are correct and detach works.")

    def test_calculate_target_q_values(self):
        """_calculate_target_q_values のテスト"""
        target_q_values = self.dqn_model_no_per._calculate_target_q_values(self.reward_batch, self.done_batch, self.next_max_q_values_dummy)
        self.assertEqual(target_q_values.shape, (self.batch_size,))

        # doneがTrueの場合のターゲットQ値を確認
        done_batch_test = torch.zeros((self.batch_size,), dtype=torch.bool, device=self.device)
        done_batch_test[0] = True
        reward_test = torch.tensor([10.0, 0.0, 0.0, 0.0], device=self.device)
        next_max_q_test = torch.tensor([100.0, 20.0, 30.0, 40.0], device=self.device)

        target_q_test = self.dqn_model_no_per._calculate_target_q_values(reward_test, done_batch_test, next_max_q_test)

        # 最初の要素は reward_test[0] に等しいはず (1 - done_batch[0] = 0 なので)
        self.assertTrue(torch.isclose(target_q_test[0], reward_test[0]))
        print("Calculate Target Q values test passed: Handles 'done' state correctly.")

    def test_huber_loss(self):
        """huber_loss メソッドのテスト (PERの有無による重み付けの確認)"""
        pred_q = torch.tensor([1.0, 2.0, 3.0, 4.0], device=self.device, requires_grad=True)
        target_q = torch.tensor([1.5, 1.8, 3.2, 4.5], device=self.device)

        # PERなしの場合のテスト
        loss_no_per, td_errors_no_per = self.dqn_model_no_per.huber_loss(pred_q, target_q, is_weights=None)
        self.assertIsInstance(loss_no_per, torch.Tensor)
        self.assertEqual(loss_no_per.ndim, 0) # スカラーであること
        self.assertEqual(td_errors_no_per.shape, (self.batch_size,))
        self.assertFalse(td_errors_no_per.requires_grad) # detachされていること

        # PERありの場合のテスト
        is_weights = torch.tensor([0.5, 1.0, 1.5, 2.0], device=self.device)
        loss_with_per, td_errors_with_per = self.dqn_model_with_per.huber_loss(pred_q, target_q, is_weights=is_weights)
        self.assertIsInstance(loss_with_per, torch.Tensor)
        self.assertEqual(loss_with_per.ndim, 0) # スカラーであること
        self.assertEqual(td_errors_with_per.shape, (self.batch_size,))
        self.assertFalse(td_errors_with_per.requires_grad) # detachされていること

        # PERありとなしで損失が異なることを確認 (is_weightsが1以外の場合)
        self.assertTrue(torch.allclose(td_errors_no_per, td_errors_with_per)) # td_errorsは同じはず
        self.assertFalse(torch.isclose(loss_no_per, loss_with_per)) # 重みが異なるので損失は異なるはず
        print("Huber Loss test passed: Handles IS weights correctly.")

    def test_dqn_update_logic(self):
        """_perform_standard_dqn_update および update メソッドのテスト"""
        total_step = 0 # ターゲットネットワークの更新はしないようにする

        # _perform_standard_dqn_update のテスト (PERなしモデル)
        initial_qnet_param_before_update = next(iter(self.dqn_model_no_per.qnet.parameters())).clone().detach()
        loss, td_errors = self.dqn_model_no_per._perform_standard_dqn_update(
            self.agent_states_batch, self.action_batch, self.reward_batch, self.next_agent_states_batch, self.done_batch, total_step,
            is_weights_batch=None # PERなしなのでNone
        )
        self.assertIsInstance(loss, float)
        self.assertIsNone(td_errors) # use_per=False なので None を返す
        # パラメータが更新されたことを確認
        updated_qnet_param = next(iter(self.dqn_model_no_per.qnet.parameters())).clone().detach()
        self.assertFalse(torch.equal(initial_qnet_param_before_update, updated_qnet_param))
        # print("DQN model (no PER) updated and parameters changed.")

        # _perform_standard_dqn_update のテスト (PERありモデル)
        initial_qnet_param_before_update_per = next(iter(self.dqn_model_with_per.qnet.parameters())).clone().detach()
        loss_per, td_errors_per = self.dqn_model_with_per._perform_standard_dqn_update(
            self.agent_states_batch, self.action_batch, self.reward_batch, self.next_agent_states_batch, self.done_batch, total_step,
            is_weights_batch=self.is_weights_batch # PERありなので重みを渡す
        )
        self.assertIsNotNone(td_errors_per) # td_errors_per != Noneであることを確認

        self.assertIsInstance(loss_per, float)
        self.assertIsNotNone(td_errors_per)
        self.assertEqual(td_errors_per.shape, (self.batch_size,)) # type:ignore :直前でNoneではないことが保証されている
        self.assertFalse(td_errors_per.requires_grad) # # type:ignore : detachされていること
        # パラメータが更新されたことを確認
        updated_qnet_param_per = next(iter(self.dqn_model_with_per.qnet.parameters())).clone().detach()
        self.assertFalse(torch.equal(initial_qnet_param_before_update_per, updated_qnet_param_per))
        # print("DQN model (with PER) updated and parameters changed, TD errors returned.")

        # update メソッドのテスト (内部で _perform_standard_dqn_update を呼ぶ)
        loss_update, td_errors_update = self.dqn_model_with_per.update(
            i=0, 
            global_states_batch=self.dummy_global_states_batch,
            actions_batch=self.action_batch, 
            rewards_batch=self.reward_batch, 
            next_global_states_batch=self.dummy_next_global_states_batch,
            dones_batch=self.done_batch, 
            total_step=total_step, 
            is_weights_batch=self.is_weights_batch
        )
        self.assertIsInstance(loss_update, float)
        self.assertIsNotNone(td_errors_update)
        self.assertEqual(td_errors_update.shape, (self.batch_size,)) # type:ignore
        print("DQN update logic test passed: _perform_standard_dqn_update and update methods work.")

    def test_weight_management(self):
        """重み管理メソッド (get_weights, set_qnet_state, set_target_state, set_optimizer_state) のテスト"""
        # 現在の重みを取得
        qnet_state, qnet_target_state, optimizer_state = self.dqn_model_no_per.get_weights()

        # QNetの重みを変更 (例: ランダムなテンソルで上書き)
        new_qnet_state = {n: torch.randn_like(p) for n, p in self.dqn_model_no_per.qnet.named_parameters()}
        self.dqn_model_no_per.set_qnet_state(new_qnet_state)
        # QNetの重みが変更されたことを確認
        for name, param in self.dqn_model_no_per.qnet.named_parameters():
            self.assertTrue(torch.equal(param, new_qnet_state[name]))

        # 元のQNetの重みに戻す
        self.dqn_model_no_per.set_qnet_state(qnet_state)
        for name, param in self.dqn_model_no_per.qnet.named_parameters():
            self.assertTrue(torch.equal(param, qnet_state[name]))
        # print("set_qnet_state successfully updated and restored QNet weights.")

        # Target Netの重みを変更
        new_qnet_target_state = {n: torch.randn_like(p) for n, p in self.dqn_model_no_per.qnet_target.named_parameters()}
        self.dqn_model_no_per.set_target_state(new_qnet_target_state)
        for name, param in self.dqn_model_no_per.qnet_target.named_parameters():
            self.assertTrue(torch.equal(param, new_qnet_target_state[name]))
        # print("set_target_state successfully updated QNetTarget weights.")

        # Optimizerの重みを変更 (内部状態は複雑なので、load_state_dictがエラーなく実行されるかを確認)
        optimizer_test = optim.Adam(QNet(self.grid_size, self.action_size).to(self.device).parameters(), lr=0.01)
        new_optimizer_state = optimizer_test.state_dict()
        self.dqn_model_no_per.set_optimizer_state(new_optimizer_state)

        # Optimizerの状態が更新されたことを確認
        self.assertEqual(len(self.dqn_model_no_per.optimizer.state), len(new_optimizer_state['state']))
        self.assertEqual(self.dqn_model_no_per.optimizer.param_groups[0]['lr'], new_optimizer_state['param_groups'][0]['lr'])
        print("Weight management test passed: State management methods work correctly.")
