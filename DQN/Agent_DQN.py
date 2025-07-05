"""
エージェントの行動を出力.
学習器の更新など.
"""
import torch
import numpy as np
import os
import sys


from utils.replay_buffer import ReplayBuffer
from DQN.dqn import DQNModel
import torch.optim as optim
from DQN.dqn import QNet # QNet クラスのインポートが必要です
from utils.replay_buffer import ReplayBuffer

np.random.seed(0)
torch.manual_seed(0)

# ターミナルの表示関連
RED = '\033[91m'
GREEN = '\033[92m'
RESET = '\033[0m'

MAX_EPSILON = 1
MIN_EPSILON = 0.01

class Agent_DQN:
    def __init__(self, args):
        self.agents_num = args.agents_number
        self.batch_size = args.batch_size
        self.decay_epsilon_step = args.decay_epsilon
        self.action_size = 5
        self.epsilon = MAX_EPSILON
        #self.learning_mode = args.learning_mode
        self.load_model = args.load_model
        self.goals_num = args.goals_number
        self.mask = args.mask
        #self.model_path = model_path
        self.device = torch.device(args.device)
        self.replay_buffer = ReplayBuffer("DQN", args.buffer_size, self.batch_size, self.device)

        # Agent_DQN の内部で DQNModel を初期化
        self.model = DQNModel(
            args.optimizer,
            args.gamma,
            args.batch_size,
            self.agents_num,
            self.goals_num,
            args.load_model, # argsからload_modelを渡す
            args.learning_rate,
            args.mask,
            args.target_update_frequency if hasattr(args, 'target_update_frequency') else 100 # デフォルト値を追加
        )

    """
        # モデルのロード
        if self.load_model == 1 or self.load_model == 2:
            self.model_path = model_path
            self.loading_model(self.model_path)

        self.model = DQNModel(args.optimizer,args.gamma,args.batch_size,args.agents_number,self.goals_num,self.load_model,args.learning_rate,self.mask)

    # 学習済みモデルの存在の確認
    def loading_model(self, model_path):
        if os.path.exists(model_path):
            print('モデルを読み込みました.')
            print(f"from {GREEN}{model_path}{RESET}\n")
        else:
            print(f"学習済みモデル {RED}{model_path}{RESET} が見つかりません.")
            print('学習する場合, load_model=0 に変更してください.\n')
            sys.exit()
    
    """

    def get_action(self, i, states):
        flat_state = np.array(states).flatten()
        states_torchtensol = torch.tensor(flat_state, dtype=torch.float32) # 1次元のテンソルに変換

        return self.nn_greedy_actor(i, states_torchtensol)

    # Q系列のNNモデル使用時のε-greedy
    def nn_greedy_actor(self, i, states:torch.Tensor):
        if self.mask:
            states = states[self.goals_num + i]

        if np.random.rand() < self.epsilon:
            return np.random.choice(self.action_size)
        else:
            states = states.to(self.device)
            qs = self.model.qnet(states)#nn.module標準の__call__ の挙動によるもの。入力データ（テンソル）を表す
            return qs.argmax().item()

    # epsilonの線形アニーリング
    def decay_epsilon(self, step):
        if step < self.decay_epsilon_step:
            self.epsilon = MIN_EPSILON + (MAX_EPSILON - MIN_EPSILON) * (self.decay_epsilon_step - step) / self.decay_epsilon_step

    def decay_epsilon_power(self,step:int,alpha=0.5):
        # ゼロ除算対策
        effective_step = max(1, step)
        self.epsilon = MAX_EPSILON * (1.0 / (effective_step ** alpha))
        # 必要に応じて MIN_EPSILON で下限を設ける
        #self.epsilon = max(MIN_EPSILON, self.epsilon)

    # 価値更新(非推奨)
    def update_brain(self, i, states, action, reward, next_state, done, episode_num):
        """
        副作用を持つため強く非推奨
        (推奨):observe_and_store_experienceとlearn_from_experienceを組み合わせて使う
        """
        """
        self.replay_buffer.add(states, action, reward, next_state, done)

        if self.load_model == 1:
            return None # 学習済みモデル使用時は更新なし

        if len(self.replay_buffer) < self.batch_size:
            return None #<-こっちでバグったら単にreturnにする?
        
        states, action, reward, next_state, done = self.replay_buffer.get_batch()

        # Qネットワークの重み更新
        scalar_loss = self.model.update(i, states, action, reward, next_state, done, episode_num)

        return scalar_loss
        """
        print("Warning: update_brainは非推奨。代わりにobserve_and_store_experienceとlearn_from_experienceを推奨。")
        # 必要であれば、互換性のために内部で呼び出しをラップすることも可能だが、非推奨であることを明確にする
        #self.observe_and_store_experience(states, action, reward, next_state, done)
        #return self.learn_from_experience(i, episode_num)


    def observe_and_store_experience(self, state, action, reward, next_state, done):
        """
        環境からの単一ステップの経験をリプレイバッファに追加する。
        """
        self.replay_buffer.add(state, action, reward, next_state, done)

    def learn_from_experience(self, i, episode_num):
        """
        リプレイバッファからバッチを取得し、モデルを学習させる。
        """
        if self.load_model == 1:
            return None # 学習済みモデル使用時は更新なし

        #if len(self.replay_buffer) < self.model.batch_size: # batch_sizeはmodelにあると仮定
        if len(self.replay_buffer) < self.batch_size: # batch_sizeはselfにある
            return None # バッチサイズに満たない場合は学習しない

        # 1. バッチデータの取得
        states, action, reward, next_state, done = self.replay_buffer.get_batch()
        
        # 2. モデルの更新 episode_numはターゲットネットワーク更新タイミングのため必要
        loss = self.model.update(i, states, action, reward, next_state, done, episode_num)

        return loss

# メインループでの呼び出し方の変更
# for i, agent in enumerate(agents):
#     # ステップごとに経験をストア
#     agent.observe_and_store_experience(states, actions[i], reward, next_state, done)
#     
#     # 学習は別のタイミングでトリガー
#     loss = agent.learn_from_experience(i, episode_num, step_count)
#     if loss is not None:
#         losses.append(loss)