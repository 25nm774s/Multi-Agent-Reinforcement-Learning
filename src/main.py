import argparse
import torch
import json
import os
import shutil

from DQN.network import AgentNetwork
from Environments.StateProcesser import ObsToTensorWrapper
from Environments.MultiAgentGridEnv import MultiAgentGridEnv, GridEnvWrapper
from DQN.MultiAgent_DQN import MARLTrainer

RED = '\033[91m'
GREEN = '\033[92m'
RESET = '\033[0m'


def load_presets(file_path="presets.json", sample_path="presets.json.sample"):
    """
    設定ファイルを読み込む。
    実体がない場合はサンプルからコピーを試みる。
    """
    # 1. presets.json がなくて sample がある場合はコピーしてあげる（親切設計）
    if not os.path.exists(file_path) and os.path.exists(sample_path):
        print(f"[*] {file_path} not found. Creating from {sample_path}...")
        shutil.copy(sample_path, file_path)

    # 2. それでもファイルがない場合は空の辞書を返す
    if not os.path.exists(file_path):
        print("[!] Warning: No preset file found. Using internal defaults.")
        return {}

    with open(file_path, 'r', encoding='utf-8') as f:
        try:
            data = json.load(f)
        except json.JSONDecodeError:
            print(f"[!] Error: {file_path} is not a valid JSON. Ignoring presets.")
            return {}

    # エイリアス展開
    presets = {k: v for k, v in data.items() if k != "aliases"}
    aliases = data.get("aliases", {})
    for alias, target in aliases.items():
        if target in presets:
            presets[alias] = presets[target]
            
    return presets

def parse_args():
    # プリセットデータの読み込み
    PRESETS = load_presets("presets.json")

    parser = argparse.ArgumentParser(description="Multi-Agent RL Parser with Presets")

    # --- 1. プリセット選択引数 ---
    # choicesに辞書のキーを渡すことで、存在しないプリセット指定をエラーにできる
    parser.add_argument('-P', '--preset', choices=PRESETS.keys(), help="Use a predefined preset from JSON")

    # --- 既存の引数 ---
    parser.add_argument('-g','--grid_size', default=4, type=int)
    parser.add_argument('-A','--agents_number', default=2, type=int)
    parser.add_argument('-G','--goals_number', default=2, type=int)
    parser.add_argument('-l','--learning_mode', choices=['Q', 'QMIX','IQL', 'VDN'], default='IQL')
    parser.add_argument('--optimizer', choices=['Adam', 'RMSProp'], default='Adam')
    parser.add_argument('--neighbor_distance', default=256, type=int)# 大きい値にしておくことで全観測になる。
    parser.add_argument('--reward_mode', choices=[0, 1, 2, 3], default=2, type=int)
    parser.add_argument('--agent_reward_processing_mode', \
                        choices=['individual', 'sum_and_distribute', 'mean_and_distribute'], default='individual', \
                        help='IQL agent reward processing mode: how environment rewards are aggregated or scaled.')
    parser.add_argument('--device', choices=['auto', 'cpu', 'cuda', 'mps'], default='auto')
    parser.add_argument('-e','--episode_number', default=1200, type=int)
    parser.add_argument('-T','--max_timestep', default=150, type=int)
    parser.add_argument('--epsilon_decay', default=0.50, type=float)
    parser.add_argument('--learning_rate', default=0.001, type=float)
    parser.add_argument('--gamma', default=0.95, type=float)
    parser.add_argument('--buffer_size', default=10000, type=int)
    parser.add_argument('--batch_size', default=32, type=int)
    parser.add_argument('--save_agent_states', choices=[0, 1], default=1, type=int)
    parser.add_argument('--window_width', default=500, type=int)
    parser.add_argument('--window_height', default=500, type=int)
    parser.add_argument('--render_mode', choices=[0, 1], default=0, type=int)
    parser.add_argument('--pause_duration', default=0.1, type=float)
    parser.add_argument('--target_update_frequency', default=500, type=int)
    parser.add_argument('--alpha', default=0.6, type=float)
    parser.add_argument('--beta', default=0.4, type=float)
    parser.add_argument('--beta_anneal_steps', default=20000, type=int)
    parser.add_argument('--use_per', choices=[0, 1], default=0, type=int)

    # 3. 1回目のパース：プリセットの有無だけを確認
    # args, unknown = parser.parse_known_args() ではなく、
    # プリセット引数だけを抽出してチェックする
    temp_args, _ = parser.parse_known_args()
    
    if temp_args.preset:
        target = PRESETS[temp_args.preset]
        if isinstance(target, str):
            target = PRESETS[target]
        
        # プリセットをデフォルト値として設定
        parser.set_defaults(**target)
    
    # 4. 最終的なパース
    # ここで sys.argv (または引数なし) を使うことで、
    # 「プリセットで上書きされたデフォルト値」を「個別のコマンド引数」がさらに上書きできる
    args = parser.parse_args() 
    
    return args

if __name__ == '__main__':
    config = parse_args()
    print(f"Loaded config: {config}")

    # auto選択時のデバイス決定ロジックを追加
    if config.device == 'auto':
        if torch.cuda.is_available():
            config.device = 'cuda'
        elif torch.backends.mps.is_available():
            config.device = 'mps'
        else:
            config.device = 'cpu'
        print(f"自動選択されたデバイス: {GREEN}{config.device}{RESET}\n")

    def q_learning():
        from Q_learn.MultiAgent_Q import MultiAgent_Q
        from Q_learn.Agent_Q import Agent
        agents:list[Agent] = [Agent(config, id) for id in range(config.agents_number)]
        simulation = MultiAgent_Q(config,agents)
        
        simulation.train(config.episode_number)

        simulation.result_save()

        simulation.render_anime(config.episode_number)
        #print("reward:",r, "done:",done)
        #print("GET: trajectry")
        #for tr in traj: print(tr)
    
    def q_play():
        from Q_learn.MultiAgent_Q import MultiAgent_Q
        from Q_learn.Agent_Q import Agent
        agents:list[Agent] = [Agent(config, id) for id in range(config.agents_number)]
        simulation = MultiAgent_Q(config,agents)

        #simulation.render_anime(config.episode_number)
        # simulation.log_disp()
            
    def dqn_process(current_conf, run_id=None):
        # Shared components
        device = torch.device(current_conf.device)

        # 1. MultiAgentGridEnv のインスタンス化
        fix_goal_pool = [(current_conf.grid_size-1,current_conf.grid_size-1),(current_conf.grid_size//4,current_conf.grid_size//3),(current_conf.grid_size-1,0),(current_conf.grid_size//4,current_conf.grid_size//6)]
        _fix_goal_from_goal_number = fix_goal_pool[:min(current_conf.goals_number, len(fix_goal_pool))]

        multi_agent_env = MultiAgentGridEnv(current_conf, fixrd_goals=_fix_goal_from_goal_number)

        # 2. ObsToTensorWrapper をインスタンス化
        shared_state_processor = ObsToTensorWrapper(
            grid_size=current_conf.grid_size,
            goals_number=current_conf.goals_number,
            agents_number=current_conf.agents_number,
            neighbor_distance=current_conf.neighbor_distance,
            device=device,
            agent_ids=multi_agent_env._agent_ids,
            goal_ids=multi_agent_env._goal_ids
        )

        # 3. GridEnvWrapper をインスタンス化
        env_wrapper_instance = GridEnvWrapper(env_instance=multi_agent_env, state_processor_instance=shared_state_processor)

        shared_agent_network = AgentNetwork(
            grid_size=current_conf.grid_size,
            output_size=env_wrapper_instance.action_space_size, # Use the dynamically retrieved action_space_size
            total_agents=current_conf.agents_number
        ).to(device)

        # Instantiate MARLTrainer. It will now create its own ReplayBuffer internally.
        trainer = MARLTrainer(
            args=current_conf,
            mode=current_conf.learning_mode,
            env_wrapper=env_wrapper_instance, # 4. GridEnvWrapper のインスタンスを渡す
            shared_agent_network=shared_agent_network,
            shared_state_processor=shared_state_processor,
            run_id=run_id
        )

        # Run training
        trainer.train() # Comment out the full training

        print(f"--- {current_conf.learning_mode} mode test finished successfully ---")    
    
    def dimensions_estimater(grid_size:int, agent_number:int)->int:
        res = 1
        for i in range(agent_number):
            res *= (grid_size * grid_size - i)

        return res

    if config.learning_mode == "Q":
        if dimensions_estimater(config.grid_size, config.agents_number)>1e6: 
            raise ValueError(f"警告:推定空間サイズ({dimensions_estimater(config.grid_size, config.agents_number)})が大きすぎます")
        
        if config.episode_number == -1:
            q_play()
        else:
            q_learning()

    elif config.learning_mode == "IQL" or config.learning_mode == "QMIX" or config.learning_mode == "VDN":
        dqn_process(config)
    else:
        print("未実装\n")