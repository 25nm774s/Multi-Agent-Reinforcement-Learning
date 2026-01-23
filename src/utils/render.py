import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
import numpy as np

class Render:
    """
    エージェントの軌跡描画およびアニメーション生成を管理するクラス。
    """
    def __init__(self, grid_size, goals_number, agents_number):
        """
        Render クラスを初期化します。

        Args:
            grid_size (int): グリッドのサイズ。
            goals_number (int): ゴールの数。
            agents_number (int): エージェントの数。
        """
        self.grid_size = grid_size
        self.goals_number = goals_number
        self.agents_number = agents_number

    def render(self, trajectory_data):
        """
        trajectory_data (状態のリスト) からエージェントの軌跡を描画します。

        Args:
            trajectory_data (list): 環境状態のリスト。各状態はゴール位置とエージェント位置のタプル。
        """
        if not trajectory_data:
            print("描画する軌跡データがありません。")
            return

        # Extract agent trajectories
        agent_trajectories = {f'agent_{i}': [] for i in range(self.agents_number)}
        goal_positions = trajectory_data[0][:self.goals_number] # Assuming goal positions are fixed throughout the trajectory_data

        for state in trajectory_data:
            agent_current_positions = state[self.goals_number:]
            for i in range(self.agents_number):
                agent_trajectories[f'agent_{i}'].append(agent_current_positions[i])

        plt.figure(figsize=(6, 6))
        ax = plt.gca()

        # Draw grid lines
        ax.set_xticks(np.arange(-.5, self.grid_size, 1), minor=True)
        ax.set_yticks(np.arange(-.5, self.grid_size, 1), minor=True)
        ax.grid(which='minor', color='black', linestyle='-', linewidth=0.5)

        # Set limits and labels
        ax.set_xlim(-0.5, self.grid_size - 0.5)
        ax.set_ylim(-0.5, self.grid_size - 0.5)
        ax.set_xlabel('X')
        ax.set_ylabel('Y')
        ax.set_title('Agent Trajectories')

        # Plot goals
        for i, goal_pos in enumerate(goal_positions):
            ax.plot(goal_pos[0], goal_pos[1], 'P', markersize=10, color='gold', label=f'Goal {i}') # 'P' for filled pentagon

        # Plot agent trajectories
        colors = plt.cm.get_cmap('tab10', self.agents_number) # Get distinct colors for agents
        for i in range(self.agents_number):
            trajectory = agent_trajectories[f'agent_{i}']
            if trajectory:
                # Extract x and y coordinates
                x_coords = [pos[0] for pos in trajectory]
                y_coords = [pos[1] for pos in trajectory]
                # Plot the trajectory
                ax.plot(x_coords, y_coords, marker='o', linestyle='-', color=colors(i), label=f'Agent {i}')
                # Mark start and end points
                ax.plot(x_coords[0], y_coords[0], 'D', markersize=8, color=colors(i), markeredgecolor='black', label=f'Agent {i} Start') # 'D' for diamond start
                if len(x_coords) > 1:
                     ax.plot(x_coords[-1], y_coords[-1], 'X', markersize=8, color=colors(i), markeredgecolor='black', label=f'Agent {i} End') # 'X' for X end

        # Add legend
        handles, labels = plt.gca().get_legend_handles_labels()
        by_label = dict(zip(labels, handles))
        ax.legend(by_label.values(), by_label.keys(), loc='upper right')

        ax.invert_yaxis() # Invert Y axis to match typical grid world indexing (0,0) at top-left
        plt.gca().set_aspect('equal', adjustable='box') # Ensure square grid cells
        plt.show()

    def render_anime(self, trajectory_data, output_filename="agent_trajectory_animation.gif", interval=200):
        """
        trajectory_data (状態のリスト) からエージェントの軌跡のアニメーションを生成し、GIFとして保存します。

        Args:
            trajectory_data (list): 環境状態のリスト。各状態はゴール位置とエージェント位置のタプル。
            output_filename (str): 保存するGIFファイルのファイル名。
            interval (int): フレーム間の遅延（ミリ秒）。
        """
        if not trajectory_data:
            print("描画する軌跡データがありません。")
            return

        # Extract initial goal positions (assuming they are fixed)
        initial_state = trajectory_data[0]
        goal_positions = initial_state[:self.goals_number]

        fig, ax = plt.subplots(figsize=(6, 6))

        # Draw grid lines
        ax.set_xticks(np.arange(-.5, self.grid_size, 1), minor=True)
        ax.set_yticks(np.arange(-.5, self.grid_size, 1), minor=True)
        ax.grid(which='minor', color='black', linestyle='-', linewidth=0.5)

        # Set limits and labels
        ax.set_xlim(-0.5, self.grid_size - 0.5)
        ax.set_ylim(-0.5, self.grid_size - 0.5)
        ax.set_xlabel('X')
        ax.set_ylabel('Y')
        ax.set_title('Agent Trajectories Animation')
        ax.invert_yaxis() # Invert Y axis to match typical grid world indexing (0,0) at top-left
        ax.set_aspect('equal', adjustable='box') # Ensure square grid cells


        # Plot goals (fixed)
        for i, goal_pos in enumerate(goal_positions):
            ax.plot(goal_pos[0], goal_pos[1], 'P', markersize=10, color='gold', label=f'Goal {i}')

        # Initialize agent markers and trajectories (initially empty)
        agent_markers = []
        agent_lines = []
        colors = plt.cm.get_cmap('tab10', self.agents_number)
        for i in range(self.agents_number):
            # Agent marker
            marker, = ax.plot([], [], marker='o', linestyle='', color=colors(i), markersize=8, label=f'Agent {i}')
            agent_markers.append(marker)
            # Agent trajectory line
            line, = ax.plot([], [], linestyle='-', color=colors(i), linewidth=1)
            agent_lines.append(line)

        # Add legend
        handles, labels = ax.get_legend_handles_labels()
        ax.legend(handles, labels, loc='upper right')

        # Add step counter text
        step_text = ax.text(0.02, 0.95, '', transform=ax.transAxes, fontsize=12,
                            verticalalignment='top', bbox=dict(boxstyle='round,pad=0.5', fc='wheat', alpha=0.5))



        def update(frame):
            """Update function for the animation."""
            current_state = trajectory_data[frame]
            current_agent_positions = current_state[self.goals_number:]

            for i in range(self.agents_number):
                # Update marker position
                pos = current_agent_positions[i]
                agent_markers[i].set_data([pos[0]], [pos[1]])

                # Update trajectory line data
                # Get current trajectory data
                x_data, y_data = agent_lines[i].get_data()
                # Append new position to trajectory data
                x_data = np.append(x_data, pos[0])
                y_data = np.append(y_data, pos[1])
                # Set updated trajectory data
                agent_lines[i].set_data(x_data, y_data)

            # Update the step counter text
            step_text.set_text(f'Step: {frame}')

            return agent_markers + agent_lines # Return all artists that were modified


        # Create the animation
        ani = FuncAnimation(fig, update, frames=len(trajectory_data), blit=True, interval=interval, repeat=False)

        # Save the animation as a GIF
        print(f"Saving animation to {output_filename}...")
        ani.save(output_filename, writer='pillow')
        print("Animation saved.")

        plt.close(fig) # Close the plot figure after saving