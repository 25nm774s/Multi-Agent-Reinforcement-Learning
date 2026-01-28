import numpy as np
import random
import torch
from collections import deque
import numpy as np
from typing import Tuple, List, Optional

from Base.Constant import ExperienceType

class SumTree:
    """
    A binary tree data structure used for efficient priority sampling
    and updating in Prioritized Experience Replay (PER).
    """
    def __init__(self, capacity):
        """
        Initializes the SumTree with a given capacity.

        Args:
            capacity (int): The maximum number of leaf nodes (experiences)
                            the tree can hold. This should be a power of 2
                            for optimal tree structure, but will work otherwise.
        """
        self.capacity = capacity
        # The tree array stores sums of priorities, total size 2*capacity - 1
        # The leaf nodes (actual priorities) start at index `capacity - 1`
        self.tree = np.zeros(2 * capacity - 1)
        # The data array stores the actual experiences' indices (or data) associated with leaves
        # This will hold the indices of the experiences in the replay buffer's main list
        self.data_idx_map = np.zeros(capacity, dtype=int)
        self.data_pointer = 0 # Points to the next available position in the data_idx_map

    def _propagate(self, idx, change):
        """
        Propagates the change up the tree from a leaf node to the root.
        """
        # If we are at the root, stop propagating up
        if idx == 0:
            return
        parent_idx = (idx - 1) // 2
        self.tree[parent_idx] += change
        self._propagate(parent_idx, change)

    def _retrieve(self, idx, value):
        """
        Retrieves the index of a leaf node corresponding to a sampled value.
        """
        left_child_idx = 2 * idx + 1
        right_child_idx = 2 * idx + 2

        if left_child_idx >= len(self.tree):
            # Reached a leaf node
            return idx

        if value <= self.tree[left_child_idx]:
            return self._retrieve(left_child_idx, value)
        else:
            return self._retrieve(right_child_idx, value - self.tree[left_child_idx])

    def add(self, priority, data_idx):
        """
        Adds a new experience (priority and data_idx) to the tree.
        If the buffer is full, it overwrites the oldest experience.
        """
        # Index of the leaf node in the tree array
        tree_idx = self.data_pointer + self.capacity - 1

        # Store the experience's index in the data array
        self.data_idx_map[self.data_pointer] = data_idx

        # Update the leaf node's priority and propagate changes up the tree
        self.update(tree_idx, priority)

        # Move data pointer to the next position
        self.data_pointer = (self.data_pointer + 1) % self.capacity

    def update(self, tree_idx, priority):
        """
        Updates the priority of a specific leaf node and propagates
        the change up the tree.

        Args:
            tree_idx (int): The index of the leaf node in the `self.tree` array.
            priority (float): The new priority value for this leaf node.
        """
        change = priority - self.tree[tree_idx]
        self.tree[tree_idx] = priority
        self._propagate(tree_idx, change)

    def get_prefix_sum_idx(self, value):
        """
        Given a value, finds the index of the leaf node whose cumulative sum
        matches or exceeds the value. Used for proportional sampling.

        Args:
            value (float): A random value sampled between 0 and total_priority.

        Returns:
            tuple: A tuple containing:
                - int: The index of the leaf node in the `self.tree` array.
                - float: The priority stored at that leaf node.
                - int: The original index of the experience in the replay buffer's data list.
        """
        tree_idx = self._retrieve(0, value) # Start retrieval from the root (index 0)
        # The data_idx_map index corresponds to the leaf node's position relative to the start of leaves
        data_buffer_idx = tree_idx - (self.capacity - 1)

        return tree_idx, self.tree[tree_idx], self.data_idx_map[data_buffer_idx]

    @property
    def total_priority(self):
        """
        Returns the sum of all priorities (the value at the root of the tree).
        This is used for normalizing sampling values.
        """
        return self.tree[0]

class ReplayBuffer:
    def __init__(self, buffer_size: int, batch_size: int, device: torch.device, alpha: float = 0.6, use_per: bool = False, n_agents: int = 0, num_channels: int = 0, grid_size: int = 0, goals_number: int = 0):
        self.buffer_size: int = buffer_size
        self.batch_size: int = batch_size
        self.device: torch.device = device
        self.alpha: float = alpha
        self.use_per: bool = use_per
        self._max_priority = 1.0

        self.n_agents: int = n_agents
        self.num_channels: int = num_channels
        self.grid_size: int = grid_size
        self.goals_number: int = goals_number

        if self.use_per:
            self.experiences: List[Optional[ExperienceType]] = [None] * buffer_size
            self.tree = SumTree(capacity=buffer_size)
            self.current_idx = 0
            self.size = 0
        else:
            self.buffer: deque[ExperienceType] = deque(maxlen=buffer_size)
            self.priorities: deque[float] = deque(maxlen=buffer_size)

    def add(self, agent_obs_tensor: torch.Tensor, global_state_tensor: torch.Tensor, actions_tensor: torch.Tensor, rewards_tensor: torch.Tensor, dones_tensor: torch.Tensor, all_agents_done_scalar: torch.Tensor, next_agent_obs_tensor: torch.Tensor, next_global_state_tensor: torch.Tensor) -> None:
        data: ExperienceType = (
            agent_obs_tensor,
            global_state_tensor,
            actions_tensor,
            rewards_tensor,
            dones_tensor,
            all_agents_done_scalar,
            next_agent_obs_tensor,
            next_global_state_tensor
        )
        if self.use_per:
            self.experiences[self.current_idx] = data
            self.tree.add(self._max_priority, self.current_idx)

            self.current_idx = (self.current_idx + 1) % self.buffer_size
            self.size = min(self.size + 1, self.buffer_size)
        else:
            self.buffer.append(data)
            self.priorities.append(0.0) # Not used for uniform, but keeping for consistency

    def __len__(self) -> int:
        if self.use_per:
            return self.size
        else:
            return len(self.buffer)

    def sample(self, beta: float) -> Optional[Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, Optional[torch.Tensor], Optional[List[int]]]]:
        buffer_len = len(self)
        if buffer_len < self.batch_size:
            return None

        if self.use_per:
            sampled_indices: List[int] = [] # Original indices in self.experiences
            sampled_priorities_from_tree: List[float] = [] # These are the raw priorities from the tree
            sampled_tree_indices: List[int] = [] # Tree indices for update_priorities

            min_priority = 1e-6 # 0ではない小さな値

            segment = self.tree.total_priority / self.batch_size

            for i in range(self.batch_size):
                a = segment * i
                b = segment * (i + 1)
                s = random.uniform(a, b)

                tree_idx, priority, data_idx = self.tree.get_prefix_sum_idx(s)
                sampled_tree_indices.append(tree_idx)
                sampled_indices.append(data_idx)#type:ignore
                sampled_priorities_from_tree.append(priority) # Store the raw priority from the tree

            all_active_raw_priorities = np.array([self.tree.tree[self.tree.capacity - 1 + j] for j in range(buffer_len)])
            adjusted_all_active_priorities = (all_active_raw_priorities + min_priority)**self.alpha
            sum_adjusted_all_active_priorities = adjusted_all_active_priorities.sum()

            adjusted_sampled_priorities = (np.array(sampled_priorities_from_tree) + min_priority)**self.alpha
            p_i_normalized = adjusted_sampled_priorities / sum_adjusted_all_active_priorities
            is_weights_np = (buffer_len * p_i_normalized) ** -beta
            max_is_weight = np.max(is_weights_np) if np.max(is_weights_np) > 0 else 1.0
            is_weights_np /= max_is_weight

            is_weights_tensor: torch.Tensor = torch.tensor(is_weights_np, dtype=torch.float32, device=self.device)#type:ignore

            sampled_experiences = [self.experiences[idx] for idx in sampled_indices]
            sampled_original_indices = sampled_tree_indices # For PER update, we need tree_idx

        else: # Uniform sampling
            try:
                sampled_indices_for_uniform: List[int] = random.sample(range(buffer_len), k=self.batch_size)
            except ValueError as e:
                print(f"Error during random.sample (Uniform): {e}")
                print(f"Buffer length: {buffer_len}, Batch size: {self.batch_size}")
                return None

            is_weights_tensor: Optional[torch.Tensor] = None
            sampled_experiences = [self.buffer[i] for i in sampled_indices_for_uniform]
            sampled_original_indices = None # Not applicable for uniform sampling

        try:
            filtered_experiences = [exp for exp in sampled_experiences if exp is not None]

            if not filtered_experiences:
                print("Warning: Sampled experiences list is empty after filtering None. Returning None.")
                return None

            # Unpack and stack the tensors
            agent_obs_tensor_batch = torch.stack([exp[0] for exp in filtered_experiences], dim=0)
            global_state_tensor_batch = torch.stack([exp[1] for exp in filtered_experiences], dim=0)
            actions_tensor_batch = torch.stack([exp[2] for exp in filtered_experiences], dim=0)
            rewards_tensor_batch = torch.stack([exp[3] for exp in filtered_experiences], dim=0)
            dones_tensor_batch = torch.stack([exp[4] for exp in filtered_experiences], dim=0)
            all_agents_done_scalar_batch = torch.stack([exp[5] for exp in filtered_experiences], dim=0)
            next_agent_obs_tensor_batch = torch.stack([exp[6] for exp in filtered_experiences], dim=0)
            next_global_state_tensor_batch = torch.stack([exp[7] for exp in filtered_experiences], dim=0)

        except Exception as e:
            print(f"Error converting sampled data to batched tensors: {e}")
            return None

        return (
            agent_obs_tensor_batch,
            global_state_tensor_batch,
            actions_tensor_batch,
            rewards_tensor_batch,
            dones_tensor_batch,
            all_agents_done_scalar_batch,
            next_agent_obs_tensor_batch,
            next_global_state_tensor_batch,
            is_weights_tensor,
            sampled_original_indices
        )

    def update_priorities(self, tree_indices: List[int], td_errors: np.ndarray) -> None:
        """
        Updates the priorities of sampled experiences in the SumTree based on their TD errors.

        Args:
            tree_indices (List[int]): A list of tree_indices (leaf node indices in SumTree)
                                      for the experiences whose priorities are to be updated.
            td_errors (np.ndarray): An array of TD errors corresponding to the sampled experiences.
        """
        if not self.use_per:
            return
        min_error = 1e-6 # Epsilon for numerical stability
        new_priorities = np.maximum(np.abs(td_errors), min_error)
        for i in range(len(tree_indices)): # Corrected typo here
            tree_idx = tree_indices[i]
            priority = new_priorities[i]
            self.tree.update(tree_idx, priority)
        # Update _max_priority to ensure new experiences are added with at least the highest observed priority
        self._max_priority = max(self._max_priority, np.max(new_priorities))
