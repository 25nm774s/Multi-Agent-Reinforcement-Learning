import unittest

# Assume the Grid and CollisionResolver class definitions
from src.Enviroments.MultiAgentGridEnv import Grid, CollisionResolver, PositionType # If in a separate file

class TestCollisionResolver(unittest.TestCase):
    """
    Test cases for the CollisionResolver class.
    """

    def test_init(self):
        """Test CollisionResolver initialization."""
        grid = Grid(10)
        resolver = CollisionResolver(grid)
        # Check if the internal _grid attribute is set correctly
        self.assertIs(resolver._grid, grid)

    def test_calculate_next_position(self):
        """Test _calculate_next_position method for all actions."""
        # No need for a Grid instance here, as it's a pure calculation
        resolver = CollisionResolver(Grid(5)) # Pass a dummy Grid instance

        current_pos = (2, 2)
        # 0: UP (y decreases)
        self.assertEqual(resolver._calculate_next_position(current_pos, 0), (2, 1))
        # 1: DOWN (y increases)
        self.assertEqual(resolver._calculate_next_position(current_pos, 1), (2, 3))
        # 2: LEFT (x decreases)
        self.assertEqual(resolver._calculate_next_position(current_pos, 2), (1, 2))
        # 3: RIGHT (x increases)
        self.assertEqual(resolver._calculate_next_position(current_pos, 3), (3, 2))
        # 4: STAY
        self.assertEqual(resolver._calculate_next_position(current_pos, 4), (2, 2))
        # Test from a boundary position
        boundary_pos = (0, 0)
        self.assertEqual(resolver._calculate_next_position(boundary_pos, 3), (1, 0)) # RIGHT
        self.assertEqual(resolver._calculate_next_position(boundary_pos, 1), (0, 1)) # DOWN


    def test_resolve_agent_movements_single_agent_valid_move(self):
        """Test single agent moving to a valid position."""
        grid = Grid(5)
        resolver = CollisionResolver(grid)
        grid.add_object('agent_0', (1, 1))

        agent_actions = {'agent_0': 3} # Move RIGHT to (2, 1)
        resolved_pos = resolver.resolve_agent_movements(agent_actions)

        # Check return value
        self.assertEqual(resolved_pos, {'agent_0': (2, 1)})
        # Check grid state update
        self.assertEqual(grid.get_object_position('agent_0'), (2, 1))

    def test_resolve_agent_movements_single_agent_out_of_bounds(self):
        """Test single agent attempting to move out of bounds."""
        grid = Grid(5) # Grid 0-4
        resolver = CollisionResolver(grid)
        grid.add_object('agent_0', (0, 0))

        # Attempt to move LEFT from (0,0)
        agent_actions = {'agent_0': 2}
        resolved_pos = resolver.resolve_agent_movements(agent_actions)

        # Check return value (should stay at current position)
        self.assertEqual(resolved_pos, {'agent_0': (0, 0)})
        # Check grid state update
        self.assertEqual(grid.get_object_position('agent_0'), (0, 0))

        # Attempt to move UP from (0,0)
        grid.set_object_position('agent_0', (0, 0)) # Reset position
        agent_actions = {'agent_0': 0}
        resolved_pos = resolver.resolve_agent_movements(agent_actions)
        self.assertEqual(resolved_pos, {'agent_0': (0, 0)})
        self.assertEqual(grid.get_object_position('agent_0'), (0, 0))

        # Attempt to move RIGHT from (4,4)
        grid.add_object('agent_1', (4, 4))
        agent_actions = {'agent_1': 3}
        resolved_pos = resolver.resolve_agent_movements(agent_actions)
        self.assertEqual(resolved_pos, {'agent_1': (4, 4)})
        self.assertEqual(grid.get_object_position('agent_1'), (4, 4))

        # Attempt to move DOWN from (4,4)
        grid.set_object_position('agent_1', (4, 4))
        agent_actions = {'agent_1': 1}
        resolved_pos = resolver.resolve_agent_movements(agent_actions)
        self.assertEqual(resolved_pos, {'agent_1': (4, 4)})
        self.assertEqual(grid.get_object_position('agent_1'), (4, 4))


    def test_resolve_agent_movements_two_agents_head_on_collision(self):
        """Test two agents attempting to move to the same position."""
        grid = Grid(5)
        resolver = CollisionResolver(grid)
        grid.add_object('agent_0', (1, 2))
        grid.add_object('agent_1', (3, 2))

        # agent_0 moves RIGHT to (2,2)
        # agent_1 moves LEFT to (2,2)
        agent_actions = {
            'agent_0': 3,
            'agent_1': 2
        }
        resolved_pos = resolver.resolve_agent_movements(agent_actions)

        # Check return value (both should stay at initial position due to collision)
        self.assertEqual(resolved_pos, {'agent_0': (1, 2), 'agent_1': (3, 2)})
        # Check grid state update
        self.assertEqual(grid.get_object_position('agent_0'), (1, 2))
        self.assertEqual(grid.get_object_position('agent_1'), (3, 2))

    def test_resolve_agent_movements_two_agents_swap_collision(self):
        """Test two agents attempting to swap positions."""
        grid = Grid(5)
        resolver = CollisionResolver(grid)
        grid.add_object('agent_0', (1, 2))
        grid.add_object('agent_1', (2, 2))

        # agent_0 moves RIGHT to (2,2)
        # agent_1 moves LEFT to (1,2)
        agent_actions = {
            'agent_0': 3,
            'agent_1': 2
        }
        resolved_pos = resolver.resolve_agent_movements(agent_actions)

        # Check return value (both should stay at initial position due to collision)
        self.assertEqual(resolved_pos, {'agent_0': (1, 2), 'agent_1': (2, 2)})
        # Check grid state update
        self.assertEqual(grid.get_object_position('agent_0'), (1, 2))
        self.assertEqual(grid.get_object_position('agent_1'), (2, 2))

    def test_resolve_agent_movements_agent_move_to_occupied_by_static(self):
        """Test agent attempting to move to a position occupied by a static object."""
        grid = Grid(5)
        resolver = CollisionResolver(grid)
        grid.add_object('agent_0', (1, 2))
        grid.add_object('obstacle_0', (2, 2)) # Add a static obstacle

        # agent_0 moves RIGHT to (2,2)
        agent_actions = {'agent_0': 3}
        resolved_pos = resolver.resolve_agent_movements(agent_actions)

        # Check return value (Currently, the resolver *allows* movement onto static objects)
        # If the rule was to prevent this, this test would need adjustment.
        # Based on the current implementation, the agent moves to the obstacle's position.
        self.assertEqual(resolved_pos, {'agent_0': (2, 2)})
        # Check grid state update
        self.assertEqual(grid.get_object_position('agent_0'), (2, 2))
        # The static object's position should not change
        self.assertEqual(grid.get_object_position('obstacle_0'), (2, 2))

    def test_resolve_agent_movements_multiple_agents_no_collision(self):
        """Test multiple agents moving without collision."""
        grid = Grid(5)
        resolver = CollisionResolver(grid)
        grid.add_object('agent_0', (1, 1))
        grid.add_object('agent_1', (3, 3))

        # agent_0 moves RIGHT to (2,1)
        # agent_1 moves UP to (3,2)
        agent_actions = {
            'agent_0': 3,
            'agent_1': 0
        }
        resolved_pos = resolver.resolve_agent_movements(agent_actions)

        # Check return value
        self.assertEqual(resolved_pos, {'agent_0': (2, 1), 'agent_1': (3, 2)})
        # Check grid state update
        self.assertEqual(grid.get_object_position('agent_0'), (2, 1))
        self.assertEqual(grid.get_object_position('agent_1'), (3, 2))

    def test_resolve_agent_movements_agent_stay(self):
        """Test agent choosing to stay."""
        grid = Grid(5)
        resolver = CollisionResolver(grid)
        grid.add_object('agent_0', (1, 1))

        agent_actions = {'agent_0': 4} # STAY
        resolved_pos = resolver.resolve_agent_movements(agent_actions)

        # Check return value
        self.assertEqual(resolved_pos, {'agent_0': (1, 1)})
        # Check grid state update
        self.assertEqual(grid.get_object_position('agent_0'), (1, 1))

    def test_resolve_agent_movements_non_existent_agent(self):
        """Test attempting to resolve movement for a non-existent agent."""
        grid = Grid(5)
        resolver = CollisionResolver(grid)
        grid.add_object('agent_0', (1, 1))

        # Action for a non-existent agent
        agent_actions = {'non_existent_agent': 3}

        with self.assertRaisesRegex(KeyError, "Agent with ID 'non_existent_agent' not found in grid."):
             resolver.resolve_agent_movements(agent_actions)

        # Ensure the existing agent's position didn't change
        self.assertEqual(grid.get_object_position('agent_0'), (1, 1))
