import unittest

# Assume the Grid class definition from the previous cell is available.
# from your_module import Grid, PositionType # If Grid was in a separate file
from src.Enviroments.Grid import Grid

class TestGrid(unittest.TestCase):
    """
    Test cases for the Grid class.
    """

    def test_init(self):
        """Test Grid initialization."""
        grid = Grid(10)
        self.assertEqual(grid.grid_size, 10)
        self.assertEqual(grid._object_positions, {})

    def test_add_object_success(self):
        """Test successful addition of an object."""
        grid = Grid(5)
        grid.add_object('agent_0', (1, 2))
        self.assertEqual(grid.get_object_position('agent_0'), (1, 2))
        self.assertEqual(len(grid.get_all_object_positions()), 1)

    def test_add_object_invalid_position(self):
        """Test adding an object at an invalid position."""
        grid = Grid(5)
        with self.assertRaises(ValueError):
            grid.add_object('agent_0', (5, 2)) # x out of bounds
        with self.assertRaises(ValueError):
            grid.add_object('agent_1', (1, 5)) # y out of bounds
        with self.assertRaises(ValueError):
            grid.add_object('agent_2', (-1, 2)) # x out of bounds
        with self.assertRaises(ValueError):
            grid.add_object('agent_3', (1, -1)) # y out of bounds

    def test_add_object_duplicate_id(self):
        """Test adding an object with a duplicate ID."""
        grid = Grid(5)
        grid.add_object('agent_0', (1, 2))
        with self.assertRaises(ValueError):
            grid.add_object('agent_0', (3, 4)) # Duplicate ID

    def test_get_object_position_success(self):
        """Test getting the position of an existing object."""
        grid = Grid(5)
        grid.add_object('agent_0', (1, 2))
        self.assertEqual(grid.get_object_position('agent_0'), (1, 2))

    def test_get_object_position_not_found(self):
        """Test getting the position of a non-existent object."""
        grid = Grid(5)
        with self.assertRaises(KeyError):
            grid.get_object_position('non_existent_agent')

    def test_get_all_object_positions(self):
        """Test retrieving all object positions."""
        grid = Grid(5)
        grid.add_object('agent_0', (1, 1))
        grid.add_object('goal_0', (4, 4))
        all_pos = grid.get_all_object_positions()
        expected_pos = {'agent_0': (1, 1), 'goal_0': (4, 4)}
        self.assertEqual(all_pos, expected_pos)
        # Ensure the returned dictionary is a copy
        all_pos['agent_0'] = (0, 0)
        self.assertEqual(grid.get_object_position('agent_0'), (1, 1))

    def test_is_position_occupied(self):
        """Test checking if a position is occupied."""
        grid = Grid(5)
        grid.add_object('agent_0', (1, 1))
        grid.add_object('agent_1', (1, 2))
        grid.add_object('goal_0', (4, 4))

        self.assertTrue(grid.is_position_occupied((1, 1)))
        self.assertTrue(grid.is_position_occupied((1, 2)))
        self.assertTrue(grid.is_position_occupied((4, 4)))
        self.assertFalse(grid.is_position_occupied((2, 2)))
        self.assertFalse(grid.is_position_occupied((0, 0)))

    def test_is_position_occupied_exclude_obj_id(self):
        """Test checking occupied position while excluding an object."""
        grid = Grid(5)
        grid.add_object('agent_0', (1, 1))
        grid.add_object('agent_1', (1, 1)) # Two objects at the same spot for this test

        # Without exclusion, the position is occupied
        self.assertTrue(grid.is_position_occupied((1, 1)))

        # Excluding agent_0, the position is still occupied by agent_1
        self.assertTrue(grid.is_position_occupied((1, 1), exclude_obj_id='agent_0'))

        # If only agent_0 was there, excluding it would make the position seem free
        grid_single = Grid(5)
        grid_single.add_object('agent_0', (1, 1))
        self.assertFalse(grid_single.is_position_occupied((1, 1), exclude_obj_id='agent_0'))


    def test_is_valid_position(self):
        """Test checking if a position is within grid bounds."""
        grid = Grid(5) # Grid size 5x5, valid indices 0-4

        self.assertTrue(grid.is_valid_position((0, 0)))
        self.assertTrue(grid.is_valid_position((4, 4)))
        self.assertTrue(grid.is_valid_position((2, 3)))

        self.assertFalse(grid.is_valid_position((5, 0))) # x out of bounds
        self.assertFalse(grid.is_valid_position((0, 5))) # y out of bounds
        self.assertFalse(grid.is_valid_position((-1, 0))) # x out of bounds
        self.assertFalse(grid.is_valid_position((0, -1))) # y out of bounds
        self.assertFalse(grid.is_valid_position((5, 5))) # both out of bounds
        self.assertFalse(grid.is_valid_position((-1, -1))) # both out of bounds


    def test_set_object_position_success(self):
        """Test setting the position of an existing object."""
        grid = Grid(5)
        grid.add_object('agent_0', (1, 1))
        grid.set_object_position('agent_0', (3, 3))
        self.assertEqual(grid.get_object_position('agent_0'), (3, 3))

    def test_set_object_position_not_found(self):
        """Test setting the position of a non-existent object."""
        grid = Grid(5)
        with self.assertRaises(KeyError):
            grid.set_object_position('non_existent_agent', (2, 2))

    def test_set_object_position_invalid_position(self):
        """Test setting the position of an object to an invalid location."""
        grid = Grid(5)
        grid.add_object('agent_0', (1, 1))
        with self.assertRaises(ValueError):
            grid.set_object_position('agent_0', (5, 2)) # x out of bounds
        with self.assertRaises(ValueError):
            grid.set_object_position('agent_0', (2, 5)) # y out of bounds


    def test_remove_object_success(self):
        """Test successful removal of an object."""
        grid = Grid(5)
        grid.add_object('agent_0', (1, 1))
        self.assertEqual(len(grid.get_all_object_positions()), 1)
        grid.remove_object('agent_0')
        self.assertEqual(len(grid.get_all_object_positions()), 0)
        with self.assertRaises(KeyError):
            grid.get_object_position('agent_0')

    def test_remove_object_not_found(self):
        """Test removing a non-existent object."""
        grid = Grid(5)
        with self.assertRaises(KeyError):
            grid.remove_object('non_existent_agent')

# This allows running the tests directly from the script
# if __name__ == '__main__':
#     unittest.main(argv=['first-arg-is-ignored'], exit=False)
