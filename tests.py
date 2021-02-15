"""Tests"""
import unittest
from unittest.mock import patch
import pandas as pd
from utils import RandomPolygons


class BaseCase(unittest.TestCase):
    """
    BaseCase class for unittest tests
    """
    def setUp(self):
        self.test_polygons = RandomPolygons(num_polygons=2, num_edges=3, total_num_polygons=3)

    def test_init_data(self):
        dict_data = [{'id': 0, 'x0': 0, 'x1': 1, 'x2': 2, 'y0': 0, 'y1': 1, 'y2': 2},
                     {'id': 1, 'x0': 1, 'x1': 2, 'x2': 4, 'y0': 3, 'y1': 2, 'y2': 4},
                     {'id': 2, 'x0': 4, 'x1': 5, 'x2': 1, 'y0': 4, 'y1': 7, 'y2': 10},
                     {'id': 3, 'x0': 7, 'x1': 10, 'x2': -1, 'y0': 0, 'y1': 3, 'y2': 10}]
        self.test_polygons.df = pd.DataFrame(dict_data)

    def tearDown(self):
        pass


class TestOperations(BaseCase):
    """
    TestOperations class to test operations on polygons
    """
    def test_delete_lowest(self):
        self.test_init_data()
        response = self.test_polygons.delete_lowest().to_dict()
        assert response.get('id', None) == 0

    def test_get_poly_by_index(self):
        self.test_init_data()
        with patch('builtins.input', return_value='1'):
            response = self.test_polygons.get_poly_by_index()
            assert all(response.index == 1)

    def test_move_poly(self):
        self.test_init_data()
        with patch('builtins.input', return_value='1'):
            response = self.test_polygons.move_poly(xt=1000, yt=1000)
            assert all(response.index == 1)
            assert all(response.x0 == 1001)

    def test_delete_poly(self):
        self.test_init_data()
        with patch('builtins.input', return_value='1') as e:
            response = self.test_polygons.delete_poly()
            assert int(e.return_value) not in response.index

    @patch('builtins.input', side_effect=['1', '0'])
    def test_replace_poly_by_id(self, mock_input):    # pylint: disable=W0613
        self.test_init_data()
        response = self.test_polygons.replace_poly_by_id()
        # we verify that the old id 1 is no longer in the collection
        assert 1 not in response.id.values

    def test_get_poly_between(self):
        self.test_init_data()
        response = self.test_polygons.get_poly_between(y_limit_min=3, y_limit_max=11)
        assert all(response.index == 2)
