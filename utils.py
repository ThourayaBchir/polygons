"""This provides a class RandomPolygons to create and manipulate polygons"""
import math
import logging
import numpy as np
import pandas as pd
from numpy import pi
import matplotlib.pyplot as plt
from matplotlib.collections import PolyCollection
import matplotlib as mpl


logging.basicConfig(level=logging.INFO)
logging.disable(logging.DEBUG)
logger = logging.getLogger(__name__)


class RandomPolygons:
    """
    This class helps to generate random initial Polygons and to do some operations on these polygons.
    arguments:
        num_polygons: the number of different polygons and is num_polygons >= 1
        num_edges: the number of edges of one polygon
        total_num_polygons: the number of polygons after replication and total_num_polygons >= num_polygons
        columns_x: the list containing x coordinates labels, like [x0, x2, x3 ...]
        columns_y: the list containing y coordinates labels, like [y0, y2, y3 ...]
        columns: the whole coordinates labels list
        df: is a Pandas dataframe that will contain the coordinates of polygons and is generated randomely
        from previous arguments
    """
    def __init__(self, num_polygons, num_edges, total_num_polygons):
        self.num_polygons = num_polygons  # initial number polygons that we will duplicate and translate or rotate
        self.num_edges = num_edges
        self.total_num_polygons = total_num_polygons
        self.columns_x = ['x'+str(i) for i in range(num_edges)]
        self.columns_y = ['y'+str(i) for i in range(num_edges)]
        self.columns = self.columns_x + self.columns_y
        self.df = None
        logger.info('Generating data ...')

    def initialize_data(self):
        """
        It generates a dataframe with random polygons coordinates
        df columns are like [x0, x1, x2 ..., y0, y1, y2 ...]
        Each row contains the coordinates of a single polygon
        """
        self.df = pd.DataFrame(np.random.randint(0, 200, size=(self.num_polygons, 2 * self.num_edges)),
                               columns=self.columns)
        logger.info('Creating a basic random collection of {} polygons of {} edges ...'.
                    format(self.num_polygons, self.num_edges))

    def replicate_random_polygons(self):
        """
        It takes the df dataframe and replicate random rows to get a final number of row of total_num_polygons
        """
        k = self.total_num_polygons - len(self.df.index)
        self.df = self.df.append(self.df.sample(n=k, replace=True))
        logger.info('Replicating random polygons to get {} in total ...'.
                    format(self.total_num_polygons))

    def translate_random_polygons(self):
        """
        It applies random translation to all the polygons
        """
        length = len(self.df)
        self.df = self.df.assign(xt=np.random.randint(0, 200, size=(length, 1)))
        self.df = self.df.assign(yt=np.random.randint(0, 200, size=(length, 1)))
        for col in self.columns_x:
            self.df[col] = self.df.apply(lambda x, c=col: x[c]+x.xt, axis=1)
        for col in self.columns_y:
            self.df[col] = self.df.apply(lambda x, c=col: x[c]+x.yt, axis=1)
        logger.info('Random translation of some polygons ...')

    @staticmethod
    def rotate(origin, point, angle=pi):
        """
        Rotate a point counterclockwise by a given angle around a given origin.
        The angle is in radians.
        """
        x, y = origin
        xp, yp = point
        xf = x + math.cos(angle) * (xp - x) - math.sin(angle) * (yp - y)
        yf = y + math.sin(angle) * (xp - x) + math.cos(angle) * (yp - y)
        return xf, yf

    def rotate_random_polygons(self):
        """
        Rotates a random fraction of polygons with 180°
        """
        df_to_rotate = self.df.sample(frac=.5)
        df_tmp = pd.DataFrame()
        for xi, yi in zip(self.columns_x, self.columns_y):
            df_tmp[[xi, yi]] = df_to_rotate.apply(lambda x, x_i=xi, y_i=yi: self.rotate((x['x0'], x['y0']),
                                                                                        (x[x_i], x[y_i])),
                                                  axis=1,
                                                  result_type="expand")

        self.df = self.df.append(df_tmp).drop(['xt', 'yt'], axis=1)
        self.df = self.df.reset_index().rename(columns={'index': 'id'})
        logger.info('Random rotation of some polygons by 180° ...')

    def drop_polygons_with_negative_y(self):
        """
        It drops polygons with negative y coordinate
        """
        for yi in self.df.columns[self.num_edges+1:]:
            dfn = self.df.loc[self.df[yi]<0]
            self.df.drop(dfn.index, axis=0, inplace=True)

    def plot_polygons(self):
        """
        Plot the polygons
        """
        coord = self.df.apply(lambda x: list(zip(
            [x[xi] for xi in self.df.columns[1:self.num_edges+1]],
            [x[xi] for xi in self.df.columns[self.num_edges+1:]])) , axis=1).to_list()

        z = np.random.random(self.total_num_polygons) * 500
        fig, ax = plt.subplots()
        coll = PolyCollection(coord, array=z, cmap=mpl.cm.jet, edgecolors='magenta')
        ax.add_collection(coll)
        ax.autoscale_view()
        fig.colorbar(coll, ax=ax)
        plt.show()

    def delete_lowest(self):
        """
        It takes the previously generated dataframe df of polygons and deletes the polygn with lowest edge.
        It deletes the first occurrence of the lowest y
        :return: df_low: a dataframe containing only the deleted polygon
        """
        # we find the min for each polygon
        self.df['y_min'] = self.df[self.df.columns[self.num_edges+1:]].apply(
            lambda x: min([x[xi] for xi in self.df.columns[self.num_edges+1:]]), axis=1)

        # then find the index of the min of these y_min
        index_min = self.df['y_min'].idxmin()

        self.df.drop('y_min', axis=1, inplace=True)
        df_low = self.df.loc[index_min]
        self.df.drop(index_min, inplace=True)
        return df_low

    def get_poly_by_index(self):
        """
        This method will ask the user to input an index from a list.
        Note: indexes are unique
        :returns: a df containg the polygon of the specified index
        """
        index = self.get_id_input()
        if index:
            return self.df.loc[self.df.index == index]
        return None

    def get_id_input(self, index_mode=True):
        """
        This is a helper method to handle the input data from user
        """
        if index_mode:
            index = input('Enter an index from {} for the polygon: '.format(set(self.df.index)))
            if index.isdigit():
                return int(index)
            logger.info('You need to enter integer from the list.')
            return None
        else:
            id1 = input('Enter the polygon id to replace from {} for the polygon: '.format(set(self.df.id)))
            id2 = input('Enter the new polygon id to from {} for the polygon: '.format(set(self.df.id)))
            if id1.isdigit() and id2.isdigit():
                return int(id1), int(id2)
            logger.info('You need to enter integers from the list.')
            return None

    def move_poly(self, xt=1000, yt=1000):
        """
        It applies a translation to the polygon corresponding to the specified index by the user by (xt, yt) vector
        :returns: a df containing the new polygon
        """
        index = self.get_id_input()
        if index:
            self.df.loc[self.df.index == index, self.columns_x] = \
                self.df.loc[self.df.index == index].apply(lambda x: x[self.columns_x]+xt, axis=1)
            self.df.loc[self.df.index == index, self.columns_y] = \
                self.df.loc[self.df.index == index].apply(lambda x: x[self.columns_y]+yt, axis=1)
            return self.df.loc[self.df.index == index]
        return None

    def delete_poly(self):
        """
        :returns: a df containg the deleted polygon of the specified index
        """
        index = self.get_id_input()
        if index and index in self.df.index:
            return self.df.drop(index)
        logger.info('The entered index {} is not from the list {} .'.format(index, self.df.index))
        return None

    def replace_poly_by_id(self):
        """
        It takes the following inputs from user:
            old_id: is th id of the polygon we want to replace
            new_id: is the id of the polygon
        :returns: the new dataframe df after replacement
        """
        ids = self.get_id_input(index_mode=False)
        if ids and len(ids) == 2 and set(ids).issubset(set(self.df.index)):
            old_id, new_id = ids
            df_new = self.df.loc[self.df.id == new_id].head(1)
            df_tmp = pd.DataFrame()
            df_tmp = df_tmp.append([df_new]*len(df_new), ignore_index=True)
            self.df.loc[self.df.id == old_id] = df_tmp.values
            return self.df
        logger.info('Choose integers from the list {}.'.format(self.df.index))
        return None

    def get_poly_between(self, y_limit_min=100, y_limit_max=500):
        """
        :param y_limit_min: the y limit to get polygons under it
        :param y_limit_max: the y limit to get polygons above it
        :return: a dataframe with the polygons satisfying these conditions
        """
        self.df['y_min'] = self.df[self.df.columns[self.num_edges:]].apply(
            lambda x: min([x[xi] for xi in self.df.columns[self.num_edges+1:]]), axis=1)

        self.df['y_max'] = self.df[self.df.columns[self.num_edges:]].apply(
            lambda x: max([x[xi] for xi in self.df.columns[self.num_edges+1:]]), axis=1)

        return self.df.loc[self.df['y_min'] >= y_limit_min].loc[self.df['y_max'] <= y_limit_max].\
            drop(['y_min', 'y_max'], axis=1)
