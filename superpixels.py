# -*- coding: utf-8 -*-
"""
Created on Thu Jul 22 00:20:39 2021

@author: adeju
"""

import numpy as np
from PIL import Image
import math
# from skimage.segmentation import slic, mark_boundaries
from numba import jit
# import time


def paintSuperpixel(superpixel, target, image=None, index=None,
                    color=[255, 255, 255]):
    '''
    Paint certain pixels according to superpixel index given.

    Parameters
    ----------
    superpixel : TYPE
        Superpixel image with indexes for each pixel.
    target : TYPE
        Image to be painted with specific superpixel.
    image : TYPE, optional
        DESCRIPTION. The default is None.
    index : TYPE, optional
        DESCRIPTION. The default is None.
    color : TYPE, optional
        DESCRIPTION. The default is [255,255,255].

    Returns
    -------
    target : TYPE
        DESCRIPTION.

    '''

    target[:, :, 0] = np.where((superpixel == index), color[0],
                               target[:, :, 0])
    target[:, :, 1] = np.where((superpixel == index), color[1],
                               target[:, :, 1])
    target[:, :, 2] = np.where((superpixel == index), color[2],
                               target[:, :, 2])

    return target


def computeSuperpixelColor(labImage, superpixel):
    '''
    Compute the centroid color in a CIELab space for each superpixel.

    Parameters
    ----------
    image : TYPE
        The reference image in CIELab colors.
    superpixel : TYPE
        The superpixel image with the indexes for each pixel.

    Returns
    -------
    superpixelColor : TYPE
        Array of CieLAB colors for each Superpixel.

    '''

    superpixelColor = [[0, 0, 0]]

    for i in range(1, superpixel.max()+1):
        color = np.median(labImage[np.where(superpixel == i)], axis=0).tolist()
        superpixelColor.append(color)

    return superpixelColor


def superpixels_to_graph(superpixel):

    offset = 1
    neighbors = np.full((superpixel.max()+1, superpixel.max()+1), False)
    for i in range(1, np.shape(superpixel)[0]-2):
        for j in range(1, np.shape(superpixel)[1]-2):
            temp = superpixel[i-offset:i+offset+1,
                              j-offset:j+offset+1].flatten()
            for k in temp:
                if k != superpixel[i, j]:
                    neighbors[superpixel[i, j]][k] = True
                    neighbors[k][superpixel[i, j]] = True

    A = []
    for i in range(0, superpixel.max()):
        temp = neighbors[i:i+1, :]
        A.append(np.unique(np.sort(np.where(temp == True)[1])))

    return A


def compute_color_distance(color_0, color_1):

    L0, a0, b0 = color_0
    L1, a1, b1 = color_1

    dist = math.sqrt(math.pow((L1-L0), 2) + math.pow((a1-a0), 2)
                     + math.pow((b1-b0), 2))

    return dist


def superpixel_region_growing(A, seed, I, T, visited):

    # visited = np.full((len(A)+1), False)
    # visited = np.full((A.max()+1), False)
    tempColor = I[seed]

    queue = []
    queue.append(seed)
    # visited[seed] = True

    backtrack = []
    backtrack.append(seed)

    while queue:  # queue
        s = queue.pop(0)
        temp = A[s]
        for k in temp:
            if np.size(tempColor) == 3:
                dist = compute_color_distance(I[k], tempColor)
            else:
                dist = compute_color_distance(I[k],
                                              np.median(tempColor, axis=0))

            if dist < T:
                if visited[k] == False:  # or superpixelClass[k] == 0:
                    queue.append(k)
                    # print("Push " + str(k))
                    tempColor = np.insert(tempColor, 0,
                                          I[k], 0)
                    tempColor = np.reshape(tempColor,
                                           (int(np.size(tempColor)/3), 3))
                    backtrack.append(k)

                    # superpixelClass[k] = classe
            visited[k] = True
        # print(np.size(queue))
        # if np.size(queue) > 50:
        #     # print(queue)
        #     queue = []
        
    return backtrack


def superpixel_forest_segmentation(A, seed, I, T):

    unvisited = list(range(len(A)))

    path_cost = np.full((np.size(unvisited), 2), np.inf)

    path_cost[seed, 0] = 0

    while np.size(unvisited) != 0:

        current_vertex = unvisited[np.argmin(path_cost[unvisited, 0])]

        unvisited_neighbour = np.intersect1d(A[current_vertex], unvisited)

        for i in unvisited_neighbour:

            dist = compute_color_distance(I[current_vertex], I[i])

            temp = max(dist, path_cost[current_vertex, 0])

            # if temp < np.inf: print(dist, path_cost[current_vertex, 0], temp)

            # temp = dist + path_cost[current_vertex, 0]

            if temp < T:
                path_cost[i] = temp, current_vertex

        # visited.append(current_vertex)
        unvisited.remove(current_vertex)

    backtrack = np.where(path_cost[:, 0] < np.inf)[0]

    return backtrack


# @jit(nopython=True)
def forest_segmentation2(A, seed, end, I, path_cost=None):
    # A = neighbors
    # seed = target
    # I = superpixelColor
    # T = maxDist

    if path_cost is None:
        visited = []
        unvisited = list(range(np.shape(np.asarray(A))[0]))

        path_cost = np.full((np.size(unvisited), 2), np.inf)

        path_cost[seed, 0] = 0

        while np.size(unvisited) != 0:

            current_vertex = unvisited[np.argmin(path_cost[unvisited, 0])]

            unvisited_neighbour = np.intersect1d(A[current_vertex], unvisited)

            for i in unvisited_neighbour:

                dist = compute_color_distance(I[current_vertex], I[i])

                temp = max(dist, path_cost[current_vertex, 0])

                # if temp < np.inf: print(dist, path_cost[current_vertex, 0], temp)

                # temp = dist + path_cost[current_vertex, 0]

                if temp < path_cost[i, 0]:
                    path_cost[i] = temp, current_vertex

            visited.append(current_vertex)
            unvisited.remove(current_vertex)

    backtrack = np.where(path_cost[:, 0] < path_cost[end, 0])[0]

    return backtrack, path_cost


def least_path(A, start, end, I):

    visited = []
    unvisited = list(range(np.shape(A)[0]))

    path_cost = np.full((np.size(unvisited), 2), np.inf)

    # start = 0
    path_cost[start, 0] = 0

    while np.size(unvisited) != 0:

        current_vertex = unvisited[np.argmin(path_cost[unvisited, 0])]

        if current_vertex == end:
            break

        unvisited_neighbour = np.intersect1d(
            A[current_vertex], unvisited)

        for i in unvisited_neighbour:
            dist = compute_color_distance(I[current_vertex],
                                          I[i])
            if dist + path_cost[current_vertex, 0] < path_cost[i, 0]:
                path_cost[i] = (dist +
                                path_cost[current_vertex, 0]), current_vertex

        visited.append(current_vertex)
        unvisited.remove(current_vertex)

    # backtrack
    backtrack = []
    backtrack.append(current_vertex)
    while current_vertex != start:

        current_vertex = np.uint(path_cost[current_vertex][1])
        backtrack.append(current_vertex)

    return np.asarray(backtrack)


def showImage(image):
    '''
    Open a image with the OS image viewer

    Parameters
    ----------
    image : TYPE
        Numpy array of a RGB or grayscale image.

    Returns
    -------
    None.

    '''

    pil_img = Image.fromarray(image)
    pil_img.show()

    return
