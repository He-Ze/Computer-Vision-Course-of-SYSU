import math
import cv2
import random
import numpy as np


class area:
    def __init__(self, num_node):
        self.parent = [i for i in range(num_node)]
        self.size = [1 for _ in range(num_node)]
        self.num_set = num_node
        self.num = num_node

    # 寻找根节点
    def find(self, u):
        if self.parent[u] == u:
            return u
        self.parent[u] = self.find(self.parent[u])
        return self.parent[u]

    # 合并两个节点
    def merge(self, u, v):
        u = self.find(u)
        v = self.find(v)  
        # 若两个父节点不同
        if u != v:
            # 子节点多的当父节点
            if self.size[u] > self.size[v]:
                self.parent[v] = u 
                self.size[u] += self.size[v]
                # 被合并的根节点的子节点数目变为1
                self.size[v] = 1  
            else:
                self.parent[u] = v 
                self.size[v] += self.size[u]
                self.size[u] = 1 
            self.num_set -= 1

    def all_area(self):
        res=[]
        i,n=0,0
        while n != self.num_set:
            if self.parent[i] == i:
                res.append(i)
                n += 1
            i += 1
        return res

    def one_area(self, x):
        root = self.find(x)
        res = []
        for i in range(self.num):
            if self.find(i) == root:
                res.append(i)
        return res


# 计算RGB距离并创建图的边
def create_edge(img, width, x1, y1, x2, y2):
    # 计算RGB距离
    r = math.pow((img[0][y1, x1] - img[0][y2, x2]), 2)
    g = math.pow((img[1][y1, x1] - img[1][y2, x2]), 2)
    b = math.pow((img[2][y1, x1] - img[2][y2, x2]), 2)
    return (y1 * width + x1, y2 * width + x2, math.sqrt(r + g + b))


# 建立图结构
def build_graph(img, width, height):
    graph = []
    for y in range(height):
        for x in range(width):
            if x < width - 1:
                graph.append(create_edge(img, width, x, y, x + 1, y))
            if y < height - 1:
                graph.append(create_edge(img, width, x, y, x, y + 1))
            if x < width - 1 and y < height - 1:
                graph.append(create_edge(img, width, x, y, x + 1, y + 1))
            if x < width - 1 and y > 0:
                graph.append(create_edge(img, width, x, y, x + 1, y - 1))
    return graph


# 分割
def segment_graph(sorted_graph, num_node, k):
    res = area(num_node)
    # 类内不相似度
    threshold = [k] * num_node
    for edge in sorted_graph:
        u = res.find(edge[0])
        v = res.find(edge[1])
        w = edge[2]
        # 如果两个节点的父节点不相同则不属于同一类
        if u != v:  
            # 如果边的权重小于阈值
            if w <= threshold[u] and w <= threshold[v]:
                # 合并两个节点
                res.merge(u, v)  
                parent = res.find(u)
                # 更新最大类内间距
                threshold[parent] = np.max([w, threshold[u], threshold[v]]) + k / res.size[parent]
    return res


# 移除面积过小的区域
def remove_small_component(res, sorted_graph, min_size):
    for edge in sorted_graph:
        u = res.find(edge[0])
        v = res.find(edge[1])
        if u != v:
            if res.size[u] < min_size or res.size[v] < min_size:
                res.merge(u, v)
    return res



def segment(img):
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    height, width, _ = img.shape
    img = np.asarray(img, dtype=float)
    img = cv2.split(img)
    # 建立图结构
    graph = build_graph(img, width, height)
    # 按照权重进行不减的排序
    def weight(edge): return edge[2]
    # 根据权重对所有的边进行排序
    sorted_graph = sorted(graph, key=weight)
    # 每个区域最小面积
    min_size = 50
    # 分割
    res = segment_graph(sorted_graph, width * height, 1)
    res = remove_small_component(res, sorted_graph, min_size)
    return res
