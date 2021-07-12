import math
import cv2
import random
import numpy as np


class area:
    def __init__(self, num_node):
        self.parent = [i for i in range(num_node)]
        self.size = [1 for _ in range(num_node)]
        self.num_set = num_node

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
    print('划分的区域个数为', res.num_set)
    return res


# 生成结果图
def generate_image(res, width, height, area):
    # 随机生成颜色
    color = [(int(random.random() * 255), int(random.random() * 255),int(random.random() * 255)) for i in range(width * height)]
    save_img = np.zeros((height, width, 3), np.uint8)
    for y in range(height):
        for x in range(width):
            color_idx = res.find(y * width + x)
            area.append(color_idx)
            save_img[y, x] = color[color_idx]
    return save_img


# 生成区域标记后的图片并计算IOU
def cal_IOU(img_gt, res, width, height, area):
    save_img = np.zeros((height, width, 3), np.uint8)
    # 划分出的所有区域
    area = list(set(area))
    # 每个区域遍历
    for i in range(len(area)):
        # 这个区域的面积
        total_area = 0
        # 在前景图中是前景的面积
        gt_area = 0
        # 寻找这个区域中是前景的点
        for y in range(height):
            for x in range(width):
                if(res.find(y * width + x) == area[i]):
                    if(img_gt[y, x, 0] > 128):
                        gt_area += 1
                    total_area += 1
        # 如果前景占比大于一半则在标记图中标为全白
        if(float(gt_area/total_area) > 0.5):
            for y in range(height):
                for x in range(width):
                    if(res.find(y * width + x) == area[i]):
                        save_img[y, x] = (255, 255, 255)
        # 否则不是前景则全黑
        else:
            for y in range(height):
                for x in range(width):
                    if(res.find(y * width + x) == area[i]):
                        save_img[y, x] = (0, 0, 0)
    # 计算IOU
    IOU = 0.0
    R1_and_R2 = 0
    R1_or_R2 = 0
    for y in range(height):
        for x in range(width):
            if(save_img[y, x, 0] > 128 and img_gt[y, x, 0] > 128):
                R1_and_R2 += 1
            if(save_img[y, x, 0] > 128 or img_gt[y, x, 0] > 128):
                R1_or_R2 += 1
    IOU = (float)(R1_and_R2/R1_or_R2)
    return IOU, save_img


if __name__ == "__main__":
    # 读取哪张图片在下面更改即可
    img_number = "952"
    # 读取相应编号原图和前景图
    img = cv2.imread("../../data/imgs/"+img_number+".png")
    img_gt = cv2.imread("../../data/gt/"+img_number+".png")
    # 先在当前目录生成原图和前景图
    cv2.imwrite(img_number+"_origin.png", img)
    cv2.imwrite(img_number+"_gt.png", img_gt)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    height, width, channel = img.shape
    img = np.asarray(img, dtype=float)
    # 分开组成grb
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
    res = segment_graph(sorted_graph, width * height, 5)
    res = remove_small_component(res, sorted_graph, min_size)
    # 生成结果图
    area = []
    img = generate_image(res, width, height, area)
    IOU, img2 = cal_IOU(img_gt, res, width, height, area)
    print(f"IOU的值为{IOU}")
    # 将IOU的值写到区域标记图的左上角
    cv2.putText(img2, "IOU=%.4f" % IOU, (5, 15), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)
    # 显示图片
    cv2.imshow('result', img)
    cv2.imshow('区域标记', img2)
    # 按序号保存图片
    cv2.imwrite(img_number+'_result.png', img)
    cv2.imwrite(img_number+'_mark.png', img2)
    cv2.waitKey(0)
    cv2.destroyAllWindows()