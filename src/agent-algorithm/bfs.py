
"""
核心思想：

    从根节点开始，一层一层地访问所有节点。

    先访问离根最近的节点（第1层），再访问第2层、第3层……

    同一层内，一般按照从左到右的顺序访问。

数据结构：队列（Queue）

    队列是先进先出（FIFO）的结构。

    把当前层的节点放入队列，然后依次弹出，同时把它们的子节点（下一层）加入队列尾部。

    这样可以保证先入队的节点（上层）先被处理，从而实现“逐层”效果。

题目：
    二叉树的层序遍历（LeetCode 102）	返回按层分组的结果	标准 BFS，记录每层大小
    二叉树的层序遍历 II（LeetCode 107）	从底向上输出层序	BFS 后反转结果列表
    二叉树的右视图（LeetCode 199）	从右边看二叉树，返回每层最右边的节点	BFS 时，记录每层最后一个节点
    二叉树的最大宽度（LeetCode 662）	计算树的最大宽度（包括空节点）	BFS 时给每个节点编号（位置索引）
    在每个树行中找最大值（LeetCode 515）	每层的最大值	BFS 并记录每层最大值
    二叉树的最小深度（LeetCode 111）	根到最近叶子节点的最短路径节点数	BFS 找到第一个叶子节点时立即返回
    填充每个节点的下一个右侧节点指针（LeetCode 116, 117）	为每个节点添加 next 指针指向右边节点	BFS 层序遍历，同层内串联
    N 叉树的层序遍历（LeetCode 429）	多叉树的层序遍历	BFS，每个节点有多个孩子，全部入队
"""
from collections import deque

binary_tree = {
    "1": ["2", "3"],
    "2": ["4", "5"],
    "3": ["6", "7"],
    "4": ["#", "#"],
    "5": ["8", "#"],
    "6": ["#", "#"],
    "7": ["9", "10"],
    "8": ["#", "#"],
    "9": ["#", "#"],
    "10": ["#", "#"]
}

def bfs_level_order(root_id):
    """BFS 层序遍历（输出节点值）"""
    if root_id == "#" or root_id not in binary_tree:
        return
    
    queue = deque([root_id])
    
    while queue:
        node = queue.popleft()          # 取出队首节点
        print(node, end=" ")            # 访问该节点
        
        left, right = binary_tree[node]
        if left != "#":                 # 左子节点存在则加入队列
            queue.append(left)
        if right != "#":                # 右子节点存在则加入队列
            queue.append(right)

# 调用
bfs_level_order("1")