class Node:
    def __init__(self, value: int, left=None, right=None):
        self.value = value
        self.left = left
        self.right = right

def build_tree_from_dict(node_id: str, tree_dict: dict) -> Node or None:
    """根据字典表示的二叉树构建 Node 对象，'#' 表示空节点"""
    if node_id == "#" or node_id not in tree_dict:
        return None
    left_id, right_id = tree_dict[node_id]
    left_node = build_tree_from_dict(left_id, tree_dict)
    right_node = build_tree_from_dict(right_id, tree_dict)
    return Node(int(node_id), left_node, right_node)

binary_tree = {
    "1": ["2", "3"],      # 根节点 1，左子节点 2，右子节点 3
    "2": ["4", "5"],      # 节点 2，左子节点 4，右子节点 5
    "3": ["6", "7"],      # 节点 3，左子节点 6，右子节点 7
    "4": ["#", "#"],      # 叶子节点（# 表示空）
    "5": ["8", "#"],      # 节点 5，左子节点 8，右子节点为空
    "6": ["#", "#"],
    "7": ["9", "10"],
    "8": ["#", "#"],
    "9": ["#", "#"],
    "10": ["#", "#"]
}
"""
        1
       / \
      2   3
     / \ / \
    4  5 6  7
      /    / \
     8    9  10
"""
root = build_tree_from_dict("1", binary_tree)
# 递归算法

## 前序遍历：root -> left -> right
def preorder(node_id):
    if node_id == "#" or node_id not in binary_tree:
        return
    print(node_id)
    left, right = binary_tree[node_id]
    preorder(left)
    preorder(right)

## 中序遍历：左 → 根 → 右
def inorder(node_id):
    if node_id == "#" or node_id not in binary_tree:
        return 
    left, right = binary_tree[node_id]
    inorder(left)
    print(node_id)
    inorder(right)


## 后序遍历：左 → 右 → 根
def postorder(node_id):
    if node_id == "#" or node_id not in binary_tree:
        return 
    left, right = binary_tree[node_id]
    postorder(left)
    postorder(right)
    print(node_id)

# 栈法：先压右 再压左
def preorder_stack(root_id):
    if root_id == "#" or root_id not in binary_tree:
        return
    stack = [root_id]
    while stack:
        node = stack.pop()
        print(node, end=" ")
        left, right = binary_tree[node]
        # 先压右，再压左（保证左先出）
        if right != "#":
            stack.append(right)
        if left != "#":
            stack.append(left)

if __name__ == "__main__":
    preorder_stack("1")