"""
leetcode_98 判断一棵树是不是二叉搜索树
思路：二叉搜索树的特点，根节点的值大于左孩子的值，小于右孩子的值，每一个根节点大于所有左子树的值，小于所有右子树的值
    二叉搜索树的中序遍历，即左根右，是满足有序列升序的，所以，我们可以想到两种解决方案
解法一：
中序遍历，序列是单调递增的即可，我们不需要保存中序遍历的每一个节点，我们只需要保存他的前继节点，和当前节点做比较，满足单调递增即可
解法二：
    递归法，只要每个子树，整棵树都满足 左子树的里的最大的 < root节点的值 < 右子树里面最小的，每一个根节点大于所有左子树的值，小于所有右子树的值
    即他是一颗二叉搜索树
时间复杂度 O（n）（因为每个节点我们只需要遍历一次）
注意点：python 中无穷大和无穷小，float("inf"), float("-inf")表示，可以使用嵌套函数来写，递归写法的参数可以维护一个树根，右子树中的最小值和左子树中的最大值
"""


# 解法一
class Solution1(object):
    def isValidBST(self, root):
        """
        :type root: TreeNode
        :rtype: bool
        """
        self.pre = None

        def isValidBSTCore(root):
            if root is None:
                return True
            if not isValidBSTCore(root.left):  # 二叉树的中序遍历，左中右，判断左子树是否满足bst
                return False
            if self.pre and self.pre.val >= root.val:  # 如果前驱节点的值大于等于根节点，则不满足升序，则不是二叉搜索树
                return False
            self.pre = root  # 前驱节点跟过来
            return isValidBSTCore(root.right)  # 判断右子树是否满足bst

        return isValidBSTCore(root)


# 解法二
class Solution2(object):
    def isValidBST(self, root):
        """
        :type root: TreeNode
        :rtype: bool
        """

        def isValidBSTCore(root, min, max):
            # 如果是一颗空树，则他是bst树
            if root == None:
                return True
            # 如果他的左子树的最大值不为空并且大于root的值,则他不是bst树
            if min is not None and min >= root.val:
                return False
            # 如果他的右子树最小值不为空并且小于root的值,则他不是bst树
            if max is not None and max <= root.val:
                return False
            # 递归，判断他的左子树和右子树是否都满足
            return isValidBSTCore(root.left, min, root.val) and isValidBSTCore(root.right, root.val, max)

        return isValidBSTCore(root, float("-inf"), float("inf"))


# leetcode
# 235
#
# leetcode
# 236
# """
