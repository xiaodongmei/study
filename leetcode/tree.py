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


"""
二叉树的层次遍历 leetcode_102 leetcode_103 leetcode_107
思路：可以利用队列（python中的可以用list替代）,queue如果当前队列里还有值，说明还有
需要遍历的节点，用一个list ll来存取下一层需要遍历的节点，在每次循环结尾让queue = ll，ll = [],遍历queue里面的
值，ll又重新收集下一层需要遍历的节点，直到queue为空，说明整棵树遍历结束，已经没有再需要遍历的了
103和107是102的变形，差别不大，103需要注意，在奇数层正序，偶数层需要把序列逆序下，再存入结果列表，我们用一个
tmp存临时遍历，避免对queue变量本身的改动，107的话就是BFS遍历完了之后，将结果逆序返回
注意点：用index可控制层，在103题，需要tmp存临时遍历，避免对queue变量本身的改动
"""


# leetcode_102

class Solution1(object):
    def levelOrder(self, root):
        """
        :type root: TreeNode
        :rtype: List[List[int]]
        """
        if not root:
            return []

        res = []
        queue = [root]

        while queue:
            res.append([node.val for node in queue])
            ll = []
            for node in queue:
                if node.left:
                    ll.append(node.left)
                if node.right:
                    ll.append(node.right)
            queue = ll
        return res


# leetcode_103
class Solution2(object):
    def zigzagLevelOrder(self, root):
        """
        :type root: TreeNode
        :rtype: List[List[int]]
        """
        if not root:
            return []
        res = []
        queue = [root]
        index = 1

        while queue:
            tmp = queue
            if not index & 1:
                tmp = tmp[::-1]
            res.append([node.val for node in tmp])
            index += 1
            ll = []
            for node in queue:
                if node.left:
                    ll.append(node.left)
                if node.right:
                    ll.append(node.right)
            queue = ll
        return res


"""
leetcode_236 二叉树的最近公共祖先
求二叉树的最近公共祖先，首先我们确定这颗树是不是二叉搜索树，如果是二叉搜索树，
我们可以利用二叉搜素树的特性，每个子树都满足，根节点的值大于整个左子树的值，小于整个有右子树的值，每个子树节点。根节点比
左孩子大，比右孩子小，
所以，求二叉搜索树的最近公共祖先，我们只要找到一个树在这两个数之间，如果根节点的值比这两个节点的值都大，则说明这两个节点的 
最低公共祖先在左子树，我们需要递归在左子树继续找，如果说这根节点的值比这两个节点的值都小，则说明这两个节点的最低公共祖先在
右子树，我们需要递归的在右子树继续找，否则，则说明根节点的值在他们两个之间，最低公共祖先就是根节点，返回根节点

如果他不是二叉搜索树，那这颗二叉树有指向前驱的（父节点）的指针吗。如果有那就从底向上遍历这，类似于路径，遇到第一个相同的节点，即就是
他们的最低公共祖先

那如果他就是一颗普通的二叉树呢，我们可以有两种思路，一种是可以从上往下找，找到这两个节点的路径，然后逆序下，然后找他们的第一个公共节点
第二种的话，就是递归，如果p,q节点一个在左子树，一个在右子树，那么他们的最低公共祖先就是根节点，如果p,q节点都在左子树，那么他们的最低公共祖先
就在左子树中找，如果p,q节点都在右子树，那么他们的最低公共祖先就在右子树中找，如果p节点或q节点就是根节点或者根节点为空，就返回root根节点就是
他们的最低公共祖先，如果p,q在左右子树中都没有找到，则返回空或直接返回
"""


# leetcode_235 二叉搜索树的最近公共祖先
class Solution(object):
    def lowestCommonAncestor(self, root, p, q):
        """
        :type root: TreeNode
        :type p: TreeNode
        :type q: TreeNode
        :rtype: TreeNode
        """
        if not root:
            return root
        if root.val > p.val and root.val > q.val:
            return self.lowestCommonAncestor(root.left, p, q)
        elif root.val < p.val and root.val < q.val:
            return self.lowestCommonAncestor(root.right, p, q)
        return root


# leetcode_236 二叉树的最近公共祖先

class Solution2(object):
    def lowestCommonAncestor(self, root, p, q):
        """
        :type root: TreeNode
        :type p: TreeNode
        :type q: TreeNode
        :rtype: TreeNode
        """
        if not root or root is p or root is q:
            return root
        left = self.lowestCommonAncestor(root.left, p, q)
        right = self.lowestCommonAncestor(root.right, p, q)
        if not left and not right:
            return
        if not left:
            return right
        if not right:
            return left
        return root


"""
leetcode_113 找二叉树中和为n的路径
给你二叉树的根节点 root 和一个整数目标和 targetSum ，找出所有 从根节点到叶子节点 路径总和等于给定目标和的路径

思路：其实就是二叉树的先序遍历，访问当前节点，把它加进路径，并用target的值减去本节点的值，判断他是否为叶子节点。即没有左孩子也
    没有右孩子，如果已经是叶子节点，说明一条路径形成，判断此时target是否为0（或者类似我解法1的写法，判断路径里节点和的总数，较为低效），
    如果为0，则把这条理解加入结果集，说明满足条件，然后继续递归遍历
    如果当前节点不是叶子节点，则继续访问他的左子树和右子树，最后，返回最终的路径结果
注意点：
    python list的神拷贝和浅拷贝问题，list.append([:])是深拷贝，还有题目的要求是返回路径节点的值列表，
    还有就是可以使用target 差值来，避免sum计算
时间复杂度 O(n^2) 空间复杂度 O（n）
"""


class Solution(object):
    def pathSum(self, root, target):
        """
        :type root: TreeNode
        :type target: int
        :rtype: List[List[int]]
        """
        if not root:
            return []
        res, path = [], []

        def pathSumCore(root, target):
            if not root:
                return
            path.append(root.val)
            if not root.left and not root.right:
                if sum([node for node in path]) == target:
                    res.append(path[:])
            if root.left:
                pathSumCore(root.left, target)
            if root.right:
                pathSumCore(root.right, target)
            path.pop()

        pathSumCore(root, target)
        return res


class Solution(object):
    def pathSum(self, root, target):
        """
        :type root: TreeNode
        :type target: int
        :rtype: List[List[int]]
        """
        if not root:
            return []
        res, path = [], []

        def pathSumCore(root, target):
            if not root:
                return
            path.append(root.val)
            target -= root.val
            if not root.left and not root.right and target == 0:
                res.append(path[:])
            pathSumCore(root.left, target)
            pathSumCore(root.right, target)
            path.pop()

        pathSumCore(root, target)
        return res


"""
将一颗二叉搜索树转化为一个有序的双向链表 leetcode_426
leetcode_114 二叉树展开为链表

"""

"""
重建二叉树 leetcode_105
从前序和中序序列构建一颗二叉树


"""


"""
二叉树的镜像 leetcode_226 
"""

"""
求二叉树的深度 leetcode_104
"""


"""
剑指offer_33 输入一个数组，判断他是不是二叉树的后序遍历序列
"""
