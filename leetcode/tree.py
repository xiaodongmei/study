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
题目：
    将一颗二叉搜索树转化为一个有序的双向链表 剑指offer36 / leetcode_426
二叉树的左孩子指向前驱，右孩子指向后继，双向循环链表，第一个节点的前驱向最后一个节点，最后一个节点的后继
指向第一个节点
思想：
    要把二叉搜索树转化成有序的循环双向链表，则要通过二叉树搜索树的中序遍历（左根右，二叉搜素树的中序遍历是单调递增的有序序列），
    在二叉树搜索树的中序遍历的中，我们要完成的事情是，进行指针的转化，即对于树中的节点，我们要记录前驱节点pre和当前的节点cur，
    使得cur.left = pre,pre.right = cur,当pre为空时，说明当前访问的节点是头节点，我们让head = cur，整个遍历的过程都要做
    这步操作，本节点遍历完，即本操作完之后，我们让pre节点紧跟过去，即pre = cur,当前节点操作完了后，继续中序遍历的模版，遍历右孩子节点
    经过二叉树的中序遍历后，我们便得到了链表的头节点head，pre此时通过遍历已经走到了链表的结尾，即已经是尾节点了，因为要构建循环双向有序
    链表，所以我们让尾节点即 pre.right = head，头节点的前驱，head.left = pre，最后返回我们所构造的链表的头节点，返回链表头节点
注意点：
    本质就是二叉树中序遍历打印的框架，我们在打印那一步，不做打印，而是完成我们指针转换的操作，记得让pre在本步操作完成之后。然后紧跟过去=cur,
    当cur为空时，说明我们已经整个二叉树也遍历完成，整个转换也完成了，最后返回头节点
时间复杂度：O（n）因为整棵树都被遍历了一遍，时间复杂度 O（n）
空间复杂度：O(n) 这个最坏情况下就是树退化成一个链表，每个都要递归压栈操作，空间复杂度 O(n)
"""


class Solution_1(object):
    def treeToDoublyList(self, root):
        """
        :type root: Node
        :rtype: Node
        """
        if not root:
            return
        self.pre = None
        self.head = None

        def treeToDoublyListCore(cur):
            if not cur:
                return
            treeToDoublyListCore(cur.left)
            if self.pre:
                self.pre.right, cur.left = cur, self.pre
            else:
                self.head = cur
            self.pre = cur
            treeToDoublyListCore(cur.right)

        treeToDoublyListCore(root)
        self.head.left, self.pre.right = self.pre, self.head
        return self.head


"""
leetcode_114 二叉树展开为链表
要求：将二叉树先序遍历，展开成单链表，空间复杂度O（1）
思路：
方法一：
    观察二叉树找规律，我们可以用这种思路，找根节点左孩子的最后一个右孩子，
然后让这个右孩子挂过去，之前当前root节点的右孩子，当前root节点的右孩子指向他的左孩子，然后将他的左孩子置空，
当前root的操作完成之后，让root = 他的右孩子，root = root.right,进行下一层同等的操作，知道再没有root节点，
整个二叉树就被我们完美转成了一颗单链表
这样的话，时间复杂度 O（n）因为本质上，每个节点只遍历了一次
        空间复杂度O（1）
方法二：
    可以利用二叉树的先序遍历，把遍历到的节点都放到一个队列中，然后在二叉树的遍历结束后，从队列中出队，先进去的要先出来，所以是 pop(0),
    建立我们的单链表
"""


class Solution1(object):
    def flatten(self, root):
        """
        :type root: TreeNode
        :rtype: None Do not return anything, modify root in-place instead.
        """
        while root:
            pleft = root.left
            if pleft:
                while pleft.right:
                    pleft = pleft.right
                pleft.right = root.right
                root.right = root.left
                root.left = None
            root = root.right


"""

"""


class Solution2(object):
    def flatten(self, root):
        """
        :type root: TreeNode
        :rtype: None Do not return anything, modify root in-place instead.
        """
        if not root:
            return
        queue = []

        def flattenCore(root):
            if not root:
                return
            queue.append(root)
            flattenCore(root.left)
            flattenCore(root.right)

        flattenCore(root)
        head = queue.pop(0)
        head.left = None
        while queue:
            tmp = queue.pop(0)
            tmp.left = None
            head.right = tmp
            head = tmp


"""
重建二叉树 leetcode_105
从前序和中序序列构建一颗二叉树
preorder = [3, 9, 8, 5, 4, 10, 20, 15, 7]
inorder = [4, 5, 8, 10, 9, 3, 15, 20, 7]

思路：利用先序序列和中序序列重建二叉树，首先先序序列是：根左右，中序序列：左根右
所以，先序序列中第一个元素就是我们的根节点，然后在中序序列中找根节点，根节点会把中序序列分为
两半，左边的就是左子树部分，右边的是右子树部分，我们只需要递归的构造他的左子树和右子树，如果这颗树
的每个子树的左子树和右子树都构造完毕了（左子树或右子树的序列不再有元素，左孩子或右孩子个数<=1,则说明构造好了）（分治的思路），那么这棵树就构造完成了
注意点：
    下标，在中序序列中找到根节点的位置，然后把序列分成了左子树和右子树，再去递归构造子树即可
    mid = inorder.index(preorder[0])
    下一次，左子树部分 为：inorder[:mid] preorder[1:mid+1]  第一个元素为根节点，已经建好，我们越过它，再去构造他的子树，树左子树边的长度为mid
          右子树部分 为：inorder[mid+1] preorder[mid+1:]
"""


class TreeNode(object):
    def __init__(self, val=0, left=None, right=None):
        self.val = val
        self.left = left
        self.right = right


class Solution(object):
    def buildTree(self, preorder, inorder):
        """
        :type preorder: List[int]
        :type inorder: List[int]
        :rtype: TreeNode
        """
        if not (preorder and inorder) or len(preorder) != len(inorder):
            return
        root = TreeNode(preorder[0])
        mid = inorder.index(preorder[0])
        root.left = self.buildTree(preorder[1:mid + 1], inorder[:mid])
        root.right = self.buildTree(preorder[mid + 1:], inorder[mid + 1:])
        return root


class Solution(object):
    def invertTree(self, root):
        """
        :type root: TreeNode
        :rtype: TreeNode
        """
        if not root:
            return
        root.left, root.right = root.right, root.left
        self.invertTree(root.left)
        self.invertTree(root.right)


"""
二叉树的镜像 leetcode_226 
思路：
    有两种方法，
    方法一：
        DFS，利用递归法，本质就是交换每个子树的左右孩子。整棵树的左右孩子都被交换了，则也就得到了这棵树的镜像
        递归结束条件。if not root,root为空，没有子树的左右孩子再需要交换，递归就结束了，
        如果有root，则交换它的左右孩子，并对它的左子树和右子树递归做同样的操作，左右子树都完成了交换，则整棵树也就都完成了交换，
        最后返回root，根节点，得到了我们二叉树的镜像
        时间复杂度：O（n）
        空间复杂度：O（h），h为树的高度（递归层数）
    方法二：
        BFS，广度优先遍历，我们可以利用队列，先进先出，先把root节点压进队列，当队列不为空时，出队，交换他的左右孩子，如果当前的
        root节点还有左右孩子，那就都压入队列，循环做同样的操作，直到整个队列里都没有元素了，说明整个树的已经被遍历完成，并且得到了
        此二叉树的镜像
        时间复杂度：
            整个树的元素都被遍历了一遍，入库，出队的，所以时间复杂度 O（n）
        空间复杂度：队列中的元素个数至少有n/2个了。所以，空间复杂度也是 O（n）
注意点：
"""


# 方法一 递归法
class Solution(object):
    def invertTree(self, root):
        """
        :type root: TreeNode
        :rtype: TreeNode
        """
        if not root:
            return
        root.left, root.right = root.right, root.left
        self.invertTree(root.left)
        self.invertTree(root.right)
        return root


# 方法二： BFS法
class Solution(object):
    def invertTree(self, root):
        """
        :type root: TreeNode
        :rtype: TreeNode
        """
        if not root:
            return
        queue = [root]
        while queue:
            tmp = queue.pop(0)
            tmp.left, tmp.right = tmp.right, tmp.left
            if tmp.left:
                queue.append(tmp.left)
            if tmp.right:
                queue.append(tmp.right)
        return root


"""
求二叉树的深度 leetcode_104
方法一：DFS
    二叉树的深度就是，二叉树到叶子节点的路径最长的那条节点的个数，等于它的左子树和右子树中这两者中最大的深度 + 本层的深度（1）
    二叉树的左子树就等于它的左子树对应的左子树和右子树中这两者中最大的深度 + 本层的深度（1）
    二叉树的右子树就等于它的左子树对应的左子树和右子树中这两者中最大的深度 + 本层的深度（1）
    所以，我们可以利用递归的写法，求得二叉树的左子树和右子树中这两者中最大者的深度+1，返回得到本二叉树的深度
    时间复杂度 O（n)
    空间复杂度 O（h）h为二叉树的高度
方法二：BFS
    可以利用层次遍历，借用队列，把本层的节点全部压入，并在里层控制一个循环，把当前本层的节点都出掉，出完了，说明本层也完了，深度+1，
    同时在这个过程中，把下一层的节点也放进去了，循环就好了，到最后，队列中没有元素了，说明整个树也遍历完了，也得到了我们二叉树的深度，
    返回即可
    时间复杂度：O（n）
    空间复杂度：O（n）
"""


# DFS
class Solution(object):
    def maxDepth(self, root):
        """
        :type root: TreeNode
        :rtype: int
        """
        if not root:
            return 0
        l = self.maxDepth(root.left)
        r = self.maxDepth(root.right)
        return max(l, r) + 1


# BFS
class Solution(object):
    def maxDepth(self, root):
        """
        :type root: TreeNode
        :rtype: int
        """
        if not root:
            return 0
        queue = [root]
        depth = 0
        while queue:
            n = len(queue)
            for i in range(n):
                node = queue.pop(0)
                if node.left:
                    queue.append(node.left)
                if node.right:
                    queue.append(node.right)
            depth += 1
        return depth


"""
二叉搜索树的后序遍历序列
剑指offer_33 输入一个数组，判断他是不是二叉树搜索树的后序遍历序列
思路：
[1,6,3,2,5]
[1,3,2,6,5]
二叉搜索树的后序遍历：
    左右根，说明整个序列最后一个节点为根节点，
    那么序列从前往后找到的第一个比根节点大的，即是右孩子，后面部分， 从这个节点到根节点前的都是右子树，因为是二叉搜索树，所以整个右子树都应该比根节点大，如有
    比根节点小的，说明它不是二叉搜索树的后序遍历序列，
    递归的检查他的子树是否也满足：父节点比左孩子（左子树）大，比右孩子（右子树）小，
    如果每子树都满足，则说明他是二叉搜索树的后序遍历序列，如果 L >=r 说明都没有左右孩子了。不可再划分了，直接返回true
注意点：l >= r 说明都没有元素可以划分了，这本质也是一种递归分治的思路，注意过程中元素的赋值
"""


class Solution(object):
    def verifyPostorder(self, postorder):
        """
        :type postorder: List[int]
        :rtype: bool
        """

        def verifyPostorderCore(l, r):
            if l >= r:
                return True
            root = postorder[r]
            k = l
            while k < r and postorder[k] < root:
                k += 1
            j = k
            while j < r:
                if postorder[j] < root:
                    return False
                j += 1
            return verifyPostorderCore(l, k - 1) and verifyPostorderCore(k, r - 1)

        return verifyPostorderCore(0, len(postorder) - 1)
