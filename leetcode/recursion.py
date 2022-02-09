"""
递归和分治
"""

"""
pow函数 leetcode50
题目描述：实现 pow(x, n) ，即计算 x 的 n 次幂函数（即，xn ）

# 解法一：快速幂 + 递归法
思路：要求x^n次方，可以求 x^(2/n);
    x^n = x^(2/n) * x^(2/n)
    同理，递归下去，就是一种分治的思想，如果n为奇数，res = x^(2/n) * x^(2/n) * x， 如果为偶数：res = x^(2/n) * x^(2/n)
    当 n= 0 时，任何数的0次方都是1
    当 n < 0 时，返回结果为 1/res,否则为res
注意点：n & 1 == 1的话，说明n为奇数，否则为偶数
      n >>= 1 n向右移1位，就等于 n /= 2
"""


# pow函数 leetcode50
class Solution1(object):
    def myPow(self, x, n):
        """
        :type x: float
        :type n: int
        :rtype: float
        """

        def myPowCore(n):
            if n == 0:
                return 1.0
            y = myPowCore(n >> 1)
            return y * y if not n & 1 else y * y * x

        return myPowCore(n) if n > 0 else 1.0 / myPowCore(-n)


"""
# 解法二： 快速幂 + 迭代法
思路：每次x *= x就相当于给x降幂，即 n >>= 1，当 n 降幂为0的时候，说明已经没有幂可降了
    我们已经得到了最终的结果，然后返回结果即可
    如果底数x ==0,直接返回0，如果次方n<0，则求其倒数的解返回，如果n为奇数，即 n & 1 != 0,
    则需要结果 res *= x
注意点：结果和底数都是flaot类型的
"""


class Solution2(object):
    def myPow(self, x, n):
        """
        :type x: float
        :type n: int
        :rtype: float
        """
        if x == 0.0:
            return 0.0
        res = 1
        if n < 0:
            x, n = 1 / x, -n
        while n:
            if n & 1:
                res *= x
            x *= x
            n >>= 1
        return res


"""
leetcode_22
题目描述：
    数字 n 代表生成括号的对数，请你设计一个函数，用于能够生成所有可能的并且 有效的 括号组合。
思路：生成有效括号组合
     可以利用DFS 暴力递归搜索基础上 + 剪枝（局部不合法 + 不再递归）来解决
     left表示左括号剩了多少，right表示右括号剩了多少 
     递归终止条件：如果left == 0 and right == 0 的话说明一种括号路径方法形成了，需要把路径加入我们的结果集
     如果  right < left 也就是right用的比left还多，这种肯定是不合法的，需要剪枝剪掉 直接return
     然后就可以进行选择，
     如果left>0  那种选择，并且继续递归，此时left数量 -1
     如果right>0 那种选择，并且继续递归，此时right数量 -1
     最后返回我们的res结果集
时间复杂度：O(2^n)
注意点：字符串的拼接可以直接用 + 
"""


class Solution(object):
    def generateParenthesis(self, n):
        """
        :type n: int
        :rtype: List[str]
        """
        res, cur = [], ''

        def generateParenthesisCore(cur, left, right):
            if left == 0 and right == 0:
                res.append(cur)
                return res
            if right < left:
                return
            if left > 0:
                generateParenthesisCore(cur + '(', left - 1, right)
            if right > 0:
                generateParenthesisCore(cur + ')', left, right - 1)

        generateParenthesisCore(cur, n, n)
        return res


"""
全排列问题
leetcode_46. 全排列
leetcode_47. 全排列 II
leetcode78. 子集问题 
leetcode_90 子集_II
leetcode_39 组合问题 
"""

"""
全排列46
题目描述：给定一个不含重复数字的数组 nums ，返回其 所有可能的全排列 。你可以 按任意顺序 返回答案
示例 1：

输入：nums = [1,2,3]
输出：[[1,2,3],[1,3,2],[2,1,3],[2,3,1],[3,1,2],[3,2,1]]
思路：全排列问题，可以利用回溯+剪枝解决，枚举所有可能的结果，对于之前已经选过的元素，就不要再选了，需要剪枝
    递归结束条件：如果当前路径长 == 所给序列长度，说明已经形成了一种排列结果，深拷贝，将它加入结果集
    在选择列表做选择，如果之前的元素已经选过了，那就continue直接剪枝
    否则做选择，然后进行下一次递归，递归结束后pop 恢复之前的状态
    就是我们回溯的框架
注意点：res.append(path[:]) 要注意为深拷贝

"""


class Solution(object):
    def permute(self, nums):
        """
        :type nums: List[int]
        :rtype: List[List[int]]
        """
        res, path = [], []

        def permuteCore(nums):
            if len(path) == len(nums):
                res.append(path[:])
                return
            for num in nums:
                if num in path:
                    continue
                path.append(num)
                permuteCore(nums)
                path.pop()

        permuteCore(nums)
        return res


"""
全排列47
题目描述：给定一个可包含重复数字的序列 nums ，按任意顺序 返回所有不重复的全排列

示例 1：
输入：nums = [1,1,2]
输出：
[[1,1,2],
[1,2,1],
[2,1,1]]

思路：
注意点：

"""


class Solution(object):
    def permuteUnique(self, nums):
        """
        :type nums: List[int]
        :rtype: List[List[int]]
        """
        if not nums:
            return []
        res, path = [], []
        used = [0] * len(nums)

        def permuteUniqueCore(nums, path, used):
            if len(path) == len(nums):
                res.append(path[:])
                return
            for i in range(len(nums)):
                if not used[i]:
                    if i > 0 and nums[i] == nums[i - 1] and not used[i - 1]:
                        continue
                    used[i] = 1
                    path.append(nums[i])
                    permuteUniqueCore(nums, path, used)
                    path.pop()


"""
leetcode_78 求子集问题
题目描述：给你一个整数数组 nums ，数组中的元素 互不相同 。返回该数组所有可能的子集（幂集）
        解集不能包含重复的子集。你可以按 任意顺序 返回解集。

示例 1：
输入：nums = [1,2,3]
输出：[[],[1],[2],[1,2],[3],[1,3],[2,3],[1,2,3]]

思路：还是回溯问题，我们套用回溯的框架，
    结束条件是：对于自己问题，路径结果都需要加进最终的结果集，所以结束条件是当在选择列表里，
    不再有选择可以选，就结束了
    我们需要记录选择的下标位置，每次从下标位置到最后一个元素之前做原则
    本次选择后，需要递归进行下一层做选择，递归完后，需要恢复状态
注意点：res.append(path[:]) 注意要深拷贝
"""


class Solution(object):
    def subsets(self, nums):
        """
        :type nums: List[int]
        :rtype: List[List[int]]
        """
        res, path = [], []

        def backTrack(nums, start):
            res.append(path[:])
            for i in range(start, len(nums)):
                path.append(nums[i])
                backTrack(nums, i + 1)
                path.pop()

        backTrack(nums, 0)
        return res


"""
leetcode_90 子集_2
题目描述：
    给你一个整数数组 nums ，其中可能包含重复元素，请你返回该数组所有可能的子集（幂集）。
    解集 不能 包含重复的子集。返回的解集中，子集可以按 任意顺序 排列。

示例 1：
输入：nums = [1,2,2]
输出：[[],[1],[1,2],[1,2,2],[2],[2,2]]

思路：如果含有重复元素的话，我们需要先对序列进行排序
    在做选择过程中，需要剪枝，如果i>start, 并且当前元素和前一个元素一样的话，需要剪枝，剪掉
注意点：if i > start and (nums[i] == nums[i - 1]):
"""


class Solution(object):
    def subsetsWithDup(self, nums):
        """
        :type nums: List[int]
        :rtype: List[List[int]]
        """
        res, path = [], []
        nums = sorted(nums)

        def backTrack(nums, start):
            res.append(path[:])
            for i in range(start, len(nums)):
                if i > start and (nums[i] == nums[i - 1]):
                    continue
                path.append(nums[i])
                backTrack(nums, i + 1)
                path.pop()

        backTrack(nums, 0)
        return res


"""
leetcode_39. 组合总和  组合问题
题目描述：
    给你一个 无重复元素 的整数数组 candidates 和一个目标整数 target ，找出 candidates 中可以使数字和为目标数 target 的 所有 不同组合 ，并以列表形式返回。你可以按 任意顺序 返回这些组合。
    candidates 中的 同一个 数字可以 无限制重复被选取 。如果至少一个数字的被选数量不同，则两种组合是不同的。 
    对于给定的输入，保证和为 target 的不同组合数少于 150 个。

示例 2：
输入: candidates = [2,3,5], target = 8
输出: [[2,2,2,2],[2,3,3],[3,5]]

思路：组合他的特点在于同一个数字元素可以被无限制重复使用，为了方便剪枝等，我们先对序列进行排序
     递归结束条件：如果当前序列数字的和 == target了，说明得到一个解，把他加入最终结果集
     如果 sum > target了，直接return,进行剪枝
     否则在选择列表做选择，如果sum + 当前这个数 > target 直接剪枝剪掉
     否则做选择，递归进入下一层决策，递归完进行恢复状态
注意点：要先对序列进行排序
"""


class Solution(object):
    def combinationSum(self, candidates, target):
        """
        :type candidates: List[int]
        :type target: int
        :rtype: List[List[int]]
        """
        res, path = [], []
        candidates = sorted(candidates)

        def backTrack(candidates, start, sum, target):
            if sum > target:
                return
            if sum == target:
                return res.append(path[:])
            for i in range(start, len(candidates)):
                if sum + candidates[i] > target:
                    return
                sum += candidates[i]
                path.append(candidates[i])
                backTrack(candidates, i, sum, target)
                sum -= candidates[i]
                path.pop()

        backTrack(candidates, 0, 0, target)
        return res


"""
N皇后问题
leetcode_51 leetcode_52

题目描述：
    n 皇后问题 研究的是如何将 n 个皇后放置在 n×n 的棋盘上，并且使皇后彼此之间不能相互攻击。
    给你一个整数 n ，返回所有不同的 n 皇后问题 的解决方案。
    每一种解法包含一个不同的 n 皇后问题 的棋子放置方案，该方案中 'Q' 和 '.' 分别代表了皇后和空位
思路：
注意点：
"""


class Solution(object):
    def solveNQueens(self, n):
        """
        :type n: int
        :rtype: List[List[str]]
        """
        if not n:
            return []
        res = []
        board = [["."] * n for _ in range(n)]

        def isvalidSite(board, row, col):
            # 判断同列是否已经有皇后了
            for i in range(len(board)):
                if board[i][col] == "Q":
                    return False

            # 判断右上角是否存在皇后攻击
            i, j = row - 1, col + 1
            while i >= 0 and j < n:
                if board[i][j] == "Q":
                    return False
                i -= 1
                j += 1

            i, j = row - 1, col - 1
            # 判断左上角是否存在皇后攻击
            while i >= 0 and j >= 0:
                if board[i][j] == "Q":
                    return False
                i -= 1
                j -= 1
            return True

        def solveNQueensCore(board, row, n):
            if row == n:
                res_tmp = []
                for temp in board:
                    temp_str = "".join(temp)
                    res_tmp.append(temp_str)
                res.append(res_tmp)
            for col in range(n):
                if not isvalidSite(board, row, col):
                    continue
                board[row][col] = "Q"
                solveNQueensCore(board, row + 1, n)
                board[row][col] = "."

        solveNQueensCore(board, 0, n)
        return res


# leetcode_52

class Solution(object):
    def totalNQueens(self, n):
        """
        :type n: int
        :rtype: List[List[str]]
        """
        if not n:
            return []
        res = []
        board = [["."] * n for _ in range(n)]

        def isvalidSite(board, row, col):
            # 判断同列是否已经有皇后了
            for i in range(len(board)):
                if board[i][col] == "Q":
                    return False

            # 判断右上角是否存在皇后攻击
            i, j = row - 1, col + 1
            while i >= 0 and j < n:
                if board[i][j] == "Q":
                    return False
                i -= 1
                j += 1

            i, j = row - 1, col - 1
            # 判断左上角是否存在皇后攻击
            while i >= 0 and j >= 0:
                if board[i][j] == "Q":
                    return False
                i -= 1
                j -= 1
            return True

        def solveNQueensCore(board, row, n):
            if row == n:
                res_tmp = []
                for temp in board:
                    temp_str = "".join(temp)
                    res_tmp.append(temp_str)
                res.append(res_tmp)
            for col in range(n):
                if not isvalidSite(board, row, col):
                    continue
                board[row][col] = "Q"
                solveNQueensCore(board, row + 1, n)
                board[row][col] = "."

        solveNQueensCore(board, 0, n)
        return len(res)


"""
有效的数独
leetcode_36 
题目描述：
    请你判断一个 9 x 9 的数独是否有效。只需要 根据以下规则 ，验证已经填入的数字是否有效即可。
    数字 1-9 在每一行只能出现一次。
    数字 1-9 在每一列只能出现一次。
    数字 1-9 在每一个以粗实线分隔的 3x3 宫内只能出现一次。（请参考示例图）

思路：
注意点：
"""

"""
解数独
leetcode_37
题目描述：
    编写一个程序，通过填充空格来解决数独问题。
    数独的解法需 遵循如下规则：
    数字 1-9 在每一行只能出现一次。
    数字 1-9 在每一列只能出现一次。
    数字 1-9 在每一个以粗实线分隔的 3x3 宫内只能出现一次。（请参考示例图）
    数独部分空格内已填入了数字，空白格用 '.' 表示

思路：
注意点：
"""


# leetcode_37 解数独
class Solution(object):
    def solveSudoku(self, board):
        """
        :type board: List[List[str]]
        :rtype: None Do not return anything, modify board in-place instead.
        """

        def solveSudokuCore(board, i, j):
            if j == 9:
                return solveSudokuCore(board, i + 1, 0)
            if i == 9:
                return True
            if board[i][j] != '.':
                return solveSudokuCore(board, i, j + 1)
            for ch in range(1, 10):
                if not isValid(board, i, j, str(ch)):
                    continue
                board[i][j] = str(ch)
                if solveSudokuCore(board, i, j + 1):
                    return True
                board[i][j] = '.'
            return False

        def isValid(board, row, col, ch):
            # 同一行是否已经有这个数字了
            for k in range(9):
                if board[row][k] == ch:
                    return False
                # 同一列是否已经有这个数字了
                if board[k][col] == ch:
                    return False
                # 小3*3的方格是否已经有这个数字了
                if board[(row // 3) * 3 + k // 3][(col // 3) * 3 + k % 3] == ch:
                    return False
            return True

        solveSudokuCore(board, 0, 0)
