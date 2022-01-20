"""

"""


# 超出时间限制了
class Solution(object):
    def climbStairs(self, n):
        """
        :type n: int
        :rtype: int
        """
        if n == 0 or n == 1:
            return 1
        return self.climbStairs(n - 1) + self.climbStairs(n - 2)


# 你的错误就在于 你的递归没有终止条件
# 记忆话搜索
class Solution(object):
    def climbStairs(self, n):
        """
        :type n: int
        :rtype: int
        """

        def climbStairsCore(i, mem):
            if i == 0 or i == 1:
                return 1
            if mem[i] == -1:
                mem[i] = climbStairsCore(i - 1, mem) + climbStairsCore(i - 2, mem)
            return mem[i]

        return climbStairsCore(n, [-1] * (n + 1))


# 对于状态的定义和状态转移方程
class Solution(object):
    def climbStairs(self, n):
        """
        :type n: int
        :rtype: int
        """
        f1, f2 = 1, 1
        for i in range(2, n + 1):
            f1, f2 = f2, f1 + f2
        return f2


"""
杨辉三角 最小路径和
递归/回溯
贪心法不行
这也是每步都在做决策么
动态规划

状态的定义
状态转移方程
从下往上的递推

dp[i,j]状态表示的是：从最下面的点，走到i，j这个点，路径之和的最小值
dp[i,j] = min(dp[i+1,j] + dp[i+1,j+1]) + triangle[i,j]
起始值
dp[m-1,j] = triangle[m-1,j]
O(m * n)

用一维数组就可以来表示 （状态压缩）
"""









