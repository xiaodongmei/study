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
找路径之和最小的那个
状态和状态转移方程都已经get
三角形最后一个节点肯定只有一个值，也就是我们的mini[0]
"""

"""
思路：这道题是一道动态规划类型的题，
    如果我们用暴力法，也就是递归（回溯）从上往下 求得所有路径，再求里面的最小值。这样时间复杂度是非常大的，O（2^n)
    所以我们可以从下往上进行递推，
    定义动态规划dp状态为：dp[i,j] 表示从下往上的节点到i，j这个节点的最小的路径和
    状态转移方程：dp[i,j] = min(dp[i-1,j],dp[i-1,j-1]) + triangle(i,j)
    关于dp状态我们可以用一维数组来存储表示 （进行降维压缩）
    
"""

"""
这段代码，首先对于参数的判断，如果为空的话，直接返回
得到三角形也就是二维数组的最后一层，res就是这层的节点，（这层的节点也就是我们dp最初始的节点）
然后从下往上进行迭代递推，我们从倒数第二层开始，一直到第0层，也就是三角形的尖尖，因为range是左闭右开的，所以是到-1，每次递减1
然后遍历当层，得到当前节点的dp状态，也就是从下往上到当前解决的最短的路径和
在循环结束了之后，我们就得到了三角形的最顶端也就是res[0]de最短路径和， 同解于我们从最顶端，从上往下找到最底端一层的最短路径和

这道题类似到生活就是：如何让人生不走弯路，找到那道最短的，最捷径的路呢？如果我们可以预知未来，从未来后面往前，我们就可以每次选则那个最优的，最短的，
那么从未来到现在，我们就可以找到那个最短的捷径
"""


class Solution(object):
    def minimumTotal(self, triangle):
        """
        :type triangle: List[List[int]]
        :rtype: int
        """
        if not triangle:
            return
        res = triangle[-1]
        for i in range(len(triangle) - 2, -1, -1):
            for j in range(len(triangle[i])):
                res[j] = min(res[j], res[j + 1]) + triangle[i][j]
        return res[0]


"""
乘积的最大子序列
数组里面的数有正数，有负数，还必须是连续的子序列
思路：求子序列乘积的最大值，序列元素都是正数，那么自然整个序列的乘机就是最大值
但是事实是序列中可能有正值，也有负值，也有零值，所以子序列在加入新的元素时，需要判断这个元素是否是负值，
如果是，则让目前子序列的最大值和最小值交换，因为当一个最大值 * 上一个负值，他也就变成了最小值，相反，最小值却变成了最大值，
我们进行两者交换，让imax中始终保存子序列的最大值，如果当前数不是负值， 则分别求子序列的最大值和最小值，
最大值 = max(imax,num)
最小值 = min(imin,num)
保存得到的最大值，max=max(maxvalue,imax)，等到循环结束后，我们就得到了子序列的最大值，返回
"""


class Solution(object):
    def maxProduct(self, nums):
        """
        :type nums: List[int]
        :rtype: int
        """
        max_value = float("-inf")
        imax, imin = 1, 1

        for num in nums:
            if num < 0:
                imax, imin = imin, imax
            imax = max(imax * num, num)
            imin = min(imin * num, num)

            max_value = max(max_value, imax)
        return max_value


"""
买卖股票的最佳时间
系列题目 121 122 123 309 188 714

如果达到利润最大化

121 只能买一次 卖一次 
122 买卖无数次 
123 买卖两次 交易只能发生两次 不能同时拥有两股股票
309 cooldown 隔一天才能买卖
188 交易只能发生K次
714 交易手续费

一个dp 一个动态规划的思路

可以买卖k次， 只能同时持有1股
dp[i] 到了第i天的最大值 maxprofits

121：最低的时候买入，最高的时候卖出
新的值 - 保存的最小值

122：后一天的价格高于前一天的化 就买进
123：

dp 状态的定义 dp[i]  到了第i天的最大利润 maxfrofits MP[i]
状态转移方程
MP[i] = MP[i-1] + (-a[i])/a[i]  但是无法知道我之前手里有没有股票，我当前的状态
MP[i][j][k] 三维状态的dp

i:天 0～n-1
j : 0/1
k：k表示我之前交易了多少次 0 ～ K


MP[i][k][j] = 

k表示我之前交易了多少次
 

不知道之前已经买卖多少次了？  不动          卖掉
MP[i,k,0] = max(MP[i-1,k,0], MP[i-1,k-1,1]+a[i])
             不动             买入
MP[i,k,1] = max(MP[i-1,k,1],MP[i-1,k-1,0] - a[i])

最大值：MP[n-1,{0..K},0] 的最大值 就是我们最后的结果

cooldown K （0，1）可以交易还是不可以交易
可以同时拥有x股 j 0~X

时间复杂度 O（n * k）
"""


# 方法一
# leetcode_121
class Solution(object):
    def maxProfit(self, prices):
        """
        :type prices: List[int]
        :rtype: int
        """
        if not prices:
            return 0
        min_price = float("inf")
        profits = float("-inf")
        for price in prices:
            min_price = min(price, min_price)
            profits = max(price - min_price, profits)
        return profits






# 方法二：
# leetcode_122
class Solution(object):
    def maxProfit(self, prices):
        """
        :type prices: List[int]
        :rtype: int
        """
        if not prices or len(prices) < 2:
            return 0
        profits = 0
        for i in range(len(prices) - 1):
            if prices[i + 1] > prices[i]:
                profits += (prices[i + 1] - prices[i])
        return profits


# leetcode_123
# 买卖股票的最佳时间 123
"""
dp[i][k][j]
dp[i][k][0] = max(dp[i-1][k][0],dp[i-1][k-1][1]+a[i])
dp[i][k][1] = max(dp[i-1][k][1],dp[i-1][k-1][0]-a[i])
三维的dp方程
dp[n - 1][K][0]，即最后一天，最多允许 K 次交易，最多获得多少利润。

"""


"""

dp[i][k][0] = max(dp[i-1][k][0], dp[i-1][k][1]+prices[i])
dp[i][k][1] = max(dp[i-1][k][1], dp[i-1][k-1][0]-prices[i])
"""


"""
// 空间复杂度优化版本
int maxProfit_k_1(int[] prices) {
    int n = prices.length;
    // base case: dp[-1][0] = 0, dp[-1][1] = -infinity
    int dp_i_0 = 0, dp_i_1 = Integer.MIN_VALUE;
    for (int i = 0; i < n; i++) {
        // dp[i][0] = max(dp[i-1][0], dp[i-1][1] + prices[i])
        dp_i_0 = Math.max(dp_i_0, dp_i_1 + prices[i]);
        // dp[i][1] = max(dp[i-1][1], -prices[i])
        dp_i_1 = Math.max(dp_i_1, -prices[i]);
    }
    return dp_i_0;
}

"""


"""
最长上升子序列
"""

"""
要想得到最大的，需要把整个原数组序列都遍历完，然后保存那个乘机最大的然后返回
这个最初始的状态应该是什么呢

维护当前最大值和最小值
当负数出现时则imax与imin进行交换再进行下一步计算


初始值：
"""


链表，二叉树，回溯，深度宽度优先遍历，图，贪心，动规，数组，哈希表

