"""
动态规划系系列就基本做完了
"""

"""
leetcode_70 爬楼梯
题目描述：
    假设你正在爬楼梯。需要 n 阶你才能到达楼顶。
    每次你可以爬 1 或 2 个台阶。你有多少种不同的方法可以爬到楼顶呢？
思路：每次爬楼梯可以爬1个或两个台阶，那么爬n层楼梯的方法 f(n)= f(n-1) + f(n-2)，最后一步爬了1步或2步他们拥有的方法和
     这个表示式和斐波拉切那个类似，可以递归做，但是画出递归树的话，我们发现会有很多重复节点，时间复杂度达到了 2^n,会超时，
     所以我们可以进行记忆话搜索，把那些已经计算过的记录下来，这些都是从上往下来递归来做的，
     这道题本质上是一个动态规划的题目，可以从下往上进行迭代：
     定义dp状态：dp[n]  n表示爬的楼梯层数
     状态转移方程：dp[n] = dp[n-1] + dp[n-2]  降维，用两个变量迭代
注意点：python dp初始化状态都为-1
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
leetcode_120  三角形的最小路径和
题目描述：
    给定一个三角形 triangle ，找出自顶向下的最小路径和。
    示例 1：
        输入：triangle = [[2],[3,4],[6,5,7],[4,1,8,3]]
        输出：11
        解释：如下面简图所示：
           2
          3 4
         6 5 7
        4 1 8 3
        自顶向下的最小路径和为 11（即，2 + 3 + 5 + 1 = 11）
        
思路：这道题是一道动态规划类型的题，
    如果我们用暴力法，也就是递归（回溯）从上往下 求得所有路径，再求里面的最小值。这样时间复杂度是非常大的，O（2^n)
    所以我们可以从下往上进行递推，
    定义动态规划dp状态为：dp[i,j] 表示从下往上的节点到i，j这个节点的最小的路径和
    状态转移方程：dp[i,j] = min(dp[i-1,j],dp[i-1,j-1]) + triangle(i,j)
    关于dp状态我们可以用一维数组来存储表示 （进行降维压缩）
    起始值
    dp[m-1,j] = triangle[m-1,j]
    O(m * n)
注意点：
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
乘积的最大子序列 leetcode_152
题目描述：给你一个整数数组 nums ，请你找出数组中乘积最大的连续子数组（该子数组中至少包含一个数字），并返回该子数组所对应的乘积
 
示例 1:
    输入: [2,3,-2,4]
    输出: 6
    解释: 子数组 [2,3] 有最大乘积 
思路：
    数组里面的数有正数，有负数，还必须是连续的子序列
    求子序列乘积的最大值，序列元素都是正数，那么自然整个序列的乘机就是最大值
    但是事实是序列中可能有正值，也有负值，也有零值，所以子序列在加入新的元素时，需要判断这个元素是否是负值，
    如果是，则让目前子序列的最大值和最小值交换，因为当一个最大值 * 上一个负值，他也就变成了最小值，相反，最小值却变成了最大值，
    我们进行两者交换，让imax中始终保存子序列的最大值，如果当前数不是负值， 则分别求子序列的最大值和最小值，
    最大值 = max(imax,num)
    最小值 = min(imin,num)
    保存得到的最大值，max=max(maxvalue,imax)，等到循环结束后，我们就得到了子序列的最大值，返回
注意点：
    python中的无穷小：float("-inf")
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
买卖股票的最佳时间系列
系列题目 121 122 123 309 188 714
如何达到利润最大化

121 只能买一次 卖一次 
122 买卖无数次 
123 买卖两次 交易只能发生两次 不能同时拥有两股股票
309 cooldown 隔一天才能买卖（卖出股票后，你无法在第二天买入股票 (即冷冻期为 1 天)）
188 交易只能发生K次
714 有交易手续费

注意点：
    你不能同时参与多笔交易（你必须在再次购买前出售掉之前的股票）
    这里的一笔交易指买入持有并卖出股票的整个过程，每笔交易你只需要为支付一次手续费。
    可以买卖k次， 只能同时持有1股

思路：
经典动态规划的题目：动态规划的本质：穷举状态，在选择中找最优解
用一个状态转移方程来解决
121：最低的时候买入，最高的时候卖出
新的值 - 保存的最小值，返回最终的最大的那个

122：后一天的价格高于前一天的化 就买进（贪心问题）

123：
dp 状态的定义 dp[i]  到了第i天的最大利润 maxfrofits MP[i]
状态转移方程
MP[i] = MP[i-1] + (-a[i])/a[i]  但是无法知道我之前手里有没有股票，我当前的状态

dp状态：
    三维状态的dp：MP[i][j][k] 三维状态的dp
    i:天 0～n-1
    j : 0/1
    k：k表示我之前交易了多少次 0 ～ K

状态转移方程：
                      不动          卖掉
MP[i,k,0] = max(MP[i-1,k,0], MP[i-1,k-1,1]+a[i])
             不动             买入
MP[i,k,1] = max(MP[i-1,k,1],MP[i-1,k-1,0] - a[i])

最大值：MP[n-1,{0..K},0] 的最大值 就是我们最后的结果

cooldown K （0，1）可以交易还是不可以交易
可以同时拥有x股 j 0~X
时间复杂度 O（n * k）
"""


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


# leetcode_122
# 贪心算法
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
dp[i][k][0] = max(dp[i-1][k][0], dp[i-1][k][1]+prices[i])
dp[i][k][1] = max(dp[i-1][k][1], dp[i-1][k-1][0]-prices[i])
三维的dp方程
dp[n - 1][K][0]，即最后一天，最多允许 K 次交易，最多获得多少利润。
"""


class Solution(object):
    def maxProfit(self, prices):
        """
        :type prices: List[int]
        :rtype: int
        """
        if not prices:
            return 0
        n, k = len(prices), 2
        dp = [[[0] * 2 for _ in range(k + 1)] for _ in range(n)]
        for i in range(n):
            for j in range(k, 0, -1):
                if i - 1 == -1:
                    dp[i][j][0] = 0
                    dp[i][j][1] = -prices[i]
                    continue
                dp[i][j][0] = max(dp[i - 1][j][0], dp[i - 1][j][1] + prices[i])
                dp[i][j][1] = max(dp[i - 1][j][1], dp[i - 1][j - 1][0] - prices[i])

        return dp[n - 1][k][0]


# 买卖股票的最佳时间 123 dp降维
class Solution(object):
    def maxProfit(self, prices):
        """
        :type prices: List[int]
        :rtype: int
        """
        # 交易次数为2次时的状态转移方程 k=2 时
        # dp[i][k][0] = max(dp[i-1][k][0],dp[i-1][k][1]+prices[i])
        # dp[i][k][1] = max(dp[i-1][k][1],dp[i-1][k-1][0] - prices[i])

        # 当 k = 2时，因为k比较小，所以我们可以枚举状态,降低维度
        # dp[i][2][0] = max(dp[i-1][2][0],dp[i-1][2][1]+prices[i])
        # dp[i][2][1] = max(dp[i-1][2][1],dp[i-1][1][0] - prices[i])
        # dp[i][1][0] = max(dp[i-1][1][0],dp[i-1][1][1]+prices[i])
        # dp[i][1][1] = max(dp[i-1][1][1],-prices[i])

        if not prices:
            return 0
        dp_i20, dp_i10 = 0, 0
        dp_i21, dp_i11 = float("-inf"), float("-inf")
        for price in prices:
            dp_i20 = max(dp_i20, dp_i21 + price)
            dp_i21 = max(dp_i21, dp_i10 - price)
            dp_i10 = max(dp_i10, dp_i11 + price)
            dp_i11 = max(dp_i11, -price)
        return dp_i20


"""
买卖股票的最佳时间，当交易可以发生k次时
"""


# 买卖股票的最佳时间，当交易可以发生k次时
class Solution(object):

    def maxProfitNotLimited(self, prices):
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

    def maxProfit(self, k, prices):
        """
        :type k: int
        :type prices: List[int]
        :rtype: int
        """
        # 买卖股票的最佳时间
        # 状态转移方程：
        # dp[i][k][0] = max(dp[i-1][k][0],dp[i-1][k][1]+prices[i])
        # dp[i][k][1] = max(dp[i-1][k][1],dp[i-1][k-1][0]-prices[i])
        # 当序列的长度为n时，最多买卖k/2次，如果k如果超过n/2了， 那就直接说明是可以
        # 把 k计做为买卖无数次了
        if not prices or k <= 0:
            return 0
        n = len(prices)
        if k > n >> 1:
            return self.maxProfitNotLimited(prices)

        dp = [[[0] * 2 for _ in range(k + 1)] for _ in range(n)]
        for i in range(n):
            dp[i][0][0] = 0
            dp[i][0][1] = float("-inf")
        for i in range(n):
            for j in range(k, 0, -1):
                if i - 1 == -1:
                    dp[i][j][0] = 0
                    dp[i][j][1] = -prices[i]
                    continue
                dp[i][j][0] = max(dp[i - 1][j][0], dp[i - 1][j][1] + prices[i])
                dp[i][j][1] = max(dp[i - 1][j][1], dp[i - 1][j - 1][0] - prices[i])
        return dp[n - 1][k][0]


"""
买卖股票的最佳时间，有cooldown期
"""


class Solution(object):
    def maxProfit(self, prices):
        """
        :type prices: List[int]
        :rtype: int
        """
        if not prices:
            return 0
        n = len(prices)
        dp_i0, dp_i1 = 0, float("-inf")
        dp_pre_operate = 0

        for i in range(n):
            temp = dp_i0
            dp_i0 = max(dp_i0, dp_i1 + prices[i])
            dp_i1 = max(dp_i1, dp_pre_operate - prices[i])
            dp_pre_operate = temp
        return dp_i0


"""
买卖股票的最佳时间，含有手续费
"""


class Solution(object):
    def maxProfit(self, prices, fee):
        """
        :type prices: List[int]
        :type fee: int
        :rtype: int
        """
        if not prices:
            return 0
        n = len(prices)
        dp_i0, dp_i1 = 0, float("-inf")
        for i in range(n):
            temp = dp_i0
            dp_i0 = max(dp_i0, dp_i1 + prices[i])
            dp_i1 = max(dp_i1, temp - prices[i] - fee)
        return dp_i0


"""
最长上升子序列 leetcode_300
状态定义：dp[i] 表示从头开始到第i个元素 的最长上升子序列的长度
状态转移方程: dp[i] = max(dp[i],dp[j]+1) 其中j <i,(相当于j是i全面的子序列部分)， nums[j] < nums[i]
            如果nums[j] < nums[i]，就把当前元素加进来，比较在子序列的最长上升子序列的长度 + 当前元素长度（1）
            和之前的dp[i]做比较，取最大值作为dp[i]的值
最后的返回值为：dp中值最大的那个，即max(dp[0]...dp[n-1])
所以范围： i 0~n (左闭右开）
         j 0~i (左闭右开）
时间复杂度： O(n^2)
空间复杂度： O（n）     
"""


class Solution(object):
    def lengthOfLIS(self, nums):
        """
        :type nums: List[int]
        :rtype: int
        """
        if not nums:
            return 0
        n = len(nums)
        dp = [1] * n
        for i in range(n):
            for j in range(i):
                if nums[j] < nums[i]:
                    dp[i] = max(dp[i], dp[j] + 1)
        return max(dp)


"""
leetcode 322 凑硬币问题
最少需要多少个硬币
状态表示：
    dp[i]  表示凑成面值为i的需要的最少硬币数
状态转移方程：
    dp[i] = min(dp[i],dp[i-coins[j]]+1)
时间复杂度：O（x * n）
X   最后要的面值
N   硬币的数量
思路：一般这种求最值的问题本质都是动态规划问题
求面值为n的所需硬币的最小值，初始化每个dp的值都为正无穷大，dp[0] = 0
然后迭代，在硬币序列里做选择，选或不选，最后得到面值为amount的dp值返回，
如果dp[amount] > amount了，说明他还是开始的初始值，无穷大，说明凑面值无结果，返回-1
注意：面值为0时也是一种结果，结果是0。需要最少硬币数为0
"""


class Solution(object):
    def coinChange(self, coins, amount):
        """
        :type coins: List[int]
        :type amount: int
        :rtype: int
        """
        # 凑硬币问题 动态规划
        # dp状态 dp[i] 表示凑成面值为i的面值需要的最少硬币数
        # 状态转移方程 dp[i] = min(dp[i],dp[i-coins[j]]+1)

        if not coins:
            return -1
        dp = [float("inf")] * (amount + 1)
        dp[0] = 0
        for i in range(1, amount + 1):
            for coin in coins:
                if coin <= i:
                    dp[i] = min(dp[i], dp[i - coin] + 1)
        return -1 if dp[amount] > amount else dp[amount]


"""
leetcode_72 编辑最短距离
word1 word2
长m  长n
从单词1变成单词2最少需要多少步的操作 （insert,delete,replace）
状态的定义：dp[i][j] 表示从word1 的前i个字符 匹配到word2 的前j个字符 最少的需要操作的步数
返回 dp[m][n]
时间复杂度 O（m*n）
空间复杂度 O（m*n）

插入，删除，替换
dp状态转移方程：
dp[i,j] = dp[i-1,j-1] if w[i] == w[j];
dp[i,j] = min(dp[i-1,j],dp[i,j-1],dp[i-1,j-1])+1
权重不同，下标变换不同

单词的最短编辑距离，注意 if word1[i - 1] == word2[j - 1]，外层的循环i，j下标都是从1开始的，
所以这里的i-1，j-1，其实就是word1[i] == word2[j],因为word的下标是从0开始的
"""


class Solution(object):
    def minDistance(self, word1, word2):
        """
        :type word1: str
        :type word2: str
        :rtype: int
        """
        # 最短编辑距离，也属于动态规划问题
        # 状态的定义 dp[i][j]  表示word1的前i个字符 匹配 word2的前j个字符需要的最短编辑
        # 步数 所以，最后返回dp[m][n] m为word1的长度，n为word2的长度
        # 状态转移方程
        # if w1[i] == w2[j]: dp[i,j] = dp[i-1,j-1]
        # dp[i,j] = min(dp[i-1,j-1],dp[i-1,j],dp[i,j-1]) + 1

        m, n = len(word1), len(word2)
        # 定义二维dp
        dp = [[0 for _ in range(n + 1)] for _ in range(m + 1)]
        # 初始化dp
        for i in range(m + 1):
            dp[i][0] = i
        for j in range(n + 1):
            dp[0][j] = j
        for i in range(1, m + 1):
            for j in range(1, n + 1):
                if word1[i - 1] == word2[j - 1]:
                    dp[i][j] = dp[i - 1][j - 1]
                else:
                    dp[i][j] = min(dp[i - 1][j - 1], dp[i - 1][j], dp[i][j - 1]) + 1
        return dp[m][n]
