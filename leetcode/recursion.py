"""
递归和分治
pow函数 leetcode50
"""

"""
# 解法一：快速幂 + 递归法
思路：要求x^n次方，可以求 x^(2/n);
    x^n = x^(2/n) * x^(2/n)
    同理，递归下去，就是一种分治的思想，如果n为奇数，res = x^(2/n) * x^(2/n) * x， 如果为偶数：res = x^(2/n) * x^(2/n)
    当 n= 0 时，任何数的0次方都是1
    当 n < 0 时，返回结果为 1/res,否则为res
注意点：n & 1 == 1的话，说明n为奇数，否则为偶数
      n >>= 1 n向右移1位，就等于 n /= 2
"""


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
求众数 leetcode_169
1.暴力法
2.map
3.sort之后，发现哪个重复元素个数最多
4.分治算法 left == right的值，则返回这个值 比较left 和 right哪个更大，返回里面那个较大值 log(n)
  如果里面没有众数，
"""

"""
思路：使用最简单的解法吧，用一个map,统计每个值对应的count,然后返回count值最大的那个数，
则他就是这一组数中出现最多的（众数）
注意点：初始化时可使用定义一个defaultdict（int）,python 没有 ++，--这些
      返回的时间可以使用max函数，遍历字典。得到那个value值最大的返回
"""


class Solution_1(object):
    def majorityElement(self, nums):
        """
        :type nums: List[int]
        :rtype: int
        """
        nums_count = defaultdict(int)
        for i in nums:
            nums_count[i] += 1
        maxvalue, maxcount, = list(num_count.items())[0]
        for i, count in num_count.items():
            if count > maxcount:
                maxcount = count
                maxvalue = i
        return maxvalue


class Solution_2(object):
    def majorityElement(self, nums):
        """
        :type nums: List[int]
        :rtype: int
        """
        nums_count = defaultdict(int)
        for i in nums:
            nums_count[i] += 1
        return max(nums_count, key=nums_count.get)
