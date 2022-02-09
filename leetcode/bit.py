"""
位运算
异或
判断一个数对应的二进制数中1的个数：(3种方法你都掌握了对吧) 嘿嘿
把x最后一位的1给消除掉
x & (x-1)
leetcode_191


leetcode_231
leetcode_338

返回二进制个数为1的总和
去除最低位的1后的数
"""


class Solution(object):
    def hammingWeight(self, n):
        """
        :type n: int
        :rtype: int
        """
        count = 0
        while n:
            n = n & (n - 1)
            count += 1
        return count


class Solution(object):
    def hammingWeight(self, n):
        """
        :type n: int
        :rtype: int
        """
        count = 0
        while n:
            count += (n & 1)
            n >>= 1
        return count


class Solution(object):
    def hammingWeight(self, n):
        """
        :type n: int
        :rtype: int
        """
        return bin(n).count("1")


# leetcode_338
class Solution(object):
    def countBits(self, n):
        """
        :type n: int
        :rtype: List[int]
        """
        res = [0] * (n + 1)
        for i in range(n + 1):
            j = i
            count = 0
            while j:
                j = j & (j - 1)
                count += 1
            res[i] = count
        return res


"""
leetcode_231  2 的幂
"""


class Solution(object):
    def isPowerOfTwo(self, n):
        """
        :type n: int
        :rtype: bool
        """
        # 分析一下，如果n是2的幂次方的话，那么n的二进制必然是这样的，n的二进制的最高为为1，
        # 其他位都为0， 那么对应的 n-1，他的二进制的话是最高位为0，其他位为1，所以他俩与的话
        # 必然结果为0,并且满足n>0，因为2^0=1,所以最小为1
        return n > 0 and n & (n - 1) == 0


"""
布隆过滤器 (bloom filter)
位图 哈希 哈希冲突 多个哈希函数
当判断这个元素不在 100%不在
当判断他的元素在的话 可能误判，它不存在却会判断到存在

速度快
误判率

布隆过滤器，哈希，哈希冲突
"""