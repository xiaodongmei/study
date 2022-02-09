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