"""
滑动窗口题目
"""

"""
哈希表相关
5.哈希算法
给两个字符串，判断是否为Anagram  (leetcode 242)
给一组数字，这些数字里面每一个都重复出现了三次，只有一个数字只出现了一个，要求在时间O（n）空间O（1）内解出来
数组中奇数个元素
一栋楼有n层，不知道鸡蛋从第几层扔下去会碎，用最少的次数找出刚好会碎的楼层
连续整数求和(leetcode 第 829 题)，要求时间复杂度小于O(N)
一个无序数组找其子序列构成的和最大，要求子序列中的元素在原数组中两两都不相邻

1.完全平方数最少的 leetcode_279
2.积水珠最多的容器 leetcode_11

去掉字符串中所有的空格

字符串的匹配 bf和kmp
"""

"""
并查集
"""

"""
字典树
"""

"""
把这些按分类刷完了，然后要进行总结，思维导图和扩展
请教朋友，看还存在什么问题，自己感觉知识图谱形成了，经典题目
思路都get了，就开始刷leetcode hot 200道。加油！！
"""



#
# class Solution:
#     def groupAnagrams(self, strs: List[str]) -> List[List[str]]:
#         dict = {}
#         for item in strs:
#             key = tuple(sorted(item))
#             dict[key] = dict.get(key, []) + [item]
#         return list(dict.values())