"""
贪心算法：问题可以分解成子问题来解决，每个子问题的最优解可以推导出最终问题的最优解，拥有最优子结构
        贪心算法在对当下做出最优的选择后，不具体回退功能，但是动态规划可以回退，会保存之前的结果，
        最终选择出问题整体的最优解决方式
"""

"""
剪枝：布局最优的分支
     全局最优的分支
     
     状态空间的扩展和搜索能力强
     接下来一步的可能的状态
     所有可能的状态空间搜素
"""

"""
剪枝问题
leetcode_51 leetcode_52 N皇后问题

打印出所有可能皇后放置的位置

DFS
每一层皇后可能放的位置
怎么判断这个位置皇后可不可以放

1.暴力枚举法
2.数组将已经存的位置记录下来，
回溯 + 剪枝
(0,1),(1,2)(2,3)
x+y = c
y-x = c

当前层做完后还要恢复现场
"""

"""
全排列问题

"""

"""
数独问题
leetcode_36 leetcode_37

对于重复的数怎么进行有效的标记和去重

要填的位置所在的行不能里面已经出现这个数
要填的位置所在的列不能里面已经出现这个数



搜索 + 剪枝
怎么搜索呢，感觉每个都要枚举一下
有冲突，就回溯，把之前的去掉

i+1，j = 0

枚举 1～9
check_valid
剪枝：
1.从选项少的入手
2.预处理
3.dancinglink
4.位运算判重
"""

"""
leetcode_69
1.二分法 无限逼近
float类型 无法直接加1。减少1
2.牛顿迭代法
二分查找
"""

"""
字典树 trie
leetcode_208

"""

"""
leetcode 212 word search II
"""
