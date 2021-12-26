# 题目：两数之和 和 三数之和问题

"""
两数之和

解法一：
思路：
    双层循环，暴力枚举法，时间复杂度 O(n2) 空间复杂度 O（1）
注意点：
    数组下标是从0开始的，range 是左闭右开的 所以，len(nums)就是最后一个元素的下标
"""


# 解法一：
class Solution1(object):
    def twoSum(self, nums, target):
        """
        :type nums: List[int]
        :type target: int
        :rtype: List[int]
        """
        for x in range(0, len(nums) - 1):
            for y in range(x + 1, len(nums)):
                if nums[x] + nums[y] == target:
                    return [x, y]
        return [-1, -1]


"""
解法二：
思路：以空间换时间的思想，可以用一个dict来存：如果本次找到就返回对应的数的下标，否则就把这个数以及这个数对应的
     下标存入dict中；遍历数组，求出当前元素对应的差diff，如果diff已经在dict中，说明找到了匹配的满足条件的，则
     返回两者的下标，如果没有找到，就把当前元素及其下标加入dict,等待后面的元素来匹配，如果匹配上了，就返回满足的结果
     
     想象一个空间，你先来的，此时没找到匹配的，那就先把你放入这个空间，后面的元素依次进行，如果在这个空间中找到匹配了的，
     就返回一对满足条件的解，如果没有找到，他也进来这个空间，如果整个列表遍历完了，没有那就是没有
     
     时间复杂度 O（n），空间复杂度 O(n)
注意点： 
    dict中需要加入当前元素及其对应的下标，key是当前元素，value是下标
"""


# 解法二
class Solution2(object):
    def twoSum(self, nums, target):
        """
        :type nums: List[int]
        :type target: int
        :rtype: List[int]
        """
        another = {}
        for i in range(len(nums)):
            diff = target - nums[i]
            if diff in another.keys():
                return another[diff], i
            another.update({nums[i]: i})


# 三数之和：

"""
三数之和：
思路：
    1.暴力枚举法 时间复杂度 O(n^3)
    2.遍历整个列表，保持当前元素不变，然后对后面的元素进行两面夹击，找满足条件的元素的结果

方法2：
思路：
 先对这个list序列进行排序，然后对list数组遍历，保持当前元素不变，后面的数进行两面夹击法，找满足条件的选项返回
 对list数组进行排序，遍历整个数组，保持当前元素不变，如果当前元素已经大于0了，说明没有满足条件的解，后面的都比当前的大，直接返回结果，
 如果在i>0的情况下，当前元素和前一个元素一样，也没有必要找后面的解，直接continue，避免重复，
 然后定义l,r分别指向后面元素的两边，在l < r下，如果当前元素+nums[l]+nums[r] = 0，则说明找到一个解，继续夹击，如果有像相同的元素
 就要略过去，即nums[l]==nums[l+1],nums[r] == nums[r-1],知道这些重复元素被略完，然后l+1，r-1，
 如果当前的sum值小于0了，则说明小了，需要l往后移移，如果sum>0了，则说明大了，需要r像左移移

注意点：
    1.入参检查，sorted(nums)和nums.sort() 这两个的区别
    2.对于重复的元素的处理，要略过，避免出现重复解
"""


class Solution(object):
    def threeSum(self, nums):
        """
        :type nums: List[int]
        :rtype: List[List[int]]
        """
        res = []
        # 对入参的检查
        if not nums or len(nums) < 3:
            return res

        # 对数组序列进行排序
        nums.sort()
        lenth = len(nums)
        # 开始遍历整个数组序列
        for i in range(lenth):
            # 如果最小的这个初始元素已经>0,那肯定没有满足条件的了，直接返回
            if nums[i] > 0:
                return res
            # 如果在i>0的情况下，当前元素和前一个元素一样，重复了，则没有必要再去后面
            # 夹击，避免出现重复解
            if i > 0 and nums[i] == nums[i - 1]:
                continue

            # 此时就要进行后面的两面夹击了，来找所有满足的条件
            l = i + 1
            r = lenth - 1
            while l < r:
                sum = nums[i] + nums[l] + nums[r]
                if sum == 0:
                    res.append((nums[i], nums[l], nums[r]))
                    # 如果发现当前l的元素和后面元素一样，直接往过跳，跳过重复的部分
                    while l < r and nums[l] == nums[l + 1]:
                        l = l + 1
                    # 如果发现当前r的元素和r左边的元素一样，直接往过跳，r--
                    while l < r and nums[r] == nums[r - 1]:
                        r = r - 1
                    l = l + 1
                    r = r - 1
                elif sum < 0:
                    l = l + 1
                else:
                    r = r - 1
        return res