"""
排序算法那些
快排 以及快排的优化
十大排序算法 快排，堆排等以及快排的优化
1.快排及快排的优化：
https://blog.csdn.net/insistGoGo/article/details/7785038
4.堆的应用
    1.返回动态数据流中的第k大数字 （leetcode 703） (小根堆)
    2.滑动窗口 （leetcode 239）
"""

"""
堆排
#### 堆排的思想：

```
1.建立大根堆（建堆，建立大根堆或小根堆）
建堆的过程：
1.找第一个倒数的有孩子的父节点，首先，我们这个堆用数组存储，我们取下标按1开始开始吧：可以推出，父子节点下标之间的关系，
父节点下表为i的话，左孩子下标 = 2*i， 右孩子下标 = 2*i+1
根据孩子推父亲节点的下标的话：父亲节点的下标 = i/2

然后我们比较父子节点和孩子节点的大小，我们可以标记一个max_index用来父子和左右孩子中最大的那个的下标，我们比较父子节点和左右孩子
节点的大小，然后将父节点和左右孩子中最大的那个进行交换，交换之后可能造成他下面子树的不稳定，所以我们要自上往下的进行调整，直到他的孩子
节点已经没有了孩子或者说他的子树的父节点已经比左右孩子都大了，说明树已经稳定了，
然后我们再继续操作下一个倒数的父节点，直到最后一个父节点，也就是 i>=1,i是父节点的下标，直到最后一个节点插入后稳定了，那整个建大根堆的
过程就完成了

总体思想：建堆从下往上， 调整堆使堆稳定，自上往下

2.堆排第二步，调整 (我们假设了这个堆下标从1开始，0号元素没有存堆元素)
思想：首先，建立大根堆后，我们的根节点就是最大的那个数，我们从叶子节点末尾从右往左开始调整，首先，让末尾倒数第一个元素金和根
     节点进行交换，交换后，堆肯定不稳定了，然后我们调用一次调整的方法自上而下的调整堆使得堆稳定，直到我们从倒数进行交换的元素
     只剩一个，那整个堆排就完成了


所以，整个堆排他的时间复杂度：O(nlogn),他存在跳跃式的交换，所以他是不稳定的

如果数组下标从0开始，则左孩子下标：2*i+1
                 则右孩子下标：2*i+2
                 则由孩子推父亲的下标的话，父亲下标为：（i-1）/2
"""


class Solution(object):
    def sortArray(self, nums):
        """
        :type nums: List[int]
        :rtype: List[int]
        """
        """
        # 堆排
        # 建大根堆
        # 堆排序
        数组下标是从0开始的，如果当前元素下标为i，
        则他的左孩子下标为2*i+1,右孩子下标为:2*i+2
        如果知道他的左孩子下标为 j，则父亲节点下标为 （j-1）/2
        """
        n = len(nums)

        def buildHeap(nums):
            i = (n - 1) >> 1
            while i >= 0:
                adjustHeap(nums, n, i)
                i -= 1

        def adjustHeap(nums, n, i):
            while (True):
                max_index = i
                l, r = (2 * i) + 1, (2 * i) + 2
                if l < n and nums[l] > nums[i]:
                    max_index = l
                if r < n and nums[r] > nums[max_index]:
                    max_index = r
                if max_index == i:
                    break
                nums[i], nums[max_index] = nums[max_index], nums[i]
                i = max_index

        def heapSort(nums):
            if not nums:
                return
            buildHeap(nums)
            for i in range(n - 1, 0, -1):
                nums[i], nums[0] = nums[0], nums[i]
                adjustHeap(nums, i, 0)

        heapSort(nums)
        return nums


"""
快排
"""

"""
快速排序
快排和快排的优化

快排的思想：
    快排的思想主要就是分治，分而治之，局部有序来保证全局有序
    首先，我们会在序列中选一个基准，比基准大的都放在基准的后面，比基准小的都放在基准的前面，通过一次划分，我们就可以得到基准最终需要放置的
    位置，并把整个序列一分为二，然后利用分治的思想继续对基准的左边的序列和右边的序列进行排序，进行递归，直到基准的两边没有元素或只剩下一个元素，
    这样就通过局部有序，分而治之，实现来整体有序

    关于快排一次划分的思想，首先我们选择基准，可以选择第一个元素或最后一个元素或三者取中或序列中随机下标的元素作为基准，基准可以与第一个元素交换，
    然后我们取第一个元素作为基准，定义low和hign指针，先从后往前找，hign--,直到找到第一个比基准小的数，说明我们需要把这个元素放到前面来，我们让
    arr[low] = arr[hign],然后后面就有可以空的小拼格 ，我们从前往后,low++，直到找到第一个比基准大的，说明他需要放到后面，arr[hign] = arr[low]
    直到low和hign指针相遇，说明一次划分完成，此时他们相遇的位置就是基准应该放置的位置，返回基准的位置，然后进行分治，递归的对基准两侧的序列进行排序，
    直到没有再需要排序的序列区间为止，整个快速排序就完成了

    直接取第一个元素作为基准的话可能会超时，所以我们可以取low和hign里面的随机下标的元素，然后让他和第一个元素交换，然后我们取第一个元素作为基准，
    这样的话，leetcode超时问题就解决了


"""

import random


class Solution(object):
    def sortArray(self, nums):
        """
        :type nums: List[int]
        :rtype: List[int]
        """

        def partition(nums, low, hign):
            # 选择low和hign之间随机下标的元素作为基准
            index = random.randint(low, hign)
            nums[low], nums[index] = nums[index], nums[low]
            paviot = nums[low]
            while low < hign:
                while low < hign and nums[hign] >= paviot:
                    hign -= 1
                nums[low] = nums[hign]
                while low < hign and nums[low] <= paviot:
                    low += 1
                nums[hign] = nums[low]
            nums[low] = paviot
            return low

        def quickSort(nums, low, hign):
            if low < hign:
                mid = partition(nums, low, hign)
                quickSort(nums, low, mid - 1)
                quickSort(nums, mid + 1, hign)

        quickSort(nums, 0, len(nums) - 1)
        return nums


import random


class Solution(object):
    def sortArray(self, nums):
        """
        :type nums: List[int]
        :rtype: List[int]
        """

        # 选择三者取中（low，hign,mid）元素中的第二大，作为基准
        def get_mid_large(nums, low, hign):
            mid_index = low + ((hign - low) >> 1)
            if nums[mid_index] > nums[hign]:
                nums[mid_index], nums[hign] = nums[hign], nums[mid_index]
            if nums[low] > nums[hign]:
                nums[low], nums[hign] = nums[hign], nums[low]
            if nums[mid_index] > nums[low]:
                nums[mid_index], nums[low] = nums[low], nums[mid_index]
            return nums[low]

        def partition(nums, low, hign):
            paviot = get_mid_large(nums, low, hign)
            while low < hign:
                while low < hign and nums[hign] >= paviot:
                    hign -= 1
                nums[low] = nums[hign]
                while low < hign and nums[low] <= paviot:
                    low += 1
                nums[hign] = nums[low]
            nums[low] = paviot
            return low

        def quickSort(nums, low, hign):
            if low < hign:
                mid = partition(nums, low, hign)
                quickSort(nums, low, mid - 1)
                quickSort(nums, mid + 1, hign)

        quickSort(nums, 0, len(nums) - 1)
        return nums


"""
快排 ： 三者取中 + 重复元素聚集
left：始终指向从左向右的 下一个需要放置 与基准相同的元素的元素的位置 
right:始终指向从右向左的 下一个需要放置 与基准相同的元素的元素的位置 

left_lenth:基准左边与基准元素相同的元素的个数
right_lenth：基准右边与基准元素相同的元素的个数

将两边的与基准元素相同的元素移到基准周围

"""
import random


class Solution(object):
    def sortArray(self, nums):
        """
        :type nums: List[int]
        :rtype: List[int]
        """

        # 选择三者取中（low，hign,mid）元素中的第二大，作为基准
        def get_mid_large(nums, low, hign):
            mid_index = low + ((hign - low) >> 1)
            if nums[mid_index] > nums[hign]:
                nums[mid_index], nums[hign] = nums[hign], nums[mid_index]
            if nums[low] > nums[hign]:
                nums[low], nums[hign] = nums[hign], nums[low]
            if nums[mid_index] > nums[low]:
                nums[mid_index], nums[low] = nums[low], nums[mid_index]
            return nums[low]

        def partition(nums, low, hign):
            paviot = get_mid_large(nums, low, hign)
            first, last = low, hign
            left, right = low, hign
            left_lenth, right_lenth = 0, 0
            while low < hign:
                while low < hign and nums[hign] >= paviot:
                    if nums[hign] == paviot:
                        nums[right], nums[hign] = nums[hign], nums[right]
                        right -= 1
                        right_lenth += 1
                    hign -= 1
                nums[low] = nums[hign]
                while low < hign and nums[low] <= paviot:
                    if nums[low] == paviot:
                        nums[left], nums[low] = nums[low], nums[left]
                        left += 1
                        left_lenth += 1
                    low += 1
                nums[hign] = nums[low]
            nums[low] = paviot

            # 把两边的元素放置到基准元素的周围
            i, j = low - 1, first
            while j < left and nums[i] != paviot:
                nums[j], nums[i] = nums[i], nums[j]
                i -= 1
                j += 1
            i, j = low + 1, last
            while j > right and nums[i] != paviot:
                nums[i], nums[j] = nums[j], nums[i]
                j -= 1
                i += 1
            return low, left_lenth, right_lenth

        def quickSort(nums, low, hign):
            if low < hign:
                mid, left_lenth, right_lenth = partition(nums, low, hign)
                quickSort(nums, low, mid - 1 - left_lenth)
                quickSort(nums, mid + 1 + right_lenth, hign)

        quickSort(nums, 0, len(nums) - 1)
        return nums


"""
归并排序
思想：归并排序的思想就是分治，分而治之，整体代码结果有点像二叉树的后序遍历，先将序列一份为二
    继续分治处理，直至序列<=1 一个元素
    局部有序，然后合并有序数组的逻辑（类似与合并两个有序链表），保证最后整体有序
"""


class Solution(object):
    def sortArray(self, nums):
        """
        :type nums: List[int]
        :rtype: List[int]
        """

        def mergeSort(nums):
            if len(nums) <= 1:
                return nums
            mid = len(nums) >> 1
            left = mergeSort(nums[:mid])
            right = mergeSort(nums[mid:])
            return mergeArray(left, right)

        def mergeArray(left, right):
            res = []
            i, j = 0, 0
            while i < len(left) and j < len(right):
                if left[i] <= right[j]:
                    res.append(left[i])
                    i += 1
                else:
                    res.append(right[j])
                    j += 1
            res += left[i:]
            res += right[j:]
            return res

        return mergeSort(nums)


"""
冒泡排序
两两比较，每一趟冒泡都会冒泡出去一个数字，
最外层循环表示冒泡的趟数，需要n-1次，下标是从0开始的，所以为 range(0,n-1)
里面这层循环表示从0开始，和除去那些已经冒泡排序确定好的数字，然后后面的两两比较，if nums[j] > nums[j + 1]就进行交换
这样就冒泡出一个最大数到后面
平均时间复杂度 O(n^2)
最好情况下 O（n）完全有序的情况下
最好情况下 O(n^2)
"""


class Solution(object):
    def sortArray(self, nums):
        """
        :type nums: List[int]
        :rtype: List[int]
        """

        # 冒泡排序
        def bubbleSort(nums):
            n = len(nums)
            for i in range(n - 1):
                for j in range(0, n - 1 - i, 1):
                    if nums[j] > nums[j + 1]:
                        nums[j], nums[j + 1] = nums[j + 1], nums[j]
            return nums

        return bubbleSort(nums)


"""
选择排序

选择排序的思想：就是先确定待排序序列，每个位置都放入那个最正确的数
比如，先确定第一个位置，然后找他后面的所有序列中最小的那个数的下标，找到后，如果最小的那个数就是第一个数，那不用交换
否则交换，第一个位置放最小的那个数，然后第二个位置继续以此类推，直到每个位置上都放入了那个正确的数 
那整个序列就是有序的了
"""


class Solution(object):
    def sortArray(self, nums):
        """
        :type nums: List[int]
        :rtype: List[int]
        """

        def selectSort(nums):
            n = len(nums)
            for i in range(n - 1):
                min_index = i
                for j in range(i + 1, n, 1):
                    if nums[j] < nums[min_index]:
                        min_index = j
                if min_index == i:
                    continue
                nums[i], nums[min_index] = nums[min_index], nums[i]
            return nums

        return selectSort(nums)
