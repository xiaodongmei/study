"""
lru cache
当缓存没满，继续进
否则，如果缓存满了，就要把最老的给淘汰了，如果这个元素之前没有在缓存中的话
双向链表
LFU 统计出现的次数，访问最高的放前面 按照频次进行排序放置的
当满了的话，淘汰掉最后一个频次最低的那个
缓存替换策略
多写多练

哈希表 + 双向链表

删除的时候，不仅链表中要删除，哈希表中也要把这个元素删除了
"""

# LRU算法的思想是：淘汰掉最近最久未使用的 我们要在O（1）的时间复杂度查找元素和插入和删除一个元素
# 所以，我们想到的数据结构是哈希➕双向链表的方式，哈希可以保证我们的找出元素时间复杂度为O(1)，使用双向链表的话
# 插入和删除时间复杂度为 O（1)
# 当我们新访问一个元素时，根据lru算法的思想我们需要看之前这个元素在不在哈希中，如果存在我们需要把他从链表中删除掉，然后已到链表的尾端（也就是队头）

# 当我们新增加一个元素时，我们需要看这个元素之前是否已经存在，也就是hash中已经有了，如果已经有了。我们就更新hash中它的值，
# 然后在链表中删除他，并把它插入双链表的尾端（队头），如果没有，并且容量已经满了，则我们需要删除掉最近最久未使用的那个元素，（从hash表和链表中都需要删除掉这个元素，）并把新的元素插入到双链表的尾端（队头）

"""
常见错误点注意：类函数注意self变量，调用类函数时要self.这种
第二，链表插入元素时先把链表都挂上去，前驱后继的啥的，然后再断链，还有就是头节点除删除时。注意，head的next已经挂过去了，
所以挂前驱时直接时，self.head.next.pre = self.head
再一个，put插入新元素时，要把这个元素先入进hash表，必须的
"""


class ListNode(object):
    def __init__(self, key=None, value=None):
        self.key = key
        self.value = value
        self.pre = None
        self.next = None


class LRUCache(object):

    def __init__(self, capacity):
        """
        :type capacity: int
        """
        self.capacity = capacity
        self.head = ListNode()
        self.tail = ListNode()
        self.head.next = self.tail
        self.tail.pre = self.head
        self.hashmap = {}

    def move_node_to_tail(self, key):
        node = self.hashmap.get(key)
        # 先从链表中删除掉这个元素
        node.pre.next = node.next
        node.next.pre = node.pre

        # 把这个元素插入到双链表的尾端
        node.next = self.tail
        node.pre = self.tail.pre
        self.tail.pre.next = node
        self.tail.pre = node

    def get(self, key):
        """
        :type key: int
        :rtype: int
        """
        if self.hashmap.get(key):
            self.move_node_to_tail(key)
        res = self.hashmap.get(key, -1)
        if res == -1:
            return res
        else:
            return res.value

    def put(self, key, value):
        """
        :type key: int
        :type value: int
        :rtype: None
        """
        if self.hashmap.get(key):
            node = self.hashmap[key]
            node.value = value
            self.move_node_to_tail(key)
        else:
            if len(self.hashmap) == self.capacity:
                self.hashmap.pop(self.head.next.key)
                self.head.next = self.head.next.next
                self.head.next.pre = self.head
            node = ListNode(key, value)
            self.hashmap[key] = node
            node.next = self.tail
            node.pre = self.tail.pre
            self.tail.pre.next = node
            self.tail.pre = node
