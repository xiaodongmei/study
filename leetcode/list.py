# Definition for singly-linked list.
# class ListNode(object):
#     def __init__(self, x):
#         self.val = x
#         self.next = None


"""
思路：链表的逆置，我们可以用三个指针，pcur指向当前要处理的节点，pre指向前驱，next用了记录下一个节点
     当断链时，我们需要先记录下next，防止后面链丢失
     然后让pcur->next指向前驱，前驱pre紧跟过来指向pcur，pcur指向next
     当pcur节点为NULL时，说明需要逆置的节点完了，此时pre指向最后一个节点，也就是逆置节点的头，返回pre
注意点：注意断链时要先把后面的链记录下来
"""


# leetcode_206
# 链表的逆置
class Solution(object):
    def reverseList(self, head):
        """
        :type head: ListNode
        :rtype: ListNode
        """
        if head == None:
            return
        pre, qcur = None, head
        while qcur:
            qnext = qcur.next
            qcur.next, pre, qcur = pre, qcur, qnext
        return pre


# 从尾到头打印链表 剑指 Offer 06
# 方法一 可以利用栈的特性 先进后出
"""
思路：从尾到头打印链表，可以利用栈，先把节点都压进栈，利用栈后进先出的特性，完成从尾到头打印链表
注意点：我们用列表表示栈，返回时，对列表元素逆置 stack[::-1]
"""


class Solution(object):
    def reversePrint(self, head):
        """
        :type head: ListNode
        :rtype: List[int]
        """
        stack = []
        while head:
            stack.append(head.val)
            head = head.next
        return stack[::-1]


# 可以利用递归的方式
"""
思路：利用递归的方式，递归的本质也是压栈
注意点：res里压进值，因为最后需要的是值列表
"""


class Solution(object):
    def reversePrint(self, head):
        """
        :type head: ListNode
        :rtype: List[int]
        """
        res = []

        def reversePrintCore(head):
            if head:
                reversePrintCore(head.next)
                res.append(head.val)

        reversePrintCore(head)
        return res


# 判断链表是否有环 leetcode_141
"""
思路：通过快慢指针就可以了，慢指针每次走一步，快指针走两步，如果他们两个相遇了，那肯定有环
注意点：需要判断slow and fast and fast.next 有值，防止越界
"""


class Solution(object):
    def hasCycle(self, head):
        """
        :type head: ListNode
        :rtype: bool
        """
        slow, fast = head, head
        while slow and fast and fast.next:
            slow = slow.next
            fast = fast.next.next
            if slow is fast:
                return True
        return False


# leetcode_142 链表中环的入口节点
"""
思路：找链表中环的入口节点：
    解法一：如果我们知道环的长度n，就可以定义快慢指针 ahead 和bebind,让ahead先走n步，
          然后behind和ahead一起走，他们相遇的地方就是环的入口节点
    解法二：可以定义快慢指针，fast和slow,fast一次走2步，slow一次走1步，当快慢指针相遇时，
          我们让一个指针point指向头节点，然后让point节点和slow指针一起走，当他们相遇时。
          此时的节点便是环的入口节点
注意点：注意防止越界问题
"""


# Definition for singly-linked list.
# class ListNode(object):
#     def __init__(self, x):
#         self.val = x
#         self.next = None

class Solution(object):
    def detectCycle(self, head):
        """
        :type head: ListNode
        :rtype: ListNode
        """
        slow, fast = head, head
        while fast and fast.next:
            slow = slow.next
            fast = fast.next.next
            if slow == fast:
                point = head
                while point != slow:
                    point = point.next
                    slow = slow.next
                return point
        return None


# 找链表中的倒数第k个节点 剑指 Offer 22

"""
思路：
    可以利用两个指针，让第一个指针先走 k-1步，然后用另一个指针指向head，两个同时走，
    当第一个指针走到最后一个元素时，此时第二个指针指向的位置就是我们要求的链表中的
    倒数第k个节点 
    倒数第k个节点 = n - k +1 = n-(k-1)
注意点：注意代码鲁棒性，对入参节点的判断，还有判断k的合法性，如果倒数第k个节点，链表都没有那么长，那肯定不行
      注意是：当第一个指针走到最后一个元素时，此时ahead.next已经为空
"""


class Solution(object):
    def getKthFromEnd(self, head, k):
        """
        :type head: ListNode
        :type k: int
        :rtype: ListNode
        """
        if head is None or k <= 0:
            return None
        pAhead = head
        for i in range(k - 1):
            if pAhead.next is not None:
                pAhead = pAhead.next
            else:
                return None
        pBehind = head
        while pAhead.next:
            pAhead = pAhead.next
            pBehind = pBehind.next
        return pBehind


# 找两个链表的第一个公共节点 剑指 Offer 52
"""
有两种解法：
解法一：如果我们求出两个链表的长度，便可以得到他们的长度差，让长的那个先走diff步，然后两个指针仔同时走
      此时他们碰到第一个相同的节点。便是两个链表的第一个公共节点
注意点：注意越界问题
"""


class Solution(object):
    def getLenth(self, head):
        count = 0
        while head:
            count += 1
            head = head.next
        return count

    def getIntersectionNode(self, headA, headB):
        """
        :type head1, head1: ListNode
        :rtype: ListNode
        """
        lenth1, lenth2 = self.getLenth(headA), self.getLenth(headB)
        diff_lenth = abs(lenth1 - lenth2)
        if lenth1 > lenth2:
            pfast = headA
            pslow = headB
        else:
            pfast = headB
            pslow = headA
        for i in range(diff_lenth):
            pfast = pfast.next
        while pfast and pslow:
            if pfast is pslow:
                return pslow
            pfast = pfast.next
            pslow = pslow.next
        return None


"""
解法二：
思想：如果当前节点不为空的话。就继续走，指向他的next，如果当前节点已经为空了，
       说明走完了，直接让他指向另一个链表的头
       你变成我，走过我走过的路。
       我变成你，走过你走过的路。
       然后我们便相遇了 
       当我们走过的路都为 L1+L2+C时，我们便相遇了 c为我们的公共部分
注意点：非常巧妙，哈哈，你的名字里的经典场景，平行时空
"""


class Solution(object):
    def getIntersectionNode(self, headA, headB):
        """
        :type head1, head1: ListNode
        :rtype: ListNode
        """
        node1, node2 = headA, headB
        while node1 != node2:
            node1 = node1.next if node1 else headB
            node2 = node2.next if node2 else headA
        return node1


"""
合并两个有序的链表 leetcode_21
有两种方法：
解法一：
思想：
    指针法，有点类似归并排序里合并两个有序数组，合并完 整体是一个有序的
    我们用两个指针分表指向这两个链表，pnewhead，pcur在合并新链表时用到，pnewhead指向新链表的头
    pcur用来跟进此时加入新链表的解点，用于把整个链表都串起来
    最后若两个链表有一个已经被合并完了，则把另一个还不为NULL的链表挂到 pcur_next 
注意点：
    注意越界问题
"""


class Solution(object):
    def mergeTwoLists(self, list1, list2):
        """
        :type list1: Optional[ListNode]
        :type list2: Optional[ListNode]
        :rtype: Optional[ListNode]
        """
        if not list1:
            return list2
        if not list2:
            return list1
        pnode1, pnode2 = list1, list2
        pnewhead, pcur = None, None
        if pnode1.val < pnode2.val:
            pnewhead = pnode1
            pcur = pnode1
            pnode1 = pnode1.next
        else:
            pnewhead = pnode2
            pcur = pnode2
            pnode2 = pnode2.next

        while pnode1 and pnode2:
            if pnode1.val < pnode2.val:
                pcur.next = pnode1
                pcur = pnode1
                pnode1 = pnode1.next
            else:
                pcur.next = pnode2
                pcur = pnode2
                pnode2 = pnode2.next
        if not pnode1 and pnode2:
            pcur.next = pnode2
        if not pnode2 and pnode1:
            pcur.next = pnode1

        return pnewhead


"""
解法二：
思想：可以利用递归来做，因为合并链表的过程本质上就是不断重复 比较链表元素大小，然后把节点挂到新链表这个过程
"""


class Solution(object):
    def mergeTwoLists(self, list1, list2):
        """
        :type list1: Optional[ListNode]
        :type list2: Optional[ListNode]
        :rtype: Optional[ListNode]
        """
        if not list1:
            return list2
        if not list2:
            return list1
        pnewhead = None

        if list1.val <= list2.val:
            pnewhead = list1
            pnewhead.next = self.mergeTwoLists(list1.next, list2)
        else:
            pnewhead = list2
            pnewhead.next = self.mergeTwoLists(list2.next, list1)
        return pnewhead
