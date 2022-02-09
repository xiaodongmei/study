"""
链表的题目全部做一遍
/*
链表常见算法题
1.链表的逆置
2.从尾到头打印链表
3.判断链表是否有环
4.找链表中的倒数第k个节点
5.寻找两个链表的第一个公共节点
6.合并两个有序的链表
leetcode_142 链表中环的入口节点
*/
"""


# Definition for singly-linked list.
# class ListNode(object):
#     def __init__(self, x):
#         self.val = x
#         self.next = None

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


# 从尾到头打印链表

# 方法一 可以利用栈的特性 先进后出
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


# 判断链表是否有环
"""
通过快慢指针就可以了，慢指针每次走一步，快指针走两步，如果他们两个相遇了，那肯定有环
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
环的入口位置该怎么定位呢，还是快慢指针吗
链表中环的入口节点
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


# 找链表中的倒数第k个节点

"""
可以利用两个指针，让第一个指针先走 k-1步，然后用另一个指针指向head，两个同时走，
当第一个指针走到最后一个元素时，此时第二个指针指向的位置就是我们要求的链表中的
倒数第k个节点 
倒数第k个节点 = n - k +1 = n-(k-1)
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


# 找两个链表的第一个公共节点

# Definition for singly-linked list.
# class ListNode(object):
#     def __init__(self, x):
#         self.val = x
#         self.next = None

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
要注意：如果当前节点不为空的话。就继续走，指向他的next，如果当前节点已经为空了，
       说明走完了，直接让他指向另一个链表的头
       你变成我，走过我走过的路。
       我变成你，走过你走过的路。
       然后我们便相遇了 
       当我们走过的路都为 L1+L2+C时，我们便相遇了 c为我们的公共部分
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
合并两个有序的链表
"""


# Definition for singly-linked list.
# class ListNode(object):
#     def __init__(self, val=0, next=None):
#         self.val = val
#         self.next = next
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
