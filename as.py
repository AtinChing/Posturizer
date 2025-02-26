sequence = input()
print(sequence)
class Node:
    def __init__(self, item, next_item):
        self.value = item
        self.next = next_item

class LinkedList:
    def __init__(self, items):
        self.head = Node(items[0], None)
        node = self.head
        for i in range(len(items)):
            node.next = Node(items[i], None)
            node = node.next

linked_list = LinkedList(sequence)

node = linked_list.head
while node.value != None:
    print(node.value, "-",)
    node = node.next
print(None)