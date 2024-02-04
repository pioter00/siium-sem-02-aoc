from copy import deepcopy


class MyQueue:
    def __init__(self, size):
        self.list = []
        self.size = size

    def push(self,item):
        self.list.append(item)
        if len(self.list) > self.size:
            del self.list[0]

    def pop(self):
        temp = deepcopy(self.list[0])
        del self.list[0]
        return temp

    def clear(self):
        self.list.clear()

    def isEmpty(self):
        return len(self.list) == 0

    def isFull(self):
        return len(self.list) >= self.size

    def size(self):
        return len(self.list)

    def __str__(self):
        return str(self.list)