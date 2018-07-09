import numpy


class PriorityTree:
    def __init__(self, capacity):
        self.current_memory_size = 0
        self.capacity = capacity
        self.data = numpy.zeros(capacity, dtype=object)
        self.priority_tree = numpy.zeros(2 * capacity - 1)

    def add(self, priority, data):
        index = self.current_memory_size + self.capacity - 1
        self.data[self.current_memory_size] = data
        self.update_priority(index, priority)
        self.current_memory_size += 1
        if self.current_memory_size >= self.capacity:
            self.current_memory_size = 0

    def _propagate_changes(self, index, change):
        parent_index = (index - 1) // 2
        self.priority_tree[parent_index] += change
        if parent_index:
            self._propagate_changes(parent_index, change)

    def get_sample(self, s):
        index = self._retrieve_sample(0, s)
        data_index = index - self.capacity + 1
        return index, self.priority_tree[index], self.data[data_index]

    def _retrieve_sample(self, index, s):
        left_child_index = 2 * index + 1
        right_child_index = left_child_index + 1

        if left_child_index >= len(self.priority_tree):
            return index

        if s <= self.priority_tree[left_child_index]:
            return self._retrieve_sample(left_child_index, s)
        else:
            return self._retrieve_sample(right_child_index, s - self.priority_tree[left_child_index])

    def update_priority(self, index, priority):
        change = priority - self.priority_tree[index]
        self.priority_tree[index] = priority
        self._propagate_changes(index, change)

    def total(self):
        return self.priority_tree[0]
