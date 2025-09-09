import math
import heapq
from collections import deque

class PipeNetwork:
    def __init__(self, n):
        self.n = n
        self.graph = [[] for _ in range(n)]
        self.coordinates = [(0, 0)] * n

    def add_pipe(self, u, v, cost):
        self.graph[u].append((v, cost))
        self.graph[v].append((u, cost))

    def set_coordinates(self, junction, x, y):
        self.coordinates[junction] = (x, y)

    def straight_distance(self, a, b):
        x1, y1 = self.coordinates[a]
        x2, y2 = self.coordinates[b]
        return math.sqrt((x1 - x2) ** 2 + (y1 - y2) ** 2)

    # DFS
    def dfs(self, start, target):
        visited = [False] * self.n
        path = []

        def dfs_visit(u):
            visited[u] = True
            path.append(u)
            if u == target:
                return True
