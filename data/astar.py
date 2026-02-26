import heapq
import numpy as np
from typing import Tuple, List, Optional


def heuristic(a: Tuple[int, int, int], b: Tuple[int, int, int]) -> int:
    return abs(a[0] - b[0]) + abs(a[1] - b[1]) + abs(a[2] - b[2])


def astar_3d(grid: np.ndarray, start: Tuple[int, int, int], end: Tuple[int, int, int]) -> Optional[List[Tuple[int, int, int]]]:
    size = grid.shape[0]
    directions = [
        (1, 0, 0), (-1, 0, 0),
        (0, 1, 0), (0, -1, 0),
        (0, 0, 1), (0, 0, -1)
    ]
    
    open_set = [(0, start)]
    came_from = {}
    g_score = {start: 0}
    f_score = {start: heuristic(start, end)}
    visited = set()
    
    while open_set:
        _, current = heapq.heappop(open_set)
        
        if current in visited:
            continue
        visited.add(current)
        
        if current == end:
            path = []
            while current in came_from:
                path.append(current)
                current = came_from[current]
            path.append(start)
            return path[::-1]
        
        for dx, dy, dz in directions:
            neighbor = (current[0] + dx, current[1] + dy, current[2] + dz)
            
            if not (0 <= neighbor[0] < size and 0 <= neighbor[1] < size and 0 <= neighbor[2] < size):
                continue
            
            if grid[neighbor] == 1:
                continue
            
            if neighbor in visited:
                continue
            
            tentative_g = g_score[current] + 1
            
            if neighbor not in g_score or tentative_g < g_score[neighbor]:
                came_from[neighbor] = current
                g_score[neighbor] = tentative_g
                f = tentative_g + heuristic(neighbor, end)
                f_score[neighbor] = f
                heapq.heappush(open_set, (f, neighbor))
    
    return None


def dijkstra_3d(grid: np.ndarray, start: Tuple[int, int, int], end: Tuple[int, int, int]) -> Optional[List[Tuple[int, int, int]]]:
    size = grid.shape[0]
    directions = [
        (1, 0, 0), (-1, 0, 0),
        (0, 1, 0), (0, -1, 0),
        (0, 0, 1), (0, 0, -1)
    ]
    
    distances = {start: 0}
    came_from = {}
    pq = [(0, start)]
    visited = set()
    
    while pq:
        dist, current = heapq.heappop(pq)
        
        if current in visited:
            continue
        visited.add(current)
        
        if current == end:
            path = []
            while current in came_from:
                path.append(current)
                current = came_from[current]
            path.append(start)
            return path[::-1]
        
        for dx, dy, dz in directions:
            neighbor = (current[0] + dx, current[1] + dy, current[2] + dz)
            
            if not (0 <= neighbor[0] < size and 0 <= neighbor[1] < size and 0 <= neighbor[2] < size):
                continue
            
            if grid[neighbor] == 1:
                continue
            
            if neighbor in visited:
                continue
            
            new_dist = dist + 1
            
            if neighbor not in distances or new_dist < distances[neighbor]:
                distances[neighbor] = new_dist
                came_from[neighbor] = current
                heapq.heappush(pq, (new_dist, neighbor))
    
    return None


if __name__ == "__main__":
    grid = np.zeros((8, 8, 8), dtype=np.float32)
    grid[2:5, 2:5, 2:5] = 1
    
    start = (0, 0, 0)
    end = (7, 7, 7)
    
    path = astar_3d(grid, start, end)
    if path:
        print(f"A*找到路径，长度: {len(path)}")
        print(f"路径: {path[:5]}...{path[-5:]}")
    else:
        print("A*未找到路径")
