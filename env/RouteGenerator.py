import numpy as np

class RouteGenerator:
    def __init__(self, bounds=(-100, 100, -100, 100), min_points=3, max_points=8):
        """
        bounds: (xmin, xmax, ymin, ymax) 平面范围，高度默认0
        """
        self.bounds = bounds
        self.min_points = min_points
        self.max_points = max_points

    def generate_route(self):
        n = np.random.randint(self.min_points, self.max_points + 1)
        waypoints = []
        for _ in range(n):
            x = np.random.uniform(self.bounds[0], self.bounds[1])
            y = np.random.uniform(self.bounds[2], self.bounds[3])
            waypoints.append([x, y, 0.0])
        return waypoints