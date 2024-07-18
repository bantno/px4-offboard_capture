import os
import ompl.base as ob
import ompl.geometric as og
import numpy as np
import random
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from scipy.interpolate import BSpline, splprep, splev

class PathPlanner:
    def __init__(self, bounds=(-4, 4), obstacles=None):
        self.space = ob.RealVectorStateSpace(3)
        self.bounds = ob.RealVectorBounds(3)
        self.bounds.setLow(bounds[0])
        self.bounds.setHigh(bounds[1])
        self.space.setBounds(self.bounds)

        self.si = ob.SpaceInformation(self.space)
        self.si.setStateValidityChecker(ob.StateValidityCheckerFn(self.isStateValid))

        self.start = ob.State(self.space)
        self.goal = ob.State(self.space)
        self.goal_above = ob.State(self.space)

        self.obstacles = obstacles if obstacles else []

        self.problem = ob.ProblemDefinition(self.si)
        self.problem.setOptimizationObjective(ob.PathLengthOptimizationObjective(self.si))
        self.planner = og.RRTstar(self.si)
        self.planner.setProblemDefinition(self.problem)
        self.planner.setRange(2.0)
        self.planner.setGoalBias(0.05)
        self.planner.setup()

    def set_random_start_and_goal(self):
        for i in range(3):
            self.start[i] = random.uniform(self.bounds.low[i], self.bounds.high[i])
            self.goal[i] = random.uniform(self.bounds.low[i], self.bounds.high[i])
        self.goal_above[0] = self.goal[0]
        self.goal_above[1] = self.goal[1]
        self.goal_above[2] = self.goal[2] + 0.75
        self.problem.setStartAndGoalStates(self.start, self.goal_above)
        self.obstacles.append((self.goal[0], self.goal[1], self.goal[2], 0.25))

    def set_start_and_goal(self,start,goal):
        for i in range(3):
            self.start[i] = start[i]
            self.goal[i] = goal[i]
        self.goal_above[0] = self.goal[0]
        self.goal_above[1] = self.goal[1]
        self.goal_above[2] = self.goal[2] + 0.75
        self.problem.setStartAndGoalStates(self.start, self.goal_above)
        self.obstacles = [(self.goal[0], self.goal[1], self.goal[2], 0.25)]

    def isStateValid(self, state):
        for obstacle in self.obstacles:
            dist = np.sqrt((state[0] - obstacle[0])**2 + (state[1] - obstacle[1])**2 + (state[2] - obstacle[2])**2)
            if dist <= obstacle[3]:
                return False
        return True

    def isPathValid(self, path):
        last_state = path.getState(path.getStateCount() - 1)
        return last_state[2] > self.goal[2]

    def compute_bezier_points(self, p0, p1, p2, p3, num_samples=100):
        t = np.linspace(0, 1, num_samples).reshape(num_samples, 1)
        curve = (1 - t)**3 * p0 + 3 * (1 - t)**2 * t * p1 + 3 * (1 - t) * t**2 * p2 + t**3 * p3
        return curve

    def smooth_with_bezier(self, points, num_samples=100):
        points = np.array(points)
        assert points.ndim == 2 and points.shape[1] == 3, "Points must be a 2-dimensional array with shape (N, 3)."
        N = points.shape[0]
        if N < 4:
            raise ValueError("At least 4 points are required to create a Bezier curve.")
        smoothed_points = []
        control_points = []
        for i in range(0, N-1, 3):
            p0 = points[i]
            p1 = points[min(i+1, N-1)]
            p2 = points[min(i+2, N-1)]
            p3 = points[min(i+3, N-1)]
            bezier_curve = self.compute_bezier_points(p0, p1, p2, p3, num_samples)
            smoothed_points.append(bezier_curve)
            control_points.extend([p0, p1, p2, p3])
        smoothed_points = np.concatenate(smoothed_points, axis=0)
        control_points = np.array(control_points).reshape(-1, 3)
        return smoothed_points, control_points

    def smooth_with_bspline(self, points, num_samples=100, degree=3, smoothness=0):
        points = np.array(points)
        assert points.ndim == 2 and points.shape[1] == 3, "Points must be a 2-dimensional array with shape (N, 3)."
        tck, _ = splprep(points.T, k=degree, s=smoothness)
        control_points = np.array(tck[1]).T
        u_new = np.linspace(0, 1, num_samples)
        smoothed_points = np.array(splev(u_new, tck)).T
        return smoothed_points, control_points

    def plot_sphere(self, ax, obstacle):
        u, v = np.mgrid[0:2*np.pi:20j, 0:np.pi:10j]
        x = obstacle[3] * np.cos(u) * np.sin(v) + obstacle[0]
        y = obstacle[3] * np.sin(u) * np.sin(v) + obstacle[1]
        z = obstacle[3] * np.cos(v) + obstacle[2]
        ax.plot_wireframe(x, y, z, color="r")

    def plot_path(self, path):
        path_states = path.getStates()
        x_vals = [state[0] for state in path_states]
        y_vals = [state[1] for state in path_states]
        z_vals = [state[2] for state in path_states]
        x_vals.append(self.goal[0])
        y_vals.append(self.goal[1])
        z_vals.append(self.goal[2])
        states = np.array([x_vals, y_vals, z_vals], dtype=float).transpose()
        smoothed_points, control_points = self.smooth_with_bezier(states)
        fig = plt.figure()
        ax = fig.add_subplot(111, projection='3d')
        ax.plot(x_vals, y_vals, z_vals, label='Planned Path')
        ax.scatter(control_points[:,0], control_points[:,1], control_points[:,2], label='Planned Path Control Points')
        ax.plot(smoothed_points[:,0], smoothed_points[:,1], smoothed_points[:,2], label='Smoothed Planned Path')
        ax.scatter([self.start[0]], [self.start[1]], [self.start[2]], color='green', label='Start')
        ax.scatter([self.goal[0]], [self.goal[1]], [self.goal[2]], color='red', label='Goal')
        ax.set_xlim([-4, 4])
        ax.set_ylim([-4, 4])
        ax.set_zlim([-4, 4])
        ax.set_xlabel('X')
        ax.set_ylabel('Y')
        ax.set_zlabel('Z')
        ax.legend(loc='center left' ,bbox_to_anchor=(1.1, 1.05))
        plt.tight_layout()
        for obstacle in self.obstacles:
            self.plot_sphere(ax, obstacle)
        plt.show()

    def solve(self, time=0.45,plot:bool=False):
        solved = self.planner.solve(time)
        if solved and self.isPathValid(self.problem.getSolutionPath()):
            path = self.problem.getSolutionPath()
            self.path = path
            if plot==True:
                self.plot_path(path)
        else:
            print("No solution found or the solution does not approach the goal from the top.")
            self.path = self.start

if __name__ == "__main__":
    planner = PathPlanner()
    planner.set_start_and_goal(start=[0.0,0.0,0.0], goal=[1.0,1.0,1.0])
    # planner.set_random_start_and_goal()
    planner.solve(plot=False)
    print(planner.path)
