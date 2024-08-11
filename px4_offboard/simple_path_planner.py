import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

def plan_path(start, goal):
    # Define the intermediate point (half a meter above the goal in the -z direction)
    intermediate = np.array([goal[0], goal[1], goal[2] - 10.0])

    # Bezier control points: start, intermediate, goal
    control_points = np.array([start, intermediate, goal])
    
    # Function to calculate the Bezier point at t
    def bezier_point(t, control_points):
        n = len(control_points) - 1
        point = np.zeros(3)
        for i in range(n + 1):
            binomial_coeff = np.math.factorial(n) / (np.math.factorial(i) * np.math.factorial(n - i))
            point += binomial_coeff * (1 - t)**(n - i) * t**i * control_points[i]
        return point

    # Generate the Bezier curve points
    t_values = np.linspace(0, 1, 100)
    path = np.array([bezier_point(t, control_points) for t in t_values])

    return path

def plot_path(start, goal, path):
    # Plotting the path
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    ax.plot(path[:, 0], path[:, 1], path[:, 2], label="Bezier Path")
    ax.scatter([start[0], goal[0]], [start[1], goal[1]], [start[2], goal[2]], color='red', label="Start/Goal")
    ax.scatter([goal[0]], [goal[1]], [goal[2] - 0.5], color='blue', label="Intermediate Point")
    ax.legend()
    ax.set_xlabel("X")
    ax.set_ylabel("Y")
    ax.set_zlabel("Z")
    ax.set_title("Path Planning with Bezier Spline")
    plt.show()

# Example usage
goal = np.array([-1, -1, -4])
start = np.array([-1.5, -1.5, -3])

path = plan_path(start, goal)
plot_path(start, goal, path)
