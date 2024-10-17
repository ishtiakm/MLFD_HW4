import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.svm import SVC

class Coordinate:
    def __init__(self, x=0.0, y=0.0):
        self.x = x
        self.y = y

    def dist(self, other_point):
        return np.sqrt((self.x - other_point.x) ** 2 + (self.y - other_point.y) ** 2)

class Cylinder:
    def __init__(self, radius, thickness, separation):
        self.radius = radius
        self.thickness = thickness
        self.separation = separation
        self.bottom_center = Coordinate(2 * radius + 1.5 * thickness, radius + thickness)
        self.top_center = Coordinate(radius + thickness, separation + radius + thickness)
        self.points = []
        self.labels = []

    def create_points(self, count):
        self.points.clear()
        self.labels.clear()

        max_x = (self.radius + self.thickness) * 3 - self.thickness / 2
        max_y = (self.radius + self.thickness) * 2 + self.separation

        lower_y_max = self.bottom_center.y
        upper_y_min = self.radius + self.thickness + self.separation
        upper_y_max = upper_y_min + self.radius + self.thickness

        while len(self.points) < count:
            x = np.random.uniform(0, max_x)
            y = np.random.uniform(0, max_y)

            new_point = Coordinate(x, y)

            if self.thickness * 0.5 + self.radius <= x <= max_x and 0 <= y <= lower_y_max:
                if self.radius <= new_point.dist(self.bottom_center) <= self.radius + self.thickness:
                    self.points.append(new_point)
                    self.labels.append(-1)
            elif 0 <= x <= (self.thickness + self.radius) * 2 and upper_y_min <= y <= upper_y_max:
                if self.radius <= new_point.dist(self.top_center) <= self.radius + self.thickness:
                    self.points.append(new_point)
                    self.labels.append(1)

    def plot_points(self, ax):
        x_vals = [point.x for point in self.points]
        y_vals = [point.y for point in self.points]
        colors = ['blue' if label == 1 else 'red' for label in self.labels]
        ax.scatter(x_vals, y_vals, c=colors, alpha=0.6)

def perceptron_learning(points, labels):
    weights = np.zeros(3)
    iteration_count = 0
    converged = False

    while not converged:
        converged = True
        for i in range(len(labels)):
            point = np.array([1, points[i].x, points[i].y])
            if np.dot(weights, point) * labels[i] <= 0:
                weights += labels[i] * point
                iteration_count += 1
                converged = False

    return weights, iteration_count


def linear_regression(points, labels):
    X = np.array([[1, point.x, point.y] for point in points])
    y = np.array(labels)
    w_lin = np.linalg.inv(X.T @ X) @ X.T @ y
    return w_lin


def run_experiment(radius=10.0, thickness=5.0, separation=5.0, n_points=2000,is_plot=1):
    def calc_y(x_val, weight):
        return (-weight[0] - weight[1] * x_val) / weight[2]

    if is_plot==1:
        fig, ax = plt.subplots()
    disk = Cylinder(radius, thickness, separation)
    disk.create_points(n_points)
    if is_plot==1:
        disk.plot_points(ax)

    pla_weights, iterations = perceptron_learning(disk.points, disk.labels)
    reg_weights = linear_regression(disk.points, disk.labels)

    x_range = np.array([0, 50])
    if is_plot==1:
        ax.plot(x_range, [calc_y(x, pla_weights) for x in x_range], color="black", linestyle='-', linewidth=1.5, label="Perceptron Learning")
        ax.plot(x_range, [calc_y(x, reg_weights) for x in x_range], color="green", linestyle='--', linewidth=1.5, label="Linear Regression")

        return fig,ax,pla_weights, reg_weights, disk
    return 0,0,pla_weights, reg_weights, disk

sep_values = np.arange(0.2, 5.2, 0.2)
iterations_list = []

for sep in sep_values:
    fig, ax, _, _, _disk = run_experiment(separation=sep,is_plot=0)
    #title="Perceptron vs Linear Regression for Separation = "+str(sep)
    #ax.set_title(title, fontsize=12)
    #ax.legend()
    #fig.tight_layout()
    #plt.show()
    _, iterations = perceptron_learning(_disk.points, _disk.labels)
    iterations_list.append(iterations)

fig, ax = plt.subplots()
ax.plot(sep_values, iterations_list, linestyle='-.', marker='s', color='purple')
ax.set_title("Separation vs PLA Iterations", fontsize=14)
ax.set_xlabel("Separation", fontsize=12)
ax.set_ylabel("Iterations", fontsize=12)
plt.savefig("sep_iter.png")
plt.show()