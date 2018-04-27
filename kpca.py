import matplotlib.pyplot as plt
import numpy as np
from sklearn import datasets


class Kpca:

    def calculate_kernel_matrix(self, input):
        m = len(input)
        kernel_matrix = np.zeros(m**2).reshape(m, m)
        for rows in range(m):
            for cols in range(m):
                kernel_matrix[rows][cols] = np.matmul(input[rows].T, input[cols])
                # A = input[rows] - input[cols]
                # kernel_matrix[rows][cols] = np.exp(- 0.7 * np.matmul(A.T, A))
                # kernel_matrix[rows][cols] = np.sqrt(np.matmul(A.T, A) + 1000)
        print(kernel_matrix)
        return kernel_matrix

    def reducing_dim(self, input, dim):
        m = len(input)
        kernel_matrix = self.calculate_kernel_matrix(input)
        # print(kernel_matrix)
        B = np.eye(m) - 1 / m * np.ones(m)
        kernel_matrix = np.matmul(np.matmul(B, kernel_matrix), B)
        print(kernel_matrix)
        eigenvalue, eigenvector = np.linalg.eig(kernel_matrix)
        index = eigenvalue.argsort()[::-1]
        eigenvector = eigenvector[:, index]
        # eigenvector = eigenvector[:, :dim]
        for i in range(len(eigenvector)):
            eigenvector[i] = np.sqrt(1 / np.abs(eigenvalue[i])) * eigenvector[i]
        return np.dot(eigenvector.T, kernel_matrix)

def main():
    data = datasets.load_digits()
    x = data.data
    y = data.target
    transform_data = Kpca().reducing_dim(x, 2)
    print(transform_data)
    x1 = transform_data[0]
    x2 = transform_data[1]
    # print(transform_data)
    cmap = plt.get_cmap()
    color_list = [cmap(i) for i in np.linspace(0, 1, len(np.unique(y)))]
    color_list = ['pink', 'green', 'black', 'blue', 'yellow', 'purple', 'orange', 'teal', 'wheat', 'red']
    class_data = {}
    class_data['class_name'] = []
    class_data['legend_name'] = []
    for i, l in enumerate(np.unique(y)):
        _x1 = x1[y == l]
        _x2 = x2[y == l]
        class_data['class_name'].append(plt.scatter(_x1, _x2, color=color_list[i]))
        class_data['legend_name'].append(l)

    plt.legend(class_data['class_name'], class_data['legend_name'], loc=1)
    plt.suptitle("PCA")
    plt.title("Digit")
    plt.xlabel("principal 1")
    plt.ylabel("principal 2")
    plt.show()


if __name__ == '__main__':
    main()