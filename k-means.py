# Developed by Bayram Baris Sari
# E-mail: bayrambariss@gmail.com
# Tel No: +90 539 593 7501    

import numpy as np
import matplotlib.pyplot as plt

# these two are for built-in functions for testing
# from sklearn.decomposition import PCA
# from sklearn.cluster import KMeans


def PCA(dataset):
    # Calculate covarience matrix
    cov = np.cov(dataset, rowvar=False)
    # print(cov)

    # Find eigenvalues and eigenvectors of the covariance matrix
    eigenvalue, eigenvector = np.linalg.eig(cov)
    # print(eigenvalue)
    # print(eigenvector)

    # descending sort of eigenvectors, according to their eigenvalues
    index = eigenvalue.argsort()[::-1]
    eigenvector = eigenvector[:, index]

    # merge 2 biggest eigenvector matrices horizontally
    matrix_w = np.hstack((eigenvector[:, 0].reshape(65, 1), eigenvector[:, 1].reshape(65, 1)))
    transformed = matrix_w.T.dot(dataset.T)
    # print(matrix_w)
    # print(transformed)

    return transformed


def k_means(data, k, sse):
    b = np.zeros((data.shape[0], k))    # 1 if the distance between points and the center is the minimum distance
    C = np.zeros((k, 2))                # center points
    # for k initial center points, I selected close to the means of the x and y values
    mean_x = (max(data[:, 0]) - min(data[:, 0])) / k + 1
    mean_y = (max(data[:, 1]) - min(data[:, 1])) / k + 1
    for i in range(k):
        rnd_x = np.random.random()
        rnd_y = np.random.random()
        C[i][0] = mean_x*(i+1) + rnd_x
        C[i][1] = mean_y*(i+1) + rnd_y

    iteration = 1
    sum_of_squared_error = 0
    while True:
        prev_sum_of_squared_error = sum_of_squared_error
        sum_of_squared_error = 0
        # for whole data points
        for index, x in enumerate(data):
            # at the beginning, minimum is the distance between x-first_centroid
            minimum = np.sqrt((x[0] - C[0][0]) ** 2 + (x[1] - C[0][1]) ** 2)
            b[index][0] = 1
            temp_index = 0
            # find the minimum distance between x and other centroids
            for idx, j in enumerate(C):
                temp = np.sqrt((x[0]-j[0])**2+(x[1]-j[1])**2)
                # if the distance between x and "this centroid", j, is smaller, min = this distance
                if temp < minimum:
                    minimum = temp
                    b[index][temp_index] = 0
                    b[index][idx] = 1
                    temp_index = idx
            # minimum distance is found, add it to sum of squared errors
            sum_of_squared_error += minimum
        # update centroids
        for index, value in enumerate(C):
            C[index][0] = np.sum(b[:, index]*data[:, 0]) / np.sum(b[:, index])
            C[index][1] = np.sum(b[:, index]*data[:, 1]) / np.sum(b[:, index])

        # if SSE changed lower than the threshold, break the loop
        if np.abs(prev_sum_of_squared_error-sum_of_squared_error) < 0.000000000000001:
            sse.append(sum_of_squared_error)
            print("Centroids converged after %d iterations for K = %d!" % (iteration, k))
            break
        iteration += 1

    return C, b


def plot_k_means(new_data, k_list, sse):
    colors = ['k', 'g', 'b', 'm', 'gold', 'r', 'dimgrey', 'peru', 'limegreen', 'teal', 'darkblue',
         'crimson', 'c', 'purple', 'dimgray', 'aqua', 'firebrick', 'olive', 'midnightblue', 'maroon']
    for k in k_list:
        means, clusters = k_means(new_data, k, sse)
        for i, data in enumerate(new_data):
            for j in range(k):
                if clusters[i][j] == 1:
                    plt.plot(data[0], data[1], 'o', markersize=1, color=colors[j])

        plt.title("K-means of PCA applied data, K= %d" % k)
        plt.ylabel('Second Eigenvector')
        plt.xlabel('First Eigenvector')
        plt.plot(means[:, 0], means[:, 1], '*', markersize=7, color='k')
        plt.show()

def main():
    # parse the data
    x = []
    file = open("Cluster.csv", 'r')
    file.readline()  # skip the first line
    data = [line.strip('\n').split(',') for line in file.readlines()]
    for row in data:
            x.append(row)
    file.close()

    # reduce its dimension to 2
    x = np.array(x, dtype=int)
    new_data = PCA(x)
    new_data = new_data.T

    k_list = [1, 5, 10, 20]

    # this plot function calculates k-means for K=1,5,10,20, and plots them
    sse = []
    plot_k_means(new_data, k_list, sse)
    print("Sum of squared errors")
    print("--------------------------------")
    for i, value in enumerate(sse):
        print("For k = %d,\t %.3f" % (k_list[i], value))


    ''' I used this built-in functions for comparing my results with the results of built-in functions
    pca = PCA(n_components=2)
    new_data = pca.fit_transform(x)
    print(new_data)
    kmeans = KMeans(n_clusters=k)   # k = 1,5,10,20
    kmeans = kmeans.fit(new_data)
    labels = kmeans.predict(new_data)
    centers = kmeans.cluster_centers_
    '''

if __name__ == "__main__":
    main()