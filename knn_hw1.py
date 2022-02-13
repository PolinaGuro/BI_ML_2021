import numpy as np


class KNNClassifier:
    """
    K-neariest-neighbor classifier using L1 loss
    """
    
    def __init__(self, k=1):
        self.k = k
    

    def fit(self, X, y):
        self.train_X = X
        self.train_y = y


    def predict(self, testData, n_loops=1):
        
        if n_loops == 0:
            distances = self.compute_distances_no_loops(testData)
        elif n_loops == 1:
            distances = self.compute_distances_one_loop(testData)
        else:
            distances = self.compute_distances_two_loops(testData)
        
        if len(np.unique(self.train_y)) == 2:
          return self.predict_labels_binary(distances)
        else:
          return self.predict_labels_multiclass(distances)


    def compute_distances_two_loops(self, X_test):
        
        num_test = len(X_test)
        num_train = len(self.train_X)
        dists = np.zeros((num_test, num_train))
        for i in range(num_test):
          for j in range(num_train):
            dists[i, j] = np.sqrt(np.sum(np.square(self.train_X[j,:] - X_test[i,:]))) # this!
        return dists
        
        pass


    def compute_distances_one_loop(self, X_test):
        
        num_test = len(X_test)
        num_train = len(self.train_X)
        dists = np.zeros((num_test, num_train))
        for i in range(num_test):
          dists[i] = np.sqrt(np.sum(np.square(self.train_X - X_test[i,:]), axis = 1)) # this
        return dists
        pass


    def compute_distances_no_loops(self, X_test):
        

        num_test = len(X_test)
        num_train = len(self.train_X)
        dists = np.zeros((num_test, num_train))
        dists = np.sqrt((X_test**2).sum(axis=1)[:, np.newaxis]
            + (self.train_X**2).sum(axis=1)
            - 2 * X_test.dot(self.train_X.T))
        return dists
        pass


    def predict_labels_binary(self, distances):
        

        n_train = distances.shape[1]
        n_test = distances.shape[0]
        prediction = np.zeros(n_test)

        numberOfClasses = 2
        #prediction = np.zeros(np.size(testData, 0))
        for i in range(n_test):
          arr_ind = np.argsort(distances[i, :])
          labels = self.train_y[arr_ind[0:self.k]]
          #вектор из количеств каждого класса в labels
          label_sum = np.zeros(numberOfClasses)
          for j in range(numberOfClasses):
          # для каждого класса делаем вектор bool а потом суммируем как int
            label_sum[j] = sum(labels == j)
          #индекс максимального элемента label_sum
          prediction[i] = np.argmax(label_sum)
        return prediction
        pass


    def predict_labels_multiclass(self, distances):
        

        n_train = distances.shape[0]
        n_test = distances.shape[0]
        prediction = np.zeros(n_test, np.int)

        # ддлина вектора уникальных значиений y_train (меток класса)
        numberOfClasses = (np.unique(self.train_y)).shape[0]
        #prediction = np.zeros(np.size(testData, 0))
        for i in range(n_test):
          arr_ind = np.argsort(distances[i, :])
          labels = self.train_y[arr_ind[0:self.k]]
          #вектор из количеств каждого класса в labels
          label_sum = np.zeros(numberOfClasses)
          for j in range(numberOfClasses):
          # для каждого класса делаем вектор bool а потом суммируем как int
            label_sum[j] = sum(labels == j)
          #индекс максимального элемента label_sum
          prediction[i] = np.argmax(label_sum)
        return prediction
        pass
