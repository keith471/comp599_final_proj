class DataPartitioner:
    '''
    This class partitions a dataset X, y into n parts of equal size
    partitions, the key datastructure of this class, is a list of two-element
    lists: [X instances, y instances]
    '''

    def __init__(self, n, X, y):
        X = X.toarray()
        self.n = n
        self.partitions = self.partition(n, X, y)

    def partition(self, n, X, y):
        chunks = []
        for i in range(len(X)):
            partitionIndex = i % n
            if partitionIndex >= len(chunks):
                chunks.append([[],[]])
            chunks[partitionIndex][0].append(X[i])
            chunks[partitionIndex][1].append(y[i])
        return chunks

    def getIthPartition(self, i):
        ''' returns the ith partition as the validation set and the remaining
        partitions as the training set '''
        validationSet = self.partitions[i]
        trainingSet = [[],[]]
        for index, partition in enumerate(self.partitions):
            if index != i:
                trainingSet[0].extend(partition[0])
                trainingSet[1].extend(partition[1])
        return trainingSet, validationSet

    def getPartitions(self):
        '''Returns all partitions in the following format:
        [(X1_train, y1_train, X1_test, y1_test),...for all n paritions]
        '''
        partitions = []
        for i in range(self.n):
            training, validation = self.getIthPartition(i)
            partitions.append((training[0], training[1], validation[0], validation[1]))
        return partitions
