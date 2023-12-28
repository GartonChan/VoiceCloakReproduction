import numpy as np
def data_loader(data_set, batchSize = 64):
    data_size = data_set.shape[0]
    if  batchSize > data_size:
        batchSize = data_size
    return data_set[np.random.choice(data_set.shape[0], batchSize)]

# print(np.random.choice(100, 10))
# print(np.random.choice(3, [0,1,2,3,4,5,6,7,8,9]))