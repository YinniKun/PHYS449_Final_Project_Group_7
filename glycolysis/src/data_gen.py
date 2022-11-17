## Data for the glycolysis model
import numpy as np
class testdata():
    def __init__(self, n_training_data, n_test_data):
        '''Data generation
        x_* attributes contain random bit strings of size n_bits
        y_* attributes are the parity of each bit string'''

        data_inputs = np.random.rand(2, n_training_data)
        data_labels = np.random.rand(2, n_training_data)
        aux_inputs = np.random.rand(6, 2)
        aux_labels = np.random.rand(6, 2)
        data_inputs= np.array(data_inputs, dtype= np.float32)
        data_labels= np.array(data_labels, dtype= np.float32)
        aux_inputs = np.array(aux_inputs, dtype= np.float32)
        aux_labels = np.array(aux_labels, dtype= np.float32)

        x_test= np.random.rand(2, 3)
        y_test= np.random.rand(1, 2)
        x_test= np.array(x_test, dtype= np.float32)
        y_test= np.array(y_test, dtype= np.float32)

        self.data_inputs = data_inputs
        self.data_labels = data_labels
        self.aux_inputs = aux_inputs
        self.aux_labels = aux_labels
        self.x_test= x_test
        self.y_test= y_test