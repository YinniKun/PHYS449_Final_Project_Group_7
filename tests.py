import pytest
from glycolysis.src.nn_gen import calculate_weights as cw
import numpy as np

class CWInput:
    test_input1 = np.asarray([20, 3000, 33124, 2.3, 85])
    expected_output1 = np.asarray([10, 100, 10000, 1, 10])

    test_input2 = np.asarray([100])
    expected_output2 = 1


@pytest.mark.parametrize("test_input, expected_out", [(CWInput.test_input1, CWInput.expected_output1),
                                                      (CWInput.test_input2, CWInput.expected_output2)])
def test_calculate_weights(test_input, expected_out):
    test_output = cw(test_input)
    if isinstance(test_output, np.ndarray):
        assert test_output.all() == expected_out.all()
    else:
        assert test_output == expected_out