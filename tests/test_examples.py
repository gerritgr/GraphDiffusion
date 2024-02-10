from graphdiffusion.pipeline import *

import pytest
from torch import nn


def test_example1():
    import traceback

    # Declare the variable as global to modify it
    import os

    os.system("cp -r examples examples_temp_for_testing")
    global TEST_MODUS_WITH_REDUCED_TRAINING
    if "TEST_MODUS_WITH_REDUCED_TRAINING" not in globals():
        TEST_MODUS_WITH_REDUCED_TRAINING = False
        TEST_MODUS_WITH_REDUCED_TRAINING_old = False
    else:
        TEST_MODUS_WITH_REDUCED_TRAINING_old = TEST_MODUS_WITH_REDUCED_TRAINING

    TEST_MODUS_WITH_REDUCED_TRAINING = True
    try:
        import examples_temp_for_testing.example1_components
    except Exception as e:  # It's a good practice to catch specific exceptions
        print(f"An error occurred: {e}")
        traceback.print_exc()
        assert False, "Failed to import examples_t"
    import examples_temp_for_testing.example1_components

    # import examples_temp_for_testing.example2_spiral
    TEST_MODUS_WITH_REDUCED_TRAINING = TEST_MODUS_WITH_REDUCED_TRAINING_old

    os.system("rm -r examples_temp_for_testing")
