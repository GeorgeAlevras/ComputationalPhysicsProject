import argparse
import numpy as np
import warnings
from main import extended_trapezoidal_rule, extended_trapezoidal_rule_3d, extended_simpsons_rule, \
    extended_simpsons_rule_3d, mc_integration
from functions import f1, f2, f3, f4, f4_mc, f5, f6, ground_state_1d, ground_state_3d

"""
Georgios Alevras - 20/12/2020
-----------------------------
Python Version used: 3.8.2
Numpy Version used: 1.19.1

Additional Dependencies: argparse, warning
------------------------------------------

        This file performs functionality-testing, unit-testing and various validations on the code to ensure that all
    functions work correctly and handle errors well.

        It contains the 9 following functions:
            1. test_functions: Tests methods for different functions, to ensure methods work for all types of functions,
        e.g. functions with singularities, repeated estimates of 0, 3D non-separable functions, etc.
            2. test_validation_check_1: Validate for existence and logical value of relative accuracy, a = 2
            3. test_validation_check_2: Validate for existence and logical value of relative accuracy, a = -0.1
            4. test_validation_check_3: Validate for existence and logical value of relative accuracy, a = 'hello
            5. test_validation_check_4: Validating for limits not matching dimensions of function, case 1
            6. test_validation_check_5: Validating for limits not matching dimensions of function, case 2
            7. test_validation_check_6: Validating for limits being logical, e.g. numbers and not letters, etc.
            8. test_validation_check_7: Validating for number of function parameters matching integrator method
            9. use_args: Reads all the parsed arguments provided by the reader in command-line and executes accordingly
        the integrators
            
"""


def test_functions():
    """Testing all methods for functions with integrands that can be undefined, f1 = sin(x)/x"""
    warnings.filterwarnings("ignore")
    print("Testing all methods for functions with integrands that can be undefined, f1 = sin(x)/x")
    int_prev_t_1, int_next_t_1, iterations_t_1, total_samples_t_1 = extended_trapezoidal_rule(f1, [0, 1], 1e-6)
    print("\tTrue value: 0.946083, Obtained with Trapezoidal: ", int_next_t_1)
    int_prev_s_1, int_next_s_1, iterations_s_1, total_samples_s_1 = extended_simpsons_rule(f1, [0, 1], 1e-6)
    print("\tTrue value: 0.946083, Obtained with Simpson's: ", int_next_s_1)
    integral_1, st_error_1, n_m_1 = mc_integration(f1, np.array([[0, 1]]), 1e-3, sampling='flat')
    print("\tTrue value: 0.946083, Obtained with MC (flat): ", integral_1)
    integral_i_1, st_error_i_1, n_m_i_1 = mc_integration(f1, np.array([[0, 1]]), 1e-3, sampling='importance')
    print("\tTrue value: 0.946083, Obtained with MC (importance): ", integral_i_1, "\n\n")
    assert int_next_t_1 >= 0
    assert integral_1 >= 0
    assert integral_i_1 >= 0

    """Testing all methods for 0 function, f2 = 0"""
    print("Testing all methods for 0 function, f2 = 0")
    int_prev_t_2, int_next_t_2, iterations_t_2, total_samples_t_2 = extended_trapezoidal_rule(f2, [0, 1], 1e-6)
    print("\tTrue value: 0, Obtained with Trapezoidal: ", int_next_t_2)
    int_prev_s_2, int_next_s_2, iterations_s_2, total_samples_s_2 = extended_simpsons_rule(f2, [0, 1], 1e-6)
    print("\tTrue value: 0, Obtained with Simpson's: ", int_next_s_2)
    integral_2, st_error_2, n_m_2 = mc_integration(f2, np.array([[0, 1]]), 1e-3, sampling='flat')
    print("\tTrue value: 0, Obtained with MC (flat): ", integral_2)
    integral_i_2, st_error_i_2, n_m_i_2 = mc_integration(f2, np.array([[0, 1]]), 1e-3, sampling='importance')
    print("\tTrue value: 0, Obtained with MC (importance): ", integral_i_2, "\n\n")
    assert int_next_t_2 == 0
    assert int_next_s_2 == 0
    assert integral_2 == 0
    assert integral_i_2 == 0

    """Testing Newton-Cotes methods return exact values for linear functions, f3 = x"""
    print("Testing Newton-Cotes methods return exact values for linear functions, f3 = x")
    int_prev_t_3, int_next_t_3, iterations_t_3, total_samples_t_3 = extended_trapezoidal_rule(f3, [0, 2], 1e-6)
    print("\tTrue value: 2.0, Obtained with Trapezoidal: ", int_next_t_3)
    int_prev_s_3, int_next_s_3, iterations_s_3, total_samples_s_3 = extended_simpsons_rule(f3, [0, 2], 1e-6)
    print("\tTrue value: 2.0, Obtained with Simpson's: ", int_next_s_3, "\n\n")
    assert int_next_t_3 == 2
    assert int_next_s_3 == 2

    """Testing all methods for 3D functions that are not separable, f4 = sin(xy) * e^(yz)"""
    limits = np.array([0.0, 1.0, 0.0, 1.0, 0.0, 1.0])
    limits = np.reshape(limits, (3, 2))
    print("Testing all methods for 3D functions that are not separable, f4 = sin(xy) * e^(yz)")
    int_prev_t_4, int_next_t_4, iterations_t_4, total_samples_t_4 = extended_trapezoidal_rule_3d(f4, limits, 1e-4)
    print("\tTrue value: 0.343464, Obtained with Trapezoidal: ", int_next_t_4)
    int_prev_s_4, int_next_s_4, iterations_s_4, total_samples_s_4 = extended_simpsons_rule_3d(f4, limits, 1e-4)
    print("\tTrue value: 0.343464, Obtained with Simpson's: ", int_next_s_4)
    integral_4, st_error_4, n_m_4 = mc_integration(f4_mc, limits, 1e-3, sampling='flat')
    print("\tTrue value: 0.343464, Obtained with MC (flat): ", integral_4)
    integral_i_4, st_error_i_4, n_m_i_4 = mc_integration(f4_mc, limits, 1e-3, sampling='importance')
    print("\tTrue value: 0.343464, Obtained with MC (importance): ", integral_i_4,  "\n\n")
    assert 0.34 < int_next_t_4 < 0.3436  # True value is 0.343464
    assert 0.34 < int_next_s_4 < 0.3436
    assert 0.32 < integral_4 < 0.36
    assert 0.32 < integral_i_4 < 0.36

    """Testing Simpson's rule returns exact values for quadratic and cubic functions, f5 = x**2 and f6 = x**3"""
    print("Testing Simpson's rule returns exact values for quadratic and cubic functions, f5 = x**2 and f6 = x**3")
    int_prev_s_5, int_next_s_5, iterations_s_5, total_samples_s_5 = extended_simpsons_rule(f5, [0, 1], 1e-2)
    print("\tTrue value: 1/3, Obtained with Simpson's: ", int_next_s_5)
    int_prev_s_6, int_next_s_6, iterations_s_6, total_samples_s_6 = extended_simpsons_rule(f6, [0, 1], 1e-2)
    print("\tTrue value: 1/4, Obtained with Simpson's: ", int_next_s_6, "\n\n")
    assert int_next_s_5 == float(1/3)

    print('\n\n*************************\nAll Function Tests Passed\n*************************\n\n')


def test_validation_check_1():
    print('\n   **********************************************************************************\n'
          '   Validate for existence and logical value of relative accuracy, e.g. test for a = 2\n'
          '   **********************************************************************************\n\n')
    print(extended_trapezoidal_rule(f1, [0, 1], 2))  # e.g. test for a = 2


def test_validation_check_2():
    print('\n   *************************************************************************************\n'
          '   Validate for existence and logical value of relative accuracy, e.g. test for a = -0.1\n'
          '   *************************************************************************************\n\n')
    print(extended_trapezoidal_rule(f1, [0, 1], -0.1))  # e.g. test for a = -0.1


def test_validation_check_3():
    print("\n   ***************************************************************************************\n"
          "   Validate for existence and logical value of relative accuracy, e.g. test for a = 'hello\n"
          "   ***************************************************************************************\n\n")
    print(extended_trapezoidal_rule(f1, [0, 1], 'hello'))  # e.g. test for a = 'hello'


def test_validation_check_4():
    print('\n   *****************************************************************************************\n'
          '   Validating for limits not matching dimensions of function, e.g. 1D limits for 3D function\n'
          '   *****************************************************************************************\n\n')
    print(extended_trapezoidal_rule_3d(ground_state_3d, [0, 2], 1e-4))  # 1D limits for 3D function


def test_validation_check_5():
    print('\n   ****************************************************************************************\n'
          '   Validating for limits not matching dimensions of function, e.g. 3 bounds for 1D function\n'
          '   ****************************************************************************************\n\n')
    print(extended_trapezoidal_rule(f1, [0, 1, 3], 2))  # 3 bounds for 1D function


def test_validation_check_6():
    print('\n   *************************************************\n'
          '   Validating for limits being logical, e.g. numbers\n'
          '   *************************************************\n\n')
    print(extended_trapezoidal_rule(ground_state_1d, ['hi', 2], 1e-4))  # limits include bound 'hi'


def test_validation_check_7():
    f = lambda x, y, z: eval(np.sin(x*y) + np.cos(y*z))
    print('\n   ***********************************************************************\n'
          '   Validating for number of function parameters matching integrator method\n'
          '   ***********************************************************************\n\n')
    print(extended_trapezoidal_rule(f, [0, 2], 1e-4))  # 3D function in 1D integrator


def use_args(args):
    if args.test_number == 0:
        test_functions()
    elif args.test_number == 1:
        test_validation_check_1()
    elif args.test_number == 2:
        test_validation_check_2()
    elif args.test_number == 3:
        test_validation_check_3()
    elif args.test_number == 4:
        test_validation_check_4()
    elif args.test_number == 5:
        test_validation_check_5()
    elif args.test_number == 6:
        test_validation_check_6()
    elif args.test_number == 7:
        test_validation_check_7()
    else:
        raise ValueError("Not a valid test, please enter a digit between 0 and 7, e.g. -t 2")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Georgios Alevras: Computational Physics Project - Testing Help',
                                     epilog='Enjoy the script :)')
    parser.add_argument('-t', '--test_number', type=float, help='Specify Test Number to perform')
    arguments = parser.parse_args()  # Parses all arguments provided at script on command-line
    use_args(arguments)  # Executes code according to arguments provided
