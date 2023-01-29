import argparse
import numpy as np
import math
from inspect import signature
from functions import ground_state_1d, ground_state_3d, ground_state_3d_mc, excited_state_3d, excited_state_3d_mc
import matplotlib.pyplot as plt

"""
Georgios Alevras - 20/12/2020
-----------------------------
Python Version used: 3.8.2
Numpy Version used: 1.19.1

Additional Dependencies: argparse, inspect
------------------------------------------

        This is the main file, working as a script, enabling a user to use the integrators and solve problems with the 
    Trapezoidal Rule, Simpson's Rule, as well as MC integration using uniform and importance sampling until a certain
    convergence criterion set by the user is met.
        
        It contains 8 functions:
            1. extended_trapezoidal_rule: Performs the extended trapezoidal rule to integrate a function in 1D
            2. extended_trapezoidal_rule_3d: Performs the extended trapezoidal rule above (extended_trapezoidal_rule)
        within 3 nested recursive calls to integrate a function in 3D
            3. extended_simpsons_rule: Performs the extended Simpson's rule to integrate a function in 1D by calling the
        extended trapezoidal rule above (extended_trapezoidal_rule) using its two adjacent iterations
            4. extended_simpsons_rule_3d: Performs the extended Simpson's rule to integrate a function in 3D by calling 
        the extended trapezoidal rule above (extended_trapezoidal_rule_3d) using its two adjacent iterations
            5. pdf: A linear PDF, returns the probability density for a random variable x
            6. pdf_inv: An inverse CDF, used to transform a uniform variate to a determined non-uniform deviate
            7. mc_integration: Performs Monte-Carlo integration to integrate a function in any dimension
            8. use_args: Reads all the parsed arguments provided by the reader in command-line and executes accordingly
        the integrators
"""


def extended_trapezoidal_rule(func, limits, rel_accuracy, enable_simpsons=False):
    """
    :param func: The integrand of the integral, type: <function func>
    :param limits: An array with the limits of integration, type: <class 'numpy.ndarray'>
    :param rel_accuracy: The relative accuracy - convergence criterion - of integration, type: <class 'float'>
    :param enable_simpsons: A flag determining whether to use Simpson's rule instead of the Trapezoidal rule
    :return: second-to-last estimation of integral, last estimation of integral, #iterations, # total samples used
    """
    if len(signature(func).parameters) != 1:  # Validation that function provided is 1D
        raise Exception("\nFunction provided is not 1-dimensional")
    if len(limits) != 2:  # Validation that limits provided are correct - 1D
        raise ValueError("\nDidn't provide correct limit dimensions. Must be ndarray of shape(2,)")
    if 1 < rel_accuracy or rel_accuracy < 0 or not isinstance(rel_accuracy, float):  # Value validation of criterion
        raise ValueError("\nPlease provide a logical convergence criterion, i.e. 0 < relative_accuracy < 1")
    a, b = limits  # Lower and upper bounds of integration
    if not isinstance(a, (int, float)) and not isinstance(b, (int, float)):  # Value validation that limits are numbers
        raise ValueError("\nPlease provide real numbers for the limits")

    h = float(b-a)  # Step size, initially defined as difference between bounds
    # First estimation using 1 trapezium, ensuring it avoids infinities and infinite loops caused by estimations of 0
    int_0 = 0.5 * func(a) + 0.5 * func(b) if not math.isnan(func(a)) and not math.isnan(func(b)) and func(a) != \
        float('inf') and func(b) != float('inf') else 0.5*func(a+0.1*h) + 0.5*func(b-0.1*h)
    mid_point = func(0.5*(a + b))
    h *= 0.5
    # Second estimation using 2 trapeziums, ensuring it avoids infinities and infinite loops caused by estimations of 0
    int_prev = h * (0.5*func(a) + mid_point + 0.5*func(b)) if not math.isnan(func(a)) and not math.isnan(func(b)) and \
        func(a) != float('inf') and func(b) != float('inf') else h * (0.5*func(a+0.1*h) + mid_point + 0.5*func(b-0.1*h))

    iterations = 2  # Index of iterations
    total_samples = 3  # Total number of samples (points) used for each estimation
    obtained_rel_accuracy = abs((int_prev - int_0) / int_0) if int_0 != 0 else 0.9

    if obtained_rel_accuracy < rel_accuracy:  # Return if the convergence criterion is met
        return int_0, int_prev, iterations, total_samples

    while obtained_rel_accuracy > rel_accuracy:  # Loop until convergence criterion is met
        new_samples = [x for x in range(1, 2**(iterations-1)+1)]  # New points to perform next estimation
        int_next = 0.5*int_prev + 0.5*h*sum(func((a+h*(2*x-1)/2)) for x in new_samples)  # Integrand  at new points
        # Update relative accuracy and validate against DivideByZero error
        obtained_rel_accuracy = abs((int_next - int_prev) / int_prev) if int_prev != 0 else 0.9

        if enable_simpsons:  # If performing Simpson's rule, use the relation with the Trapezoidal rule
            if iterations == 2:
                # First estimation, ensuring it avoids infinities and infinite loops caused by estimations of 0
                int_prev_s = h * ((1/3)*func(a) + (4/3)*mid_point + (1/3) * func(b)) if not math.isnan(func(a)) and not\
                    math.isnan(func(b)) and func(a) != float('inf') and func(b) != float('inf') else \
                    h * ((1/3)*func(a+0.1*h) + (4/3)*mid_point + (1/3) * func(b+0.1*h))  # First Simpson's estimation
                int_next_s = (4/3)*int_next - (1/3)*int_prev  # Second Simpson's rule estimation
                # Update relative accuracy and validate against DivideByZero error
                obtained_rel_accuracy = abs((int_next_s - int_prev_s) / int_prev_s) if int_prev_s != 0 else 0.9
            else:
                int_prev_s = (4/3)*int_prev - (1/3)*int_0  # Update previous Simpson's rule estimation
                int_next_s = (4/3)*int_next - (1/3)*int_prev  # Update next Simpson's rule estimation
                # Update relative accuracy and validate against DivideByZero error
                obtained_rel_accuracy = abs((int_next_s - int_prev_s) / int_prev_s) if int_prev_s != 0 else 0.9
        if int_prev == 0 and int_next == 0 and iterations > 3:
            return 0, 0, iterations, total_samples
        int_0 = int_prev  # Update initial estimation
        store_int_prev = int_prev  # Temporary variable to store and return previous estimation
        int_prev = int_next  # Update previous estimation
        total_samples += len(new_samples)  # Add new samples to total samples after each iteration
        h *= 0.5  # Halve step size for next estimation
        iterations += 1  # Increment iteration index

    if enable_simpsons:
        return int_prev_s, int_next_s, iterations, total_samples
    else:
        return store_int_prev, int_next, iterations, total_samples


def extended_trapezoidal_rule_3d(func, limits, rel_accuracy, enable_simpsons=False):
    """
    :param func: The integrand of the integral, type: <function func>
    :param limits: An array with the limits of integration, type: <class 'numpy.ndarray'>
    :param rel_accuracy: The relative accuracy - convergence criterion - of integration, type: <class 'float'>
    :param enable_simpsons: A flag determining whether to use Simpson's rule instead of the Trapezoidal rule
    :return: second-to-last estimation of integral, last estimation of integral, #iterations, # total samples used
    """
    if len(signature(func).parameters) != 3:   # Validation that function provided is 3D
        raise Exception("\nFunction provided is not 3-dimensional")
    if len(limits) != 3:   # Validation that limits provided are correct - 3D
        raise ValueError("\nDidn't provide correct limit dimensions. Must be ndarray of shape(3, 2)")
    if 1 < rel_accuracy or rel_accuracy < 0 or not isinstance(rel_accuracy, float):  # Value validation of criterion
        raise ValueError("\nPlease provide a logical convergence criterion, i.e. 0 < relative_accuracy < 1")
    if not isinstance(limits[0][0], (int, float)) and not isinstance(limits[0][1], (int, float)) and not \
            isinstance(limits[1][0], (int, float)) and not isinstance(limits[1][1], (int, float)) and not \
            isinstance(limits[2][0], (int, float)) and not isinstance(limits[2][1], (int, float)):
        raise ValueError("\nPlease provide real numbers for the limits")
    limits = np.reshape(limits, (3, 2))

    def outermost_integrand(x, y):  # Performs final integral - outermost of 3
        return extended_trapezoidal_rule(lambda z: func(x, y, z), limits[2], rel_accuracy, enable_simpsons)[1]

    def middle_integrand(x):  # Performs second integral - middle one of 3
        return extended_trapezoidal_rule(lambda y: outermost_integrand(x, y), limits[1], rel_accuracy, enable_simpsons)[1]

    # Performs first integral - innermost of 3
    int_prev, int_next, iterations, total_samples = extended_trapezoidal_rule(middle_integrand, limits[0], rel_accuracy,
                                                                              enable_simpsons)
    total_samples **= 3  # Total samples in each dimension raised to number of dimensions, 3 for 3D
    iterations **= 3  # Iterations in each dimension raised to number of dimensions, 3 for 3D
    return int_prev, int_next, iterations, total_samples


def extended_simpsons_rule(func, limits, rel_accuracy):
    """
    :param func: The integrand of the integral, type: <function func>
    :param limits: An array with the limits of integration, type: <class 'numpy.ndarray'>
    :param rel_accuracy: The relative accuracy - convergence criterion - of integration, type: <class 'float'>
    :return: second-to-last estimation of integral, last estimation of integral, #iterations, # total samples used
    """
    return extended_trapezoidal_rule(func, limits, rel_accuracy, enable_simpsons=True)


def extended_simpsons_rule_3d(func, limits, rel_accuracy):
    """
    :param func: The integrand of the integral, type: <function func>
    :param limits: An array with the limits of integration, type: <class 'numpy.ndarray'>
    :param rel_accuracy: The relative accuracy - convergence criterion - of integration, type: <class 'float'>
    :return: second-to-last estimation of integral, last estimation of integral, #iterations, # total samples used
    """
    return extended_trapezoidal_rule_3d(func, limits, rel_accuracy, enable_simpsons=True)


def pdf(x, a=-1, b=1.5):
    """
    :param x: A random variable
    :param a: coefficient of x (in linear equation for PDF)
    :param b: constant (in linear equation for PDF)
    :return: the probability density for the random variable x
    """
    return a * x + b


def pdf_inv(x, a=-1, b=1.5):
    """
    :param x: A random variable
    :param a: coefficient of x (in linear equation for PDF)
    :param b: constant (in linear equation for PDF)
    :return: the inverse cumulative probability density for the random variable x
    """
    return -(b / a) - np.sqrt((b / a) ** 2 + 2 * x / a)


def mc_integration(func, limits, rel_accuracy, sampling='flat', to_plot=False):
    """
    :param func: The integrand of the integral, type: <function func>
    :param limits: An array with the limits of integration, type: <class 'numpy.ndarray'>
    :param rel_accuracy: The relative accuracy - convergence criterion - of integration, type: <class 'float'>
    :param sampling: Choice of sampling method, uniform (flat) or importance sampling
    :param to_plot: Flag to determine whether to produce plots for checking and validation of method
    :return: second-to-last estimation of integral, last estimation of integral, #iterations, # total samples used
    """
    dimensions = len(limits)
    lower_limits = np.array([limits[:, 0]])
    upper_limits = np.array([limits[:, 1]])
    lower_limits = np.reshape(lower_limits, (dimensions, 1))  # Reshape in order to vectorise code
    upper_limits = np.reshape(upper_limits, (dimensions, 1))  # Reshape in order to vectorise code
    steps = upper_limits - lower_limits
    volume = np.product(steps)  # Volume spanned by steps in each dimension

    n = m = int(5e3)  # n is a variable, m will remain constant at 5e3
    for i in range(30):  # Will stop looping at 5,497,558,139,000,000,000 samples if convergence criterion not met
        counter = int(n/m)  # The number of times m fits into n
        n_m = counter*m  # Total number of samples used for an estimation
        store_sum = 0  # var will store the integral sum contribution from each loop iteration
        store_error = 0  # var will store the error sum contribution from each loop iteration
        for k in range(counter):
            probabilities = np.zeros((dimensions, m))
            for j in range(dimensions):  # Limit bounds of each random number generated to increase uniformity
                l = np.linspace(0, 1-1/m, m)
                u = np.linspace(0+1/m, 1, m)
                probs = np.random.uniform(l, u)  # Enforce smaller order uniformity to increase overall uniformity
                np.random.shuffle(probs)
                probabilities[j] = probs
            weight = 1  # Weight set to 1 if using uniform (flat) sampling
            if sampling != 'flat':  # If importance sampling
                probabilities = pdf_inv(probabilities)  # Obtain probabilities using the CDF of the wanted PDF
                weight = np.product(pdf(probabilities), axis=0)  # Adjust the weight for a given PDF
            samples = (lower_limits + steps*probabilities)  # Points in space of integral to evaluate function
            func_values = func(samples)/weight  # Weighted function evaluations at sample space
            store_sum += np.sum(func_values)  # Increment with sum of weighted function evaluations
            store_error += np.sum((func_values - np.full((dimensions, m), np.average(func_values)))**2)

        integral = (volume/n_m)*store_sum  # Once the loop is finished, calculate the integral and error
        st_error = (volume / np.sqrt(n_m)) * (1 / np.sqrt(n_m - 1)) * np.sqrt(store_error) / integral if integral != 0 \
            else (volume / np.sqrt(n_m)) * (1 / np.sqrt(n_m - 1)) * np.sqrt(store_error)
        print('\tMC Integration (' + str(sampling) + ') Integral:', integral, 'with', n, ' random samples. \tCurrent '
                                                                         'relative standard error:', round(st_error, 8))
        n *= 2  # Multiply n by 2 to estimate with twice as many points

        if st_error < rel_accuracy:
            if to_plot and sampling == 'flat':  # Plotting of probability histograms
                plt.figure(1)
                freq, p, _ = plt.hist(x=probabilities[0], bins=100, color='b', alpha=0.7, rwidth=0.9, label='Average: '
                                                                          + str(round(np.average(probabilities[0]), 5)))
                plt.title('Histogram of Probabilities (Uniform)', fontname='Times New Roman', fontsize=16)
                plt.xlabel('Probabilities', fontname='Times New Roman', fontsize=12)
                plt.ylabel('Frequency', fontname='Times New Roman', fontsize=12)
                plt.legend()
                plt.savefig('probabilities_flat_hist_1.png')
                plt.show()
            elif to_plot and sampling == 'importance':  # Plotting of probability histograms
                plt.figure(1)
                freq, p, _ = plt.hist(x=probabilities[0], bins=100, color='b', alpha=0.7, rwidth=0.9, label='Average: '
                                                                          + str(round(np.average(probabilities[0]), 5)))
                plt.title('Histogram of Probabilities (Linear Importance)', fontname='Times New Roman', fontsize=16)
                plt.xlabel('Probabilities', fontname='Times New Roman', fontsize=12)
                plt.ylabel('Frequency', fontname='Times New Roman', fontsize=12)
                plt.legend()
                plt.savefig('probabilities_linear_hist_1.png')
                plt.show()
            return integral, round(st_error, 8), n_m


def use_args(args):
    """
    :param args: arguments provided by user in command-line
    :return: None
    """
    # Output the help message of the script if no arguments are provided
    if args.user_fun is None and args.fun is None and args.accuracy is None and args.limits is None and args.dimensions\
            is None:
        parser.print_help()
        exit(1)
    elif args.limits is None:  # Validate that integration limits have been provided
        raise ValueError("\nPlease provide limits of integration")
    elif args.accuracy is None:  # Validate that a relative accuracy has been provided
        raise ValueError("\nPlease provide a relative accuracy")
    elif args.fun is None and args.user_fun is None:  # Validate either user-defined or code-specified function given
        raise ValueError("\nYou need to either provide a built-in function with '-u', or specify a function using '-f'")
    elif args.fun is not None and args.user_fun is not None:  # Validation same as above
        raise ValueError("\nYou must provide either-or of a built-in and a user-supplied function, not both")
    elif args.fun is None and args.user_fun is not None:
        if args.dimensions is None:  # In case of a user-defined function, its dimensions must also be provided
            raise ValueError("\nYou must also provide the dimensions of integration when writing a user-defined "
                             "function")
        elif args.dimensions is not None:
            if len(args.limits) != 2*args.dimensions:  # Validate that integration limits match dimensions
                raise ValueError("\nLimits must match integration dimensions")
            if args.dimensions == 1:
                f = lambda x: eval(args.user_fun)  # Obtain user-defined 1D function
                int_prev_t, int_next_t, iterations_t, total_samples_t = extended_trapezoidal_rule(f, args.limits,
                                                                                                  args.accuracy)
                print("Trapezoidal Rule:\t\t\t\t", str(int_next_t) + " Took ", str(iterations_t),
                      "iterations, and a total of",
                      str(total_samples_t), "samples")
                int_prev_s, int_next_s, iterations_s, total_samples_s = extended_simpsons_rule(f, args.limits,
                                                                                               args.accuracy)
                print("Simpson's Rule:\t\t\t\t\t", str(int_next_s) + " Took ", str(iterations_s),
                      "iterations, and a total of",
                      str(total_samples_s), "samples")
                limits_mc = np.array([args.limits])
                integral, st_error, n_m = mc_integration(f, limits_mc, args.accuracy)
                print("Monte Carlo Integration (Uniform Sampling):\t", str(integral) + " Took a total of", str(n_m),
                      "samples, with a standard error: ", str(st_error))
                integral_i, st_error_i, n_m_i = mc_integration(f, limits_mc, args.accuracy, sampling='importance')
                print("Monte Carlo Integration (Importance Sampling):\t", str(integral_i) + " Took a total of",
                      str(n_m_i), "samples, with a standard error: ", str(st_error_i))
            elif args.dimensions == 3:
                f = lambda x, y, z: eval(args.user_fun)  # Obtain user-defined 3D function
                limits = np.array(args.limits)
                if np.shape(limits)[0] != 6:  # Validate that integration limits match dimensions
                    raise ValueError("\nPlease provide correct limits of integration")
                limits = np.reshape(limits, (3, 2))
                nt_prev_t, int_next_t, iterations_t, total_samples_t = extended_trapezoidal_rule_3d(f, limits,
                                                                                                    args.accuracy)
                print("Trapezoidal Rule:\t\t\t\t", str(int_next_t) + " Took ", str(iterations_t),
                      "iterations, and a total of",
                      str(total_samples_t), "samples")
                int_prev_s, int_next_s, iterations_s, total_samples_s = extended_simpsons_rule_3d(f, limits,
                                                                                                  args.accuracy)
                print("Simpson's Rule:\t\t\t\t\t", str(int_next_s) + " Took ", str(iterations_s),
                      "iterations, and a total of", str(total_samples_s), "samples")
            else:  # Validate that dimensions of function are either 1D or 3D, to allow Newton-Cotes methods to work
                raise ValueError("\nNewton-Cotes has been coded for 1D and 3D functions only. Please provide '-d 1' or "
                                 "-d 3'")
    elif args.user_fun is None and args.fun is not None:
        # Dictionary holds functions, maps function names to the functions
        funcs = {'sho_gs_1d': ground_state_1d, 'sho_gs_3d': (ground_state_3d, ground_state_3d_mc), 'sho_es_3d':
                                                            (excited_state_3d, excited_state_3d_mc)}
        if args.fun not in funcs.keys():  # Validate function provided exists in source-code
            raise ValueError("\nFunction not recognised. Please enter one of the following, or your own using the -u "
                             "argument \n\tsho_gs_1d\n\tsho_gs_3d\n\tsho_es_3d")
        if args.fun == 'sho_gs_1d':
            user_func = funcs[args.fun]  # Obtain function from dictionary
            int_prev_t, int_next_t, iterations_t, total_samples_t = extended_trapezoidal_rule(user_func, args.limits,
                                                                                              args.accuracy)
            print("Trapezoidal Rule:\t\t\t\t", str(int_next_t) + " Took ", str(iterations_t),
                  "iterations, and a total of", str(total_samples_t), "samples")
            int_prev_s, int_next_s, iterations_s, total_samples_s = extended_simpsons_rule(user_func, args.limits,
                                                                                           args.accuracy)
            print("Simpson's Rule:\t\t\t\t\t", str(int_next_s) + " Took ", str(iterations_s),
                  "iterations, and a total of", str(total_samples_s), "samples")
            limits_mc = np.array([args.limits])
            integral, st_error, n_m = mc_integration(user_func, limits_mc, args.accuracy)
            print("Monte Carlo Integration (Uniform Sampling):\t", str(integral) + " Took a total of", str(n_m),
                  "samples, with a standard error: ", str(st_error))
            integral_i, st_error_i, n_m_i = mc_integration(user_func, limits_mc, args.accuracy, sampling='importance')
            print("Monte Carlo Integration (Importance Sampling):\t", str(integral_i) + " Took a total of", str(n_m_i),
                  "samples, with a standard error: ", str(st_error_i))
        elif args.fun == 'sho_gs_3d':
            user_func, user_func_mc = funcs[args.fun]  # Obtain function from dictionary
            limits = np.array(args.limits)
            if np.shape(limits)[0] != 6:  # Validate that integration limits match dimensions
                raise ValueError("\nPlease provide correct limits of integration")
            limits = np.reshape(limits, (3, 2))
            nt_prev_t, int_next_t, iterations_t, total_samples_t = extended_trapezoidal_rule_3d(user_func, limits,
                                                                                                args.accuracy)
            print("Trapezoidal Rule:\t\t\t\t", str(int_next_t) + " Took ", str(iterations_t),
                  "iterations, and a total of", str(total_samples_t), "samples")
            int_prev_s, int_next_s, iterations_s, total_samples_s = extended_simpsons_rule_3d(user_func, limits,
                                                                                              args.accuracy)
            print("Simpson's Rule:\t\t\t\t\t", str(int_next_s) + " Took ", str(iterations_s),
                  "iterations, and a total of", str(total_samples_s), "samples")
            integral, st_error, n_m = mc_integration(user_func_mc, limits, args.accuracy)
            print("Monte Carlo Integration (Uniform Sampling):\t", str(integral) + " Took a total of", str(n_m),
                  "samples, with a standard error: ", str(st_error))
            integral_i, st_error_i, n_m_i = mc_integration(user_func_mc, limits, args.accuracy, sampling='importance')
            print("Monte Carlo Integration (Importance Sampling):\t", str(integral_i) + " Took a total of", str(n_m_i),
                  "samples, with a standard error: ", str(st_error_i))
        elif args.fun == 'sho_es_3d':
            user_func, user_func_mc = funcs[args.fun]  # Obtain function from dictionary
            limits = np.array(args.limits)
            if np.shape(limits)[0] != 6:
                raise ValueError("\nPlease provide correct limits of integration")
            limits = np.reshape(limits, (3, 2))
            nt_prev_t, int_next_t, iterations_t, total_samples_t = extended_trapezoidal_rule_3d(user_func, limits,
                                                                                                args.accuracy)
            print("Trapezoidal Rule:\t\t\t\t", str(int_next_t) + " Took ", str(iterations_t),
                  "iterations, and a total of", str(total_samples_t), "samples")
            int_prev_s, int_next_s, iterations_s, total_samples_s = extended_simpsons_rule_3d(user_func, limits,
                                                                                              args.accuracy)
            print("Simpson's Rule:\t\t\t\t\t", str(int_next_s) + " Took ", str(iterations_s),
                  "iterations, and a total of", str(total_samples_s), "samples")
            integral, st_error, n_m = mc_integration(user_func_mc, limits, args.accuracy)
            print("Monte Carlo Integration (Uniform Sampling):\t", str(integral) + " Took a total of", str(n_m),
                  "samples, with a standard error: ", str(st_error))
            integral_i, st_error_i, n_m_i = mc_integration(user_func_mc, limits, args.accuracy, sampling='importance')
            print("Monte Carlo Integration (Importance Sampling):\t", str(integral_i) + " Took a total of", str(n_m_i),
                  "samples, with a standard error: ", str(st_error_i))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Georgios Alevras: Computational Physics Project - Script Help',
                                     epilog='Enjoy the script :)')
    parser.add_argument('-u', '--user_fun', help='User-defined function')
    parser.add_argument('-d', '--dimensions', type=int, help='Dimensions of user-defined function')
    parser.add_argument('-f', '--fun', help='Function name, defined in source-code')
    parser.add_argument('-l', '--limits', nargs="*", type=float, help='Integration limits, e.g. 1D: 0 1, 2D: 0 1 0 2, '
                                                                      '3D: 0 1 0 2 2 5')
    parser.add_argument('-a', '--accuracy', type=float, help='Convergence criterion - relative accuracy')
    arguments = parser.parse_args()  # Parses all arguments provided at script on command-line
    use_args(arguments)  # Executes code according to arguments provided
