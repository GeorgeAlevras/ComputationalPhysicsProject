import numpy as np
import matplotlib.pyplot as plt
import scipy.stats as st
import argparse
from functions import ground_state_1d, ground_state_3d_mc, ground_state_3d_for_plotting, f1
from main import mc_integration

"""
Georgios Alevras - 20/12/2020
-----------------------------
Python Version used: 3.8.2
Numpy Version used: 1.19.1
Scipy Version used: 1.5.2

Additional Dependencies: argparse
---------------------------------

        This file performs functionality-testing, unit-testing and various validations on the code to ensure that all
    functions work correctly and handle errors well.

        It contains the 7 following functions:
            1. plot_1d_function: Plots the 1D ground state wavefunction squared
            2. plot_3d_function: Plots the 3D ground state wavefunction squared, fixed at 10 different values of z
            3. trapezoidal_errors_with_samples: Plots errors from analytic solutions against number of samples for
                    different functions, both 1D and 3D
            4. mc_errors_with_samples: Plots errors from analytic solutions against number of samples for
                    different functions for 1D
            5. mc_sampling: Validates the PDFs used for MC as being actually uniform and linear
            6. mc_results_distribution: Validates that MC results are random, produce Gaussian distribution and that 
                    the spread is narrower for same number of samples when using importance sampling
            7. use_args: Reads all the parsed arguments provided by the reader in command-line and executes accordingly
        the integrators

"""


def plot_1d_function():
    # Plots the 1D ground state wavefunction squared
    x_range_1d = np.linspace(0, 2, 1000)
    y = ground_state_1d(x_range_1d)
    plt.plot(x_range_1d, y, label='PDF of 1D Ground State')
    plt.xlabel("X-axis [scaled, dimensionless variable]", fontname='Times New Roman', fontsize=12)
    plt.ylabel('|Ψ$_0$(x)|$^2$', fontname='Times New Roman', fontsize=12)
    plt.legend()
    plt.savefig('1d_ground_state.png')
    plt.show()
    return True


def plot_3d_function():
    # Plots the 3D ground state wavefunction squared, fixed at 10 different values of z
    ax = plt.axes(projection='3d')
    x = np.linspace(0, 2)
    y = np.linspace(0, 2)
    x, y = np.meshgrid(x, y)
    z_list = []
    for i in range(11):  # z is fixed from 0 to 10
        z_list.append(ground_state_3d_for_plotting(x, y, z=i/5))
    ax.plot_surface(x, y, z_list[0], rstride=1, cstride=1,
                    cmap='viridis', edgecolor='none')
    ax.plot_surface(x, y, z_list[1], rstride=1, cstride=1,
                    cmap='viridis', edgecolor='none')
    ax.plot_surface(x, y, z_list[2], rstride=1, cstride=1,
                    cmap='viridis', edgecolor='none')
    ax.plot_surface(x, y, z_list[3], rstride=1, cstride=1,
                    cmap='viridis', edgecolor='none')
    ax.plot_surface(x, y, z_list[4], rstride=1, cstride=1,
                    cmap='viridis', edgecolor='none')
    ax.plot_surface(x, y, z_list[5], rstride=1, cstride=1,
                    cmap='viridis', edgecolor='none')
    ax.plot_surface(x, y, z_list[6], rstride=1, cstride=1,
                    cmap='viridis', edgecolor='none')
    ax.plot_surface(x, y, z_list[7], rstride=1, cstride=1,
                    cmap='viridis', edgecolor='none')
    ax.plot_surface(x, y, z_list[8], rstride=1, cstride=1,
                    cmap='viridis', edgecolor='none')
    ax.plot_surface(x, y, z_list[0], rstride=1, cstride=1,
                    cmap='viridis', edgecolor='none')
    ax.plot_surface(x, y, z_list[10], rstride=1, cstride=1,
                    cmap='viridis', edgecolor='none')
    ax.set_title('PDF of 3D Ground State')
    ax.set_xlabel("X-Axis [scaled, dimensionless variable]")
    ax.set_ylabel("Y-Axis [scaled, dimensionless variable]")
    ax.set_zlabel('|Ψ$_0$(x, y, z)|$^2$')
    plt.savefig('3d_ground_state.png')
    plt.show()
    return True


def trapezoidal_errors_with_samples():
    """Errors from sin(x) with Trapezoidal Rule"""
    x_1d = [3, 9, 17, 65, 257, 513]  # Number of samples used
    y_1d = [9.617e-3, 5.987e-4, 1.4965e-4, 9.83526e-6, 5.84535e-7, 1.4613e-7]  # Errors from analytic solution
    x_r_1d = np.linspace(2, 513, 5110)
    y_theory_1d = [0.04 * x ** -2 for x in x_r_1d]  # Best-fit curve found
    y_theory_1d_er = [0.04 * x ** -2 for x in x_1d]
    p_value_1d = round(st.chisquare(y_1d, y_theory_1d_er)[1], 8)  # P-value: observed and expected data

    """Errors from  x**2+y**2+z**2 with Trapezoidal Rule"""
    x_3d = [5, 9, 33, 129]  # Number of samples used
    y_3d = [5.7129e-2, 6.527e-3, 4.836e-4, 4.7517e-5]  # Errors from analytic solution
    x_r_3d = np.linspace(4, 129, 125)
    y_theory_3d = [2.75 * x ** -2 / 3 for x in x_r_3d]  # Best-fit curve found
    y_theory_3d_er = [2.75 * x ** -2 / 3 for x in x_3d]
    p_value_3d = round(st.chisquare(y_3d, y_theory_3d_er)[1], 8)  # P-value: observed and expected data

    plt.figure(1)
    plt.plot(x_1d, y_1d, 'x', label='Errors Obtained')
    plt.plot(x_r_1d, y_theory_1d, label='y = 0.04*x$^-$$^2$, \nP-value: ' + str(p_value_1d))
    plt.xlabel('Number of Samples Used', fontname='Times New Roman', fontsize=12)
    plt.ylabel('Absolute Error, Deviation from Analytic Solution', fontname='Times New Roman', fontsize=12)
    plt.legend()
    plt.savefig('error_with_samples_1d.png')

    plt.figure(2)
    plt.plot(x_3d, y_3d, 'x', label='Errors Obtained')
    plt.plot(x_r_3d, y_theory_3d, label='y = 2.75*x$^-$$^2$$^/$$^3$, \nP-value: ' + str(p_value_3d))
    plt.xlabel('Number of Samples Used', fontname='Times New Roman', fontsize=12)
    plt.ylabel('Absolute Error, Deviation from Analytic Solution', fontname='Times New Roman', fontsize=12)
    plt.legend()
    plt.savefig('error_with_samples_3d.png')
    plt.show()
    return True


def mc_errors_with_samples():
    """Errors from sin(x) with uniform sampling MC integration"""
    x_1d = [5000, 320000, 40960000]  # Number of samples used
    y_1d = [4.7481e-7, 5.17136e-8, 3.7485e-9]  # Errors from analytic solution
    x_r_1d = np.linspace(4000, 40960000, 10000)
    y_theory_1d = [0.00003 * 1/np.sqrt(x) for x in x_r_1d]
    y_theory_1d_er = [0.0000327 * 1/np.sqrt(x) for x in x_1d]
    p_value_1d = round(st.chisquare(y_1d, y_theory_1d_er)[1], 10)  # P-value: observed and expected data

    plt.figure(1)
    plt.plot(x_1d, y_1d, 'x', label='Errors Obtained')
    plt.plot(x_r_1d, y_theory_1d, label='y = 0.0000327*$\sqrt{x}$$^-$$^2$, \nP-value: ' + str(p_value_1d))
    plt.xlabel('Number of Samples Used', fontname='Times New Roman', fontsize=12)
    plt.ylabel('Absolute Error, Deviation from Analytic Solution', fontname='Times New Roman', fontsize=12)
    plt.legend()
    plt.savefig('mc_error_with_samples_1d.png')
    plt.show()


def mc_sampling():
    """Plots the histograms of probabilities with uniform and importance sampling"""
    mc_integration(f1, np.array([[0, 1]]), 1e-3, sampling='flat', to_plot=True)
    mc_integration(f1, np.array([[0, 1]]), 1e-3, sampling='importance', to_plot=True)
    return True


def mc_results_distribution():
    """Runs MC integration 1200 times and plots histogram of values"""
    mc_integrals = []
    mc_integrals_i = []
    limits = np.array([0.0, 2.0, 0.0, 2.0, 0.0, 2.0])
    limits = np.reshape(limits, (3, 2))
    for i in range(1200):
        mc_integrals.append(mc_integration(ground_state_3d_mc, limits, 0.01, sampling='flat')[0])
        mc_integrals_i.append(mc_integration(ground_state_3d_mc, limits, 0.005, sampling='importance')[0])

    avg = round(np.average(mc_integrals), 7)
    avg_i = round(np.average(mc_integrals_i), 7)
    std = round(np.std(mc_integrals), 7)
    std_i = round(np.std(mc_integrals_i), 7)

    plt.figure(1)
    freq, p, _ = plt.hist(x=mc_integrals, bins=25, color='#0064f3', alpha=0.5, rwidth=0.9, label=' Average: ' + str(avg)
                                                                                       + '\nSt. Deviation: ' + str(std))
    freq_i, p_i, _ = plt.hist(x=mc_integrals_i, bins=25, color='#f91202', alpha=0.5, rwidth=0.9, label=' Average: ' +
                                                                        str(avg_i) + '\nSt. Deviation: ' + str(std_i))
    plt.title('Histogram of MC Values', fontname='Times New Roman', fontsize=16)
    plt.xlabel('MC Integration Estimates', fontname='Times New Roman', fontsize=12)
    plt.ylabel('Frequency', fontname='Times New Roman', fontsize=12)
    plt.legend()
    plt.savefig('mc_values_hist_1.png')
    plt.show()


def use_args(args):
    if args.plot_number == 0:
        plot_1d_function()
    elif args.plot_number == 1:
        plot_3d_function()
    elif args.plot_number == 2:
        trapezoidal_errors_with_samples()
    elif args.plot_number == 3:
        mc_sampling()
    elif args.plot_number == 4:
        mc_results_distribution()
    elif args.plot_number == 5:
        mc_errors_with_samples()
    else:
        raise ValueError("Not a valid test, please enter a digit between 0 and 4, e.g. -p 2")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Georgios Alevras: Computational Physics Project - Plots & Results '
                                                 'Help', epilog='Enjoy the script :)')
    parser.add_argument('-p', '--plot_number', type=float, help='Specify Plot Number to visualise')
    arguments = parser.parse_args()  # Parses all arguments provided at script on command-line
    use_args(arguments)  # Executes code according to arguments provided
