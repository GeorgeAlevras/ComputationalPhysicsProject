import numpy as np

"""
Georgios Alevras - 20/12/2020
-----------------------------
Python Version used: 3.8.2
Numpy Version used: 1.19.1
--------------------------

        This is a secondary file containing all the functions (integrands) to be evaluated with the integrators in the 
    main file.
        
        It contains 13 functions:
            1. ground_state_1d: The PDF of the ground state in time-independent SHO in 1D
            2. ground_state_3d: The PDF of the ground state in time-independent SHO in 3D
            3. ground_state_3d_mc: The PDF of the ground state in time-independent SHO in 3D
            4. excited_state_3d: The PDF of the 1st excited state in time-independent SHO in 3D with one orbital
        angular momentum
            5. excited_state_3d_mc: The PDF of the 1st excited state in time-independent SHO in 3D with one orbital
        angular momentum
            6. f1: sin(x)/x
            7. f2: 0
            8. f3: x
            9. f4: sin(xy) e^(yz)
            10. f4_mc: sin(xy) e^(yz)
            11. f5: x**2
            12. f6: x**3
            13. ground_state_3d_for_plotting: The PDF of the ground state in time-independent SHO in 3D, coded in a way
        that facilitates its plotting
"""


def ground_state_1d(x):
    return (1/np.sqrt(np.pi)) * np.exp(-x**2)


def ground_state_3d(x, y, z):
    variables = np.array([x, y, z])
    psi_squared = (1/np.sqrt(np.pi)) * np.exp(-variables**2)
    return np.product(psi_squared)


def ground_state_3d_mc(x_v):
    x, y, z = x_v
    psi_x_squared = (1/np.sqrt(np.pi)) * np.exp(-x**2)
    psi_y_squared = (1/np.sqrt(np.pi)) * np.exp(-y**2)
    psi_z_squared = (1/np.sqrt(np.pi)) * np.exp(-z**2)
    return psi_x_squared * psi_y_squared * psi_z_squared


def excited_state_3d(x, y, z):
    variables = np.array([x, y, z])
    psi_0_squared = (1/np.sqrt(np.pi)) * np.exp(-variables**2)
    psi_1_squared = (2/np.sqrt(np.pi)) * variables**2 * np.exp(-variables**2)
    return 0.5 * psi_0_squared[2] * (psi_0_squared[1]*psi_1_squared[0] + psi_0_squared[0]*psi_1_squared[1])


def excited_state_3d_mc(x_v):
    x, y, z = x_v
    psi0_x_squared = (1/np.sqrt(np.pi)) * np.exp(-x**2)
    psi0_y_squared = (1/np.sqrt(np.pi)) * np.exp(-y**2)
    psi0_z_squared = (1/np.sqrt(np.pi)) * np.exp(-z**2)
    psi1_x_squared = (2/np.sqrt(np.pi)) * x**2 * np.exp(-x**2)
    psi1_y_squared = (2/np.sqrt(np.pi)) * y**2 * np.exp(-y**2)
    return 0.5 * psi0_z_squared * (psi1_x_squared*psi0_y_squared + psi0_x_squared*psi1_y_squared)


def f1(x):
    return np.sin(x)/x


def f2(x):
    return 0


def f3(x):
    return x


def f4(x, y, z):
    return np.sin(x*y) * np.exp(y*z)


def f4_mc(x_v):
    x, y, z = x_v
    return np.sin(x*y) * np.exp(y*z)


def f5(x):
    return x**2


def f6(x):
    return x**3


def ground_state_3d_for_plotting(x, y, z=0):
    psi_x_squared = (1/np.sqrt(np.pi)) * np.exp(-x**2)
    psi_y_squared = (1/np.sqrt(np.pi)) * np.exp(-y**2)
    psi_z_squared = (1/np.sqrt(np.pi)) * np.exp(-z**2)
    return psi_x_squared * psi_y_squared * psi_z_squared


if __name__ == '__main__':
    print('Empty script, run the main.py file with --help flag to get started.')
