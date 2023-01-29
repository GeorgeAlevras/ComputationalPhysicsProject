import numpy as np
from numba import njit
from numba.typed import List


@njit
def metropolis(func, a, b, rel_accuracy, sampling='flat'):
    dimensions = len(a)
    steps = np.zeros(dimensions)  # Will hold new steps for each points
    std = np.zeros(dimensions)  # List of standard deviations for Gaussians to generate new points
    volume = 1
    for i in range(dimensions):
        steps[i] = b[i]-a[i]
        std[i] = 0.25*steps[i]
        volume *= steps[i]

    n = int(1e3)
    for l in range(20):
        rejected = 0  # Number of rejected points from new steps
        integral = 0  # Estimate of integral
        error = 0  # Variable keeps error
        for i in range(n):
            x_original = np.zeros(dimensions)  # Holds original points in each dimension
            x_test =  np.zeros(dimensions)  # Holds new points in each dimension
            for j in range(dimensions):
                x_original[j] = np.random.uniform(a[j], b[j])  # Initial point in each dimension
            for j in range(dimensions):
                # Take small step from original point in each dimension
                x_test[j] = x_original[j] + np.random.normal(0, std[j])

            p_x = helper(x_original, a, b, sampling)  # Probability of original points
            p_x_t = helper(x_test, a, b, sampling)  # Probability of new points
            if p_x_t >= p_x:  # If probability of new point > probability of original point
                x_original = x_test  # Change to new point
                p_x = p_x_t  # Change to new probability
            else:
                tmp = np.random.uniform(0, 1)  # Dummy probability between 0 and 1
                if p_x_t/p_x > tmp:  # If alpha is larger than the dummy
                    x_original = x_test  # Change to new point
                    p_x = p_x_t  # Change to new probability
                else:
                    rejected += 1  # If alpha is smaller, reject it

            integral += (1/n)*func(x_original)/p_x  # Add to integral sum current estimate
            error += (1/n)*func(x_original**2)/p_x  # Add to error sum current error

        if (1-rejected/n) < 0.23:  # if alpha is smaller than 'optimum' value
            for j in range(dimensions):
                std[j] -= abs((1-rejected/n)-0.23)  # If smaller, reduce the standard deviation
        else:
            for j in range(dimensions):
                std[j] += abs((1-rejected/n)-0.23)  # If larger, increase the standard deviation

        st_error = np.sqrt((error - integral**2)/n)/integral  # Final relative standard error
        print('\nIntegral:', integral)
        print('Error:', st_error)
        print('Acceptance ratio:', (1-rejected/n))
        print('Samples:', n)
        n *= 2
        if st_error < rel_accuracy:
            return integral, st_error, n, (1-rejected/n)


@njit
def helper(x1, l, u, sampling='flat'):
    for i in range(len(x1)):
        if not l[i] < x1[i] < u[i]:
            return 0
    if sampling == 'flat':
        p = 1
        for i in range(len(l)):
            p *= abs(u[i]-l[i])
        return 1/p
    elif sampling == 'importance':
        p = 1
        for i in range(len(x1)):
            p *= pdf_inv((x1[i]-l[i])/(u[i]-l[i]))
        return pdf(p)


@njit
def pdf(x, a=-1, b=1.5):
    return a * x + b


@njit
def pdf_inv(x, a=-1, b=1.5):
    return -(b / a) - np.sqrt((b / a) ** 2 + 2 * x / a)


@njit
def ground_state_3d(x_v):
    x, y, z = x_v
    psi_x_squared = (1/np.sqrt(np.pi)) * np.exp(-x**2)
    psi_y_squared = (1/np.sqrt(np.pi)) * np.exp(-y**2)
    psi_z_squared = (1/np.sqrt(np.pi)) * np.exp(-z**2)
    return psi_x_squared * psi_y_squared * psi_z_squared


a = [0, 0, 0]
b = [2, 2, 2]
a_t = List()
[a_t.append(x) for x in a]
b_t = List()
[b_t.append(x) for x in b]
integral, error, samples, acceptance = metropolis(ground_state_3d, a_t, b_t, 1e-3)
