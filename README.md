# Computational Physics Coursework - 3D Quantum Systems Integrator #

This was my coursework representing 60% of the course 'Computational Physics' - obtaining a grade of 84.4%. The project involved developing numerical methods of integration through Newton-Coates and Monte Carlo integration to solve 1D and 3D quantum systems that cannot be solved analytically. The speed of convergence for the different methods were compared, as well as the scaling of errors with the number of samples for 1D and 3D systems for the different methods.

## Organisation ##
The repository contains:
- The report submitted for the coursework `./alevras_georgios_project_report.pdf`
- The file with the main source code that runs integrator methods `./main.py`
- A script that performs all test validations (unit testing, edge-case testing, type and value errors) `./testing.py`
- A script that produces all the plots and figures used and shown in the report `./plots_and_results.py`
- A file containing all the functions used in the project `./functions.py`
- A file attempting to do adaptive sampling for MC integration through the metropolis algorithm `./metropolis.py`
- A directory containing all the plots produced and shown in the report `./plots`

--------------------------------------------------------------------------------------------------------

## Short Code Explanation and Execution ##
main.py:
- Integrate any 1D function of your choice
- Integrate any 3D function of your choice (does not include MC because of how lambda is defined and obtained from command line. However, MC works just fine for 3D as validated, if one wants to try it, must hard-code a function and use that, e.g. functions.py lines: 82-84)
- Integrate any of the 3 functions provided at script: Ground state 1D, Ground state 3D, 3D state with 1 unit of orbital angular momentum
- Arguments that must be specified:
    - User-defined function -u argument, or hard-coded function -f argument
	- If `-u` selected, must specify dimensions of integration with `-d` argument
	- limits of integration with `-l` argument
	- relative accuracy with `-a` argument
Do not leave blanks when specifying your own function, or even better put in brackets, so that the lambda eval function can understand what constitutes variables and mathematical operations

E.g. 
- Integrate sin(x) from 0 to 1 at rel. acc = 0.001:
    `python main.py -u np.sin(x) -d 1 -l 0 1 -a 1e-3`
- Integrate sin(xy)*e^(yz) from x: 0 to 1, y: 0 to 2, z: 0 to 3 rel. acc = 0.01:
	`python main.py -u np.sin(x*y)*np.exp(y*z) -d 3 -l 0 1 0 2 0 3 -a 1e-2`
- Integrate x^2 + y^2 + z^2 from x: 0 to 1, y: 0 to 1, z: 0 to 1 rel. acc = 0.01:
	`python main.py -u x**2+y**2+z**2 -d 3 -l 0 1 0 1 0 1 -a 1e-2`
- Integrate 1D Ground State from 0 to 2 at rel. acc = 0.0001:
	`python main.py -f sho_gs_1d -l 0 2 -a 1e-4`
- Integrate 3D Ground State from x: 0 to 2, y: 0 to 2, z: 0 to 2 rel. acc = 0.01:
	`python main.py -f sho_gs_3d -l 0 2 0 2 0 2 -a 1e-2`
- Integrate 3D State with orbital momentum from x: 0 to 2, y: 0 to 2, z: 0 to 2 rel. acc = 0.01:
	`python main.py -f sho_es_3d -l 0 2 0 2 0 2 -a 1e-2`
--------------------------------------------------------------------------------------------------------

testing.py
- Perform any of the 8 available tests described in lines: 20-31
- Arguments that must be specified:
	- Test number `-t` argument, argument can range from 0 to 7

E.g.
- Test methods for edge cases, singularities, 0 relative accuracies, etc.:
	`python testing.py -t 0`
- Test methods do not accept unreasonable values of relative accury:
	`python testing.py -t 1`
--------------------------------------------------------------------------------------------------------

plots_and_results.py
- Produce all the plots shown in the report
- Arguments that must be specified:
    - Plot number -p argument, argument can range from 0 to 5

E.g.
- Plot 1D Ground State PDF:
    `python plots_and_results.py -p 0`
- Plot 3D Ground State PDF for fixed z-values:
    `python plots_and_results.py -p 1`
- Plot Trapezoidal error-samples relation:
    `python plots_and_results.py -p 2`
--------------------------------------------------------------------------------------------------------

metropolis.py
- Run the metropolis algorithm with adaptive sampling
- No arguments needed, just run the file as such:
	`python metropolis.py`
