# mcmc_line_fitting
Metropolis-Hastings Markov Chain Monte Carlo Line Fitting Routine

Developed by: Jared Keown

The mcmc_line_fitting.py code was designed to find the set of model parameters that best describe 
molecular gas emission spectra from dense, star forming clouds.  It currently 
uses the HILL5 radiative transfer infall/outflow model by De Vries et al. (2005),
but can easily be adapted to fit other models to different types of data.

This directory also contains a sample spectra that can be used to test the code, as well as
the mcmc_params_plotter.py code that creates histograms and triangle plots of the 
maximum likelihood parameters of the model. 
