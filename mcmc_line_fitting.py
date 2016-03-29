################################################################################
## Metropolis Hastings Markov-Chain Monte Carlo emission line fitting routine
## Developed by: Jared Keown

## This code was designed to find the set of model parameters that best describe 
## molecular gas emission spectra from dense, star forming clouds.  It currently 
## uses the HILL5 radiative transfer infall/outflow model by De Vries et al. (2005),
## but can easily be adapted to fit other models to different types of data.
 
## Uses the traditional weighted least squares equation as the error function
##  - This error function can easily be altered for other MCMC applications
## Uses a Gaussian likelihood function, forcing the likelihood to be maximum
##    when the error function is minimal
## Stores the minimal likelihood parameter sets in a separate text file
##  - This text file can be loaded with another script for plotting histograms

################################################################################

import numpy,math
import matplotlib.pyplot as plt
import random as random

#Print the minimal likelihood parameter sets in a text file with name given below
file_name = 'mcmc_line_fitting_output.dat'

#Create file with name file_name and overwrite if it already exists
with open(file_name, 'w') as text_file:
	# Include a parameters key as the first line to remind you 
	# which parameters correspond to which columns in the file
	text_file.write('#parameters key: [T_pk, sig, tau0, v_in, v_lsr]')
	text_file.close()

#Text file with spectra data
#The file contains two columns:
#First column = local standard of rest velocity (V_lsr, i.e., x-value of spectra)
#Second column = Brightness Temperature at given V_lsr (T_B, i.e., y-value of spectra) 

spectra = "l694-dcop21-red-p00p00.dat"

#Create final spectrum and characterize standard deviation for error estimate
#Import the data
velocity1, temp1 = numpy.loadtxt(spectra, unpack=True)

#Cut the far right and left dip off the spectra
comb1 = zip(velocity1, temp1)
x1 = []
y1 = []
for v,t in comb1:
	if  v < (18.) and v > (1.):
		x1.append(v)
		y1.append(t)

#Define the estimated error based on standard deviation
#over portion of spectra where no emission was detected
#i.e., RMS noise  
no_emission = temp1[numpy.where(velocity1 < 8.)]
err1 = numpy.std(no_emission)

#Define the HILL5 model
def p_eval(x,p):
	x = numpy.array(x)
	T_pk = p[0]          # Peak Excitation Temperature
	sig = p[1]          # Velocity dispersion of all lines
	Tau0 = p[2]          # Total optical depth of the hyperfines
	V_in = p[3]          # infall velocity
	V_lsr = p[4]          # Systemic Velocity

	#Constants and Transition Frequency
	T_0 = 2.73 		#cosmic background temperature
	v1 = 144077.29*(10**6) 	#transition frequency of given tracer (in 1/s)
	h = (6.626*(10**-27))	#Planck's constant (in J*s)
	k = (1.381*(10**-16)) 	#Boltzmann constant (in J/Kelvin)
	T_1 = (h*v1)/k		#variable for planck corrected brightness temp. equations

	Vf = V_lsr+V_in
	Vr = V_lsr-V_in

	#Front and rear optical depth equations
	a = (x-Vf)**2
	b = 2.*(sig**2)
	c = numpy.exp(-(a/b))
	t_f = Tau0*c
	d = (x-Vr)**2
	e = 2.*(sig**2)
	f = numpy.exp(-(d/e))
	t_r = Tau0*f
	
	#Below two lines came from Chris De Vries website:
	#https://github.com/devries/analytic_infall/blob/master/hill5.c
	#Necessary for setting low values of t_f and t_r to 1.0
	subf = numpy.where(t_f>0.0001, (1.-numpy.exp(-t_f))/t_f, 1.)
	subr = numpy.where(t_r>0.0001, (1.-numpy.exp(-t_r))/t_r, 1.)

	#Planck corrected brightness temperatures
	JT_pk = T_1/(numpy.exp(T_1/T_pk)-1.)
	JT_0 = T_1/(numpy.exp(T_1/T_0)-1.)

	r = JT_pk - JT_0

	#Line brightness temperature equation from De Vries website: 
	return (r*(subf-(numpy.exp(-t_f)*subr)))

# Define a function that evaluates -ln(P) for input parameters
def evaluateLogLikelihood(params):
    T_peak = params[0] 
    sigma = params[1]  
    tau_not = params[2]
    v_infall = params[3]
    v_local = params[4]
    
    T_brightness = p_eval(x1,p=[T_peak,sigma,tau_not,v_infall,v_local])

    # Compute P (where P is the traditional weighted least squares equation)
    chi_squared = sum(((numpy.array(T_brightness)-numpy.array(y1))**2)/(2*(err1**2)))
    
    # return log likelihood
    return chi_squared*-1.0

#Define given starting parameters for HILL5 model and LogLikelihood function
params_0 = [8.0, 0.1, 1.5, 0.08, 9.6]

#Define max size of step to alter parameters each iteration
#can change if need different step sizes for each parameter
d_T = 0.005
d_sig = 0.005
d_tau = 0.005
d_v_in = 0.005
d_v_lsr = 0.005

#Define uniformly drawn steps to change parameters each iteration

def del_T():
	del_T = random.uniform(-d_T,d_T)
	return del_T

def del_sig():
	del_sig = random.uniform(-d_sig,d_sig)
	return del_sig

def del_tau():
	del_tau = random.uniform(-d_tau,d_tau)
	return del_tau

def del_vin():
	del_vin = random.uniform(-d_v_in,d_v_in)
	return del_vin

def del_vlsr():
	del_vlsr = random.uniform(-d_v_lsr,d_v_lsr)
	return del_vlsr

#Set accepted counter to zero
accepted  = 0

# Metropolis-Hastings with 10**6 iterations.
while(accepted < 10.**6.):
	old_params = params_0
    	old_loglik = evaluateLogLikelihood(old_params)
    	# Suggest new candidate from uniform distribution
	# Also force values to be greater than specified lower limits
	new_params = [-1., -1., -1., -2., -1.]
   	
	while(new_params[0] < 0.):	
		new_params[0] = old_params[0] + del_T()
	while(new_params[1] < 0.):	
		new_params[1] = old_params[1] + del_sig()
	while(new_params[2] < 0.):	
		new_params[2] = old_params[2] + del_tau()
	while(new_params[3] < -1.):
		new_params[3] = old_params[3] + del_vin()
	while(new_params[4] < 0.):
		new_params[4] = old_params[4] + del_vlsr()

    	new_loglik = evaluateLogLikelihood(new_params)

    	# Accept new candidate if it passes test
    	if (new_loglik > old_loglik):
		accepted_parameters = str(new_params)	

		#Append the accepted parameter set in the text file
		with open(file_name, 'a') as text_file:
			text_file.write('\n' + accepted_parameters[1:(len(accepted_parameters)-1)])
			text_file.close()
		#Replace initial paramater set with newly accepted set
		params_0 = new_params
        	accepted = accepted + 1  # monitor acceptance
		print accepted
    	else:
       		u = random.uniform(0.0,1.0)
        	if (u < math.exp(new_loglik - old_loglik)):
            		accepted_parameters = str(new_params)	

			#Print the values in a text file
			with open(file_name, 'a') as text_file:
				text_file.write('\n' + accepted_parameters[1:(len(accepted_parameters)-1)])
				text_file.close()
			#Replace initial paramater set with newly accepted set
			params_0 = new_params
        		accepted = accepted + 1  # monitor acceptance
			print accepted
        	else:
			#If the new parameter set does not pass test, restart with the old parameter set
            		params_0=old_params
