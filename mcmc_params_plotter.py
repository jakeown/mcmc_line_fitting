import numpy,math
import matplotlib.pyplot as plt
import random as random
import pylab


def mcmc_plot(param_file):
	params = numpy.loadtxt(param_file, delimiter=',', unpack=True) # params = [T_pk, sig, tau, v_in, v_lsr]
	params_names = ['T_pk', 'sig', 'tau', 'v_in', 'v_lsr']
	plot_names = ['T$_{pk}$', '$\sigma$', r'$\tau$', 'v$_{in}$', 'v$_{lsr}$']
	counter=0
	nbins = 40
	for i in params_names: 
		#Plot 1D Histograms
		fig1 = plt.figure()
		plt.hist(params[counter], bins=40, label='40 bins', range = [min(params[counter]), max(params[counter])], 			facecolor='g')
		plt.legend(loc='upper left')
		plt.title(plot_names[counter]+' histogram')
		plt.ylabel('counts')
		plt.xlabel(plot_names[counter])
		#plt.show()
		pylab.savefig(i +'_hist.pdf')
		plt.close()

		#Plot iterations vs parameter plot
		fig1 = plt.figure()
		plt.scatter(numpy.array(range(len(params[counter])))+1, params[counter])
		plt.title(plot_names[counter] + ' - ' + str(len(params[counter])) +' iterations')
		plt.ylabel(plot_names[counter])
		plt.xlabel('iteration')
		#plt.show()
		pylab.savefig(i +'_iterations.pdf')
		plt.close()

		# Create the 2D histograms
		nrange = [[min(params[3]), max(params[3])],[min(params[counter]), max(params[counter])]]
		H, xedges, yedges = numpy.histogram2d(params[3],params[counter],bins=nbins,range = nrange)
		H = numpy.rot90(H)
		H = numpy.flipud(H)
		# Plot 2D histogram using pcolor
		fig4 = plt.figure()
		plt.pcolormesh(xedges,yedges,H)
		plt.xlabel('v$_{in}$')
		plt.ylabel(plot_names[counter])
		plt.title(plot_names[counter]+' vs v$_{in}$ - ' + str(len(params[3])) + ' iterations')
		plt.xlim(min(xedges), max(xedges))
		plt.ylim(min(yedges), max(yedges))
		cbar = plt.colorbar()
		cbar.ax.set_ylabel('Counts')
		#plt.show()
		pylab.savefig(i +'_v_in_2dhist.pdf')
		plt.close()
		counter += 1

def mcmc_plot_triangle(param_file, spectra, xlims=[8.7, 10.7], ylims=[-0.5,5.0]):
	params = numpy.loadtxt(param_file, delimiter=',', unpack=True) # params = [T_pk, sig, tau, v_in, v_lsr]
	
	params_names = ['T_pk', 'sig', 'tau', 'v_in', 'v_lsr']
	plot_names = ['T$_{pk}$', '$\sigma$', r'$\tau$', 'v$_{in}$', 'v$_{lsr}$']
	axes = [1,7,13,19,25]
	axes_2d = [6,11,16,21,12,17,22,18,23,24]
	nbins = 40
	counter=0
	counter2=0
	fig1 = plt.figure(figsize=(8,8))
	for i in params_names:
		ax1 = fig1.add_subplot(5,5,axes[counter]) 
		#Plot 1D Histograms
		ax1.hist(params[counter], bins=40, label='40 bins', range = [min(params[counter]), max(params[counter])], histtype='step')
		if counter==4:
			plt.xlabel(plot_names[counter])
		ax1.set_xticklabels([])
		ax1.set_yticklabels([])
		#plt.xticks(rotation=70)
		plt.xlim(min(params[counter]), max(params[counter]))
		if counter==0:
			params_T = [params[1], params[2], params[3], params[4]]
		elif counter==1:
			params_T = [params[2], params[3], params[4]]
		elif counter==2:				
			params_T = [params[3], params[4]]
		elif counter==3:
			params_T = [params[4]]
		if counter<=3:
			for i in range(len(params_T)):
				# Create the 2D histograms
				min_x = params[counter]
				min_y = params_T[i]

				nrange = [[min(min_x), max(min_x)],[min(min_y), max(min_y)]]
			
				H, xedges, yedges = numpy.histogram2d(params[counter],params_T[i],bins=nbins,range = nrange)
				H = numpy.rot90(H)
				H = numpy.flipud(H)
				# Plot 2D histogram using pcolor
				ax1 = fig1.add_subplot(5,5,axes_2d[counter2])
				pcol = plt.pcolormesh(xedges,yedges,H, linewidth=0, rasterized=True)
				pcol.set_edgecolor("face")
				if axes_2d[counter2]==21 or axes_2d[counter2]==22 or axes_2d[counter2]==23 or axes_2d[counter2]==24:
					plt.xlabel(plot_names[counter])
					plt.xticks(rotation=75)
				else:
					#plt.xticks(rotation=70)
					ax1.set_xticklabels([])
				if counter==0:
					plt.ylabel(plot_names[i+1])
				else:
					ax1.set_yticklabels([])
				plt.xlim(min(xedges), max(xedges))
				plt.ylim(min(yedges), max(yedges))
				for tick in ax1.get_xticklines():
					tick.set_color('white')
				for tick in ax1.get_yticklines():
					tick.set_color('white')
				#cbar = plt.colorbar()
				#cbar.ax.set_ylabel('Counts')
				counter2+=1
		counter += 1
	x1,y1 = numpy.loadtxt(spectra, unpack=True)
	ax = fig1.add_subplot(5,5,5)
	ax.plot( x1, y1,'k', linewidth=1, label="Data", drawstyle='steps')
	plt.ylim(ylims)
	plt.xlim(xlims)
	plt.xlabel('V$_{lsr}$')
	plt.ylabel('T$_{B}$')
	fig1.subplots_adjust(hspace=0.1, wspace=0.1)
	pylab.savefig('mcmc_params_triangle_plot.pdf')
	plt.close()

mcmc_plot(param_file='mcmc_line_fitting_output.dat')
mcmc_plot_triangle(param_file='mcmc_line_fitting_output.dat', spectra='l694-dcop21-red-p00p00.dat')
