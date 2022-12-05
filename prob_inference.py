import os 
import sorobn as sbn
import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)
import numpy as np 
import pandas as pd 
import matplotlib.pyplot as plt
import pprint
import time
from collections import defaultdict
from scipy.special import rel_entr as KLDiv

from alarm_net import Alarm_Network
from asia_net import Asia_Network
from insurance_net import Insurance_Network

def aggregate(dict_list):
	# average results
	results = defaultdict(list)

	for d in dict_list:
		for k,v in d.items():
			results[k].append(v)

	for k,v in results.items():
		results[k] = np.mean(v)
	return results

def timeit_wrapper(func):
	# computes average CPU execution time over N iterations
	def wrapper(*args, **kwargs):
		niter = 10
		results = []
		start = time.process_time()
		for _ in range(niter):
			results.append(func(*args, **kwargs))
		end = time.process_time()
		duration = (end-start)/niter
		print(f'Avg CPU time: {duration}')

		# aggregate probabilities
		return aggregate(results), duration
	return wrapper 


class BNAnalysis:
	""" Takes a BN and runs a series of approximate/exact inference methods in 
	order to compare performance """

	def __init__(self, network):
		self.network = network
		self.network.prepare()
		return

	@timeit_wrapper
	def infer(self, alg, q, E, N=1000):
		"""
		Makes an inference
		Args:
			alg: string representing algorithm choice. Options are
				'exact', 'gibbs', 'likelihood', or 'rejection'
			q: query variable(s)
			E: dictionary containing evidence
			N: number of iterations for approx inference. Ignored for exact
		Returns:
			P(q|E) as estimated/computed with specified algorithm
		"""

		out = self.network.query(q, event=E, algorithm=alg, n_iterations=N)
		return out
	
	def run_all_inference_methods(self, query, evidence, N=100):
		"""
		Makes an inference using approximate and exact BN inference algs
		Args:
			query: query variable(s)
			evidence: dictionary containing evidence
			N: number of iterations for approx inference. Ignored for exact
		Returns:
			P(q|E) as estimated/computed with specified algorithm
		"""
		exact_dist, t_exact = self.infer(alg='exact',
				 q=query,
				 E=evidence,
				 )

		pprint.pprint(f'Exact Inference {exact_dist}'); print('\n')

		rej_dist, t_rej = self.infer(alg='rejection',
					q=query,
					E=evidence,
					N=N)
		pprint.pprint(f'Approx Inference (rejection): {rej_dist}'); print('\n')

		lw_dist, t_lw = self.infer(alg='likelihood',
					q=query,
					E=evidence,
					N=N)
		pprint.pprint(f'Approx Inference (likelihood): {lw_dist}'); print('\n')

		gibbs_dist, t_gibbs = self.infer(alg='gibbs',
					q=query,
					E=evidence,
					N=N)
		pprint.pprint(f'Approx Inference (gibbs): {gibbs_dist}'); print('\n')
		order = sorted(exact_dist.keys())
		entropies = {}
		for idx, aprx in enumerate([rej_dist, lw_dist, gibbs_dist]):
			try:
				approx_list = [aprx[k] for k in order]
				entropies[idx]= sum(KLDiv(list(exact_dist.values()),approx_list))
			except:
				continue 
				
		pprint.pprint(entropies)

		
		return exact_dist, rej_dist, gibbs_dist, lw_dist

	def KL_iteration_plot(self, alg, query, evidence, max_n=1000):
		# Plot KL divergence, runtime vs iteration
		entropies = []
		times = []
		N_iter = []
		percents = []

		# compute exact distribution
		exact_dist, t = self.infer(alg='exact',
				 q=query,
				 E=evidence,
				 )

		order = sorted(exact_dist.keys())
		# compute approx distribution as a function of # iterations
		duration = 0
		for i in range(0, max_n, 500):
			approx_dist, t = self.infer(alg,
					q=query,
					E=evidence,
					N=i+1)
			duration+=t

			try:
				approx_list = [approx_dist[k] for k in order]
				entropies.append(
					sum(KLDiv(list(exact_dist.values()),approx_list))
				)
				times.append(duration)
				N_iter.append(i)
				percents.append(
					np.abs(exact_dist[order[0]] - approx_list[0])/exact_dist[order[0]] * 100
				)
			except:
				continue

		# plot runtime
		plt.plot(N_iter,times)
		plt.ylabel('Time')
		plt.xlabel('Iteration')
		name = os.path.dirname(__file__) + '/plots/' + 'Time' +'.png'
		plt.savefig(fname=name)
		plt.clf()
		# plot kl divergence
		plt.plot(N_iter, entropies)
		plt.ylabel('KL Divergence')
		plt.xlabel('Iteration')
		name = os.path.dirname(__file__) + '/plots/' + 'KL' +'.png'
		plt.savefig(fname=name)
		plt.clf()
		# plot % error
		plt.plot(N_iter, percents)
		plt.ylabel('Percent Error')
		plt.xlabel('Iteration')
		name = os.path.dirname(__file__) + '/plots/' + 'Percent_Err' +'.png'
		plt.savefig(fname=name)
		
		return  approx_dist, exact_dist

if __name__ == '__main__':
	
	BNA = BNAnalysis(Asia_Network)
	# compute P(q|E) with exact (variable elimination) & approx inference algs
	query = 'Lung'
	evidence = {'Asia': 'yes', 'Smoke': 'no'}
	exact, rej, gibbs, lw = BNA.run_all_inference_methods(query, evidence, N=1000)
	
	BNA = BNAnalysis(Insurance_Network)
	# compute P(q|E) with exact (variable elimination) & approx inference algs
	query = 'PropCost'
	evidence = {'Age': 'Senior'}
	exact, rej, gibbs, lw = BNA.run_all_inference_methods(query, evidence, N=1000)

	query = 'Age'
	evidence = {'MedCost': 'Million', 'RiskAversion': 'Psychopath', 'Theft': 'True'}
	exact, rej, gibbs, lw = BNA.run_all_inference_methods(query, evidence, N=100000)

	# query = 'PropCost'
	# evidence = {'Age': 'Senior'}
	# aprx, exact = BNA.KL_iteration_plot('likelihood', query, evidence, max_n=6000)
