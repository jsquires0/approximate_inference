# this is a test script, proving functionality of code
# I've chosen to use the sorobn library written by Max Halford
# https://github.com/MaxHalford/sorobn
# This library is more transparent than high performance BN libraries
# As the project proceeds, I may revise/tweak the implementation

import sorobn as sbn
import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)
import numpy as np 
import pandas as pd 
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
		niter = 1
		results = []
		start = time.process_time()
		for _ in range(niter):
			results.append(func(*args, **kwargs))
		end = time.process_time()
		print(f'Avg CPU time: {(end-start)/niter}')

		# aggregate probabilities
		return aggregate(results)
	return wrapper 


class Driver:
	""" Takes a BN and runs a series of approximate/exact inference methods in 
	order to compare performance """

	def __init__(self, network):
		self.network = network
		self.network.prepare()
		return

	@timeit_wrapper
	def infer(self, alg, q, E, N=100):
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
		exact_dist = self.infer(alg='exact',
				 q=query,
				 E=evidence,
				 )

		pprint.pprint(f'Exact Inference {exact_dist}'); print('\n')

		rej_dist = self.infer(alg='rejection',
					q=query,
					E=evidence,
					N=N)
		pprint.pprint(f'Approx Inference (rejection): {rej_dist}'); print('\n')

		gibbs_dist = self.infer(alg='gibbs',
					q=query,
					E=evidence,
					N=N)
		pprint.pprint(f'Approx Inference (gibbs): {gibbs_dist}'); print('\n')

		lw_dist = self.infer(alg='likelihood',
					q=query,
					E=evidence,
					N=N)
		pprint.pprint(f'Approx Inference (likelihood): {lw_dist}'); print('\n')


		return exact_dist, rej_dist, gibbs_dist, lw_dist


if __name__ == '__main__':
	
	driver = Driver(Asia_Network)
	# # compute P(q|E) with exact (variable elimination) & approx inference algs
	query = 'Lung'
	evidence = {'Asia': 'yes', 'Smoke': 'no'}
	exact, rej, gibbs, lw = driver.run_all_inference_methods(query, evidence)
	

	driver = Driver(Insurance_Network)
	# compute P(q|E) with exact (variable elimination) & approx inference algs
	query = 'PropCost'
	evidence = {'Age': 'Adolescent', 'Antilock': 'False', 'Mileage': 'FiftyThou', 'MakeModel': 'SportsCar' }
	exact, rej, gibbs, lw = driver.run_all_inference_methods(query, evidence)