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

from alarm_net import Alarm_Network


def timeit_wrapper(func):
	# computes average CPU execution time over N iterations
	def wrapper(*args, **kwargs):
		niter = 10
		results = np.zeros(niter)
		start = time.process_time()
		for i in range(niter):
			try:
				results[i] = func(*args, **kwargs)[True]
			except: # p_false = 1.0
				results[i] = 0.0
		end = time.process_time()
		print(f'Avg CPU time: {(end-start)/niter}')

		# aggregate probabilities
		p_true = np.mean(results)
		return {True: p_true, False: 1-p_true}
	return wrapper 


class Driver:
	""" Takes a BN and runs a series of approximate/exact inference methods in 
	order to compare performance """

	def __init__(self, network):
		self.network = network
		self.network.prepare()
		return

	@timeit_wrapper
	def infer(self, alg, q, E, N=10000):
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


if __name__ == '__main__':

	# Testing algorithms on Pearl's Alarm network, found in:
	# Artificial Intelligence: A Modern Approach, Russel & Norvig 2021
	driver = Driver(Alarm_Network)

	# compute P(q|E) with exact (variable elimination) & approx inference algs
	result = driver.infer(alg='exact',
				 q='Burglary',
				 E={'MaryCalls':True, 'JohnCalls': True}
				 )
	pprint.pprint(f'Exact Inference {result}'); print('\n')

	result = driver.infer(alg='rejection',
				q='Burglary',
				E={'MaryCalls':True, 'JohnCalls': True}
				)
	pprint.pprint(f'Approx Inference (rejection): {result}'); print('\n')

	result = driver.infer(alg='gibbs',
			q='Burglary',
			E={'MaryCalls':True, 'JohnCalls': True}
			)
	pprint.pprint(f'Approx Inference (gibbs): {result}'); print('\n')

	result = driver.infer(alg='likelihood',
		q='Burglary',
		E={'MaryCalls':True, 'JohnCalls': True},
		N = 100000,
		)
	pprint.pprint(f'Approx Inference (likelihood): {result}'); print('\n')