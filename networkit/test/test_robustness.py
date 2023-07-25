#!/usr/bin/env python3
import numpy as np
import os
import random
import unittest

import networkit as nk

class TestRobustness(unittest.TestCase):

	def setUp(self):
		pass

	def smallGraph():
		G = nk.Graph(6)
		edges = [(0,1), (0,2), (1,3), (2,3), (1,2), (1,4), (3,5), (4,5)]
		# G=

		#     0
		#    / \
		#   1 - 2
		#   |\ /
		#   4 3
		#   |/
		#   5

		for (u, v) in edges:
			G.addEdge(u,v)
		return G

	def testStGreedy_GRIP(self):
		nk.engineering.setSeed(1, True)
		random.seed(1)
		for directed in [True, False]:
			for weighted in [True, False]:
				g = nk.generators.ErdosRenyiGenerator(100, 0.15, directed).generate()
				if weighted:
					g = nk.graphtools.toWeighted(g)
					g.forEdges(lambda u, v, ew, eid: g.setWeight(u, v, random.random()))
				apsp = nk.distance.APSP(g)
				apsp.run()
				listDistances = apsp.getDistances()
				arrayDistances = apsp.getDistances(asarray=True)
				self.assertIsInstance(listDistances, list)
				self.assertIsInstance(arrayDistances, np.ndarray)
				np.testing.assert_allclose(listDistances, arrayDistances)
	