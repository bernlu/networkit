# distutils: language=c++

from libcpp.vector cimport vector
from libcpp cimport bool as bool_t

from typing import List, Optional

from .base cimport _Algorithm, Algorithm
from .dynbase cimport _DynAlgorithm
from .dynbase import DynAlgorithm
from .dynamics cimport _GraphEvent, GraphEvent
from .graph cimport _Graph, Graph, Edge
from .structures cimport count, index, node


cdef extern from "<optional>" namespace "std" nogil:
	cdef cppclass optional[double]:
		optional()
		optional& operator=[double](double&)
	
	optional[double] make_optional[double](double)

cdef extern from "<networkit/Globals.hpp>" namespace "NetworKit":

	index _none "NetworKit::none"

none = _none


cdef extern from "<networkit/robustness/RobustnessGreedy.hpp>" namespace "NetworKit::RobustnessGreedy":
	cdef enum _Problem "NetworKit::RobustnessGreedy::Problem":
		GLOBAL_IMPROVEMENT,
		LOCAL_IMPROVEMENT,
		GLOBAL_REDUCTION,

cdef extern from "<networkit/robustness/RobustnessGreedy.hpp>" namespace "NetworKit::RobustnessGreedy":
	cdef enum _Metric "NetworKit::RobustnessGreedy::Metric":
		RESISTANCE,
		FOREST,
		AUTOMATIC
	
cdef extern from "<networkit/robustness/RobustnessGreedy.hpp>":
	cdef cppclass _RobustnessGreedy "NetworKit::RobustnessGreedy" (_Algorithm):
		void resetFocus(node) except +
		double getResultValue() except +
		vector[Edge] getResultItems() except +

class RobustnessProblem:
		GLOBAL_IMPROVEMENT = _Problem.GLOBAL_IMPROVEMENT
		LOCAL_IMPROVEMENT = _Problem.LOCAL_IMPROVEMENT
		GLOBAL_REDUCTION = _Problem.GLOBAL_REDUCTION

class RobustnessMetric:
	RESISTANCE = _Metric.RESISTANCE
	FOREST = _Metric.FOREST
	AUTOMATIC = _Metric.AUTOMATIC

cdef class RobustnessGreedy(Algorithm):
	""" 
	RobustnessGreedy

	Abstract Base Class for different greedy algorithms to solve graph robustness problems.
	"""
	def __init__(self, *args, **kwargs):
		if type(self) == RobustnessGreedy:
			raise RuntimeError("Error, you may not use RobustnessGreedy directly, use a sub-class instead")

	def resetFocus(self, node focusNode):
		"""
		resetFocus(focusNode)

		Resets the algorithm and sets a new focus for local robustness problems.

		Parameters
		----------
		focusNode : int
			Node index.
		"""
		(<_RobustnessGreedy*>(self._this)).resetFocus(focusNode)

	def getResultValue(self) -> float:
		""" 
		getResultValue()

		Returns the sum of all gains of edges picked.

		Returns
		-------
		float
			The total gain.
		"""
		return (<_RobustnessGreedy*>(self._this)).getResultValue()
	
	def getResultItems(self) -> List[(int, int)]:
		""" 
		getResultValue()

		Returns a list of all picked edges.

		Returns
		-------
		List[(int, int)]
			List of all picked edges.
		"""
		result = (<_RobustnessGreedy*>(self._this)).getResultItems()
		return [(r.u, r.v) for r in result]


cdef extern from "<networkit/robustness/StGreedy.hpp>":
	cdef cppclass _StGreedy "NetworKit::StGreedy" (_RobustnessGreedy):
		_StGreedy(_Graph, count, _Problem, _Metric, node) except +

cdef class StGreedy(RobustnessGreedy):
	""" 
	StGreedy(G, k, robustnessProblem, metric=None, focusNode = none)

	StGreedy algorithm for the graph robustness problem.
	The algorithm computes the laplacian pseudoinverse and uses the submodular greedy algorithm to pick k edges.
	
	Currently, three problem types are supported:
	- networkit.robustness.RobustnessProblem.GLOBAL_IMPROVEMENT - maximize the robustness of G by adding k edges to G.
	- networkit.robustness.RobustnessProblem.LOCAL_IMPROVEMENT	- maximize the robustness of G by adding k edges to G. All edges have to be adjacent to the :code:`focusNode`
	- networkit.robustness.RobustnessProblem.GLOBAL_REDUCTION 	- minimize the robustness of G by removing k edges from G.

	Two types of robustness metric are supported:
	- networkit.robustness.RobustnessMetric.RESISTANCE 	- uses the effective resistance r(u,v)
	- networkit.robustness.RobustnessMetric.FOREST		- uses the forest distance f(u,v)
	By default, resistance is used for improvement problems and forest distance for reduction problems.

	Parameters
	----------
	G : networkit.Graph
		Input graph (undirected).
	k : int
		Number of edges to add/remove to/from G.
	robustnessProblem : networkit.robustness.RobustnessProblem
		The robustness problem to solve
	metric : networkit.robustness.RobustnessMetric, optional
		Metric for robustness computation. Default: depends on :code:`robustnessProblem`
	focusNode : node, optional
		Node to which all edges have to be adjacent. Only used for the LOCAL_IMPROVEMENT problem type.
	"""

	def __cinit__(self, Graph G, k, robustnessProblem, metric = RobustnessMetric.AUTOMATIC, focusNode = none):
		self._this = new _StGreedy(G._this, k, robustnessProblem, metric, focusNode)


cdef extern from "<networkit/robustness/SimplStoch.hpp>":
	cdef cppclass _SimplStoch "NetworKit::SimplStoch" (_RobustnessGreedy):
		_SimplStoch(_Graph, count, _Problem, double, bool_t, optional[double], _Metric, node) except +

cdef class SimplStoch(RobustnessGreedy):
	""" 
	SimplStoch(G, k, robustnessProblem, epsilon, useJLT=False, solverEpsilon=None, metric=None, focusNode = none)

	SimplStoch algorithm for the graph robustness problem.
	The algorithm computes the laplacian pseudoinverse and uses the stochastic greedy algorithm to pick k edges.
	Optionally, this algorithm can approximate the robustness using the Johnson Lindenstrauss Lemma instead of the true laplacian pseudoinverse.

	Currently, three problem types are supported:
	- networkit.robustness.RobustnessProblem.GLOBAL_IMPROVEMENT - maximize the robustness of G by adding k edges to G.
	- networkit.robustness.RobustnessProblem.LOCAL_IMPROVEMENT	- maximize the robustness of G by adding k edges to G. All edges have to be adjacent to the :code:`focusNode`
	- networkit.robustness.RobustnessProblem.GLOBAL_REDUCTION 	- minimize the robustness of G by removing k edges from G.

	Two types of robustness metric are supported:
	- networkit.robustness.RobustnessMetric.RESISTANCE 	- uses the effective resistance r(u,v)
	- networkit.robustness.RobustnessMetric.FOREST		- uses the forest distance f(u,v)
	By default, resistance is used for improvement problems and forest distance for reduction problems.

	Parameters
	----------
	G : networkit.Graph
		Input graph (undirected).
	k : int
		Number of edges to add/remove to/from G.
	robustnessProblem : networkit.robustness.RobustnessProblem
		The robustness problem to solve
	epsilon : float
		Accuracy parameter for the stochastic greedy algorithm
	useJLT : bool
		Choice of approximating the metric using Johnson Lindenstrauss. Default: False
	solverEpsilon : float, optional
		Accuracy parameter for the JLT approximation. Default: 0.55
	metric : networkit.robustness.RobustnessMetric, optional
		Metric for robustness computation. Default: depends on :code:`robustnessProblem`
	focusNode : node, optional
		Node to which all edges have to be adjacent. Only used for the LOCAL_IMPROVEMENT problem type.
	"""
	def __cinit__(self, Graph G, k: int, robustnessProblem, epsilon: float, useJLT: bool = False, solverEpsilon: Optional[float] = None, metric = RobustnessMetric.AUTOMATIC, focusNode = none):
		cdef optional[double] _solverEpsilon
		if solverEpsilon is not None:
			_solverEpsilon = make_optional[double](solverEpsilon)
		self._this = new _SimplStoch(G._this, k, robustnessProblem, epsilon, useJLT, _solverEpsilon, metric, focusNode)


cdef extern from "<networkit/robustness/ColStoch.hpp>":
	cdef cppclass _ColStoch "NetworKit::ColStoch" (_RobustnessGreedy):
		_ColStoch(_Graph, count, _Problem, double, double, bool_t, optional[double], _Metric, node) except +

cdef class ColStoch(RobustnessGreedy):
	""" 
	ColStoch(G, k, robustnessProblem, epsilon, diagEpsilon=10, useJLT=False, solverEpsilon=None, metric=None, focusNode = none)

	ColStoch algorithm for the graph robustness problem.
	The algorithm computes the laplacian pseudoinverse using lazy evaluation and uses the stochastic greedy algorithm to pick k edges.
	ColStoch applies a heuristic to the nodes and only considers edges between nodes with high[low] centrality [depending on the robustness problem]
	Optionally, this algorithm can approximate the robustness using the Johnson Lindenstrauss Lemma instead of the true laplacian pseudoinverse.

	Currently, three problem types are supported:
	- networkit.robustness.RobustnessProblem.GLOBAL_IMPROVEMENT - maximize the robustness of G by adding k edges to G.
	- networkit.robustness.RobustnessProblem.LOCAL_IMPROVEMENT	- maximize the robustness of G by adding k edges to G. All edges have to be adjacent to the :code:`focusNode`
	- networkit.robustness.RobustnessProblem.GLOBAL_REDUCTION 	- minimize the robustness of G by removing k edges from G.

	Two types of robustness metric are supported:
	- networkit.robustness.RobustnessMetric.RESISTANCE 	- uses the effective resistance r(u,v)
	- networkit.robustness.RobustnessMetric.FOREST		- uses the forest distance f(u,v)
	By default, resistance is used for improvement problems and forest distance for reduction problems.

	Parameters
	----------
	G : networkit.Graph
		Input graph (undirected).
	k : int
		Number of edges to add/remove to/from G.
	robustnessProblem : networkit.robustness.RobustnessProblem
		The robustness problem to solve
	epsilon : float
		Accuracy parameter for the stochastic greedy algorithm
	diagEpsilon : float, optional
		Accuracy parameter for the diagonal approximation heuristic. Default: 10
	useJLT : bool
		Choice of approximating the metric using Johnson Lindenstrauss. Default: False
	solverEpsilon : float, optional
		Accuracy parameter for the JLT approximation. Default: 0.55
	metric : networkit.robustness.RobustnessMetric, optional
		Metric for robustness computation. Default: depends on :code:`robustnessProblem`
	focusNode : node, optional
		Node to which all edges have to be adjacent. Only used for the LOCAL_IMPROVEMENT problem type.
	"""
	def __cinit__(self, Graph G, k: int, robustnessProblem, epsilon: float, diagEpsilon: float = 10, useJLT: bool = False, solverEpsilon: Optional[float] = None, metric = RobustnessMetric.AUTOMATIC, focusNode = none):
		cdef optional[double] _solverEpsilon
		if solverEpsilon is not None:
			_solverEpsilon = make_optional[double](solverEpsilon)
		self._this = new _ColStoch(G._this, k, robustnessProblem, epsilon, diagEpsilon, useJLT, _solverEpsilon, metric, focusNode)


cdef extern from "<networkit/robustness/SpecStoch.hpp>":
	cdef cppclass _SpecStoch "NetworKit::SpecStoch" (_RobustnessGreedy):
		_SpecStoch(_Graph, count, _Problem, double, count, _Metric, node) except +

cdef class SpecStoch(RobustnessGreedy):
	""" 
	SpecStoch(G, k, robustnessProblem, epsilon, numberOfEigenpairs, metric=None, focusNode = none)

	SpecStoch algorithm for the graph robustness problem.
	The algorithm computes the laplacian pseudoinverse using a spectral low rank approximation and uses the stochastic greedy algorithm to pick k edges.
	
	Currently, three problem types are supported:
	- networkit.robustness.RobustnessProblem.GLOBAL_IMPROVEMENT - maximize the robustness of G by adding k edges to G.
	- networkit.robustness.RobustnessProblem.LOCAL_IMPROVEMENT	- maximize the robustness of G by adding k edges to G. All edges have to be adjacent to the :code:`focusNode`
	- networkit.robustness.RobustnessProblem.GLOBAL_REDUCTION 	- minimize the robustness of G by removing k edges from G.

	Two types of robustness metric are supported:
	- networkit.robustness.RobustnessMetric.RESISTANCE 	- uses the effective resistance r(u,v)
	- networkit.robustness.RobustnessMetric.FOREST		- uses the forest distance f(u,v)
	By default, resistance is used for improvement problems and forest distance for reduction problems.

	Parameters
	----------
	G : networkit.Graph
		Input graph (undirected).
	k : int
		Number of edges to add/remove to/from G.
	robustnessProblem : networkit.robustness.RobustnessProblem
		The robustness problem to solve
	epsilon : float
		Accuracy parameter for the stochastic greedy algorithm
	numberOfEigenpairs : int
		Number of eigenpairs to compute for the laplacian approximation
	metric : networkit.robustness.RobustnessMetric, optional
		Metric for robustness computation. Default: depends on :code:`robustnessProblem`
	focusNode : node, optional
		Node to which all edges have to be adjacent. Only used for the LOCAL_IMPROVEMENT problem type.
	"""
	def __cinit__(self, Graph G, k: int, robustnessProblem, epsilon: float, numberOfEigenpairs: int, metric = RobustnessMetric.AUTOMATIC, focusNode = none):
		self._this = new _SpecStoch(G._this, k, robustnessProblem, epsilon, numberOfEigenpairs,
		 metric, focusNode)




cdef extern from "<networkit/robustness/DynLaplacianInverseSolver.hpp>":
	
	cdef cppclass _DynLaplacianInverseSolver "NetworKit::DynLaplacianInverseSolver" (_Algorithm, _DynAlgorithm):
		double totalResistanceDifference(_GraphEvent) except +

cdef class DynLaplacianInverseSolver(Algorithm, DynAlgorithm):
	def __init__(self, *args, **kwargs):
		if type(self) == DynLaplacianInverseSolver:
			raise RuntimeError("Error, you may not use RobustnessGreedy directly, use a sub-class instead")
	def totalResistanceDifference(self, ev: GraphEvent) -> float:
		return (<_DynLaplacianInverseSolver*>(self._this)).totalResistanceDifference(_GraphEvent(ev.type, ev.u, ev.v, ev.w))


cdef extern from "<networkit/robustness/DynFullLaplacianInverseSolver.hpp>":
	
	cdef cppclass _DynFullLaplacianInverseSolver "NetworKit::DynFullLaplacianInverseSolver" (_Algorithm, _DynAlgorithm):
		_DynFullLaplacianInverseSolver(_Graph) except +

cdef class DynFullLaplacianInverseSolver(DynLaplacianInverseSolver):
	def __cinit__(self, Graph G):
		self._this = new _DynFullLaplacianInverseSolver(G._this)

cdef extern from "<networkit/robustness/DynLazyLaplacianInverseSolver.hpp>":
	
	cdef cppclass _DynLazyLaplacianInverseSolver "NetworKit::DynLazyLaplacianInverseSolver" (_Algorithm, _DynAlgorithm):
		_DynLazyLaplacianInverseSolver(_Graph, double) except +

cdef class DynLazyLaplacianInverseSolver(DynLaplacianInverseSolver):
	def __cinit__(self, Graph G, double tolerance):
		self._this = new _DynLazyLaplacianInverseSolver(G._this, tolerance)