networkit_add_module(robustness
    SubmodularGreedy.cpp
	StochasticGreedy.cpp
	StGreedy.cpp
	SimplStoch.cpp
	RobustnessGreedy.cpp
    )

networkit_module_link_modules(robustness
    algebraic auxiliary coarsening community components distance graph io structures)

add_subdirectory(test)
