from functools import partial
from openfermion.ops.operators.ising_operator import IsingOperator
from openfermion.ops.operators.qubit_operator import QubitOperator
from zquantum.qaoa.ansatzes.farhi_ansatz import QAOAFarhiAnsatz
from zquantum.qaoa.estimators import CvarEstimator
from zquantum.optimizers.scipy_optimizer import ScipyOptimizer
from zquantum.core.estimation import (
    estimate_expectation_values_by_averaging,
    allocate_shots_uniformly,
    calculate_exact_expectation_values,
)
from zquantum.core.cost_function import AnsatzBasedCostFunction
from qequlacs.simulator import QulacsSimulator
from zquantum.core.openfermion import change_operator_type
import numpy as np
import matplotlib.pyplot as plt


def main():
    # TODO: load from data of interest
    cost_hamiltonian = QubitOperator("[Z0]")
    cost_hamiltonian = change_operator_type(cost_hamiltonian, IsingOperator)
    number_of_layers = 2

    ansatz = QAOAFarhiAnsatz(number_of_layers, cost_hamiltonian=cost_hamiltonian)
    backend = QulacsSimulator()

    # TODO: load initial parameters from data
    initial_params = np.random.uniform(-np.pi, np.pi, ansatz.number_of_params)

    # TODO: select method of choice
    estimation_method = calculate_exact_expectation_values
    # estimation_method = estimate_expectation_values_by_averaging
    # estimation_method = CvarEstimator(alpha=0.3)

    estimation_preprocessors = []
    # TODO: this is optional, if we need sampling for our calculations,
    # calculate_exact_expectation_values does not require it.
    # shot_allocation = partial(allocate_shots_uniformly, number_of_shots=10000)
    # estimation_preprocessors = [shot_allocation]

    # TODO: select optimizer of choice
    # from zquantum.optimizers.basin_hopping import BasinHoppingOptimizer
    # optimizer = BasinHoppingOptimizer()
    # from zquantum.optimizers.cma_es_optimizer import CMAESOptimizer
    # optimizer = CMAESOptimizer(sigma_0=0.8)
    optimizer = ScipyOptimizer(method="L-BFGS-B")

    cost_function = AnsatzBasedCostFunction(
        cost_hamiltonian, ansatz, backend, estimation_method, estimation_preprocessors
    )

    # When
    opt_results = optimizer.minimize(cost_function, initial_params, keep_history=True)
    print("Optimal cost function value:", opt_results.opt_value)

    values_history = [entry.value for entry in opt_results.history]

    plt.plot(values_history)
    plt.show()


if __name__ == "__main__":
    main()