from networkx.classes.graph import Graph
from zquantum.core.graph import generate_random_regular_graph, save_graph
from zquantum.qaoa.problems import solve_maxcut_by_exhaustive_search
import time

# Target values:
# 10 nodes: -13
# 20 nodes – -26


def main():
    target_costs = {10: -13, 12: -16, 20: -26}
    graph_size = 14
    max_n_graphs = 25
    graph_counter = 0
    for i in range(1000):
        start_time = time.time()
        graph = generate_random_regular_graph(graph_size, 3)
        cut_value, solution = solve_maxcut_by_exhaustive_search(graph)
        print(i, cut_value, time.time() - start_time)
        if cut_value == target_costs[graph_size]:
            save_graph(graph, f"graph_{graph_size}_{graph_counter}.json")
            graph_counter += 1
            if graph_counter == max_n_graphs:
                break


def generate_regular_graph_with_given_cost(number_of_nodes, target_cost, n_tries):
    for i in range(n_tries):
        start_time = time.time()
        graph = generate_random_regular_graph(number_of_nodes, 3)
        cut_value, solution = solve_maxcut_by_exhaustive_search(graph)
        print(i, cut_value, time.time() - start_time)
        if cut_value == target_cost:
            save_graph(graph, f"graph.json")
            break


if __name__ == "__main__":
    main()
