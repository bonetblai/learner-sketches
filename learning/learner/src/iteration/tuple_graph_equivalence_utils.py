import logging

from typing import List, Dict, MutableSet
from collections import defaultdict

from .tuple_graph_equivalence import TupleGraphEquivalence
from .iteration_data import IterationData

from ..preprocessing import PreprocessingData


def compute_tuple_graph_equivalences(preprocessing_data: PreprocessingData,
                                     iteration_data: IterationData) -> None:
    """ Computes information for all subgoal states, tuples and rules over F.
    """
    num_nodes = 0

    gfa_state_id_to_tuple_graph_equivalence: Dict[int, TupleGraphEquivalence] = dict()

    for gfa_state in iteration_data.gfa_states:
        instance_idx = gfa_state.get_faithful_abstraction_index()
        instance_data = preprocessing_data.instance_datas[instance_idx]

        if instance_data.gfa.is_deadend_state(gfa_state.get_faithful_abstract_state_index()):
            continue

        gfa_state_global_idx = gfa_state.get_global_index()
        tuple_graph = preprocessing_data.gfa_state_global_idx_to_tuple_graph[gfa_state_global_idx]
        tuple_graph_vertices_by_distance = tuple_graph.get_vertices_grouped_by_distance()
        tuple_graph_states_by_distance = tuple_graph.get_states_grouped_by_distance()

        t_idx_to_r_idxs: Dict[int, MutableSet[int]] = dict()
        t_idx_to_distance: Dict[int, int] = dict()
        r_idx_to_deadend_distance: Dict[int, int] = dict()

        for s_distance, mimir_ss_states_prime in enumerate(tuple_graph_states_by_distance):
            for mimir_ss_state_prime in mimir_ss_states_prime:
                gfa_state_prime = preprocessing_data.state_finder.get_gfa_state_from_ss_state_idx(instance_idx, instance_data.mimir_ss.get_state_index(mimir_ss_state_prime))
                gfa_state_prime_global_idx = gfa_state_prime.get_global_index()
                instance_prime_idx = gfa_state_prime.get_faithful_abstraction_index()
                instance_data_prime = preprocessing_data.instance_datas[instance_prime_idx]
                gfa_state_prime_idx = gfa_state_prime.get_index()

                r_idx = iteration_data.gfa_state_global_idx_to_state_pair_equivalence[gfa_state_global_idx].subgoal_gfa_state_id_to_r_idx[gfa_state_prime_global_idx]

                if instance_data_prime.gfa.is_deadend_state(gfa_state_prime_idx):
                    r_idx_to_deadend_distance[r_idx] = min(r_idx_to_deadend_distance.get(r_idx, float("inf")), s_distance)

        for s_distance, tuple_vertex_group in enumerate(tuple_graph_vertices_by_distance):
            for tuple_vertex in tuple_vertex_group:
                t_idx = tuple_vertex.get_index()
                r_idxs = set()
                for mimir_ss_state_prime in tuple_vertex.get_states():
                    gfa_state_prime = preprocessing_data.state_finder.get_gfa_state_from_ss_state_idx(instance_idx, instance_data.mimir_ss.get_state_index(mimir_ss_state_prime))
                    gfa_state_prime_global_idx = gfa_state_prime.get_global_index()
                    r_idx = iteration_data.gfa_state_global_idx_to_state_pair_equivalence[gfa_state_global_idx].subgoal_gfa_state_id_to_r_idx[gfa_state_prime_global_idx]
                    r_idxs.add(r_idx)
                t_idx_to_distance[t_idx] = s_distance
                t_idx_to_r_idxs[t_idx] = r_idxs
                num_nodes += 1

        gfa_state_id_to_tuple_graph_equivalence[gfa_state_global_idx] = TupleGraphEquivalence(t_idx_to_r_idxs, t_idx_to_distance, r_idx_to_deadend_distance)

    logging.info(f"Tuple graph equivalence construction statistics: num_nodes={num_nodes}")

    return gfa_state_id_to_tuple_graph_equivalence


def minimize_tuple_graph_equivalences(preprocessing_data: PreprocessingData,
                                      iteration_data: IterationData):
    num_kept_nodes = 0
    num_orig_nodes = 0

    for gfa_state in iteration_data.gfa_states:
        instance_idx = gfa_state.get_faithful_abstraction_index()
        instance_data = preprocessing_data.instance_datas[instance_idx]

        if instance_data.gfa.is_deadend_state(gfa_state.get_faithful_abstract_state_index()):
            continue

        gfa_state_global_idx = gfa_state.get_global_index()
        tuple_graph = preprocessing_data.gfa_state_global_idx_to_tuple_graph[gfa_state_global_idx]
        tuple_graph_equivalence = iteration_data.gfa_state_global_idx_to_tuple_graph_equivalence[gfa_state_global_idx]
        # compute order
        order = defaultdict(set)
        for t_idx_1 in tuple_graph_equivalence.t_idx_to_r_idxs.keys():
            r_idxs_1 = frozenset(tuple_graph_equivalence.t_idx_to_r_idxs[t_idx_1])
            for t_idx_2 in tuple_graph_equivalence.t_idx_to_r_idxs.keys():
                if t_idx_1 == t_idx_2:
                    continue
                r_idxs_2 = frozenset(tuple_graph_equivalence.t_idx_to_r_idxs[t_idx_2])
                if r_idxs_1.issubset(r_idxs_2) and r_idxs_1 != r_idxs_2:
                    # t_2 gets dominated by t_1
                    order[t_idx_2].add(t_idx_1)
        # select tuple nodes according to order
        selected_t_idxs = set()
        representative_r_idxs = set()
        for tuple_vertex_group in tuple_graph.get_vertices_grouped_by_distance():
            for tuple_vertex in tuple_vertex_group:
                t_idx = tuple_vertex.get_index()
                r_idxs = frozenset(tuple_graph_equivalence.t_idx_to_r_idxs[t_idx])
                if order.get(t_idx, 0) != 0:
                    continue
                if r_idxs in representative_r_idxs:
                    continue
                representative_r_idxs.add(r_idxs)
                # found tuple with minimal number of rules
                selected_t_idxs.add(t_idx)

        # restrict to selected tuples
        t_idx_to_r_idxs: Dict[int, MutableSet[int]] = dict()
        t_idx_to_distance: Dict[int, int] = dict()
        for t_idx, r_idxs in tuple_graph_equivalence.t_idx_to_r_idxs.items():
            if t_idx in selected_t_idxs:
                t_idx_to_r_idxs[t_idx] = r_idxs
                num_kept_nodes += 1
            num_orig_nodes += 1
        for t_idx, distance in tuple_graph_equivalence.t_idx_to_distance.items():
            if t_idx in selected_t_idxs:
                t_idx_to_distance[t_idx] = distance

        iteration_data.gfa_state_global_idx_to_tuple_graph_equivalence[gfa_state_global_idx] = TupleGraphEquivalence(t_idx_to_r_idxs, t_idx_to_distance, tuple_graph_equivalence.r_idx_to_deadend_distance)

    logging.info(f"Tuple graph equivalence minimization statistics: num_orig_nodes= {num_orig_nodes}, num_kept_nodes={num_kept_nodes}")
