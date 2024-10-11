import logging
import math

from collections import defaultdict, deque
from termcolor import colored
from typing import Dict, Set, List, Deque, MutableSet, Tuple, Any

import pymimir as mm
import dlplan.core as dlplan_core
import dlplan.policy as dlplan_policy

from .iteration_data import IterationData
from ..preprocessing import PreprocessingData, InstanceData

from .ltl_base import DFA
from .ltl_policy import LTLPolicy


class LTLSketch:
    def __init__(self, ltl_policy: LTLPolicy, width: int, dfa: DFA):
        self.ltl_policy = ltl_policy
        self.width = width
        self.dfa = dfa
        assert self.width == 0, "Only LTL policies supported..."

    def _verify_bounded_width(self,
                              preprocessing_data: PreprocessingData,
                              iteration_data: IterationData,
                              instance_data: InstanceData,
                              require_optimal_width=False):
        """
        Performs forward search over R-reachable states.
        Initially, the R-reachable states are all initial states.
        For each R-reachable state there must be a satisfied subgoal tuple.
        If optimal width is required, we do not allow R-compatible states
        that are closer than the closest satisfied subgoal tuple.
        """
        instance_data_gfa_states = instance_data.gfa.get_states()

        # The queue contains pairs (Q, gfa_state_index).
        queue: Deque[Tuple[int, mm.GlobalFaithfulAbstractState]] = deque()
        visited: MutableSet[Tuple[int, mm.GlobalFaithfulAbstractState]] = set()
        # Dominik (25-07-2024): checked, use index.
        # Blai (08-10-2024): manually set initial state 0
        for gfa_state_idx in instance_data.initial_gfa_state_idxs:
            print(f"Initial: {gfa_state_idx}")
        #    queue.append(gfa_state_idx)
        #    visited.add(gfa_state_idx)
        for gfa_state_idx in [0]: # HACK: fix initial states, and initial dfa state (there could be an initial "automatic" dfa transition)
            gfa_state = instance_data_gfa_states[gfa_state_idx]
            dlplan_ss_initial_state = preprocessing_data.state_finder.get_dlplan_ss_state(gfa_state)
            initial_q = self.dfa.initial_state(dlplan_ss_initial_state, instance_data.denotations_caches)
            pair = (initial_q, gfa_state_idx)
            queue.append(pair)
            visited.add(pair)
            logging.debug(f"Queue: PUSH {pair} (initial)")
        # byproduct for acyclicity check
        subgoal_states_per_r_reachable_state: Dict[Tuple[int, int], Set[Tuple[int, int]]] = defaultdict(set)
        cur_instance_idx = instance_data.idx
        while queue:
            # Ensure that we do not reassign instance_data accidentally
            assert cur_instance_idx == instance_data.idx

            q_root, gfa_root_idx = queue.pop()
            logging.debug(f"Queue: POP {(q_root, gfa_root_idx)}, Q={queue}")
            gfa_root = instance_data_gfa_states[gfa_root_idx]
            gfa_root_global_idx = gfa_root.get_global_index()
            #print(f'Root: q_root={q_root}, gfa_root_idx={gfa_root_idx}, gfa_root_global_idx={gfa_root_global_idx}')

            # Dominik (25-07-2024): checked, use index.
            if instance_data.gfa.is_deadend_state(gfa_root_idx):
                logging.info(f"Deadend state is r_reachable: state={gfa_root_idx}")
                return False, []
            elif self.dfa.is_accepting_state(q_root):
                continue

            tuple_graph = preprocessing_data.gfa_state_global_idx_to_tuple_graph[gfa_root_global_idx]
            tuple_graph_vertices_by_distance = tuple_graph.get_vertices_grouped_by_distance()
            tuple_graph_states_by_distance = tuple_graph.get_states_grouped_by_distance()

            dlplan_ss_root = preprocessing_data.state_finder.get_dlplan_ss_state(gfa_root)

            ḧas_bounded_width = False
            min_compatible_distance = math.inf

            mapped_instance_idx = gfa_root.get_faithful_abstraction_index()
            mapped_instance_data = preprocessing_data.instance_datas[mapped_instance_idx]

            for s_distance, tuple_vertex_group in enumerate(tuple_graph_vertices_by_distance):
                for mimir_ss_state_prime in tuple_graph_states_by_distance[s_distance]:
                    mapped_gfa_state_prime = preprocessing_data.state_finder.get_gfa_state_from_ss_state_idx(mapped_instance_idx, mapped_instance_data.mimir_ss.get_state_index(mimir_ss_state_prime))
                    mapped_gfa_state_prime_global_idx = mapped_gfa_state_prime.get_global_index()
                    dlplan_ss_state_prime = preprocessing_data.state_finder.get_dlplan_ss_state(mapped_gfa_state_prime)

                    rule, q_prime = self.ltl_policy.evaluate(q_root, dlplan_ss_root, dlplan_ss_state_prime, instance_data.denotations_caches, self.dfa)
                    if rule is not None and q_prime is not None:
                        min_compatible_distance = min(min_compatible_distance, s_distance)
                        subgoal_states_per_r_reachable_state[(q_root, gfa_root_global_idx)].add((q_prime, mapped_gfa_state_prime_global_idx))
                        # Important: unmap the mapped gfa state to the original instance_data.gfa
                        # since the goal is to check whether sketch solves the given instance_data.
                        unmapped_gfa_state_prime_idx = instance_data.gfa.get_abstract_state_index(mapped_gfa_state_prime.get_global_index())
                        pair = (q_prime, unmapped_gfa_state_prime_idx)
                        if pair not in visited:
                            visited.add(pair)
                            queue.append(pair)
                            logging.debug(f"Queue: PUSH {pair}, Q={queue}")

                # Check whether there exists a subgoal tuple for which all underlying states are subgoal states
                found_subgoal_tuple = False
                for tuple_vertex in tuple_vertex_group:
                    is_subgoal_tuple = True
                    for mimir_ss_state_prime in tuple_vertex.get_states():
                        mapped_gfa_state_prime = preprocessing_data.state_finder.get_gfa_state_from_ss_state_idx(mapped_instance_idx, mapped_instance_data.mimir_ss.get_state_index(mimir_ss_state_prime))
                        mapped_gfa_state_prime_global_idx = mapped_gfa_state_prime.get_global_index()
                        dlplan_ss_state_prime = preprocessing_data.state_finder.get_dlplan_ss_state(mapped_gfa_state_prime)

                        rule, q_prime = self.ltl_policy.evaluate(q_root, dlplan_ss_root, dlplan_ss_state_prime, instance_data.denotations_caches, self.dfa)
                        if rule is not None and q_prime is not None:
                            min_compatible_distance = min(min_compatible_distance, s_distance)
                            subgoal_states_per_r_reachable_state[(q_root, gfa_root_global_idx)].add((q_prime, mapped_gfa_state_prime_global_idx))
                        else:
                            is_subgoal_tuple = False
                    if is_subgoal_tuple:
                        found_subgoal_tuple = True
                        break

                # Decide whether width is bounded or not
                if found_subgoal_tuple:
                    if require_optimal_width and min_compatible_distance < s_distance:
                        logging.info(colored(f"Optimal width disproven.", "red"))
                        logging.info(f"Min compatible distance: {min_compatible_distance}")
                        logging.info(f"Subgoal tuple distance: {s_distance}")
                        return False, []
                    else:
                        ḧas_bounded_width = True
                        break

            if not ḧas_bounded_width:
                logging.info(colored(f"Sketch FAILS to bound width of a state in {instance_data.instance_filepath}/{instance_data.idx}, source={str(dlplan_ss_root)}/{gfa_root_global_idx}", "red"))
                return False, []

        logging.info(colored(f"Sketch has BOUNDED WIDTH on {instance_data.mimir_ss.get_problem().get_filepath()}", "red"))
        return True, subgoal_states_per_r_reachable_state

    def _verify_acyclicity(self, instance_data: InstanceData, r_compatible_successors: Dict[Tuple[int, int], Set[Tuple[int, int]]]):
        """
        Returns True iff sketch is acyclic, i.e., no infinite trajectories s1,s2,... are possible.
        """
        for (q, s_idx), successors in r_compatible_successors.items():
            # The depth-first search is the iterative version where the current path is explicit in the stack.
            # https://en.wikipedia.org/wiki/Depth-first_search
            stack = [((q, s_idx), iter(successors))]
            pairs_on_path: Set[Tuple[int, int]] = {(q, s_idx)}
            frontier: Set[Tuple[int, int]] = set()  # the generated states, to ensure that they are only added once to the stack
            while stack:
                (source_q, source_idx), iterator = stack[-1]
                pairs_on_path.add((source_q, source_idx))
                try:
                    (target_q, gfa_target_idx) = next(iterator)
                    if self.dfa.is_accepting_state(target_q):
                        continue
                    if (target_q, gfa_target_idx) in pairs_on_path:
                        logging.info(colored(f"Sketch CYCLES on  {instance_data.mimir_ss.get_problem().get_filepath()}/{instance_data.idx}", "red"))
                        for pair in pairs_on_path:
                            print(f"{pair}")
                        print(f"{(target_q, gfa_target_idx)}")
                        return False
                    if (target_q, gfa_target_idx) not in frontier:
                        frontier.add((target_q, gfa_target_idx))
                        stack.append(((target_q, gfa_target_idx), iter(r_compatible_successors.get((target_q, gfa_target_idx), []))))
                except StopIteration:
                    pairs_on_path.discard((source_q, source_idx))
                    stack.pop(-1)

        logging.info(colored(f"Sketch is ACYCLIC on {instance_data.mimir_ss.get_problem().get_filepath()}", "red"))
        return True

    def _compute_state_b_values(self, booleans: List[dlplan_policy.NamedBoolean], numericals: List[dlplan_policy.NamedNumerical], state: dlplan_core.State, denotations_caches: dlplan_core.DenotationsCaches):
        return tuple([boolean.get_element().evaluate(state, denotations_caches) for boolean in booleans] + [numerical.get_element().evaluate(state, denotations_caches) > 0 for numerical in numericals])

    def _verify_goal_separating_features(self,
                                         preprocessing_data: PreprocessingData,
                                         iteration_data: IterationData,
                                         instance_data: InstanceData):
        """
        Returns True iff sketch features separate goal from nongoal states.
        """
        goal_b_values = set()
        nongoal_b_values = set()
        booleans = self.dlplan_policy.get_booleans()
        numericals = self.dlplan_policy.get_numericals()
        for gfa_state_idx, gfa_state in enumerate(instance_data.gfa.get_states()):
            new_instance_idx = gfa_state.get_abstraction_index()
            new_instance_data = preprocessing_data.instance_datas[new_instance_idx]
            dlplan_ss_state = preprocessing_data.state_finder.get_dlplan_ss_state(gfa_state)
            b_values = self._compute_state_b_values(booleans, numericals, dlplan_ss_state, new_instance_data.denotations_caches)
            separating = True
            if instance_data.gfa.is_goal_state(gfa_state_idx):
                goal_b_values.add(b_values)
                if b_values in nongoal_b_values:
                    separating = False
            else:
                nongoal_b_values.add(b_values)
                if b_values in goal_b_values:
                    separating = False
            if not separating:
                print("Features do not separate goals from non goals")
                print("Booleans:")
                print("State:", str(dlplan_ss_state))
                print("b_values:", b_values)
                return False
        return True

    def solves(self,
               preprocessing_data: PreprocessingData,
               iteration_data: IterationData,
               instance_data: InstanceData,
               enable_goal_separating_features: bool):
        """
        Returns True iff the sketch solves the instance, i.e.,
            (1) subproblems have bounded width,
            (2) sketch only classifies delta optimal state pairs as good,
            (3) sketch is acyclic, and
            (4) sketch features separate goals from nongoal states. """
        logging.info(colored(f"Verifying sketch solvability on {instance_data.mimir_ss.get_problem().get_filepath()}", "red"))
        bounded, subgoal_states_per_r_reachable_state = self._verify_bounded_width(preprocessing_data, iteration_data, instance_data)
        if not bounded:
            return False
        if enable_goal_separating_features:
            if not self._verify_goal_separating_features(preprocessing_data, iteration_data, instance_data):
                return False
        if not self._verify_acyclicity(instance_data, subgoal_states_per_r_reachable_state):
            return False

        logging.info(colored(f"Sketch SOLVES {instance_data.mimir_ss.get_problem().get_filepath()}", "red"))
        return True

    def minimize(self, policy_builder: Any):
        minimized_ltl_policy = self.ltl_policy.minimize(policy_builder)
        return LTLSketch(minimized_ltl_policy, self.width, self.dfa)

    def print(self):
        print(f"Numer of automata states: {len(self.ltl_policy.dlplan_policies)}")
        print(f"Numer of sketch rules: {[(f'q{q}',len(self.ltl_policy.dlplan_rules[q])) for q in self.ltl_policy.dlplan_policies]}")
        print(f"Number of selected features: {[(f'q{q}', len(self.ltl_policy.dlplan_policies[q].get_booleans()) + len(self.ltl_policy.dlplan_policies[q].get_numericals())) for q in self.ltl_policy.dlplan_policies]}")
        print(f"Maximum complexity of selected features: {[(f'q{q}', max([0] + [boolean.get_element().compute_complexity() for boolean in self.ltl_policy.dlplan_policies[q].get_booleans()] + [numerical.get_element().compute_complexity() for numerical in self.ltl_policy.dlplan_policies[q].get_numericals()])) for q in self.ltl_policy.dlplan_policies]}")
        self.ltl_policy.print()
