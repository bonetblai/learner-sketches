import logging

from pathlib import Path
from termcolor import colored
from typing import Set, List, MutableSet, Dict

import pymimir as mm
from dlplan.policy import PolicyMinimizer

from .src.exit_codes import ExitCode
from .src.iteration import EncodingType, ASPFactory, ClingoExitCode, IterationData, LearningStatistics, LTLSketch, LTLD2sepDlplanPolicyFactory, compute_feature_pool, compute_per_state_feature_valuations, compute_state_pair_equivalences, compute_tuple_graph_equivalences, minimize_tuple_graph_equivalences
from .src.util import Timer, create_experiment_workspace, change_working_directory, write_file, change_dir, memory_usage, add_console_handler, print_separation_line
from .src.preprocessing import InstanceData, PreprocessingData, StateFinder, compute_instance_datas, compute_tuple_graphs
from .src.iteration import make_dfa, DFA


def _compute_smallest_unsolved_instance(
        scc_initial_states: Set[int],
        scc_final_states: Set[int],
        preprocessing_data: PreprocessingData,
        iteration_data: IterationData,
        selected_instance_datas: List[InstanceData],
        sketch: LTLSketch,
        enable_goal_separating_features: bool):
    for instance_data in selected_instance_datas:
        if not sketch.solves(scc_initial_states, scc_final_states, preprocessing_data, iteration_data, instance_data, enable_goal_separating_features):
            return instance_data
    return None


def ltl_learn_sketch_for_problem_class(
    domain_filepath: Path,
    problems_directory: Path,
    workspace: Path,
    width: int,
    disable_closed_Q: bool = False,
    max_num_states_per_instance: int = 10000,
    max_time_per_instance: int = 10,
    encoding_type: EncodingType = EncodingType.D2,
    max_num_rules: int = 4,
    enable_goal_separating_features: bool = True,
    disable_feature_generation: bool = False,
    enable_incomplete_feature_pruning: bool = False,
    concept_complexity_limit: int = 9,
    role_complexity_limit: int = 9,
    boolean_complexity_limit: int = 9,
    count_numerical_complexity_limit: int = 9,
    distance_numerical_complexity_limit: int = 9,
    feature_limit: int = 1000000,
    additional_booleans: List[str] = None,
    additional_numericals: List[str] = None,
    enable_dump_files: bool = False,
    ppltl_goal_str: str = "",
    ltl_labels: Path = None,
):
    assert encoding_type == EncodingType.D2_LTL

    # Setup arguments and workspace
    if additional_booleans is None:
        additional_booleans = []
    if additional_numericals is None:
        additional_numericals = []
    instance_filepaths = list(problems_directory.iterdir())
    add_console_handler(logging.getLogger(), logging.INFO)
    create_experiment_workspace(workspace)
    change_working_directory(workspace)

    # Keep track of time
    total_timer = Timer()
    preprocessing_timer = Timer()
    asp_timer = Timer(stopped=True)
    verification_timer = Timer(stopped=True)

    # Generate data
    with change_dir("input"):
        logging.info(colored("Constructing InstanceDatas...", "blue"))
        domain_data, instance_datas, num_ss_states, num_gfa_states = compute_instance_datas(domain_filepath, instance_filepaths, disable_closed_Q, max_num_states_per_instance, max_time_per_instance, enable_dump_files)
        if instance_datas is None:
            raise Exception("Failed to create InstanceDatas.")

        state_finder = StateFinder(domain_data, instance_datas)

        logging.info(colored("Initializing TupleGraphs...", "blue"))
        gfa_state_id_to_tuple_graph: Dict[int, mm.TupleGraph] = compute_tuple_graphs(domain_data, instance_datas, state_finder, width, enable_dump_files)

    preprocessing_data = PreprocessingData(domain_data, instance_datas, state_finder, gfa_state_id_to_tuple_graph)
    preprocessing_timer.stop()

    # Generate DFA from PPLTL spec (if any)
    ppltl_dfa = make_dfa(ppltl_goal_str)
    #print(ppltl_dfa)
    logging.info(f'dfa: transitions: {[(q1, q2, str(label)) for (q1, q2, label) in ppltl_dfa.transitions]}')
    logging.info(f'dfa: initial: {ppltl_dfa.initial}')
    logging.info(f'dfa: accepting: {ppltl_dfa.accepting}')
    logging.info(f'dfa: labels: {[str(label) for label in ppltl_dfa.labels]}')
    logging.info(f'dfa: alphabet: {ppltl_dfa.alphabet}')
    if ltl_labels is not None:
        logging.info(f"Reading label denotations from {ltl_labels}")
        with ltl_labels.open("r") as fd:
            denotations = [denotation.strip() for denotation in fd.readlines()]
        syntactic_element_factory = preprocessing_data.domain_data.syntactic_element_factory
        denotations = DFA.parse_denotations(denotations, syntactic_element_factory)

        if len(denotations) == len(ppltl_dfa.alphabet):
            ppltl_dfa.set_denotations(denotations)
        else:
            logging.error(f"Error: insufficient LTL labels; expecting {len(ppltl_dfa.alphabet)} denotation(s) but got {len(denotations)}")
            return
    else:
        logging.error(f"Error: LTL labels must be supplied when using --ppltl-goal; use --ltl-labels")
        return

    logging.info(f'Training: instances={[instance_data.mimir_ss.get_problem().get_filepath() for instance_data in instance_datas]}')

    # Learn sketch in stages using SCCs of LTL automata
    sketch = LTLSketch(None, width, ppltl_dfa)
    for scc_index, scc in ppltl_dfa.sccs.items():
        scc_initial_states = ppltl_dfa.scc_initial_states[scc_index]
        scc_final_states = ppltl_dfa.scc_exit_points[scc_index]
        logging.info(f"Working SCC: index={scc_index}, scc={scc}, initial={scc_initial_states}, final={scc_final_states}")
        if scc.issubset(ppltl_dfa.accepting): continue

        iteration_data = IterationData()
        with change_dir("iterations"):
            i = 0
            with change_dir(str(i), enable=enable_dump_files):
                selected_instance_idxs = [0]
                create_experiment_workspace(workspace)
                while True:
                    logging.info(colored(f"Iteration: {i}", "red"))

                    preprocessing_timer.resume()
                    iteration_data.instance_datas = [preprocessing_data.instance_datas[subproblem_idx] for subproblem_idx in selected_instance_idxs]
                    for instance_data in iteration_data.instance_datas:
                        write_file(f"dlplan_ss_{instance_data.idx}.dot", instance_data.dlplan_ss.to_dot(1))
                        write_file(f"mimir_ss_{instance_data.idx}.dot", str(instance_data.mimir_ss))
                        write_file(f"mimir_fa_{instance_data.idx}.dot", str(instance_data.gfa.get_abstractions()[instance_data.idx]))
                        write_file(f"mimir_gfa_{instance_data.idx}.dot", str(instance_data.gfa))
                        logging.info(f"    id: {instance_data.idx}, problem_filepath: {instance_data.mimir_ss.get_problem().get_filepath()}, num_states: {instance_data.mimir_ss.get_num_states()}, num_state_equivalences: {instance_data.gfa.get_num_states()}")

                    logging.info(colored("Initialize global faithful abstract states...", "blue"))
                    gfa_states : MutableSet[mm.GlobalFaithfulAbstractState] = set()
                    for instance_data in iteration_data.instance_datas:
                        gfa_states.update(instance_data.gfa.get_states())
                    iteration_data.gfa_states = list(gfa_states)

                    logging.info(colored("Initializing DomainFeatureData...", "blue"))
                    iteration_data.feature_pool = compute_feature_pool(
                        preprocessing_data,
                        iteration_data,
                        gfa_state_id_to_tuple_graph,
                        state_finder,
                        disable_feature_generation,
                        enable_incomplete_feature_pruning,
                        concept_complexity_limit,
                        role_complexity_limit,
                        boolean_complexity_limit,
                        count_numerical_complexity_limit,
                        distance_numerical_complexity_limit,
                        feature_limit,
                        additional_booleans,
                        additional_numericals)

                    logging.info(colored("Constructing PerStateFeatureValuations...", "blue"))
                    iteration_data.gfa_state_global_idx_to_feature_evaluations = compute_per_state_feature_valuations(preprocessing_data, iteration_data)

                    logging.info(colored("Constructing StatePairEquivalenceDatas...", "blue"))
                    iteration_data.state_pair_equivalences, iteration_data.gfa_state_global_idx_to_state_pair_equivalence = compute_state_pair_equivalences(preprocessing_data, iteration_data)

                    logging.info(colored("Constructing TupleGraphEquivalences...", "blue"))
                    iteration_data.gfa_state_global_idx_to_tuple_graph_equivalence = compute_tuple_graph_equivalences(preprocessing_data, iteration_data)

                    logging.info(colored("Minimizing TupleGraphEquivalences...", "blue"))
                    minimize_tuple_graph_equivalences(preprocessing_data, iteration_data)
                    preprocessing_timer.stop()

                    asp_timer.resume()

                    d2_facts = set()
                    symbols = None
                    j = 0
                    while True:
                        asp_factory = ASPFactory(encoding_type, enable_goal_separating_features, max_num_rules)
                        facts = asp_factory.make_facts(preprocessing_data, iteration_data, ppltl_dfa, scc, scc_initial_states, scc_final_states)
                        if j == 0:
                            #d2_facts.update(asp_factory.make_initial_d2_facts(preprocessing_data, iteration_data))
                            d2_facts.update(asp_factory.make_initial_d2_facts_alt(preprocessing_data, iteration_data))
                            logging.info(f"Number of initial D2 facts: {len(d2_facts)}")
                        elif j > 0:
                            unsatisfied_d2_facts = asp_factory.make_unsatisfied_d2_facts(iteration_data, symbols)
                            d2_facts.update(unsatisfied_d2_facts)
                            logging.info(f"Number of unsatisfied D2 facts: {len(unsatisfied_d2_facts)}")
                            assert len(unsatisfied_d2_facts) > 0
                        logging.info(f"Number of D2 facts: {len(d2_facts)} of {len(iteration_data.state_pair_equivalences) ** 2}")
                        facts.extend(list(d2_facts))

                        logging.info(colored("Grounding Logic Program...", "blue"))
                        asp_factory.ground(facts)

                        logging.info(colored("Solving Logic Program...", "blue"))
                        symbols, returncode = asp_factory.solve()

                        if returncode in [ClingoExitCode.UNSATISFIABLE, ClingoExitCode.EXHAUSTED]:
                            logging.info(colored("ASP is unsatisfiable!", "red"))
                            logging.info(colored(f"No sketch of width {width} exists that solves all instances!", "red"))
                            exit(ExitCode.UNSOLVABLE)
                        elif returncode == ClingoExitCode.UNKNOWN:
                            logging.info(colored("ASP solving throws unknown error!", "red"))
                            exit(ExitCode.UNKNOWN)
                        elif returncode == ClingoExitCode.INTERRUPTED:
                            logging.info(colored("ASP solving iterrupted!", "red"))
                            exit(ExitCode.INTERRUPTED)

                        asp_factory.print_statistics()

                        dlplan_policy = LTLD2sepDlplanPolicyFactory().make_dlplan_policy_from_answer_set(symbols, preprocessing_data, iteration_data)
                        sketch.replace(dlplan_policy)
                        logging.info("Learned the following sketch:")
                        sketch.print()
                        if _compute_smallest_unsolved_instance(scc_initial_states, scc_final_states, preprocessing_data, iteration_data, iteration_data.instance_datas, sketch, enable_goal_separating_features) is None:
                            # Stop adding D2-separation constraints
                            # if sketch solves all training instances
                            break
                        j += 1

                    asp_timer.stop()

                    verification_timer.resume()
                    logging.info(colored("Verifying learned sketch...", "blue"))
                    assert _compute_smallest_unsolved_instance(scc_initial_states, scc_final_states, preprocessing_data, iteration_data, iteration_data.instance_datas, sketch, enable_goal_separating_features) is None
                    smallest_unsolved_instance = _compute_smallest_unsolved_instance(scc_initial_states, scc_final_states, preprocessing_data, iteration_data, instance_datas, sketch, enable_goal_separating_features)
                    verification_timer.stop()

                    if smallest_unsolved_instance is None:
                        logging.info(colored("Sketch solves all instances!", "red"))
                        break
                    else:
                        if smallest_unsolved_instance.idx > max(selected_instance_idxs):
                            selected_instance_idxs = [smallest_unsolved_instance.idx]
                        else:
                            selected_instance_idxs.append(smallest_unsolved_instance.idx)
                        logging.info(f"Smallest unsolved instance: {smallest_unsolved_instance.mimir_ss.get_problem().get_filepath()} (idx={smallest_unsolved_instance.idx})")
                        logging.info(f"Selected instances {selected_instance_idxs}")
                    i += 1

    total_timer.stop()

    # Check that assembled policy solves whole problem
    #assert _compute_smallest_unsolved_instance(scc_initial_states, scc_final_states, preprocessing_data, iteration_data, iteration_data.instance_datas, sketch, enable_goal_separating_features) is None

    # Output the result
    with change_dir("output"):
        print_separation_line()
        logging.info(colored("Summary:", "green"))

        learning_statistics = LearningStatistics(
            num_training_instances=len(instance_datas),
            num_selected_training_instances=len(iteration_data.instance_datas),
            num_states_in_selected_training_instances=sum(instance_data.gfa.get_num_states() for instance_data in iteration_data.instance_datas),
            num_states_in_complete_selected_training_instances=sum(instance_data.mimir_ss.get_num_states() for instance_data in iteration_data.instance_datas),
            num_features_in_pool=len(iteration_data.feature_pool))
        learning_statistics.print()
        print_separation_line()

        logging.info("Resulting sketch:")
        sketch.print()
        print_separation_line()

        logging.info("Resulting minimized sketch:")
        sketch_minimized = sketch.minimize(domain_data.policy_builder)
        sketch_minimized.print()
        print_separation_line()

        create_experiment_workspace(workspace / "output")
        #write_file(f"sketch_{width}.txt", str(sketch.dlplan_policy))
        #write_file(f"sketch_minimized_{width}.txt", str(sketch_minimized.dlplan_policy))

        print_separation_line()
        logging.info(f"Preprocessing time: {int(preprocessing_timer.get_elapsed_sec()) + 1} seconds.")
        logging.info(f"ASP time: {int(asp_timer.get_elapsed_sec()) + 1} seconds.")
        logging.info(f"Verification time: {int(verification_timer.get_elapsed_sec()) + 1} seconds.")
        logging.info(f"Total time: {int(total_timer.get_elapsed_sec()) + 1} seconds.")
        logging.info(f"Total memory: {int(memory_usage() / 1024)} GiB.")
        logging.info(f"Num states in training data before symmetry pruning: {num_ss_states}")
        logging.info(f"Num states in training data after symmetry pruning: {num_gfa_states}")
        print_separation_line()

        print(flush=True)
