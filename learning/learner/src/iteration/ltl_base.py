import logging

import re
from collections import deque
from typing import Dict, Set, Tuple, List, abstractmethod

import dlplan.core as dlplan_core
from .feature_pool import Feature
from ltlf2dfa.parser.ltlf import LTLfParser
from ltlf2dfa.parser.ppltl import PPLTLParser

class Term(object):
    def __init__(self):
        pass

    @abstractmethod
    def get_atoms(self) -> Set[str]:
        pass

    @abstractmethod
    def is_consistent(self, interp: Dict[str, bool]) -> bool:
        pass

    @abstractmethod
    def __str__(self) -> str:
        pass

class Literal(Term):
    def __init__(self, atom: str, negated: bool = False):
        super().__init__()
        self.atom : str = atom
        self.negated : bool = negated

    def get_atoms(self) -> Set[str]:
        return set([self.atom])

    def is_consistent(self, interp: Dict[str, bool]) -> bool:
        value = interp[self.atom] != self.negated
        #print(f'is_consistent: interp={interp}, literal={str(self)}, value={value}')
        return interp[self.atom] != self.negated

    def __str__(self) -> str:
        return self.atom if not self.negated else f'~{self.atom}'

class Conjunction(Term):
    def __init__(self, literals: List[Literal] = []):
        super().__init__()
        self.literals : List[Literal] = literals

    def add(self, literal: Literal):
        self.literals.append(literal)

    def get_atoms(self) -> List[str]:
        list_atoms = [literal.get_atoms() for literal in self.literals]
        return set([atom for atoms in list_atoms for atom in atoms])

    def is_consistent(self, interp: Dict[str, bool]) -> bool:
        values = [literal.is_consistent(interp) for literal in self.literals]
        return False not in values

    def __str__(self) -> str:
        return 'true' if len(self.literals) == 0 else ' & '.join([str(literal) for literal in self.literals])

class DFA(object):
    def __init__(self, formula_str, formula):
        self.formula_str = formula_str
        self.formula = formula
        self.dfa_dot = self.formula.to_dfa()
        self.states, self.transitions, self.initial, self.accepting = self._parse_dfa(self.dfa_dot)
        self.labels : List[Term] = [label for (src, dst, label) in self.transitions]
        self.alphabet = [label.get_atoms() for label in self.labels]
        self.alphabet = set([atom for atoms in self.alphabet for atom in atoms]) - {'true'}
        self.initial = 1
        self.accepting = set([2])

        self.features_map : Dict[str, Tuple[Feature, str, int]] = None
        self.features : List[Feature] = None

        self.num_states = len(self.states)
        self.num_transitions = len(self.transitions)

        self.tr_function: Dict[int, Dict[Term, int]] = dict()
        for (src, dst, label) in self.transitions:
            if src not in self.tr_function:
                self.tr_function[src] = dict()
            self.tr_function[src][label] = dst

        # Compute SCCs of automata with entry/exit points
        self.sccs: Dict[int, Set[int]] = dict()
        self.lowlinks: List[int] = [None] * (1 + self.num_states)
        self._compute_strongly_connected_components(self.lowlinks)

        self.entries: List[List[int]] = [[] for _ in range(len(self.sccs))]
        self.exits: List[List[int]] = [[] for _ in range(len(self.sccs))]
        for q in range(1, 1 + self.num_states):
            scc_index_q = self.lowlinks[q]
            for _, qp in self.tr_function[q].items():
                scc_index_qp = self.lowlinks[qp]
                if scc_index_qp != scc_index_q:
                    self.entries[scc_index_qp].append(q)
                    self.exits[scc_index_q].append(qp)

    def _parse_dfa(self, dfa_dot):
        init_pattern = re.compile(r'\s*init\s->\s(\d+);', re.MULTILINE)
        edge_pattern = re.compile(r'\s*(\d+)\s->\s(\d+)\s\[label="(.*)"\];', re.MULTILINE)
        accepting_pattern = re.compile(r'\s*node \[shape = doublecircle\];( \d+;)+$', re.MULTILINE)
        initial = int(init_pattern.search(dfa_dot).group(1)[0])
        transitions = [(int(src), int(dst), self._parse_label(label)) for (src, dst, label) in edge_pattern.findall(dfa_dot)]
        states = set([state for pair in [tr[0:2] for tr in transitions] for state in pair])
        accepting = set([int(state.rstrip(';')) for state in accepting_pattern.search(dfa_dot).groups()])
        return states, transitions, initial, accepting

    def _parse_label(self, label: str) -> Term:
        literals = [literal.strip() for literal in label.split('&')]
        return self._parse_literals(literals)

    def _parse_literals(self, literals: List[str]) -> Term:
        if len(literals) == 1 and literals[0] == 'true':
            return Conjunction()
        elif len(literals) == 1:
            negated = literals[0][0] == '~'
            atom = literals[0].strip('~')
            return Literal(atom, negated)
        else:
            parsed_literals = [self._parse_literals([literal]) for literal in literals]
            return Conjunction(parsed_literals)

    # Compute the SCCs of the DFA using Tarjan's algorithm
    def _find_component(self, q: int, index: int, indices: List[int], lowlinks: List[int], in_stack: List[bool], S: deque) -> int:
        # Set (depth) index of vertex to smallest unused index
        indices[q] = index
        lowlinks[q] = index
        index = index + 1
        S.append(q)
        in_stack[q] = True

        # Expand vertex
        for _, qp in self.tr_function[q].items():
            if indices[qp] is None:
                # Successor qp has not yet been visited; recurse on it
                index = self._find_component(qp, index, indices, lowlinks, in_stack, S)
                lowlinks[q] = min(lowlinks[q], lowlinks[qp])
            elif in_stack[qp]:
                # Successor qp is in stack and hence in the current SCC
                # Update its lowlink
                lowlinks[q] = min(lowlinks[q], lowlinks[qp])

        # If q's index equals its lowlink. then q is "entry point" of its SCC.
        # The vertices in its SCC are in the stack down to q
        if lowlinks[q] == indices[q]:
            assert lowlinks[q] not in self.sccs
            new_scc: Set[int] = set()
            while True:
                assert len(S) > 0
                qp = S.pop()
                in_stack[qp] = False
                new_scc.add(qp)
                if qp == q: break
            self.sccs[lowlinks[q]] = new_scc

        return index

    def _compute_strongly_connected_components(self, lowlinks: List[int]):
        indices: List[int] = [None] * (1 + self.num_states)
        in_stack: List[bool] = [False] * (1 + self.num_states)
        assert len(lowlinks) == 1 + self.num_states
        S = deque()
        index = 0

        for q in range(1, 1 + self.num_states):
            if indices[q] is None:
                index = self._find_component(q, index, indices, lowlinks, in_stack, S)

    def set_features(self, labels: List[str], syntactic_element_factory):
        self.features_map: Dict[str, Tuple[Feature, int]] = dict()
        self.features: List[Feature] = []
        for i, label_str in enumerate(labels):
            end_of_feature = label_str.rfind(')')
            feature_str = label_str[:end_of_feature + 1]
            condition = label_str[end_of_feature + 1]
            value = int(label_str[end_of_feature + 2:])
            assert condition in ['>', '=']
            assert value == 0

            feature = None
            if feature_str[0] == 'b':
                boolean = syntactic_element_factory.parse_boolean(feature_str)
                feature = Feature(boolean, boolean.compute_complexity() + 1)
            elif feature_str[0] == 'n':
                numerical = syntactic_element_factory.parse_numerical(feature_str)
                feature = Feature(numerical, numerical.compute_complexity() + 1)
            else:
                logging.error(f"Error: Unrecognized feature in label '{label}'")
                return

            self.features_map[chr(ord('a') + i)] = (feature, condition, value)
            self.features.append(feature)

    def get_labels_interpretation(self, dlplan_ss_state: dlplan_core.State, denotations_caches: dlplan_core.DenotationsCaches) -> Dict[str, bool]:
        labels_interpretation: Dict[str, bool] = dict()
        for (label, (feature, condition, value)) in self.features_map.items():
            feature_value = int(feature.dlplan_feature.evaluate(dlplan_ss_state, denotations_caches))
            labels_interpretation[label] = feature_value > value if condition == '>' else feature_value == value
            #print(f'Feature: feature={str(feature._dlplan_feature)}/{feature.complexity}, dlplan_ss_state={dlplan_ss_state}, value={labels_interpretation[label]}')
        #print(f'DFA: interpretations: dlplan_ss_state={dlplan_ss_state}, labels_interpretation={labels_interpretation}')
        return labels_interpretation

    def initial_state(self, dlplan_ss_state: dlplan_core.State, denotations_caches: dlplan_core.DenotationsCaches) -> int:
        labels_interpretation = self.get_labels_interpretation(dlplan_ss_state, denotations_caches)
        assert self.initial in self.tr_function
        for label, q in self.tr_function[self.initial].items():
            if label.is_consistent(labels_interpretation):
                #print(f'Initial: state={dlplan_ss_state}, interp={labels_interpretation}, label={label}')
                return q
        #print(f'Initial: state={dlplan_ss_state}, interp={labels_interpretation}, label={None}')
        raise False

    def next_state(self, q: int, dlplan_ss_state: dlplan_core.State, denotations_caches: dlplan_core.DenotationsCaches) -> Tuple[Term, int]:
        labels_interpretation = self.get_labels_interpretation(dlplan_ss_state, denotations_caches)
        if q in self.tr_function:
            for label, qp in self.tr_function[q].items():
                if label.is_consistent(labels_interpretation):
                    return label, qp
        return None, None

    def is_accepting_state(self, q: int) -> bool:
        return q in self.accepting

    def __str__(self):
        return self.dfa_dot

def make_dfa(formula_str: str, ppltl: bool = True):
    parser = PPLTLParser() if ppltl else LTLfParser()
    formula = parser(formula_str)
    return DFA(formula_str, formula)
 
