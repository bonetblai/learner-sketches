import logging

import re
from collections import deque
from typing import Dict, Set, Tuple, List, Any, abstractmethod

import dlplan.core as dlplan_core
from .feature_pool import Feature
from ltlf2dfa.parser.ltlf import LTLfParser
from ltlf2dfa.parser.ppltl import PPLTLParser

def _split(string: str, lpar: str, rpar: str, splitchar: str) -> List[str]:
    #print(f'_split: string=|{string}|, lpar=|{lpar}|, rpar=|{rpar}|, splitchar=|{splitchar}|')
    start, level, i, n = 0, 0, 0, len(string)
    substrings = []
    while i < n:
        if level == 0 and string[i] == splitchar:
            substrings.append(string[start:i])
            #print(f'_split:     substring=|{substrings[-1]}|')
            start = i + 1
        elif string[i] == lpar:
            level = level + 1
        elif string[i] == rpar:
            level = level - 1
        if level < 0:
            raise ValueError(f"Badly formed string=|{string}|")
        i = i + 1
    #print(f'level={level}, start={start}, i={i}, n={n}')

    if level == 0 and start < i + 1:
        substrings.append(string[start:])
        #print(f'_split:     substring=|{substrings[-1]}|')
    elif level != 0:
        raise ValueError(f"Badly formed string=|{string}|")

    return substrings

class Term(object):
    def __init__(self):
        pass

    @abstractmethod
    def get_atoms(self) -> Set[str]:
        raise RuntimeError("Abstract method 'get_atom' called")

    @abstractmethod
    def is_consistent(self, interp: Dict[str, bool]) -> bool:
        raise RuntimeError("Abstract method 'is_consistent' called")

    @abstractmethod
    def __str__(self) -> str:
        raise RuntimeError("Abstract method '__str__' called")

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

class Denotation(object):
    def __init__(self):
        pass

    @abstractmethod
    def evaluate(self, dlplan_ss_state: dlplan_core.State, denotations_caches: dlplan_core.DenotationsCaches) -> bool:
        raise RuntimeError("Abstract method 'evaluate' called")

    @abstractmethod
    def __str__(self) -> str:
        raise RuntimeError("Abstract method '__str__' called")

class Not(Denotation):
    def __init__(self, denotation: Denotation):
        self.denotation = denotation

    def evaluate(self, dlplan_ss_state: dlplan_core.State, denotations_caches: dlplan_core.DenotationsCaches) -> bool:
        return not self.denotation.evaluate(dlplan_ss_state, denotations_caches)

    def __str__(self) -> str:
        return f"Not[{str(denotation)}]"

class And(Denotation):
    def __init__(self, denotations: List[Denotation]):
        self.denotations = denotations

    def evaluate(self, dlplan_ss_state: dlplan_core.State, denotations_caches: dlplan_core.DenotationsCaches) -> bool:
        return all([denotation.evaluate(dlplan_ss_state, denotations_caches) for denotation in self.denotations])

    def __str__(self) -> str:
        return f"And[{','.join([str(denotation) for denotation in self.denotations])}]"

class Or(Denotation):
    def __init__(self, denotations: List[Denotation]):
        self.denotations = denotations

    def evaluate(self, dlplan_ss_state: dlplan_core.State, denotations_caches: dlplan_core.DenotationsCaches) -> bool:
        return any([denotation.evaluate(dlplan_ss_state, denotations_caches) for denotation in self.denotations])

    def __str__(self) -> str:
        return f"Or[{','.join([str(denotation) for denotation in self.denotations])}]"

class Equal(Denotation):
    def __init__(self, feature: str, reference_value: int, syntactic_element_factory: Any):
        self.reference_value = reference_value
        if feature[:2] == 'b_':
            boolean = syntactic_element_factory.parse_boolean(feature)
            self.feature = Feature(boolean, boolean.compute_complexity() + 1)
        elif feature[:2] == 'n_':
            numerical = syntactic_element_factory.parse_numerical(feature)
            self.feature = Feature(numerical, numerical.compute_complexity() + 1)
        else:
            logging.error(f"Unexpected feature '{feature}' in Equal")
            self.feature = None

    def evaluate(self, dlplan_ss_state: dlplan_core.State, denotations_caches: dlplan_core.DenotationsCaches) -> bool:
        feature_value = int(self.feature.dlplan_feature.evaluate(dlplan_ss_state, denotations_caches))
        return feature_value == self.reference_value

    def __str__(self) -> str:
        return f"Equal[{str(self.feature._dlplan_feature)},{self.value}]"

class GreaterThan(Denotation):
    def __init__(self, feature: str, reference_value: int, syntactic_element_factory: Any):
        self.reference_value = reference_value
        if feature[:2] == 'b_':
            boolean = syntactic_element_factory.parse_boolean(feature)
            self.feature = Feature(boolean, boolean.compute_complexity() + 1)
        elif feature[:2] == 'n_':
            numerical = syntactic_element_factory.parse_numerical(feature)
            self.feature = Feature(numerical, numerical.compute_complexity() + 1)
        else:
            logging.error(f"Unexpected feature '{feature}' in GreaterThan")
            self.feature = None

    def evaluate(self, dlplan_ss_state: dlplan_core.State, denotations_caches: dlplan_core.DenotationsCaches) -> bool:
        feature_value = int(self.feature.dlplan_feature.evaluate(dlplan_ss_state, denotations_caches))
        return feature_value > self.reference_value

    def __str__(self) -> str:
        return f"GreaterThan[{str(self.feature._dlplan_feature)},{self.value}]"

class DFA(object):
    def __init__(self, formula_str, formula):
        self.formula_str = formula_str
        self.formula = formula
        self.dfa_dot = self.formula.to_dfa()
        self.states, self.transitions, self.initial, self.accepting = self._parse_dfa(self.dfa_dot)
        self.labels : List[Term] = [label for (src, dst, label) in self.transitions]
        self.alphabet = [label.get_atoms() for label in self.labels]
        self.alphabet = set([atom for atoms in self.alphabet for atom in atoms]) - {'true'}

        self.denotations_map : Dict[str, Denotation] = None
        self.denotations : List[Denotation] = None

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

        self.scc_entry_points: List[Set[int]] = [set() for _ in range(len(self.sccs))]
        self.scc_exit_points: List[Set[int]] = [set() for _ in range(len(self.sccs))]
        for q in range(1, 1 + self.num_states):
            scc_index_q = self.lowlinks[q]
            for _, qp in self.tr_function[q].items():
                scc_index_qp = self.lowlinks[qp]
                if scc_index_qp != scc_index_q:
                    self.scc_entry_points[scc_index_qp].add(q)
                    self.scc_exit_points[scc_index_q].add(qp)

        self.scc_initial_states: List[Set[int]] = [set() for _ in range(len(self.sccs))]
        for scc_index in range(len(self.sccs)):
            entry_points = self.scc_entry_points[scc_index] if len(self.scc_entry_points[scc_index]) > 0 else set([self.initial])
            for q in entry_points:
                for _, qp in self.tr_function[q].items():
                    if self.lowlinks[qp] == scc_index:
                        self.scc_initial_states[scc_index].add(qp)

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

    @staticmethod
    def _parse_denotation(denotation_str: str, syntactic_element_factory: Any) -> Denotation:
        try:
            start = denotation_str.index('(')
            end = denotation_str.rindex(')')
        except Exception as e:
            raise e

        print(f'HOLA: denotation_str=|{denotation_str}|, start={start}, end={end}')
        if denotation_str[:start] == 'b_not':
            denotation = DFA._parse_denotation(denotation_str[1+start:end].strip(), syntactic_element_factory)
            return Not(denotation)
        elif denotation_str[:start] == 'b_and':
            items = [DFA._parse_denotation(item.strip(), syntactic_element_factory) for item in _split(denotation_str[1+start:end], '(', ')', ',')]
            return And(items)
        elif denotation_str[:start] == 'b_or':
            items = [DFA._parse_denotation(item.strip(), syntactic_element_factory) for item in _split(denotation_str[1+start:end], '(', ')', ',')]
            return Or(items)
        else:
            try:
                end = denotation_str.rindex(')')
            except Exception as e:
                raise e
            feature = denotation_str[:end + 1]
            condition = denotation_str[end + 1]
            reference_value = int(denotation_str[end + 2:])
            if condition == '>':
                return GreaterThan(feature, reference_value, syntactic_element_factory)
            elif condition == '=':
                return Equal(feature, reference_value, syntactic_element_factory)
            else:
                raise ValueError(f"Unexpected condition '{condition}' in denotation")

    @staticmethod
    def parse_denotations(denotation_strs: List[str], syntactic_element_factory: Any) -> List[Denotation]:
        denotations = []
        for denotation_str in denotation_strs:
            if len(denotation_str) > 0 and denotation_str[0] != '#':
                denotations.append(DFA._parse_denotation(denotation_str, syntactic_element_factory))
        logging.info(f"{len(denotations)} denotation(s)")
        return denotations

    def set_denotations(self, denotations: List[Denotation]):
        self.denotations_map: Dict[str, Denotation] = dict()
        self.denotations: List[Denotation] = []
        for i, denotation in enumerate(denotations):
            self.denotations_map[chr(ord('a') + i)] = denotation
            self.denotations.append(denotation)

    def get_labels_interpretation(self, dlplan_ss_state: dlplan_core.State, denotations_caches: dlplan_core.DenotationsCaches) -> Dict[str, bool]:
        labels_interpretation: Dict[str, bool] = dict()
        for label, denotation in self.denotations_map.items():
            labels_interpretation[label] = denotation.evaluate(dlplan_ss_state, denotations_caches)
        #print(f'DFA: interpretations: dlplan_ss_state={dlplan_ss_state}, labels_interpretation={labels_interpretation}')
        return labels_interpretation

    def get_initial_states(self, scc_initial_states: Set[int], dlplan_ss_state: dlplan_core.State, denotations_caches: dlplan_core.DenotationsCaches) -> List[int]:
        labels_interpretation = self.get_labels_interpretation(dlplan_ss_state, denotations_caches)

        initial_states = []
        for q in scc_initial_states:
            for label, qp in self.tr_function[q].items():
                if label.is_consistent(labels_interpretation):
                    #print(f'Initial: state={dlplan_ss_state}, interp={labels_interpretation}, label={label}, q={q}, qp={qp}")
                    initial_states.append(qp)

        assert len(initial_states) > 0
        return initial_states

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
 
