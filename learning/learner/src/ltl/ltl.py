import re

from ltlf2dfa.parser.ltlf import LTLfParser
from ltlf2dfa.parser.ppltl import PPLTLParser

class DFA(object):
    def __init__(self, formula_str, formula):
        self.formula_str = formula_str
        self.formula = formula
        self.dfa_dot = self.formula.to_dfa()
        self.states, self.transitions, self.initial, self.accepting = _parse_dfa(self.dfa_dot)
        self.labels = [label for (src, dst, label) in self.transitions]
        self.alphabet = [[item.strip().lstrip('~') for item in label.split('&')] for label in self.labels]
        self.alphabet = set([item for list in self.alphabet for item in list]) - {'true'}
        self.num_states = len(self.states)
        self.num_transitions = len(self.transitions)
    def __str__(self):
        return self.dfa_dot

def make_dfa(formula_str: str, ppltl: bool = True):
    parser = PPLTLParser() if ppltl else LTLfParser()
    formula = parser(formula_str)
    return DFA(formula_str, formula)

def _parse_dfa(dfa_dot):
    init_pattern = re.compile(r'\s*init\s->\s(\d+);', re.MULTILINE)
    edge_pattern = re.compile(r'\s*(\d+)\s->\s(\d+)\s\[label="(.*)"\];', re.MULTILINE)
    accepting_pattern = re.compile(r'\s*node \[shape = doublecircle\];( \d+;)+$', re.MULTILINE)
    initial = init_pattern.search(dfa_dot).group(1)
    transitions = [(int(src), int(dst), label) for (src, dst, label) in edge_pattern.findall(dfa_dot)]
    states = set([state for pair in [tr[0:2] for tr in transitions] for state in pair])
    accepting = set([int(state.rstrip(';')) for state in accepting_pattern.search(dfa_dot).groups()])
    return states, transitions, initial, accepting
 
