import logging

import dlplan.core as dlplan_core
import dlplan.policy as dlplan_policy
from dlplan.policy import PolicyMinimizer

from typing import Dict, Set, Tuple, List, Union, MutableSet, Any
from collections import defaultdict

#from ..iteration import IterationData
#from ..preprocessing import PreprocessingData
from .ltl_base import DFA


class LTLRule:
    def __init__(self, dlplan_rule: dlplan_policy.Rule, q1: int, label: str, q2: int):
        self.dlplan_rule = dlplan_rule
        self.q1 = q1
        self.label = label
        self.q2 = q2

    def __str__(self):
        ltl_rule_str = f'((:q1 {self.q1}) (:label "{self.label}") (:q2 {self.q2}) {self.dlplan_rule})'
        return ltl_rule_str

class LTLPolicy:
    def __init__(self, dlplan_rules: Dict[int, Set[LTLRule]] = None, dlplan_policies: Dict[int, dlplan_policy.Policy] = None):
        self.dlplan_rules = dlplan_rules if dlplan_rules is not None else dict()
        self.dlplan_policies = dlplan_policies if dlplan_policies is not None else dict()

    @staticmethod
    def make_policy(ltl_rules: Set[LTLRule], policy_builder: Any):
        dlplan_rules: Dict[int, Set[LTLRule]] = defaultdict(set)
        for ltl_rule in ltl_rules:
            q1 = ltl_rule.q1
            dlplan_rules[q1].add(ltl_rule.dlplan_rule)

        dlplan_policies: Dict[int, dlplan_policy.policy] = dict()
        for q1 in dlplan_rules:
            dlplan_rules_for_state = dlplan_rules[q1]
            dlplan_policies[q1] = policy_builder.make_policy(dlplan_rules_for_state)

        return LTLPolicy(dlplan_rules, dlplan_policies)

    # If transition (q,s) -> (q',s') is compatible with rule r, return (q',r). Else, return (None, None)
    def evaluate(self, q: int, dlplan_ss_state: dlplan_core.State, dlplan_ss_state_prime: dlplan_core.State, denotations_caches: dlplan_core.DenotationsCaches, dfa: DFA):
        if q in self.dlplan_rules:
            label, q_prime = dfa.next_state(q, dlplan_ss_state_prime, denotations_caches)
            for rule in self.dlplan_rules[q]:
                conditions = rule.evaluate_conditions(dlplan_ss_state, denotations_caches)
                effects = rule.evaluate_effects(dlplan_ss_state, dlplan_ss_state_prime, denotations_caches)
                if conditions and effects:
                    return rule, q_prime
        return None, None

    def minimize(self, policy_builder: Any):
        minimized_dlplan_rules: Dict[int, Set[LTLRule]] = defaultdict(set)
        minimized_dlplan_policies: Dict[int, dlplan_policy.policy] = dict()
        for q, dlplan_policy in self.dlplan_policies.items():
            dlplan_policy_minimized = PolicyMinimizer().minimize(dlplan_policy, policy_builder)
            minimized_dlplan_policies[q] = dlplan_policy_minimized
            for rule in dlplan_policy_minimized.get_rules():
                minimized_dlplan_rules[q].add(rule)

        return LTLPolicy(minimized_dlplan_rules, minimized_dlplan_policies)

    def print(self):
        keys = sorted(self.dlplan_policies.keys())
        for q in keys:
            print(f'State q{q}:')
            print(self.dlplan_policies[q])
