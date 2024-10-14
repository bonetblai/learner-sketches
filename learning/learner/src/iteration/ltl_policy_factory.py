import re

import dlplan.core as dlplan_core
import dlplan.policy as dlplan_policy

from clingo import Symbol
from typing import Set, List, Union, MutableSet

from .iteration_data import IterationData
from .dlplan_policy_factory import DlplanPolicyFactory
from ..preprocessing import PreprocessingData
from .ltl_policy import LTLRule, LTLPolicy


class LTLD2sepDlplanPolicyFactory(DlplanPolicyFactory):
    """
    Encoding where rules are implicit in the D2-separation.
    """
    def make_dlplan_policy_from_answer_set(self, symbols: List[Symbol], preprocessing_data: PreprocessingData, iteration_data: IterationData) -> LTLPolicy:
        policy_builder = preprocessing_data.domain_data.policy_builder
        dlplan_features = set()
        dfa_tr = dict()
        for symbol in symbols:
            if symbol.name == "select":
                #_print_symbol(symbol)
                f_idx = symbol.arguments[0].number
                dlplan_features.add(iteration_data.feature_pool[f_idx].dlplan_feature)
            elif symbol.name == "dfa_tr":
                #_print_symbol(symbol)
                q1 = symbol.arguments[0].number
                q2 = symbol.arguments[1].number
                label = symbol.arguments[2].string
                if q1 not in dfa_tr: dfa_tr[q1] = dict()
                dfa_tr[q1][label] = q2
            elif symbol.name == "dfa_consistent":
                #_print_symbol(symbol)
                pass

        rules = set()
        for symbol in symbols:
            if symbol.name == "good_new":
                #_print_symbol(symbol)
                q1 = symbol.arguments[0].number
                r_idx = symbol.arguments[1].number
                rule = iteration_data.state_pair_equivalences[r_idx]
                conditions = set()
                for condition in rule.get_conditions():
                    f_idx = int(condition.get_named_element().get_key()[1:])
                    dlplan_feature = iteration_data.feature_pool[f_idx].dlplan_feature
                    if dlplan_feature in dlplan_features:
                        conditions.add(condition)
                effects = set()
                for effect in rule.get_effects():
                    f_idx = int(effect.get_named_element().get_key()[1:])
                    dlplan_feature = iteration_data.feature_pool[f_idx].dlplan_feature
                    if dlplan_feature in dlplan_features:
                        effects.add(effect)
                assert q1 in dfa_tr
                for label in dfa_tr[q1]:
                    assert label in dfa_tr[q1]
                    q2 = dfa_tr[q1][label]
                    dlplan_rule = policy_builder.make_rule(conditions, effects)
                    ext_rule = LTLRule(dlplan_rule, q1, label, q2)
                    rules.add(ext_rule)
            elif symbol.name == "good_new3":
                #_print_symbol(symbol)
                pass

        ltl_policy = LTLPolicy.make_policy(rules, policy_builder)
        return ltl_policy

def _print_symbol(symbol):
    if symbol.name == "good_new":
        print(f'ASP: Symbol: good_new({symbol.arguments[0].number},{symbol.arguments[1].number})')
        pass
    elif symbol.name == "good_new3":
        print(f'ASP: Symbol: good_new3({symbol.arguments[0].number},{symbol.arguments[1].number},{symbol.arguments[2].number})')
        pass
    elif symbol.name == "order_new":
        print(f'ASP: Symbol: order_new({symbol.arguments[0].number},{symbol.arguments[1].number})')
        pass
    elif symbol.name == "order_new4":
        print(f'ASP: Symbol: order_new4({symbol.arguments[0].number},{symbol.arguments[1].number},{symbol.arguments[2].number},{symbol.arguments[3].number})')
        pass
    elif symbol.name == "r_reachable_new":
        print(f'ASP: Symbol: r_reachable_new({symbol.arguments[0].number},{symbol.arguments[1].number})')
        pass
    elif symbol.name == "subgoal_distance_new":
        print(f'ASP: Symbol: subgoal_distance_new({symbol.arguments[0].number},{symbol.arguments[1].number},{symbol.arguments[2].number})')
        pass
    elif symbol.name == "s_distance":
        print(f'ASP: Symbol: s_distance({symbol.arguments[0].number},{symbol.arguments[1].number},{symbol.arguments[2].number})')
        pass
    elif symbol.name == "feature_condition":
        print(f'ASP: Symbol: feature_condition({symbol.arguments[0].number},{symbol.arguments[1].number},"{symbol.arguments[2].string}")')
        pass
    elif symbol.name == "feature_effect":
        print(f'ASP: Symbol: feature_effect({symbol.arguments[0].number},{symbol.arguments[1].number},"{symbol.arguments[2].string}")')
        pass
    elif symbol.name == "d2_separate":
        #print(f'ASP: Symbol: d2_separate({symbol.arguments[0].number},{symbol.arguments[1].number})')
        pass
    elif symbol.name == "cover":
        print(f'ASP: Symbol: cover({symbol.arguments[0].number},{symbol.arguments[1].number},{symbol.arguments[2].number})')
        pass
    elif symbol.name == "dfa_tr":
        print(f'ASP: Symbol: dfa_tr({symbol.arguments[0].number},{symbol.arguments[1].number},"{symbol.arguments[2].string}")')
        pass
    elif symbol.name == "dfa_consistent":
        print(f'ASP: Symbol: dfa_consistent({symbol.arguments[0].number},"{symbol.arguments[1].string}")')
        pass
    elif symbol.name == "initial":
        print(f'ASP: Symbol: initial({symbol.arguments[0].number})')
        pass
 
