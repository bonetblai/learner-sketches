% Optimization smallest sum over feature complexities
#minimize { C@2,complexity(F, C) : complexity(F, C), feature(F), select(F) }.

% Choose features
{ select(F) } :- feature(F).

% Generate good equivalence classes
{ good(C) } :- state_pair_class(C).
% (NEW) Choose good pairs of qstates and equivalence classes
{ good_class(Q, C) } :- state_pair_class(C), dfa_state(Q).

% Define R-reachable states, base and inductive case
r_reachable(S) :- initial(S).
r_reachable(S2) :- r_reachable(S), good(S, S2), subgoal_distance(S, D), s_distance(S, S2, D), nongoal(S).
% (NEW) Define R-reachable pairs; base and inductive case
r_reachable(Q, S) :- dfa_initial(Q), initial(S).
r_reachable(Q2, S2) :- r_reachable(Q, S), good(S, S2), subgoal_distance(Q, S, D), s_distance(S, S2, D), nongoal(S), dfa_tr(Q, Q2, LABEL), dfa_compatible(S2, LABEL).

% Define a distance from which to pick at least one subgoal tuple for each R-reachable state
{ subgoal_distance(S, D) : t_distance(S, _, D), D > 0 } != 0 :- r_reachable(S), nongoal(S).
% (NEW) Choose a distance from which to pick at least one subgoal tuple for each R-reachable pair
{ subgoal_distance(Q, S, D) : t_distance(S, _, D), D > 0 } != 0 :- r_reachable(Q, S), nongoal(S).

% Require unsolvable state to not be R-reachable
:- unsolvable(S), r_reachable(S).
% (NEW) Require unsolvable state to not be R-reachable
:- unsolvable(S), r_reachable(Q, S).

% Define at least one subgoal tuple at the selected distance
{ subgoal(S, T) : t_distance(S, T, D) } != 0 :- subgoal_distance(S, D), r_reachable(S).
% (NEW) Choose at least one subgoal tuple at the selected distance
{ subgoal(Q, S, T) : t_distance(S, T, D) } != 0 :- subgoal_distance(Q, S, D), r_reachable(Q, S).

% Require that all equivalence classes C underlying a tuple T of subproblem P[S] are good, effectively bounding the width of P[S]
:- not good(C), subgoal(S, T), contain(S, T, C).
% (NEW) Require that all equivalence classes C underlying a tuple T of subproblem P[S] are good, effectively bounding the width of P[S]
:- not good_class(Q, C), subgoal(Q, S, T), contain(S, T, C).
% (Optimal-width): Require solvable states S' closer than subgoal to not be assigned to any rule
:- D < D2, r_distance(S, C, D), subgoal_distance(S, D2), good(C).
% (NEW) (Optimal-width): Require solvable states S' closer than subgoal to not be assigned to any rule
:- D < D2, r_distance(S, C, D), subgoal_distance(Q, S, D2), good_class(Q, C).

% Require D2-separation
:- good(C), not good(C2), d2_separate(C, C2), state_pair_class(C), state_pair_class(C2), { feature(F) : select(F), feature_effect(C, F, V), feature_effect(C2, F, V2), V != V2 } = 0, { feature(F) : select(F), feature_condition(C, F, V), feature_condition(C2, F, V2), V != V2 } = 0.
% (NEW) Require D2-separation
:- good_class(Q, C), not good_class(Q, C2), d2_separate(C, C2), state_pair_class(C), state_pair_class(C2), { feature(F) : select(F), feature_effect(C, F, V), feature_effect(C2, F, V2), V != V2 } = 0, { feature(F) : select(F), feature_condition(C, F, V), feature_condition(C2, F, V2), V != V2 } = 0.

% Define ``good`` pairs of state classes similar
good(S, S2) :- good(C), cover(S, S2, C).
% (NEW) Define ``good`` pairs of state classes similar
good(S, S2) :- good_class(Q, C), cover(S, S2, C).
% (Termination): Sketch must define strict partial order over R-reachable states
% Source of this formulation: https://users.aalto.fi/~rintanj1/papers/GebserJR14kr.pdf
order(S, S2) :- r_reachable(S), r_reachable(S2), nongoal(S), good(S, S2), order(S2).
order(S) :- r_reachable(S), order(S, S2) : good(S, S2), r_reachable(S), r_reachable(S2), nongoal(S).
:- r_reachable(S), nongoal(S), not order(S).
% (NEW) (Termination): Sketch must define strict partial order over R-reachable states
% Source of this formulation: https://users.aalto.fi/~rintanj1/papers/GebserJR14kr.pdf
order_new(Q, S, Q2, S2) :- r_reachable(Q, S), r_reachable(Q2, S2), nongoal(S), good(S, S2), dfa_tr(Q, Q2, LABEL), dfa_compatible(S2, LABEL), order_new(Q2, S2).
order_new(Q, S) :- r_reachable(Q, S), order_new(Q, S, Q2, S2) : good(S, S2), r_reachable(Q, S), r_reachable(Q2, S2), nongoal(S), dfa_tr(Q, Q2, LABEL), dfa_compatible(S2, LABEL).
:- r_reachable(Q, S), nongoal(S), not order_new(Q, S).

% Display
#show select/1.
#show numerical/1.
#show boolean/1.
#show good/1.
#show feature_condition/3.
#show feature_effect/3.
#show good_class/2.        % (NEW)
#show order_new/2.         % (NEW)
