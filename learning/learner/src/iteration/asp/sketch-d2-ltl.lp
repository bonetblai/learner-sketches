% Optimization smallest sum over feature complexities
#minimize { C@2,complexity(F, C) : complexity(F, C), feature(F), select(F) }.

% TODO:
%  1. Fix addition of d2_separate() facts [difficult] try first with all pairs
%  2. Remove goal atoms / concepts ; e.g. at_g

% Choose features
{ select(F) } :- feature(F).

% (OLD) Generate good equivalence classes
%{ good(C) } :- state_pair_class(C).
% (NEW) Choose good pairs of qstates and equivalence classes
{ good_new(Q, C) } :- state_pair_class(C), dfa_state(Q).

% (NEW) Nongoal dfa states
nongoal_new(Q) :- dfa_state(Q), not dfa_accepting(Q).

% (OLD) Define R-reachable states, base and inductive case
%r_reachable(S) :- initial(S).
%r_reachable(S2) :- r_reachable(S), good(S, S2), subgoal_distance(S, D), s_distance(S, S2, D), nongoal(S).
% (NEW) Define R-reachable pairs; base and inductive case
r_reachable_new(Q2, S) :- S =  0, initial(S), dfa_initial(Q), dfa_tr(Q, Q2, LABEL), dfa_consistent(S, LABEL). % HACK
r_reachable_new(Q2, S) :- S = 16, initial(S), dfa_initial(Q), dfa_tr(Q, Q2, LABEL), dfa_consistent(S, LABEL). % HACK
r_reachable_new(Q2, S) :- S = 32, initial(S), dfa_initial(Q), dfa_tr(Q, Q2, LABEL), dfa_consistent(S, LABEL). % HACK
r_reachable_new(Q2, S2) :- r_reachable_new(Q, S), good_new3(Q, S, S2), subgoal_distance_new(Q, S, D), s_distance(S, S2, D), nongoal_new(Q), dfa_tr(Q, Q2, LABEL), dfa_consistent(S2, LABEL).

% (OLD) Define a distance from which to pick at least one subgoal tuple for each R-reachable state
%{ subgoal_distance(S, D) : t_distance(S, _, D), D > 0 } != 0 :- r_reachable(S), nongoal(S).
% (NEW) Choose a distance from which to pick at least one subgoal tuple for each R-reachable pair
{ subgoal_distance_new(Q, S, D) : t_distance(S, _, D), D > 0 } != 0 :- r_reachable_new(Q, S), nongoal_new(Q).

% (OLD) Require unsolvable state to not be R-reachable
%:- unsolvable(S), r_reachable(S).
% (NEW) Require unsolvable state to not be R-reachable
:- unsolvable(S), r_reachable_new(Q, S).

% (OLD) Define at least one subgoal tuple at the selected distance
%{ subgoal(S, T) : t_distance(S, T, D) } != 0 :- subgoal_distance(S, D), r_reachable(S).
% (NEW) Choose at least one subgoal tuple at the selected distance
{ subgoal_new(Q, S, T) : t_distance(S, T, D) } != 0 :- subgoal_distance_new(Q, S, D), r_reachable_new(Q, S).

% (OLD) Require that all equivalence classes C underlying a tuple T of subproblem P[S] are good, effectively bounding the width of P[S]
%:- not good(C), subgoal(S, T), contain(S, T, C).
% (OLD) (Optimal-width): Require solvable states S' closer than subgoal to not be assigned to any rule
%:- D < D2, r_distance(S, C, D), subgoal_distance(S, D2), good(C).

% (NEW) Require that all equivalence classes C underlying a tuple T of subproblem P[S] are good, effectively bounding the width of P[S]
:- not good_new(Q, C), subgoal_new(Q, S, T), contain(S, T, C).
% (NEW) (Optimal-width): Require solvable states S' closer than subgoal to not be assigned to any rule
:- D < D2, r_distance(S, C, D), subgoal_distance_new(Q, S, D2), good_new(Q, C).

% (OLD) Require D2-separation
%:- good(C), not good(C2), d2_separate(C, C2), state_pair_class(C), state_pair_class(C2), { feature(F) : select(F), feature_effect(C, F, V), feature_effect(C2, F, V2), V != V2 } = 0, { feature(F) : select(F), feature_condition(C, F, V), feature_condition(C2, F, V2), V != V2 } = 0.
% (NEW) Require D2-separation
:- good_new(Q, C), not good_new(Q, C2), d2_separate(C, C2), state_pair_class(C), state_pair_class(C2), { feature(F) : select(F), feature_effect(C, F, V), feature_effect(C2, F, V2), V != V2 } = 0, { feature(F) : select(F), feature_condition(C, F, V), feature_condition(C2, F, V2), V != V2 } = 0.

% (OLD) Define ``good`` pairs of state classes similar
%good(S, S2) :- good(C), cover(S, S2, C).
% (OLD) (Termination): Sketch must define strict partial order over R-reachable states
% (OLD) Source of this formulation: https://users.aalto.fi/~rintanj1/papers/GebserJR14kr.pdf
%order(S, S2) :- r_reachable(S), r_reachable(S2), nongoal(S), good(S, S2), order(S2).
%order(S) :- r_reachable(S), order(S, S2) : good(S, S2), r_reachable(S), r_reachable(S2), nongoal(S).
%:- r_reachable(S), nongoal(S), not order(S).

% (NEW) Define ``good`` pairs of state classes similar
good_new3(Q, S, S2) :- good_new(Q, C), cover(S, S2, C).
% (NEW) (Termination): Sketch must define strict partial order over R-reachable states
% (NEW) Source of this formulation: https://users.aalto.fi/~rintanj1/papers/GebserJR14kr.pdf
order_new4(Q, S, Q2, S2) :- r_reachable_new(Q, S), r_reachable_new(Q2, S2), nongoal_new(Q), good_new3(Q, S, S2), dfa_tr(Q, Q2, LABEL), dfa_consistent(S2, LABEL), order_new(Q2, S2).
order_new(Q, S) :- r_reachable_new(Q, S), order_new4(Q, S, Q2, S2) : good_new3(Q, S, S2), r_reachable_new(Q, S), r_reachable_new(Q2, S2), nongoal_new(Q), dfa_tr(Q, Q2, LABEL), dfa_consistent(S2, LABEL).
:- r_reachable_new(Q, S), nongoal_new(Q), not order_new(Q, S).

% Display
#show select/1.
#show numerical/1.
#show boolean/1.
%#show good/1.
#show feature_condition/3.
#show feature_effect/3.

#show s_distance/3.
#show r_reachable_new/2.
#show subgoal_distance_new/3.
#show good_new/2.          % (NEW)
#show good_new3/3.         % (NEW)
#show order_new/2.         % (NEW)
#show order_new4/4.        % (NEW)
#show dfa_tr/3.
#show dfa_consistent/2.
#show initial/1.
#show cover/3.
#show d2_separate/2.
