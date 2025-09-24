from pref_voting.scoring_methods import plurality, borda, anti_plurality
from pref_voting.c1_methods import condorcet, copeland, copeland, llull, uc_gill, top_cycle, banks, slater
from pref_voting.iterative_methods import instant_runoff_tb, plurality_with_runoff_put, coombs, baldwin, weak_nanson
from pref_voting.margin_based_methods import minimax, split_cycle, river, stable_voting, ranked_pairs
from pref_voting.combined_methods import blacks
from pref_voting.other_methods import kemeny_young

import axioms as ax
import axioms_continuous as ax_cont
from utils import rand_rule

# Dictionaries of voting rules

dict_rules = {
    'Plurality': plurality,
    'Borda': borda,
    'Copeland': copeland,
    'RandomRule': rand_rule
}


# The names are as given in the pref_voting package
dict_rules_all_fast = {
    'Plurality': plurality,
    'Borda': borda,
    'Anti-Plurality': anti_plurality,
    'Copeland': copeland,
    'Llull': llull,
    'Uncovered Set': uc_gill,
    'Top Cycle': top_cycle,
    'Banks': banks,
    #'Slater': slater,    # Slow-ish but quite okay
    #'Minimax': minimax,  # Slow! Caused kernel crashes with 7 alternatives
    #'Split Cycle': split_cycle,  # Slow! Caused kernel crashes with 7 alternatives
    #'River': river,    # Slow!
    'Stable Voting': stable_voting,
    #'Ranked Pairs': ranked_pairs,   # Slow!
    'Blacks': blacks,
    'Instant Runoff TB': instant_runoff_tb,
    'Plurality With Runoff PUT': plurality_with_runoff_put,
    'Coombs': coombs,
    'Baldwin': baldwin,
    'Weak Nanson': weak_nanson,
    'Kemeny-Young': kemeny_young
}

# The names are as given in the `pref_voting`` package 
DICT_RULES_ALL = {
    'Plurality': plurality,
    'Borda': borda,
    'Anti-Plurality': anti_plurality,
    'Copeland': copeland,
    'Llull': llull,
    'Uncovered Set': uc_gill,
    'Top Cycle': top_cycle,
    'Banks': banks,
    'Slater': slater,
    'Minimax': minimax,  # Slow! Caused kernel crashes with 7 alternatives
    'Split Cycle': split_cycle,  # Slow! Caused kernel crashes with 7 alternatives
    'River': river,
    'Stable Voting': stable_voting,  # Slow!
    'Ranked Pairs': ranked_pairs,
    'Blacks': blacks,
    'Instant Runoff TB': instant_runoff_tb,
    'PluralityWRunoff PUT': plurality_with_runoff_put,
    'Coombs': coombs,
    'Baldwin': baldwin,
    'Weak Nanson': weak_nanson,
    'Kemeny-Young': kemeny_young
}


DICT_RULES_COLORS = {
    'Plurality': 'r',
    'Borda': 'b',
    'Copeland': 'y',
    'RandomRule': 'c'
}

# More colloquial names of sampling methods
DICT_SAMPLING_METHODS = {
    'IC' : 'IC',
    'URN-R' : 'Urn', 
    'MALLOWS-RELPHI' : 'Mallows', 
    'euclidean' : 'Euclidean', 
}

DICT_AXIOMS = {
    'Condorcet':ax.ax_condorcet,
    'Independence':ax.ax_independence,
} 

DICT_AXIOMS_ALL = {
    'Condorcet':ax.ax_condorcet,
    'Independence':ax.ax_independence,
}

DICT_AXIOMS_CONT = {
    'No_winner': {'fn': ax_cont.ax_no_winners_cont},
    'Inadmissible': {'fn': ax_cont.ax_inadmissibility_cont},
    'Condorcet1': {'fn': ax_cont.ax_condorcet1_cont, 'uses_distance': True},
    'Condorcet2': {'fn': ax_cont.ax_condorcet2_cont},
    'Independence': {'fn': ax_cont.ax_independence_cont, 'uses_distance': True},
}


