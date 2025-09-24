from random import shuffle, randint, choice, sample
import itertools
import torch
import numpy as np
from pref_voting.profiles import Profile 

from utils import profiles_to_ranks, PAD_PREF


def ax_condorcet(rule_mapping, profiles_tensor):
    '''
    Condorcet axiom: If a Condorcet winner exists, it must be selected by the rule.
    Args:
        rule_mapping (callable): Function mapping profiles_tensor to binary prediction tensor [B, A].
        profiles_tensor (torch.Tensor): Shape [B, V, A], where B is the batch size, V is the number of voters, and A is the number of alternatives.
    Returns:
        torch.Tensor: Shape [B], values in {-1, 0, 1}.
    '''
    B, V, A = profiles_tensor.shape
    results = torch.zeros(B, dtype=torch.int8)
    binary_preds = rule_mapping(profiles_tensor).bool()  # binary winner predictions [B, A]
    # Note: no valid mask is needed

    for b in range(B):
        # Remove padding and convert to list-of-rankings 
        rankings = [
            tuple(int(a) for a in voter if a != PAD_PREF)
            for voter in profiles_tensor[b]
            if (voter != PAD_PREF).any()
        ]
        if not rankings:  # empty profile (all padding)
            continue

        condorset_winner = Profile(rankings).condorcet_winner()  # int or None
        if condorset_winner is None:
            continue 

        # Compare with rule's winners
        winners = torch.nonzero(binary_preds[b]).squeeze(1).tolist()  # indices list
        if len(winners) == 1 and winners[0] == condorset_winner:
            results[b] = 1  # satisfied
        else:
            results[b] = -1  # violated

    # print('[DEBUG] profiles_tensor:', profiles_tensor)
    # print('[DEBUG] orig_winners:', binary_preds)
    # print('[DEBUG] results:', results)
    
    return results  # [B]


def ax_independence(rule_mapping, profiles_tensor):
    B, V, A = profiles_tensor.shape
    
    orig_winners = rule_mapping(profiles_tensor).bool()

    valid_mask = profiles_tensor.ne(PAD_PREF)
    valid_alt_mask = valid_mask.any(dim=1)
    valid_voter_mask = valid_mask.any(dim=2)
    num_valid_alts = valid_alt_mask.sum(dim=1)
    num_valid_voters = valid_voter_mask.sum(dim=1)
    
    satisfaction = torch.ones(B, dtype=torch.int8)  # initialize as all profiles satisfied
    
    # Filter trivial cases
    trivial_setups = (num_valid_alts < 3) | (num_valid_voters < 2)  # ToDo think
    satisfaction[trivial_setups] = 0

    # Check each profile
    # print(f'Profiles to check in batch: {B}')
    for b in range(B):
        if satisfaction[b] == 0:
            # print(f'Profile number {b}: NOT APPLICABLE')
            continue

        valid_profile= profiles_tensor[b][valid_mask[b].any(dim=1)][:, valid_mask[b].any(dim=0)].tolist()  # [valid V, valid A]
        valid_winner_set = torch.where(orig_winners[b] & valid_alt_mask[b])[0].tolist()
        valid_loser_set = torch.where(~orig_winners[b] & valid_alt_mask[b])[0].tolist()
        
        # print(f'\nChecking profile number {b}')
        # print('Profile:', valid_profile)
        # print('Original winners:', valid_winner_set)
        # print('Original losers:', valid_loser_set)

        # Check each winner-loser pair
        violation_found = False
        for w in valid_winner_set:
            for l in valid_loser_set:
                # print(f'Checking pair Winner {w} vs Loser {l}')

                allowed_rankings = []
                for ranking in valid_profile:
                    possible_rankings = [
                        p
                        for p in list(
                            itertools.permutations(range(len(ranking)))
                        )
                        if (p.index(w) > p.index(l))
                        == ((ranking.index(w) > ranking.index(l)))
                    ]
                    allowed_rankings.append(possible_rankings)
                    # print('[DEBUG] for ranking', ranking)
                    # print('[DEBUG] appending possible_rankings', possible_rankings)

                new_profiles = []
                for choice_of_rankings in list(
                        itertools.product(*allowed_rankings)
                    ):
                       new_profiles.append(choice_of_rankings)
                    #    print('[DEBUG] appending new profile:', choice_of_rankings)

                new_profiles_tensor = torch.tensor(new_profiles)  # [permutations, valid V, valid A]
                new_profiles_tensor_padded = torch.full((new_profiles_tensor.size(0), V, A), PAD_PREF)  # [permutations, V, A]
                new_profiles_tensor_padded[:new_profiles_tensor.size(0), 
                                           :new_profiles_tensor.size(1),
                                           :new_profiles_tensor.size(2)] = new_profiles_tensor


                winners_of_permuted_profile = rule_mapping(new_profiles_tensor_padded).bool()  # [permutations, A]
                # If any loser becomes a winner
                # if (winners_of_permuted_profile[:, l] != False).any():
                violating_indices = torch.where(winners_of_permuted_profile[:, l])[0]
                if len(violating_indices) > 0:
                    violation_found = True
                    # print(f'  [VIOLATION] Loser {l} became a winner in {len(violating_indices)} permutation(s):')
                    # for idx in violating_indices.tolist():
                    #     permuted_profile = new_profiles_tensor_padded[idx]
                    #     winners = winners_of_permuted_profile[idx].nonzero(as_tuple=True)[0].tolist()
                    #     print(f'    Permuted profile {idx}:')
                    #     print(permuted_profile)
                    #     print(f'    Winners: {winners}')

            if violation_found:
                break
        
        if violation_found:
            satisfaction[b] = -1
            # print(f'Profile number {b}: VIOLATION FOUND')
        else:
            satisfaction[b] = 1 # todo duplicate
            # print(f'Profile number {b}: SATISFIED')

    # # --- Final summary
    # total_na = (satisfaction == 0).sum().item()
    # total_satisfied = (satisfaction == 1).sum().item()
    # total_violations = (satisfaction == -1).sum().item()
    # print('\n' + '='*50)
    # print('[IIA SUMMARY]')
    # print(f'Total batches: {B}')
    # print(f'Not applicable: {total_na}')
    # print(f'Satisfied: {total_satisfied}')
    # print(f'Violations: {total_violations}')
    # print('='*50)
    # # ---
    
    return satisfaction

