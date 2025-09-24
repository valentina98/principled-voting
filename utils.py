from random import shuffle, randint, sample
import torch
import torch.nn.functional as F
from pref_voting.profiles import Profile

PAD_PREF = -1  # the padding value for rankings in X. A full padded row of alternatives means a padded voter.
PAD_WIN = 0  # the target value for invalid winners in Y. Note that Y is one-hot encoded

def winners_to_binary (winners_list, A):
    '''
    Recasts winning set into characteristic function (binary list) 

    Voting rules output a list of winners among the alternatives (cast as list). 
    Sometimes we turn this into a binary list where the n-th entry is 1 iff the 
    n-th alternative is a winner.

    Example: winner_to_vec([0,1,3], 5) = [1, 1, 0, 1, 0]
    '''
    vec = []
    for i in range(A):
        vec.append(int(i in winners_list))
    return vec

def profiles_tensor_to_onehot(profiles_tensor):
    '''
    Converts a batch of padded profile tensors [B, V, A]
    into one-hot encoded tensors [B, V, A, A].
    Padded alternatives (with rank PAD_PREF) are represented as all-zero vectors.
    '''
    B, V, A = profiles_tensor.shape
    mask = profiles_tensor != PAD_PREF
    return F.one_hot(profiles_tensor.masked_fill(~mask, 0).long(), num_classes=A) * mask.unsqueeze(-1).float()

def tensor_to_profile_list(batch_tensor):
    '''
    Converts a batch of padded profile tensors [B, V, A]
    into a list of pref_voting.Profile objects.
    '''
    profiles = []
    for election_tensor in batch_tensor:
        prof = []
        for voter_ranking in election_tensor:
            ranking = voter_ranking.tolist()
            if all(x == -1 for x in ranking):
                continue  # skip fully padded voters
            filtered_ranking = [int(x) for x in ranking if x != -1]
            prof.append(filtered_ranking)
        profiles.append(Profile(prof))
    return profiles

def pad_profile_tensor(rankings: list[list[int]], max_V: int, max_A: int) -> torch.Tensor:
    '''
    Pads a profile (list of rankings) to a fixed-size tensor using -1 padding.

    Args:
        rankings (list of list of int): Raw preference rankings.
        max_V (int): Max number of voters (rows).
        max_A (int): Max number of alternatives (columns).

    Returns:
        torch.Tensor: Tensor of shape [max_V, max_A], padded with -1.
    '''
    padded = torch.full((max_V, max_A), -1, dtype=torch.long)
    for i, ranking in enumerate(rankings):
        length = min(len(ranking), max_A)
        padded[i, :length] = torch.tensor(ranking[:length], dtype=torch.long)
    return padded


def positional_scoring_winners(profile, scoring_vector, m_max):
    '''
    Given a profile and a scoring_vector of length m_max,
    computes the winners using the scoring rule defined by that vector.
    For a profile with m alternatives (m = profile.num_cands), we use the 
    first m entries of scoring_vector.
    '''
    m = profile.num_cands
    # Use the first m entries as the rule for this profile.
    rule = scoring_vector[:m]
    alt_scores = [0.0 for _ in range(m)]
    for ranking in profile.rankings:
        # For each ranking, assign points according to positions
        for pos, alt in enumerate(ranking):
            if pos < m and alt in profile.candidates:
                alt_scores[alt] += rule[pos]
    max_score = max(alt_scores) if alt_scores else 0.0
    # Declare as winners all alternatives that achieve max_score.
    winners = [a for a, s in enumerate(alt_scores)
               if a in profile.candidates and abs(s - max_score) < 1e-9]
    return winners
    # ToDo: vectorize scoring instead of using for loops

def profiles_to_ranks(profiles_tensor):
    B, V, A = profiles_tensor.shape
    pt = profiles_tensor.to(torch.long)  # ensure integer IDs
    ranks = torch.full((B, V, A), PAD_PREF, dtype=torch.long)  # [B, V, A]
    mask = pt.ne(-1)  # valid positions
    b, v, j = mask.nonzero(as_tuple=True)  # indices of valid slots
    a = pt[b, v, j]  # alt IDs at those slots
    ranks[b, v, a] = j  # write position as rank
    return ranks

# The random rule
def rand_rule(prof):
    '''
    A voting rule that a outputs a random winning set, no matter the input
    '''
    list_of_alternatives = prof.candidates
    num_winners = randint(1,len(list_of_alternatives))
    list_winners = sample(list_of_alternatives, num_winners)
    return list_winners

