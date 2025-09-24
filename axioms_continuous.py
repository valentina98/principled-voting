from pref_voting.profiles import Profile
from random import sample, shuffle
import torch
from torch import nn
import torch.nn.functional as F

from utils import profiles_to_ranks, PAD_PREF

# Note: Crafting semantic losses is tricky. The variables on which
# the output depends require gradients and are used for the updates.
# Even a 0 value can affect the loss depending on how it is assigned
# to be 0. A general good practice is to compare model output with 
# a target output in a matrix form.


# Distance functions for computing the continuous versions of axioms.
# The predictions x are always expected as logits but the targets y can be:
# - logits: arbitrary numbers in [-inf, inf] with a given mean and variance
# - probability distribution: numbers in [0, 1] that sum to 1
# - non normalized probabilities: numbers in [0, 1] that do not sum to 1
# - one-hot mask: values in [0, 1] with only one 1
# - multilabel mask: values in [0, 1] with multiple 1s
distance_functions = {
    # DISTANCES - metrics satisfying mathematical properties like symmetry and triangle inequality
    # - don't distinguish between confident vs. uncertain predictions
    # - less sensitive to small errors than KLD, suboptimal for classification
    # - use only when comparing raw outputs (e.g., to enforce invariance axioms)
    # Euclidean distance
    # Expects prediction logits and target logits 
    'L2': lambda x, y: torch.mean(nn.PairwiseDistance(p=2)(x, y)),
    # Expects prediction logits and target probability distribution
    # Suboptimal compared to other losses
    'L2_prob': lambda x, y: torch.mean(nn.PairwiseDistance(p=2)(x.softmax(dim=1), y)),
    # Expects prediction logits and target multi-label mask
    'L2_mask_normalized': lambda x, y: torch.mean(
        nn.PairwiseDistance(p=2)(x.softmax(dim=1), y / y.sum(dim=1, keepdim=True).clamp(min=1e-8))
    ),
    # Cosine similarity between x and y vectors
    # Expects prediction logits and target logits
    'cos_sim': lambda x, y: -torch.mean(nn.CosineSimilarity(dim=1, eps=1e-8)(x, y)),
    # Expects prediction logits and target probability distributions
    'cos_sim_prob': lambda x, y: -torch.mean(nn.CosineSimilarity(dim=1, eps=1e-8)(x.softmax(dim=1), y)),

    # DIVERGENCES - measure of dissimilarity between probability distributions
    # The original Kullback-Leibler divergence
    # Expects prediction logits and target logits
    # useful to guide the model to match the same logits after permutations
    # e.g for anonymity, neutrality, independence, etc.
    'KLD': lambda x, y: nn.KLDivLoss(reduction='none', log_target=True)(
        # Note: reduction='batchmean' could be used but we do it manually to be able to apply masks on the output
        x.log_softmax(dim=1), 
        y.log_softmax(dim=1)  # Note: y should be logits!
    ),
    # Clamped for stability - useful when the target y is sparse or masked (e.g. y=[5,-100,0])
    # Note: after clamping, y is no longer a proper log-probability distribution
    'KLD_safe': lambda x, y: nn.KLDivLoss(reduction='none', log_target=True)(
        # Note: reduction='batchmean' could be used but we do it manually to be able to apply masks on the output
        x.log_softmax(dim=1),
        torch.clamp(y.log_softmax(dim=1), min=-10)  # , max=0 
    ),
    # Expects prediction logits and target probability distribution
    'KLD_prob': lambda x, y: nn.KLDivLoss(reduction='none')(  
        # Note: reduction='batchmean' could be used but we do it manually to be able to apply masks on the output
        x.log_softmax(dim=1),
        y
    ),
    # Expects prediction logits and target probability distribution
    # clamping is added to avoid zero values in y which would cause NaNs from log(0)
    # But after clamping y no longer sums to 1 and is no longer a strict distribution
    'KLD_prob_safe': lambda x, y: nn.KLDivLoss(reduction='none')(
        # Note: reduction='batchmean' could be used but we do it manually to be able to apply masks on the output
        x.log_softmax(dim=1), 
        torch.clamp(y, min=1e-8)
    ), 
    # Expects prediction logits and target multilabel mask
    # The mask is normalized to form a valid distribution for KL divergence
    'KLD_prob_normalized': lambda x, y: nn.KLDivLoss(reduction='none')(
        # Note: reduction='batchmean' could be used but we do it manually to be able to apply masks on the output
        x.log_softmax(dim=1), 
        y / y.sum(dim=1, keepdim=True)
    ),
    # Jensen-Shannon divergence - a smoother symmetric version of KLD
    # Expects prediction and target logits
    'JS_divergence_safe': lambda x, y: 0.5 * nn.KLDivLoss(reduction='none')(
        x.log_softmax(dim=1),
        ((x.softmax(dim=1) + y.softmax(dim=1)) / 2).clamp(min=1e-8)
    ) + 0.5 * nn.KLDivLoss(reduction='none')(
        y.log_softmax(dim=1),
        ((x.softmax(dim=1) + y.softmax(dim=1)) / 2).clamp(min=1e-8)
    ),

    # LOSSES - functions quantifying prediction error
    # Squared error between predicted and target distributions
    # Expects prediction and target logits
    # This is valid only for a single-target or soft-label setting
    'MSE': lambda x, y: nn.MSELoss()(
        x.softmax(dim=1),
        y.softmax(dim=1)
    ),
    # Squared error between predicted and target distributions
    # Expects prediction logits and target multilabel mask
    # Suitable for multi-label setups, allows independent probability per class
    'MSE_prob_normalized': lambda x, y: nn.MSELoss()(
        x.sigmoid(),
        y
    ),
    # Binary Cross-Entropy loss for logits
    # Expects prediction logits and target multilabel mask
    # This is more stable than MSE1
    # 'BCE': lambda x, y: nn.BCEWithLogitsLoss()(x, y)
    'BCE': lambda x, y: nn.BCEWithLogitsLoss(reduction='none')(x, y)  
    # We don't apply reduction so that we can apply a mask to the output and ignore positions
    # Note: this is not the same as masking the logits + masking the target (it is a different optimization problem)
}

def ax_no_winners_cont(model, profiles_tensor, threshold=0.5):  
    valid_mask = profiles_tensor.ne(PAD_PREF).any(dim=1).float()  # [B, V, A]
    logits = model(profiles_tensor)  # [B, A]
    masked_probs = torch.sigmoid(logits) * valid_mask  # [B, A]
    max_probs = masked_probs.max(dim=1).values  # [B]
    # We want at least one of the probabilities for each profile in 
    # to be above the threshold, so the loss, i.e., how much this is not fulfilled, is 
    # threshold - highest prob, and 0 if threshold < highest prob
    loss = torch.clamp(threshold - max_probs, min=0.0)  # [B]
    return loss.mean()  # average across batch

def ax_inadmissibility_cont(model, profiles_tensor, distance=distance_functions['BCE']):
    # B, V, A = profiles_tensor.shape
    inad_mask = profiles_tensor.eq(PAD_PREF).all(dim=1).float()  # [B, A]
    valid_mask = profiles_tensor.ne(PAD_PREF).any(dim=1).float()  # [B, A]
    logits = model(profiles_tensor)  # [B, A]
    # Get nonaveraged distances and take only the inadmissible positions into account
    distances = distance(logits, valid_mask) * inad_mask   
    distance_per_prof = distances.sum(dim=1) #/ inad_mask.sum(dim=1).clamp_min(1)
    return distance_per_prof.mean()

# We want the probability of the Condorcet winner to be 1 so we use KLD_prob with
# a target mask, which ignores the distance of the other alternatives.
# We don't use BCE because we are not doing binary classification (we don't care
# about the non Condorcet winners).
def ax_condorcet1_cont(model, profiles_tensor, distance=distance_functions['KLD_prob']):
    B, V, A = profiles_tensor.shape
    # valid_mask = profiles_tensor.ne(PAD_PREF).any(dim=1).float()  # [B, A]
    logits = model(profiles_tensor)  # [B, A]
    # masked_logits = logits * valid_mask  # [B, A]
    target = torch.zeros_like(logits)  # [B, A]
    for i in range(B):
        valid_rankings = [
            [int(x) for x in ranking.tolist() if x != PAD_PREF]
            for ranking in profiles_tensor[i]
            if (ranking != PAD_PREF).any()
        ]
        winner = Profile(valid_rankings).condorcet_winner()
        if winner is not None:
            target[i, winner] = 1.0

    # Note: We only care about the distance of the Condorcet winner to winning.
    # Logits corresponding to other valid or invalid alternatives are not be taken into account.
    distances = distance(logits, target) * target  # apply target as a mask to ignore other non-Condorcet winner alternatives
    return distances.sum(dim=1).mean()  # average loss across batch


# Note: This is a soft constraint, not a classification task.
# The model is only penalized for assigning low score to the Condorcet winner.
# It doesn't penalize other alternatives or enforce a full ranking, so it isn't
# effective on its own and should be used to complement other axioms.
def ax_condorcet2_cont(model, profiles_tensor):
    B, V, A = profiles_tensor.shape
    # preds = torch.sigmoid(model(profiles_tensor))  # [B, A]
    # target = torch.zeros_like(preds)  # [B, A]
    logits = model(profiles_tensor)  # [B, A]
    target = torch.zeros_like(logits)  # [B, A]
    for i in range(B):
        valid_rankings = [
            [int(x) for x in ranking.tolist() if x != PAD_PREF]
            for ranking in profiles_tensor[i]
            if (ranking != PAD_PREF).any()
        ]
        winner = Profile(valid_rankings).condorcet_winner()
        if winner is not None:
            target[i, winner] = 1.0

    # correct_preds = (preds * target).sum(dim=1)  # [B]
    # condorcet_mask = target.sum(dim=1) == 1  # [B] mask for profiles with Condorcet winners
    # return (1 - correct_preds)[condorcet_mask].mean()
    
    # condorcet_mask = target == 1  # [B, A] mask for Condorcet winners
    # distances = distance(logits, target) * target 
    # return distances.sum(dim=1).mean()  # average loss across batch


def ax_independence_cont(model, profiles_tensor, distance=distance_functions['KLD']):
    B, V, A = profiles_tensor.shape
    logits = model(profiles_tensor)  # [B, A]

    valid_mask = profiles_tensor.ne(PAD_PREF)  # [B, V, A]
    # Require >= 2 voters and >= 3 alternatives per profile, independence is not defined otherwise
    valid_voters_mask = valid_mask.any(dim=2)  # [B, V], a voter is valid if they rank at least one alternative
    num_valid_voters = valid_voters_mask.sum(dim=1)  # [B]
    valid_alts_mask = valid_mask.any(dim=1)  # [B, A], an alternative is valid if any voter ranks it
    num_valid_alts = valid_alts_mask.sum(dim=1)  # [B]
    nontrivial_mask = (num_valid_voters >= 2) & (num_valid_alts >= 3)  # [B]
    if not nontrivial_mask.any():
        return torch.tensor(0.0, dtype=torch.float)

    # Create a copy and shuffle only irrelevant alternatives
    perm_profiles_tensor = profiles_tensor.clone()
    for b in range(B):
        if not nontrivial_mask[b]:
            # print(f'Profile {profiles_tensor[b]} is trivial. Skipping it.')
            continue  # no permutations applied, independence is meaningless for <= 2 alternatives

        for v in range(V):
            ranking = perm_profiles_tensor[b, v][perm_profiles_tensor[b, v] != PAD_PREF].tolist()
            # print('perm_profiles_tensor[b, v]', perm_profiles_tensor[b, v])
            # print('ranking', ranking)
            if len(ranking) == 0:  # skip padded voters
                continue

            # Relevant pick alternatives p and q
            p, q = sample(ranking, 2)
            pos = {a: idx for idx, a in enumerate(ranking)}
            if pos[p] > pos[q]:
                p, q = q, p  # swap so that p is the preferred alternative than q

            # Obtain a ranking where irrelevant alternatives are shuffled and the relative order of p and q is kept
            others = [x for x in ranking if x not in (p, q)]
            shuffle(others)  # randomize the irrelevant set
            insert_pos_p, insert_pos_q = sorted(sample(range(len(ranking)), 2))  # position of p is before position of q
            new_ranking = others[:insert_pos_p] + [p] + others[insert_pos_p:insert_pos_q-1] + [q] + others[insert_pos_q-1:]
            perm_profiles_tensor[b, v] = torch.tensor(new_ranking + [PAD_PREF] * (A - len(new_ranking)))

    logits_permuted_ia = model(perm_profiles_tensor)  # [B, A]

    # Note: we don't need to mask out trivial cases as they are not permuted and the distance is 0.
    # Should we mask the invalid alternatives?
    distances = distance(logits, logits_permuted_ia) # * valid_alts_mask
    return distances.sum(dim=1).mean()  # average loss across batch
