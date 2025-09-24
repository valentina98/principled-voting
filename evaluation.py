import numpy as np
import torch
import utils_dicts
import utils
from itertools import product
from utils import tensor_to_profile_list
from utils import PAD_PREF

def axiom_satisfaction(
    rule_mapping,
    profile_tensor,  # [B, V, A]
    axioms_check_model  # list of axiom names to check (e.g. ['Borda', ...])
):
    axioms_dict = {
        name: utils_dicts.DICT_AXIOMS_ALL[name]
        for name in axioms_check_model
        if name in utils_dicts.DICT_AXIOMS_ALL
    }

    results = {}
    for axiom_name, axiom_fn in axioms_dict.items():

        try:
            sat_tensor = axiom_fn(rule_mapping, profile_tensor)  # [B] with -1, 0, 1
        except Exception as e:
            print(f"[Axiom Failure] '{axiom_name}' raised: {e}")
            continue

        applicable_mask = sat_tensor != 0
        num_applicable = applicable_mask.sum().item()

        metrics = {
            'cond_satisfaction': (  # satisfaction conditioned on the axiom being applicable
                (sat_tensor[applicable_mask] == 1).float().mean().item()
                if num_applicable > 0 else float('nan')
            ),
            'absolute_satisfaction': (sat_tensor != -1).float().mean().item(),
            'percent_applicable': applicable_mask.float().mean().item(),
        }

        # print(f'[axiom_satisfaction DEBUG] Axiom: {axiom_fn} | num_applicable: {num_applicable}')

        results[axiom_name] = metrics

    return results

def rule_similarity(
        rule_mapping, 
        profile_tensor,  # [B, V, A]
        rule_comparison_list  # list of rule names to compare against (e.g. ['Borda', ...]
        ):

    B, V, A = profile_tensor.shape
    preds = rule_mapping(profile_tensor)  # [B, A]

    # Convert tensor profiles to Profile objects
    profiles = tensor_to_profile_list(profile_tensor)

    similarities = {}
    for rule_name in rule_comparison_list:
        rule_fn = utils_dicts.dict_rules_all_fast[rule_name]
        # Apply rule to each Profile object
        ref_preds_list = []
        for prof in profiles:
            winners = rule_fn(prof)  # expects Profile object
            vec = utils.winners_to_binary(winners, A)  # binary list
            ref_preds_list.append(vec)
        
        ref_preds = torch.tensor(ref_preds_list, dtype=torch.float32)

        hamming = (preds != ref_preds).sum(dim=1).float() / A
        identity_accu = (preds == ref_preds).all(dim=1).float()
        overlap_accu = ((preds.bool() & ref_preds.bool()).sum(dim=1) > 0).float()  # 1D tensor of 0s and 1s
        subset_accu = (preds <= ref_preds).all(dim=1).float()
        superset_accu = (preds >= ref_preds).all(dim=1).float()

        similarities[rule_name] = {
            'hamming': hamming.mean().item(),
            'identity_accu': identity_accu.mean().item(),
            'overlap_accu': overlap_accu.mean().item(),
            'subset_accu': subset_accu.mean().item(),
            'superset_accu': superset_accu.mean().item(),
        }

    return similarities

def admissibility(rule_mapping, profile_tensor):
    B, V, A = profile_tensor.shape

    preds = rule_mapping(profile_tensor)  # [B, A]

    valid_mask = profile_tensor.ne(PAD_PREF).any(dim=1)  # [B, A]

    has_winner = preds.bool().any(dim=1)  # at least one predicted winner
    has_valid_winner = (preds.bool() & valid_mask).any(dim=1)  # at least one admissible prediction
    all_preds_admissible = (preds.bool() & valid_mask).sum(dim=1) == preds.bool().sum(dim=1)

    no_winner_at_all = (~has_winner).float().mean().item()
    no_admissible_winner = (has_winner & ~has_valid_winner).float().mean().item()
    all_admissible_winner = (has_winner & all_preds_admissible).float().mean().item()
    some_inadmissible_winner = (has_winner & ~all_preds_admissible & has_valid_winner).float().mean().item()

    return {
        'no_winner': no_winner_at_all,
        'no_adm_winner': no_admissible_winner,
        'all_adm_winner': all_admissible_winner,
        'some_inadm_winner': some_inadmissible_winner,
    }

