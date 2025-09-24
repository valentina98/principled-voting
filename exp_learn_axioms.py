'''
The implementation of experiment 3
'''
import os
import time
import numpy as np
import random
import json
import itertools
from datetime import datetime
import torch
from torch import nn
from torch.utils.data import TensorDataset, DataLoader

from pref_voting_dataset import PrefVotingDataset
import utils_dicts
import evaluation
import axioms_continuous
from visualizations import Visualizations
from models.mlp import MLP
from models.cnn import CNN
from models.wec import WEC
from models.set_transformer import SetTransformer


def experimentAxioms(
        architecture,
        max_num_voters,
        max_num_alternatives,
        election_sampling,
        axioms_check_model,  # list of axioms to check for model
        axiom_opt,  # dict of axiom optimization parameters
        comp_rules_axioms,  # list of rules to compare for axioms
        comp_rules_similarity,  # list of rules to compare for similarity
        num_gradient_steps, 
        report_intervals,
        train_dev_dataset_type='standard',  # type of training dataset, 'condorcet_only' or 'standard'
        dev_dataset_size=100,
        eval_dataset_size=None,  # leave None for full evaluation dataset
        architecture_parameters={},
        random_seed=None,
        batch_size=32,
        distance=None,  # distance metric for continuous axioms
        learning_rate = 1e-3,
        prediction_threshold=0.5,
        learning_scheduler = None,
        weight_decay = 0,
        load_model_from = None,
        save_model=False,
    ):

    # -------------------------------------------------------------------------
    # SET UP BASICS (Seeds, Paths, Distance Functions, Axioms, Saving Location)
    # -------------------------------------------------------------------------
    timing_log = {}
    start_time = time.time()

    assert (
        architecture in ['MLP', 'CNN', 'WEC', 'SetTransformer']  # ToDo Deep Sets don't work
    ), f"The supported architectures are 'MLP', 'CNN', 'WEC', 'SetTransformer' and 'PointerNetworks' but '{architecture}' was specified."

    if distance is None:
        distance_fn = None
        print(f'[Exp3] Using default distance functions for each axiom')
    else:
        assert distance in axioms_continuous.distance_functions, \
        f"The supported distances are None, {', '.join(list(axioms_continuous.distance_functions.keys()))}, but '{distance}' was given"
        distance_fn = axioms_continuous.distance_functions[distance]
        print(f'[Exp3] Using distance function: {distance}')

    # Set global seeds
    if random_seed is not None:
        random.seed(random_seed)
        np.random.seed(random_seed)
        torch.manual_seed(random_seed)
        torch.cuda.manual_seed(random_seed)
        torch.use_deterministic_algorithms(True)
        torch.cuda.manual_seed_all(random_seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

    # Worker seed function for DataLoader
    def seed_worker(worker_id):  # worker id is automatic
        '''
        Ensures deterministic but different seeds for each DataLoader worker.
        '''
        worker_seed = torch.initial_seed() % 2**32
        np.random.seed(worker_seed)
        random.seed(worker_seed)

    # Set up the generator for DataLoader shuffling
    g = torch.Generator()
    if random_seed is not None:
        g.manual_seed(random_seed)

    # Set up saving location
    prob_model = election_sampling['probmodel']
    current_time = datetime.now().strftime('%Y-%m-%d_%H-%M-%S')
    location = f'./results/exp3/{architecture}/exp3_{current_time}_{prob_model}'
    os.makedirs(location, exist_ok=True)
    print(f'Saving location: {location}')


    # -------------------------------------------------------------------------
    # GENERATING DATA
    # -------------------------------------------------------------------------
    
    gen_data_start = time.time()

    total_train_size = batch_size * (num_gradient_steps + 1)  # training set is generated with at least one batch
    train_dev_dataset_size = total_train_size + dev_dataset_size
    
    # Construct dataset path according to the parameters
    train_dev_type_suffix = f'_{train_dev_dataset_type}' if train_dev_dataset_type != 'standard' else ''
    train_dev_dataset_path = f'results/unsupervised_{max_num_voters}v_{max_num_alternatives}a_{train_dev_dataset_size}{train_dev_type_suffix}.pt'
    if eval_dataset_size is None:
        eval_dataset_path = f'results/unsupervised_{max_num_voters}v_{max_num_alternatives}a_eval.pt'
    else:
        eval_dataset_path = f'results/unsupervised_{max_num_voters}v_{max_num_alternatives}a_{eval_dataset_size}_eval.pt'

    if total_train_size:
        # Generate training and development datasets with the requested size
        dataset_train_dev = PrefVotingDataset(
            dataset_path=train_dev_dataset_path,
            max_num_voters=max_num_voters,
            max_num_alternatives=max_num_alternatives,
            election_sampling=election_sampling,
            num_samples=train_dev_dataset_size,
            random_seed=random_seed,
            dataset_type=train_dev_dataset_type,
        )
        
        X_train_profs, X_dev_profs = \
            dataset_train_dev.get_dataset_splits(
                [total_train_size, dev_dataset_size]
            )
        train_dataset = TensorDataset(X_train_profs)
        train_loader = DataLoader(  # dataLoader with deterministic shuffle
            train_dataset, 
            batch_size=batch_size, 
            shuffle=True, 
            num_workers=8,  # adjust based on hardware
            worker_init_fn=seed_worker,
            generator=g,  # makes worker-level shuffling deterministic
            )

    # Generate full evaluation dataset
    dataset_test_full = PrefVotingDataset(
        dataset_path=eval_dataset_path,
        max_num_voters=max_num_voters,
        max_num_alternatives=max_num_alternatives,
        election_sampling=election_sampling,
        num_samples=eval_dataset_size,
        random_seed=random_seed,
    )
    X_test_profs = dataset_test_full.get_full_dataset()
    test_dataset = TensorDataset(X_test_profs)
    test_loader = DataLoader(
        test_dataset, 
        batch_size=batch_size, 
        shuffle=False
        )

    timing_log['generate_data'] = time.time() - gen_data_start


    # -------------------------------------------------------------------------
    # NEURAL NETWORK INITIALIZATION
    # -------------------------------------------------------------------------
    
    model_init_start = time.time()
    print('[Exp3] Now initializing the model')


    # ToDo: X should be one-hot when we pass it to the models other than WEC. Remember to convert
    if architecture == 'MLP':
        exp_model = MLP(max_num_voters, max_num_alternatives, architecture_parameters)
    elif architecture == 'CNN':
        exp_model = CNN(max_num_voters, max_num_alternatives, architecture_parameters)
    elif architecture == 'WEC':
        exp_model = WEC(max_num_voters, max_num_alternatives)
    
        # Initialize/load the embeddings
        load_embeddings_from = architecture_parameters.get('load_embeddings_from', None)       
        if load_embeddings_from:
            embeddings_path = os.path.join(load_embeddings_from, 'pre_embeddings.bin')
            exp_model.load_embeddings(path=embeddings_path)

            we_corpus_size = architecture_parameters.get('we_corpus_size')
            we_size = architecture_parameters.get('we_size')
            we_window = architecture_parameters.get('we_window')
            we_algorithm = architecture_parameters.get('we_algorithm')
            if we_corpus_size is not None or we_size is not None or we_window is not None or we_algorithm is not None:
                print(f'[Exp3 Warning] Some architecture parameters for the WEC model are ignored because embeddings are loaded from {load_embeddings_from}')
        else:
            # Pretrain embeddings with the full list of possible permutations of voters and alternatives
            exp_model.initialize_embeddings(architecture_parameters, X_test_profs, random_seed)
        
        exp_model.initialize_model(architecture_parameters)
        if save_model and location:
            print('[Exp3] Saving the word embeddings')
            exp_model.word_embeddings.save(os.path.join(location, 'pre_embeddings.bin'))
    elif architecture == 'SetTransformer':
        exp_model = SetTransformer(max_num_voters, max_num_alternatives, architecture_parameters)

    # Load weights
    if load_model_from:
        checkpoint = torch.load(f'{load_model_from}/model.pth')
        exp_model.load_state_dict(checkpoint['model_state_dict'])
        print(f'[Exp3] Loaded model weights from {load_model_from}/model.pth')

    exp_optimizer = torch.optim.AdamW(
        exp_model.parameters(), 
        lr=learning_rate,
        weight_decay=weight_decay
    )

    if load_model_from is not None and 'optimizer_state_dict' in checkpoint:
        exp_optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        print(f'[Exp3] Loaded optimizer state from {load_model_from}/model.pth')

    if learning_scheduler is not None:
        scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(
            exp_optimizer, 
            T_0 = learning_scheduler
        )
        print(f'[Exp3] Learning scheduler initialized with T_0 = {learning_scheduler}')

    active_axioms = []
    for ax_name, ax_config in axiom_opt.items():
        spec = utils_dicts.DICT_AXIOMS_CONT.get(ax_name)
        if not spec:
            raise ValueError(f'[Exp3]: Axiom  "{ax_name}" was not found in the configured axioms for optimization.')
        kwargs = {'distance': distance_fn} if ('distance' in spec and distance_fn is not None) else {}
        weight = ax_config['weight']
        active_axioms.append((ax_name, spec['fn'], weight, kwargs))

    if not active_axioms:
        raise ValueError('[Exp3] No axioms are selected for optimization - training will do nothing.')
    else:
        print('[Exp3] Active axioms for optimization:',
            ', '.join(f"{n} (weight: {w})" for n,_,w,_ in active_axioms))
            
    # Print discrete axioms for evaluation
    print('[Exp3] Selected axioms for evaluation:', axioms_check_model)

    timing_log['model_initialization'] = time.time() - model_init_start


    # -------------------------------------------------------------------------
    # NEURAL NETWORK TRAINING
    # -------------------------------------------------------------------------
    
    print('[Exp3] Now starting to train')
    train_start = time.time()
    learning_curve = {}
    train_loss_sum = 0.0
    train_loss_count = 0
    exp_model.train()
    # if total_train_size != 0: # ToDo test if 0
    loader_cycle = itertools.cycle(train_loader)

    for step in range(num_gradient_steps + 1):  # no update on the 0th step, just eval
        train_batch = next(loader_cycle)   
        X_train_batch = train_batch[0]
                    
        # Compute loss using active axioms and schedule
        loss_terms = {}
        for axiom, loss_fn, weight, kwargs in active_axioms:
            loss_terms[axiom] = weight * loss_fn(exp_model, X_train_batch, **kwargs)
        loss = torch.stack(list(loss_terms.values())).sum() / len(loss_terms)  # average loss

        # Backpropagation
        if step != 0:  # no update on the 0th step, just eval
            exp_optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(exp_model.parameters(), max_norm=1.0)  # prevent gradients from exploding (by limiting their norm)
            
            if loss.item() > 0:  # avoid noisy updates when no violations are present
                exp_optimizer.step()
                if learning_scheduler is not None:
                    scheduler.step()

        # Keep track of the training loss
        train_loss_sum += loss.item()
        train_loss_count += 1

        # EVALUATION ON DEV SET
        # if (report_intervals > 0 and step % report_intervals == 0) or (step == num_gradient_steps - 1):
        if step % report_intervals == 0:
            exp_model.eval()
            with torch.no_grad():

                # --- Average Train Loss ---
                if train_loss_count > 0:
                    train_loss_avg = train_loss_sum / train_loss_count
                else:
                    train_loss_avg = float('nan')  # safety

                # Reset counters for next interval
                train_loss_sum = 0.0
                train_loss_count = 0

                # --- Dev Loss ---
                dev_loss_terms = {}
                for axiom, loss_fn, weight, kwargs in active_axioms:
                    dev_loss_terms[axiom] = weight * loss_fn(exp_model, X_dev_profs, **kwargs)
                dev_loss = torch.stack(tuple(dev_loss_terms.values())).mean().item()
                # dev_loss = (sum(dev_loss_terms.values()) / len(dev_loss_terms)).item()  # same as above

                print(f'[Step {step:5d}] Average train Loss: {train_loss_avg:.4f} | Dev Loss: {dev_loss:.4f}')

                # --- Admissibility Evaluation ---
                rule_mapping = exp_model.model2rule(threshold=prediction_threshold)  # learned mapping of profiles to winners

                admissibility = evaluation.admissibility(rule_mapping, X_dev_profs)
                adm_str = ' | '.join(f'{k} {v:.2f}' for k, v in sorted(admissibility.items()))
                print(f'[Step {step:5d}] Admissibility: {adm_str}')

                # --- Axiom Satisfaction Evaluation ---
                axiom_satisfactions = evaluation.axiom_satisfaction(
                    rule_mapping=rule_mapping,
                    profile_tensor=X_dev_profs,
                    axioms_check_model=axioms_check_model,
                )
                sat_str = ' | '.join(f"{k} {v['cond_satisfaction']*100:.1f}%" for k, v in sorted(axiom_satisfactions.items()))
                print(f'[Step {step:5d}] Axiom Satisfaction: {sat_str}')

                # --- Rule Similarity Evaluation ---
                similarity_results = evaluation.rule_similarity(
                        rule_mapping, X_dev_profs, comp_rules_similarity
                    )
                sim_str = ' | '.join(f"{k} {v['identity_accu']*100:.1f}%" for k, v in sorted(similarity_results.items()))
                print(f'[Step {step:5d}] Rule Similarity: {sim_str}')

                # Save learning curve
                learning_curve[f'{step}'] = {
                    'train_loss_avg': train_loss_avg,
                    'dev_loss': dev_loss,
                    'dev_loss_terms': {k: v.item() for k, v in dev_loss_terms.items()},
                    'admissibility': admissibility,
                    'axiom_satisfaction': axiom_satisfactions,
                    'similarity_results': similarity_results,
                }

                exp_model.train()

    timing_log['training_total'] = time.time() - train_start
    print('[Exp3] Finished training')


    if save_model:
        # Save both the model state and the optimizer state 
        # to be able to continue training
        torch.save({
            'arguments' : [max_num_voters, max_num_alternatives, architecture_parameters],
            'model_state_dict': exp_model.state_dict(),
            'optimizer_state_dict': exp_optimizer.state_dict()
            }, f'{location}/model.pth')


    # -------------------------------------------------------------------------
    # FINAL EVALUATION ON TEST SET
    # -------------------------------------------------------------------------
    print('\n=== FINAL EVALUATION ON TEST SET ===')

    exp_model.eval()
    with torch.no_grad():

        # --- Loss Evaluation ---
        test_loss_start = time.time()
        
        test_loss = 0.0
        num_batches = 0
        for batch in test_loader:
            X_test_batch = batch[0]  # unpack the batch

            test_loss_terms = {}
            for axiom, loss_fn, weight, kwargs in active_axioms:
                if axiom not in axiom_opt:
                    continue
                test_loss_terms[axiom] = weight * loss_fn(exp_model, X_test_batch, **kwargs)
            
            # Note the last batch may be smaller
            test_loss += torch.stack(tuple(test_loss_terms.values())).mean().item()  
            num_batches += 1

        avg_test_loss = test_loss/num_batches if num_batches > 0 else float('nan')
        print(f'\nAverage Test Loss: {avg_test_loss:.4f}')

        timing_log['test_loss_evaluation'] = time.time() - test_loss_start

        # --- Admissibility Evaluation ---
        rule_mapping = exp_model.model2rule(threshold=prediction_threshold)  # learned mapping of profiles to winners

        print('\n=== Admissibility Evaluation on Test Set ===')
        admissibility_test_start = time.time()

        admissibility_summary_test = compute_admissibility_batched(rule_mapping, test_loader)
        for metric_name, metric_value in sorted(admissibility_summary_test.items()):
            print(f'    {metric_name:<30}: {metric_value:.4f}')

        timing_log['admissibility_evaluation_test'] = time.time() - admissibility_test_start

        # --- Axiom Satisfaction Evaluation ---
        print('\n=== Axiom Satisfaction Evaluation on Test Set ===')
        axiom_satisfaction_test_start = time.time()

        axiom_satisfaction_test = compute_axiom_satisfaction_batched(
            rule_mapping=rule_mapping,
            data_loader=test_loader,
            axioms_check_model=axioms_check_model,
        )
        for axiom_name, metrics in sorted(axiom_satisfaction_test.items()):
            cond_sat = metrics['cond_satisfaction'] * 100
            print(f'    {axiom_name:<30}: {cond_sat:.2f}%')

        timing_log['axiom_satisfaction_evaluation_test'] = time.time() - axiom_satisfaction_test_start

        # --- Rule Similarity Evaluation ---
        print('\n=== Rule Similarity Evaluation on Test Set ===')
        rule_similarity_test_start = time.time()

        similarities_test = compute_rule_similarity_batched(
            rule_mapping=rule_mapping,
            data_loader=test_loader,
            rule_comparison_list=comp_rules_similarity,
        )
        for rule_name, result in sorted(similarities_test.items()):
            identity_acc = result['identity_accu'] * 100
            print(f'    {rule_name:<30}: {identity_acc:.2f}%')

        timing_log['rule_similarity_evaluation_test'] = time.time() - rule_similarity_test_start

    # -------------------------------------------------------------------------
    # SAVE RESULTS
    # -------------------------------------------------------------------------

    results_file = f'{location}/results.json'

    end_time = time.time()
    timing_log['total_runtime'] = end_time - start_time

    # Training configuration
    config = {
        'location': location,
        'architecture': architecture,
        'max_num_voters': max_num_voters,
        'max_num_alternatives': max_num_alternatives,
        'election_sampling': election_sampling,
        'num_gradient_steps': num_gradient_steps,
        'report_intervals': report_intervals,
        'eval_dataset_size': eval_dataset_size,
        'architecture_parameters': architecture_parameters,
        'axioms_check_model': axioms_check_model,
        'axiom_opt': axiom_opt,
        # 'comp_rules_axioms': comp_rules_axioms,  # ToDo: not ued?
        'comp_rules_similarity': comp_rules_similarity,
        'distance': distance,
        'random_seed': random_seed,
        'batch_size': batch_size,
        'learning_rate': learning_rate,
        'learning_scheduler': learning_scheduler,
        'weight_decay': weight_decay,
        'save_model': save_model,
        'load_model_from': load_model_from,
    }

    # Evaluation results
    results = {
        'avg_test_loss': avg_test_loss,
        'admissibility': admissibility_summary_test,
        'axiom_satisfaction_model': axiom_satisfaction_test,
        'rule_similarity': similarities_test,
        'learning_curve': learning_curve,
        'timing_log': timing_log,
    }

    # Save everything to a file
    with open(results_file, 'w') as json_file:
        json.dump({**config, **results}, json_file, indent=4)

    print(f'\nFinal results saved to {results_file}')
    

    # Time report
    print('\n=== TIME REPORT ===')
    for key, value in timing_log.items():
        print(f'{key:<40}: {value / 60:.2f} min')
    print('=====================\n')

    # Plot learning curve
    Visualizations.plot_learning_curve_unsupervised(location)

    return location


def compute_admissibility_batched(rule_mapping, data_loader):

    aggregated_results = {}
    for batch in data_loader:
        batch = batch[0]  # unpack the batch (get X)
        batch_results = evaluation.admissibility(rule_mapping, batch)

        # Gather results
        for key, value in batch_results.items():
            if key not in aggregated_results:
                aggregated_results[key] = []
            aggregated_results[key].append(value)

    # Aggregate results over batches (the same size batches are assumed)
    return {key: sum(values) / len(values) for key, values in aggregated_results.items()}

def compute_axiom_satisfaction_batched(rule_mapping, data_loader, axioms_check_model):

    aggregated_lists = {}
    for batch in data_loader:
        batch = batch[0]  # unpack the batch (get X)

        batch_results = evaluation.axiom_satisfaction(
            rule_mapping=rule_mapping,
            profile_tensor=batch,
            axioms_check_model=axioms_check_model
        )

        # Aggregate batch results
        for name in batch_results:
            if name not in aggregated_lists:
                aggregated_lists[name] = {}
            for metric_key, metric_val in batch_results[name].items():
                if metric_key not in aggregated_lists[name]:
                    aggregated_lists[name][metric_key] = []
                aggregated_lists[name][metric_key].append(metric_val)

    # Average results per axiom and metric
    averaged_results = {
        name: {
            metric: sum(values) / len(values) if values else float('nan')
            for metric, values in metrics.items()
        }
        for name, metrics in aggregated_lists.items()
    }

    return averaged_results

def compute_rule_similarity_batched(rule_mapping, data_loader, rule_comparison_list):
    aggregated_results = []

    for batch, in data_loader:
        batch_result = evaluation.rule_similarity(
            rule_mapping, batch, rule_comparison_list
        )
        aggregated_results.append(batch_result)

    # Aggregate metrics across batches (same size batches are assumed)
    final_results = {}
    for rule_name in rule_comparison_list:
        final_results[rule_name] = {
            'identity_accu': sum(batch[rule_name]['identity_accu'] for batch in aggregated_results) / len(aggregated_results)
        }

    return final_results




