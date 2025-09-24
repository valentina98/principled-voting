import os
import time
import numpy as np
import random
import json
import itertools
import torch
from torch import nn
from torch.utils.data import TensorDataset, DataLoader
from datetime import datetime

from pref_voting_dataset import PrefVotingDataset
from visualizations import Visualizations
from models.mlp import MLP
from models.cnn import CNN
from models.wec import WEC
from models.set_transformer import SetTransformer


def experimentBorda(
        architecture,
        max_num_voters,
        max_num_alternatives,
        election_sampling,
        num_gradient_steps,
        report_intervals,
        dev_dataset_size=100,
        eval_dataset_size=None,  # leave None for full evaluation dataset
        architecture_parameters={},
        random_seed=None,
        batch_size=64,  # ToDo: change to 32
        learning_rate=1e-3,
        prediction_threshold=0.5,
        save_model=False,
        voting_rule='Borda',
        # Note: No learning_scheduler and weight_decay; we try to overfit to the voting rule and regularization only makes that harder
    ):     
    '''
    Runs a training experiment to learn a voting rule from profile data, using the specified neural network architecture.

    Args:
        architecture (str): Model architecture to use ('WEC').
        max_num_voters (int): Maximum number of voters allowed in a profile.
        max_num_alternatives (int): Maximum number of alternatives (candidates) allowed in a profile.
        election_sampling (dict): Dictionary describing the sampling method for generating profiles (e.g., {'probmodel': 'IC'}).
        num_gradient_steps (int): Number of training steps (gradient updates). Based on this is determined the size of the training set.
        report_intervals (int): Number of steps between evaluations on the dev set.
        dev_dataset_size (int): Size of the development dataset. If None, uses a default size of 100.
        eval_dataset_size (int): Size of the evaluation dataset. If None, uses the full dataset size based on max_num_voters and max_num_alternatives.
        architecture_parameters (dict, optional): Architecture-specific parameters. If not provided, defaults are used.
        random_seed (int, optional): Random seed for reproducibility. Defaults to None.
        batch_size (int, optional): Batch size for training. Defaults to 64.
        learning_rate (float, optional): Learning rate for the optimizer. Defaults to 1e-3.
        prediction_threshold (float, optional): Threshold for deciding winners from model outputs. Defaults to 0.5.
        save_model (bool, optional): Whether to save the trained model. Defaults to False.
        load_model_from (str, optional): Path to a previously saved model checkpoint to load. Defaults to None.
        voting_rule (str, optional): Name of the voting rule used to generate labels. Defaults to 'Borda'.

    Returns:
        location (str): Path to the directory where results and model are saved.

    Saves model and dataset, logs results, and plots the learning curve.
    '''

    # -------------------------------------------------------------------------
    # SET UP BASICS (Seeds, Paths, Distance Functions)
    # -------------------------------------------------------------------------
    timing_log = {}
    start_time = time.time()

    assert (
        architecture in ['MLP', 'CNN', 'WEC', 'SetTransformer', 'PointerNetworks']  # ToDo Deep Sets don't work
    ), f"The supported architectures are 'MLP', 'CNN', 'WEC', 'SetTransformer' and 'PointerNetworks' but '{architecture}' was specified."

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

    # Worker seed function for deterministic DataLoader 
    # Pytorch automatically sets a different one for each worker as long
    # as a torch.Generator is passed with the seed
    def seed_worker(worker_id):
        # Ensures deterministic but different seeds for each DataLoader worker
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
    location = f'./results/exp_borda/{architecture}/exp_{current_time}_{prob_model}'
    os.makedirs(location, exist_ok=True)
    print(f'[ExpBorda] Saving location: {location}')


    # -------------------------------------------------------------------------
    # GENERATING DATA
    # -------------------------------------------------------------------------
    
    gen_data_start = time.time()

    total_train_size = batch_size * (num_gradient_steps + 1)  # generate at least 1 train batch
    
    # Construct dataset path according to the parameters
    prefix = f'{voting_rule.lower()}_' if voting_rule else ''  # empty string if no voting rule is specified
    train_dataset_path = f'results/{prefix}{max_num_voters}v_{max_num_alternatives}a_{total_train_size}_train.pt'
    dev_dataset_path = f'results/{prefix}{max_num_voters}v_{max_num_alternatives}a_{dev_dataset_size}_dev.pt'
    if eval_dataset_size is None:
        eval_dataset_path = f'results/{prefix}{max_num_voters}v_{max_num_alternatives}a_eval.pt'
    else:
        eval_dataset_path = f'results/{prefix}{max_num_voters}v_{max_num_alternatives}a_{eval_dataset_size}_eval.pt'

    # Generate labeled datasets with given size (load from file or generates new)
    dataset_train = PrefVotingDataset(
        dataset_path=train_dataset_path,
        max_num_voters=max_num_voters,
        max_num_alternatives=max_num_alternatives,
        election_sampling=election_sampling,
        num_samples=total_train_size,  # total size of the training and dev datasets
        random_seed=random_seed,
        voting_rule=voting_rule,  # we need labels according to the voting rule
    )
    dataset_dev = PrefVotingDataset(
        dataset_path=dev_dataset_path,
        max_num_voters=max_num_voters,
        max_num_alternatives=max_num_alternatives,
        election_sampling=election_sampling,
        num_samples=dev_dataset_size,  # total size of the training and dev datasets
        random_seed=random_seed,
        voting_rule=voting_rule,  # we need labels according to the voting rule
    )
    # Generate complete labeled evaluation dataset
    dataset_test = PrefVotingDataset(
        dataset_path=eval_dataset_path, # test dataset is saved in the same location with '_test' suffix
        max_num_voters=max_num_voters,
        max_num_alternatives=max_num_alternatives,
        election_sampling=election_sampling,
        num_samples=None,  # the full dataset of all permutations for the max_num_voters and max_num_alternatives will be generated
        random_seed=random_seed,
        voting_rule=voting_rule,  # we need labels according to the voting rule
    )

    print(f'[ExpBorda] Partition sizes | Train: {dataset_train.num_samples} profiles | Dev: {dataset_dev.num_samples} | Test: {dataset_test.num_samples} profiles')

    # Create DataLoaders for batching
    train_loader = DataLoader(
        dataset_train, 
        batch_size=batch_size, 
        shuffle=True, 
        num_workers=8,  # adjust based on hardware
        worker_init_fn=seed_worker,
        generator=g,  # this makes worker-level shuffling deterministic
    )
    # Note: dev loader is not deeded because the dev set is small and passed through the model at once
    test_loader = DataLoader(dataset_test, batch_size=batch_size, shuffle=False)

    timing_log['generate_data'] = time.time() - gen_data_start

    # -------------------------------------------------------------------------
    # NEURAL NETWORK INITIALIZATION
    # -------------------------------------------------------------------------
    
    model_init_start = time.time()

    print('[ExpBorda] Now initializing the model')

    if architecture == 'MLP':
        exp_model = MLP(
            max_num_voters=max_num_voters,
            max_num_alternatives=max_num_alternatives,
            architecture_parameters=architecture_parameters
        )
    elif architecture == 'CNN':
        exp_model = CNN(
            max_num_voters=max_num_voters,
            max_num_alternatives=max_num_alternatives,
            architecture_parameters=architecture_parameters
        )
    elif architecture == 'WEC':
        exp_model = WEC(max_num_voters, max_num_alternatives)
        # Pretrain embeddings with the full list of possible permutations of voters and alternatives
        exp_model.initialize_embeddings(architecture_parameters, dataset_train.X, random_seed)
        exp_model.initialize_model(architecture_parameters)
        if save_model:
            print('[ExpBorda] Saving the word embeddings')
            exp_model.word_embeddings.save(f'{location}/pre_embeddings.bin')
    elif architecture == 'SetTransformer':
        exp_model = SetTransformer(
            max_num_voters=max_num_voters,
            max_num_alternatives=max_num_alternatives,
            architecture_parameters=architecture_parameters
        )
    elif architecture == 'PointerNetworks':
        raise NotImplementedError('PointerNetworks architecture is not implemented yet.')

    exp_optimizer = torch.optim.AdamW(
        exp_model.parameters(), 
        lr=learning_rate,
    )

    loss_fn = nn.BCEWithLogitsLoss()  # combines sigmoid + binary cross-entropy; accepts logits directly
        
    timing_log['model_initialization'] = time.time() - model_init_start


    # -------------------------------------------------------------------------
    # NEURAL NETWORK TRAINING
    # -------------------------------------------------------------------------
    
    print('[ExpBorda] Now starting to train')
    train_start = time.time()
    learning_curve = {}
    train_loss_sum = 0.0
    train_loss_count = 0
    exp_model.train()
    loader_cycle = itertools.cycle(train_loader)
    
    for step in range(num_gradient_steps + 1):  # no update on the 0th step, just eval
        X_train_batch, Y_train_batch = next(loader_cycle)
        
        logits = exp_model(X_train_batch)  # Shape: [batch_size, num_alternatives]
        loss = loss_fn(logits, Y_train_batch.float())  # BCE expects float targets

        # Log spikes or NaNs
        if torch.isnan(loss): # this should not happen (just a safeguard)
            print(f'[Step {step}] ❌ NaN loss encountered. Halting training.')
            break
        if loss.item() > 1e3: # this should not happen unless the training is going extremely wrong
            print(f'[Step {step}] ⚠️ Loss spike: {loss.item():.4f}')

        if step != 0:  # no update on the 0th step, just eval
            # Backpropagation
            exp_optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(exp_model.parameters(), max_norm=1.0)  # prevent gradients from exploding (by limiting their norm)
            exp_optimizer.step()
        
        # Keep track of the training loss
        train_loss_sum += loss.item()  # gradients are not tracked because of .item() 
        train_loss_count += 1

        # EVALUATION ON DEV SET
        if step % report_intervals == 0:
            exp_model.eval()
            with torch.no_grad():

                # --- Average Train Loss ---
                # Compute average train loss over the interval
                if train_loss_count > 0:
                    train_loss_avg = train_loss_sum / train_loss_count
                else:
                    train_loss_avg = float('nan')  # safety
                # Reset counters for next interval
                train_loss_sum = 0.0
                train_loss_count = 0

                # --- Dev Loss ---
                dev_logits = exp_model(dataset_dev.X)
                dev_preds = (torch.sigmoid(dev_logits) >= prediction_threshold).long()
                dev_loss = loss_fn(dev_logits, dataset_dev.Y.float())

                dev_exact_match = (dev_preds == dataset_dev.Y).all(dim=1)
                dev_exact_match_acc = dev_exact_match.sum().item() / dev_dataset_size

                dev_hamming_dist = (dev_preds != dataset_dev.Y).float().sum(dim=1)
                dev_hamming_acc = (1 - dev_hamming_dist / dataset_dev.Y.float().size(1)).sum().item() / dev_dataset_size

                learning_curve[f'{step}'] = {
                    'train_loss_avg': float(train_loss_avg),
                    'dev_loss': float(dev_loss),
                    'dev_exact_match_acc': float(dev_exact_match_acc),
                    'dev_hamming_acc': float(dev_hamming_acc),
                }

                print(f'[Step {step:5d}] Avg Train Loss: {train_loss_avg:.4f} | Dev Loss: {dev_loss:.4f} | Exact Match: {dev_exact_match_acc:.2%} | Hamming: {dev_hamming_acc:.2%}')

                exp_model.train()
       
    timing_log['training_total'] = time.time() - train_start
    print('[ExpBorda] Finished training')

    if save_model:
        # We save both the model state and the optimizer state to be able to 
        # continue training later on.
        torch.save({
            'arguments' : [max_num_voters, max_num_alternatives, architecture_parameters],
            'model_state_dict': exp_model.state_dict(),
            'optimizer_state_dict': exp_optimizer.state_dict()
            }, f'{location}/model.pth')


    # -------------------------------------------------------------------------
    # FINAL EVALUATION ON TEST SET
    # -------------------------------------------------------------------------

    print('[ExpBorda] Evaluating on the test set')

    test_evaluation_start = time.time()

    exp_model.eval()
    with torch.no_grad():
        test_loss_sum = 0.0
        exact_match_count = 0
        hamming_acc_sum = 0.0

        for X_test_batch, Y_test_batch in test_loader:
            test_logits_batch = exp_model(X_test_batch)

            # Collect eval loss
            test_loss_batch = loss_fn(test_logits_batch, Y_test_batch.float())
            test_loss_sum += test_loss_batch.item()  
            
            test_preds_batch = (torch.sigmoid(test_logits_batch) >= prediction_threshold).long()

            # Collect exact matches (all labels correct)
            exact_match_count += (test_preds_batch == Y_test_batch.long()).all(dim=1).sum().item()

            # Collect hamming accuracy  (1 - mean Hamming distance of the batch)
            hamming_acc_batch = 1.0 - (test_preds_batch != Y_test_batch.long()).float().mean(dim=1)
            hamming_acc_sum += hamming_acc_batch.sum().item()

            # print('DEBUG: Test batch loss:', test_loss_batch.item())
            # print('DEBUG: Test batch exact match count:', exact_match_count)
            # print('DEBUG: Test batch hamming accuracy sum:', hamming_acc_sum)

        test_loss_avg = test_loss_sum / dataset_test.num_samples  # average loss over all batches
        test_exact_match_acc = exact_match_count / dataset_test.num_samples
        test_hamming_acc = hamming_acc_sum / dataset_test.num_samples
            
    test_results = {
        'test_loss': float(test_loss_avg),
        'test_exact_match_acc': float(test_exact_match_acc),
        'test_hamming_acc': float(test_hamming_acc),
    }

    timing_log['test_evaluation_total'] = time.time() - test_evaluation_start


    # -------------------------------------------------------------------------
    # SAVE RESULTS
    # -------------------------------------------------------------------------

    results_file_location = f'{location}/results.json'

    # Store training configuration
    config = {
        'location': location,
        'architecture': architecture,
        'max_num_voters': max_num_voters,
        'max_num_alternatives': max_num_alternatives,
        'election_sampling': election_sampling,
        'architecture_parameters': architecture_parameters,
        'num_gradient_steps': num_gradient_steps,
        'report_intervals': report_intervals,
        'dev_dataset_size': dev_dataset_size,
        'eval_dataset_size': eval_dataset_size,
        'random_seed': random_seed,
        'batch_size': batch_size,        
        'learning_rate': learning_rate,
        'prediction_threshold': prediction_threshold,
        'save_model': save_model,
        'voting_rule': voting_rule,
    }

    # Store evaluation results
    results = {
        'test_results': test_results,
        'total_runtime': timing_log.get('total_runtime', 0),
        'learning_curve': learning_curve,
    }

    # Save config and results to file
    with open(results_file_location, 'w') as json_file:
        json.dump({**config, **results}, json_file, indent=4)

    print(f'[ExpBorda] Final results saved to {results_file_location}')
    

    # FINAL EVALUATION REPORT
    print('\n=== Final Evaluation on Test Set ===')
    print(f'Test Loss             : {test_loss_avg:.4f}')
    print(f'Exact Match Accuracy  : {test_exact_match_acc:.2%}')
    print(f'Hamming Accuracy      : {test_hamming_acc:.2%}')

    # TIME REPORT
    end_time = time.time()
    timing_log['total_runtime'] = end_time - start_time

    print('\n=== TIME REPORT ===')
    for key, value in timing_log.items():
        print(f'{key}: {value / 60:.2f} min')
    print('=====================\n')
    

    # PLOT LEARNING CURVE
    Visualizations.plot_learning_curve_supervised(location)

    return location
