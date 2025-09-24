import os
import torch
import itertools
import math
import numpy as np
from torch.utils.data import Dataset
from pref_voting.generate_profiles import generate_profile, generate_profile_with_groups

from utils import winners_to_binary, tensor_to_profile_list, PAD_PREF, PAD_WIN
import utils_dicts


class PrefVotingDataset(Dataset):
    '''
    A dataset for preference voting profiles.

    - If dataset_path exists, we load numeric data from disk.
    - Otherwise, we generate raw data (list of profiles) and convert them to numeric.
    - If a voting rule is set, we produce a multi-label target Y for each profile.

    Note: samples are the number of elections. In each sample there is a set number of voters and alternatives.
    '''
    def __init__(
        self,
        dataset_path,
        max_num_voters,
        max_num_alternatives,
        election_sampling,
        num_samples=None,  # if None, the full dataset is generated
        random_seed=None,
        voting_rule=None,  # if specified, labels will be generated using this rule
        dataset_type='standard',  # 'standard', 'condorcet_only'
        dtype=torch.int8,  # data type for the profiles
    ):
        super().__init__()
        self.dataset_path = dataset_path
        self.max_num_voters = max_num_voters
        self.max_num_alternatives = max_num_alternatives
        self.election_sampling = election_sampling
        self.num_samples = num_samples
        self.random_seed = random_seed
        self.voting_rule = voting_rule
        self.dataset_type = dataset_type
        self.dtype = dtype

        # Placeholders
        self.X = None
        self.Y = None
        
        # Additional flags
        self._exhaustive = False
        
        # If dataset size wasn't provided, an exhaustive dataset is expected
        if num_samples is None:
            self.num_samples = sum(math.factorial(m)**n
                                   for n in range(1, self.max_num_voters + 1)
                                   for m in range(1, self.max_num_alternatives + 1))
            print(f'[PrefVotingDataset] The number of samples for exhaustive enumeration is {self.num_samples}.')

            # If eval dataset is too big, we have to sample. Even with lazy loading it wouldn't be feasible
            # to do full evaluation on 5x5 profiles
            if self.num_samples > 300000:  
                print(f'[PrefVotingDataset] The number of samples to evaluate is too big, so 300,000 random samples will be used.')
                self.num_samples = 300000
                # ToDo: ensure IC distribution is used
            else:
                self._exhaustive = True  # generate all possible profiles

        self._prepare_data()  # initialize the data


    # Dataset interface

    def __len__(self):
        return self.num_samples
    
    def __getitem__(self, idx):
        return (self.X[idx], self.Y[idx]) if self.Y is not None else self.X[idx]
            
    def __repr__(self):
        return f'<PrefVotingDataset: {len(self)} samples, rule={self.voting_rule}>' # , arch={self.architecture} todo
    
    #  Private methods

    def _prepare_data(self):
        '''Check for existing dataset on disk; if not present, generate, encode, and save.'''
        
        if os.path.exists(self.dataset_path):
            # Load from disk
            print(f'[PrefVotingDataset] A dataset file is found, loading dataset...')
            saved_data = torch.load(self.dataset_path)
            self.X = saved_data['X']
            self.Y = saved_data.get('Y')  # None if no labels are saved
            if self.num_samples and len(self.X) < self.num_samples: 
                raise Exception(f"[PrefVotingDataset] Loaded dataset has {len(self.X)} samples, but {self.num_samples} are needed according to 'num_samples'! Remove the file or provide a different 'dataset_path'.")
            if self.voting_rule and self.Y is None:  # a dataset based on a voting rule should have labels
                raise Exception(f"[PrefVotingDataset] Warning: 'voting_rule' is set but no labels found in saved file.")
            print(f'[PrefVotingDataset] Dataset loaded from {self.dataset_path}')

        else:
            print('[PrefVotingDataset] No dataset file found. Generating dataset from scratch...')

            # Generate
            if self._exhaustive:
                print(f'[PrefVotingDataset] All possible profiles with up to {self.max_num_voters} voters and {self.max_num_alternatives} alternatives will be generated...')
                self._generate_raw_profiles_all()
            elif self.dataset_type == 'condorcet_only':  # generate only profiles with Condorcet winner
                print(f'[PrefVotingDataset] Generating {self.dataset_type} dataset with {self.num_samples} sample profiles...')
                self._generate_raw_profiles_condorcet()
            else:
                print(f'[PrefVotingDataset] Generating {self.num_samples} random sample profiles...')
                self._generate_raw_profiles_random()
            if self.voting_rule:
                print(f'[PrefVotingDataset] Finding the winners using the voting rule: {self.voting_rule}...')
                self._find_winners()

            # Save to disk
            torch.save({'X': self.X, 'Y': self.Y}, self.dataset_path) #  rankings in X, one-hot encoded winners in Y
            print(f'[PrefVotingDataset] Dataset generated and saved to {self.dataset_path}')

        print(f"[PrefVotingDataset] The dataset contains: {len(self.X)} samples | {len(self.Y) if self.Y is not None else 'no'} labels | {self.max_num_voters} voters | {self.max_num_alternatives} alternatives")


    # ToDo: simplify using Do Not Repeat Yourself (DRY) principle
    def _generate_raw_profiles_all(self):  # ToDo test
        '''
        Generates all possible election profiles for the given maximum number of voters and alternatives.
        Smaller profiles are padded to fixed dimensions using -1 for missing entries.
        '''
        self.X = torch.full((self.num_samples, self.max_num_voters, self.max_num_alternatives), PAD_PREF, dtype=self.dtype)

        count = 0
        for nv in range(1, self.max_num_voters + 1):  # 1, ..., max_num_voters
            for na in range(1, self.max_num_alternatives + 1):  # 1, ..., max_num_alternatives
                all_rankings = list(itertools.permutations(range(na)))  # generate all rankings for 0 to na including
                # print('DEBUG all_rankings', all_rankings)
                for ranking_combo in itertools.product(all_rankings, repeat=nv):  # generate all combinations of rankings for nv
                    self.X[count, :nv, :na] = torch.tensor([list(p) for p in ranking_combo], dtype=self.dtype)
                    count += 1
                
                
    def _generate_raw_profiles_random(self):
        '''
        Generates a batch of random election profiles from a distribution and their corresponding 
        winners (if a voting rule is provided). Profiles are padded to fixed dimensions
        using -1 for missing entries.
        '''
        self.X = torch.full((self.num_samples, self.max_num_voters, self.max_num_alternatives), PAD_PREF, dtype=self.dtype)
        
        # Generate profile sets with maximum or less voters and alternatives
        print(f'[PrefVotingDataset] Profiles with up to {self.max_num_voters} voters and {self.max_num_alternatives} alternatives will be generated...')
        
        if self.election_sampling.get('seed') is not None:
            raise ValueError('[PrefVotingDataset] Setting a seed for election sampling will generate the same profile throughout the dataset.')

        # We should sample more often the bigger setups where we can have more interesting profiles
        # We can decide the probability of na and nv happening as:
        total = sum(sum(math.factorial(na)**nv
                for na in range(1, self.max_num_alternatives + 1))
                for nv in range(1, self.max_num_voters + 1))
        na_probabilities = [(na, sum(math.factorial(na)**i for i in range(1, self.max_num_voters + 1))/total)
                            for na in range(1, self.max_num_alternatives + 1)]
        nv_probabilities = [(nv, sum(math.factorial(j)**nv for j in range(1, self.max_num_alternatives + 1))/total)
                            for nv in range(1, self.max_num_voters + 1)]
        # print(f'[PrefVotingDataset DEBUG] total: {total}')
        # print(f'[PrefVotingDataset DEBUG] na_probabilities: {na_probabilities}')
        # print(f'[PrefVotingDataset DEBUG] nv_probabilities: {nv_probabilities}')

        # Decide how often to sample each setup
        weights = [wa*wv for _, wa in na_probabilities for _, wv in nv_probabilities]
        setups = [(na, nv) for na, _ in na_probabilities for nv, _ in nv_probabilities]
        # Ensure the total counts sum to num_samples
        counts = np.random.multinomial(self.num_samples, weights)
        count = 0
        for (na, nv), num_profs in zip(setups, counts):
            if num_profs == 0:
                continue
            profs = generate_profile(na, nv, num_profiles=num_profs, **self.election_sampling)
            if not isinstance(profs, list):  # if only one profile is generated, it is not returned as a list
                profs = [profs]
            profs = list(map(lambda p: p.rankings, profs))  # extract rankings from profiles
            self.X[count:count+num_profs, :nv, :na] = torch.tensor(profs, dtype=self.dtype)
            count += num_profs

        # Shuffle the dataset to ensure setups are mixed; assign to object attribute
        self.X = self.X[torch.randperm(self.num_samples)]
              
    def _find_winners(self):
        self.Y = torch.full(
            (self.num_samples, self.max_num_alternatives),
            PAD_WIN, dtype=self.dtype
        )

        profiles_list = tensor_to_profile_list(self.X)  # convert the batch tensor to a list of Profile objects

        for i in range(self.num_samples):
            rule_fn = utils_dicts.DICT_RULES_ALL[self.voting_rule]
            if rule_fn is None:
                raise ValueError(f"[PrefVotingDataset] Voting rule '{self.voting_rule}' not found in the dictionary")
            winners = rule_fn(profiles_list[i])  # the list of winners, e.g [0, 2] for winning alternatives 0 and 2
            # print(f'[DEBUG] winners {winners}')
            winners = winners_to_binary(winners, self.max_num_alternatives)  # convert to binary list (e.g [1, 0, 1, 0] for na=4 and winners [0, 2])
            self.Y[i, :len(winners)] = torch.tensor(winners, dtype=self.dtype)  # max_num_alternatives >= len(winners)


    # Public methods
    
    def get_dataset_splits(self, split_sizes, return_y=False): # e.g. [800, 200, 200]
        assert sum(split_sizes) == len(self.X), 'Sum of splits must equal dataset size.' 

        g = torch.Generator()
        if self.random_seed is not None:
            g.manual_seed(self.random_seed)

        # ToDo: here deterministic should be fine?
        subsets = torch.utils.data.random_split(self.X, split_sizes, generator=g)

        if return_y:
            return [(self.X[subset.indices], self.Y[subset.indices]) for subset in subsets]
        
        return [self.X[subset.indices] for subset in subsets]

    def get_full_dataset(self, return_y=False):
        '''Returns the full dataset as a tensor.'''
        if return_y and self.Y is not None:
            return self.X, self.Y
        elif return_y:
            raise ValueError('No labels available in this dataset.')
        return self.X
