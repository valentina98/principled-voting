import torch
import itertools
import os
from torch import nn
from random import sample
import math

# Word2Vec and FastText are a libraries for word embeddings
# They are used to convert words into vectors of numbers
from gensim.models import Word2Vec #, FastText 


class WEC(nn.Module):
    def __init__(self, 
                 max_num_voters: int, 
                 max_num_alternatives: int
                 ):
        super(WEC, self).__init__()
        self.max_num_voters = max_num_voters
        self.max_num_alternatives = max_num_alternatives
        
    def _preprocess_profiles(self, profile_tensor):
        B, V, A = profile_tensor.shape

        profile_sentences = [
            [self.ranking_to_string(profile_tensor[b, v].tolist()) for v in range(V)]
            for b in range(B)
        ]

        unique_words = {word for sentence in profile_sentences for word in sentence}
        word_to_index = {word: self.word_embeddings.wv.key_to_index.get(word, self.unk_idx) for word in unique_words}

        batch_indices = torch.tensor([
            [word_to_index[word] for word in sentence]
            for sentence in profile_sentences
        ], dtype=torch.long).view(B, self.max_num_voters)

        return batch_indices

    def _zero_pad_embedding(self):
        if self.pad_idx is not None:
            with torch.no_grad():
                self.embeddings.weight[self.pad_idx].zero_()

    # Call this on a model instance to initialize embeddings
    def initialize_embeddings(self, 
                              architecture_parameters, 
                              X_for_pretraining=None, 
                              random_seed=None, 
                              train_embeddings=True  # in the experiments we always train them
                              ):
        we_corpus_size = architecture_parameters.get('we_corpus_size', int(1e4))
        we_size = architecture_parameters.get('we_size', 100)
        we_window = architecture_parameters.get('we_window', self.max_num_alternatives)
        we_algorithm = architecture_parameters.get('we_algorithm', 1)

        print('[WEC] Pretraining word embeddings')

        # Convert tensor of profiles to tokenized sentences (one sentence per profile)
        train_sentences = [
            [WEC.ranking_to_string(ranking) 
            for ranking in profile.tolist()]
            for profile in X_for_pretraining
        ]

        # Add 'PAD' and 'UNK' tokens
        train_sentences = [['PAD'], ['UNK']] + train_sentences

        if we_corpus_size and len(train_sentences) > we_corpus_size:
            # ToDo: not limit by default?
            train_sentences = train_sentences[:we_corpus_size]  # limit embedding corpus

        pre_embeddings = Word2Vec(
            train_sentences,
            vector_size=we_size,
            window=we_window,
            min_count=1,
            workers=4,  # number of CPU cores to use
            sg=we_algorithm,
            seed=random_seed,
        )
        self.word_embeddings = pre_embeddings
        print('[WEC] Done pretraining word embeddings')
        
        print('[WEC] Number of word embedding tokens: ', len(self.word_embeddings.wv.key_to_index.keys()))
        tokens = sorted(self.word_embeddings.wv.key_to_index.keys())
        print('[WEC] Word embedding tokens: ', ', '.join(str(t) for t in tokens))

        self.embeddings = nn.Embedding.from_pretrained(  # the actual embeddings
            torch.FloatTensor(pre_embeddings.wv.vectors), freeze=not train_embeddings  
        )

        self.unk_idx = self.word_embeddings.wv.key_to_index['UNK']
        self.pad_idx = self.word_embeddings.wv.key_to_index['PAD']

        # Zero out the PAD embedding to avoid learning from it
        self._zero_pad_embedding()

        return pre_embeddings

    # Call this on a model instance to load embeddings
    def load_embeddings(self, path, train_embeddings=False):

        print(f'[WEC] Loading pretrained word embeddings from {path}')


        if os.path.exists(path):

            self.word_embeddings = Word2Vec.load(path)

            self.word_embeddings = Word2Vec.load(path, mmap='r')
            self.embeddings = nn.Embedding.from_pretrained(
                torch.FloatTensor(self.word_embeddings.wv.vectors), freeze=not train_embeddings
            )

            self.unk_idx = self.word_embeddings.wv.key_to_index['UNK']
            self.pad_idx = self.word_embeddings.wv.key_to_index['PAD']

            # Zero out the PAD embedding to avoid learning from it
            self._zero_pad_embedding()

            print('[WEC] Number of word embedding tokens:', len(self.word_embeddings.wv))
            tokens = sorted(self.word_embeddings.wv.key_to_index.keys())
            print('[WEC] Word embedding tokens: ', ', '.join(str(t) for t in tokens))
            return self.word_embeddings
        else:
            raise FileNotFoundError(f'Embedding file {path} not found!')

    # Call this on a model instance after the word embedding are initialized/loaded
    def initialize_model(self, architecture_parameters):        
        # Extract model architecture hyperparameters
        hidden_size = architecture_parameters.get('hidden_size', 256)
        layer_norm = architecture_parameters.get('layer_norm', True)  # helps with training stability
        dropout = architecture_parameters.get('dropout', 0.0)
                
        self.linear_hidden1 = nn.Linear(self.word_embeddings.vector_size, hidden_size)
        self.linear_hidden2 = nn.Linear(hidden_size, hidden_size)
        self.linear_hidden3 = nn.Linear(hidden_size, hidden_size)
        self.linear_output = nn.Linear(hidden_size, self.max_num_alternatives)

        if layer_norm:
            self.norm1 = nn.LayerNorm(hidden_size)
            self.norm2 = nn.LayerNorm(hidden_size)
            self.norm3 = nn.LayerNorm(hidden_size)
        else:
            self.norm1 = self.norm2 = self.norm3 = nn.Identity()

        self.dropout = nn.Dropout(p=dropout)
        self.relu = nn.ReLU()

    # This can be called after the embeddings and model are initialized
    def forward(self, x):
        # ToDo: x should be accepted either with 2 or 3 dimensions
        if x.dim() == 2 and x.dtype == torch.long:
            # Input is already token indices
            token_indices = x
        elif x.dim() == 3:
            # Raw profile tensor [B, V, A]
            token_indices = self._preprocess_profiles(x)
        else:
            raise ValueError(f'Unsupported input shape: {x.shape} and dtype {x.dtype}')

        if self.embeddings is None:
            raise ValueError('Word embeddings not initialized!')

        x = self.embeddings(token_indices)  # [batch, sentence_len, emb_dim]
        x = x.mean(dim=1)  
        # ToDo: What if we mask in the forward pass?
        # mask = token_indices.ne(self.pad_idx)  # [B, V] (True where NOT PAD)
        # lengths = mask.sum(dim=1, keepdim=True).clamp(min=1)  # [B, 1]  (# real voters)
        # x = (x * mask.unsqueeze(-1)).sum(dim=1) / lengths  # masked mean [B, D]

        x = self.dropout(self.relu(self.norm1(self.linear_hidden1(x))))
        x = self.dropout(self.relu(self.norm2(self.linear_hidden2(x))))
        x = self.dropout(self.relu(self.norm3(self.linear_hidden3(x))))
        x = self.linear_output(x)

        return x

    def _model2rule(self, profile_tensor, threshold=0.5):
        logits = self.forward(profile_tensor)  # [batch_size, num_alternatives]

        # print('[DEBUG WEC] logits', logits.shape, logits)

        binary_preds = (torch.sigmoid(logits) > threshold).long()

        ## ToDo: try this out
        ## Select winners: top-1 (or all max if tied)
        # max_vals, _ = logits.max(dim=1, keepdim=True)  
        # binary_preds = (logits == max_vals).long()

        # We don't mask out because we check the admissibility. But we could.
        # if not full:
        #     # Mask out completely padded alternatives
        #     valid_mask = profile_tensor.ne(self.pad_idx).any(dim=1)
        #     binary_preds = binary_preds * valid_mask

        return binary_preds

    def model2rule(self, threshold=0.5):
        return lambda profile_tensor: self._model2rule(profile_tensor, threshold=threshold)


    # ToDo: try model2logits for discrete axiom check (no sampling)
    # def _model2logits(self, profile_tensor):
    #     logits = self.forward(profile_tensor)  # [batch_size, num_alternatives]
    #     return logits

    # def model2logits(self):
    #     return lambda profile_tensor: self._model2rule(profile_tensor)
    

    @staticmethod
    def ranking_to_string(ranking):
        # Missing voter rankings are possible when full=False so we use a padding token
        # that will be ignored in the forward pass
        if all(x == -1 for x in ranking):
            return 'PAD'
        
        # For example: 210, 10-1, PAD, 021, etc
        return ''.join(map(str, map(int, ranking)))

