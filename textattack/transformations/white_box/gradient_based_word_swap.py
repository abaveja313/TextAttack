import torch

from textattack.shared import utils
from textattack.shared.validators import validate_model_gradient_word_swap_compatibility
from textattack.transformations import Transformation

class GradientBasedWordSwap(Transformation):
    """ Uses the model's gradient to suggest replacements for a given word.
        
        Based off of HotFlip: White-Box Adversarial Examples for Text 
            Classification (Ebrahimi et al., 2018).
        
            https://arxiv.org/pdf/1712.06751.pdf
        
        Arguments:
            model (nn.Module): The model to attack. Model must have a 
                `word_embeddings` matrix and `convert_id_to_word` function.
            top_n (int): the number of top words to return at each index
            replace_stopwords (bool): whether or not to replace stopwords
    """
    def __init__(self, model, top_n=1, replace_stopwords=False):
        validate_model_gradient_word_swap_compatibility(model)
        if not hasattr(model, 'word_embeddings'):
            raise ValueError('Model needs word embedding matrix for gradient-based word swap')
        if not hasattr(model, 'lookup_table'):
            raise ValueError('Model needs lookup table for gradient-based word swap')
        if not hasattr(model, 'zero_grad'):
            raise ValueError('Model needs `zero_grad()` for gradient-based word swap')
        if not hasattr(model.tokenizer, 'convert_id_to_word'):
            raise ValueError('Tokenizer needs `convert_id_to_word()` for gradient-based word swap')
        if not hasattr(model.tokenizer, 'pad_id'):
            raise ValueError('Tokenizer needs `pad_id` for gradient-based word swap')
        if not hasattr(model.tokenizer, 'oov_id'):
            raise ValueError('Tokenizer needs `oov_id` for gradient-based word swap')
        self.loss = torch.nn.CrossEntropyLoss()
        self.model = model
        self.pad_id = self.model.tokenizer.pad_id
        self.oov_id = self.model.tokenizer.oov_id
        self.top_n = top_n
        if replace_stopwords:
            self.stopwords = set()
        else:
            from nltk.corpus import stopwords
            self.stopwords = set(stopwords.words('english'))

    def _get_replacement_words_by_grad(self, text, indices_to_replace):
        """ Returns returns a list containing all possible words to replace
            `word` with, based off of the model's gradient.
            
            Arguments:
                text (TokenizedText): The full text input to perturb
                word_index (int): index of the word to replace
        """
        self.model.train()
       
        lookup_table = self.model.lookup_table.to(utils.get_device())
        lookup_table_transpose = lookup_table.transpose(0,1)
        
        # set backward hook on the word embeddings for input x
        emb_hook = Hook(self.model.word_embeddings, backward=True)
    
        self.model.zero_grad()
        predictions = self._call_model(text)
        original_label = predictions.argmax()
        y_true = torch.Tensor([original_label]).long().to(utils.get_device())
        loss = self.loss(predictions, y_true)
        loss.backward()
    
        # grad w.r.t to word embeddings
        emb_grad = emb_hook.output[0].to(utils.get_device()).squeeze()
    
        # grad differences between all flips and original word (eq. 1 from paper)
        vocab_size = lookup_table.size(0)
        diffs = torch.zeros(len(indices_to_replace), vocab_size)
        for j, word_idx in enumerate(indices_to_replace):
            # Get the grad w.r.t the one-hot index of the word.
            b_grads = emb_grad[word_idx].view(1,-1).mm(lookup_table_transpose).squeeze()
            a_grad = b_grads[text.ids[0][word_idx]]
            diffs[j] = b_grads-a_grad
        
        # Don't change to the pad token.
        diffs[:, self.model.tokenizer.pad_id] = 0
        
        # Find best indices within 2-d tensor by flattening.
        word_idxs_sorted_by_grad = (-diffs).flatten().argsort()
        
        candidates = []
        num_words_in_text, num_words_in_vocab = diffs.shape
        for idx in word_idxs_sorted_by_grad.tolist():
            idx_in_diffs = idx // num_words_in_vocab
            idx_in_vocab = idx % (num_words_in_vocab)
            idx_in_sentence = indices_to_replace[idx_in_diffs]
            word = self.model.tokenizer.convert_id_to_word(idx_in_vocab)
            if not has_letter(word): 
                # Do not consider words that are solely letters or punctuation.
                continue
            candidates.append((word, idx_in_sentence))
            if len(candidates) == self.top_n:
                break
            
        self.model.eval()
        return candidates
    
    def _call_model(self, text):
        """ A helper function to query `self.model` with TokenizedText `text`.
        """
        ids = torch.tensor(text.ids[0])
        ids = ids.to(next(self.model.parameters()).device)
        ids = ids.unsqueeze(0)
        return self.model(ids)

    def __call__(self, tokenized_text, indices_to_replace=None):
        """
        Returns a list of all possible transformations for `text`.
            
        If indices_to_replace is set, only replaces words at those indices.
        
        """
        words = tokenized_text.words
        if not indices_to_replace:
            indices_to_replace = list(range(len(words)))
        
        transformations = []
        word_swaps = []
        # Don't replace stopwords.
        indices_to_replace = [i for i in indices_to_replace if not (words[i].lower() in self.stopwords)]
        transformations = []
        for word, idx in self._get_replacement_words_by_grad(tokenized_text, indices_to_replace):
            transformations.append(tokenized_text.replace_word_at_index(idx, word))
        return transformations

def has_letter(word):
    for c in word:
        if c.isalpha(): return True
    return False
    
class Hook:
    def __init__(self, module, backward=False):
        if backward:
            self.hook = module.register_backward_hook(self.hook_fn)
        else:
            self.hook = module.register_forward_hook(self.hook_fn)
            
    def hook_fn(self, module, input, output):
        self.input = [x.to(utils.get_device()) for x in input]
        self.output = [x.to(utils.get_device()) for x in output]
        
    def close(self):
        self.hook.remove()