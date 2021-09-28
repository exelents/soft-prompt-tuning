import torch
import torch.nn as nn

class SoftEmbedding(nn.Module):
    def __init__(self, 
                wte: nn.Embedding,
                n_tokens: int = 10,
                n_prompts: int = 1,
                random_range: float = 0.5,
                initialize_from_vocab: bool = True):
        """appends learned embedding to 

        Args:
            wte (nn.Embedding): original transformer word embedding
            n_tokens (int, optional): number of tokens for task. Defaults to 10.
            random_range (float, optional): range to init embedding (if not initialize from vocab). Defaults to 0.5.
            initialize_from_vocab (bool, optional): initalizes from default vocab. Defaults to True.
        """
        super(SoftEmbedding, self).__init__()
        self.wte = wte
        self.n_tokens = n_tokens
        self.n_prompts = n_prompts
        self.learned_embedding = nn.parameter.Parameter(
            self.initialize_embedding(wte,
                                      n_tokens,
                                      n_prompts,
                                      random_range,
                                      initialize_from_vocab
                                      )
        )
            
    def initialize_embedding(self, 
                             wte: nn.Embedding,
                             n_tokens: int = 10,
                             n_prompts: int = 1,
                             random_range: float = 0.5, 
                             initialize_from_vocab: bool = True):
        """initializes learned embedding

        Args:
            same as __init__

        Returns:
            torch.float: initialized using original schemes
        """
        if initialize_from_vocab:
            return torch.cat([
                self.wte.weight[:n_tokens].clone().detach().unsqueeze(0)
                for _ in range(n_prompts)
            ], 0)
        return torch.FloatTensor(n_prompts, n_tokens, wte.weight.size(1)).uniform_(-random_range, random_range)
            
    def forward(self, tokens):
        """run forward pass

        Args:
            tokens (torch.long): input tokens before encoding

        Returns:
            torch.float: encoding of text concatenated with learned task specifc embedding
        """
        input_embedding = self.wte(tokens[:, self.n_tokens:])
        index_prompt_tokens = tokens[:, 0]
        if not torch.any(index_prompt_tokens == -1):
            learned_embedding = self.learned_embedding.index_select(0, index_prompt_tokens)
            return torch.cat([learned_embedding, input_embedding], 1)
        else:
            return input_embedding