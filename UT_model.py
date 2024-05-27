import math
import logging
import torch
import torch.nn as nn
import torch.nn.functional as F

from ACT import ACTModule

logger = logging.getLogger(__name__)

class MultiHeadCausalAttention(nn.Module):
    def __init__(self, embed_size, heads):

        super(MultiHeadCausalAttention, self).__init__()
        self.embed_size = embed_size 
        self.heads = heads 
        # Ensure the embedding size is divisible by the number of heads for equal division
        assert embed_size % heads == 0, "Embedding size needs to be divisible by heads"

        # Define linear transformations for queries, keys, and values
        self.values = nn.Linear(embed_size, embed_size)
        self.keys = nn.Linear(embed_size, embed_size)
        self.queries = nn.Linear(embed_size, embed_size)
        # Output projection layer
        self.proj_out = nn.Linear(embed_size, embed_size)

    def forward(self, x):
        # Extract dimensions: batch size, sequence length, and embedding size
        batch_size, seq_len, embed_size = x.size()

        # Process queries, keys, values: split into 'heads' number of heads, each with head dimension (embed_size/heads)
        # Using .view we divide queries, keys and values for each head
        ## Before transpose: [batch_size, seq_len, heads, embed_size/heads] | After transpose: [batch_size, heads, seq_len, embed_size/heads]
        all_queries = self.queries(x).view(batch_size, seq_len, self.heads, embed_size//self.heads).transpose(1,2)
        all_keys = self.keys(x).view(batch_size, seq_len, self.heads, embed_size//self.heads).transpose(1,2)      
        all_values = self.values(x).view(batch_size, seq_len, self.heads, embed_size//self.heads).transpose(1,2)  

        queries_keys = (all_queries @ all_keys.transpose(-1,-2)) * (1/math.sqrt(all_keys.size(-1)))

        # Create a causal mask to mask out future tokens (prevent attending to future positions)
        causal_mask = torch.triu(torch.ones(seq_len, seq_len) * float('-inf'), diagonal=1).to(queries_keys.device)

        # Apply the causal mask by adding it to the scaled dot-product scores
        masked_queries_keys = queries_keys + causal_mask[None, None, :, :] 

        attn_score = F.softmax(masked_queries_keys, dim=-1) 

        out = (attn_score @ all_values).transpose(1,2).contiguous().view(batch_size, seq_len, embed_size)  

        # Apply the final linear projection layer
        out = self.proj_out(out)  # Transform back to original embed size

        return out


class TransitionFunction(nn.Module):
    """TransitionFunction Block is the position-wise feed-forward, just becuase it's named so 
    in the universal tranformer, I adopted it here"""
    def __init__(self, embed_size, ff_size):
        super().__init__()
        self.fc1 = nn.Linear(embed_size, ff_size)
        self.fc2 = nn.Linear(ff_size, embed_size)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return x
    

class GPTDecoderLayer(nn.Module):
    def __init__(self, embed_size, heads, ff_size, attn_dropout_rate=0.1, resid_dropout_rate=0.1):
        super().__init__()
        # Pre-attention layer normalisation
        self.norm_attn = nn.LayerNorm(embed_size)  
        # Pre-feed-forward layer normalisation
        self.norm_ff = nn.LayerNorm(embed_size) 
        # Causal attention mechanism
        self.attn_dropout = nn.Dropout(attn_dropout_rate)
        self.attention = MultiHeadCausalAttention(embed_size, heads)
        # Feed-forward network
        self.feed_forward = TransitionFunction(embed_size, ff_size)
        #residual dropout
        self.resid_dropout = nn.Dropout(resid_dropout_rate)

    def forward(self, x):
        attn = self.attn_dropout(self.attention(self.norm_attn(x)))  
        x = self.resid_dropout(x + attn) 

        ff = self.feed_forward(self.norm_ff(x))  
        x = self.resid_dropout(x + ff)  
        return x
    

class UTModel(nn.Module):
    def __init__(self, vocab_size, max_context, embed_size, ff_size, num_layers, num_heads, act=True, embed_dropout_rate=0.1, resid_dropout_rate=0.1, attn_dropout_rate=0.1):
        super(UTModel, self).__init__() 
        self.embedding = nn.Embedding(vocab_size, embed_size)
        self.embed_dropout = nn.Dropout(embed_dropout_rate)
        self.act = act
        self.max_context = max_context

        # Below I fixed time encoding length to 1000, I don't to test beyond this limit
        assert self.max_context < 10000, "Please modify so max_contextcan be less than 10000, just becuase of time encoding limit!. "
        self.time_encoding = nn.Parameter(torch.ones(1, 10000, embed_size))
        if self.act:
            self.position_encoding = nn.Parameter(torch.ones(1, num_layers, embed_size))

        if self.act:
            self.transformation_fn = GPTDecoderLayer(embed_size, num_heads, ff_size, attn_dropout_rate=attn_dropout_rate, resid_dropout_rate=resid_dropout_rate)
            self.act_module = ACTModule(embed_size, max_hop=num_layers)
        else:
            # Create a ModuleList of GPTDecoderLayer instances.
            self.layers = nn.ModuleList([GPTDecoderLayer(embed_size, num_heads, ff_size) for _ in range(num_layers)])

        self.unembedding = nn.Linear(embed_size, vocab_size)
        self.apply(self._init_weights)
        logger.info("Number of parameters: %e",sum(p.numel() for p in self.parameters()))
    
    def _init_weights(self, module):
        if isinstance(module, (nn.Linear, nn.Embedding)):
            nn.init.normal_(module.weight, std=0.02)

        if isinstance(module, (nn.Linear)):
            if module.bias is not None:
                if not getattr(module.bias, "_no_reinit", False):
                    nn.init.zeros_(module.bias)

        elif isinstance(module, nn.LayerNorm):
            module.weight.data.fill_(1.0)
            module.bias.data.zero_()
    
    def UT_optimizer(self, train_config):
        decay = set()
        whitelist_weight_modules = (nn.Linear)

        for mn, m in self.named_modules():
            for pn, _ in m.named_parameters():
                fpn = "%s.%s" %(mn, pn) if mn else pn
                if pn.endswith('weight') and isinstance(m, whitelist_weight_modules):
                    decay.add(fpn)
 
        param_dict = {pn: p  for pn, p in self.named_parameters()}
        
        optim_groups = [
            {"params":[param_dict[pn] for pn in sorted(list(decay))], "weight_decay": train_config.weight_decay},
            {"params":[param_dict[pn] for pn in param_dict.keys() if pn not in sorted(list(decay))], "weight_decay": 0.0}
            ]
        
        optimizer = torch.optim.AdamW(optim_groups, lr=train_config.learning_rate, betas=train_config.AdamwBetas)
        return optimizer

    def forward(self, input_ids, targets=None):
        x = self.embed_dropout(self.embedding(input_ids)) 
        if not self.act:
            x = x + self.time_encoding[:,:input_ids.shape[1],:]

        if self.act:
            x, (remainders, n_updates) = self.act_module(x, self.time_encoding, self.position_encoding, self.transformation_fn)
        else:
            for layer in self.layers:  
                x = layer(x)

        logits = self.unembedding(x)

        loss=None
        if targets is not None:
            loss = F.cross_entropy(logits.view(-1, logits.size(-1)), targets.view(-1))

        return (logits, loss, (remainders, n_updates)) if self.act else (logits, loss)

    def generate(self, idx, num_tokens):
        tokens_len = idx.shape[1]
        token_ponder_time = {}
        generate_ponder_time = []
        for i in range(num_tokens):
            block_idx = idx[: , -self.max_context:]
            if self.act:
                logits, _, (_, n_updates) = self.forward(block_idx)
                generate_ponder_time.append(n_updates.mean().item())
            else:
                logits, _ = self.forward(block_idx)
            logits = logits.reshape(1, tokens_len, -1)[:,-1,:]
            probs = F.softmax(logits, dim=1)
            next_token = torch.multinomial(probs, num_samples=1)
            idx = torch.cat((idx, next_token), dim=1)
            if tokens_len < self.max_context:
                tokens_len+=1
            
            if self.act:
               token_ponder_time[i] = (next_token.item(), n_updates.mean().item())

        return (idx, sum(generate_ponder_time)/len(generate_ponder_time), token_ponder_time) if self.act else idx