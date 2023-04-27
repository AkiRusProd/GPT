import torch
import torch.nn as nn



class MultiHeadAttention(nn.Module):
    def __init__(self, d_model, n_heads, dropout=0.1):
        super(MultiHeadAttention, self).__init__()
        self.d_model = d_model
        self.n_heads = n_heads
        self.scale = torch.sqrt(torch.FloatTensor([d_model])).to("cuda")
        self.dropout = nn.Dropout(dropout)

        assert d_model % n_heads == 0

        self.depth = d_model // n_heads

        self.wq = nn.Linear(d_model, d_model)
        self.wk = nn.Linear(d_model, d_model)
        self.wv = nn.Linear(d_model, d_model)

        self.fc = nn.Linear(d_model, d_model)

    def forward(self, q, k, v, mask=None):
        batch_size = q.size(0)
        q = self.wq(q).view(batch_size, -1, self.n_heads, self.depth).transpose(1, 2)
        k = self.wk(k).view(batch_size, -1, self.n_heads, self.depth).transpose(1, 2)
        v = self.wv(v).view(batch_size, -1, self.n_heads, self.depth).transpose(1, 2)

        attn = torch.matmul(q, k.transpose(2, 3)) / self.scale
        if mask is not None:
            # print(mask.shape, attn.shape)
            mask = mask.unsqueeze(1)#.repeat(1, self.n_heads, 1, 1)
            attn = attn.masked_fill(mask == 0, -1e9)
        attn = self.dropout(nn.Softmax(dim = -1)(attn))
        x = torch.matmul(attn, v)
        x = x.transpose(1, 2).contiguous().view(batch_size, -1, self.n_heads * self.depth)
        x = self.fc(x)
        return x, attn
    


class PositionalEncoding(nn.Module):

    def __init__(self, d_model, max_len=5000):
        """
        Inputs
            d_model - Hidden dimensionality of the input.
            max_len - Maximum length of a sequence to expect.
            https://uvadlc-notebooks.readthedocs.io/en/latest/tutorial_notebooks/tutorial6/Transformers_and_MHAttention.html
        """
        super().__init__()

        # Create matrix of [SeqLen, HiddenDim] representing the positional encoding for max_len inputs
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-torch.log(torch.tensor([10000.0])) / d_model)) 
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)

        # register_buffer => Tensor which is not a parameter, but should be part of the modules state.
        # Used for tensors that need to be on the same device as the module.
        # persistent=False tells PyTorch to not add the buffer to the state dict (e.g. when we save the model)
        self.register_buffer('pe', pe, persistent=False)

    def forward(self, x):
        x = x + self.pe[:, :x.size(1)]
        return x


class LearnablePositionalEncoding(nn.Module):
    def __init__(self,max_len, d_model):
        super(LearnablePositionalEncoding, self).__init__()
        self.d_model = d_model
        self.max_len = max_len
        self.embedding = nn.Embedding(max_len, d_model)

    def forward(self, x):
        """ x: (batch_size, d_model)
        """
        # print(x.shape[1], self.d_model, self.max_len)
        # assert x.shape[1] > self.max_len
        pos =  torch.arange(0, x.shape[1], dtype=torch.long, device=x.device).unsqueeze(0)
    
        x = self.embedding(pos).to(x.device)  # (batch_size, seq_len, d_model)
        return x
    
    


class PositionwiseFeedForward(nn.Module):
    def __init__(self, d_model, d_ff, dropout=0.1):
        super(PositionwiseFeedForward, self).__init__()
        self.fc_1 = nn.Linear(d_model, d_ff)
        self.fc_2 = nn.Linear(d_ff, d_model)
        self.dropout = nn.Dropout(dropout)

        self.activation = nn.GELU()

    def forward(self, x):
        x = self.fc_1(x)
        x = self.activation(x)
        x = self.dropout(x)
        x = self.fc_2(x)

        return x


class DecoderLayer(nn.Module):
    def __init__(self, d_model, n_heads, d_ff, attn_dropout, resid_dropout, ff_dropout):
        super(DecoderLayer, self).__init__()

        self.self_attn_norm = nn.LayerNorm(d_model)
        self.ff_norm = nn.LayerNorm(d_model)

        self.self_attn = MultiHeadAttention(d_model, n_heads, attn_dropout)
        self.ff = PositionwiseFeedForward(d_model, d_ff, ff_dropout)
        self.resid_dropout = nn.Dropout(resid_dropout)


    def forward(self, x, mask):
        _x, attention = self.self_attn(x, x, x, mask)
        x = self.self_attn_norm(x + self.resid_dropout(_x))

        _x = self.ff(x)
        x = self.ff_norm(x + self.resid_dropout(_x))

        return x, attention
    

class Decoder(nn.Module):
    def __init__(self, vocab_size, d_model, n_layers, n_heads,  d_ff, attn_dropout, resid_dropout, ff_dropout, embed_dropout, max_len, pad_idx):
        super(Decoder, self).__init__()
        self.pad_idx = pad_idx

        self.token_embedding = nn.Embedding(vocab_size, d_model)
        self.pos_embedding = LearnablePositionalEncoding(max_len, d_model) # PositionalEncoding(max_len, d_model)
        self.embed_dropout = nn.Dropout(embed_dropout)
        self.scale = 1 #torch.sqrt(torch.FloatTensor([d_model])).to("cuda")

        self.layers = nn.ModuleList([DecoderLayer(d_model, n_heads, d_ff, attn_dropout, resid_dropout, ff_dropout) for _ in range(n_layers)])

        # self.fc = nn.Linear(d_model, vocab_size)
        # self.activation = None # no activation, because CrossEntropyLoss is already using Softmax

    def forward(self, x, mask = None):
        if mask is None:
            mask = self.get_pad_mask(x).to(x.device) & self.get_sub_mask(x).to(x.device) # check the mask in batch dataset
            # print(mask.shape, x.shape)
            # print(self.pad_idx)
            # print(self.get_pad_mask(x))
            # print(mask)

        # Const positional embedding
        # x = self.token_embedding(x) * self.scale
        # x = self.pos_embedding(x)
        # x = self.embed_dropout(x)

        # Learnable positional embedding
        x = self.token_embedding(x) * self.scale + self.pos_embedding(x)
        x = self.embed_dropout(x)

        attentions = []
        for layer in self.layers:
            x, attention = layer(x, mask)

            attentions.append(attention)

        # x = self.fc(x)
        return x, attentions
    
    def get_pad_mask(self, x):
        #x: (batch_size, seq_len)
        return (x != self.pad_idx).type(torch.int32)[:, None, :]

    def get_sub_mask(self, x):
        #x: (batch_size, seq_len)
        seq_len = x.shape[1]
        subsequent_mask = torch.triu(torch.ones((seq_len, seq_len)), diagonal = 1).type(torch.int32)
        subsequent_mask = torch.logical_not(subsequent_mask)
        return subsequent_mask

# class GPT(nn.Module):
#     def __init__(self, vocab_size, d_model, n_layers, n_heads,  d_ff, attn_dropout, resid_dropout, ff_dropout, embed_dropout, max_len = 5000):
#         super(GPT, self).__init__()
#         self.decoder = Decoder(vocab_size, d_model, n_layers, n_heads,  d_ff, attn_dropout, resid_dropout, ff_dropout, embed_dropout, max_len)

#     def forward(self, x):
#         x, attentions = self.decoder(x)
#         return x
       
class GPT(nn.Module):
    def __init__(self, vocab_size, d_model, n_layers, n_heads,  d_ff, attn_dropout, resid_dropout, ff_dropout, embed_dropout, max_len, pad_idx):
        super(GPT, self).__init__()
        self.decoder = Decoder(vocab_size, d_model, n_layers, n_heads,  d_ff, attn_dropout, resid_dropout, ff_dropout, embed_dropout, max_len, pad_idx)
        vocab_size, d_model = self.decoder.token_embedding.weight.size()
        self.fc = nn.Linear(d_model, vocab_size)
        self.fc.weight = self.decoder.token_embedding.weight
        # self.activation = None # no activation, because CrossEntropyLoss is already using Softmax

    def forward(self, x, mask = None):
        x, attentions = self.decoder(x, mask)
        x = self.fc(x)
        return x, attentions
    


# class GPTCLSHead():
#     def __init__(self, gpt, n_classes, cls_token_id, dropout=0.1):
#         super(GPTCLSHead, self).__init__()
#         self.gpt = gpt
#         self.cls_token_id = cls_token_id
#         vocab_size, d_model = gpt.decoder.token_embedding.weight.size()
#         self.fc_1 = nn.Linear(d_model, vocab_size)
#         self.fc_1.weight = self.gpt.decoder.token_embedding.weight
#         self.fc_2 = nn.Linear(d_model, n_classes)
#         self.dropout = nn.Dropout(dropout)

#     def forward(self, x):
#         _x = self.gpt(x)

#         lm_logits = self.fc_1(_x)
#         cls_logits = self.fc_2(self.dropout(_x[x.eq(self.cls_token_id)]))

#         return lm_logits, cls_logits
        







        

