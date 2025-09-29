import torch
import torch.nn as nn
import torch.nn.functional as F
import math

class Embeddings(nn.Module):
    def __init__(self,vocab_size,d_model):
        super().__init__()
        self.vocab_size = vocab_size
        self.d_model = d_model
        self.emmbedings = nn.Embedding(vocab_size,d_model)
    
    def forward(self,x):
        return self.emmbedings(x)*math.sqrt(self.d_model)
    
class positional_encoding(nn.Module):
    def __init__(self,seq_len:int,d_model:int,dropout:int):
        super().__init__()
        self.seq_len = seq_len
        self.d_model = d_model
        PE = torch.zeros(seq_len,d_model)
        POS = torch.arange(0,seq_len,dtype=torch.float).unsqueeze(1)
        self.dropout = nn.Dropout(dropout)

        div_term = torch.exp(torch.arange(0,d_model,2).float()*(-math.log(10000.0)/d_model))
        PE[:,0::2] = torch.sin(POS*div_term)
        PE[:,1::2] = torch.cos(POS*div_term)

        self.register_buffer('pe',PE.unsqueeze(0))

    def forward(self,x):
        x = x + (self.pe[:,:x.shape[1],:]).requires_grad_(False)
        return self.dropout(x)

class RoPE(nn.Module):
    def __init__(self,dim,scale=40):
        super().__init__()
        assert dim % 2 == 0,"dim not even"
        self.dim = dim
        self.freq = 1.0 / (10000 ** (torch.arange(0, dim//2,2).float() / (dim//2)))
        self.register_buffer("inv_freq", self.freq)
        self.scale = scale

    def forward(self,seq_len):
        t = torch.arange(seq_len,device=self.inv_freq.device).type_as(self.inv_freq) / self.scale
        freqs = torch.einsum("i,j->ij",t,self.inv_freq)
        return torch.cat((freqs,freqs),dim=-1)
    
def rotate_half(x):
    x1,x2 = x.chunk(2,dim=-1)
    return torch.cat((-x2,x1),dim=-1)

def apply_RoPE(x,cos,sin):
    x_rotate,x_base = x.split(cos.shape[-1],dim=-1)
    x_rotate = (x_rotate*cos) + (rotate_half(x_rotate)*sin)
    return torch.cat([x_rotate,x_base],dim=-1)

class layer_norm(nn.Module):
    def __init__(self,eps:float = 10**-6):
        super().__init__()
        self.eps = eps
        self.alpha = nn.Parameter(torch.ones(1))
        self.beta = nn.Parameter(torch.zeros(1))

    def forward(self,x):
        mean = torch.mean(x,keepdim=True,dim=-1)
        std = torch.std(x,dim=-1,keepdim=True)
        return self.alpha * (x-mean)/(std+self.eps) + self.beta

class add_norm_layer(nn.Module):
    def __init__(self,dropout):
        super().__init__()
        self.dropout = nn.Dropout(dropout)
        self.alpha = nn.Parameter(torch.ones([1]))
        self.beta = nn.Parameter(torch.zeros([1]))
        self.eps = 1e-3
    
    def forward(self,x,sub_layer):
        x = self.alpha*(x-torch.mean(x,dim=-1,keepdim=True))/(torch.std(x,dim=-1,keepdim=True)+self.eps) +self.beta
        x = sub_layer(x)
        return x + self.dropout(x)

class MultiHeadAttention(nn.Module):
    def __init__(self,d_model:int,h:int,dropout:float):
        super().__init__()
        self.d_model = d_model
        self.h = h
        self.dropout = nn.Dropout(dropout)
        self.q = nn.Linear(self.d_model,self.d_model,bias=False)
        self.k = nn.Linear(self.d_model,self.d_model,bias=False)
        self.v = nn.Linear(self.d_model,self.d_model,bias=False)
        self.w0 = nn.Linear(self.d_model,self.d_model,bias=False)
        self.dk = self.d_model//self.h

    @staticmethod
    def attention_mechanism(q,k,v,mask,dropout):
        dk = q.shape[-1]
 
        attention_scores = (q @ k.transpose(-2,-1))/math.sqrt(dk)
        if mask is not None:
            attention_scores.masked_fill_(mask==0,-1e9)
        attention_scores = torch.softmax(attention_scores,dim=-1) 
        if dropout is not None: 
            attention_scores = dropout(attention_scores)
        
        return (attention_scores @ v),attention_scores
    
    def forward(self,q,k,v,mask):

        query = self.q(q)
        key = self.k(k)
        values = self.v(v)

        query = query.view(query.shape[0],query.shape[1],self.h,self.dk).transpose(1,2)
        key = key.view(key.shape[0],key.shape[1],self.h,self.dk).transpose(1,2)
        values = values.view(values.shape[0],values.shape[1],self.h,self.dk).transpose(1,2)

        x,self.attention_scores = MultiHeadAttention.attention_mechanism(query,key,values,mask,self.dropout)

        x = x.transpose(1,2).contiguous().view(x.shape[0],-1,self.h*self.dk)
        return self.w0(x)

class Multihead_Latent_attention_Decoupled_rope(nn.Module):
    def __init__(self,n_head,d_model,dropout):
        super().__init__()
        self.n_head = n_head
        self.d_head = d_model // self.n_head
        self.qk_rope_dim = self.d_head // 2
        self.qk_nope_dim = self.d_head // 2
        self.q_proj_dim = d_model // 2
        self.d_kv_comp = (d_model) // 2

        self.W_dkv = nn.Linear(d_model,self.d_kv_comp,bias=False)
        self.W_ukv = nn.Linear(self.d_kv_comp,d_model + (self.n_head * self.qk_nope_dim))

        self.W_dq = nn.Linear(d_model,self.d_kv_comp,bias=False)

        self.W_uk = nn.Linear(self.d_kv_comp,n_head*self.qk_nope_dim,bias=False)
        self.W_uv = nn.Linear(self.d_kv_comp,n_head*self.qk_nope_dim,bias=False)
        self.W_uq = nn.Linear(self.d_kv_comp,n_head*self.qk_nope_dim,bias=False)

        self.W_qr = nn.Linear(self.d_kv_comp,n_head*self.qk_rope_dim,bias=False)
        self.W_kr = nn.Linear(self.d_kv_comp,n_head*self.qk_rope_dim,bias=False)

        self.rotary = RoPE(self.qk_rope_dim)
        self.W0 = nn.Linear(n_head*self.d_head,d_model,bias=False)

        self.q_layernorm = torch.nn.LayerNorm(self.q_proj_dim)
        self.kv_layernorm = torch.nn.LayerNorm(self.d_kv_comp)
        self.dropout = nn.Dropout(dropout)

    def forward(self,x,kv_cache = None, mask=None, dropout = None):
        batch_size,seq_len,dim = x.shape
        """
        c_kv = self.W_dkv(x)
        k = self.W_uk(c_kv).view(batch_size,seq_len,self.n_head,self.d_head)
        v = self.W_uv(c_kv).view(batch_size,seq_len,self.n_head,self.d_head)
        """ 
        c_q = self.W_dq(x)
        c_q = self.q_layernorm(c_q)
        Q = self.W_uq(c_q).view(batch_size,seq_len,self.n_head,self.qk_nope_dim)
        Q_for_rope = self.W_qr(c_q).view(batch_size,seq_len,self.n_head,self.qk_rope_dim)

        if kv_cache is None:
            compressed_kv = self.W_dkv(x)
            KV_for_Lora,K_for_RoPE = torch.split(compressed_kv,[self.d_kv_comp,self.qk_rope_dim],dim=-1)
            KV_for_Lora = self.kv_layernorm(KV_for_Lora)
        else:
            new_kv = self.W_dkv(x)
            compressed_kv = torch.cat([kv_cache,new_kv],dim=-1)
            new_kv,new_k_for_rope = torch.split(new_kv,[self.d_kv_comp,self.qk_rope_dim],dim=-1)
            old_kv,old_k_for_rope = torch.split(kv_cache,[self.d_kv_comp,self.qk_rope_dim],dim=-1)

            new_kv = self.kv_layernorm(new_kv)
            old_kv = self.kv_layernorm(old_kv)
            KV_for_Lora = torch.cat([old_kv,new_kv],dim=1)
            K_for_RoPE = torch.cat([old_k_for_rope,new_k_for_rope],dim=1)

        KV = self.W_ukv(KV_for_Lora)
        KV = KV.view(batch_size,-1,self.n_head,self.d_head+self.qk_nope_dim).transpose(1,2)
        K,V = torch.split(KV,[self.qk_nope_dim,self.d_head],dim=-1)
        S_full = K.size(2)

        K_for_RoPE = K_for_RoPE.view(batch_size,-1,1,self.qk_rope_dim).transpose(1,2)
        rotary_emb = self.rotary(seq_len)
        cos = torch.cos(rotary_emb).view(1,seq_len,1,-1)
        sin = torch.sin(rotary_emb).view(1,seq_len,1,-1)

        Q_for_rope = apply_RoPE(Q_for_rope,cos,sin)
        K_for_RoPE = apply_RoPE(K_for_RoPE,cos,sin)

        K_for_rope = K_for_rope.repeat(1, self.n_heads, 1, 1)

        q_heads = torch.cat([Q,Q_for_rope],dim=-1)
        k_heads = torch.cat([K,K_for_RoPE],dim=-1)
        v_heads = V

        dk = q_heads.shape[-1]
        attention_scores = (q_heads @ k_heads.transpose(-2,-1))/math.sqrt(dk)
        if mask is not None:
            attention_scores.masked_fill_(mask==0,-1e9)
        attention_scores = torch.softmax(attention_scores,dim=-1) 
        if dropout is not None: 
            attention_scores = self.dropout(attention_scores)

        out = attention_scores @ v_heads
        out = out.transpose(1,2).reshape(batch_size,seq_len,dim)

        return self.W0(out), compressed_kv

class feed_forward(nn.Module):
    def __init__(self,d_model,d_ff,dropout):
        super().__init__()
        self.l1 = nn.Linear(d_model,d_ff)
        self.l2 = nn.Linear(d_ff,d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self,x):
        return self.dropout(self.l2(self.l1(x)))

class Encoder_Block(nn.Module):
    def __init__(self,attention_block:Multihead_Latent_attention_Decoupled_rope,feed_forward:feed_forward,dropout:float):
        super().__init__()
        self.attention = attention_block
        self.feed_forward = feed_forward
        self.dropout = dropout
        self.AN_list = nn.ModuleList([add_norm_layer(dropout=dropout) for _ in range(2)])
        self.KV_cache = None
    
    def forward(self,x,mask):
        return (self.AN_list[1](self.AN_list[0](x ,lambda x: self.attention(x,self.KV_cache,mask,self.dropout)),self.feed_forward))

class Encoder(nn.Module):
    def __init__(self,Module_list:nn.ModuleList):
        super().__init__()
        self.layers = Module_list
        self.norm = layer_norm()
    
    def forward(self,x,mask):
        for i in self.layers:
            x = i(x,mask)
        return self.norm(x)
 
class Decoder_Block(nn.Module):
    def __init__(self,attention_layer:Multihead_Latent_attention_Decoupled_rope,cross_attention_layer:Multihead_Latent_attention_Decoupled_rope,feed_forward:feed_forward,dropout:float):
        super().__init__()
        self.attention = attention_layer
        self.cross_attention = cross_attention_layer
        self.feed_forward = feed_forward
        self.AN_list = nn.ModuleList([add_norm_layer(dropout) for _ in range(3)])

    def forward(self,x,encoder_output,src_mask,trg_mask):
        x = self.AN_list[0](x,lambda x: self.attention(x,x,x,trg_mask))
        x = self.AN_list[1](x,lambda x: self.cross_attention(x,encoder_output,encoder_output,src_mask))
        return self.AN_list[2](x,self.feed_forward)

class Decoder(nn.Module):
    def __init__(self,Module_list:nn.ModuleList):
        super().__init__()
        self.layers = Module_list
        self.norm = layer_norm()

    def forward(self,x,encoder_output,src_mask,trg_mask):
        for i in self.layers:
            x = i(x,encoder_output,src_mask,trg_mask)
        return self.norm(x)

class projection_layer(nn.Module):
    def __init__(self,d_model,vocab_size):
        super().__init__()
        self.linear = nn.Linear(d_model,vocab_size)

    def forward(self,x):
        return self.linear(x)

class Transformer(nn.Module):
    def __init__(self,src_emmbedings:Embeddings,trg_emmbedings:Embeddings,Encoder:Encoder,Decoder:Decoder,projection:projection_layer):
        super().__init__()
        self.src_emmbedings = src_emmbedings
#        self.src_pos_emmbedings = src_pos_emmbedings
        self.trg_emmbedings = trg_emmbedings
#        self.trg_pos_emmbedings = trg_pos_emmbedings
        self.Encoder = Encoder
        self.Decoder = Decoder
        self.projection = projection

    def encode(self,src,src_mask):
        src = self.src_emmbedings(src)
#        src = self.src_pos_emmbedings(src)
        return self.Encoder(src,src_mask)
    
    def decode(self,src,trg,src_mask,trg_mask):
        trg = self.trg_emmbedings(trg)
#        trg = self.trg_pos_emmbedings(trg)
        return self.Decoder(src,trg,src_mask,trg_mask)
    
    def project(self,x):
        return self.projection(x)

def Transfromer_builder(src_vocab_size:int,trg_vocab_size:int,src_seq_len:int,trg_seq_len:int,d_model:int = 512,dropout:float = 0.1,d_ff = 2048,N:int = 6,h:int = 8)->Transformer:
    
    src_emmbeding = Embeddings(src_vocab_size,d_model)
    #src_pos_emmbeddings = positional_encoding(src_seq_len,d_model,dropout)
    trg_emmbeding = Embeddings(trg_vocab_size,d_model)
    #trg_pos_emmbeddings = positional_encoding(trg_seq_len,d_model,dropout)

    encoder_list = []
    for i in range(N):
        src_attention = Multihead_Latent_attention_Decoupled_rope(h,d_model,dropout)
        src_feed_forward = feed_forward(d_model,d_ff,dropout)
        encoder_list.append(Encoder_Block(src_attention,src_feed_forward,dropout))

    decoder_list = []
    for i in range(N):
        trg_attention = Multihead_Latent_attention_Decoupled_rope(d_model,h,dropout)
        trg_cross_attention = Multihead_Latent_attention_Decoupled_rope(d_model,h,dropout)
        trg_feed_forward = feed_forward(d_model,d_ff,dropout)
        decoder_list.append(Decoder_Block(trg_attention,trg_cross_attention,trg_feed_forward,dropout))
    
    encoder = Encoder(nn.ModuleList(encoder_list))
    decoder = Decoder(nn.ModuleList(decoder_list))
    projection = projection_layer(d_model,trg_vocab_size)

    transformer = Transformer(src_emmbeding,trg_emmbeding,encoder,decoder,projection)

    for i in transformer.parameters():
        if i.dim() >1:
            nn.init.xavier_uniform(i)
    
    return transformer