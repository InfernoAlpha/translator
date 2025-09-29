import torch
import torch.nn as nn
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

class feed_forward(nn.Module):
    def __init__(self,d_model,d_ff,dropout):
        super().__init__()
        self.l1 = nn.Linear(d_model,d_ff)
        self.l2 = nn.Linear(d_ff,d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self,x):
        return self.dropout(self.l2(self.l1(x)))

class Encoder_Block(nn.Module):
    def __init__(self,attention_block:MultiHeadAttention,feed_forward:feed_forward,dropout:float):
        super().__init__()
        self.attention = attention_block
        self.feed_forward = feed_forward
        self.AN_list = nn.ModuleList([add_norm_layer(dropout=dropout) for _ in range(2)])
    
    def forward(self,x,mask):
        return (self.AN_list[1](self.AN_list[0](x ,lambda x: self.attention(x,x,x,mask)),self.feed_forward))

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
    def __init__(self,attention_layer:MultiHeadAttention,cross_attention_layer:MultiHeadAttention,feed_forward:feed_forward,dropout:float):
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
    def __init__(self,src_emmbedings:Embeddings,src_pos_emmbedings:Embeddings,trg_emmbedings:Embeddings,trg_pos_emmbedings:Embeddings,Encoder:Encoder,Decoder:Decoder,projection:projection_layer):
        super().__init__()
        self.src_emmbedings = src_emmbedings
        self.src_pos_emmbedings = src_pos_emmbedings
        self.trg_emmbedings = trg_emmbedings
        self.trg_pos_emmbedings = trg_pos_emmbedings
        self.Encoder = Encoder
        self.Decoder = Decoder
        self.projection = projection

    def encode(self,src,src_mask):
        src = self.src_emmbedings(src)
        src = self.src_pos_emmbedings(src)
        return self.Encoder(src,src_mask)
    
    def decode(self,src,trg,src_mask,trg_mask):
        trg = self.trg_emmbedings(trg)
        trg = self.trg_pos_emmbedings(trg)
        return self.Decoder(src,trg,src_mask,trg_mask)
    
    def project(self,x):
        return self.projection(x)

def Transfromer_builder(src_vocab_size:int,trg_vocab_size:int,src_seq_len:int,trg_seq_len:int,d_model:int = 512,dropout:float = 0.1,d_ff = 2048,N:int = 6,h:int = 8)->Transformer:
    
    src_emmbeding = Embeddings(src_vocab_size,d_model)
    src_pos_emmbeddings = positional_encoding(src_seq_len,d_model,dropout)
    trg_emmbeding = Embeddings(trg_vocab_size,d_model)
    trg_pos_emmbeddings = positional_encoding(trg_seq_len,d_model,dropout)

    encoder_list = []
    for i in range(N):
        src_attention = MultiHeadAttention(d_model,h,dropout)
        src_feed_forward = feed_forward(d_model,d_ff,dropout)
        encoder_list.append(Encoder_Block(src_attention,src_feed_forward,dropout))

    decoder_list = []
    for i in range(N):
        trg_attention = MultiHeadAttention(d_model,h,dropout)
        trg_cross_attention = MultiHeadAttention(d_model,h,dropout)
        trg_feed_forward = feed_forward(d_model,d_ff,dropout)
        decoder_list.append(Decoder_Block(trg_attention,trg_cross_attention,trg_feed_forward,dropout))
    
    encoder = Encoder(nn.ModuleList(encoder_list))
    decoder = Decoder(nn.ModuleList(decoder_list))
    projection = projection_layer(d_model,trg_vocab_size)

    transformer = Transformer(src_emmbeding,src_pos_emmbeddings,trg_emmbeding,trg_pos_emmbeddings,encoder,decoder,projection)

    for i in transformer.parameters():
        if i.dim() >1:
            nn.init.xavier_uniform(i)
    
    return transformer