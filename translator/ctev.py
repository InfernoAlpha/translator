import torch
import torch.nn as nn
import math

class embeddings(nn.Module):
    def __init__(self,d_model:int,vocab_size:int):
        super().__init__()
        self.d_model = d_model
        self.vocab_size = vocab_size
        self.Embeddings = nn.Embedding(vocab_size,d_model)

    def forward(self,x):
        return self.Embeddings(x)*math.sqrt(self.d_model)
        

class positional_embeddings(nn.Module):
    def __init__(self,d_model:int,seq_len:int,dropout:float):
        super().__init__()
        self.d_model = d_model
        self.seq_len = seq_len
        self.dropout = nn.Dropout(dropout)

        pe = torch.zeros(seq_len,d_model)
        pos = torch.arange(0,seq_len,dtype=torch.float).unsqueeze(1)

        div_term = torch.exp(torch.arange(0,d_model,2).float()*(-math.log(10000.0)/d_model))
        pe[:,0::2] = torch.sin(pos * div_term)
        pe[:,1::2] = torch.cos(pos * div_term)
        
        pe = pe.unsqueeze(0) #(1,seq_len,d_model)
        
        self.register_buffer('pe',pe)
    
    def forward(self,x):
        #here the model adds the positional and contextual emmbendings x.shape[1]
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
    
class feed_forward(nn.Module):
    def __init__(self,d_model:int,d_ff:int,dropout:float):
        super().__init__()
        self.layer1 = nn.Linear(d_model,d_ff)
        self.layer2 = nn.Linear(d_ff,d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self,x):
        return self.layer2(self.dropout(torch.relu(self.layer1(x))))
    
class MultiHeadAttention(nn.Module):
    def __init__(self,h:int,d_model:int,dropout:float):
        super().__init__()
        self.d_model = d_model
        self.h = h
        self.k = nn.Linear(d_model,d_model,bias=False)
        self.q = nn.Linear(d_model,d_model,bias=False)
        self.v = nn.Linear(d_model,d_model,bias=False)
        self.W0 = nn.Linear(d_model,d_model,bias=False)
        self.dk = self.d_model//self.h
        self.dropout = nn.Dropout(dropout)
    
    @staticmethod
    def attention(q,k,v,mask,dropout:nn.Dropout):
        dk = q.shape[-1]

        attentions_scores = (q @ k.transpose(-2,-1))/math.sqrt(dk)
        if mask is not None:
            attentions_scores.masked_fill_(mask==0,-1e9)
        attentions_scores = torch.softmax(attentions_scores,dim=-1)
        if dropout is not None:
            attentions_scores = dropout(attentions_scores)

        return (attentions_scores @ v),attentions_scores
    def forward(self,q,k,v,mask):    
        query = self.q(q)
        key = self.k(k)
        value = self.v(v)

        #(Batch,seq_len,d_model) --> (Batch,seq_len,h,dk) --> (Batch,h,seq_len,dk) x.shape[0]
        query = query.view(query.shape[0],query.shape[1],self.h,self.dk).transpose(1,2)
        key = key.view(key.shape[0],key.shape[1],self.h,self.dk).transpose(1,2)
        value = value.view(value.shape[0],value.shape[1],self.h,self.dk).transpose(1,2)

        x,self.attention_scores = MultiHeadAttention.attention(query,key,value,mask,self.dropout)

        # -1(placeholder) in .view automatically calculates the value given all the other values
        x = x.transpose(1,2).contiguous().view(x.shape[0],-1,self.h*self.dk)

        return self.W0(x)

class residual_networks(nn.Module):
    def __init__(self,dropout:float):
        super().__init__()
        self.dropout = nn.Dropout(dropout)
        self.norm = layer_norm()
    def forward(self,x,sublayer):
        return x + self.dropout(sublayer(self.norm(x)))

class Encoder_block(nn.Module):
    def __init__(self,self_attention_block:MultiHeadAttention,self_feed_forward:feed_forward,dropout:float):
        super().__init__()
        self.dropout = nn.Dropout(dropout)
        self.attention = self_attention_block
        self.re = nn.ModuleList([residual_networks(dropout=dropout) for _ in range(2)])    
        self.ff = self_feed_forward
    def forward(self,x,mask):
        return (self.re[1](self.re[0](x,lambda x: self.attention(x,x,x,mask)),self.ff))

class Encoder(nn.Module):
    def __init__(self,layers:nn.ModuleList):
        super().__init__()
        self.layers = layers
        self.norm = layer_norm()

    def forward(self,x,mask):
        for layer in self.layers:
            x = layer(x,mask)
        return self.norm(x)

class decoder_block(nn.Module):
    def __init__(self,self_attention:MultiHeadAttention,cross_attention:MultiHeadAttention,Feed_forward:feed_forward,dropout:float):
        super().__init__()
        self.attention = self_attention
        self.cross_attention = cross_attention
        self.ff = Feed_forward
        self.residual_connection = nn.ModuleList([residual_networks(dropout=dropout) for _ in range(3)])
        self.dropout = nn.Dropout(dropout)

    def forward(self,x,encoder_output,src_mask,trg_mask):
        x = self.residual_connection[0](x,lambda x: self.attention(x,x,x,trg_mask))
        x = self.residual_connection[1](x,lambda x: self.cross_attention(x,encoder_output,encoder_output,src_mask))
        x = self.residual_connection[2](x, self.ff)
        return x

class Decoder(nn.Module):
    def __init__(self,layers:nn.ModuleList):
        super().__init__()
        self.layers = layers
        self.norm = layer_norm()

    def forward(self,x,encoder_outputs,src_mask,trg_mask):
        for layer in self.layers:
            x = layer(x,encoder_outputs,src_mask,trg_mask)
        
        return self.norm(x)

class Projection_Layer(nn.Module):
    def __init__(self,d_model:int,vocab_size:int,):
        super().__init__()
        self.linear = nn.Linear(d_model,vocab_size)

    def forward(self,x):
        #(batch,seq_len,d_model)-->(batch,seq_len,vocab_size)
        return self.linear(x)

class Transformer(nn.Module):
    def __init__(self,src_emmbedding:embeddings,src_pos_emmbeddings:positional_embeddings,trg_emmbedding:embeddings,trg_pos_emmbeddings:positional_embeddings,encoder:Encoder,Decoder:Decoder,projection:Projection_Layer):
        super().__init__()
        self.src_emmbeddings = src_emmbedding
        self.src_pos_emmbeddings = src_pos_emmbeddings
        self.trg_emmbeddings = trg_emmbedding
        self.trg_pos_emmbeddings = trg_pos_emmbeddings
        self.encoder = encoder
        self.decoder = Decoder
        self.projection = projection
        
    def encode(self,src,src_mask):
        src = self.src_emmbeddings(src)
        src = self.src_pos_emmbeddings(src)
        return self.encoder(src,src_mask)

    def Decode(self,src,trg,src_mask,trg_mask):
        trg = self.trg_emmbeddings(trg)
        trg = self.trg_pos_emmbeddings(trg)
        return self.decoder(trg,src,src_mask,trg_mask)

    def projecter(self,trg):
        return self.projection(trg)

def Transfromer_builder(src_vocab_size:int,trg_vocab_size:int,src_seq_len:int,trg_seq_len:int,d_model:int = 512,dropout:float = 0.1,d_ff = 2048,N:int = 6,h:int = 8)->Transformer:
    
    src_embed = embeddings(d_model=d_model,vocab_size=src_vocab_size)
    src_pos_embed = positional_embeddings(d_model=d_model,seq_len=src_seq_len,dropout=dropout)
    trg_embed = embeddings(d_model=d_model,vocab_size=trg_vocab_size)
    trg_pos_embed = positional_embeddings(d_model=d_model,seq_len=trg_seq_len,dropout=dropout)
    
    encoder_list = []
    for i in range(N):
        src_attention = MultiHeadAttention(h,d_model=d_model,dropout=dropout)
        src_ff = feed_forward(d_model=d_model,d_ff=d_ff,dropout=dropout)
        encoder_list.append(Encoder_block(src_attention,src_ff,dropout=dropout))

    decoder_list = []
    for i in range(N):
        trg_attention = MultiHeadAttention(h,d_model=d_model,dropout=dropout)
        trg_cross_attention = MultiHeadAttention(h,d_model=d_model,dropout=dropout)
        trg_ff = feed_forward(d_model=d_model,d_ff=d_ff,dropout=dropout)
        decoder_list.append(decoder_block(trg_attention,trg_cross_attention,trg_ff,dropout=dropout))

    encoder = Encoder(nn.ModuleList(encoder_list))
    decoder = Decoder(nn.ModuleList(decoder_list))
    projection = Projection_Layer(d_model,trg_vocab_size)

    transformer =  Transformer(src_embed,src_pos_embed,trg_embed,trg_pos_embed,encoder,decoder,projection=projection)
    for i in transformer.parameters():
        if i.dim() >1:
            nn.init.xavier_uniform(i)
    return transformer