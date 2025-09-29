from typing import Any
import torch
import torch.nn as nn
from torch.utils.data import Dataset

class BilingualDataset(Dataset):
    def __init__(self,ds,tokenizer_src,tokenizer_trg,src_lang,trg_lang,seq_len)->None:
        super().__init__()

        self.ds = ds
        self.src_tokenizer = tokenizer_src
        self.trg_tokenizer = tokenizer_trg
        self.src_lang = src_lang
        self.trg_lang = trg_lang
        self.seq_len = seq_len

        self.sos_token = torch.tensor([tokenizer_trg.token_to_id("[SOS]")], dtype = torch.int64)
        self.eos_token = torch.tensor([tokenizer_trg.token_to_id("[EOS]")], dtype = torch.int64)
        self.pad_token = torch.tensor([tokenizer_trg.token_to_id("[PAD]")], dtype = torch.int64)

    def __len__(self):
        return len(self.ds)
    
    def __getitem__(self,index:Any)->Any:
        src_trg_pair = self.ds[index]
        src_text = src_trg_pair['en']
        trg_text = src_trg_pair['te']


        src_tokens = self.src_tokenizer.encode(src_text).ids
        trg_tokens = self.trg_tokenizer.encode(trg_text).ids

        enc_num_padding_tokens = self.seq_len - len(src_tokens) - 2
        dec_num_padding_tokens = self.seq_len - len(trg_tokens) - 1

        if enc_num_padding_tokens < 0 or dec_num_padding_tokens < 0:
            raise ValueError("sentence is too long")
        
        encoder_input = torch.cat([
            self.sos_token,
            torch.tensor(src_tokens,dtype=torch.int64),
            self.eos_token,
            torch.tensor([self.pad_token]*enc_num_padding_tokens,dtype=torch.int64)
        ],dim=0)
        
        decoder_input = torch.cat([
            self.sos_token,
            torch.tensor(trg_tokens,dtype=torch.int64),
            torch.tensor([self.pad_token]*dec_num_padding_tokens,dtype=torch.int64),

        ],dim=0)

        label = torch.cat([
           torch.tensor(trg_tokens,dtype=torch.int64),
           self.eos_token,
           torch.tensor([self.pad_token]*dec_num_padding_tokens,dtype=torch.int64)
        ],dim=0)
        

        assert encoder_input.size(0) == self.seq_len
        assert decoder_input.size(0) == self.seq_len
        assert label.size(0) == self.seq_len

        return {
            "encoder_input" : encoder_input,
            "decoder_input" : decoder_input,
            "encoder_mask"  : (encoder_input != self.pad_token).unsqueeze(0).unsqueeze(0).int(), # (1,1,seq_len)
            "decoder_mask"  : (decoder_input != self.pad_token).unsqueeze(0).int() & casual_mask(decoder_input.size(0)), # (1,seq_len) -> (1,seq_len,seq_len)
            "label"         : label,
            "src_text"      : src_text,
            "trg_text"      : trg_text
        }
    
def casual_mask(size):
    mask = torch.triu(torch.ones((1,size,size)),diagonal=1).type(torch.int)
    return mask == 0