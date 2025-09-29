import torch
import torch.nn as nn
from torch.utils.data import DataLoader,Dataset,random_split 
from torch.utils.tensorboard import SummaryWriter
import torchmetrics.text

from ctev_database import BilingualDataset , casual_mask
from ctev2 import Transfromer_builder
from config import get_weights_file_path,get_config,latest_weights_file_path

from datasets import load_dataset
from tokenizers import Tokenizer 
from tokenizers.models import WordLevel,WordPiece,BPE
from tokenizers.trainers import WordLevelTrainer,WordPieceTrainer,BpeTrainer
from tokenizers.pre_tokenizers import Whitespace

from pathlib import Path
from tqdm import tqdm
import warnings

def get_all_sentence(ds,lang):
    for item in ds:
        yield item[lang]

def tokenizer_builder(config,ds,lang):

    tokenizer_path = Path(config['tokenizer_file'].format(lang)) 

    if not Path.exists(tokenizer_path):
        tokenizer = Tokenizer(WordPiece(unk_token= '[UNK]'))
        tokenizer.pre_tokenizer = Whitespace()
        trainer = WordPieceTrainer(special_tokens = ["[UNK]","[PAD]","[SOS]","[EOS]"],min_frequency = 2)
        tokenizer.train_from_iterator(get_all_sentence(ds,lang),trainer=trainer)
        tokenizer.save(str(tokenizer_path))
    else:
        tokenizer = Tokenizer.from_file(str(tokenizer_path))

    return tokenizer

def get_ds(config):

    ds_raw = load_dataset(f"{config['datasource']}", f"default", split='train')

    tokenizer_src = tokenizer_builder(config,ds_raw,config['lang_src'])
    tokenizer_trg = tokenizer_builder(config,ds_raw,config['lang_trg'])

    train_ds_size = int(0.9*len(ds_raw))
    test_ds_size = len(ds_raw) - train_ds_size
    train_ds_raw,test_ds_raw = random_split(ds_raw,[train_ds_size,test_ds_size])

    train_ds = BilingualDataset(train_ds_raw,tokenizer_src,tokenizer_trg,config['lang_src'],config['lang_trg'],config['seq_len'])
    test_ds = BilingualDataset(test_ds_raw,tokenizer_src,tokenizer_trg,config['lang_src'],config['lang_trg'],config['seq_len'])

    max_len_src = 0
    max_len_trg = 0

    for item in ds_raw:
        src_ids = tokenizer_src.encode(item[config['lang_src']]).ids
        trg_ids = tokenizer_trg.encode(item[config['lang_trg']]).ids
        max_len_src = max(max_len_src, len(src_ids))
        max_len_trg = max(max_len_trg, len(trg_ids))
    
    print(f'Max length of source sentence: {max_len_src}')
    print(f'Max length of target sentence: {max_len_trg}')

    train_dataloder = DataLoader(train_ds,batch_size=config['batch_size'],shuffle=True)
    test_dataloder = DataLoader(test_ds,batch_size=1,shuffle=True) # batch is 1 in train dataloder as it is predicting a single world

    return train_dataloder,test_dataloder,tokenizer_src,tokenizer_trg

def get_model(config,src_vocab_len,trg_vocab_len):

    model = Transfromer_builder(src_vocab_size=src_vocab_len,trg_vocab_size=trg_vocab_len,src_seq_len=config['seq_len'],trg_seq_len=config['seq_len'],d_model=config['d_model'])
    
    return model

def greedy_decode(model,encoder,encoder_mask,tokenizer_src,tokenizer_trg,max_len,device):
    sos_idx = tokenizer_trg.token_to_id('[SOS]')
    eos_idx = tokenizer_trg.token_to_id('[EOS]')
    
    encoder_output = model.encode(encoder,encoder_mask)

    decoder_input = torch.empty(1,1).fill_(sos_idx).type_as(encoder).to(device)
    while True:
        if decoder_input.size(1) == max_len:
            break

        decoder_mask = casual_mask(decoder_input.size(1)).type_as(encoder_mask).to(device)

        out = model.Decode(encoder_output,decoder_input,encoder_mask,decoder_mask)

        prob = model.projecter(out[:,-1])
        _ ,next_word = torch.max(prob,dim=1)

        decoder_input = torch.cat([decoder_input,torch.empty(1,1).type_as(encoder).fill_(next_word.item()).to(device)],dim=1)

        if next_word == eos_idx:
            break
    return decoder_input.squeeze(0)

def test_loop(model,test_ds,tokenizer_src,tokenizer_trg,max_len,device,print_msg,global_step,writer,num_examples = 2):
    model.eval() 
    count = 0

    source_txts = []
    target_txts = []
    predicted_texts = []

    console_width = 80

    with torch.no_grad():
        for batch in test_ds:
            count += 1
            encoder_input = batch['encoder_input'].to(device)
            encoder_mask = batch['encoder_mask'].to(device)

            assert encoder_input.size(0) == 1 ,"batch size must be 1 as your testing"

            model_out = greedy_decode(model,encoder_input,encoder_mask,tokenizer_src,tokenizer_trg,max_len,device)

            source_txt = batch['src_text'][0]
            target_txt = batch['trg_text'][0]
            model_out_text = tokenizer_trg.decode(model_out.detach().cpu().numpy())


            source_txts.append(source_txt)
            target_txts.append(target_txt)
            predicted_texts.append(model_out_text)

            print_msg('-'*console_width)
            print_msg(f'{f'SOURCE: ':>12}{source_txt}')
            print_msg(f'{f'TARGET: ':>12}{target_txt}')

            out_txt = ''
            for i in model_out_text:
                if i == '#':
                    continue
                else:
                    out_txt += i

            print_msg(f'{f'PREDICTED: ':>12}{out_txt}')

            if count == num_examples:
                print_msg('-'*console_width)
                break
    if writer: 
        metric = torchmetrics.text.CharErrorRate()
        cer = metric(model_out_text, target_txt)
        writer.add_scalar('validation cer', cer, global_step)
        writer.flush()

        metric = torchmetrics.text.WordErrorRate()
        wer = metric(model_out_text, target_txt)
        writer.add_scalar('validation wer', wer, global_step)
        writer.flush()

        metric = torchmetrics.text.BLEUScore()
        bleu = metric(model_out_text, target_txt)
        writer.add_scalar('validation BLEU', bleu, global_step)
        writer.flush()


def train_model(config):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    Path(f"{config['datasource']}_{config['model_folder']}").mkdir(parents=True, exist_ok=True)

    print(f"the device used is {device}")

    train_dataloder,test_dataloder,tokenizer_src,tokenizer_trg = get_ds(config)
    model = get_model(config,tokenizer_src.get_vocab_size(),tokenizer_trg.get_vocab_size()).to(device)

    writer = SummaryWriter(config['experiment_name'])
    optimizer = torch.optim.Adam(params= model.parameters(),lr=config['lr'],eps=1e-9)

    initial_epoch = 0
    global_step = 0

    preload = config['preload']
    model_filename = latest_weights_file_path(config) if preload == 'latest' else get_weights_file_path(config, preload) if preload else None
    if model_filename:
        print(f'Preloading model {model_filename}')
        state = torch.load(model_filename)
        model.load_state_dict(state['model_state_dict'])
        initial_epoch = state['epoch'] + 1
        optimizer.load_state_dict(state['optimizer_state_dict'])
        global_step = state['global_step']
    else:
        print('No model to preload, starting from scratch')

    loss_fn = nn.CrossEntropyLoss(ignore_index=tokenizer_src.token_to_id('[PAD]'),label_smoothing=0.1).to(device)

    for epoch in range(initial_epoch,config['num_epochs']):
        torch.cuda.empty_cache()
        model.train()
        batch_itrator = tqdm(train_dataloder,desc=f'processing epoch {epoch:02d}')

        for batch in batch_itrator:
            

            encoder_input = batch['encoder_input'].to(device)#(B,seq_len)
            decoder_input = batch['decoder_input'].to(device)#(B,seq_len)
            encoder_mask = batch['encoder_mask'].to(device)#(B,1,1,seq_len)
            decoder_mask = batch['decoder_mask'].to(device)#(B,1,seq_len,seq_len)
            encoder_output = model.encode(encoder_input,encoder_mask)
            decoder_output = model.decode(encoder_output,decoder_input,encoder_mask,decoder_mask)
            proj_output = model.project(decoder_output) #(B,seq_len,trg_vocab_size)

            label = batch['label'].to(device)#(B,seq_len)

            #(B,seq_len,trg_vocab_size) --> (B*seq_len,trg_vocab_size)
            loss = loss_fn(proj_output.view(-1,tokenizer_trg.get_vocab_size()),label.view(-1))
            batch_itrator.set_postfix({"loss" : f"{loss.item():6.3f}"})

            #log the loss
            writer.add_scalar('train_loss' , loss.item(),global_step)
            writer.flush()

            loss.backward()

            optimizer.step()
            optimizer.zero_grad(set_to_none=True)

            global_step += 1

        test_loop(model,test_dataloder,tokenizer_src,tokenizer_trg,config['seq_len'],device,lambda msg:batch_itrator.write(msg),global_step,writer)

        model_filename = get_weights_file_path(config,f'{epoch:02d}')
        torch.save({
            'epoch' : epoch,
            'model_state_dict' : model.state_dict(),
            'optimizer_state_dict' : optimizer.state_dict(),
            'global_step' : global_step 
        },model_filename)

if __name__ == '__main__':
    warnings.filterwarnings('ignore')
    config = get_config()
    train_model(config)