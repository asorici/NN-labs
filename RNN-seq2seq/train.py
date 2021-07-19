import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np

import random
import time
import math
from model import Encoder, Decoder, Seq2Seq
from utils import NumeralsDataset, generate_dataset
from torchtext.legacy.data import BucketIterator

from utils import SRC, TRG

# set the random seeds for deterministic results
SEED = 1234

random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
BATCH_SIZE = 32

INPUT_DIM = 14  # 10 digits + <, >, _ and ?
OUTPUT_DIM = 11  # 7 digits + <, >, _ and ?
ENC_EMB_DIM = 64
DEC_EMB_DIM = 64
HID_DIM = 256
N_LAYERS = 2
ENC_DROPOUT = 0.5
DEC_DROPOUT = 0.5

N_EPOCHS = 40
CLIP = 1


def init_weights(m):
    for name, param in m.named_parameters():
        nn.init.uniform_(param.data, -0.08, 0.08)


def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def epoch_time(start_time, end_time):
    elapsed_time = end_time - start_time
    elapsed_mins = int(elapsed_time / 60)
    elapsed_secs = int(elapsed_time - (elapsed_mins * 60))
    return elapsed_mins, elapsed_secs


def train(model, iterator, optimizer, criterion, clip):
    model.train()
    
    epoch_loss = 0
    
    for i, batch in enumerate(iterator):
        src = batch.src
        trg = batch.trg
        
        optimizer.zero_grad()
        
        output = model(src, trg)

        # trg = [trg len, batch size]
        # output = [trg len, batch size, output dim]
        
        output_dim = output.shape[-1]
        
        output = output[1:].view(-1, output_dim)
        trg = trg[1:].view(-1)

        # trg = [(trg len - 1) * batch size]
        # output = [(trg len - 1) * batch size, output dim]
        
        loss = criterion(output, trg)
        
        loss.backward()
        
        torch.nn.utils.clip_grad_norm_(model.parameters(), clip)
        
        optimizer.step()
        
        epoch_loss += loss.item()
    
    return epoch_loss / len(iterator)


def evaluate(model, iterator, criterion):
    model.eval()
    
    epoch_loss = 0
    
    with torch.no_grad():
        for i, batch in enumerate(iterator):
            src = batch.src
            trg = batch.trg
            
            output = model(src, trg, 0)  # turn off teacher forcing

            # trg = [trg len, batch size]
            # output = [trg len, batch size, output dim]
            
            output_dim = output.shape[-1]
            
            output = output[1:].view(-1, output_dim)
            trg = trg[1:].view(-1)

            # trg = [(trg len - 1) * batch size]
            # output = [(trg len - 1) * batch size, output dim]
            
            loss = criterion(output, trg)
            
            epoch_loss += loss.item()
    
    return epoch_loss / len(iterator)


if __name__ == "__main__":
    enc = Encoder(INPUT_DIM, ENC_EMB_DIM, HID_DIM, N_LAYERS, ENC_DROPOUT)
    dec = Decoder(OUTPUT_DIM, DEC_EMB_DIM, HID_DIM, N_LAYERS, DEC_DROPOUT)
    
    model = Seq2Seq(enc, dec, device).to(device)
    model.apply(init_weights)

    print(f'The model has {count_parameters(model):,} trainable parameters')
    print(model)

    optimizer = optim.Adam(model.parameters())
    
    numeral_examples = generate_dataset()
    numerals_dataset = NumeralsDataset(numeral_examples)
    
    train_data, test_data = numerals_dataset.split(split_ratio=0.8)
    
    SRC.build_vocab(train_data)
    TRG.build_vocab(train_data)
    TRG_PAD_IDX = TRG.vocab.stoi[TRG.pad_token]
    
    train_iterator, test_iterator = BucketIterator.splits((train_data, test_data),
                                                          batch_sizes=(BATCH_SIZE, BATCH_SIZE),
                                                          device=device,
                                                          sort_key=lambda x: len(x.src),
                                                          
                                                          # Sort all examples in data using `sort_key`.
                                                          sort=True,

                                                          # Shuffle data on each epoch run.
                                                          shuffle=True,

                                                          # Use `sort_key` to sort examples in each batch.
                                                          sort_within_batch=True
                                                          )
    

    criterion = nn.CrossEntropyLoss(ignore_index=TRG_PAD_IDX)

    

    best_valid_loss = float('inf')

    for epoch in range(N_EPOCHS):
    
        start_time = time.time()
    
        train_loss = train(model, train_iterator, optimizer, criterion, CLIP)
        # valid_loss = evaluate(model, valid_iterator, criterion)
    
        end_time = time.time()
    
        epoch_mins, epoch_secs = epoch_time(start_time, end_time)
    
        # if valid_loss < best_valid_loss:
        #     best_valid_loss = valid_loss
        #     torch.save(model.state_dict(), 'tut1-model.pt')
        #
        print(f'Epoch: {epoch + 1:02} | Time: {epoch_mins}m {epoch_secs}s')
        print(f'\tTrain Loss: {train_loss:.3f} | Train PPL: {math.exp(train_loss):7.3f}')
        
        model_name = "numeral-conversion-model-sort_by_src_all-batch_%s-epochs_%s-dropout_%s.pt" % (str(BATCH_SIZE), str(N_EPOCHS),
                                                                                    str(DEC_DROPOUT))
        
        torch.save(model.state_dict(), 'models/' + model_name)
        # print(f'\t Val. Loss: {valid_loss:.3f} |  Val. PPL: {math.exp(valid_loss):7.3f}')

    test_loss = evaluate(model, test_iterator, criterion)
    print(f'| Test Loss: {test_loss:.3f} | Test PPL: {math.exp(test_loss):7.3f} |')

