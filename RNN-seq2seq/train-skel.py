import random
import time
import math
from argparse import ArgumentParser

import torch
import torch.nn as nn
import torch.optim as optim
from torchtext.legacy.data import BucketIterator
import numpy as np

from model import Encoder, Decoder, Seq2Seq
from utils import NumeralsDataset, generate_dataset, load_configurations
from utils import SRC, TRG


"""
Method to initialize the weights of a model to have random values in between -0.08 and 0.08
"""
def init_weights(m):
    for name, param in m.named_parameters():
        nn.init.uniform_(param.data, -0.08, 0.08)

"""
Auxiliary methods to count the parameters of the model and return time elapsed between two epochs
"""
def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def epoch_time(start_time, end_time):
    elapsed_time = end_time - start_time
    elapsed_mins = int(elapsed_time / 60)
    elapsed_secs = int(elapsed_time - (elapsed_mins * 60))
    return elapsed_mins, elapsed_secs


def train(model, iterator, optimizer, criterion, clip):
    """
    Main training function
    :param model: The used seq2seq model
    :param iterator: The dataset iterator used to produce batches of training data
    :param optimizer: The used optimizer
    :param criterion: The loss criterion
    :param clip: clip treshold for gradient stabilization
    """
    # set the model in training mode
    model.train()
    epoch_loss = 0
    
    # main training loop
    for i, batch in enumerate(iterator):
        # get batch of source (arabic numeral) and target (roman numeral) sequences
        src = batch.src
        trg = batch.trg
        
        optimizer.zero_grad()
        
        # Tensor shape indications:
        #   trg = [trg len, batch size]
        #   output = [trg len, batch size, output dim]
        # TODO: get the output predicted by the model
        # output = ...

        # TODO: Create views of the output and target tensors so as to apply the CrossEntropyLoss criterion
        #   over all tokens in the target sequence, ignoring the first. See torch.tensor.view()
        #   trg = [(trg len - 1) * batch size]
        #   output = [(trg len - 1) * batch size, output dim]
        # output = ...
        # trg = ...
        
        # TODO: apply the CrossEntropyLoss between output and trg
        loss = None
        loss.backward()
        
        # Clip gradients if their norm is too large to ensure training stability
        torch.nn.utils.clip_grad_norm_(model.parameters(), clip)
        
        # Perform one step of optimization
        optimizer.step()
        
        epoch_loss += loss.item()
    
    return epoch_loss / len(iterator)


def evaluate(model, iterator, criterion):
    """
    Method to evaluate a trained model
    :param model:
    :param iterator:
    :param criterion:
    :return:
    """
    # set model in evaluation mode
    model.eval()
    epoch_loss = 0
    
    with torch.no_grad():
        for i, batch in enumerate(iterator):
            src = batch.src
            trg = batch.trg

            # Tensor shape indications
            #   trg = [trg len, batch size]
            #   output = [trg len, batch size, output dim]
            
            # TODO: get the output predicted by the model, WITHOUT applying teacher forcing
            # output = ...

            # TODO: Obtain views of the output and target tensors as in the training case
            #   trg = [(trg len - 1) * batch size]
            #   output = [(trg len - 1) * batch size, output dim]
            # output = ...
            # trg = ...
            
            # TODO: apply the CrossEntropy loss criterion
            loss = None
            epoch_loss += loss.item()
    
    return epoch_loss / len(iterator)


if __name__ == "__main__":
    arg_parser = ArgumentParser(description='.')
    arg_parser.add_argument('--config', type=str, required=True, default="config.yml")
    args = arg_parser.parse_args()
    
    config = load_configurations(args.config)

    SEED = config["SEED"]
    
    INPUT_DIM = config["INPUT_DIM"]
    OUTPUT_DIM = config["OUTPUT_DIM"]
    ENC_EMB_DIM = config["ENC_EMB_DIM"]
    DEC_EMB_DIM = config["DEC_EMB_DIM"]
    HID_DIM = config["HID_DIM"]
    N_LAYERS = config["N_LAYERS"]
    ENC_DROPOUT = config["ENC_DROPOUT"]
    DEC_DROPOUT = config["ENC_DROPOUT"]
    
    BATCH_SIZE = config["BATCH_SIZE"]
    N_EPOCHS = config["N_EPOCHS"]
    CLIP = config["CLIP"]
    
    # set the randomness SEED to ensure repeatable results across runs of the script
    random.seed(SEED)
    np.random.seed(SEED)
    torch.manual_seed(SEED)
    
    # set the device to run on
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # create and parameterize encoder and decoder models
    enc = Encoder(INPUT_DIM, ENC_EMB_DIM, HID_DIM, N_LAYERS, ENC_DROPOUT)
    dec = Decoder(OUTPUT_DIM, DEC_EMB_DIM, HID_DIM, N_LAYERS, DEC_DROPOUT)
    
    # create seq2se1 model and initialize its weights
    model = Seq2Seq(enc, dec, device).to(device)
    model.apply(init_weights)

    print(f'The model has {count_parameters(model):,} trainable parameters')
    print(model)
    
    # define the optimizer
    optimizer = optim.Adam(model.parameters())
    
    # generate the dataset examples
    numeral_examples = generate_dataset()
    numerals_dataset = NumeralsDataset(numeral_examples)
    
    # split dataset in 80% for training and the rest for testing
    train_data, test_data = numerals_dataset.split(split_ratio=0.8)
    
    # build the vocabularies for source and target numerals: this should results in a vocabulary of
    # size 14 for arabic numerals and 11 for roman numerals
    SRC.build_vocab(train_data)
    TRG.build_vocab(train_data)
    TRG_PAD_IDX = TRG.vocab.stoi[TRG.pad_token]
    
    # Create a batch iterator - use the torchtext BucketIterator
    # The BucketIterator *sorts* the examples by the length of their `source` numerals
    # This is an important setup, because it allows the network to learn over batches that represent:
    # units, tens, hundreds, thousands
    #
    # NOTE! Such sorting is only possible since the dataset is small.
    # TODO: study what happens if you DO NOT sort all the examples in the dataset: i.e. set sort = False in the
    #  configuration of the BucketIterator
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
    
    # The loss criterion is the CrossEntropyLoss
    criterion = nn.CrossEntropyLoss(ignore_index=TRG_PAD_IDX)

    # Run the training over several epochs
    best_valid_loss = float('inf')
    
    for epoch in range(N_EPOCHS):
    
        start_time = time.time()
        train_loss = train(model, train_iterator, optimizer, criterion, CLIP)
        end_time = time.time()
    
        epoch_mins, epoch_secs = epoch_time(start_time, end_time)
    
        print(f'EPOCH: {epoch + 1:02} | Time: {epoch_mins}m {epoch_secs}s')
        print(f'\tTRAIN Loss: {train_loss:.3f} | Train PPL: {math.exp(train_loss):7.3f}')
        
        if (epoch + 1) % 10 == 5:
            # test the current model
            test_loss = evaluate(model, test_iterator, criterion)
            print(f'\tTEST Loss: {test_loss:.3f} | Test PPL: {math.exp(test_loss):7.3f} |')
        
    # Save the trained model to be
    model_name = "numeral-conversion-model-sort_by_src_all-batch_%s-epochs_%s-dropout_%s.pt" \
                 % (str(BATCH_SIZE), str(N_EPOCHS), str(DEC_DROPOUT))
    torch.save(model.state_dict(), 'models/' + model_name)
    
    # Test the model at the end
    test_loss = evaluate(model, test_iterator, criterion)
    print(f'| TEST Loss: {test_loss:.3f} | Test PPL: {math.exp(test_loss):7.3f} |')

