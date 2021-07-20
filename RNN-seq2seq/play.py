import random
import torch
import numpy as np
from model import Encoder, Decoder, Seq2Seq

from utils import NumeralsDataset, generate_dataset, load_configurations
from utils import SRC, TRG
from torchtext.legacy.data import BucketIterator

from argparse import ArgumentParser


def play(model, iterator):
    model.eval()
    
    with torch.no_grad():
        for i, batch in enumerate(iterator):
            src = batch.src
            trg = batch.trg
            
            # print("src", src[:-1, 0])
            # print("trg", trg[:-1, 0])
            
            src_numeral = "".join([SRC.vocab.itos[idx] for idx in src[:, 0]])
            trg_numeral = "".join([TRG.vocab.itos[idx] for idx in trg[:, 0]])
            
            print("src_numeral", src_numeral)
            print("trg_numeral", trg_numeral)
            
            output = model(src, trg, 0)  # turn off teacher forcing
            
            # trg = [trg len, batch size]
            # output = [trg len, batch size, output dim]
            
            output_dim = output.shape[-1]
            
            output = output[1:].view(-1, output_dim)
            trg = trg[1:].view(-1)
            
            # trg = [(trg len - 1) * batch size]
            # output = [(trg len - 1) * batch size, output dim]
            
            # print(output.shape)
            # print(output.argmax(1))
            pred_numeral = "".join([TRG.vocab.itos[idx] for idx in output.argmax(1)])
            print("predicted_numeral", pred_numeral)
            print("\n")
            

if __name__=="__main__":
    arg_parser = ArgumentParser(description='.')
    arg_parser.add_argument('--config', type=str, required=True, default="config.yml")
    arg_parser.add_argument('--model', type=str, required=True)
    args = arg_parser.parse_args()

    config = load_configurations(args.config)
    model_path = args.model

    SEED = config["SEED"]

    INPUT_DIM = config["INPUT_DIM"]
    OUTPUT_DIM = config["OUTPUT_DIM"]
    ENC_EMB_DIM = config["ENC_EMB_DIM"]
    DEC_EMB_DIM = config["DEC_EMB_DIM"]
    HID_DIM = config["HID_DIM"]
    N_LAYERS = config["N_LAYERS"]
    ENC_DROPOUT = config["ENC_DROPOUT"]
    DEC_DROPOUT = config["ENC_DROPOUT"]

    # in the play setting we run 1 example at a time to be able to visualize it more easily
    BATCH_SIZE = 1

    random.seed(SEED)
    np.random.seed(SEED)
    torch.manual_seed(SEED)
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    enc = Encoder(INPUT_DIM, ENC_EMB_DIM, HID_DIM, N_LAYERS, ENC_DROPOUT)
    dec = Decoder(OUTPUT_DIM, DEC_EMB_DIM, HID_DIM, N_LAYERS, DEC_DROPOUT)
    
    model = Seq2Seq(enc, dec, device).to(device)
    model.load_state_dict(torch.load(model_path))

    # Get the dataset. The predefined random SEED ensures we get the same split
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
    
    play(model, test_iterator)

