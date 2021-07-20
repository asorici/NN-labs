import torch
import torch.nn as nn

import random

START_TOKEN = "<"
END_TOKEN = ">"
PAD_TOKEN = "_"
UNK_TOKEN = "?"


class Encoder(nn.Module):
    """
    The Encoder Module
    """
    def __init__(self, input_dim=14, emb_dim=64, hid_dim=256, n_layers=2, dropout=0.1):
        """
        :param input_dim: size of the input vocabulary (default 10 digits + 4 extra tokens)
        :param emb_dim: size of the embedding of digits and extra tokens
        :param hid_dim: size of the hidden dimension for the LSTM cell
        :param n_layers: number of LSTM cells
        :param dropout: probability of dropout
        """
        super().__init__()
        
        self.hid_dim = hid_dim
        self.n_layers = n_layers
        
        # TODO: Define the embedding Layer encoding from the input vocabulary dimension to the embedding dimension
        #  see nn.Embedding
        # self.embedding = ...
        
        # TODO: Define the LSTM cells for the encoder module
        #  specify the size of the input, size of the hidden layer, number of LSTM layers and dropout probability
        #  see nn.LSTM
        # self.rnn = ...
        
        self.dropout = nn.Dropout(dropout)
    
    def forward(self, src):
        # The source sequence src is of size [src len, batch size]

        # TODO: obtain the embedding of the sequence to which dropout is applied
        #  Dimension of the embedded results will be [src len, batch size, emb dim]
        embedded = None

        # TODO: run the embedded sequence through the encoder (LSTM layers in self.rnn)
        #  obtain results of the following shapes
        #       outputs = [src len, batch size, hid dim * n directions]
        #       hidden = [n layers * n directions, batch size, hid dim]
        #       cell = [n layers * n directions, batch size, hid dim]
        hidden = None
        cell = None

        # Return the hidden state (h_T) that "encodes" the sequence, as well as the corresponding LSTM `cell` state
        return hidden, cell


class Decoder(nn.Module):
    """
    The Decoder Module
    """
    def __init__(self, output_dim=10, emb_dim=64, hid_dim=256, n_layers=2, dropout=0.1):
        """
        :param output_dim: size of the output vocabulary (default 7 elements + 4 extra tokens)
        :param emb_dim: size of the embedding of roman number representations and extra tokens
        :param hid_dim: size of the hidden dimension for the LSTM cell
        :param n_layers: number of LSTM cells
        :param dropout: probability of dropout
        """
        super().__init__()
        
        self.output_dim = output_dim
        self.hid_dim = hid_dim
        self.n_layers = n_layers
        
        self.embedding = nn.Embedding(output_dim, emb_dim)
        
        self.rnn = nn.LSTM(emb_dim, hid_dim, n_layers, dropout=dropout)
        
        self.fc_out = nn.Linear(hid_dim, output_dim)
        
        self.dropout = nn.Dropout(dropout)
    
    def forward(self, input, hidden, cell):
        # Tensor shape indications
        #   input = [batch size]
        #   hidden = [n layers * n directions, batch size, hid dim]
        #   cell = [n layers * n directions, batch size, hid dim]

        #   n directions in the decoder will both always be 1, therefore:
        #       hidden = [n layers, batch size, hid dim]
        #       context = [n layers, batch size, hid dim]
        #       input = [1, batch size]
        input = input.unsqueeze(0)

        # embedded shape = [1, batch size, emb dim]
        embedded = self.dropout(self.embedding(input))

        # Tensor shape indications
        #   output = [seq len, batch size, hid dim * n directions]
        #   hidden = [n layers * n directions, batch size, hid dim]
        #   cell = [n layers * n directions, batch size, hid dim]
        #   seq len and n directions will always be 1 in the decoder, therefore:
        #       output = [1, batch size, hid dim]
        #       hidden = [n layers, batch size, hid dim]
        #       cell = [n layers, batch size, hid dim]
        output, (hidden, cell) = self.rnn(embedded, (hidden, cell))

        # prediction shape = [batch size, output dim]
        prediction = self.fc_out(output.squeeze(0))

        return prediction, hidden, cell


class Seq2Seq(nn.Module):
    """
    Sequence to Sequence model combining the encoder and decoder
    """
    def __init__(self, encoder, decoder, device):
        super().__init__()
        
        self.encoder = encoder
        self.decoder = decoder
        self.device = device
        
        assert encoder.hid_dim == decoder.hid_dim, \
            "Hidden dimensions of encoder and decoder must be equal!"
        assert encoder.n_layers == decoder.n_layers, \
            "Encoder and decoder must have equal number of layers!"
    
    def forward(self, src, trg, teacher_forcing_ratio=0.5):
        # Tensor shape indications:
        #   src = [src len, batch size]
        #   trg = [trg len, batch size]
        
        # teacher_forcing_ratio is probability to use teacher forcing
        # e.g. if teacher_forcing_ratio is 0.75 we use ground-truth inputs 75% of the time
        batch_size = trg.shape[1]
        trg_len = trg.shape[0]
        trg_vocab_size = self.decoder.output_dim

        # tensor to store decoder outputs
        outputs = torch.zeros(trg_len, batch_size, trg_vocab_size).to(self.device)

        # TODO: encode the input sequence `src`
        #  obtain th last hidden state of the encoder ito use as the initial hidden state of the decoder
        # hidden, cell = ...

        # TODO: decode the output, one token at a time
        # first input to the decoder is the <sos> tokens
        input = trg[0, :]
        for t in range(1, trg_len):
            # TODO: insert input token embedding, previous hidden and previous cell states to decoder
            #  receive output tensor (predictions) and new hidden and cell states
            # output, hidden, cell = ...

            # place predictions in a tensor holding predictions for each token
            outputs[t] = output

            # decide if we are going to use teacher forcing or not
            teacher_force = random.random() < teacher_forcing_ratio

            # TODO: get the highest predicted token from our predictions
            # top1 = ...

            # TODO: if teacher forcing, use actual next token as next input
            #       if not, use predicted token
            # input = ...
        
        return outputs

