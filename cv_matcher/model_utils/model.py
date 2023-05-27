import sys
sys.path.append('.\\data')
import torch
from torch import nn
from constants import MODEL_PARAMETERS



class SiameseCVNet(nn.Module):
    def __init__(self, vac_vocab_size, res_vocab_size,
                 embedding_dim, rnn_hidden_dim, rnn_type, dropout, bidir,
                 hidden_layers, fc1_output=512, fc2_output=128):

        super(SiameseCVNet, self).__init__()

        self.vac_vocab_size = vac_vocab_size
        self.res_vocab_size = res_vocab_size
        self.rnn_type = rnn_type
        self.bidir = bidir

        self.embedding_dim = embedding_dim
        self.rnn_hidden_dim = rnn_hidden_dim
        self.hidden_layers = hidden_layers

        if rnn_type == 'LSTM':
            self.fc1_input_one = 2 * ((
                                              int(bidir) + 1) * self.hidden_layers * self.rnn_hidden_dim + self.rnn_hidden_dim + self.embedding_dim)

        elif rnn_type == 'GRU':
            self.fc1_input_one = 2 * ((
                                              int(bidir) + 1) * self.hidden_layers * self.rnn_hidden_dim + self.rnn_hidden_dim + self.embedding_dim) - self.rnn_hidden_dim

        # но мы конкатенируем 2 сэмпла!
        self.fc1_input = 2 * self.fc1_input_one

        self.fc1_output = fc1_output
        self.fc2_output = fc2_output

        self.vac_embed = nn.Embedding(vac_vocab_size, embedding_dim)
        self.res_embed = nn.Embedding(res_vocab_size, embedding_dim)

        if rnn_type == 'LSTM':
            self.rnn = nn.LSTM(input_size=embedding_dim,
                               hidden_size=rnn_hidden_dim,
                               num_layers=hidden_layers,
                               batch_first=True,
                               bidirectional=bidir)

        elif rnn_type == 'GRU':
            self.rnn = nn.GRU(input_size=embedding_dim,
                              hidden_size=rnn_hidden_dim,
                              num_layers=hidden_layers,
                              batch_first=True,
                              bidirectional=bidir)

        fc1 = nn.Linear(self.fc1_input, self.fc1_output)
        relu = nn.ReLU()
        fc2 = nn.Linear(self.fc1_output, self.fc2_output)
        sigmoid = nn.Sigmoid()

        if dropout > 0:
            self.nn_head = nn.Sequential(
                fc1,
                relu,
                nn.Dropout(dropout),
                fc2,
                sigmoid
            )
        else:
            self.nn_head = nn.Sequential(
                fc1,
                relu,
                fc2,
                sigmoid
            )

    def forward(self, vac_text, res_text):
        vac_embeds = self.vac_embed(vac_text)
        res_embeds = self.res_embed(res_text)

        catted_output_vac = self.forward_one(vac_embeds)
        catted_output_res = self.forward_one(res_embeds)

        # конкатенируем и пускаем через dense
        catted_output = torch.cat((catted_output_vac, catted_output_res), dim=-1)
        sigm_output = self.nn_head(catted_output)

        return sigm_output

    def forward_one(self, batch):
        '''
        image there is just a tensor of embeddings
        shit with dims for sure
        '''

        # print('sample:', batch.shape)

        if self.rnn_type == 'LSTM':
            rnn_output, (hidden_states, cell_states) = self.rnn(batch)
        elif self.rnn_type == 'GRU':
            rnn_output, hidden_states = self.rnn(batch)

        # print('rnn output:', rnn_output.shape)

        embed_max_pool = batch.max(dim=1)[0]
        embed_avg_pool = batch.sum(dim=1) / len(batch)

        rnn_max_pool = rnn_output.max(dim=1)[0]
        rnn_avg_pool = rnn_output.sum(dim=1) / len(rnn_output)

        # тут 0 ось -- кол-во слоев в rnn-блоке
        hidden_states = torch.cat([hidden_states[i, :, :] for i in range(hidden_states.shape[0])], dim=-1)

        if self.rnn_type == 'LSTM':
            cell_states = torch.cat([cell_states[i, :, :] for i in range(cell_states.shape[0])], dim=-1)
            catted_output = torch.cat((embed_max_pool, embed_avg_pool, rnn_max_pool,
                                       rnn_avg_pool, hidden_states, cell_states), dim=-1)
        elif self.rnn_type == 'GRU':
            catted_output = torch.cat((embed_max_pool, embed_avg_pool, rnn_max_pool,
                                       rnn_avg_pool, hidden_states), dim=-1)

        return catted_output


def create_model():
    model = SiameseCVNet(
        embedding_dim=MODEL_PARAMETERS["embedding_dim"],
        vac_vocab_size=MODEL_PARAMETERS["vac_vocab_size"],
        res_vocab_size=MODEL_PARAMETERS["res_vocab_size"],
        rnn_hidden_dim=MODEL_PARAMETERS["rnn_hidden_dim"],
        rnn_type=MODEL_PARAMETERS["rnn_type"],
        dropout=MODEL_PARAMETERS["dropout"],
        bidir=MODEL_PARAMETERS["bidir"],
        hidden_layers=MODEL_PARAMETERS["hidden_layers"],
        fc1_output=MODEL_PARAMETERS["fc1_output"],
        fc2_output=MODEL_PARAMETERS["fc2_output"]
    )

    return model
