import torch
from torch import nn


class S2VT(nn.Module):
    def __init__(
        self,
        vocab_size,
        word2index,
        frame_dim=4096,
        hidden=500,
        dropout=0.5,
        n_step=80,
    ):
        super(S2VT, self).__init__()
        self.frame_dim = frame_dim
        self.hidden = hidden
        self.n_step = n_step

        self.word2index = word2index

        self.drop = nn.Dropout(p=dropout)
        self.linear1 = nn.Linear(frame_dim, hidden)
        self.linear2 = nn.Linear(hidden, vocab_size)

        self.lstm1 = nn.LSTM(hidden, hidden, batch_first=True, dropout=dropout)
        self.lstm2 = nn.LSTM(2 * hidden, hidden, batch_first=True, dropout=dropout)

        self.embedding = nn.Embedding(vocab_size, hidden)

    def forward(self, video, caption=None):
        batch_size = video.size(0)
        video = video.contiguous().view(-1, self.frame_dim)
        video = self.drop(video)
        video = self.linear1(video)
        video = video.view(-1, self.n_step, self.hidden)
        padding = torch.zeros([batch_size, self.n_step - 1, self.hidden])
        video = torch.cat((video, padding), 1)
        vid_out, state_vid = self.lstm1(video)

        if self.training:
            caption = self.embedding(caption[:, 0 : self.n_step - 1])
            padding = torch.zeros([batch_size, self.n_step, self.hidden])
            caption = torch.cat((padding, caption), 1)
            caption = torch.cat((caption, vid_out), 2)

            cap_out, state_cap = self.lstm2(caption)
            cap_out = cap_out[:, self.n_step :, :]
            cap_out = cap_out.contiguous().view(-1, self.hidden)
            cap_out = self.drop(cap_out)
            cap_out = self.linear2(cap_out)
            return cap_out
        else:
            padding = torch.zeros([batch_size, self.n_step, self.hidden])
            cap_input = torch.cat((padding, vid_out[:, 0 : self.n_step, :]), 2)
            cap_out, state_cap = self.lstm2(cap_input)

            bos_id = self.word2index["<BOS>"] * torch.ones(batch_size, dtype=torch.long)
            bos_id = bos_id
            cap_input = self.embedding(bos_id)
            cap_input = torch.cat((cap_input, vid_out[:, self.n_step, :]), 1)
            cap_input = cap_input.view(batch_size, 1, 2 * self.hidden)

            cap_out, state_cap = self.lstm2(cap_input, state_cap)
            cap_out = cap_out.contiguous().view(-1, self.hidden)
            cap_out = self.drop(cap_out)
            cap_out = self.linear2(cap_out)
            cap_out = torch.argmax(cap_out, 1)

            caption = []
            caption.append(cap_out)
            for i in range(self.n_step - 2):
                cap_input = self.embedding(cap_out)
                cap_input = torch.cat(
                    (cap_input, vid_out[:, self.n_step + 1 + i, :]), 1
                )
                cap_input = cap_input.view(batch_size, 1, 2 * self.hidden)

                cap_out, state_cap = self.lstm2(cap_input, state_cap)
                cap_out = cap_out.contiguous().view(-1, self.hidden)
                cap_out = self.drop(cap_out)
                cap_out = self.linear2(cap_out)
                cap_out = torch.argmax(cap_out, 1)
                caption.append(cap_out)
            return caption
