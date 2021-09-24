# -*-coding:utf-8-*- 
import torch
import torch.nn as nn
import torch.nn.functional as F
from hyparams import HyParams as hp

class MAG(nn.Module):
    def __init__(self, hidden_size, beta_shift, dropout_prob):
        super(MAG, self).__init__()
        print(
            "Initializing MAG with beta_shift:{} hidden_prob:{}".format(
                beta_shift, dropout_prob
            )
        )

        self.W_hv = nn.Linear(hp.VISUAL_DIM + hp.TEXT_DIM, hp.TEXT_DIM)
        self.W_ha = nn.Linear(hp.ACOUSTIC_DIM + hp.TEXT_DIM, hp.TEXT_DIM)

        self.W_v = nn.Linear(hp.VISUAL_DIM, hp.TEXT_DIM)
        self.W_a = nn.Linear(hp.ACOUSTIC_DIM, hp.TEXT_DIM)

        self.beta_shift = beta_shift

        self.LayerNorm = nn.LayerNorm(hidden_size)
        self.dropout = nn.Dropout(dropout_prob)

    def forward(self, text_embedding, visual, acoustic):
        eps = 1e-6
        weight_v = F.relu(self.W_hv(torch.cat((visual, text_embedding), dim=-1)))
        weight_a = F.relu(self.W_ha(torch.cat((acoustic, text_embedding), dim=-1)))
        # weight_v = F.sigmoid(self.W_hv(torch.cat((visual, text_embedding), dim=-1)))
        # weight_a = F.sigmoid(self.W_ha(torch.cat((acoustic, text_embedding), dim=-1)))

        h_m = weight_v * self.W_v(visual) + weight_a * self.W_a(acoustic)

        em_norm = text_embedding.norm(2, dim=-1)
        hm_norm = h_m.norm(2, dim=-1)

        hm_norm_ones = torch.ones(hm_norm.shape, requires_grad=True).to(hp.DEVICE)
        hm_norm = torch.where(hm_norm == 0, hm_norm_ones, hm_norm)

        thresh_hold = (em_norm / (hm_norm + eps)) * self.beta_shift

        ones = torch.ones(thresh_hold.shape, requires_grad=True).to(hp.DEVICE)

        alpha = torch.min(thresh_hold, ones)
        alpha = alpha.unsqueeze(dim=-1)

        acoustic_vis_embedding = alpha * h_m

        embedding_output = self.dropout(
            self.LayerNorm(acoustic_vis_embedding + text_embedding)
        )

        return embedding_output


class WM(nn.Module):
    def __init__(self, hidden_size, beta_shift, dropout_prob):
        super(WM, self).__init__()
        print(
            "Initializing WHAT with beta_shift:{} hidden_prob:{}".format(
                beta_shift, dropout_prob
            )
        )
        self.W_v = nn.Linear(hp.VISUAL_DIM, hp.TEXT_DIM)
        self.W_a = nn.Linear(hp.ACOUSTIC_DIM, hp.TEXT_DIM)

        self.W_v_ = nn.Linear(hp.VISUAL_DIM, hp.TEXT_DIM)
        self.W_a_ = nn.Linear(hp.ACOUSTIC_DIM, hp.TEXT_DIM)

        self.W_hv = nn.Linear(hp.VISUAL_DIM + hp.TEXT_DIM, hp.TEXT_DIM)
        self.W_ha = nn.Linear(hp.ACOUSTIC_DIM + hp.TEXT_DIM, hp.TEXT_DIM)

        self.beta_shift = beta_shift
        self.LayerNorm = nn.LayerNorm(hidden_size)
        self.dropout = nn.Dropout(dropout_prob)

    def forward(self, text, visual, acoustic):
        eps = 1e-6
        new_v = self.W_v(visual)
        new_a = self.W_a(acoustic)
        aligned_v = torch.bmm(F.softmax(torch.bmm(text, torch.transpose(new_v, 1, 2)), dim=-1), visual)
        aligned_a = torch.bmm(F.softmax(torch.bmm(text, torch.transpose(new_a, 1, 2)), dim=-1), acoustic)

        weight_v = F.relu(self.W_hv(torch.cat((aligned_v, text), dim=-1)))
        weight_a = F.relu(self.W_ha(torch.cat((aligned_a, text), dim=-1)))

        h_m = weight_v * self.W_v_(aligned_v) + weight_a * self.W_a_(aligned_a)

        em_norm = text.norm(2, dim=-1)
        hm_norm = h_m.norm(2, dim=-1)

        hm_norm_ones = torch.ones(hm_norm.shape, requires_grad=True).to(hp.DEVICE)
        hm_norm = torch.where(hm_norm == 0, hm_norm_ones, hm_norm)

        thresh_hold = (em_norm / (hm_norm + eps)) * self.beta_shift

        ones = torch.ones(thresh_hold.shape, requires_grad=True).to(hp.DEVICE)

        alpha = torch.min(thresh_hold, ones)
        alpha = alpha.unsqueeze(dim=-1)

        av_embedding = alpha * h_m

        embedding_output = self.dropout(
            self.LayerNorm(av_embedding + text)
        )

        return embedding_output