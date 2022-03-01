# -*-coding:utf-8-*- 
import torch
import torch.nn as nn
import torch.nn.functional as F
from hyparams import HyParams as hp
from utils import get_device

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


class WA(nn.Module):
    def __init__(self, hidden_size, beta_shift, dropout_prob):
        super(WA, self).__init__()
        print(
            "Initializing WA with beta_shift:{} hidden_prob:{}".format(
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


class Sparsemax(nn.Module):
    """Sparsemax function."""

    def __init__(self, dim=None):
        """Initialize sparsemax activation
        Args:
            dim (int, optional): The dimension over which to apply the sparsemax function.
        """
        super(Sparsemax, self).__init__()

        self.dim = -1 if dim is None else dim


    def forward(self, input):
        """Forward function.
        Args:
            input (torch.Tensor): Input tensor. First dimension should be the batch size
        Returns:
            torch.Tensor: [batch_size x number_of_logits] Output tensor
        """
        # Sparsemax currently only handles 2-dim tensors,
        # so we reshape to a convenient shape and reshape back after sparsemax

        input = input.transpose(0, self.dim)
        original_size = input.size()
        input = input.reshape(input.size(0), -1)
        input = input.transpose(0, 1)
        dim = 1

        number_of_logits = input.size(dim)

        # Translate input by max for numerical stability
        input = input - torch.max(input, dim=dim, keepdim=True)[0].expand_as(input)

        # Sort input in descending order.
        # (NOTE: Can be replaced with linear time selection method described here:
        # http://stanford.edu/~jduchi/projects/DuchiShSiCh08.html)
        zs = torch.sort(input=input, dim=dim, descending=True)[0]
        range = torch.arange(start=1, end=number_of_logits + 1, step=1, dtype=input.dtype, device=get_device()).view(1, -1)
        range = range.expand_as(zs)

        # Determine sparsity of projection
        bound = 1 + range * zs
        cumulative_sum_zs = torch.cumsum(zs, dim)
        is_gt = torch.gt(bound, cumulative_sum_zs).type(input.type())
        k = torch.max(is_gt * range, dim, keepdim=True)[0]

        # Compute threshold function
        zs_sparse = is_gt * zs

        # Compute taus
        taus = (torch.sum(zs_sparse, dim, keepdim=True) - 1) / k
        taus = taus.expand_as(input)

        # Sparsemax
        self.output = torch.max(torch.zeros_like(input), input - taus)

        # Reshape back to original shape
        output = self.output
        output = output.transpose(0, 1)
        output = output.reshape(original_size)
        output = output.transpose(0, self.dim)

        return output


class WSA(nn.Module):
    def __init__(self, hidden_size, beta_shift, dropout_prob):
        super(WSA, self).__init__()
        print(
            "Initializing WSA with beta_shift:{} hidden_prob:{}".format(
                beta_shift, dropout_prob
            )
        )
        self.W_v = nn.Linear(hp.VISUAL_DIM, hp.TEXT_DIM)
        self.W_a = nn.Linear(hp.ACOUSTIC_DIM, hp.TEXT_DIM)

        self.W_v_ = nn.Linear(hp.VISUAL_DIM, hp.TEXT_DIM)
        self.W_a_ = nn.Linear(hp.ACOUSTIC_DIM, hp.TEXT_DIM)

        self.W_hv = nn.Linear(hp.VISUAL_DIM + hp.TEXT_DIM, hp.TEXT_DIM)
        self.W_ha = nn.Linear(hp.ACOUSTIC_DIM + hp.TEXT_DIM, hp.TEXT_DIM)

        self.sparsemax = Sparsemax()

        self.beta_shift = beta_shift
        self.LayerNorm = nn.LayerNorm(hidden_size)
        self.dropout = nn.Dropout(dropout_prob)

    def forward(self, text, visual, acoustic):
        eps = 1e-6
        new_v = self.W_v(visual)
        new_a = self.W_a(acoustic)

        aligned_v = torch.bmm(self.sparsemax(torch.bmm(text, torch.transpose(new_v, 1, 2))), visual)
        aligned_a = torch.bmm(self.sparsemax(torch.bmm(text, torch.transpose(new_a, 1, 2))), acoustic)

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