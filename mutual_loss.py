import torch
import torch.nn as nn


class MutualLearningLoss(nn.Module):

    def __init__(self, eta=1.0, lamb=1.0):
        super().__init__()

        self.eta = eta
        self.lamb = lamb

    def forward(self, target, global_prob, a_prob, v_prob, a_uct, v_uct, global_uct, batch_idx):
        global_uct = torch.clamp(global_uct, min=0.9)

        av_prob = torch.stack((a_prob, v_prob), dim=-1)
        av_uct = torch.stack((a_uct, v_uct), dim=-1)

        max_prob, max_index = torch.max(av_prob, dim=-1)
        max_uct = torch.gather(av_uct, dim=-1, index=max_index.unsqueeze(-1)).squeeze()

        mean_prob = torch.mean(av_prob, dim=-1)
        mean_uct = torch.mean(av_uct, dim=-1)

        mse_loss = nn.MSELoss(reduction='none')

        is_odd = batch_idx % 2

        pos_loss = is_odd * global_uct.unsqueeze(-1) * \
                   ((1 - max_uct) * mse_loss(global_prob, max_prob.detach()) * target) + \
                   (1 - is_odd) * (1 - global_uct).unsqueeze(-1) * \
                   ((1 - max_uct) * mse_loss(global_prob.detach(), max_prob) * target)

        neg_loss = is_odd * global_uct.unsqueeze(-1) * \
                   ((1 - mean_uct) * mse_loss(global_prob, mean_prob.detach()) * (1 - target)) + \
                   (1 - is_odd) * (1 - global_uct).unsqueeze(-1) * \
                   ((1 - mean_uct) * mse_loss(global_prob.detach(), mean_prob) * (1 - target))

        total_loss = pos_loss + self.lamb * neg_loss
        total_loss = total_loss.mean()

        return total_loss
