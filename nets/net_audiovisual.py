import torch
import torch.nn as nn
import torch.nn.functional as F


def exp_evidence(y, temp=0.8):
    return torch.exp(torch.div(torch.clamp(y, -10, 10), temp))


def get_p_and_u_from_logit(x):
    alpha = exp_evidence(x) + torch.ones_like(x)
    p = alpha[..., 0] / torch.sum(alpha, dim=-1)
    u = 2 / torch.sum(alpha, dim=-1)
    return p, u


class LabelSmoothingNCELoss(nn.Module):

    def __init__(self, classes, smoothing=0.0, dim=-1):
        super(LabelSmoothingNCELoss, self).__init__()
        self.confidence = 1.0 - smoothing
        self.smoothing = smoothing
        self.cls = classes
        self.dim = dim

    def forward(self, pred, target):
        pred = pred.softmax(dim=self.dim)
        with torch.no_grad():
            true_dist = torch.zeros_like(pred)
            true_dist.fill_(self.smoothing / (self.cls - 1))
            true_dist.scatter_(1, target.data.unsqueeze(1), self.confidence)
        return -torch.mean(torch.log(torch.sum(true_dist * pred, dim=self.dim)))


class Encoder(nn.Module):
    def __init__(self, encoder_layer, num_layers, norm=None, d_model=512, nhead=1, dim_feedforward=512, dropout=0.1):
        super(Encoder, self).__init__()
        self.layers = []
        if encoder_layer == 'HANLayer':
            for i in range(num_layers):
                self.layers.append(HANLayer(d_model=d_model, nhead=nhead,
                                            dim_feedforward=dim_feedforward, dropout=dropout))
        else:
            raise ValueError('wrong encoder layer')
        self.layers = nn.ModuleList(self.layers)
        self.num_layers = num_layers
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.norm = norm

    def forward(self, src_a, src_v, mask=None, src_key_padding_mask=None, with_ca=True):
        output_a = src_a
        output_v = src_v

        for i in range(self.num_layers):
            output_a = self.layers[i](src_a, src_v, src_mask=mask,
                                      src_key_padding_mask=src_key_padding_mask, with_ca=with_ca)
            output_v = self.layers[i](src_v, src_a, src_mask=mask,
                                      src_key_padding_mask=src_key_padding_mask, with_ca=with_ca)
            src_a = output_a
            src_v = output_v

        if self.norm:
            output_a = self.norm1(output_a)
            output_v = self.norm2(output_v)

        return output_a, output_v


class HANLayer(nn.Module):
    def __init__(self, d_model, nhead, dim_feedforward=512, dropout=0.1):
        super(HANLayer, self).__init__()
        self.self_attn = nn.MultiheadAttention(d_model, nhead, dropout=dropout)
        self.cm_attn = nn.MultiheadAttention(d_model, nhead, dropout=dropout)

        # Implementation of Feedforward model
        self.linear1 = nn.Linear(d_model, dim_feedforward)
        self.dropout = nn.Dropout(dropout)
        self.linear2 = nn.Linear(dim_feedforward, d_model)

        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout11 = nn.Dropout(dropout)
        self.dropout12 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)

    def forward(self, src_q, src_v, src_mask=None, src_key_padding_mask=None, with_ca=True):
        """Pass the input through the encoder layer.

        Args:
            src_q: the sequence to the encoder layer (required).
            src_v: the sequence to the encoder layer (required).
            src_mask: the mask for the src sequence (optional).
            src_key_padding_mask: the mask for the src keys per batch (optional).
            with_ca: whether to use audio-visual cross-attention
        Shape:
            see the docs in Transformer class.
        """
        src_q = src_q.permute(1, 0, 2)
        src_v = src_v.permute(1, 0, 2)

        if with_ca:
            src1 = self.cm_attn(src_q, src_v, src_v, attn_mask=src_mask,
                                key_padding_mask=src_key_padding_mask)[0]
            src2 = self.self_attn(src_q, src_q, src_q, attn_mask=src_mask,
                                  key_padding_mask=src_key_padding_mask)[0]

            src_q = src_q + self.dropout11(src1) + self.dropout12(src2)
            src_q = self.norm1(src_q)
        else:
            src2 = self.self_attn(src_q, src_q, src_q, attn_mask=src_mask,
                                  key_padding_mask=src_key_padding_mask)[0]

            src_q = src_q + self.dropout12(src2)
            src_q = self.norm1(src_q)

        src_q = src_q + self.dropout2(self.linear2(self.dropout(F.relu(self.linear1(src_q)))))
        src_q = self.norm2(src_q)
        return src_q.permute(1, 0, 2)


class MMIL_Net(nn.Module):

    def __init__(self, num_layers=1, temperature=0.2, att_dropout=0.1, cls_dropout=0.5):
        super(MMIL_Net, self).__init__()

        self.fc_prob = nn.Linear(512, 25)
        self.fc_prob_beta = nn.Linear(512, 25)
        self.fc_frame_att = nn.Linear(512, 25 * 2)
        self.fc_global = nn.Sequential(
            nn.Linear(512, 512),
            nn.ReLU(),
            nn.Linear(512, 25),
            nn.Sigmoid(),
        )
        self.fc_a = nn.Sequential(
            nn.Linear(128, 512),
            nn.ELU(),
        )
        self.fc_v = nn.Sequential(
            nn.Linear(2048, 512),
            nn.ELU(),
        )
        self.fc_st = nn.Sequential(
            nn.Linear(512, 512),
            nn.ELU(),
        )
        self.fc_fusion = nn.Sequential(
            nn.Linear(1024, 512),
            nn.ELU(),
        )
        self.hat_encoder = Encoder('HANLayer', num_layers, norm=None, d_model=512,
                                   nhead=1, dim_feedforward=512, dropout=att_dropout)

        self.v2a = nn.Sequential(
            nn.Linear(512, 512),
            nn.Dropout(0.2),
            nn.LeakyReLU(0.2),
        )

        self.a2v = nn.Sequential(
            nn.Linear(512, 512),
            nn.Dropout(0.2),
            nn.LeakyReLU(0.2),
        )

        self.classifier1 = nn.Sequential(
            nn.Linear(512, 512),
            nn.LeakyReLU(0.2),
        )

        self.classifier2 = nn.Linear(512, 25)

        self.temp = temperature
        if cls_dropout != 0:
            self.dropout = nn.Dropout(p=cls_dropout)
        else:
            self.dropout = None

    def forward(self, audio, visual, visual_st, with_ca=True):
        b, t, d = visual_st.size()
        x1 = self.fc_a(audio)
        # 2d and 3d visual feature fusion
        vid_s = self.fc_v(visual).permute(0, 2, 1).unsqueeze(-1)
        vid_s = F.avg_pool2d(vid_s, (8, 1)).squeeze(-1).permute(0, 2, 1)
        vid_st = self.fc_st(visual_st)
        x2 = torch.cat((vid_s, vid_st), dim=-1)
        x2 = self.fc_fusion(x2)

        # Mutual Learning
        x1_embed = self.classifier1(x1)
        x2_embed = self.classifier1(x2)
        ori_logit = self.classifier2(x1_embed + x2_embed)
        ori_alpha = exp_evidence(ori_logit.mean(dim=1)).add(1.0)
        global_uct = 25 / torch.sum(ori_alpha, dim=-1)
        ori_prob = nn.Sigmoid()(ori_logit)
        global_prob = ori_prob.mean(dim=1)

        # HAN
        # x1, x2 = self.hat_encoder(x1, x2, with_ca=with_ca)
        if not with_ca:
            x1, x2 = self.hat_encoder(x1, x2, with_ca=False)
        else:
            x1_woca, x2_woca = self.hat_encoder(x1, x2, with_ca=False)
            x1_ca, x2_ca = self.hat_encoder(x1, x2, with_ca=True)
            ratio = 0.8
            x1 = x1_ca * ratio + x1_woca * (1 - ratio)
            x2 = x2_ca * ratio + x2_woca * (1 - ratio)

        # noise contrastive
        # please refer to https://github.com/Yu-Wu/Modaily-Aware-Audio-Visual-Video-Parsing
        xx2_after = F.normalize(x2, p=2, dim=-1)
        xx1_after = F.normalize(x1, p=2, dim=-1)
        sims_after = xx1_after.bmm(xx2_after.permute(0, 2, 1)).squeeze(1) / self.temp
        sims_after = sims_after.reshape(-1, 10)
        mask_after = torch.zeros(b, 10)
        mask_after = mask_after.long()
        for i in range(10):
            mask_after[:, i] = i
        mask_after = mask_after.cuda()
        mask_after = mask_after.reshape(-1)

        # prediction
        if self.dropout is not None:
            x2 = self.dropout(x2)
        x = torch.cat([x1.unsqueeze(-2), x2.unsqueeze(-2)], dim=-2)

        c = 25
        b, t, m, _ = x.shape  # 128, 10, 2, 512

        frame_alpha_logit = self.fc_prob(x)  # 128, 10, 2, 25
        x_a = x[:, :, 0]  # 128, 10, 512
        x_v = x[:, :, 1]  # 128, 10, 512
        x_v2a = x_a + self.v2a(x_a + x_v)
        x_a2v = x_v + self.a2v(x_a + x_v)
        frame_beta_a = self.fc_prob_beta(x_v2a)  # 128, 10, 25
        frame_beta_v = self.fc_prob_beta(x_a2v)
        frame_beta_logit = torch.stack((frame_beta_a, frame_beta_v), dim=2)
        frame_logit = torch.stack((frame_alpha_logit, frame_beta_logit), dim=-1)

        # attentive MMIL pooling
        frame_att_before_softmax = self.fc_frame_att(x).reshape(b, t, m, c, 2)
        frame_att = torch.softmax(frame_att_before_softmax, dim=1)
        temporal_logit = frame_att * frame_logit
        # frame-wise probability
        a_logit = temporal_logit[:, :, 0, :, :].sum(dim=1)
        v_logit = temporal_logit[:, :, 1, :, :].sum(dim=1)

        a_prob, a_uct = get_p_and_u_from_logit(a_logit)
        v_prob, v_uct = get_p_and_u_from_logit(v_logit)
        frame_prob, frame_uct = get_p_and_u_from_logit(frame_logit)

        return global_prob, a_prob, v_prob, frame_prob, sims_after, mask_after, \
               global_uct, a_uct, v_uct, frame_uct
