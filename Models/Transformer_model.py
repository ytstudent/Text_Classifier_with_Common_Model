import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import copy
from data_processing import DataProcess


class Config(object):
    """配置参数"""

    def __init__(self):
        self.model_name = 'Transformer'
        self.pad_size = 100  # 每句话处理成的长度(短填长切)
        processing = DataProcess(self.pad_size)
        self.inputs_training, self.target_training, self.inputs_testing, self.target_testing = processing.get_data()

        self.embedding_pretrained = None
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')  # 设备

        self.dropout = 0.5  # 随机失活
        self.require_improvement = 2000  # 若超过1000batch效果还没提升，则提前结束训练
        self.num_classes = 4  # 类别数

        self.n_vocab = len(processing.dictionary)  # 词表大小，在运行时赋值
        self.num_epochs = 5  # epoch数
        self.batch_size = 64  # mini-batch大小
        self.pad_size = 100  # 每句话处理成的长度(短填长切)
        self.learning_rate = 1e-4  # 学习率
        self.d_model = 256
        self.d_k = 64    # d_k = d_v = d_model/ n_head
        self.d_v = 64
        self.embed = self.embedding_pretrained.size(1) \
            if self.embedding_pretrained is not None else self.d_model  # 字向量维度

        self.hidden = 256
        self.last_hidden = 128
        self.n_head = 4
        self.num_encoder = 2


'''Attention Is All You Need'''


class Transformer(nn.Module):
    def __init__(self, config):
        super(Transformer, self).__init__()
        if config.embedding_pretrained is not None:
            self.embedding = nn.Embedding.from_pretrained(config.embedding_pretrained, freeze=False)
        else:
            self.embedding = nn.Embedding(config.n_vocab, config.embed, padding_idx=config.n_vocab - 1)

        self.postion_embedding = Positional_Encoding(config.embed, config.pad_size, config.dropout, config.device)
        self.encoder = Encoder(config.d_model, config.n_head, config.d_k, config.d_v, config.hidden, config.dropout)
        self.encoders = nn.ModuleList([
            copy.deepcopy(self.encoder)
            # Encoder(config.d_model, config.n_head, config.hidden, config.dropout)

            for _ in range(config.num_encoder)])

        self.decoder = Decoder(config.d_model, config.n_head, config.d_k, config.d_v, config.hidden, config.dropout)
        self.decoders = nn.ModuleList([
            copy.deepcopy(self.decoder)
            # Encoder(config.d_model, config.n_head, config.hidden, config.dropout)

            for _ in range(config.num_encoder)])

        self.fc1 = nn.Linear(config.pad_size * config.d_model, config.num_classes)
        # self.fc2 = nn.Linear(config.last_hidden, config.num_classes)
        # self.fc1 = nn.Linear(config.d_model, config.num_classes)

    def forward(self, x):
        out = self.embedding(x.long())
        out = self.postion_embedding(out)
        for encoder in self.encoders:
            out = encoder(out)
        out = out.view(out.size(0), -1)
        # out = torch.mean(out, 1)
        out = self.fc1(out)
        return out


class Encoder(nn.Module):
    def __init__(self, d_model, n_head, d_k, d_v, hidden, dropout=0.1):
        super(Encoder, self).__init__()
        self.attention = Multi_Head_Attention(d_model, n_head, d_k, d_v, dropout=dropout)
        self.feed_forward = Position_wise_Feed_Forward(d_model, hidden, dropout=dropout)

    def forward(self, x):
        out, _ = self.attention(x, x, x, mask=None)  # [65, 100, 300]
        out = self.feed_forward(out)  # [65, 100, 300]
        return out


class Decoder(nn.Module):
    ''' Compose with three layers '''

    def __init__(self, d_model, hidden, n_head, d_k, d_v, dropout=0.1):
        super(Decoder, self).__init__()
        self.slf_attn = Multi_Head_Attention(d_model, n_head, d_k, d_v, dropout=dropout)
        self.enc_attn = Multi_Head_Attention(d_model, n_head, d_k, d_v, dropout=dropout)
        self.pos_ffn = Position_wise_Feed_Forward(d_model, hidden, dropout=dropout)

    def forward(self, dec_input, enc_output, non_pad_mask=None, slf_attn_mask=None, dec_enc_attn_mask=None):
        dec_output, dec_slf_attn = self.slf_attn(
            dec_input, dec_input, dec_input, mask=slf_attn_mask)
        dec_output *= non_pad_mask

        dec_output, dec_enc_attn = self.enc_attn(
            dec_output, enc_output, enc_output, mask=dec_enc_attn_mask)
        dec_output *= non_pad_mask

        dec_output = self.pos_ffn(dec_output)
        dec_output *= non_pad_mask

        return dec_output, dec_slf_attn, dec_enc_attn


class Scaled_Dot_Product_Attention(nn.Module):
    """
    Scaled Dot-Product Attention
    """
    """
    Args:
        Q: [batch_size, len_Q, dim_Q]
        K: [batch_size, len_K, dim_K]
        V: [batch_size, len_V, dim_V]
        scale: 缩放因子 论文为根号dim_K
    Return:
        self-attention后的张量，以及attention张量
    """

    def __init__(self, scale, attention_dropout=0.1):
        super(Scaled_Dot_Product_Attention, self).__init__()
        self.scale = scale
        self.dropout = nn.Dropout(attention_dropout)
        self.softmax = nn.Softmax(dim=2)

    def forward(self, q, k, v, mask=None):
        attention = torch.bmm(q, k.transpose(1, 2))
        attention = attention / self.scale

        # if mask is not None:
        #     attention = attention.masked_fill(mask, -np.inf)

        attention = self.softmax(attention)
        attention = self.dropout(attention)
        output = torch.bmm(attention, v)
        return output, attention


class Multi_Head_Attention(nn.Module):
    def __init__(self, d_model, n_head, d_k, d_v, dropout=0.1):
        super(Multi_Head_Attention, self).__init__()
        self.d_k = d_k
        self.d_v = d_v
        self.n_head = n_head

        self.w_qs = nn.Linear(d_model, n_head * self.d_k)
        self.w_ks = nn.Linear(d_model, n_head * self.d_k)
        self.w_vs = nn.Linear(d_model, n_head * self.d_v)

        nn.init.normal_(self.w_qs.weight, mean=0, std=np.sqrt(2.0 / (d_model + d_k)))
        nn.init.normal_(self.w_ks.weight, mean=0, std=np.sqrt(2.0 / (d_model + d_k)))
        nn.init.normal_(self.w_vs.weight, mean=0, std=np.sqrt(2.0 / (d_model + d_v)))

        self.attention = Scaled_Dot_Product_Attention(scale=np.power(d_k, 0.5))
        self.layer_norm = nn.LayerNorm(d_model)

        self.fc = nn.Linear(n_head * self.d_v, d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, q, k, v, mask=None):
        d_k, d_v, n_head = self.d_k, self.d_v, self.n_head
        sz_b, len_q, _ = q.size()
        sz_b, len_k, _ = k.size()
        sz_b, len_v, _ = v.size()

        residual = q

        q = self.w_qs(q).view(sz_b, len_q, n_head, d_k)  # [batch_size, sent_len, n_head, v.size] eg:[65, 100, 5, 64]
        k = self.w_ks(k).view(sz_b, len_k, n_head, d_k)
        v = self.w_vs(v).view(sz_b, len_v, n_head, d_v)

        q = q.permute(2, 0, 1, 3).contiguous().view(-1, len_q, d_k)  # (n*b) x lq x dk     [325, 100, 64]
        k = k.permute(2, 0, 1, 3).contiguous().view(-1, len_k, d_k)  # (n*b) x lk x dk
        v = v.permute(2, 0, 1, 3).contiguous().view(-1, len_v, d_v)  # (n*b) x lv x dv
        # mask = mask.repeat(n_head, 1, 1)  # (n*b) x .. x ..
        output, attention = self.attention(q, k, v, mask=None)

        output = output.view(n_head, sz_b, len_q, d_v)  # [5, 65, 100, 64]

        output = output.permute(1, 2, 0, 3).contiguous().view(sz_b, len_q, -1)  # b x lq x (n*dv)  [65, 100, 320]

        output = self.dropout(self.fc(output))
        output = self.layer_norm(output + residual)  # [65, 100, 300]

        return output, attention


class Positional_Encoding(nn.Module):
    def __init__(self, embed, pad_size, dropout, device):
        super(Positional_Encoding, self).__init__()
        self.device = device
        self.pe = torch.tensor(
            [[pos / (10000.0 ** (i // 2 * 2.0 / embed)) for i in range(embed)] for pos in range(pad_size)])
        self.pe[:, 0::2] = np.sin(self.pe[:, 0::2])
        self.pe[:, 1::2] = np.cos(self.pe[:, 1::2])
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):  # x.size:  [65, 100, 300]
        out = x + nn.Parameter(self.pe, requires_grad=False).to(self.device)
        out = self.dropout(out)  # [65, 100, 300]
        return out


class Position_wise_Feed_Forward(nn.Module):
    def __init__(self, d_model, hidden, dropout=0.1):
        super(Position_wise_Feed_Forward, self).__init__()
        self.fc1 = nn.Linear(d_model, hidden)
        self.fc2 = nn.Linear(hidden, d_model)
        self.dropout = nn.Dropout(dropout)
        self.layer_norm = nn.LayerNorm(d_model)

    def forward(self, x):
        out = self.fc1(x)
        out = F.relu(out)
        out = self.fc2(out)
        out = self.dropout(out)
        out = out + x  # 残差连接
        out = self.layer_norm(out)
        return out
