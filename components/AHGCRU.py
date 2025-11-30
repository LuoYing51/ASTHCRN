import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from torch_geometric.nn import HypergraphConv

class HyperGCNBlock(nn.Module):
    def __init__(self, in_c, out_c, hidden_c=None):
        super(HyperGCNBlock, self).__init__()

        if hidden_c is None:
            hidden_c = out_c

        self.conv1 = HypergraphConv(in_c, hidden_c)
        self.conv2 = HypergraphConv(hidden_c, out_c)
        self.layernorm1 = nn.LayerNorm(hidden_c)
        self.layernorm2 = nn.LayerNorm(out_c)
        self.dropout = nn.Dropout(0.1)

        if in_c != out_c:
            self.residual_proj = nn.Linear(in_c, out_c)
        else:
            self.residual_proj = nn.Identity()

    def forward(self, x, hyperedge_index, hyperedge_weight):

        residual = self.residual_proj(x)
        x = self.conv1(x, hyperedge_index, hyperedge_weight)
        x=self.dropout(self.layernorm1(F.relu(residual +x)))
        return x

class HGCRU(nn.Module):
    def __init__(self, in_feature, hidden_dim):
        super(HGCRU, self).__init__()
        self.hidden_dim = hidden_dim

        self.gate_block = HyperGCNBlock(in_c=in_feature + hidden_dim, out_c=2 * hidden_dim)
        self.candidate_block = HyperGCNBlock(in_c=in_feature + hidden_dim, out_c=hidden_dim)

    def forward(self, x, state, hyperedge_index, hyperedge_weight):

        input_and_state = torch.cat((x, state), dim=1)

        z_r = self.gate_block(input_and_state, hyperedge_index, hyperedge_weight)
        z_r = torch.sigmoid(z_r)
        z, r = torch.split(z_r, self.hidden_dim, dim=1)

        candidate_input = torch.cat((x, r * state), dim=1)

        hc = self.candidate_block(candidate_input, hyperedge_index, hyperedge_weight)
        hc = torch.tanh(hc)

        h = (1 - z) * state + z * hc
        return h

# adaptive hypergraph convolutional recurrent unit
class AHGCRU(nn.Module):
    def __init__(self, device, num_nodes, in_channels, hidden_dim,out_channels,
                 hyperedge_rate, HGCNADP_topk, embed_dims):
        super(AHGCRU, self).__init__()
        self.device = device
        self.num_nodes = num_nodes
        self.out_channels = out_channels
        self.hidden_dim = hidden_dim

        self.TempHG = AdaHCM(
            DEVICE=device,
            num_of_vertices=num_nodes,
            HGCNADP_topk=HGCNADP_topk,
            hyperedge_rate=hyperedge_rate,
            embed_dims=embed_dims
        )

        self.HGCRU = HGCRU(in_feature=in_channels, hidden_dim=self.hidden_dim)

        self.conv = nn.Conv2d(self.hidden_dim, self.out_channels, 1)

    def forward(self, x):
        """
        :param x:  (B, N, F_in, T)
        """
        B, N, F_in, T = x.shape

        state = torch.zeros(B * N, self.hidden_dim).to(self.device)

        outputs = []

        for t in range(T):
            current_x = x[:, :, :, t].reshape(B * N, F_in)
            HE, HEW = self.TempHG(current_x.reshape(B ,N, F_in,1))
            state = self.HGCRU(current_x, state, HE, HEW)
            outputs.append(state.reshape(B, N, self.hidden_dim))

        final_output_ = torch.stack(outputs, dim=-1)
        final_output=self.conv(final_output_.permute(0,2,1,3)).permute(0,2,1,3)

        return final_output

#adaptive hypergraph construction module
class AdaHCM(nn.Module):
    def __init__(self, DEVICE, num_of_vertices,HGCNADP_topk,hyperedge_rate,embed_dims):
        super(AdaHCM, self).__init__()
        self.device = DEVICE
        self.HGCNADP_topk=HGCNADP_topk
        self.nodevec = nn.Parameter(torch.randn(num_of_vertices, embed_dims), requires_grad=True).to(DEVICE)
        self.edgevec = nn.Parameter(torch.randn(math.ceil(hyperedge_rate * num_of_vertices), embed_dims), requires_grad=True).to(DEVICE)

    def forward(self, x):
        B, N, _, _ = x.shape
        DE = torch.tanh(2 * self.nodevec)
        EE = torch.tanh(2 * self.edgevec).transpose(1, 0)
        adj = F.relu(torch.tanh(2 * torch.matmul(DE, EE)))

        adj = adj.repeat(B, 1, 1)
        HE = []
        HEW = []
        B, N, M = adj.shape
        for i in range(B):

            edge_index = adj[i, :, :].nonzero(as_tuple=False).t()

            edge_index[0] += i * N
            edge_index[1] += i * M
            HE.append(edge_index)

        HE = torch.cat(HE, dim=1)
        for i in range(B):
            edge_weight = adj[i, :, :].view(-1)
            HEW.append(edge_weight)
        HEW = torch.cat(HEW, dim=0)
        HEW = HEW[torch.nonzero(HEW)].squeeze(-1)
        return HE, HEW
