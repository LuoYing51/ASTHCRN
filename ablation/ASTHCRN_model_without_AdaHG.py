import torch.nn as nn

from ASTHCRN.ablation.AHGCRU_without_AdaHG import AHGCRU_without_AdaHG


class STBlock(nn.Module):

    def __init__(self, embed_size, num_nodes, d_inner,
                 hyperedge_index,
                 device,
                 HGCNADP_topk,
                 hyperedge_rate,
                 HGCNADP_embed_dims, ):
        super(STBlock, self).__init__()

        self.num_nodes = num_nodes
        self.device = device

        self.AHGCN_GRUcell = AHGCRU_without_AdaHG(
            device=device,
            num_nodes=self.num_nodes,
            in_channels=embed_size,
            hidden_dim=d_inner,
            out_channels=embed_size,
            hyperedge_index=hyperedge_index,
            hyperedge_rate=hyperedge_rate,
            HGCNADP_topk=HGCNADP_topk,
            embed_dims=HGCNADP_embed_dims
        )

    def forward(self, x):  # t参数为兼容旧接口，实际已不需要
        # B, N, T, C = x.shape
        residual_x = x

        final_output = self.AHGCN_GRUcell(x.permute(0, 1, 3, 2)).permute(0, 1, 3, 2)

        final_output = residual_x + final_output
        return final_output


class STBlocks(nn.Module):

    def __init__(
            self,
            embed_size,
            num_layers,
            num_nodes,
            d_inner,
            device,
            hyperedge_index,
            HGCNADP_topk,
            hyperedge_rate,
            HGCNADP_embed_dims,
            dropout,

    ):
        super(STBlocks, self).__init__()
        self.embed_size = embed_size
        self.layers = nn.ModuleList(
            [
                STBlock(
                    embed_size, num_nodes, d_inner,
                    hyperedge_index,
                    device,
                    HGCNADP_topk,
                    hyperedge_rate,
                    HGCNADP_embed_dims,

                )
                for _ in range(num_layers)
            ]
        )

        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        out = self.dropout(x)
        for layer in self.layers:
            out = layer(out)
        return out


class main(nn.Module):
    def __init__(
            self,
            in_channels,
            output_dim,
            embed_size,
            d_inner,
            num_layers,
            T_dim,
            output_T_dim,
            num_nodes,
            hyperedge_index,
            device,
            HGCNADP_topk,
            hyperedge_rate,
            HGCNADP_embed_dims,
            dropout=0.1,
    ):
        super(main, self).__init__()

        self.device = device
        self.embed_size = embed_size

        self.conv1 = nn.Conv2d(in_channels, embed_size, 1)

        self.Transformer_layers = STBlocks(
            embed_size,
            num_layers,
            num_nodes,
            d_inner,
            device,
            hyperedge_index,
            HGCNADP_topk,
            hyperedge_rate,
            HGCNADP_embed_dims,
            dropout,
        )

        self.temporal_conv = nn.Sequential(
            nn.Conv2d(T_dim, output_T_dim, 1),
            nn.ReLU())

        self.dim_conv = nn.Conv2d(embed_size, output_dim, 1)

    def forward(self, x):
        input_Transformer = self.conv1(x.permute(0, 3, 2, 1))

        input_Transformer = input_Transformer.permute(0, 2, 3, 1)
        output_Transformer = self.Transformer_layers(input_Transformer)
        output_Transformer = output_Transformer.permute(0, 2, 1, 3)

        out = self.temporal_conv(output_Transformer)
        out = out.permute(0, 3, 2, 1)
        out = self.dim_conv(out)

        out = out.permute(0, 3, 2, 1)

        return out
