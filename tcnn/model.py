"""The shared block of the TCNN ("Pair-wise shared network weights and classification", Fig. 1).

    VIS pool5 features --> Flatten - Dense(4096) FC6 - Dropout - Dense(4096) FC7 - Dropout --+
                                    (same weights for both branches)                        +--> L1 layer --> Softmax(2)
    NIR pool5 features --> Flatten - Dense(4096) FC6 - Dropout - Dense(4096) FC7 - Dropout --+

* The two branches are one set of weights applied to the VIS and the NIR features in
  turn ("passed through the shared block sequentially, where the weights are learned
  simultaneously", Sec. II-B).
* L1 layer, Eq. (3)-(4): element-wise absolute difference d_m = |f_v,m - f_n,m|, m = 1..4096.
* A dense layer with 2 outputs and softmax then predicts 1 (genuine pair, same class)
  or 0 (imposter pair).
* Dropout after each dense layer is optional: the paper reports results without dropout
  and with p = 0.25 (Table I).
"""
from __future__ import annotations

import torch
from torch import nn

FEATURE_DIM = 512 * 7 * 7
HIDDEN = 4096


class TCNN(nn.Module):
    def __init__(self, in_dim: int = FEATURE_DIM, hidden: int = HIDDEN, dropout: float = 0.25):
        super().__init__()
        self.branch = nn.Sequential(  # shared by the VIS and the NIR branch
            nn.Flatten(),
            nn.Linear(in_dim, hidden), nn.ReLU(inplace=True), nn.Dropout(dropout),   # FC6
            nn.Linear(hidden, hidden), nn.ReLU(inplace=True), nn.Dropout(dropout),   # FC7
        )
        self.classifier = nn.Linear(hidden, 2)  # softmax over {imposter = 0, genuine = 1}
        for m in self.modules():
            if isinstance(m, nn.Linear):  # Glorot-uniform weights, zero bias (Keras Dense default)
                nn.init.xavier_uniform_(m.weight)
                nn.init.zeros_(m.bias)

    def embed(self, feats: torch.Tensor) -> torch.Tensor:
        """f_shared = conv_shared(F, W_shared): output of FC7 for a batch of pool5 features."""
        return self.branch(feats)

    def head(self, f_vis: torch.Tensor, f_nir: torch.Tensor) -> torch.Tensor:
        """L1 layer (|f_vis - f_nir|, Eq. 3-4) followed by the 2-way classifier; returns logits."""
        return self.classifier(torch.abs(f_vis - f_nir))

    def forward(self, vis_feats: torch.Tensor, nir_feats: torch.Tensor) -> torch.Tensor:
        return self.head(self.embed(vis_feats), self.embed(nir_feats))
