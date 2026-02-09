import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from base_model import SequenceModel


# =============================================================================
# Temporal Encoder Components
# =============================================================================

class PositionalEncoding(nn.Module):
    """Sinusoidal positional encoding for temporal sequences."""

    def __init__(self, d_model, max_len=100):
        super().__init__()
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        self.register_buffer("pe", pe)

    def forward(self, x):
        return x + self.pe[:x.shape[1], :]


class TemporalAttentionBlock(nn.Module):
    """Multi-head self-attention block with pre-norm and FFN for temporal modeling."""

    def __init__(self, d_model, nhead, dropout):
        super().__init__()
        self.d_model = d_model
        self.nhead = nhead
        self.qtrans = nn.Linear(d_model, d_model, bias=False)
        self.ktrans = nn.Linear(d_model, d_model, bias=False)
        self.vtrans = nn.Linear(d_model, d_model, bias=False)

        self.attn_dropout = []
        if dropout > 0:
            for i in range(nhead):
                self.attn_dropout.append(nn.Dropout(p=dropout))
            self.attn_dropout = nn.ModuleList(self.attn_dropout)

        self.norm1 = nn.LayerNorm(d_model, eps=1e-5)
        self.norm2 = nn.LayerNorm(d_model, eps=1e-5)
        self.ffn = nn.Sequential(
            nn.Linear(d_model, d_model),
            nn.ReLU(),
            nn.Dropout(p=dropout),
            nn.Linear(d_model, d_model),
            nn.Dropout(p=dropout)
        )

    def forward(self, x):
        x = self.norm1(x)
        q = self.qtrans(x)
        k = self.ktrans(x)
        v = self.vtrans(x)

        dim = int(self.d_model / self.nhead)
        att_output = []
        for i in range(self.nhead):
            if i == self.nhead - 1:
                qh = q[:, :, i * dim:]
                kh = k[:, :, i * dim:]
                vh = v[:, :, i * dim:]
            else:
                qh = q[:, :, i * dim:(i + 1) * dim]
                kh = k[:, :, i * dim:(i + 1) * dim]
                vh = v[:, :, i * dim:(i + 1) * dim]
            atten_ave_matrixh = torch.softmax(torch.matmul(qh, kh.transpose(1, 2)), dim=-1)
            if self.attn_dropout:
                atten_ave_matrixh = self.attn_dropout[i](atten_ave_matrixh)
            att_output.append(torch.matmul(atten_ave_matrixh, vh))
        att_output = torch.concat(att_output, dim=-1)

        xt = x + att_output
        xt = self.norm2(xt)
        att_output = xt + self.ffn(xt)

        return att_output


class RMSNorm(nn.Module):
    """Root Mean Square Layer Normalization."""

    def __init__(self, d_model, eps=1e-6):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(d_model))
        self.eps = eps

    def forward(self, x):
        rms = torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps)
        return x * rms * self.weight


# =============================================================================
# Component (1): Disentangled Representation
#   Asymmetric Dual-Stream Encoder — separates sequence-derived stock momentum
#   from state-derived macro context.
# =============================================================================

class AsymmetricDualStreamEncoder(nn.Module):
    """Asymmetric Dual-Stream Encoder for disentangled representation.

    Encodes each market-index stream independently to produce control vectors
    that capture distinct macro regimes.

    Parameters
    ----------
    index_dim : int
        Dimension of each market index feature vector (default: 21).
    control_dim : int
        Output control-vector dimension.
    single_stream : bool
        If True, use a single encoder (one market index only); the output
        is duplicated to maintain the dual-vector interface.
    """

    def __init__(self, index_dim=21, control_dim=64, single_stream=False):
        super().__init__()
        self.single_stream = single_stream
        self.index_dim = index_dim

        if single_stream:
            self.encoder = nn.Sequential(
                nn.Linear(self.index_dim, control_dim),
                nn.GELU(),
                nn.Linear(control_dim, control_dim)
            )
        else:
            self.encoder_primary = nn.Sequential(
                nn.Linear(self.index_dim, control_dim),
                nn.GELU(),
                nn.Linear(control_dim, control_dim)
            )
            self.encoder_secondary = nn.Sequential(
                nn.Linear(self.index_dim, control_dim),
                nn.GELU(),
                nn.Linear(control_dim, control_dim)
            )

    def forward(self, x):
        """Encode macro context into two control vectors.

        Parameters
        ----------
        x : torch.Tensor
            Macro features, shape ``[..., D]``.

        Returns
        -------
        z_primary, z_secondary : torch.Tensor
            Encoded control vectors, each ``[..., control_dim]``.
        """
        if self.single_stream:
            m = x[..., :self.index_dim]
            z = self.encoder(m)
            return z, z
        else:
            if x.shape[-1] < 2 * self.index_dim:
                m_primary = x
                m_secondary = x
            else:
                m_primary = x[..., :self.index_dim]
                m_secondary = x[..., self.index_dim:2 * self.index_dim]
            z_primary = self.encoder_primary(m_primary)
            z_secondary = self.encoder_secondary(m_secondary)
            return z_primary, z_secondary


# =============================================================================
# Component (2): Regime-Dependent Topology
#   Regime-Conditioned Graph Attention with Macro Anchors and Alpha Guard.
#
#   - Regime Gate:   element-wise query modulation conditioned on macro state.
#   - Macro Anchors: virtual key-value nodes for global conditioning.
#   - Alpha Guard:   gated residual connection with RMSNorm.
# =============================================================================

class RegimeConditionedGraphAttention(nn.Module):
    """Regime-Conditioned Graph Attention with Macro Anchors.

    Parameters
    ----------
    d_model : int
        Hidden dimension.
    nhead : int
        Number of attention heads.
    dropout : float
        Dropout rate.
    use_alpha_guard : bool
        Enable Alpha Guard (gated residual + RMSNorm).
    use_macro_anchor : bool
        Inject Macro Anchor virtual nodes into key-value sets.
    use_regime_gate : bool
        Apply Regime Gate element-wise query modulation.
    use_zero_init : bool
        Zero-initialize the output projection for residual stability.
    """

    def __init__(self, d_model, nhead=4, dropout=0.1,
                 use_alpha_guard=True, use_macro_anchor=True,
                 use_regime_gate=True, use_zero_init=True):
        super().__init__()
        self.d_model = d_model
        self.nhead = nhead
        self.head_dim = d_model // nhead
        self.use_alpha_guard = use_alpha_guard
        self.use_macro_anchor = use_macro_anchor
        self.use_regime_gate = use_regime_gate

        # QKV projections
        self.to_q = nn.Linear(d_model, d_model, bias=False)
        self.to_k = nn.Linear(d_model, d_model, bias=False)
        self.to_v = nn.Linear(d_model, d_model, bias=False)

        # Regime Gate: g_R = sigma(W_R [z_primary || z_secondary])
        if self.use_macro_anchor and self.use_regime_gate:
            self.regime_gate = nn.Sequential(
                nn.Linear(d_model * 2, d_model),
                nn.Sigmoid()
            )

        # Alpha Guard: gated residual with RMSNorm
        if self.use_alpha_guard:
            self.alpha_guard_gate = nn.Linear(d_model * 2, d_model)
            self.alpha_guard_norm = RMSNorm(d_model)

        # Output projection (optionally zero-initialized)
        self.out_proj = nn.Linear(d_model, d_model)
        if use_zero_init:
            nn.init.constant_(self.out_proj.weight, 0)
            nn.init.constant_(self.out_proj.bias, 0)

        self.dropout = nn.Dropout(dropout)
        self.scale = 1.0 / math.sqrt(self.head_dim)

    def forward(self, stock_feats, z_primary, z_secondary, keep_ratio=1.0):
        """
        Parameters
        ----------
        stock_feats : torch.Tensor
            Stock representations ``[N, D]``.
        z_primary, z_secondary : torch.Tensor
            Macro control vectors from the Dual-Stream Encoder, each ``[1, D]``.
        keep_ratio : float
            Fraction of top-k neighbours to retain (sparsity schedule).

        Returns
        -------
        output : torch.Tensor
            Updated stock representations ``[N, D]``.
        raw_scores : torch.Tensor
            Mean-over-heads attention logits (before softmax), used by the
            Structural Hinge Loss during training.
        gate_mean : float
            Mean Alpha Guard activation (0.0 if disabled).
        attn_weights : torch.Tensor
            Head-averaged attention weights ``[N, M]``.
        """
        N, D = stock_feats.shape

        # --- 1. Query construction with optional Regime Gate modulation ---
        if self.use_macro_anchor:
            # Macro Anchors injected as virtual key-value nodes
            nodes = torch.cat([z_primary, z_secondary, stock_feats], dim=0)

            if self.use_regime_gate:
                z_concat = torch.cat([z_primary, z_secondary], dim=-1)
                regime_modulation = self.regime_gate(z_concat)
                q_modulated = self.to_q(stock_feats) * regime_modulation
            else:
                q_modulated = self.to_q(stock_feats)

            k_input = nodes
            v_input = nodes
            num_anchor_nodes = 2
        else:
            # No macro anchors: stock-only attention
            q_modulated = self.to_q(stock_feats)
            k_input = stock_feats
            v_input = stock_feats
            num_anchor_nodes = 0

        # --- 2. Multi-head attention ---
        q = q_modulated.view(N, self.nhead, self.head_dim)
        k = self.to_k(k_input).view(N + num_anchor_nodes, self.nhead, self.head_dim)
        v = self.to_v(v_input).view(N + num_anchor_nodes, self.nhead, self.head_dim)

        scores = torch.einsum('nhd,mhd->nhm', q, k) * self.scale
        raw_scores = scores.mean(dim=1)

        # --- 3. Top-k sparsification ---
        if keep_ratio < 1.0:
            if self.use_macro_anchor:
                anchor_scores = scores[..., :2]
                stock_scores = scores[..., 2:]
            else:
                anchor_scores = None
                stock_scores = scores

            k_keep = max(5, int(N * keep_ratio))
            topk_val, _ = torch.topk(stock_scores, k_keep, dim=-1)
            threshold = topk_val[..., -1].unsqueeze(-1)
            mask = stock_scores >= threshold
            stock_scores_masked = stock_scores.masked_fill(~mask, -1e9)

            if self.use_macro_anchor:
                scores = torch.cat([anchor_scores, stock_scores_masked], dim=-1)
            else:
                scores = stock_scores_masked

        attn_weights = F.softmax(scores, dim=-1)
        attn_weights = self.dropout(attn_weights)

        output = torch.einsum('nhm,mhd->nhd', attn_weights, v)
        output = output.contiguous().view(N, D)
        output = self.out_proj(output)

        # --- 4. Alpha Guard: G = sigma(W_G [h || o]), out = RMSNorm(h + G * o) ---
        if self.use_alpha_guard:
            concat_feat = torch.cat([stock_feats, output], dim=-1)
            gate = torch.sigmoid(self.alpha_guard_gate(concat_feat))
            gated_residual = stock_feats + gate * output
            final_out = self.alpha_guard_norm(gated_residual)
            gate_mean = gate.mean().item()
        else:
            final_out = output
            gate_mean = 0.0

        attn_weights_mean = attn_weights.mean(dim=1)  # [N, M]
        return final_out, raw_scores, gate_mean, attn_weights_mean


# =============================================================================
# Full HMSA Network
# =============================================================================

class HMSANetwork(nn.Module):
    """Hierarchical Macro-conditioned Stock Attention Network.

    Integrates three components from the paper:

    1. **Disentangled Representation** via :class:`AsymmetricDualStreamEncoder`.
    2. **Regime-Dependent Topology** via :class:`RegimeConditionedGraphAttention`
       (Regime Gate + Macro Anchors + Alpha Guard).
    3. **Structural Regularization** via the margin-based Structural Hinge Loss
       applied externally during training (see ``base_model.SequenceModel``).

    Parameters
    ----------
    d_feat : int
        Per-timestep input feature dimension.
    d_model : int
        Hidden dimension.
    dropout : float
        Dropout rate.
    universe : str or None
        Market universe identifier (``'csi300'``, ``'csi800'``, or ``'sp500'``).
        Determines macro feature extraction:
        - ``'csi300'``: uses z300 (first 21 dims) or z300+z500 (42 dims)
        - ``'csi800'``: uses z300+z500 (42 dims)
        - ``'sp500'``: uses single market index (21 dims)
    use_single_index : bool
        For CSI300 only: if True, use only z300 (21 dims); if False, use z300+z500 (42 dims).
    """

    def __init__(self, d_feat, d_model, dropout=0.1, universe=None, use_single_index=False):
        super().__init__()

        self.universe = universe
        self.use_single_index = use_single_index

        # Determine macro feature dimensions based on universe
        if universe == 'csi300':
            if use_single_index:
                # CSI300: single index (z300 only, 21 dims)
                self.macro_dim = 21
                self.single_stream = True
            else:
                # CSI300: dual index (z300+z500, 42 dims)
                self.macro_dim = 42
                self.single_stream = False
        elif universe == 'csi800':
            # CSI800: always dual index (z300+z500, 42 dims)
            self.macro_dim = 42
            self.single_stream = False
        elif universe == 'sp500':
            # SP500: single index (21 dims)
            self.macro_dim = 21
            self.single_stream = True
        else:
            # Default: assume dual index (42 dims)
            self.macro_dim = 42
            self.single_stream = False

        # Stock encoder: project per-timestep features to d_model
        self.stock_encoder = nn.Sequential(
            nn.Linear(d_feat, d_model),
            nn.GELU(),
            nn.Dropout(dropout)
        )

        # Temporal encoder: positional encoding + transformer block
        self.temporal_encoder = nn.Sequential(
            PositionalEncoding(d_model),
            TemporalAttentionBlock(d_model=d_model, nhead=4, dropout=dropout)
        )

        # Asymmetric Dual-Stream Encoder (macro context stream)
        self.dual_stream_encoder = AsymmetricDualStreamEncoder(
            index_dim=21,
            control_dim=d_model,
            single_stream=self.single_stream
        )

        # Regime-Conditioned Graph Attention (full model: all components enabled)
        self.graph_attention = RegimeConditionedGraphAttention(
            d_model, nhead=4, dropout=dropout,
            use_alpha_guard=True,
            use_macro_anchor=True,
            use_regime_gate=True,
            use_zero_init=True
        )

        # Prediction head
        self.prediction_head = nn.Sequential(
            nn.Linear(d_model, 64),
            nn.GELU(),
            nn.Linear(64, 1)
        )
        self.current_ratio = 1.0

    def set_sparsity_ratio(self, epoch, total_epochs):
        """Linearly decay keep-ratio from 1.0 to target during [20%, 80%] of training."""
        start_decay = int(total_epochs * 0.2)
        end_decay = int(total_epochs * 0.8)
        target_ratio = 0.35
        if epoch < start_decay:
            self.current_ratio = 1.0
        elif epoch > end_decay:
            self.current_ratio = target_ratio
        else:
            progress = (epoch - start_decay) / (end_decay - start_decay)
            self.current_ratio = 1.0 - progress * (1.0 - target_ratio)
        return self.current_ratio

    def forward(self, x):
        # Extract stock features and macro features
        # Input shape: [N, T, d_feat + 63] where 63 = 3 x 21 (full macro features)
        stock_feats = x[..., :-63]
        market_full = x[..., -63:]

        # Extract macro features based on universe configuration
        if self.universe == 'csi300':
            if self.use_single_index:
                # CSI300: single index (z300 only, first 21 dims)
                market_info = market_full[..., :21]
            else:
                # CSI300: dual index (z300+z500, first 42 dims)
                market_info = market_full[..., :42]
        elif self.universe == 'csi800':
            # CSI800: dual index (z300+z500, first 42 dims)
            market_info = market_full[..., :42]
        elif self.universe == 'sp500':
            # SP500: single index (first 21 dims)
            market_info = market_full[..., :21]
        else:
            # Default: use first 42 dims (z300+z500)
            market_info = market_full[..., :42]

        # Stock momentum stream (sequence-derived)
        h = self.stock_encoder(stock_feats)
        h = self.temporal_encoder(h)
        h_last = h[:, -1, :]

        # Macro context stream (state-derived)
        market_last = market_info[0, -1, :]
        z_primary, z_secondary = self.dual_stream_encoder(market_last.unsqueeze(0))

        # Regime-Conditioned Graph Attention
        h_out, raw_scores, gate_mean, attn_weights = self.graph_attention(
            h_last, z_primary, z_secondary, keep_ratio=self.current_ratio
        )

        pred = self.prediction_head(h_out).squeeze(-1)

        debug_stats = {
            'gate_mean': gate_mean
        }

        return pred, raw_scores, debug_stats, attn_weights


# =============================================================================
# Model Wrapper (Training & Evaluation Interface)
# =============================================================================

class HMSAModel(SequenceModel):
    """HMSA model wrapper providing training loop, evaluation, and loss computation.

    Parameters
    ----------
    d_feat : int
        Input feature dimension.
    d_model : int
        Hidden dimension.
    use_hinge_loss : bool
        Use Structural Hinge Loss for graph regularization (Component 3).
    universe : str
        Market universe identifier (``'csi300'``, ``'csi800'``, or ``'sp500'``).
    use_single_index : bool
        For CSI300 only: if True, use only z300 (21 dims); if False, use z300+z500 (42 dims).
        Ignored for other universes.
    """

    def __init__(self, d_feat, d_model, use_hinge_loss=True, universe=None, use_single_index=False, **kwargs):
        super().__init__(use_hinge_loss=use_hinge_loss, **kwargs)
        self.model = HMSANetwork(
            d_feat=d_feat,
            d_model=d_model,
            universe=universe,
            use_single_index=use_single_index
        )
        self.use_hinge_loss = use_hinge_loss
        self.init_model()

    def forward(self, x):
        pred, _, debug_stats, _ = self.model(x)
        return pred
