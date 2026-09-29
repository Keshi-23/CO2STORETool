"""
DeepONet architectures with uncertainty-aware output heads.

Architectures
--------------
BaselineDeepONet
    Unconstrained four-output DeepONet.
    Closure is encouraged through an optional closure penalty.

ClosureDeepONet
    Composition-constrained DeepONet.
    Softmax guarantees:

        STTI + RTI + STI + MTI = 1

    exactly for every prediction.

Both architectures independently control the depth of the branch and
trunk networks through:

    n_branch_layers
    n_trunk_layers

Uncertainty heads
-----------------
BaselineDeepONet
    point      -> sigmoid outputs
    hetero     -> (mu, log variance)
    bayes_last -> Bayesian final projection

ClosureDeepONet
    point       -> softmax composition
    logitnormal -> composition mean + ALR uncertainty
    bayes_last  -> Bayesian trunk perturbation
"""

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


# =============================================================================
# FEATURE ATTENTION
# =============================================================================

class FeatureAttention(nn.Module):
    def __init__(self, d):
        super().__init__()

        self.attn = nn.Sequential(
            nn.Linear(d, d),
            nn.GroupNorm(1, d),
            nn.SiLU(),
            nn.Linear(d, d),
            nn.Sigmoid()
        )

    def forward(self, x):
        return x * self.attn(x)


# =============================================================================
# CROSS ATTENTION FUSION
# =============================================================================

class CrossAttentionFusion(nn.Module):
    def __init__(self, dim, heads=4):
        super().__init__()

        if dim % heads != 0:
            raise ValueError(
                f"dim={dim} must be divisible by heads={heads}"
            )

        self.heads = heads
        self.head_dim = dim // heads
        self.scale = self.head_dim ** 0.5

        self.query = nn.Linear(dim, dim)
        self.key = nn.Linear(dim, dim)
        self.value = nn.Linear(dim, dim)

        self.out = nn.Linear(dim, dim)

    def forward(self, b, t):
        """
        Parameters
        ----------
        b : Tensor
            Branch representation, shape (B, dim)

        t : Tensor
            Trunk representation, shape (B, dim)
        """

        B = b.size(0)

        q = self.query(t).view(
            B, self.heads, self.head_dim
        )

        k = self.key(b).view(
            B, self.heads, self.head_dim
        )

        v = self.value(b).view(
            B, self.heads, self.head_dim
        )

        scores = (
            torch.sum(q * k, dim=-1, keepdim=True)
            / self.scale
        )

        attention = torch.softmax(
            scores,
            dim=1
        )

        context = torch.sum(
            attention * v,
            dim=1
        )

        context = context.unsqueeze(1).expand(
            -1,
            self.heads,
            -1
        ).reshape(B, -1)

        return self.out(context) + t


# =============================================================================
# RANDOM FOURIER FEATURES
# =============================================================================

class FourierTime(nn.Module):
    """
    Random Fourier features for time-dependent trunk inputs.

    sigma controls the temporal bandwidth.
    """

    def __init__(
        self,
        d_in,
        n_freq=32,
        sigma=0.3
    ):
        super().__init__()

        self.register_buffer(
            "B",
            torch.randn(d_in, n_freq) * sigma
        )

    def forward(self, t):

        p = 2.0 * np.pi * t @ self.B

        return torch.cat(
            [
                t,
                torch.sin(p),
                torch.cos(p)
            ],
            dim=-1
        )


# =============================================================================
# BAYESIAN LINEAR
# =============================================================================

class BayesianLinear(nn.Module):
    """
    Mean-field variational Bayesian linear layer.

    Used as the Bayesian last layer for epistemic uncertainty.
    """

    def __init__(
        self,
        d_in,
        d_out,
        prior_sigma=0.1,
        zero_init=False,
        rho0=-6.0
    ):
        super().__init__()

        if zero_init:
            w0 = torch.zeros(d_out, d_in)
        else:
            w0 = torch.randn(
                d_out,
                d_in
            ) / np.sqrt(d_in)

        self.w_mu = nn.Parameter(w0)

        self.w_rho = nn.Parameter(
            torch.full(
                (d_out, d_in),
                rho0
            )
        )

        self.b_mu = nn.Parameter(
            torch.zeros(d_out)
        )

        self.b_rho = nn.Parameter(
            torch.full(
                (d_out,),
                rho0
            )
        )

        self.prior_sigma = prior_sigma

    def forward(self, x, sample=True):

        w_sig = F.softplus(self.w_rho)
        b_sig = F.softplus(self.b_rho)

        mu = F.linear(
            x,
            self.w_mu,
            self.b_mu
        )

        if not sample:
            return mu

        var = F.linear(
            x ** 2,
            w_sig ** 2,
            b_sig ** 2
        )

        return (
            mu
            + torch.sqrt(var + 1e-12)
            * torch.randn_like(mu)
        )

    def kl(self):

        out = torch.zeros(
            (),
            device=self.w_mu.device
        )

        for mu, rho in (
            (self.w_mu, self.w_rho),
            (self.b_mu, self.b_rho)
        ):

            sig = F.softplus(rho)

            out = out + (
                torch.log(
                    self.prior_sigma / sig
                )
                + (
                    sig ** 2
                    + mu ** 2
                ) / (
                    2.0 * self.prior_sigma ** 2
                )
                - 0.5
            ).sum()

        return out


# =============================================================================
# MLP BUILDER
# =============================================================================

def _mlp(
    input_dim,
    output_dim,
    width,
    n_layers,
    p_drop=0.0,
    spectral=False,
    groups=4
):
    """
    Build an MLP with independently configurable depth.

    n_layers means the number of hidden processing layers.

    Example:
        n_layers=1
            input -> hidden -> output

        n_layers=2
            input -> hidden -> hidden -> output

    This matches the layer-count philosophy used in the original DeepONet
    implementation.
    """

    if n_layers < 1:
        raise ValueError(
            f"n_layers must be >= 1, got {n_layers}"
        )

    def linear(i, o):

        layer = nn.Linear(i, o)

        if spectral:
            layer = nn.utils.spectral_norm(layer)

        return layer

    layers = [
        linear(input_dim, width),
        nn.GroupNorm(groups, width),
        nn.SiLU()
    ]

    if p_drop > 0.0:
        layers.append(
            nn.Dropout(p_drop)
        )

    # Additional hidden layers
    for _ in range(n_layers - 1):

        layers.extend(
            [
                linear(width, width),
                nn.GroupNorm(groups, width),
                nn.SiLU()
            ]
        )

        if p_drop > 0.0:
            layers.append(
                nn.Dropout(p_drop)
            )

    layers.append(
        linear(width, output_dim)
    )

    return nn.Sequential(*layers)


# =============================================================================
# BASELINE DEEPONET
# =============================================================================

class BaselineDeepONet(nn.Module):
    """
    Unconstrained DeepONet.

    Branch depth:
        n_branch_layers

    Trunk depth:
        n_trunk_layers

    head='point'
        Four sigmoid outputs.

    head='hetero'
        Returns:
            mu, logvar

    bayes_last=True
        Uses Bayesian final projection.
    """

    def __init__(
        self,
        db,
        dt,
        width=128,
        n_branch_layers=2,
        n_trunk_layers=8,
        p_drop=0.0,
        head='point',
        bayes_last=False,
        n_out=4
    ):
        super().__init__()

        self.head = head
        self.n_out = n_out
        self.bayes_last = bayes_last

        self.n_branch_layers = n_branch_layers
        self.n_trunk_layers = n_trunk_layers

        # ---------------------------------------------------------------
        # Feature attention
        # ---------------------------------------------------------------

        self.attn = FeatureAttention(db)

        # ---------------------------------------------------------------
        # Branch network
        # ---------------------------------------------------------------

        self.branch = _mlp(
            input_dim=db,
            output_dim=width,
            width=width,
            n_layers=n_branch_layers,
            p_drop=p_drop,
            spectral=True
        )

        # ---------------------------------------------------------------
        # Trunk network
        # ---------------------------------------------------------------

        self.trunk = _mlp(
            input_dim=dt,
            output_dim=width,
            width=width,
            n_layers=n_trunk_layers,
            p_drop=p_drop,
            spectral=True
        )

        # ---------------------------------------------------------------
        # Cross attention
        # ---------------------------------------------------------------

        self.fuse = CrossAttentionFusion(
            width,
            heads=4
        )

        # ---------------------------------------------------------------
        # Output head
        # ---------------------------------------------------------------

        k = (
            2 * n_out
            if head == 'hetero'
            else n_out
        )

        self.proj = nn.Sequential(
            nn.Linear(width, 64),
            nn.GroupNorm(4, 64),
            nn.SiLU()
        )

        self.out = (
            BayesianLinear(64, k)
            if bayes_last
            else nn.Linear(64, k)
        )

    def forward(
        self,
        xb,
        xt,
        sample=True
    ):

        branch = self.branch(
            self.attn(xb)
        )

        trunk = self.trunk(xt)

        f = self.fuse(
            branch,
            trunk
        )

        f = self.proj(f)

        if self.bayes_last:

            z = self.out(
                f,
                sample=sample
            )

        else:

            z = self.out(f)

        # ---------------------------------------------------------------
        # Heteroscedastic Gaussian
        # ---------------------------------------------------------------

        if self.head == 'hetero':

            mu = z[:, :self.n_out]

            logvar = z[:, self.n_out:]

            return (
                torch.sigmoid(mu),
                logvar.clamp(-12.0, 4.0)
            )

        # ---------------------------------------------------------------
        # Point prediction
        # ---------------------------------------------------------------

        return torch.sigmoid(z)

    def kl(self):

        if self.bayes_last:

            return self.out.kl()

        return torch.zeros(
            (),
            device=next(self.parameters()).device
        )


# =============================================================================
# CLOSURE DEEPONET
# =============================================================================

class ClosureDeepONet(nn.Module):
    """
    Closure-constrained DeepONet.

    Branch depth:
        n_branch_layers

    Trunk depth:
        n_trunk_layers

    head='point'
        Softmax composition.

    head='logitnormal'
        Returns:
            mean composition,
            log sigma in ALR space.

    The softmax guarantees:

        STTI + RTI + STI + MTI = 1

    exactly.
    """

    def __init__(
        self,
        db,
        dt,
        p=128,
        width=192,
        n_branch_layers=4,
        n_trunk_layers=4,
        n_out=4,
        n_freq=32,
        sigma=0.3,
        p_drop=0.0,
        head='point',
        bayes_last=False
    ):
        super().__init__()

        self.p = p
        self.n_out = n_out
        self.head = head
        self.bayes_last = bayes_last

        self.n_branch_layers = n_branch_layers
        self.n_trunk_layers = n_trunk_layers

        # ---------------------------------------------------------------
        # Feature attention
        # ---------------------------------------------------------------

        self.attn = FeatureAttention(db)

        # ---------------------------------------------------------------
        # Branch
        #
        # Output = n_out * p
        # ---------------------------------------------------------------

        self.branch = _mlp(
            input_dim=db,
            output_dim=p * n_out,
            width=width,
            n_layers=n_branch_layers,
            p_drop=p_drop,
            spectral=False
        )

        # ---------------------------------------------------------------
        # Fourier time encoding
        # ---------------------------------------------------------------

        self.fourier = FourierTime(
            dt,
            n_freq=n_freq,
            sigma=sigma
        )

        # ---------------------------------------------------------------
        # Trunk
        # ---------------------------------------------------------------

        self.trunk = _mlp(
            input_dim=dt + 2 * n_freq,
            output_dim=p,
            width=width,
            n_layers=n_trunk_layers,
            p_drop=p_drop,
            spectral=False
        )

        # ---------------------------------------------------------------
        # Bias and temperature
        # ---------------------------------------------------------------

        self.bias = nn.Parameter(
            torch.zeros(n_out)
        )

        self.tau = nn.Parameter(
            torch.tensor(0.0)
        )

        # ---------------------------------------------------------------
        # Logit-normal uncertainty head
        # ---------------------------------------------------------------

        if head == 'logitnormal':

            self.logsig = _mlp(
                input_dim=db + dt + 2 * n_freq,
                output_dim=n_out - 1,
                width=64,
                n_layers=2,
                p_drop=0.0,
                spectral=False
            )

        # ---------------------------------------------------------------
        # Bayesian trunk projection
        # ---------------------------------------------------------------

        self.bayes = (
            BayesianLinear(
                p,
                p,
                zero_init=True
            )
            if bayes_last
            else None
        )

    def _logits(
        self,
        xb,
        xt,
        sample=True
    ):

        # ---------------------------------------------------------------
        # Branch
        # ---------------------------------------------------------------

        B = self.branch(
            self.attn(xb)
        ).view(
            -1,
            self.n_out,
            self.p
        )

        # ---------------------------------------------------------------
        # Fourier trunk input
        # ---------------------------------------------------------------

        te = self.fourier(xt)

        # ---------------------------------------------------------------
        # Trunk
        # ---------------------------------------------------------------

        T = self.trunk(te)

        # ---------------------------------------------------------------
        # Bayesian perturbation
        # ---------------------------------------------------------------

        if self.bayes is not None:

            T = T + self.bayes(
                T,
                sample=sample
            )

        # ---------------------------------------------------------------
        # DeepONet contraction
        # ---------------------------------------------------------------

        logits = (
            B
            * T.unsqueeze(1)
        ).sum(-1)

        logits = (
            logits / np.sqrt(self.p)
            + self.bias
        )

        # Learnable temperature
        logits = (
            logits
            * torch.exp(self.tau)
        )

        return logits, te

    def forward(
        self,
        xb,
        xt,
        sample=True
    ):

        logits, te = self._logits(
            xb,
            xt,
            sample=sample
        )

        # ---------------------------------------------------------------
        # Closure
        # ---------------------------------------------------------------

        mean = torch.softmax(
            logits,
            dim=1
        )

        # ---------------------------------------------------------------
        # Logit-normal uncertainty
        # ---------------------------------------------------------------

        if self.head == 'logitnormal':

            ls = self.logsig(
                torch.cat(
                    [xb, te],
                    dim=-1
                )
            ).clamp(
                -8.0,
                3.0
            )

            return mean, ls

        return mean

    def kl(self):

        if self.bayes is not None:

            return self.bayes.kl()

        return torch.zeros(
            (),
            device=next(self.parameters()).device
        )