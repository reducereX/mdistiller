"""
SFW-SupCon: Semantic Force Weighted Supervised Contrastive Distillation
=======================================================================
Integration for mdistiller framework (mdistiller/distillers/SFWSupCon.py).

Algorithm summary
-----------------
1. Teacher backbone is FROZEN.  Only a lightweight projection MLP attached to
   the teacher is updated jointly (at LR/10) — the CRD-paper approach.
2. Both teacher and student project their penultimate features down to a shared
   64-d contrastive space via their own MLP heads.
3. The SFW-SupCon loss pulls same-class pairs together and pushes cross-class
   pairs apart, with pull/push weights derived from the teacher's softmax
   confidence (α · p_teacher for pull; adaptive β · (1 − p_teacher) for push).
4. A momentum memory bank (K = 4096, m = 0.5) stores the full dataset's teacher
   projections and corresponding logits so that each mini-batch sees far more
   negatives than the batch size alone provides.
5. The CE loss on the student is computed from the backbone features directly
   and its gradient does NOT flow back through the contrastive projection head
   (the two paths are independent, as in the original code).

Projection dimensions
---------------------
* Teacher (ResNet-50 for CIFAR-100): penultimate dim = 2048
    2048 → 512 → ReLU → 256 → ReLU → 64   (downsampling MLP)
* Student (ResNet-8×4 / ResNet-32×4 / etc.):  penultimate dim varies by model,
    read at runtime from `student.get_stage_channels()[-1]`.
    e.g. resnet32x4 → 256;  resnet8x4 → 256; vgg13 → 512; wrn_40_2 → 128
    student_dim → max(student_dim, 128) → ReLU → 64  (upsampling if needed,
    downsampling if already large).

Why 64-d?
---------
64-d gives a compact but sufficiently expressive hypersphere.  The original
code's projector already bottlenecks to 64 (128→64).  Uniform initialisation
on S^63 has enough capacity to represent 100 CIFAR classes well, and the small
size keeps the memory bank cheap (4096 × 64 × 4 bytes = 1 MB).  You could try
128 if you suspect under-capacity, but 64 worked well in the original notebook.

Usage (mdistiller config YAML)
------------------------------
  distiller: SFWSupCon
  SFWSupCon:
    alpha: 1.0          # pull weight multiplier
    beta: 10.0          # push weight multiplier
    temperature: 0.07
    proj_dim: 64
    bank_size: 4096     # set 0 to disable memory bank
    bank_momentum: 0.5
    adaptive_beta: true
    ce_weight: 1.0      # weight for standard cross-entropy term
    kd_weight: 1.0      # weight for contrastive distillation term

Register in mdistiller/__init__.py:
    from .SFWSupCon import SFWSupCon

Add teacher projection optimizer in tools/train.py  (see note at bottom of
this file) so teacher.projection is updated at LR/10.
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F

from ._base import Distiller  # mdistiller base class


# ---------------------------------------------------------------------------
# Projection heads
# ---------------------------------------------------------------------------

class _DownsampleMLP(nn.Module):
    """Maps a high-dim feature vector down to `out_dim`.

    Architecture mirrors the teacher projection in the original notebook:
        in_dim → hidden → ReLU → hidden/4 → ReLU → out_dim
    For very large in_dim (≥ 1024) we add one extra bottleneck stage.
    """

    def __init__(self, in_dim: int, out_dim: int = 64):
        super().__init__()
        if in_dim >= 1024:
            # e.g. ResNet-50: 2048 → 512 → 256 → 64
            h1 = in_dim // 4
            h2 = max(h1 // 2, out_dim * 2)
            self.net = nn.Sequential(
                nn.Linear(in_dim, h1),
                nn.ReLU(inplace=True),
                nn.Linear(h1, h2),
                nn.ReLU(inplace=True),
                nn.Linear(h2, out_dim),
            )
        else:
            # e.g. ResNet-32×4: 256 → 128 → 64
            h1 = max(in_dim, out_dim * 2)
            self.net = nn.Sequential(
                nn.Linear(in_dim, h1),
                nn.ReLU(inplace=True),
                nn.Linear(h1, out_dim),
            )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class _UpsampleMLP(nn.Module):
    """Maps a low-dim feature vector UP to `out_dim`.

    Used when the student's penultimate dim < out_dim (rare for CIFAR
    models but included for completeness, e.g. WRN-16-1 → 64 features).
        in_dim → in_dim*2 → ReLU → out_dim
    """

    def __init__(self, in_dim: int, out_dim: int = 64):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, in_dim * 2),
            nn.ReLU(inplace=True),
            nn.Linear(in_dim * 2, out_dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


def _make_projector(in_dim: int, out_dim: int = 64) -> nn.Module:
    """Factory: choose down- or up-sampling MLP based on dimensionality."""
    if in_dim >= out_dim:
        return _DownsampleMLP(in_dim, out_dim)
    else:
        return _UpsampleMLP(in_dim, out_dim)


# ---------------------------------------------------------------------------
# Momentum Memory Bank
# ---------------------------------------------------------------------------

class MomentumMemoryBank(nn.Module):
    """
    Index-based momentum memory bank (CRD-style).

    Stores normalised teacher projections + labels + logits for the entire
    training set.  At each forward pass it:
      1. Performs a momentum update for the current batch's indices.
      2. Samples K random entries as negatives and writes them into
         `self.queue` / `self.queue_labels` / `self.queue_logits` so the
         loss function can read them without any extra plumbing.

    Parameters
    ----------
    n_data : int
        Total number of training samples (e.g. 50 000 for CIFAR-100).
    dim : int
        Projection dimensionality (must match proj_dim, default 64).
    K : int
        Number of negatives sampled per forward pass (bank "window").
    momentum : float
        EMA coefficient.  0.5 as in original code.
    num_classes : int
        Number of output classes (100 for CIFAR-100).
    """

    def __init__(
        self,
        n_data: int = 50_000,
        dim: int = 64,
        K: int = 4096,
        momentum: float = 0.5,
        num_classes: int = 100,
    ):
        super().__init__()
        self.n_data = n_data
        self.dim = dim
        self.K = K
        self.momentum = momentum
        self.num_classes = num_classes
        # Expose `size` so loss functions can guard on `memory_bank.size > 0`
        self.size = K

        # --- persistent buffers (no gradient) ---
        stdv = 1.0 / math.sqrt(dim / 3)
        self.register_buffer(
            "memory_features",
            F.normalize(
                torch.rand(n_data, dim).mul_(2 * stdv).add_(-stdv), dim=1
            ),
        )
        self.register_buffer("memory_labels", torch.zeros(n_data, dtype=torch.long))
        self.register_buffer("memory_logits", torch.zeros(n_data, num_classes))

        # "Queue" view — overwritten every forward pass, read by loss functions
        self.register_buffer("queue", torch.zeros(dim, K))          # [dim, K]
        self.register_buffer("queue_labels", torch.zeros(K, dtype=torch.long))
        self.register_buffer("queue_logits", torch.zeros(K, num_classes))

    @torch.no_grad()
    def forward(self, features, indices, labels, logits, update=True):
        if update:
            features_norm = F.normalize(features.detach(), dim=1)  # stays on GPU
            old_f = self.memory_features[indices]
            new_f = F.normalize(
                self.momentum * old_f + (1.0 - self.momentum) * features_norm, dim=1
            )
            self.memory_features[indices] = new_f
            self.memory_labels[indices] = labels
            old_l = self.memory_logits[indices]
            self.memory_logits[indices] = (
                self.momentum * old_l + (1.0 - self.momentum) * logits.detach()
            )

        sample_idx = torch.randperm(self.n_data, device=self.memory_features.device)[: self.K]
        bank_f = self.memory_features[sample_idx]
        bank_l = self.memory_labels[sample_idx]
        bank_lg = self.memory_logits[sample_idx]

        self.queue = bank_f.T.contiguous()
        self.queue_labels = bank_l
        self.queue_logits = bank_lg

        return bank_f.detach()


# ---------------------------------------------------------------------------
# Loss function
# ---------------------------------------------------------------------------

def _sfwsupcon_loss_with_bank(
    student_proj: torch.Tensor,      # [B, dim]  — NOT yet normalised
    teacher_proj: torch.Tensor,      # [B, dim]  — NOT yet normalised
    teacher_logits: torch.Tensor,    # [B, C]
    labels: torch.Tensor,            # [B]
    memory_bank: "MomentumMemoryBank | None",
    alpha: float = 1.0,
    beta: float = 10.0,
    tau: float = 0.07,
    adaptive_beta: bool = True,
) -> torch.Tensor:
    """
    SFW-SupCon loss (with or without memory bank).

    Positive pairs  = same-class samples within the batch.
    Negative pairs  = cross-class samples in batch  +  bank negatives (if any).

    Pull weight  : w_pull  = alpha  * p_teacher(y | x_i)
    Push weight  : w_push  = beta_eff * (1 − p_teacher(y_j | x_i))
      where beta_eff = beta / (p_target + 0.5)  [adaptive] or just beta.

    The loss is then:
        L = −log( w_pull · Σ_{j+} exp(s·t/τ)  /
                  (w_pull · Σ_{j+} exp(s·t/τ)  +  Σ_{j−} w_push_j · exp(s·t/τ)) )
    normalised by alpha to keep gradient magnitude stable.
    """
    B = student_proj.shape[0]
    device = student_proj.device

    s_norm = F.normalize(student_proj, dim=1)      # [B, dim]
    t_norm = F.normalize(teacher_proj, dim=1)      # [B, dim]
    teacher_probs = F.softmax(teacher_logits, dim=1)  # [B, C]

    labels_col = labels.view(-1, 1)                # [B, 1]

    # ---- similarities -------------------------------------------------------
    sim_batch = torch.matmul(s_norm, t_norm.T) / tau   # [B, B]

    if memory_bank is not None and memory_bank.size > 0:
        bank_feats = memory_bank.queue.T.detach()       # [K, dim]
        sim_bank = torch.matmul(s_norm, bank_feats.T) / tau  # [B, K]
        all_sims = torch.cat([sim_batch, sim_bank], dim=1)   # [B, B+K]
    else:
        all_sims = sim_batch                              # [B, B]

    # Numerical stability: subtract row max (detached)
    sim_max, _ = torch.max(all_sims, dim=1, keepdim=True)
    all_sims = torch.clamp(all_sims - sim_max.detach(), min=-50.0, max=50.0)
    exp_all = torch.exp(all_sims)                         # [B, B(+K)]

    exp_batch = exp_all[:, :B]                            # [B, B]
    if memory_bank is not None and memory_bank.size > 0:
        exp_bank = exp_all[:, B:]                         # [B, K]

    # ---- in-batch masks -----------------------------------------------------
    # mask_pos[i,j] = 1 iff same class (including self)
    mask_pos = torch.eq(labels_col, labels_col.T).float()   # [B, B]
    # Remove self-similarity from positives (diagonal = 0)
    mask_pos.fill_diagonal_(0.0)
    mask_neg = 1.0 - torch.eq(labels_col, labels_col.T).float()  # [B, B]

    # ---- pull weights -------------------------------------------------------
    p_target = torch.gather(teacher_probs, 1, labels_col).view(-1)  # [B]
    w_pull = alpha * p_target                                         # [B]

    # ---- push weights (in-batch) --------------------------------------------
    # p_negative_class[i, j] = p_teacher(label_i | x_j)
    labels_row = labels_col.view(1, -1).expand(B, -1)  # [B, B]
    p_negative_class = torch.gather(teacher_probs, 1, labels_row)  # [B, B]

    if adaptive_beta:
        beta_eff = (beta / (p_target + 0.5)).view(-1, 1)             # [B, 1]
        w_push = beta_eff * (1.0 - p_negative_class)                 # [B, B]
    else:
        w_push = beta * (1.0 - p_negative_class)                     # [B, B]

    # ---- numerator & in-batch denominator -----------------------------------
    sum_pos_exp = (exp_batch * mask_pos).sum(dim=1)           # [B]
    
    # guaranteed instance-matched positive: student_i · teacher_i (diagonal)
    # this eliminates degenerate anchors when no same-class pairs exist in batch
    diag_exp = torch.diagonal(exp_batch)                      # [B]
    numerator_term = w_pull * (sum_pos_exp + diag_exp)        # [B]
    
    # numerator_term = w_pull * sum_pos_exp                     # [B]
    weighted_neg_exp = (exp_batch * w_push * mask_neg).sum(dim=1)  # [B]
    denominator_term = numerator_term + weighted_neg_exp      # [B]

    # ---- bank denominator ---------------------------------------------------
    if memory_bank is not None and memory_bank.size > 0:
        bank_labels = memory_bank.queue_labels               # [K]
        bank_logits = memory_bank.queue_logits               # [K, C]
        bank_probs = F.softmax(bank_logits, dim=1)           # [K, C]

        # mask_bank_neg[i, k] = 1 iff bank[k] is a different class from i
        mask_bank_neg = (~torch.eq(
            labels_col,                    # [B, 1]
            bank_labels.view(1, -1)        # [1, K]
        )).float()                         # [B, K]

        # p_bank_at_target[i, k] = p_teacher(label_i | bank_sample_k)
        # labels.view(-1) → [B];  bank_probs → [K, C]
        # We want bank_probs[:, label_i]  →  gather along dim=1 using labels
        # bank_probs.T → [C, K];  index with labels → [B, K]
        p_bank_at_target = bank_probs[:, labels.view(-1)].T  # [B, K]

        if adaptive_beta:
            beta_factor = (beta / (p_target + 0.5)).view(-1, 1)   # [B, 1]
            w_push_bank = beta_factor * (1.0 - p_bank_at_target)   # [B, K]
        else:
            w_push_bank = beta * (1.0 - p_bank_at_target)          # [B, K]

        weighted_bank_sum = (w_push_bank * exp_bank * mask_bank_neg).sum(dim=1)
        denominator_term = denominator_term + weighted_bank_sum    # [B]

    # ---- final loss ---------------------------------------------------------
    eps = 1e-8
    loss = -torch.log((numerator_term + eps) / (denominator_term + eps))
    return (loss / alpha).mean()

def _supcon_pretrain_loss(t_proj, labels, tau=0.07):
    """
    Standard supervised contrastive loss applied to teacher projections only.
    Used to pretrain the teacher projector so it has meaningful semantic
    structure before the bank starts being populated and the student starts
    distilling from it.
    """
    t_norm = F.normalize(t_proj, dim=1)
    sim = torch.matmul(t_norm, t_norm.T) / tau   # [2B, 2B]

    # numerical stability
    sim_max, _ = torch.max(sim, dim=1, keepdim=True)
    sim = sim - sim_max.detach()

    labels_col = labels.view(-1, 1)
    mask_pos = torch.eq(labels_col, labels_col.T).float()
    # exclude self from positives (every other same-class sample IS a positive)
    mask_pos.fill_diagonal_(0.0)
    # mask out self from denominator too
    mask_self = 1.0 - torch.eye(sim.shape[0], device=sim.device)

    exp_sim = torch.exp(sim) * mask_self
    log_prob = sim - torch.log(exp_sim.sum(dim=1, keepdim=True) + 1e-8)

    # mean log-prob over positives per anchor
    pos_count = mask_pos.sum(dim=1).clamp(min=1.0)
    loss = -(mask_pos * log_prob).sum(dim=1) / pos_count

    return loss.mean()

# ---------------------------------------------------------------------------
# Main Distiller class
# ---------------------------------------------------------------------------

class SFWSupCon(Distiller):
    """
    Semantic Force Weighted Supervised Contrastive Distillation.

    Drop-in for mdistiller.  Add to mdistiller/distillers/__init__.py:

        from .SFWSupCon import SFWSupCon

    Expected cfg keys (under cfg.SFWSupCon):
        alpha          (float, default 1.0)
        beta           (float, default 10.0)
        temperature    (float, default 0.07)
        proj_dim       (int,   default 64)
        bank_size      (int,   default 4096;  0 = no bank)
        bank_momentum  (float, default 0.5)
        adaptive_beta  (bool,  default True)
        ce_weight      (float, default 1.0)
        kd_weight      (float, default 1.0)

    Notes
    -----
    * The teacher backbone weights are NEVER updated here.  Only the teacher
      projection MLP (self.teacher_projector) is trainable.  You must add it
      to the optimiser in your training script — see `get_learnable_parameters`.
    * The memory bank requires dataset indices to be passed in `data` dict as
      `data["index"]`.  Use the IndexedCIFAR100 wrapper below or mdistiller's
      CRD dataset wrapper.
    * The CE loss gradient is detached from the contrastive path (student
      encoder features are shared but the classifier branch has its own path).
    """

    def __init__(self, student, teacher, cfg):
        super().__init__(student, teacher)

        # ---- hyper-parameters -----------------------------------------------
        sfw_cfg = cfg.SFWSupCon
        self.alpha = getattr(sfw_cfg, "alpha", 1.0)
        self.beta = getattr(sfw_cfg, "beta", 10.0)
        self.tau = getattr(sfw_cfg, "temperature", 0.07)
        self.proj_dim = getattr(sfw_cfg, "proj_dim", 64)
        self.bank_size = getattr(sfw_cfg, "bank_size", 4096)
        self.bank_momentum = getattr(sfw_cfg, "bank_momentum", 0.5)
        self.adaptive_beta = getattr(sfw_cfg, "adaptive_beta", True)
        self.ce_weight = getattr(sfw_cfg, "ce_weight", 1.0)
        self.kd_weight = getattr(sfw_cfg, "kd_weight", 1.0)

        # ---- infer feature dimensions ---------------------------------------
        # mdistiller models expose get_stage_channels(); last entry = penultimate dim
        self.s_dim = self._get_feat_dim(student)
        self.t_dim = self._get_feat_dim(teacher)

        # ---- projection heads -----------------------------------------------
        # Teacher projector: large → 64  (frozen backbone, trainable head)
        self.teacher_projector = _make_projector(self.t_dim, self.proj_dim)
        # Student projector: varies → 64
        self.student_projector = _make_projector(self.s_dim, self.proj_dim)

        # ---- memory bank (initialised lazily once n_data is known) ----------
        self.memory_bank: "MomentumMemoryBank | None" = None
        self._bank_size_cfg = self.bank_size   # stash; build in forward on 1st call
        self._n_data: int = 0
        self._bank_ready: bool = False

        # ---- CE criterion ---------------------------------------------------
        self.ce_loss_fn = nn.CrossEntropyLoss()

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _get_feat_dim(model: nn.Module) -> int:
        """Return the penultimate feature dimension of an mdistiller model."""
        if hasattr(model, "get_stage_channels"):
            return model.get_stage_channels()[-1]
        # Fallback: try fc / classifier weight shape
        for name in ("fc", "classifier", "head", "linear"):
            layer = getattr(model, name, None)
            if layer is not None and hasattr(layer, "in_features"):
                return layer.in_features
        raise RuntimeError(
            "Cannot infer feature dimension from model.  "
            "Implement get_stage_channels() or set s_dim/t_dim manually."
        )

    def _init_bank(self, n_data: int, device: torch.device):
        # """Lazy initialisation of the memory bank once dataset size is known."""
        # if self._bank_size_cfg > 0 and not self._bank_ready:
        #     self.memory_bank = MomentumMemoryBank(
        #         n_data=n_data,
        #         dim=self.proj_dim,
        #         K=self._bank_size_cfg,
        #         momentum=self.bank_momentum,
        #         num_classes=self.student.fc.out_features,  # CIFAR-100 → 100
        #     ).to(device)
        #     self._bank_ready = True
        if self._bank_ready:          # ← already initialized, skip
            return
        if self.bank_size > 0:
            self.memory_bank = MomentumMemoryBank(
                n_data=50_000,
                dim=self.proj_dim,
                K=self.bank_size,
                momentum=self.bank_momentum,
                num_classes=self.student.fc.out_features,
            ).to(device)
            self._bank_ready = True
        else:
            self.memory_bank = None
            self._bank_ready = False

    def get_learnable_parameters(self):
        """
        Return (name, param) pairs for the optimiser.

        Includes:
          - all student parameters
          - teacher_projector parameters (joint training at lower LR)

        The teacher backbone itself is excluded (frozen).

        In your tools/train.py, split these into two param groups:
            [
              {"params": student_params,           "lr": base_lr},
              {"params": teacher_projector_params, "lr": base_lr / 10},
            ]
        Use the `is_teacher_proj` flag we attach below.
        """
        params = []
        for n, p in self.student.named_parameters():
            params.append((f"student.{n}", p))
        for n, p in self.teacher_projector.named_parameters():
            params.append((f"teacher_projector.{n}", p))
        for n, p in self.student_projector.named_parameters():
            params.append((f"student_projector.{n}", p))
        return params

    def get_extra_parameters(self):
        """
        Called by mdistiller's train.py to get distiller-specific params.
        Returns a list of parameter groups suitable for torch.optim.

        The teacher projector is updated at LR/10 (joint training, CRD-style).
        The student projector is updated at the base LR alongside the student.
        """
        return [
            {
                "params": list(self.teacher_projector.parameters()),
                "lr_scale": 0.1,   # tools/train.py multiplies base_lr by this
                "name": "teacher_projector",
            },
            {
                "params": list(self.student_projector.parameters()),
                "lr_scale": 1.0,
                "name": "student_projector",
            },
        ]

    # ------------------------------------------------------------------
    # Forward
    # ------------------------------------------------------------------

    def forward_train(self, image, target, index, epoch=None, **kwargs):
        # ----- unpack two-view or single-view input -------------------------
        if isinstance(image, (list, tuple)):
            view1, view2 = image[0], image[1]
        else:
            view1 = view2 = image

        device = view1.device
        images_dual = torch.cat([view1, view2], dim=0)
        labels_dual = torch.cat([target, target], dim=0)
        indices_dual = torch.cat([index, index], dim=0)

        # ----- teacher forward (backbone frozen) ----------------------------
        with torch.no_grad():
            t_logits, t_feats = self.teacher(images_dual)
            t_feat = t_feats["pooled_feat"]

        t_proj = self.teacher_projector(t_feat.detach())

        # ----- student forward ----------------------------------------------
        s_logits, s_feats = self.student(images_dual)
        s_feat = s_feats["pooled_feat"]
        s_proj = self.student_projector(s_feat)

        # ----- memory bank init + update ------------------------------------
        self._init_bank(50_000, device)

        # ============================================================
        # PHASE 1: Teacher projector pretraining (epochs 1..TP_PRETRAIN)
        # ============================================================
        TP_PRETRAIN_EPOCHS = 30  # tune this; notebook used ~10 with cosine probe

        if epoch is not None and epoch <= TP_PRETRAIN_EPOCHS:
            # Train teacher_projector with supervised contrastive loss on t_proj alone.
            # NO bank used yet (bank is still random — would just inject noise).
            # NO student contrastive — just CE on student.
            kd_loss = _supcon_pretrain_loss(t_proj, labels_dual, tau=self.tau)

            # Don't update bank during pretraining — wait until projector is good
            # (otherwise bank fills up with mid-pretraining garbage)

            if epoch == 1 and not getattr(self, "_pretrain_announced", False):
                print(f"\n[PRETRAIN] Teacher projector pretraining for {TP_PRETRAIN_EPOCHS} epochs")
                self._pretrain_announced = True

        # ============================================================
        # PHASE 2: Full SFW-SupCon distillation (epoch > TP_PRETRAIN)
        # ============================================================
        else:
            # Now bank starts updating — projector is competent, entries are meaningful
            if self.memory_bank is not None:
                self.memory_bank(t_proj, indices_dual, labels_dual, t_logits, update=True)

            if epoch == TP_PRETRAIN_EPOCHS + 1 and not getattr(self, "_phase2_announced", False):
                print(f"\n[PHASE 2] Switching to full SFW-SupCon distillation with bank")
                self._phase2_announced = True

            kd_loss = _sfwsupcon_loss_with_bank(
                student_proj=s_proj,
                teacher_proj=t_proj,
                teacher_logits=t_logits,
                labels=labels_dual,
                memory_bank=self.memory_bank,
                alpha=self.alpha,
                beta=self.beta,
                tau=self.tau,
                adaptive_beta=self.adaptive_beta,
            )

        # ----- CE loss on view-1 only (always active) -----------------------
        s_logits_v1 = self.student.fc(s_feat[:view1.shape[0]].detach())
        ce_loss = self.ce_loss_fn(s_logits_v1, target)

        losses_dict = {
            "loss_ce": ce_loss,
            "loss_kd": kd_loss,
        }
        return s_logits_v1, losses_dict

# ---------------------------------------------------------------------------
# Dataset wrapper — required for index-based memory bank
# ---------------------------------------------------------------------------

class CIFAR100WithIndex(torch.utils.data.Dataset):
    """
    Thin wrapper around an existing CIFAR-100 Dataset that returns
    (image, target, index).

    Usage (in mdistiller's get_cifar100_dataloaders or your own loader):

        from mdistiller.distillers.SFWSupCon import CIFAR100WithIndex
        base_train = torchvision.datasets.CIFAR100(...)
        train_set  = CIFAR100WithIndex(base_train)

    The SFWSupCon distiller reads `data["index"]` in forward_train().
    mdistiller's default collate puts the 3-tuple into {"image", "target", "index"}
    if you rename appropriately, or you can write a custom collate_fn.

    Alternatively, use mdistiller's existing CRD index dataset if it's already
    set up in your codebase — they are equivalent.
    """

    def __init__(self, base_dataset):
        self.base = base_dataset

    def __len__(self):
        return len(self.base)

    def __getitem__(self, idx):
        img, target = self.base[idx]
        return img, target, idx


# ---------------------------------------------------------------------------
# Convenience: build two-view transform compatible with mdistiller
# ---------------------------------------------------------------------------

class TwoViewTransform:
    """Applies `base_transform` twice independently to produce two views."""

    def __init__(self, base_transform):
        self.t = base_transform

    def __call__(self, x):
        return self.t(x), self.t(x)