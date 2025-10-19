# models.py

from __future__ import annotations
import contextlib, math, warnings
from typing import Dict

import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import MT5ForConditionalGeneration, T5Tokenizer

# ---------- geoopt integration ------------------------------------------------
import geoopt # User added
from geoopt import ManifoldParameter
from geoopt.manifolds import PoincareBall, Euclidean
GEOOPT_AVAILABLE = True # User hardcoded

# ---------- project-specific --------------------------------------------------

from stgcn_layers import Graph, get_stgcn_chain
from config import mt5_path

try:
    from utils import is_main_process
except ImportError:
    def is_main_process(): # Fallback if utils or is_main_process is not available
        if GEOOPT_AVAILABLE and torch.distributed.is_available() and torch.distributed.is_initialized():
            return torch.distributed.get_rank() == 0
        return True # Assume main process if not in distributed setting or unsure

# ============================================================================ #
#  Helper: truncated normal initialiser
# ============================================================================ #
def _no_grad_trunc_normal_(tensor, mean, std, a, b):
    def norm_cdf(x):
        return (1.0 + math.erf(x / math.sqrt(2.0))) / 2.0
    if (mean < a - 2 * std) or (mean > b + 2 * std):
        warnings.warn("mean is more than 2 std from [a, b] in trunc_normal_", stacklevel=2)

    with torch.no_grad():
        l = norm_cdf((a - mean) / std)
        u = norm_cdf((b - mean) / std)
        tensor.uniform_(2 * l - 1, 2 * u - 1).erfinv_()
        tensor.mul_(std * math.sqrt(2.0)).add_(mean)
        tensor.clamp_(min=a, max=b)
        return tensor

def trunc_normal_(tensor, mean=0.0, std=1.0, a=-2.0, b=2.0):
    return _no_grad_trunc_normal_(tensor, mean, std, a, b)

# ============================================================================ #
#  Hyperbolic utilities (active only if geoopt is available)
# ============================================================================ #
if PoincareBall is not None: # Effectively always true given GEOOPT_AVAILABLE = True above

    class HyperbolicProjection(nn.Module):
        """
        Linear -> tangent-space -> exp-map to the Poincaré ball.
        Autocast-safe: matmul in weight dtype, geo-math in fp32.
        """
        def __init__(self, dim_in: int, dim_out: int, manifold: PoincareBall):
            super().__init__()
            if not isinstance(manifold, PoincareBall):
                raise TypeError("manifold must be geoopt.manifolds.PoincareBall")
            self.manifold = manifold
            self.proj = nn.Linear(dim_in, dim_out, bias=True)
            self.log_scale  = nn.Parameter(torch.tensor(0.0, dtype=torch.float32))

        def forward(self, x: torch.Tensor) -> torch.Tensor:
            w_dtype = self.proj.weight.dtype
            y_tan   = self.proj(x.to(w_dtype)) * self.log_scale.to(w_dtype).exp()
            out     = self.manifold.expmap0(y_tan.float(), project=True)
            return out.to(x.dtype)

    class HyperbolicContrastiveLoss(nn.Module):
        def __init__(self, manifold: PoincareBall, label_smoothing: float = 0.1):
            super().__init__()
            if not isinstance(manifold, PoincareBall):
                raise TypeError("manifold must be geoopt.manifolds.PoincareBall")
            self.manifold = manifold
            self.temp         = nn.Parameter(torch.tensor(1.0))
            self.margin_base  = nn.Parameter(torch.tensor(0.3))
            self.loss_fct     = nn.CrossEntropyLoss(label_smoothing=label_smoothing, ignore_index=-100)

        def pair_loss(self, p: torch.Tensor, t: torch.Tensor) -> Dict:
            bsz  = p.shape[0]
            if bsz == 0:
                return {"loss": torch.tensor(0.0, device=p.device, requires_grad=True),
                        "sim_mean": torch.tensor(0.0, device=p.device),
                        "margin": self.margin_base.detach(),
                        "temp": torch.sigmoid(self.temp).detach()}
            dist = self.manifold.dist(p.unsqueeze(1), t.unsqueeze(0))
            sims = -dist
            tau = torch.sigmoid(self.temp) * 1.99 + 0.01
            logits = sims / tau
            eye  = torch.eye(bsz, device=logits.device, dtype=torch.bool)
            margin_cuda = self.margin_base.to(logits.dtype)
            logits = logits + margin_cuda * (~eye)
            targets = torch.arange(bsz, device=p.device)
            loss = self.loss_fct(logits, targets)
            sim_mean_pos = sims.diag().mean().detach()
            return {"loss": loss, "sim_mean": sim_mean_pos,
                    "margin": self.margin_base.detach(), "temp": tau.detach()}

    def weighted_frechet_mean_origin(points: torch.Tensor, weights: torch.Tensor,
                                     manifold: PoincareBall, max_iter=50, tol=1e-5):
        w_norm = weights / (weights.sum(dim=0, keepdim=True) + 1e-8)
        mu = points[0]
        for _ in range(max_iter):
            logmap_mu_points = manifold.logmap(mu.unsqueeze(0), points)
            bar_tan = (w_norm.unsqueeze(-1) * logmap_mu_points).sum(dim=0)
            mu_next = manifold.expmap(mu, bar_tan, project=True)
            dist_diff = manifold.dist(mu_next, mu)
            if (dist_diff < tol).all(): break
            mu = mu_next
        return mu

# ============================================================================ #
#  Original Uni-Sign Model with added Hyperbolic Projections
# ============================================================================ #
class Uni_Sign(nn.Module):

    def __init__(self, args):
        super().__init__()
        self.args    = args
        self.use_hyp = bool(getattr(args, "use_hyperbolic", False) and GEOOPT_AVAILABLE)

        self.modes = ["body", "left", "right", "face_all"]
        initial_gcn_dim = 64

        self.graph, As, self.proj_linear = {}, [], nn.ModuleDict()
        for m in self.modes:
            g = Graph(layout=m, strategy="distance", max_hop=1)
            self.graph[m] = g
            As.append(torch.tensor(g.A, dtype=torch.float32, requires_grad=False))
            self.proj_linear[m] = nn.Linear(3, initial_gcn_dim)

        self.gcn_modules        = nn.ModuleDict()
        self.fusion_gcn_modules = nn.ModuleDict()

        final_dim_gcn = -1 # Will be determined by the first GCN chain
        for i, m in enumerate(self.modes):
            current_spatial_k = As[i].shape[0]
            gcn, d_mid = get_stgcn_chain(initial_gcn_dim, "spatial", (1, current_spatial_k), As[i].clone(), True)
            fus, d_out = get_stgcn_chain(d_mid, "temporal", (5, current_spatial_k), As[i].clone(), True)
            if i == 0: final_dim_gcn = d_out
            self.gcn_modules[m]        = gcn
            self.fusion_gcn_modules[m] = fus

        if "right" in self.modes and "left" in self.modes:
            self.gcn_modules["left"]        = self.gcn_modules["right"]
            self.fusion_gcn_modules["left"] = self.fusion_gcn_modules["right"]
            self.proj_linear["left"]        = self.proj_linear["right"]

        concat_dim    = final_dim_gcn * len(self.modes)
        self.part_para = nn.Parameter(torch.zeros(concat_dim))

        mt5_cfg           = MT5ForConditionalGeneration.from_pretrained(mt5_path).config
        self.mt5_model    = MT5ForConditionalGeneration.from_pretrained(mt5_path)
        self.mt5_tokenizer= T5Tokenizer.from_pretrained(mt5_path, legacy=False)
        self.mt5_dim      = mt5_cfg.d_model
        self.pose_proj    = nn.Linear(concat_dim, self.mt5_dim)

        if self.use_hyp and GEOOPT_AVAILABLE:
            self.hyp_dim    = args.hyp_dim
            self.manifold   = PoincareBall(c=args.init_c, learnable=True)
            self.hyp_proj_body  = HyperbolicProjection(final_dim_gcn, self.hyp_dim, self.manifold)
            self.hyp_proj_right = HyperbolicProjection(final_dim_gcn, self.hyp_dim, self.manifold)
            self.hyp_proj_left  = HyperbolicProjection(final_dim_gcn, self.hyp_dim, self.manifold)
            self.hyp_proj_face  = HyperbolicProjection(final_dim_gcn, self.hyp_dim, self.manifold)
            self.hyp_proj_text  = HyperbolicProjection(self.mt5_dim,   self.hyp_dim, self.manifold)

            # For 'token' mode attention
            self.hyp_attn_W = geoopt.ManifoldParameter(torch.randn(args.hyp_dim, args.hyp_dim), manifold=self.manifold)
            self.hyp_attn_b = geoopt.ManifoldParameter(torch.zeros(args.hyp_dim), manifold=self.manifold)


            self.geom_loss        = HyperbolicContrastiveLoss(self.manifold, args.label_smoothing_hyp)
            self.loss_alpha_logit = nn.Parameter(torch.tensor(0.0))
            self.register_buffer("global_step", torch.tensor(0, dtype=torch.long))
            self.total_steps      = max(int(getattr(args, 'total_steps', 1)), 1)
            self.text_cmp_mode    = getattr(args, "hyp_text_cmp", "pooled")
            if self.text_cmp_mode not in {"pooled", "attn", "token"}:
                raise ValueError(f"--hyp_text_cmp must be pooled|attn|token, got {self.text_cmp_mode}")
            if self.text_cmp_mode == "attn":
                self.text_pool_attn = nn.Linear(self.mt5_dim, 1)
            self.hyp_text_emb_src = getattr(args, "hyp_text_emb_src", "token")
            if self.hyp_text_emb_src not in {"token", "decoder"}:
                raise ValueError(f"--hyp_text_emb_src must be 'token' or 'decoder', got {self.hyp_text_emb_src}")

        self.apply(self._init_weights)

    @staticmethod
    def _init_weights(m):
        if isinstance(m, nn.Linear):
            if not (GEOOPT_AVAILABLE and isinstance(m.weight, ManifoldParameter)):
                trunc_normal_(m.weight, std=0.02)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.ones_(m.weight); nn.init.zeros_(m.bias)

    def forward(self, src_input: Dict, tgt_input: Dict) -> Dict[str, torch.Tensor]:
        if self.mt5_model is None or self.mt5_tokenizer is None:
            raise RuntimeError("mT5 model or tokenizer not loaded.")

        out, compute_dtype = {}, self.pose_proj.weight.dtype
        autocast_ctx = contextlib.nullcontext() # Assuming AMP is handled by DeepSpeed or externally

        # ========== Pose encoding =======================================
        with autocast_ctx:
            feats, pooled, body_feat = [], {}, None
            active_modes = [m for m in self.modes if m in src_input]
            if not active_modes: raise ValueError("src_input contains no data for any defined modes.")

            for part in active_modes:
                x = self.proj_linear[part](src_input[part].to(dtype=compute_dtype)).permute(0,3,1,2)
                gcn_out = self.gcn_modules[part](x)
                if body_feat is not None:
                    if part == "left"   and body_feat.shape[-1] >= 2: gcn_out = gcn_out + body_feat[..., -2][..., None].detach()
                    elif part == "right" and body_feat.shape[-1] >= 1: gcn_out = gcn_out + body_feat[..., -1][..., None].detach()
                    elif part == "face_all" and body_feat.shape[-1] >= 1: gcn_out = gcn_out + body_feat[...,  0][..., None].detach()
                if part == "body": body_feat = gcn_out
                gcn_out = self.fusion_gcn_modules[part](gcn_out)
                pool_sp = gcn_out.mean(dim=-1).transpose(1,2)
                feats.append(pool_sp)
                pooled[part] = pool_sp.mean(dim=1)

            concatenated_feats = torch.cat(feats, dim=-1)
            pose_features_biased = concatenated_feats
            if len(active_modes) == len(self.modes): # Only add bias if all parts present (original logic)
                pose_features_biased = concatenated_feats + self.part_para
            pose_emb = self.pose_proj(pose_features_biased)

            # ========== mT5  ============================================
            prefix_ids    = src_input["prefix_ids"].long()
            prefix_mask   = src_input["prefix_mask"]
            inputs_embeds = torch.cat([self.mt5_model.encoder.embed_tokens(prefix_ids), pose_emb], dim=1)
            attention_mask= torch.cat([prefix_mask, src_input["attention_mask"]], dim=1)
            labels        = tgt_input["labels_ids"].long()
            labels_masked = labels.clone()
            labels_masked[labels_masked == self.mt5_tokenizer.pad_token_id] = -100

            mt5_out = self.mt5_model(inputs_embeds=inputs_embeds, attention_mask=attention_mask,
                                     labels=labels_masked, return_dict=True, output_hidden_states=True)
            logits  = mt5_out.logits
            ce_loss = F.cross_entropy(logits.view(-1, logits.size(-1)).float(),
                                      labels_masked.view(-1),
                                      label_smoothing=getattr(self.args, 'label_smoothing', 0.0),
                                      ignore_index=-100)
            out["ce_loss"] = ce_loss.detach()

        # ========== Hyperbolic branch ====================================
        margin_loss = torch.tensor(0.0, device=ce_loss.device)
        # Default alpha for CE loss, gets updated if hyperbolic branch runs
        alpha_scalar = torch.tensor(getattr(self.args, 'alpha', 1.0), device=ce_loss.device)
        geom_out     = {}
        current_step_eval_tensors_for_out = {} # For out["eval_figure_data"]

        if self.use_hyp and GEOOPT_AVAILABLE:
            with torch.cuda.amp.autocast(enabled=False): # Geoopt operations in fp32
                if not all(part in pooled for part in self.modes):
                    if self.args.eval and is_main_process(): # Print warning only if in eval & main process
                        warnings.warn(
                            "Skipping hyperbolic branch & eval_figure_data: Missing pooled features.",
                            stacklevel=1
                        )
                else:
                    # ---- Pose parts projection to Poincaré ball ----
                    hyp_body  = self.hyp_proj_body (pooled["body"].float())
                    hyp_left  = self.hyp_proj_left (pooled["left"].float())
                    hyp_right = self.hyp_proj_right(pooled["right"].float())
                    hyp_face  = self.hyp_proj_face (pooled["face_all"].float())
                    pose_points_stacked = torch.stack([hyp_body, hyp_left, hyp_right, hyp_face])

                    # ---- Fréchet Mean Calculation ----
                    d0s = torch.stack([self.manifold.dist0(p) for p in pose_points_stacked])
                    w = torch.softmax(d0s, dim=0)
                    mu_mfd = weighted_frechet_mean_origin(pose_points_stacked, w, self.manifold)

                    # ---- Text embeddings ----
                    mask_bool = (labels != self.mt5_tokenizer.pad_token_id)
                    if self.hyp_text_emb_src == "token":
                        txt_e = self.mt5_model.encoder.embed_tokens(labels.clamp_min(0))
                    else: # "decoder"
                        if mt5_out.decoder_hidden_states is None:
                            raise RuntimeError("Decoder hidden states not available for hyp_text_emb_src='decoder'.")
                        txt_e = mt5_out.decoder_hidden_states[-1]

                    # ---- Contrastive Loss Calculation (based on text_cmp_mode) ----
                    # This section defines: margin_loss, geom_out
                    # And potentially: hyp_text_p, att (for attn mode), hyp_text, attn_weights, text_context (for token mode)
                    if self.text_cmp_mode == "pooled":
                        txt_mean = (txt_e * mask_bool.unsqueeze(-1).float()).sum(dim=1) / \
                                   mask_bool.float().sum(dim=1, keepdim=True).clamp_min(1)
                        hyp_text_p = self.hyp_proj_text(txt_mean.float())
                        geom_out = self.geom_loss.pair_loss(mu_mfd, hyp_text_p)
                        margin_loss = geom_out["loss"]
                    elif self.text_cmp_mode == "attn":
                        att_raw = self.text_pool_attn(txt_e.float()).squeeze(-1)
                        att_raw = att_raw.masked_fill(~mask_bool, -torch.inf)
                        att     = att_raw.softmax(dim=-1).unsqueeze(-1)
                        txt_mean = (att * txt_e).sum(dim=1)
                        hyp_text_p = self.hyp_proj_text(txt_mean.float())
                        geom_out = self.geom_loss.pair_loss(mu_mfd, hyp_text_p)
                        margin_loss = geom_out["loss"]
                    elif self.text_cmp_mode == "token":
                        B, K, D = pose_points_stacked.shape[1], pose_points_stacked.shape[0], self.hyp_dim

                        # (a) Hyperbolic Tokenization (Values)
                        # Project all Euclidean text tokens to hyperbolic space. These are the 'values'.
                        hyp_text_tokens = self.hyp_proj_text(txt_e.float()) # Shape: (B, T, D)

                        # (b) Hyperbolic Attention
                        # (b.1) Queries from pose parts.
                        # Reshape from (K, B, D) -> (B, K, 1, D) for broadcasting.
                        queries = pose_points_stacked.transpose(0, 1).unsqueeze(2)

                        # (b.2) Create attention keys via Möbius transformation (M ⊗c v + b).
                        # We unsqueeze hyp_text_tokens to (B, 1, T, D) to broadcast against K queries.
                        keys = self.manifold.mobius_add(
                            self.manifold.mobius_matvec(self.hyp_attn_W, hyp_text_tokens.unsqueeze(1)),
                            self.hyp_attn_b,
                            project=True
                        )

                        # (b.3) Compute attention scores: negative geodesic distance.
                        attn_logits = -self.manifold.dist(queries, keys) # Shape: (B, K, T)

                        # (b.4) Apply padding mask and softmax to get attention weights.
                        attn_logits = attn_logits.masked_fill(~mask_bool.unsqueeze(1), -torch.inf)
                        attn_weights = attn_logits.softmax(dim=-1) # Shape: (B, K, T)

                        # (b.5) Compute context vectors {cp} as hyperbolic weighted midpoint of the values.
                        # This computes K distinct context vectors for each item in the batch.
                        text_contexts = self.manifold.weighted_midpoint(
                            hyp_text_tokens.unsqueeze(1), # Values
                            weights=attn_weights,         # Weights
                            dim=2,                        # Aggregate over the token dimension (T)
                            project=True
                        ) # Shape: (B, K, D)

                        # (c) Compute final loss: average of K contrastive losses.
                        # Reshape poses and contexts to (B*K, D) to compute loss in one batch.
                        all_poses = pose_points_stacked.transpose(0, 1).reshape(B * K, -1)
                        all_texts = text_contexts.reshape(B * K, -1)

                        geom_out = self.geom_loss.pair_loss(all_poses, all_texts)
                        margin_loss = geom_out["loss"]
                       
                    # --- Populate eval_figure_data FOR CURRENT BATCH if in global eval mode ---
                    # Uncomment this section if you want to store tensors for eval
                    if self.args.eval: # Check global CLI --eval flag
                        temp_tensors = {}
                        if 'hyp_body' in locals() and hyp_body is not None: temp_tensors["hyp_body"] = hyp_body.detach().cpu()
                        if 'hyp_left' in locals() and hyp_left is not None: temp_tensors["hyp_left"] = hyp_left.detach().cpu()
                        if 'hyp_right' in locals() and hyp_right is not None: temp_tensors["hyp_right"] = hyp_right.detach().cpu()
                        if 'hyp_face' in locals() and hyp_face is not None: temp_tensors["hyp_face"] = hyp_face.detach().cpu()
                        
                        # These names depend on the text_cmp_mode logic execution
                        if 'hyp_text' in locals() and hyp_text is not None: temp_tensors["hyp_text"] = hyp_text.detach().cpu()
                        elif 'hyp_text_p' in locals() and hyp_text_p is not None: temp_tensors["hyp_text"] = hyp_text_p.detach().cpu()
                        
                        if 'attn_weights' in locals() and attn_weights is not None: temp_tensors["attn_weights"] = attn_weights.detach().cpu()
                        elif 'att' in locals() and att is not None and self.text_cmp_mode == "attn": temp_tensors["attn_weights"] = att.squeeze(-1).detach().cpu()
                        
                        if 'text_context' in locals() and text_context is not None: temp_tensors["text_context"] = text_context.detach().cpu()
                        if 'mask_bool' in locals() and mask_bool is not None: temp_tensors["text_mask_bool"] = mask_bool.detach().cpu()
                        if 'mu_mfd' in locals() and mu_mfd is not None: temp_tensors["mu_mfd"] = mu_mfd.detach().cpu()
                        if 'w' in locals() and w is not None: temp_tensors["frechet_weights_w"] = w.detach().cpu()
                        
                        if temp_tensors and is_main_process(): # Log if tensors are populated
                             print(f"[DEBUG model.forward] Populated current_step_eval_tensors_for_out with keys: {list(temp_tensors.keys())}")
                        current_step_eval_tensors_for_out = temp_tensorsV

                    # ---- Loss Blend ----
                    prog = self.global_step.item() / self.total_steps if self.total_steps > 0 else 0
                    a_base_val = getattr(self.args, 'alpha', 0.8)
                    a_base = a_base_val + (0.1 * prog)
                    a_learn = torch.sigmoid(self.loss_alpha_logit) * 0.2
                    alpha_scalar = (a_base + a_learn).clamp(0.1, 1.0) # Update alpha_scalar

                    # ---- Logging hyperbolic metrics to 'out' ----
                    if geom_out:
                        log_weights_cond = 'w' in locals() and w is not None and w.numel() > 0
                        out.update({
                            "hyp_sim_mean": geom_out.get("sim_mean", torch.tensor(0.0)),
                            "temperature": geom_out.get("temp", torch.tensor(0.0)),
                            "effective_margin": geom_out.get("margin", torch.tensor(0.0)),
                            "curvature": self.manifold.c.abs().detach(),
                            "weights_fm_body" : w[0].mean().detach() if log_weights_cond and w.shape[0] > 0 else torch.tensor(0.0),
                            "weights_fm_left" : w[1].mean().detach() if log_weights_cond and w.shape[0] > 1 else torch.tensor(0.0),
                            "weights_fm_right": w[2].mean().detach() if log_weights_cond and w.shape[0] > 2 else torch.tensor(0.0),
                            "weights_fm_face" : w[3].mean().detach() if log_weights_cond and w.shape[0] > 3 else torch.tensor(0.0),
                            "margin_loss_val": margin_loss.detach(), # Actual hyp loss value
                            "alpha_hyp": alpha_scalar.detach(),      # Blended alpha
                        })
        # else: current_step_eval_tensors_for_out remains {}, margin_loss remains 0, alpha_scalar uses default

        # --- Final Loss Calculation ---
        loss = ce_loss.float() # Default to CE loss
        if self.use_hyp and GEOOPT_AVAILABLE and geom_out: # If hyp branch ran successfully and produced geom_out
            # alpha_scalar would have been updated by the loss blend logic inside the hyp branch
            loss = alpha_scalar * ce_loss.float() + (1 - alpha_scalar) * margin_loss
            
        # ========== 4. Outputs =============================================
        out.update({
            "loss": loss,
            "margin_loss": margin_loss.detach(), # Will be 0 if hyp branch didn't run or geom_out empty
            "alpha": alpha_scalar.detach(),      # Will be default args.alpha or blended value
            "inputs_embeds": inputs_embeds.detach(),
            "attention_mask": attention_mask.detach(),
        })
        out["eval_figure_data"] = current_step_eval_tensors_for_out # Add current batch's data (empty if not args.eval or hyp skip)
        return out

    @torch.no_grad()
    def generate(self, pc: Dict[str, torch.Tensor],
                 *, max_new_tokens: int = 100, num_beams: int = 4, **kwargs) -> torch.Tensor:
        # ... (Your existing generate method - it seems fine) ...
        if not {"inputs_embeds", "attention_mask"} <= pc.keys():
            if "body" in pc and "prefix_ids" in pc:
                with torch.no_grad():
                    compute_dtype = self.pose_proj.weight.dtype
                    feats, _, body_feat = [], {}, None # pooled not needed for generate
                    active_modes = [m for m in self.modes if m in pc]
                    if not active_modes: raise ValueError("Input pc contains no data for defined modes.")
                    for part in active_modes:
                        x = self.proj_linear[part](pc[part].to(dtype=compute_dtype)).permute(0,3,1,2)
                        gcn_out = self.gcn_modules[part](x)
                        if body_feat is not None:
                            if part == "left" and body_feat.shape[-1] >= 2: gcn_out += body_feat[..., -2][..., None].detach()
                            elif part == "right" and body_feat.shape[-1] >= 1: gcn_out += body_feat[..., -1][..., None].detach()
                            elif part == "face_all" and body_feat.shape[-1] >= 1: gcn_out += body_feat[..., 0][..., None].detach()
                        if part == "body": body_feat = gcn_out
                        gcn_out = self.fusion_gcn_modules[part](gcn_out)
                        pool_sp = gcn_out.mean(dim=-1).transpose(1,2)
                        feats.append(pool_sp)
                    concatenated_feats = torch.cat(feats, dim=-1)
                    pose_features_biased = concatenated_feats
                    if len(active_modes) == len(self.modes):
                         pose_features_biased += self.part_para
                    pose_emb = self.pose_proj(pose_features_biased)
                    prefix_ids    = pc["prefix_ids"].long()
                    prefix_mask   = pc["prefix_mask"]
                    if self.mt5_model is None: raise RuntimeError("mT5 model not loaded.")
                    inputs_embeds  = torch.cat([self.mt5_model.encoder.embed_tokens(prefix_ids), pose_emb], dim=1)
                    if "attention_mask" not in pc: raise ValueError("Pose attention_mask missing in pc for generation.")
                    attention_mask = torch.cat([prefix_mask, pc["attention_mask"]], dim=1)
                    pc_out = {"inputs_embeds": inputs_embeds, "attention_mask": attention_mask}
            else:
                raise ValueError("generate: need 'inputs_embeds'/'attention_mask', or full src_input dict in pc.")
        else:
            pc_out = pc
        if self.mt5_model is None: raise RuntimeError("Cannot generate, mT5 model not loaded.")
        return self.mt5_model.generate(
            inputs_embeds  = pc_out["inputs_embeds"],
            attention_mask = pc_out["attention_mask"],
            max_new_tokens = max_new_tokens,
            num_beams      = num_beams,
            **kwargs
        )

# ============================================================================ #
#  (deprecated) helper – kept for checkpoint compatibility with original UniSign
# ============================================================================ #
def get_requires_grad_dict(model: nn.Module) -> Dict[str, torch.Tensor]:
    # ... (Your existing get_requires_grad_dict method) ...
    warnings.warn("get_requires_grad_dict is deprecated. Use model.state_dict().",
                  DeprecationWarning, stacklevel=2)
    param_req = {n: p.requires_grad for n, p in model.named_parameters()}
    if GEOOPT_AVAILABLE:
        for n,p in model.named_parameters():
            if isinstance(p, ManifoldParameter):
                warnings.warn(f"{n} is ManifoldParameter – may need geoopt to reload.", stacklevel=2)
    dup_map = {k.replace("left","right"):v for k,v in param_req.items() if "left" in k}
    param_req.update(dup_map)
    return {k:v for k,v in model.state_dict().items() if param_req.get(k, False)}
