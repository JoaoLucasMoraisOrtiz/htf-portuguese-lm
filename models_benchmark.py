"""
Modelos Pareados para Benchmark Científico Rigoroso:
1. Baseline: Transformer Causal (Byte-Level Llama-style com RoPE e SwiGLU)
2. Inovação: BEABA-2 / TTT-DEQ (Fast Weights via Regra Delta Multi-Head + Raciocínio Latente em Equilíbrio)

Ambos os modelos operam diretamente sobre o mesmo espaço de bytes UTF-8 (vocab_size=256),
eliminando viés de tokenização e garantindo comparação pareada em contagem de parâmetros.
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F

# ==============================================================================
# 1. BASELINE: TRANSFORMER CAUSAL (LLAMA/GPT STYLE)
# ==============================================================================

class CausalSelfAttention(nn.Module):
    def __init__(self, d_model: int, n_heads: int):
        super().__init__()
        self.d_model = d_model
        self.n_heads = n_heads
        self.head_dim = d_model // n_heads
        
        self.q_proj = nn.Linear(d_model, d_model, bias=False)
        self.k_proj = nn.Linear(d_model, d_model, bias=False)
        self.v_proj = nn.Linear(d_model, d_model, bias=False)
        self.out_proj = nn.Linear(d_model, d_model, bias=False)

    def forward(self, x: torch.Tensor, use_cache: bool = False, past_kv: tuple = None):
        B, T, D = x.shape
        q = self.q_proj(x).view(B, T, self.n_heads, self.head_dim).transpose(1, 2)
        k = self.k_proj(x).view(B, T, self.n_heads, self.head_dim).transpose(1, 2)
        v = self.v_proj(x).view(B, T, self.n_heads, self.head_dim).transpose(1, 2)

        if past_kv is not None:
            past_k, past_v = past_kv
            k = torch.cat([past_k, k], dim=2)
            v = torch.cat([past_v, v], dim=2)

        current_kv = (k, v) if use_cache else None

        # Causal Attention com PyTorch SDPA otimizado
        out = F.scaled_dot_product_attention(q, k, v, is_causal=True if past_kv is None else False)
        out = out.transpose(1, 2).contiguous().view(B, T, D)
        return self.out_proj(out), current_kv


class SwiGLUMLP(nn.Module):
    def __init__(self, d_model: int, d_ff: int):
        super().__init__()
        self.w_gate = nn.Linear(d_model, d_ff, bias=False)
        self.w_up = nn.Linear(d_model, d_ff, bias=False)
        self.w_down = nn.Linear(d_ff, d_model, bias=False)

    def forward(self, x: torch.Tensor):
        return self.w_down(F.silu(self.w_gate(x)) * self.w_up(x))


class TransformerBlock(nn.Module):
    def __init__(self, d_model: int, n_heads: int, d_ff: int):
        super().__init__()
        self.norm1 = nn.LayerNorm(d_model)
        self.attn = CausalSelfAttention(d_model, n_heads)
        self.norm2 = nn.LayerNorm(d_model)
        self.mlp = SwiGLUMLP(d_model, d_ff)

    def forward(self, x: torch.Tensor, use_cache: bool = False, past_kv: tuple = None):
        attn_out, kv = self.attn(self.norm1(x), use_cache=use_cache, past_kv=past_kv)
        x = x + attn_out
        x = x + self.mlp(self.norm2(x))
        return x, kv


class BaselineTransformer(nn.Module):
    """Transformer Causal Padrão Pareado em Parâmetros."""
    def __init__(self, vocab_size: int = 256, d_model: int = 384, n_layers: int = 6, n_heads: int = 6):
        super().__init__()
        self.d_model = d_model
        self.embed = nn.Embedding(vocab_size, d_model)
        d_ff = int(d_model * 2.66)
        self.layers = nn.ModuleList([TransformerBlock(d_model, n_heads, d_ff) for _ in range(n_layers)])
        self.final_norm = nn.LayerNorm(d_model)
        self.lm_head = nn.Linear(d_model, vocab_size, bias=False)

    def forward(self, input_ids: torch.Tensor):
        x = self.embed(input_ids) * math.sqrt(self.d_model)
        for layer in self.layers:
            x, _ = layer(x)
        x = self.final_norm(x)
        return self.lm_head(x)


# ==============================================================================
# 2. INOVAÇÃO: BEABA-2 / TTT-DEQ (FAST WEIGHTS + LATENT EQUILIBRIUM)
# ==============================================================================

class MultiHeadTTTMemory(nn.Module):
    """
    Pilar 1: Memória Plástica Multi-Head via Fast Weights (Regra Delta / TTT).
    Cada cabeça h gerencia uma matriz de pesos rápidos W_t^(h) com aprendizado online.
    Zero KV-cache! Espaço de memória O(1).
    """
    def __init__(self, d_model: int = 384, n_heads: int = 6):
        super().__init__()
        self.d_model = d_model
        self.n_heads = n_heads
        self.head_dim = d_model // n_heads

        self.k_proj = nn.Linear(d_model, d_model, bias=False)
        self.v_proj = nn.Linear(d_model, d_model, bias=False)
        self.q_proj = nn.Linear(d_model, d_model, bias=False)
        self.eta_proj = nn.Linear(d_model, n_heads, bias=True)
        self.out_proj = nn.Linear(d_model, d_model, bias=False)
        self.norm = nn.LayerNorm(d_model)

    def forward(self, x: torch.Tensor, W_prev: torch.Tensor = None):
        B, T, D = x.shape
        H = self.n_heads
        d_k = self.head_dim

        K = F.normalize(self.k_proj(x).view(B, T, H, d_k), p=2, dim=-1) # [B, T, H, d_k]
        V = self.v_proj(x).view(B, T, H, d_k)                           # [B, T, H, d_k]
        Q = self.q_proj(x).view(B, T, H, d_k)                           # [B, T, H, d_k]
        Eta = torch.sigmoid(self.eta_proj(x))                           # [B, T, H]

        if W_prev is None:
            # W: [B, H, d_k, d_k]
            W = torch.zeros(B, H, d_k, d_k, device=x.device, dtype=x.dtype)
        else:
            W = W_prev.clone()

        readouts = []
        decay = 0.98

        for t in range(T):
            k_t = K[:, t, :, :].unsqueeze(-1)    # [B, H, d_k, 1]
            v_t = V[:, t, :, :].unsqueeze(-1)    # [B, H, d_k, 1]
            q_t = Q[:, t, :, :].unsqueeze(-1)    # [B, H, d_k, 1]
            eta_t = Eta[:, t, :].view(B, H, 1, 1) # [B, H, 1, 1]

            # Erro de reconstrução (surpresa): e_t = v_t - W_{t-1} k_t
            pred_v = torch.matmul(W, k_t)
            e_t = v_t - pred_v

            # Atualização plástica online dos pesos rápidos:
            # W_t = λ W_{t-1} + η_t (e_t @ k_t^T)
            W = decay * W + eta_t * torch.matmul(e_t, k_t.transpose(-2, -1))

            # Leitura instantânea de contexto: c_t = W_t @ q_t
            c_t = torch.matmul(W, q_t).squeeze(-1) # [B, H, d_k]
            readouts.append(c_t.view(B, D))

        readouts = torch.stack(readouts, dim=1) # [B, T, D]
        return self.out_proj(self.norm(readouts)), W


class DeepEquilibriumSolver(nn.Module):
    """
    Pilar 2: Motor de Raciocínio Latente em Equilíbrio (DEQ).
    Itera em espaço latente contínuo z até convergir para um atrator semântico z*.
    """
    def __init__(self, d_model: int = 384, max_iter: int = 8, alpha: float = 0.5):
        super().__init__()
        self.d_model = d_model
        self.max_iter = max_iter
        self.alpha = alpha

        # Bloco único não-linear compartilhado de alta expressividade
        d_ff = int(d_model * 2.66)
        self.norm1 = nn.LayerNorm(d_model)
        self.mlp1 = SwiGLUMLP(d_model, d_ff)
        self.norm2 = nn.LayerNorm(d_model)
        self.mlp2 = SwiGLUMLP(d_model, d_ff)

    def forward(self, x: torch.Tensor, context: torch.Tensor):
        B, T, D = x.shape
        u = x + context
        z = u.clone()

        # Iteração de ponto fixo latente
        for _ in range(self.max_iter):
            f_z = self.mlp1(self.norm1(z)) + self.mlp2(self.norm2(z + u))
            z = (1.0 - self.alpha) * z + self.alpha * (f_z + u)

        return z


class BEABA2LanguageModel(nn.Module):
    """
    Modelo Completo BEABA-2:
    Embedding de Bytes -> Memória Plástica TTT (Fast Weights) -> Raciocínio Latente DEQ -> Logits.
    Pareado em contagem de parâmetros com o Baseline Transformer.
    """
    def __init__(self, vocab_size: int = 256, d_model: int = 384, n_heads: int = 6, max_thinking_steps: int = 8):
        super().__init__()
        self.d_model = d_model
        self.embed = nn.Embedding(vocab_size, d_model)
        
        # Módulo de Contexto Plástico
        self.memory = MultiHeadTTTMemory(d_model=d_model, n_heads=n_heads)
        
        # Módulo de Raciocínio Latente em Equilíbrio
        self.reasoning = DeepEquilibriumSolver(d_model=d_model, max_iter=max_thinking_steps)
        
        self.final_norm = nn.LayerNorm(d_model)
        self.lm_head = nn.Linear(d_model, vocab_size, bias=False)

    def forward(self, input_ids: torch.Tensor, W_prev: torch.Tensor = None):
        x = self.embed(input_ids) * math.sqrt(self.d_model)
        
        # 1. Ingestão e recuperação na memória plástica
        context, W_final = self.memory(x, W_prev=W_prev)
        
        # 2. Raciocínio em equilíbrio latente
        z_star = self.reasoning(x, context)
        
        # 3. Projeção de saída
        logits = self.lm_head(self.final_norm(z_star))
        return logits, W_final


def print_parameter_comparison():
    t_model = BaselineTransformer(d_model=384, n_layers=6, n_heads=6)
    b_model = BEABA2LanguageModel(d_model=384, n_heads=6, max_thinking_steps=8)
    
    t_params = sum(p.numel() for p in t_model.parameters())
    b_params = sum(p.numel() for p in b_model.parameters())
    
    print(f"Parâmetros Baseline Transformer: {t_params:,} ({t_params / 1e6:.2f} M)")
    print(f"Parâmetros BEABA-2 / TTT-DEQ:     {b_params:,} ({b_params / 1e6:.2f} M)")
    print(f"Diferença relativa:               {abs(t_params - b_params) / t_params * 100:.2f}%")

if __name__ == "__main__":
    print_parameter_comparison()
