"""text_encoder の attention をクエリ方向に分割して計算する(Windows 向け)。

Windows の PyTorch には flash attention のカーネルが無く、因果マスク付きの SDPA は math カーネルに落ちて
スコア行列(トークン数² × ヘッド数 × 2〜4 バイト)を丸ごと確保する。参照画像 2 枚の編集(約 2.3k トークン)で
1GB 超の単発アロケーションになり、6GB カードでは TE の中で OOM した(実測、RTX 3050)。Linux では flash /
mem-efficient カーネルが使われるのでこの行列は作られない。

ここではクエリを chunk 行ずつに分け、各チャンクについて明示的なマスク(因果なら下三角)を作って SDPA を呼ぶ。
行ごとの softmax なので結果は分割しない場合と同じ(bf16 の丸め順序の差を除く)。スコア行列は chunk × N に収まる。
transformers の AttentionInterface に "sdpa_chunked" として登録し、config._attn_implementation で切り替える。
"""

from __future__ import annotations

import torch
from transformers import AttentionInterface
from transformers.integrations.sdpa_attention import repeat_kv, sdpa_attention_forward

CHUNK = 512


def chunked_sdpa(module, query, key, value, attention_mask, dropout: float = 0.0, scaling=None,
                 is_causal=None, **kwargs):
    q_len, kv_len = query.shape[2], key.shape[2]
    if q_len <= CHUNK or dropout:
        return sdpa_attention_forward(module, query, key, value, attention_mask, dropout=dropout,
                                      scaling=scaling, is_causal=is_causal, **kwargs)
    if getattr(module, "num_key_value_groups", 1) > 1:
        key = repeat_kv(key, module.num_key_value_groups)
        value = repeat_kv(value, module.num_key_value_groups)
    causal = is_causal if is_causal is not None else getattr(module, "is_causal", True)
    causal = bool(causal) and attention_mask is None
    out = torch.empty_like(query)
    for i0 in range(0, q_len, CHUNK):
        i1 = min(i0 + CHUNK, q_len)
        if attention_mask is not None:
            mask = attention_mask[:, :, i0:i1, :kv_len] if attention_mask.dim() == 4 else attention_mask
        elif causal:
            rows = torch.arange(i0, i1, device=query.device)[:, None]
            cols = torch.arange(kv_len, device=query.device)[None, :]
            mask = (cols <= rows)[None, None]          # True = attend(下三角、この行までのキー)
        else:
            mask = None
        out[:, :, i0:i1] = torch.nn.functional.scaled_dot_product_attention(
            query[:, :, i0:i1], key, value, attn_mask=mask, dropout_p=0.0, scale=scaling, is_causal=False)
    return out.transpose(1, 2).contiguous(), None


AttentionInterface.register("sdpa_chunked", chunked_sdpa)


def enable(te, chunk: int) -> None:
    """te の言語モデルとビジョン塔の attention を分割版に切り替える。"""
    global CHUNK
    CHUNK = int(chunk)
    for cfg in (te.config, te.config.text_config, te.config.vision_config,
                te.model.language_model.config, te.model.visual.config):
        try:
            cfg._attn_implementation = "sdpa_chunked"
        except Exception:  # noqa: BLE001
            pass
