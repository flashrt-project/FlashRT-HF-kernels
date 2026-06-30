import sys
import os
import argparse
from pathlib import Path

import torch
import torch.nn.functional as F

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
sys.path.insert(0, str(HERE))

import pi05_decoder_loop_hub as dec  # noqa: E402
import pi05_hf_decoder_e2e as e2e  # noqa: E402

FP8_MAX = 448.0


def _dyn_scale(x: torch.Tensor, device: torch.device) -> torch.Tensor:
    return torch.clamp(x.float().abs().max() / FP8_MAX, min=1e-12).reshape(1).to(
        device=device, dtype=torch.float32
    )


class DynFp8VisionRuntime(e2e.Fp8HubVisionRuntime):
    def _setup_fp8(self) -> None:
        super()._setup_fp8()
        self._ones_h = torch.ones((e2e.VIS_H,), device=self.w.device, dtype=torch.bfloat16)

    def __call__(self) -> torch.Tensor:
        ff = self._fp8ffn
        self.layout.patch_im2col_bf16(self.images, out=self.patches)
        self.gemm.bf16_linear_bf16(self.patches, self.w.patch_w, out=self.x)
        self.residual.bias_residual_bf16(self.x, self.w.pos_embed, self.w.patch_b, out=self.x)
        self.norm.layer_norm_bf16(self.x, self.w.ln1_w[0], self.w.ln1_b[0], out=self.normed)
        for i in range(e2e.VIS_L):
            qi = _dyn_scale(self.normed, self.w.device)
            self.gemm.channel_scale_quantize_fp8_static_bf16(self.normed, self._ones, qi, out=self._nf)
            ff.fp8_gemm_bf16(self._nf, self._qkv_f[i], qi, self._qkv_s[i], out=self.qkv_buf)
            self.qkv_buf.add_(self.w.qkv_b[i])
            self.qkv.qkv_split_bf16(
                self.qkv_buf.view(self.num_views, e2e.VIS_SEQ_PER_VIEW, 3 * e2e.VIS_D),
                e2e.VIS_NH,
                e2e.VIS_HD,
                self.q,
                self.k,
                self.v,
            )
            self.attn_kernel.fwd(self.q, self.k, self.v, out=self.attn_bthd, p_dropout=0.0, is_causal=False)
            oi = _dyn_scale(self.attn, self.w.device)
            self.gemm.channel_scale_quantize_fp8_static_bf16(self.attn, self._ones, oi, out=self._af)
            ff.fp8_gemm_bf16(self._af, self._o_f[i], oi, self._o_s[i], out=self.normed)
            self.residual.bias_residual_bf16(self.x, self.normed, self.w.o_b[i], out=self.x)
            self.norm.layer_norm_bf16(self.x, self.w.ln2_w[i], self.w.ln2_b[i], out=self.normed)

            fi = _dyn_scale(self.normed, self.w.device)
            self.gemm.channel_scale_quantize_fp8_static_bf16(self.normed, self._ones, fi, out=self._nf)
            ff.fp8_gemm_bf16(self._nf, self._up_f[i], fi, self._up_s[i], out=self._hbf)
            self._hbf.add_(self.w.up_b[i])
            self._hbf.copy_(F.gelu(self._hbf.float(), approximate="tanh").to(torch.bfloat16))
            hi = _dyn_scale(self._hbf, self.w.device)
            self.gemm.channel_scale_quantize_fp8_static_bf16(self._hbf, self._ones_h, hi, out=self._hf)
            ff.fp8_gemm_bf16(self._hf, self._dn_f[i], hi, self._dn_s[i], out=self.normed)
            self.residual.bias_residual_bf16(self.x, self.normed, self.w.down_b[i], out=self.x)
            if i != e2e.VIS_L - 1:
                self.norm.layer_norm_bf16(self.x, self.w.ln1_w[i + 1], self.w.ln1_b[i + 1], out=self.normed)
        self.norm.layer_norm_bf16(self.x, self.w.final_norm_w, self.w.final_norm_b, out=self.normed)
        self.gemm.bf16_linear_bias_bf16(self.normed, self.w.projector_w, self.w.projector_b, out=self.encoder_x_vision)
        return self.encoder_x_input


class DynFp8EncoderRuntime(e2e.Fp8HubEncoderRuntime):
    def enable_fp8(self, qkv_in_scale, o_in_scale) -> None:
        super().enable_fp8(qkv_in_scale, o_in_scale)
        self._ones_h = torch.ones((e2e.ENC_H,), device=self.w.device, dtype=torch.bfloat16)
        seq = self.x.shape[0]
        self._hidden_bf = [
            torch.empty((seq, e2e.ENC_H), device=self.w.device, dtype=torch.bfloat16)
            for _ in range(self.w.layers)
        ]

    def __call__(self) -> tuple[torch.Tensor, torch.Tensor]:
        self.x.copy_(self.encoder_x_input)
        for i in range(self.w.layers):
            self.norm.rms_norm_bf16(self.x, self.w.ones, out=self.normed)
            qi = _dyn_scale(self.normed, self.w.device)
            self.gemm.channel_scale_quantize_fp8_static_bf16(self.normed, self.w.ones, qi, out=self._xf)
            self.ffn.fp8_gemm_bf16(self._xf, self._qkv_w_fp8[i], qi, self._qkv_w_scale[i], out=self.qkv_buf)
            self.qkv.qkv_split_rope_kvcache_bf16(
                self.qkv_buf.view(1, self.x.shape[0], -1),
                self.rope,
                e2e.ENC_NH,
                e2e.DEC_NKV,
                e2e.DEC_HD,
                0,
                self.q,
                self.k_all[i : i + 1],
                self.v_all[i : i + 1],
            )
            if i == self.w.layers - 1:
                break
            self.attn_kernel.fwd(
                self.q,
                self.k_all[i : i + 1],
                self.v_all[i : i + 1],
                out=self.attn_bthd,
                p_dropout=0.0,
                is_causal=False,
            )
            oi = _dyn_scale(self.attn, self.w.device)
            self.gemm.channel_scale_quantize_fp8_static_bf16(self.attn, self._oh, oi, out=self._af)
            self.ffn.fp8_gemm_bf16(self._af, self._o_w_fp8[i], oi, self._o_w_scale[i], out=self.attn_o)
            self.residual.gate_residual_bf16(self.x, self.attn_o, self.ones_gate, out=self.x)

            self.norm.rms_norm_bf16(self.x, self.w.ones, out=self.normed)
            fi = _dyn_scale(self.normed, self.w.device)
            self.gemm.channel_scale_quantize_fp8_static_bf16(self.normed, self.w.ones, fi, out=self.ffn_fp8)
            self.ffn.fp8_gemm_bf16(self.ffn_fp8, self.w.gate_up_w_fp8[i], fi, self.w.gate_up_w_scale[i], out=self.gate_up[i])
            gate, up = self.gate_up[i].float().chunk(2, dim=1)
            self._hidden_bf[i].copy_((F.gelu(gate, approximate="tanh") * up).to(torch.bfloat16))
            hidden = self._hidden_bf[i]
            hi = _dyn_scale(hidden, self.w.device)
            self.gemm.channel_scale_quantize_fp8_static_bf16(hidden, self._ones_h, hi, out=self.hidden_fp8[i])
            self.ffn.fp8_gemm_bf16(self.hidden_fp8[i], self.w.down_w_fp8[i], hi, self.w.down_w_scale[i], out=self.ffn_out[i])
            self.residual.gate_residual_bf16(self.x, self.ffn_out[i], self.ones_gate, out=self.x)
        return self.k_all, self.v_all


class DynFp8DecoderLoop(dec.Fp8HubDecoderLoop):
    def enable_fp8_projections(self, qkv_in_scale, o_in_scale) -> None:
        super().enable_fp8_projections(qkv_in_scale, o_in_scale)
        self._ones_h = torch.ones((dec.DEC_H,), device=self.w.device, dtype=torch.bfloat16)
        cs = self.x.shape[0]
        self._hidden_bf = [
            torch.empty((cs, dec.DEC_H), device=self.w.device, dtype=torch.bfloat16)
            for _ in range(self.w.layers)
        ]

    def __call__(self) -> torch.Tensor:
        w = self.w
        s = self.s
        self.noise.copy_(s.noise0)
        for step in range(w.steps):
            self.gemm.bf16_linear_bias_bf16(self.noise, w.action_in_w, w.action_in_b, out=self.x)
            for i in range(w.layers):
                self.adapt.ada_rms_norm_style_bf16(
                    self.x, w.ones, w.style_attn[step, i], out=self.normed, gate_out=self.gate
                )
                qi = _dyn_scale(self.normed, w.device)
                self.gemm.channel_scale_quantize_fp8_static_bf16(self.normed, self._ones_d, qi, out=self._nf)
                self.ffn.fp8_gemm_bf16(self._nf, self._qkv_w_fp8[i], qi, self._qkv_w_scale[i], out=self.qkv_buf)
                self.k_cache[:, : s.encoder_seq_len].copy_(dec._layer_kv(s.encoder_k, i, name="encoder_k"))
                self.v_cache[:, : s.encoder_seq_len].copy_(dec._layer_kv(s.encoder_v, i, name="encoder_v"))
                self.qkv.qkv_split_rope_kvcache_bf16(
                    self.qkv_buf.view(1, s.chunk_size, -1),
                    s.rope,
                    dec.DEC_NH,
                    dec.DEC_NKV,
                    dec.DEC_HD,
                    s.encoder_seq_len,
                    self.q,
                    self.k_cache,
                    self.v_cache,
                )
                self.attn_kernel.fwd(self.q, self.k_cache, self.v_cache, out=self.attn_bthd, p_dropout=0.0, is_causal=False)
                oi = _dyn_scale(self.attn, w.device)
                self.gemm.channel_scale_quantize_fp8_static_bf16(self.attn, self._ones_a, oi, out=self._af)
                self.ffn.fp8_gemm_bf16(self._af, self._o_w_fp8[i], oi, self._o_w_scale[i], out=self.attn_o)
                self.residual.gate_residual_bf16(self.x, self.attn_o, self.gate, out=self.x)

                self.adapt.ada_rms_norm_style_bf16(
                    self.x, w.ones, w.style_ffn[step, i], out=self.normed, gate_out=self.ffn_gate
                )
                fi = _dyn_scale(self.normed, w.device)
                self.gemm.channel_scale_quantize_fp8_static_bf16(self.normed, self._ones_d, fi, out=self.ffn_fp8)
                self.ffn.fp8_gemm_bf16(self.ffn_fp8, w.gate_up_w_fp8[i], fi, w.gate_up_w_scale[i], out=self.gate_up[i])
                gate, up = self.gate_up[i].float().chunk(2, dim=1)
                self._hidden_bf[i].copy_((F.gelu(gate, approximate="tanh") * up).to(torch.bfloat16))
                hidden = self._hidden_bf[i]
                hi = _dyn_scale(hidden, w.device)
                self.gemm.channel_scale_quantize_fp8_static_bf16(hidden, self._ones_h, hi, out=self.hidden_fp8[i])
                self.ffn.fp8_gemm_bf16(self.hidden_fp8[i], w.down_w_fp8[i], hi, w.down_w_scale[i], out=self.ffn_out[i])
                self.residual.gate_residual_bf16(self.x, self.ffn_out[i], self.ffn_gate, out=self.x)
            self.adapt.ada_rms_norm_style_bf16(
                self.x, w.ones, w.style_final[step], out=self.final_normed, gate_out=self.final_gate
            )
            self.gemm.bf16_linear_bias_bf16(self.final_normed, w.action_out_w, w.action_out_b, out=self.action)
            self.residual.gate_residual_bf16(self.noise, self.action, self.noise_gate, out=self.noise)
        return self.noise


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run PI0.5 full all-FP8 static-vs-dynamic calibration e2e."
    )
    parser.add_argument("mode", choices=("static", "dynamic"))
    parser.add_argument(
        "--checkpoint",
        default=None,
        help=(
            "Path to pi05_libero_pytorch checkpoint directory or model.safetensors. "
            "Defaults to $PI05_CHECKPOINT or ../checkpoints/pi05_libero_pytorch."
        ),
    )
    parser.add_argument(
        "--encoder-bundle",
        default=str(ROOT / "internal-tests/runtime-demo/pi05-real-images-encoder-x-kv-frame50.pt"),
    )
    parser.add_argument(
        "--calibration-input",
        default=str(ROOT / "internal-tests/runtime-demo/pi05-hf-vision-encoder-decoder-frame50-decoder-static-scales.json"),
    )
    parser.add_argument(
        "--encoder-calibration-input",
        default=str(ROOT / "internal-tests/runtime-demo/pi05-hf-vision-encoder-frame50-static-scales.json"),
    )
    parser.add_argument("--warmup", type=int, default=8)
    parser.add_argument("--iters", type=int, default=30)
    parser.add_argument("--no-cuda-graph", action="store_true")
    args = parser.parse_args()

    mode = args.mode
    if mode == "dynamic":
        e2e.Fp8HubVisionRuntime = DynFp8VisionRuntime
        e2e.Fp8HubEncoderRuntime = DynFp8EncoderRuntime
        dec.Fp8HubDecoderLoop = DynFp8DecoderLoop

    checkpoint = (
        args.checkpoint
        or os.environ.get("PI05_CHECKPOINT")
        or str(ROOT.parent / "checkpoints/pi05_libero_pytorch")
    )

    sys.argv = [
        "e2e",
        "run-vision-encoder-decoder",
        "--encoder-bundle",
        args.encoder_bundle,
        "--checkpoint",
        checkpoint,
        "--calibration-input",
        args.calibration_input,
        "--encoder-calibration-input",
        args.encoder_calibration_input,
        "--warmup",
        str(args.warmup),
        "--iters",
        str(args.iters),
        "--fp8-projections",
    ]
    if not args.no_cuda_graph:
        sys.argv.append("--cuda-graph")
    e2e.main()


if __name__ == "__main__":
    main()
