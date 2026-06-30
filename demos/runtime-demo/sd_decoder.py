import argparse, os, sys, time, torch, torch.nn.functional as F
from pathlib import Path
HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
sys.path.insert(0, str(HERE))
import pi05_decoder_loop_hub as dec
FP8=448.0
class DynDecoder(dec.HubDecoderLoop):
    """Same as base (BF16 QKV/O, FP8 FFN) but FFN scales computed per-forward -> split GeGLU."""
    def _sc(self, a): return torch.clamp(a/FP8, min=1e-12).reshape(1).to(self.w.device, torch.float32)
    def __call__(self):
        w=self.w; s=self.s; self.noise.copy_(s.noise0)
        H=w.down_w_fp8[0].shape[1]
        oH=torch.ones(H, device=w.device, dtype=torch.bfloat16)
        for step in range(w.steps):
            self.gemm.bf16_linear_bias_bf16(self.noise, w.action_in_w, w.action_in_b, out=self.x)
            for i in range(w.layers):
                self.adapt.ada_rms_norm_style_bf16(self.x, w.ones, w.style_attn[step,i], out=self.normed, gate_out=self.gate)
                self.gemm.bf16_linear_bf16(self.normed, w.qkv_w[i], out=self.qkv_buf)
                self.k_cache[:, :s.encoder_seq_len].copy_(dec._layer_kv(s.encoder_k,i,name="encoder_k"))
                self.v_cache[:, :s.encoder_seq_len].copy_(dec._layer_kv(s.encoder_v,i,name="encoder_v"))
                self.qkv.qkv_split_rope_kvcache_bf16(self.qkv_buf.view(1,s.chunk_size,-1), s.rope, dec.DEC_NH, dec.DEC_NKV, dec.DEC_HD, s.encoder_seq_len, self.q, self.k_cache, self.v_cache)
                if self.attention_backend=="fa2":
                    self.attn_kernel.fwd(self.q,self.k_cache,self.v_cache,out=self.attn_bthd,p_dropout=0.0,is_causal=False)
                else:
                    self.attn.copy_(dec._sdpa_gqa(self.q,self.k_cache,self.v_cache))
                self.gemm.bf16_linear_bf16(self.attn, w.o_w[i], out=self.attn_o)
                self.residual.gate_residual_bf16(self.x, self.attn_o, self.gate, out=self.x)
                # --- DYNAMIC FFN: per-forward input + hidden scales, split geglu ---
                self.adapt.ada_rms_norm_style_bf16(self.x, w.ones, w.style_ffn[step,i], out=self.normed, gate_out=self.ffn_gate)
                in_dyn=self._sc(self.normed.float().abs().max())
                ffn_fp8=self.gemm.channel_scale_quantize_fp8_static_bf16(self.normed, w.ones, in_dyn)
                gup=self.ffn.fp8_gemm_bf16(ffn_fp8, w.gate_up_w_fp8[i], in_dyn, w.gate_up_w_scale[i])
                g,u=gup.chunk(2,dim=-1); hid=(F.gelu(g,approximate='tanh')*u).contiguous()
                hid_dyn=self._sc(hid.float().abs().max())
                hf=self.gemm.channel_scale_quantize_fp8_static_bf16(hid, oH, hid_dyn)
                fo=self.ffn.fp8_gemm_bf16(hf, w.down_w_fp8[i], hid_dyn, w.down_w_scale[i])
                self.residual.gate_residual_bf16(self.x, fo, self.ffn_gate, out=self.x)
            self.adapt.ada_rms_norm_style_bf16(self.x, w.ones, w.style_final[step], out=self.final_normed, gate_out=self.final_gate)
            self.gemm.bf16_linear_bias_bf16(self.final_normed, w.action_out_w, w.action_out_b, out=self.action)
            self.residual.gate_residual_bf16(self.noise, self.action, self.noise_gate, out=self.noise)
        return self.noise

parser = argparse.ArgumentParser(description="Run PI0.5 decoder-loop static-vs-dynamic FP8 FFN benchmark.")
parser.add_argument("--encoder-kv-bundle", default=str(ROOT / "internal-tests/runtime-demo/pi05-real-encoder-kv-frame50.pt"))
parser.add_argument(
    "--checkpoint",
    default=None,
    help="Path to pi05_libero_pytorch checkpoint. Defaults to $PI05_CHECKPOINT or ../checkpoints/pi05_libero_pytorch.",
)
parser.add_argument("--calibration-input", default=str(ROOT / "internal-tests/runtime-demo/pi05-decoder-loop-hub-static-scales.json"))
parser.add_argument("--warmup", type=int, default=8)
parser.add_argument("--iters", type=int, default=30)
args = parser.parse_args()
B=args.encoder_kv_bundle
CK=args.checkpoint or os.environ.get("PI05_CHECKPOINT") or str(ROOT.parent / "checkpoints/pi05_libero_pytorch")
CAL=args.calibration_input
dev=torch.device("cuda")
bundle=torch.load(B, map_location="cpu", weights_only=False)
ck=dec._resolve_weight_path(CK) if hasattr(dec,"_resolve_weight_path") else Path(CK)/"model.safetensors"
L=int(bundle["layers"]); S=int(bundle["steps"])
w=dec.DecoderWeights(ck, layers=L, steps=S, device=dev)
st=dec.DecoderState.__new__(dec.DecoderState)
st.noise0=bundle["noise0"].to(dev,torch.bfloat16).contiguous()
st.encoder_k=bundle["encoder_k"][:L].to(dev,torch.bfloat16).contiguous(); st.encoder_v=bundle["encoder_v"][:L].to(dev,torch.bfloat16).contiguous()
st.chunk_size=int(bundle["chunk_size"]); st.encoder_seq_len=int(bundle["encoder_seq_len"]); st.rope=dec._make_rope(st.chunk_size,st.encoder_seq_len,dev)
calp=Path(CAL)
ref=dec.TorchDecoderReference(w, st, scale_safety=1.0, calibration_input=(calp if calp.exists() else None))
def build(cls): return cls(w, st, ref, local_gemm_artifact=None, local_qkv_artifact=None, local_ffn_artifact=None, local_residual_artifact=None, attention_backend="fa2")
def tm(fn,it=args.iters):
    for _ in range(args.warmup): fn()
    torch.cuda.synchronize(); s=time.time()
    for _ in range(it): fn()
    torch.cuda.synchronize(); return (time.time()-s)/it*1e3
def cos(a,b): return F.cosine_similarity(a.float().flatten(),b.float().flatten(),dim=0).item()
expected=ref()
stat=build(dec.HubDecoderLoop); a_s=stat(); l_s=tm(stat)
dyn=build(DynDecoder); a_d=dyn(); l_d=tm(dyn)
gs=dec.Captured(stat); lgs=tm(gs.replay)
gd=dec.Captured(dyn); lgd=tm(gd.replay)
print("\n=== DECODER (complete FP8 FFN, real KV) STATIC vs DYNAMIC ===")
print(f"  action cos vs ref:  STATIC {cos(a_s,expected):.5f} | DYNAMIC {cos(a_d,expected):.5f}")
print(f"  eager  ms:  STATIC {l_s:.2f} | DYNAMIC {l_d:.2f}  ({l_d/l_s:.2f}x)")
print(f"  graph  ms:  STATIC {lgs:.2f} | DYNAMIC {lgd:.2f}  ({lgd/lgs:.2f}x)")
