import argparse, os, sys
from pathlib import Path

import torch, torch.nn.functional as F
import pi05_decoder_loop_hub as dec

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
FP8=448.0
class DynDecoder(dec.HubDecoderLoop):
    def _sc(self,a): return torch.clamp(a/FP8,min=1e-12).reshape(1).to(self.w.device,torch.float32)
    def __call__(self):
        w=self.w; s=self.s; self.noise.copy_(s.noise0)
        H=w.down_w_fp8[0].shape[1]; oH=torch.ones(H,device=w.device,dtype=torch.bfloat16)
        for step in range(w.steps):
            self.gemm.bf16_linear_bias_bf16(self.noise,w.action_in_w,w.action_in_b,out=self.x)
            for i in range(w.layers):
                self.adapt.ada_rms_norm_style_bf16(self.x,w.ones,w.style_attn[step,i],out=self.normed,gate_out=self.gate)
                self.gemm.bf16_linear_bf16(self.normed,w.qkv_w[i],out=self.qkv_buf)
                self.k_cache[:,:s.encoder_seq_len].copy_(dec._layer_kv(s.encoder_k,i,name="encoder_k"))
                self.v_cache[:,:s.encoder_seq_len].copy_(dec._layer_kv(s.encoder_v,i,name="encoder_v"))
                self.qkv.qkv_split_rope_kvcache_bf16(self.qkv_buf.view(1,s.chunk_size,-1),s.rope,dec.DEC_NH,dec.DEC_NKV,dec.DEC_HD,s.encoder_seq_len,self.q,self.k_cache,self.v_cache)
                if self.attention_backend=="fa2": self.attn_kernel.fwd(self.q,self.k_cache,self.v_cache,out=self.attn_bthd,p_dropout=0.0,is_causal=False)
                else: self.attn.copy_(dec._sdpa_gqa(self.q,self.k_cache,self.v_cache))
                self.gemm.bf16_linear_bf16(self.attn,w.o_w[i],out=self.attn_o)
                self.residual.gate_residual_bf16(self.x,self.attn_o,self.gate,out=self.x)
                self.adapt.ada_rms_norm_style_bf16(self.x,w.ones,w.style_ffn[step,i],out=self.normed,gate_out=self.ffn_gate)
                in_dyn=self._sc(self.normed.float().abs().max())
                ffn_fp8=self.gemm.channel_scale_quantize_fp8_static_bf16(self.normed,w.ones,in_dyn)
                gup=self.ffn.fp8_gemm_bf16(ffn_fp8,w.gate_up_w_fp8[i],in_dyn,w.gate_up_w_scale[i])
                g,u=gup.chunk(2,dim=-1); hid=(F.gelu(g,approximate='tanh')*u).contiguous()
                hid_dyn=self._sc(hid.float().abs().max())
                hf=self.gemm.channel_scale_quantize_fp8_static_bf16(hid,oH,hid_dyn)
                fo=self.ffn.fp8_gemm_bf16(hf,w.down_w_fp8[i],hid_dyn,w.down_w_scale[i])
                self.residual.gate_residual_bf16(self.x,fo,self.ffn_gate,out=self.x)
            self.adapt.ada_rms_norm_style_bf16(self.x,w.ones,w.style_final[step],out=self.final_normed,gate_out=self.final_gate)
            self.gemm.bf16_linear_bias_bf16(self.final_normed,w.action_out_w,w.action_out_b,out=self.action)
            self.residual.gate_residual_bf16(self.noise,self.action,self.noise_gate,out=self.noise)
        return self.noise
parser = argparse.ArgumentParser(description="Run full e2e with only decoder FFN made dynamic.")
parser.add_argument("mode", choices=("static", "dynamic"))
parser.add_argument(
    "--checkpoint",
    default=None,
    help="Path to pi05_libero_pytorch checkpoint. Defaults to $PI05_CHECKPOINT or ../checkpoints/pi05_libero_pytorch.",
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
mode=args.mode
if mode=="dynamic": dec.HubDecoderLoop=DynDecoder
import pi05_hf_decoder_e2e as e2e
checkpoint = args.checkpoint or os.environ.get("PI05_CHECKPOINT") or str(ROOT.parent / "checkpoints/pi05_libero_pytorch")
sys.argv=["e2e","run-vision-encoder-decoder",
  "--encoder-bundle",args.encoder_bundle,
  "--checkpoint",checkpoint,
  "--calibration-input",args.calibration_input,
  "--encoder-calibration-input",args.encoder_calibration_input,
  "--warmup",str(args.warmup),"--iters",str(args.iters),"--no-fp8-projections"]
if not args.no_cuda_graph:
  sys.argv.append("--cuda-graph")
e2e.main()
