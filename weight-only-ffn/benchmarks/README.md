# Benchmark Protocol

The benchmark pre-quantizes static weights outside the timed region and measures
complete FFN calls and standalone model-oriented linear projections. It reports:

- PyTorch eager with exact dequantized weights
- warmed `torch.compile(mode="max-autotune-no-cudagraphs")`
- every diagnostic native variant
- production auto acceptance/rejection, selected latency, and speedup
- max/mean/p99 absolute error and cosine similarity

CUDA events bracket warmed steady-state iterations on the current stream. Each
reported latency is the median of three measurement rounds. Compilation and
weight preparation are excluded.

An accepted auto dispatch must be within 5% of the fastest diagnostic variant
and at least 2% faster than the stronger eager/compile baseline.
Diagnostic-only timings are not public production claims.
