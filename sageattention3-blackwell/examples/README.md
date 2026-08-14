# Examples

`basic_usage.py` shows allocation before CUDA Graph capture and an allocation-
free FP4 attention call. Inputs must already satisfy SageAttention3 centering
and 128-token padding; see the package README for the preprocessing contract.
