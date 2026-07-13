from kernels import get_kernel


int4 = get_kernel("flashrt/int4-blackwell", version=2)

print("NVFP4 E2M1:", int4.codebook_probe("e2m1"))
print("uniform INT4:", int4.codebook_probe("ab"))
