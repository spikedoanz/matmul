# Factors
clock_speed = 4.8e9  # 4.8 GHz in Hz
num_cores = 12
simd_factor = 4  # AVX2 for double-precision
fma_factor = 2
superscalar_factor = 6

theoretical_max_flops = clock_speed * num_cores * simd_factor * fma_factor * superscalar_factor

print(f"Theoretical Maximum FLOPS: {theoretical_max_flops:.2e} FLOPS, {theoretical_max_flops/1e12:.2f} TFLOPS")
