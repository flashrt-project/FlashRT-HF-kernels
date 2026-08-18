// Restricted Marlin selector for FlashRT's BF16 x NVFP4 W4A16 path.
// The package deliberately instantiates only the M <= 16 configurations it
// publishes, keeping build time and artifact size bounded.
if (a_type == vllm::kBFloat16 && b_type == vllm::kFE2M1f &&
    c_type == vllm::kBFloat16 && s_type == vllm::kFE4M3fn &&
    threads == 256 && thread_m_blocks == 1 && thread_n_blocks == 8 &&
    thread_k_blocks == 8 && m_block_size_8 == true && stages == 4 &&
    group_blocks == 1 && is_zp_float == false)
  kernel = Marlin<vllm::kBFloat16.id(), vllm::kFE2M1f.id(),
                  vllm::kBFloat16.id(), vllm::kFE4M3fn.id(), 256, 1, 8, 8,
                  true, 4, 1, false>;
else if (a_type == vllm::kBFloat16 && b_type == vllm::kFE2M1f &&
         c_type == vllm::kBFloat16 && s_type == vllm::kFE4M3fn &&
         threads == 128 && thread_m_blocks == 1 && thread_n_blocks == 8 &&
         thread_k_blocks == 4 && m_block_size_8 == true && stages == 4 &&
         group_blocks == 1 && is_zp_float == false)
  kernel = Marlin<vllm::kBFloat16.id(), vllm::kFE2M1f.id(),
                  vllm::kBFloat16.id(), vllm::kFE4M3fn.id(), 128, 1, 8, 4,
                  true, 4, 1, false>;
else if (a_type == vllm::kBFloat16 && b_type == vllm::kFE2M1f &&
         c_type == vllm::kBFloat16 && s_type == vllm::kFE4M3fn &&
         threads == 128 && thread_m_blocks == 1 && thread_n_blocks == 4 &&
         thread_k_blocks == 8 && m_block_size_8 == true && stages == 4 &&
         group_blocks == 1 && is_zp_float == false)
  kernel = Marlin<vllm::kBFloat16.id(), vllm::kFE2M1f.id(),
                  vllm::kBFloat16.id(), vllm::kFE4M3fn.id(), 128, 1, 4, 8,
                  true, 4, 1, false>;
else if (a_type == vllm::kBFloat16 && b_type == vllm::kFE2M1f &&
         c_type == vllm::kBFloat16 && s_type == vllm::kFE4M3fn &&
         threads == 256 && thread_m_blocks == 1 && thread_n_blocks == 8 &&
         thread_k_blocks == 8 && m_block_size_8 == false && stages == 4 &&
         group_blocks == 1 && is_zp_float == false)
  kernel = Marlin<vllm::kBFloat16.id(), vllm::kFE2M1f.id(),
                  vllm::kBFloat16.id(), vllm::kFE4M3fn.id(), 256, 1, 8, 8,
                  false, 4, 1, false>;
else if (a_type == vllm::kBFloat16 && b_type == vllm::kFE2M1f &&
         c_type == vllm::kBFloat16 && s_type == vllm::kFE4M3fn &&
         threads == 128 && thread_m_blocks == 1 && thread_n_blocks == 8 &&
         thread_k_blocks == 4 && m_block_size_8 == false && stages == 4 &&
         group_blocks == 1 && is_zp_float == false)
  kernel = Marlin<vllm::kBFloat16.id(), vllm::kFE2M1f.id(),
                  vllm::kBFloat16.id(), vllm::kFE4M3fn.id(), 128, 1, 8, 4,
                  false, 4, 1, false>;
else if (a_type == vllm::kBFloat16 && b_type == vllm::kFE2M1f &&
         c_type == vllm::kBFloat16 && s_type == vllm::kFE4M3fn &&
         threads == 128 && thread_m_blocks == 1 && thread_n_blocks == 4 &&
         thread_k_blocks == 8 && m_block_size_8 == false && stages == 4 &&
         group_blocks == 1 && is_zp_float == false)
  kernel = Marlin<vllm::kBFloat16.id(), vllm::kFE2M1f.id(),
                  vllm::kBFloat16.id(), vllm::kFE4M3fn.id(), 128, 1, 4, 8,
                  false, 4, 1, false>;
