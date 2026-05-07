.amdgcn_target "amdgcn-amd-amdhsa--gfx950"
.text
.globl Custom_Cijk_Alik_Bljk_F4BS_MXA32_MXB32_MT128x128x256_MI16x16x1_kgen_gfx950
.p2align 8
.type Custom_Cijk_Alik_Bljk_F4BS_MXA32_MXB32_MT128x128x256_MI16x16x1_kgen_gfx950,@function

Custom_Cijk_Alik_Bljk_F4BS_MXA32_MXB32_MT128x128x256_MI16x16x1_kgen_gfx950:
    // === DTL Interleaved Setup ===
    s_load_dword s10, s[0:1], 0                   // M
    s_load_dword s11, s[0:1], 4                   // N
    s_load_dword s12, s[0:1], 12                  // K
    s_load_dwordx2 s[8:9], s[0:1], 16             // D ptr
    s_load_dwordx2 s[4:5], s[0:1], 32             // A ptr
    s_load_dwordx2 s[6:7], s[0:1], 48             // B ptr
    s_waitcnt lgkmcnt(0)                          // wait kernargs

    // WorkGroupMappingXCC=2: remap for L2 locality
    s_add_u32 s46, s10, 127                       // M + 127
    s_lshr_b32 s46, s46, 7                        // ceil(M/128)
    s_add_u32 s14, s11, 127                       // N + 127
    s_lshr_b32 s14, s14, 7                        // ceil(N/128)
    s_mul_i32 s46, s46, s14                       // numWG = tiles_m * tiles_n
    s_lshr_b32 s14, s2, 1                         // old_wg / 2
    s_and_b32 s15, s2, 1                          // old_wg % 2 (XCC lane)
    s_lshr_b32 s46, s46, 1                        // numWG / 2
    s_mul_i32 s15, s15, s46                       // XCC_lane * (numWG / WGMXCC)
    s_add_u32 s2, s14, s15                        // remapped WG serial

    s_mov_b32 s15, s2                             // save wg_serial
    s_add_u32 s14, s10, 127                       // M + 127
    s_lshr_b32 s14, s14, 7                        // numWG_m = ceil(M/128)
    s_sub_u32 s2, s14, 1                          // numWG_m - 1
    s_and_b32 s3, s14, s2                         // numWG_m & (numWG_m-1) == 0 if power-of-2
    s_ff1_i32_b32 s3, s14                         // log2(numWG_m)
    s_lshr_b32 s3, s15, s3                        // tile_n = serial >> log2(numWG_m)
    s_sub_u32 s2, s14, 1                          // numWG_m - 1
    s_and_b32 s2, s15, s2                         // tile_m = serial & (numWG_m - 1)

    v_lshrrev_b32 v1, 6, v0                       // wave_id = tid >> 6
    v_and_b32 v2, 63, v0                          // lane_id = tid & 63
    v_lshrrev_b32 v3, 1, v1                       // wave_m = wave_id >> 1
    v_and_b32 v4, 1, v1                           // wave_n = wave_id & 1

    // DTL offset: 8 threads/row
    v_lshrrev_b32 v29, 3, v0                      // thread_row
    v_and_b32 v30, 7, v0                          // thread_col_group
    v_lshlrev_b32 v30, 4, v30                     // * 16 -> col_bytes
    s_lshr_b32 s26, s12, 1                        // s_k_stride = K * 0.5
    // Swizzled LDS write offset (pair=2, eff_cols=16)
    v_lshrrev_b32 v33, 4, v30                     // thread_col = col_bytes / 16
    v_and_b32 v32, 1, v29                         // m_row % 2
    v_lshlrev_b32 v32, 3, v32                     // * 8 -> half_offset
    v_lshrrev_b32 v31, 1, v29                     // lds_row = m_row / 2
    v_add_u32 v33, v32, v33                       // half_offset + col
    v_add_u32 v33, v33, v31                       // + lds_row (rotation)
    v_and_b32 v33, 15, v33                        // % 16
    v_lshrrev_b32 v32, 1, v29                     // lds_row = row / 2
    v_lshlrev_b32 v39, 8, v32                     // lds_row * 256
    v_lshlrev_b32 v33, 4, v33                     // swizzled_col * 16
    v_add_u32 v39, v39, v33                       // + swizzled_col_bytes

    v_lshlrev_b32 v40, 7, v29                     // row * 128 (LDS stride)
    v_add_u32 v40, v40, v30                       // + col_bytes -> v_lds_wr_off

    s_mov_b32 s47, 0                              // buf_wr_db = 0 (buffer 0)

    v_mul_lo_u32 v5, s26, v29                     // row * K*elem
    v_add_u32 v5, v5, v30                         // + col_bytes
    v_mov_b32 v6, v5                              // B offset = same

    // SRD A
    s_mul_i32 s14, s2, 128                        // wg_id * 128
    s_mul_i32 s14, s14, s26                       // * K*elem
    s_add_u32 s16, s4, s14                        // SRD_A lo
    s_addc_u32 s17, s5, 0                         // SRD_A hi
    s_mov_b32 s18, 0xFFFFFFFF                     // limit
    s_mov_b32 s19, 0x20000                        // flags

    // SRD B
    s_mul_i32 s14, s3, 128                        // wg_id * 128
    s_mul_i32 s14, s14, s26                       // * K*elem
    s_add_u32 s20, s6, s14                        // SRD_B lo
    s_addc_u32 s21, s7, 0                         // SRD_B hi
    s_mov_b32 s22, 0xFFFFFFFF                     // limit
    s_mov_b32 s23, 0x20000                        // flags

    // Scalar offset for DTL lines (32 rows/load)
    s_mul_i32 s27, s26, 32                        // soffset = 32 * K*elem
    s_mov_b32 s28, s27                            // same

    // LDS write base for DTL
    v_lshlrev_b32 v29, 7, v29                     // row * 128
    v_add_u32 v29, v29, v30                       // + col_bytes -> per-thread LDS offset
    v_readfirstlane_b32 s24, v29                  // LDS write base A
    s_add_u32 s25, s24, 16384                     // LDS write base B = A + 16384

    // LDS read addresses
    v_and_b32 v29, 15, v2                         // lane_row = lane_id % 16
    v_lshrrev_b32 v30, 4, v2                      // k_group = lane_id / 16
    // LDS read A (paired-row swizzle)
    v_lshlrev_b32 v7, 6, v3                       // wave_m * 64
    v_add_u32 v7, v7, v29                         // + lane_row -> m_row_a
    v_lshrrev_b32 v31, 1, v7                      // lds_row = m_row / 2
    v_lshlrev_b32 v31, 8, v31                     // row_base = lds_row * 256
    v_mov_b32 v35, v30                            // k_col = k_group
    v_and_b32 v37, 1, v7                          // m_row % 2
    v_lshlrev_b32 v37, 3, v37                     // * 8 -> half_offset
    v_lshrrev_b32 v38, 1, v7                      // lds_row = m_row / 2
    v_add_u32 v35, v37, v35                       // half_offset + k_col
    v_add_u32 v35, v35, v38                       // + lds_row (rotation)
    v_and_b32 v35, 15, v35                        // % 16
    v_lshlrev_b32 v7, 4, v35                      // swizzled_col * 16
    v_add_u32 v7, v7, v31                         // + row_base
    v_add_u32 v38, v35, 4                         // k_col + 4 (ki=1)
    v_and_b32 v38, 15, v38                        // % 16
    v_lshlrev_b32 v9, 4, v38                      // swizzled_col * 16
    v_add_u32 v9, v9, v31                         // + row_base
    v_lshlrev_b32 v41, 6, v3                      // m_row_base = wave_m * 64
    v_add_u32 v41, v41, v29                       // + lane_row (persistent)
    v_mov_b32 v42, v30                            // k_group (persistent)

    // LDS read B (paired-row swizzle)
    v_and_b32 v29, 15, v2                         // lane_row = lane_id % 16
    v_lshlrev_b32 v8, 6, v4                       // wave_n * 64
    v_add_u32 v8, v8, v29                         // + lane_row -> n_row_b
    v_lshrrev_b32 v31, 1, v8                      // lds_row = n_row / 2
    v_lshlrev_b32 v31, 8, v31                     // row_base = lds_row * 256
    s_mov_b32 s14, 16384                          // lds_b_off=16384
    v_add_u32 v31, v31, s14                       // + lds_b_offset=16384
    v_mov_b32 v35, v30                            // k_col = k_group
    v_and_b32 v37, 1, v8                          // m_row % 2
    v_lshlrev_b32 v37, 3, v37                     // * 8 -> half_offset
    v_lshrrev_b32 v38, 1, v8                      // lds_row = m_row / 2
    v_add_u32 v35, v37, v35                       // half_offset + k_col
    v_add_u32 v35, v35, v38                       // + lds_row (rotation)
    v_and_b32 v35, 15, v35                        // % 16
    v_lshlrev_b32 v8, 4, v35                      // swizzled_col * 16
    v_add_u32 v8, v8, v31                         // + row_base
    v_add_u32 v38, v35, 4                         // k_col + 4 (ki=1)
    v_and_b32 v38, 15, v38                        // % 16
    v_lshlrev_b32 v10, 4, v38                     // swizzled_col * 16
    v_add_u32 v10, v10, v31                       // + row_base
    v_lshlrev_b32 v43, 6, v4                      // n_row_base = wave_n * 64
    v_and_b32 v29, 15, v2                         // lane_row (re-derive for save)
    v_add_u32 v43, v43, v29                       // + lane_row (persistent)


    // Init 64 accumulators
    v_accvgpr_write_b32 acc0, 0
    v_accvgpr_write_b32 acc1, 0
    v_accvgpr_write_b32 acc2, 0
    v_accvgpr_write_b32 acc3, 0
    v_accvgpr_write_b32 acc4, 0
    v_accvgpr_write_b32 acc5, 0
    v_accvgpr_write_b32 acc6, 0
    v_accvgpr_write_b32 acc7, 0
    v_accvgpr_write_b32 acc8, 0
    v_accvgpr_write_b32 acc9, 0
    v_accvgpr_write_b32 acc10, 0
    v_accvgpr_write_b32 acc11, 0
    v_accvgpr_write_b32 acc12, 0
    v_accvgpr_write_b32 acc13, 0
    v_accvgpr_write_b32 acc14, 0
    v_accvgpr_write_b32 acc15, 0
    v_accvgpr_write_b32 acc16, 0
    v_accvgpr_write_b32 acc17, 0
    v_accvgpr_write_b32 acc18, 0
    v_accvgpr_write_b32 acc19, 0
    v_accvgpr_write_b32 acc20, 0
    v_accvgpr_write_b32 acc21, 0
    v_accvgpr_write_b32 acc22, 0
    v_accvgpr_write_b32 acc23, 0
    v_accvgpr_write_b32 acc24, 0
    v_accvgpr_write_b32 acc25, 0
    v_accvgpr_write_b32 acc26, 0
    v_accvgpr_write_b32 acc27, 0
    v_accvgpr_write_b32 acc28, 0
    v_accvgpr_write_b32 acc29, 0
    v_accvgpr_write_b32 acc30, 0
    v_accvgpr_write_b32 acc31, 0
    v_accvgpr_write_b32 acc32, 0
    v_accvgpr_write_b32 acc33, 0
    v_accvgpr_write_b32 acc34, 0
    v_accvgpr_write_b32 acc35, 0
    v_accvgpr_write_b32 acc36, 0
    v_accvgpr_write_b32 acc37, 0
    v_accvgpr_write_b32 acc38, 0
    v_accvgpr_write_b32 acc39, 0
    v_accvgpr_write_b32 acc40, 0
    v_accvgpr_write_b32 acc41, 0
    v_accvgpr_write_b32 acc42, 0
    v_accvgpr_write_b32 acc43, 0
    v_accvgpr_write_b32 acc44, 0
    v_accvgpr_write_b32 acc45, 0
    v_accvgpr_write_b32 acc46, 0
    v_accvgpr_write_b32 acc47, 0
    v_accvgpr_write_b32 acc48, 0
    v_accvgpr_write_b32 acc49, 0
    v_accvgpr_write_b32 acc50, 0
    v_accvgpr_write_b32 acc51, 0
    v_accvgpr_write_b32 acc52, 0
    v_accvgpr_write_b32 acc53, 0
    v_accvgpr_write_b32 acc54, 0
    v_accvgpr_write_b32 acc55, 0
    v_accvgpr_write_b32 acc56, 0
    v_accvgpr_write_b32 acc57, 0
    v_accvgpr_write_b32 acc58, 0
    v_accvgpr_write_b32 acc59, 0
    v_accvgpr_write_b32 acc60, 0
    v_accvgpr_write_b32 acc61, 0
    v_accvgpr_write_b32 acc62, 0
    v_accvgpr_write_b32 acc63, 0

    // Init MX constant scale = 1.0 (E8M0 0x7F)
    v_mov_b32 v20, 0x7F7F7F7F                     // scale = 1.0 for all byte lanes

    // === MX Scale Setup (direct VGPR, no LDS) ===
    s_load_dwordx2 s[30:31], s[0:1], 40           // scale A ptr (MXSA)
    s_load_dwordx2 s[32:33], s[0:1], 56           // scale B ptr (MXSB)
    s_load_dword s34, s[0:1], 88                  // strideMXSA0
    s_load_dword s35, s[0:1], 104                 // strideMXSB0
    s_waitcnt lgkmcnt(0)                          // wait scale kernargs

    // Scale SRD A
    s_mul_i32 s14, s2, 4                          // wg_id_x * 4 (MT/32)
    s_mul_i32 s14, s14, s34                       // * stride_scale_a
    s_add_u32 s36, s30, s14                       // SRD_scaleA lo
    s_addc_u32 s37, s31, 0                        // SRD_scaleA hi
    s_mov_b32 s38, 0xFFFFFFFF                     // limit
    s_mov_b32 s39, 0x20000                        // flags

    // Scale SRD B
    s_mul_i32 s14, s3, 4                          // wg_id_y * 4 (MT/32)
    s_mul_i32 s14, s14, s35                       // * stride_scale_b
    s_add_u32 s40, s32, s14                       // SRD_scaleB lo
    s_addc_u32 s41, s33, 0                        // SRD_scaleB hi
    s_mov_b32 s42, 0xFFFFFFFF                     // limit
    s_mov_b32 s43, 0x20000                        // flags

    // Scale A wave-level voffset
    v_lshlrev_b32 v29, 6, v3                      // wave_m * 64
    v_mul_lo_u32 v21, s34, v29                    // wave_m_base * stride_scale_a -> voffset_scale_a

    // Scale B wave-level voffset
    v_lshlrev_b32 v29, 6, v4                      // wave_n * 64
    v_mul_lo_u32 v22, s35, v29                    // wave_n_base * stride_scale_b -> voffset_scale_b

    // Precompute DTL soffsets
    s_mul_i32 s48, s27, 1                         // dtl_soff_a[1] = 1 * soffset_a
    s_mul_i32 s49, s27, 2                         // dtl_soff_a[2] = 2 * soffset_a
    s_mul_i32 s50, s27, 3                         // dtl_soff_a[3] = 3 * soffset_a
    s_mul_i32 s51, s28, 1                         // dtl_soff_b[1] = 1 * soffset_b
    s_mul_i32 s52, s28, 2                         // dtl_soff_b[2] = 2 * soffset_b
    s_mul_i32 s53, s28, 3                         // dtl_soff_b[3] = 3 * soffset_b

    // Scale A per-lane voffset
    v_mul_lo_u32 v29, 64, v3                      // wave_m * 64
    v_and_b32 v30, v2, 15                         // lane_id & 15 (M-row within MFMA tile)
    v_add_u32 v29, v29, v30                       // M-row relative to wave's start
    v_mul_lo_u32 v124, s34, v29                   // * stride_scale_a -> byte offset

    // Scale B per-lane voffset
    v_mul_lo_u32 v29, 64, v4                      // wave_n * 64
    v_and_b32 v30, v2, 15                         // lane_id & 15 (N-row within MFMA tile)
    v_add_u32 v29, v29, v30                       // N-row relative to wave's start
    v_mul_lo_u32 v125, s35, v29                   // * stride_scale_b -> byte offset

    // Precompute scale soffsets
    s_mul_i32 s54, s34, 16                        // soff_a[1] = stride * 16
    s_mul_i32 s55, s34, 32                        // soff_a[2] = stride * 32
    s_mul_i32 s56, s34, 48                        // soff_a[3] = stride * 48
    s_mul_i32 s57, s35, 16                        // soff_b[1] = stride * 16
    s_mul_i32 s58, s35, 32                        // soff_b[2] = stride * 32
    s_mul_i32 s59, s35, 48                        // soff_b[3] = stride * 48

    // Precompute swizzled A read addresses
    v_mov_b32 v31, v41                            // m_row = base (mi=0)
    v_lshrrev_b32 v32, 1, v31                     // lds_row = m_row / 2
    v_lshlrev_b32 v32, 8, v32                     // row_base = lds_row * 256
    v_mov_b32 v35, v42                            // k_col = k_group
    v_and_b32 v37, 1, v31                         // m_row % 2
    v_lshlrev_b32 v37, 3, v37                     // * 8 -> half_offset
    v_lshrrev_b32 v38, 1, v31                     // lds_row = m_row / 2
    v_add_u32 v35, v37, v35                       // half_offset + k_col
    v_add_u32 v35, v35, v38                       // + lds_row (rotation)
    v_and_b32 v35, 15, v35                        // % 16
    v_lshlrev_b32 v44, 4, v35                     // swizzled_col * 16
    v_add_u32 v44, v44, v32                       // + row_base
    v_add_u32 v38, v35, 4                         // k_col + 4 (ki=1)
    v_and_b32 v38, 15, v38                        // % 16
    v_lshlrev_b32 v45, 4, v38                     // swizzled_col * 16
    v_add_u32 v45, v45, v32                       // + row_base
    v_add_u32 v31, v41, 16                        // m_row = base + 16 (mi=1)
    v_lshrrev_b32 v32, 1, v31                     // lds_row = m_row / 2
    v_lshlrev_b32 v32, 8, v32                     // row_base = lds_row * 256
    v_mov_b32 v35, v42                            // k_col = k_group
    v_and_b32 v37, 1, v31                         // m_row % 2
    v_lshlrev_b32 v37, 3, v37                     // * 8 -> half_offset
    v_lshrrev_b32 v38, 1, v31                     // lds_row = m_row / 2
    v_add_u32 v35, v37, v35                       // half_offset + k_col
    v_add_u32 v35, v35, v38                       // + lds_row (rotation)
    v_and_b32 v35, 15, v35                        // % 16
    v_lshlrev_b32 v46, 4, v35                     // swizzled_col * 16
    v_add_u32 v46, v46, v32                       // + row_base
    v_add_u32 v38, v35, 4                         // k_col + 4 (ki=1)
    v_and_b32 v38, 15, v38                        // % 16
    v_lshlrev_b32 v47, 4, v38                     // swizzled_col * 16
    v_add_u32 v47, v47, v32                       // + row_base
    v_add_u32 v31, v41, 32                        // m_row = base + 32 (mi=2)
    v_lshrrev_b32 v32, 1, v31                     // lds_row = m_row / 2
    v_lshlrev_b32 v32, 8, v32                     // row_base = lds_row * 256
    v_mov_b32 v35, v42                            // k_col = k_group
    v_and_b32 v37, 1, v31                         // m_row % 2
    v_lshlrev_b32 v37, 3, v37                     // * 8 -> half_offset
    v_lshrrev_b32 v38, 1, v31                     // lds_row = m_row / 2
    v_add_u32 v35, v37, v35                       // half_offset + k_col
    v_add_u32 v35, v35, v38                       // + lds_row (rotation)
    v_and_b32 v35, 15, v35                        // % 16
    v_lshlrev_b32 v48, 4, v35                     // swizzled_col * 16
    v_add_u32 v48, v48, v32                       // + row_base
    v_add_u32 v38, v35, 4                         // k_col + 4 (ki=1)
    v_and_b32 v38, 15, v38                        // % 16
    v_lshlrev_b32 v49, 4, v38                     // swizzled_col * 16
    v_add_u32 v49, v49, v32                       // + row_base
    v_add_u32 v31, v41, 48                        // m_row = base + 48 (mi=3)
    v_lshrrev_b32 v32, 1, v31                     // lds_row = m_row / 2
    v_lshlrev_b32 v32, 8, v32                     // row_base = lds_row * 256
    v_mov_b32 v35, v42                            // k_col = k_group
    v_and_b32 v37, 1, v31                         // m_row % 2
    v_lshlrev_b32 v37, 3, v37                     // * 8 -> half_offset
    v_lshrrev_b32 v38, 1, v31                     // lds_row = m_row / 2
    v_add_u32 v35, v37, v35                       // half_offset + k_col
    v_add_u32 v35, v35, v38                       // + lds_row (rotation)
    v_and_b32 v35, 15, v35                        // % 16
    v_lshlrev_b32 v50, 4, v35                     // swizzled_col * 16
    v_add_u32 v50, v50, v32                       // + row_base
    v_add_u32 v38, v35, 4                         // k_col + 4 (ki=1)
    v_and_b32 v38, 15, v38                        // % 16
    v_lshlrev_b32 v51, 4, v38                     // swizzled_col * 16
    v_add_u32 v51, v51, v32                       // + row_base

    // Precompute swizzled B read addresses
    v_mov_b32 v31, v43                            // n_row = base (ni=0)
    v_lshrrev_b32 v32, 1, v31                     // lds_row = n_row / 2
    v_lshlrev_b32 v32, 8, v32                     // row_base = lds_row * 256
    s_mov_b32 s14, 16384                          // lds_b_off=16384
    v_add_u32 v32, v32, s14                       // + lds_b_offset
    v_mov_b32 v35, v42                            // k_col = k_group
    v_and_b32 v37, 1, v31                         // m_row % 2
    v_lshlrev_b32 v37, 3, v37                     // * 8 -> half_offset
    v_lshrrev_b32 v38, 1, v31                     // lds_row = m_row / 2
    v_add_u32 v35, v37, v35                       // half_offset + k_col
    v_add_u32 v35, v35, v38                       // + lds_row (rotation)
    v_and_b32 v35, 15, v35                        // % 16
    v_lshlrev_b32 v52, 4, v35                     // swizzled_col * 16
    v_add_u32 v52, v52, v32                       // + row_base
    v_add_u32 v38, v35, 4                         // k_col + 4 (ki=1)
    v_and_b32 v38, 15, v38                        // % 16
    v_lshlrev_b32 v53, 4, v38                     // swizzled_col * 16
    v_add_u32 v53, v53, v32                       // + row_base
    v_add_u32 v31, v43, 16                        // n_row = base + 16 (ni=1)
    v_lshrrev_b32 v32, 1, v31                     // lds_row = n_row / 2
    v_lshlrev_b32 v32, 8, v32                     // row_base = lds_row * 256
    s_mov_b32 s14, 16384                          // lds_b_off=16384
    v_add_u32 v32, v32, s14                       // + lds_b_offset
    v_mov_b32 v35, v42                            // k_col = k_group
    v_and_b32 v37, 1, v31                         // m_row % 2
    v_lshlrev_b32 v37, 3, v37                     // * 8 -> half_offset
    v_lshrrev_b32 v38, 1, v31                     // lds_row = m_row / 2
    v_add_u32 v35, v37, v35                       // half_offset + k_col
    v_add_u32 v35, v35, v38                       // + lds_row (rotation)
    v_and_b32 v35, 15, v35                        // % 16
    v_lshlrev_b32 v54, 4, v35                     // swizzled_col * 16
    v_add_u32 v54, v54, v32                       // + row_base
    v_add_u32 v38, v35, 4                         // k_col + 4 (ki=1)
    v_and_b32 v38, 15, v38                        // % 16
    v_lshlrev_b32 v55, 4, v38                     // swizzled_col * 16
    v_add_u32 v55, v55, v32                       // + row_base
    v_add_u32 v31, v43, 32                        // n_row = base + 32 (ni=2)
    v_lshrrev_b32 v32, 1, v31                     // lds_row = n_row / 2
    v_lshlrev_b32 v32, 8, v32                     // row_base = lds_row * 256
    s_mov_b32 s14, 16384                          // lds_b_off=16384
    v_add_u32 v32, v32, s14                       // + lds_b_offset
    v_mov_b32 v35, v42                            // k_col = k_group
    v_and_b32 v37, 1, v31                         // m_row % 2
    v_lshlrev_b32 v37, 3, v37                     // * 8 -> half_offset
    v_lshrrev_b32 v38, 1, v31                     // lds_row = m_row / 2
    v_add_u32 v35, v37, v35                       // half_offset + k_col
    v_add_u32 v35, v35, v38                       // + lds_row (rotation)
    v_and_b32 v35, 15, v35                        // % 16
    v_lshlrev_b32 v56, 4, v35                     // swizzled_col * 16
    v_add_u32 v56, v56, v32                       // + row_base
    v_add_u32 v38, v35, 4                         // k_col + 4 (ki=1)
    v_and_b32 v38, 15, v38                        // % 16
    v_lshlrev_b32 v57, 4, v38                     // swizzled_col * 16
    v_add_u32 v57, v57, v32                       // + row_base
    v_add_u32 v31, v43, 48                        // n_row = base + 48 (ni=3)
    v_lshrrev_b32 v32, 1, v31                     // lds_row = n_row / 2
    v_lshlrev_b32 v32, 8, v32                     // row_base = lds_row * 256
    s_mov_b32 s14, 16384                          // lds_b_off=16384
    v_add_u32 v32, v32, s14                       // + lds_b_offset
    v_mov_b32 v35, v42                            // k_col = k_group
    v_and_b32 v37, 1, v31                         // m_row % 2
    v_lshlrev_b32 v37, 3, v37                     // * 8 -> half_offset
    v_lshrrev_b32 v38, 1, v31                     // lds_row = m_row / 2
    v_add_u32 v35, v37, v35                       // half_offset + k_col
    v_add_u32 v35, v35, v38                       // + lds_row (rotation)
    v_and_b32 v35, 15, v35                        // % 16
    v_lshlrev_b32 v58, 4, v35                     // swizzled_col * 16
    v_add_u32 v58, v58, v32                       // + row_base
    v_add_u32 v38, v35, 4                         // k_col + 4 (ki=1)
    v_and_b32 v38, 15, v38                        // % 16
    v_lshlrev_b32 v59, 4, v38                     // swizzled_col * 16
    v_add_u32 v59, v59, v32                       // + row_base

    s_mov_b32 s61, 0                              // rd_db = 0
    s_mov_b32 s60, 32768                          // DB step = 32768

    s_lshr_b32 s13, s12, 8                        // k_tiles = K / 256

    // Pipeline ramp-up stage 0/2
    s_mov_b32 m0, s24                             // m0 = LDS base A
    buffer_load_dwordx4 v5, s[16:19], 0, offen offset:0, lds// DTL A[0]
    s_add_u32 m0, m0, 4096                        // m0 += 4096
    buffer_load_dwordx4 v5, s[16:19], s48, offen offset:0, lds// DTL A[1]
    s_add_u32 m0, m0, 4096                        // m0 += 4096
    buffer_load_dwordx4 v5, s[16:19], s49, offen offset:0, lds// DTL A[2]
    s_add_u32 m0, m0, 4096                        // m0 += 4096
    buffer_load_dwordx4 v5, s[16:19], s50, offen offset:0, lds// DTL A[3]
    s_mov_b32 m0, s25                             // m0 = LDS base B
    buffer_load_dwordx4 v6, s[20:23], 0, offen offset:0, lds// DTL B[0]
    s_add_u32 m0, m0, 4096                        // m0 += 4096
    buffer_load_dwordx4 v6, s[20:23], s51, offen offset:0, lds// DTL B[1]
    s_add_u32 m0, m0, 4096                        // m0 += 4096
    buffer_load_dwordx4 v6, s[20:23], s52, offen offset:0, lds// DTL B[2]
    s_add_u32 m0, m0, 4096                        // m0 += 4096
    buffer_load_dwordx4 v6, s[20:23], s53, offen offset:0, lds// DTL B[3]
    s_waitcnt vmcnt(0)                            // wait all loads
    s_barrier                                     // sync tile 0

    // Pipeline ramp-up stage 1/2
    s_cmp_le_u32 s13, 1                           // skip if k_tiles <= 1
    s_cbranch_scc1 pgr_skip_1                     // not enough tiles for stage 1
    s_add_u32 s16, s16, 128                       // s_srd_a += 128
    s_addc_u32 s17, s17, 0                        // carry
    s_add_u32 s24, s24, s60                       // s_lds_wr_a_sg += db
    s_mov_b32 m0, s24                             // m0 = LDS base A
    buffer_load_dwordx4 v5, s[16:19], 0, offen offset:0, lds// DTL A[0]
    s_add_u32 m0, m0, 4096                        // m0 += 4096
    buffer_load_dwordx4 v5, s[16:19], s48, offen offset:0, lds// DTL A[1]
    s_add_u32 m0, m0, 4096                        // m0 += 4096
    buffer_load_dwordx4 v5, s[16:19], s49, offen offset:0, lds// DTL A[2]
    s_add_u32 m0, m0, 4096                        // m0 += 4096
    buffer_load_dwordx4 v5, s[16:19], s50, offen offset:0, lds// DTL A[3]
    s_add_u32 s20, s20, 128                       // s_srd_b += 128
    s_addc_u32 s21, s21, 0                        // carry
    s_add_u32 s25, s25, s60                       // s_lds_wr_b_sg += db
    s_mov_b32 m0, s25                             // m0 = LDS base B
    buffer_load_dwordx4 v6, s[20:23], 0, offen offset:0, lds// DTL B[0]
    s_add_u32 m0, m0, 4096                        // m0 += 4096
    buffer_load_dwordx4 v6, s[20:23], s51, offen offset:0, lds// DTL B[1]
    s_add_u32 m0, m0, 4096                        // m0 += 4096
    buffer_load_dwordx4 v6, s[20:23], s52, offen offset:0, lds// DTL B[2]
    s_add_u32 m0, m0, 4096                        // m0 += 4096
    buffer_load_dwordx4 v6, s[20:23], s53, offen offset:0, lds// DTL B[3]
pgr_skip_1:

k_loop:

    s_waitcnt vmcnt(0)                            // wait DTL loads from prev iter
    s_waitcnt lgkmcnt(0)                          // wait LDS writes from prev iter
    s_barrier                                     // sync workgroup
    ds_read_b128 v[92:95], v44                    // LR A m0k0 b0
    ds_read_b128 v[60:63], v52                    // LR B n0k0
    buffer_load_dword v108, v124, s[36:39], 0, offen// scale A m0 k0
    buffer_load_dword v110, v124, s[36:39], s54, offen// scale A m1 k0
    buffer_load_dword v109, v124, s[36:39], 0, offen offset:4// scale A m0 k1
    buffer_load_dword v111, v124, s[36:39], s54, offen offset:4// scale A m1 k1
    buffer_load_dword v116, v125, s[40:43], 0, offen// scale B n0 k0
    buffer_load_dword v118, v125, s[40:43], s57, offen// scale B n1 k0
    buffer_load_dword v117, v125, s[40:43], 0, offen offset:4// scale B n0 k1
    buffer_load_dword v119, v125, s[40:43], s57, offen offset:4// scale B n1 k1
    ds_read_b128 v[68:71], v54                    // LR B n1k0
    ds_read_b128 v[76:79], v56                    // LR B n2k0
    buffer_load_dword v120, v125, s[40:43], s58, offen// scale B n2 k0
    buffer_load_dword v122, v125, s[40:43], s59, offen// scale B n3 k0
    buffer_load_dword v121, v125, s[40:43], s58, offen offset:4// scale B n2 k1
    buffer_load_dword v123, v125, s[40:43], s59, offen offset:4// scale B n3 k1
    ds_read_b128 v[84:87], v58                    // LR B n3k0
    ds_read_b128 v[96:99], v45                    // LR A m0k1 b0
    ds_read_b128 v[64:67], v53                    // LR B n0k1
    ds_read_b128 v[72:75], v55                    // LR B n1k1
    ds_read_b128 v[80:83], v57                    // LR B n2k1
    ds_read_b128 v[88:91], v59                    // LR B n3k1
    ds_read_b128 v[100:103], v46                  // LR A m1k0 b1
    ds_read_b128 v[104:107], v47                  // LR A m1k1 b1
    buffer_load_dword v112, v124, s[36:39], s55, offen// scale A m2 k0
    buffer_load_dword v114, v124, s[36:39], s56, offen// scale A m3 k0
    buffer_load_dword v113, v124, s[36:39], s55, offen offset:4// scale A m2 k1
    buffer_load_dword v115, v124, s[36:39], s56, offen offset:4// scale A m3 k1
    s_waitcnt lgkmcnt(12)                         // auto-wait at pos 17
    v_mfma_scale_f32_16x16x128_f8f6f4 acc[0:3], v[92:95], v[60:63], acc[0:3], v108, v116, cbsz:4 blgp:4// MFMA m0_n0_k0
    s_waitcnt lgkmcnt(11)                         // auto-wait at pos 18
    v_mfma_scale_f32_16x16x128_f8f6f4 acc[4:7], v[92:95], v[68:71], acc[4:7], v108, v118, cbsz:4 blgp:4// MFMA m0_n1_k0
    s_waitcnt lgkmcnt(9)                          // auto-wait at pos 19
    v_mfma_scale_f32_16x16x128_f8f6f4 acc[8:11], v[92:95], v[76:79], acc[8:11], v108, v120, cbsz:4 blgp:4// MFMA m0_n2_k0
    s_waitcnt lgkmcnt(8)                          // auto-wait at pos 20
    v_mfma_scale_f32_16x16x128_f8f6f4 acc[12:15], v[92:95], v[84:87], acc[12:15], v108, v122, cbsz:4 blgp:4// MFMA m0_n3_k0
    s_waitcnt lgkmcnt(6)                          // auto-wait at pos 21
    v_mfma_scale_f32_16x16x128_f8f6f4 acc[0:3], v[96:99], v[64:67], acc[0:3], v109, v117, cbsz:4 blgp:4// MFMA m0_n0_k1
    s_waitcnt lgkmcnt(5)                          // auto-wait at pos 22
    v_mfma_scale_f32_16x16x128_f8f6f4 acc[4:7], v[96:99], v[72:75], acc[4:7], v109, v119, cbsz:4 blgp:4// MFMA m0_n1_k1
    s_waitcnt lgkmcnt(4)                          // auto-wait at pos 23
    v_mfma_scale_f32_16x16x128_f8f6f4 acc[8:11], v[96:99], v[80:83], acc[8:11], v109, v121, cbsz:4 blgp:4// MFMA m0_n2_k1
    s_waitcnt lgkmcnt(3)                          // auto-wait at pos 24
    v_mfma_scale_f32_16x16x128_f8f6f4 acc[12:15], v[96:99], v[88:91], acc[12:15], v109, v123, cbsz:4 blgp:4// MFMA m0_n3_k1
    ds_read_b128 v[92:95], v48                    // LR A m2k0 b0
    ds_read_b128 v[96:99], v49                    // LR A m2k1 b0
    s_waitcnt lgkmcnt(4)                          // auto-wait at pos 27
    v_mfma_scale_f32_16x16x128_f8f6f4 acc[16:19], v[100:103], v[60:63], acc[16:19], v110, v116, cbsz:4 blgp:4// MFMA m1_n0_k0
    v_mfma_scale_f32_16x16x128_f8f6f4 acc[20:23], v[100:103], v[68:71], acc[20:23], v110, v118, cbsz:4 blgp:4// MFMA m1_n1_k0
    v_mfma_scale_f32_16x16x128_f8f6f4 acc[24:27], v[100:103], v[76:79], acc[24:27], v110, v120, cbsz:4 blgp:4// MFMA m1_n2_k0
    v_mfma_scale_f32_16x16x128_f8f6f4 acc[28:31], v[100:103], v[84:87], acc[28:31], v110, v122, cbsz:4 blgp:4// MFMA m1_n3_k0
    s_waitcnt lgkmcnt(3)                          // auto-wait at pos 31
    v_mfma_scale_f32_16x16x128_f8f6f4 acc[16:19], v[104:107], v[64:67], acc[16:19], v111, v117, cbsz:4 blgp:4// MFMA m1_n0_k1
    v_mfma_scale_f32_16x16x128_f8f6f4 acc[20:23], v[104:107], v[72:75], acc[20:23], v111, v119, cbsz:4 blgp:4// MFMA m1_n1_k1
    v_mfma_scale_f32_16x16x128_f8f6f4 acc[24:27], v[104:107], v[80:83], acc[24:27], v111, v121, cbsz:4 blgp:4// MFMA m1_n2_k1
    v_mfma_scale_f32_16x16x128_f8f6f4 acc[28:31], v[104:107], v[88:91], acc[28:31], v111, v123, cbsz:4 blgp:4// MFMA m1_n3_k1
    ds_read_b128 v[100:103], v50                  // LR A m3k0 b1
    ds_read_b128 v[104:107], v51                  // LR A m3k1 b1
    s_waitcnt lgkmcnt(3)                          // auto-wait at pos 37
    v_mfma_scale_f32_16x16x128_f8f6f4 acc[32:35], v[92:95], v[60:63], acc[32:35], v112, v116, cbsz:4 blgp:4// MFMA m2_n0_k0
    v_mfma_scale_f32_16x16x128_f8f6f4 acc[36:39], v[92:95], v[68:71], acc[36:39], v112, v118, cbsz:4 blgp:4// MFMA m2_n1_k0
    v_mfma_scale_f32_16x16x128_f8f6f4 acc[40:43], v[92:95], v[76:79], acc[40:43], v112, v120, cbsz:4 blgp:4// MFMA m2_n2_k0
    v_mfma_scale_f32_16x16x128_f8f6f4 acc[44:47], v[92:95], v[84:87], acc[44:47], v112, v122, cbsz:4 blgp:4// MFMA m2_n3_k0
    s_waitcnt lgkmcnt(2)                          // auto-wait at pos 41
    v_mfma_scale_f32_16x16x128_f8f6f4 acc[32:35], v[96:99], v[64:67], acc[32:35], v113, v117, cbsz:4 blgp:4// MFMA m2_n0_k1
    v_mfma_scale_f32_16x16x128_f8f6f4 acc[36:39], v[96:99], v[72:75], acc[36:39], v113, v119, cbsz:4 blgp:4// MFMA m2_n1_k1
    v_mfma_scale_f32_16x16x128_f8f6f4 acc[40:43], v[96:99], v[80:83], acc[40:43], v113, v121, cbsz:4 blgp:4// MFMA m2_n2_k1
    v_mfma_scale_f32_16x16x128_f8f6f4 acc[44:47], v[96:99], v[88:91], acc[44:47], v113, v123, cbsz:4 blgp:4// MFMA m2_n3_k1
    s_waitcnt lgkmcnt(1)                          // auto-wait at pos 45
    v_mfma_scale_f32_16x16x128_f8f6f4 acc[48:51], v[100:103], v[60:63], acc[48:51], v114, v116, cbsz:4 blgp:4// MFMA m3_n0_k0
    v_mfma_scale_f32_16x16x128_f8f6f4 acc[52:55], v[100:103], v[68:71], acc[52:55], v114, v118, cbsz:4 blgp:4// MFMA m3_n1_k0
    v_mfma_scale_f32_16x16x128_f8f6f4 acc[56:59], v[100:103], v[76:79], acc[56:59], v114, v120, cbsz:4 blgp:4// MFMA m3_n2_k0
    v_mfma_scale_f32_16x16x128_f8f6f4 acc[60:63], v[100:103], v[84:87], acc[60:63], v114, v122, cbsz:4 blgp:4// MFMA m3_n3_k0
    s_waitcnt lgkmcnt(0)                          // auto-wait at pos 49
    v_mfma_scale_f32_16x16x128_f8f6f4 acc[48:51], v[104:107], v[64:67], acc[48:51], v115, v117, cbsz:4 blgp:4// MFMA m3_n0_k1
    v_mfma_scale_f32_16x16x128_f8f6f4 acc[52:55], v[104:107], v[72:75], acc[52:55], v115, v119, cbsz:4 blgp:4// MFMA m3_n1_k1
    v_mfma_scale_f32_16x16x128_f8f6f4 acc[56:59], v[104:107], v[80:83], acc[56:59], v115, v121, cbsz:4 blgp:4// MFMA m3_n2_k1
    v_mfma_scale_f32_16x16x128_f8f6f4 acc[60:63], v[104:107], v[88:91], acc[60:63], v115, v123, cbsz:4 blgp:4// MFMA m3_n3_k1
    // Toggle all precomputed read addresses (ADD)
    v_add_u32 v44, v44, s60                       // rd_a_m0_k0 += db
    v_add_u32 v45, v45, s60                       // rd_a_m0_k1 += db
    v_add_u32 v46, v46, s60                       // rd_a_m1_k0 += db
    v_add_u32 v47, v47, s60                       // rd_a_m1_k1 += db
    v_add_u32 v48, v48, s60                       // rd_a_m2_k0 += db
    v_add_u32 v49, v49, s60                       // rd_a_m2_k1 += db
    v_add_u32 v50, v50, s60                       // rd_a_m3_k0 += db
    v_add_u32 v51, v51, s60                       // rd_a_m3_k1 += db
    v_add_u32 v52, v52, s60                       // rd_b_n0_k0 += db
    v_add_u32 v53, v53, s60                       // rd_b_n0_k1 += db
    v_add_u32 v54, v54, s60                       // rd_b_n1_k0 += db
    v_add_u32 v55, v55, s60                       // rd_b_n1_k1 += db
    v_add_u32 v56, v56, s60                       // rd_b_n2_k0 += db
    v_add_u32 v57, v57, s60                       // rd_b_n2_k1 += db
    v_add_u32 v58, v58, s60                       // rd_b_n3_k0 += db
    v_add_u32 v59, v59, s60                       // rd_b_n3_k1 += db
    s_sub_u32 s13, s13, 1                         // k_tiles--
    s_cmp_gt_u32 s13, 1                           // k_tiles > 1?
    s_cbranch_scc0 load_skip_all                  // skip producers (drain)
    s_sub_u32 s60, 0, s60                         // negate db_step for next toggle
    s_add_u32 s16, s16, 128                       // s_srd_a += 128
    s_addc_u32 s17, s17, 0                        // carry
    s_add_u32 s24, s24, s60                       // s_lds_wr_a_sg += db
    s_mov_b32 m0, s24                             // m0 = LDS base A
    buffer_load_dwordx4 v5, s[16:19], 0, offen offset:0, lds// DTL A[0]
    s_add_u32 m0, m0, 4096                        // m0 += 4096
    buffer_load_dwordx4 v5, s[16:19], s48, offen offset:0, lds// DTL A[1]
    s_add_u32 m0, m0, 4096                        // m0 += 4096
    buffer_load_dwordx4 v5, s[16:19], s49, offen offset:0, lds// DTL A[2]
    s_add_u32 m0, m0, 4096                        // m0 += 4096
    buffer_load_dwordx4 v5, s[16:19], s50, offen offset:0, lds// DTL A[3]
    s_add_u32 s20, s20, 128                       // s_srd_b += 128
    s_addc_u32 s21, s21, 0                        // carry
    s_add_u32 s25, s25, s60                       // s_lds_wr_b_sg += db
    s_mov_b32 m0, s25                             // m0 = LDS base B
    buffer_load_dwordx4 v6, s[20:23], 0, offen offset:0, lds// DTL B[0]
    s_add_u32 m0, m0, 4096                        // m0 += 4096
    buffer_load_dwordx4 v6, s[20:23], s51, offen offset:0, lds// DTL B[1]
    s_add_u32 m0, m0, 4096                        // m0 += 4096
    buffer_load_dwordx4 v6, s[20:23], s52, offen offset:0, lds// DTL B[2]
    s_add_u32 m0, m0, 4096                        // m0 += 4096
    buffer_load_dwordx4 v6, s[20:23], s53, offen offset:0, lds// DTL B[3]

load_skip_all:
    s_cmp_gt_u32 s13, 1                           // k_tiles > 1? (exit to drain)
    s_cbranch_scc1 k_loop                         // loop

    // Drain stage 0
    s_waitcnt vmcnt(0)                            // wait DTL loads
    s_waitcnt lgkmcnt(0)                          // wait LDS writes
    s_barrier                                     // sync workgroup
    ds_read_b128 v[92:95], v44                    // LR A m0k0 b0
    ds_read_b128 v[60:63], v52                    // LR B n0k0
    buffer_load_dword v108, v124, s[36:39], 0, offen// scale A m0 k0
    buffer_load_dword v110, v124, s[36:39], s54, offen// scale A m1 k0
    buffer_load_dword v109, v124, s[36:39], 0, offen offset:4// scale A m0 k1
    buffer_load_dword v111, v124, s[36:39], s54, offen offset:4// scale A m1 k1
    buffer_load_dword v116, v125, s[40:43], 0, offen// scale B n0 k0
    buffer_load_dword v118, v125, s[40:43], s57, offen// scale B n1 k0
    buffer_load_dword v117, v125, s[40:43], 0, offen offset:4// scale B n0 k1
    buffer_load_dword v119, v125, s[40:43], s57, offen offset:4// scale B n1 k1
    ds_read_b128 v[68:71], v54                    // LR B n1k0
    ds_read_b128 v[76:79], v56                    // LR B n2k0
    buffer_load_dword v120, v125, s[40:43], s58, offen// scale B n2 k0
    buffer_load_dword v122, v125, s[40:43], s59, offen// scale B n3 k0
    buffer_load_dword v121, v125, s[40:43], s58, offen offset:4// scale B n2 k1
    buffer_load_dword v123, v125, s[40:43], s59, offen offset:4// scale B n3 k1
    ds_read_b128 v[84:87], v58                    // LR B n3k0
    ds_read_b128 v[96:99], v45                    // LR A m0k1 b0
    ds_read_b128 v[64:67], v53                    // LR B n0k1
    ds_read_b128 v[72:75], v55                    // LR B n1k1
    ds_read_b128 v[80:83], v57                    // LR B n2k1
    ds_read_b128 v[88:91], v59                    // LR B n3k1
    ds_read_b128 v[100:103], v46                  // LR A m1k0 b1
    ds_read_b128 v[104:107], v47                  // LR A m1k1 b1
    buffer_load_dword v112, v124, s[36:39], s55, offen// scale A m2 k0
    buffer_load_dword v114, v124, s[36:39], s56, offen// scale A m3 k0
    buffer_load_dword v113, v124, s[36:39], s55, offen offset:4// scale A m2 k1
    buffer_load_dword v115, v124, s[36:39], s56, offen offset:4// scale A m3 k1
    v_mfma_scale_f32_16x16x128_f8f6f4 acc[0:3], v[92:95], v[60:63], acc[0:3], v108, v116, cbsz:4 blgp:4// MFMA m0_n0_k0
    v_mfma_scale_f32_16x16x128_f8f6f4 acc[4:7], v[92:95], v[68:71], acc[4:7], v108, v118, cbsz:4 blgp:4// MFMA m0_n1_k0
    v_mfma_scale_f32_16x16x128_f8f6f4 acc[8:11], v[92:95], v[76:79], acc[8:11], v108, v120, cbsz:4 blgp:4// MFMA m0_n2_k0
    v_mfma_scale_f32_16x16x128_f8f6f4 acc[12:15], v[92:95], v[84:87], acc[12:15], v108, v122, cbsz:4 blgp:4// MFMA m0_n3_k0
    v_mfma_scale_f32_16x16x128_f8f6f4 acc[0:3], v[96:99], v[64:67], acc[0:3], v109, v117, cbsz:4 blgp:4// MFMA m0_n0_k1
    v_mfma_scale_f32_16x16x128_f8f6f4 acc[4:7], v[96:99], v[72:75], acc[4:7], v109, v119, cbsz:4 blgp:4// MFMA m0_n1_k1
    v_mfma_scale_f32_16x16x128_f8f6f4 acc[8:11], v[96:99], v[80:83], acc[8:11], v109, v121, cbsz:4 blgp:4// MFMA m0_n2_k1
    v_mfma_scale_f32_16x16x128_f8f6f4 acc[12:15], v[96:99], v[88:91], acc[12:15], v109, v123, cbsz:4 blgp:4// MFMA m0_n3_k1
    ds_read_b128 v[92:95], v48                    // LR A m2k0 b0
    ds_read_b128 v[96:99], v49                    // LR A m2k1 b0
    v_mfma_scale_f32_16x16x128_f8f6f4 acc[16:19], v[100:103], v[60:63], acc[16:19], v110, v116, cbsz:4 blgp:4// MFMA m1_n0_k0
    v_mfma_scale_f32_16x16x128_f8f6f4 acc[20:23], v[100:103], v[68:71], acc[20:23], v110, v118, cbsz:4 blgp:4// MFMA m1_n1_k0
    v_mfma_scale_f32_16x16x128_f8f6f4 acc[24:27], v[100:103], v[76:79], acc[24:27], v110, v120, cbsz:4 blgp:4// MFMA m1_n2_k0
    v_mfma_scale_f32_16x16x128_f8f6f4 acc[28:31], v[100:103], v[84:87], acc[28:31], v110, v122, cbsz:4 blgp:4// MFMA m1_n3_k0
    v_mfma_scale_f32_16x16x128_f8f6f4 acc[16:19], v[104:107], v[64:67], acc[16:19], v111, v117, cbsz:4 blgp:4// MFMA m1_n0_k1
    v_mfma_scale_f32_16x16x128_f8f6f4 acc[20:23], v[104:107], v[72:75], acc[20:23], v111, v119, cbsz:4 blgp:4// MFMA m1_n1_k1
    v_mfma_scale_f32_16x16x128_f8f6f4 acc[24:27], v[104:107], v[80:83], acc[24:27], v111, v121, cbsz:4 blgp:4// MFMA m1_n2_k1
    v_mfma_scale_f32_16x16x128_f8f6f4 acc[28:31], v[104:107], v[88:91], acc[28:31], v111, v123, cbsz:4 blgp:4// MFMA m1_n3_k1
    ds_read_b128 v[100:103], v50                  // LR A m3k0 b1
    ds_read_b128 v[104:107], v51                  // LR A m3k1 b1
    v_mfma_scale_f32_16x16x128_f8f6f4 acc[32:35], v[92:95], v[60:63], acc[32:35], v112, v116, cbsz:4 blgp:4// MFMA m2_n0_k0
    v_mfma_scale_f32_16x16x128_f8f6f4 acc[36:39], v[92:95], v[68:71], acc[36:39], v112, v118, cbsz:4 blgp:4// MFMA m2_n1_k0
    v_mfma_scale_f32_16x16x128_f8f6f4 acc[40:43], v[92:95], v[76:79], acc[40:43], v112, v120, cbsz:4 blgp:4// MFMA m2_n2_k0
    v_mfma_scale_f32_16x16x128_f8f6f4 acc[44:47], v[92:95], v[84:87], acc[44:47], v112, v122, cbsz:4 blgp:4// MFMA m2_n3_k0
    v_mfma_scale_f32_16x16x128_f8f6f4 acc[32:35], v[96:99], v[64:67], acc[32:35], v113, v117, cbsz:4 blgp:4// MFMA m2_n0_k1
    v_mfma_scale_f32_16x16x128_f8f6f4 acc[36:39], v[96:99], v[72:75], acc[36:39], v113, v119, cbsz:4 blgp:4// MFMA m2_n1_k1
    v_mfma_scale_f32_16x16x128_f8f6f4 acc[40:43], v[96:99], v[80:83], acc[40:43], v113, v121, cbsz:4 blgp:4// MFMA m2_n2_k1
    v_mfma_scale_f32_16x16x128_f8f6f4 acc[44:47], v[96:99], v[88:91], acc[44:47], v113, v123, cbsz:4 blgp:4// MFMA m2_n3_k1
    v_mfma_scale_f32_16x16x128_f8f6f4 acc[48:51], v[100:103], v[60:63], acc[48:51], v114, v116, cbsz:4 blgp:4// MFMA m3_n0_k0
    v_mfma_scale_f32_16x16x128_f8f6f4 acc[52:55], v[100:103], v[68:71], acc[52:55], v114, v118, cbsz:4 blgp:4// MFMA m3_n1_k0
    v_mfma_scale_f32_16x16x128_f8f6f4 acc[56:59], v[100:103], v[76:79], acc[56:59], v114, v120, cbsz:4 blgp:4// MFMA m3_n2_k0
    v_mfma_scale_f32_16x16x128_f8f6f4 acc[60:63], v[100:103], v[84:87], acc[60:63], v114, v122, cbsz:4 blgp:4// MFMA m3_n3_k0
    v_mfma_scale_f32_16x16x128_f8f6f4 acc[48:51], v[104:107], v[64:67], acc[48:51], v115, v117, cbsz:4 blgp:4// MFMA m3_n0_k1
    v_mfma_scale_f32_16x16x128_f8f6f4 acc[52:55], v[104:107], v[72:75], acc[52:55], v115, v119, cbsz:4 blgp:4// MFMA m3_n1_k1
    v_mfma_scale_f32_16x16x128_f8f6f4 acc[56:59], v[104:107], v[80:83], acc[56:59], v115, v121, cbsz:4 blgp:4// MFMA m3_n2_k1
    v_mfma_scale_f32_16x16x128_f8f6f4 acc[60:63], v[104:107], v[88:91], acc[60:63], v115, v123, cbsz:4 blgp:4// MFMA m3_n3_k1
    // Toggle all precomputed read addresses (ADD)
    v_add_u32 v44, v44, s60                       // rd_a_m0_k0 += db
    v_add_u32 v45, v45, s60                       // rd_a_m0_k1 += db
    v_add_u32 v46, v46, s60                       // rd_a_m1_k0 += db
    v_add_u32 v47, v47, s60                       // rd_a_m1_k1 += db
    v_add_u32 v48, v48, s60                       // rd_a_m2_k0 += db
    v_add_u32 v49, v49, s60                       // rd_a_m2_k1 += db
    v_add_u32 v50, v50, s60                       // rd_a_m3_k0 += db
    v_add_u32 v51, v51, s60                       // rd_a_m3_k1 += db
    v_add_u32 v52, v52, s60                       // rd_b_n0_k0 += db
    v_add_u32 v53, v53, s60                       // rd_b_n0_k1 += db
    v_add_u32 v54, v54, s60                       // rd_b_n1_k0 += db
    v_add_u32 v55, v55, s60                       // rd_b_n1_k1 += db
    v_add_u32 v56, v56, s60                       // rd_b_n2_k0 += db
    v_add_u32 v57, v57, s60                       // rd_b_n2_k1 += db
    v_add_u32 v58, v58, s60                       // rd_b_n3_k0 += db
    v_add_u32 v59, v59, s60                       // rd_b_n3_k1 += db

    // === Store D via buffer SRD ===
    // SRD for D matrix (raw buffer mode)
    s_mov_b32 s64, s8                             // SRD_D base lo
    s_mov_b32 s65, s9                             // SRD_D base hi
    s_mov_b32 s66, 0xFFFFFFFF                     // SRD_D size (unlimited)
    s_mov_b32 s67, 0x20000                        // SRD_D flags: raw buffer

    v_and_b32 v29, 15, v2                         // lane_n = lane_id % 16
    v_lshrrev_b32 v30, 4, v2                      // lane_id / 16
    v_lshlrev_b32 v30, 2, v30                     // * 4 -> lane_m_base
    v_lshlrev_b32 v26, 6, v3                      // wave_m * 64
    v_add_u32 v26, v26, v30                       // + lane_m_base
    s_mul_i32 s15, s2, 128                        // wg_id_x * 128
    v_add_u32 v26, s15, v26                       // + wg_base_m -> global_row
    v_lshlrev_b32 v27, 6, v4                      // wave_n * 64
    v_add_u32 v27, v27, v29                       // + lane_n
    s_mul_i32 s15, s3, 128                        // wg_id_y * 128
    v_add_u32 v27, s15, v27                       // + wg_base_n -> global_col
    v_mul_lo_u32 v27, s10, v27                    // global_col * M
    v_add_u32 v26, v26, v27                       // + global_row -> col-major linear index
    v_lshlrev_b32 v26, 1, v26                     // * 2 -> byte offset

    // Store 64 elements (4x4x4) via buffer_store_short
    s_lshl_b32 s14, s10, 1                        // col_stride = M * 2 bytes

    s_mov_b32 s15, 0                              // soffset = 0 (ni=0)
    v_accvgpr_read_b32 v29, acc0                  // acc[0] a0
    v_accvgpr_read_b32 v30, acc1                  // acc[1] a1
    v_cvt_pk_bf16_f32 v28, v29, v30               // pk cvt a0,a1
    v_accvgpr_read_b32 v29, acc2                  // acc[2] a2
    v_accvgpr_read_b32 v30, acc3                  // acc[3] a3
    v_cvt_pk_bf16_f32 v29, v29, v30               // pk cvt a2,a3
    buffer_store_dwordx2 v[28:29], v26, s[64:67], s15, offen nt// store D m0_n0 GWVW=4
    v_accvgpr_read_b32 v29, acc16                 // acc[16] a0
    v_accvgpr_read_b32 v30, acc17                 // acc[17] a1
    v_cvt_pk_bf16_f32 v28, v29, v30               // pk cvt a0,a1
    v_accvgpr_read_b32 v29, acc18                 // acc[18] a2
    v_accvgpr_read_b32 v30, acc19                 // acc[19] a3
    v_cvt_pk_bf16_f32 v29, v29, v30               // pk cvt a2,a3
    buffer_store_dwordx2 v[28:29], v26, s[64:67], s15, offen offset:32 nt// store D m1_n0 GWVW=4
    v_accvgpr_read_b32 v29, acc32                 // acc[32] a0
    v_accvgpr_read_b32 v30, acc33                 // acc[33] a1
    v_cvt_pk_bf16_f32 v28, v29, v30               // pk cvt a0,a1
    v_accvgpr_read_b32 v29, acc34                 // acc[34] a2
    v_accvgpr_read_b32 v30, acc35                 // acc[35] a3
    v_cvt_pk_bf16_f32 v29, v29, v30               // pk cvt a2,a3
    buffer_store_dwordx2 v[28:29], v26, s[64:67], s15, offen offset:64 nt// store D m2_n0 GWVW=4
    v_accvgpr_read_b32 v29, acc48                 // acc[48] a0
    v_accvgpr_read_b32 v30, acc49                 // acc[49] a1
    v_cvt_pk_bf16_f32 v28, v29, v30               // pk cvt a0,a1
    v_accvgpr_read_b32 v29, acc50                 // acc[50] a2
    v_accvgpr_read_b32 v30, acc51                 // acc[51] a3
    v_cvt_pk_bf16_f32 v29, v29, v30               // pk cvt a2,a3
    buffer_store_dwordx2 v[28:29], v26, s[64:67], s15, offen offset:96 nt// store D m3_n0 GWVW=4
    s_mul_i32 s15, s14, 16                        // soffset = 16 * col_stride (ni=1)
    v_accvgpr_read_b32 v29, acc4                  // acc[4] a0
    v_accvgpr_read_b32 v30, acc5                  // acc[5] a1
    v_cvt_pk_bf16_f32 v28, v29, v30               // pk cvt a0,a1
    v_accvgpr_read_b32 v29, acc6                  // acc[6] a2
    v_accvgpr_read_b32 v30, acc7                  // acc[7] a3
    v_cvt_pk_bf16_f32 v29, v29, v30               // pk cvt a2,a3
    buffer_store_dwordx2 v[28:29], v26, s[64:67], s15, offen nt// store D m0_n1 GWVW=4
    v_accvgpr_read_b32 v29, acc20                 // acc[20] a0
    v_accvgpr_read_b32 v30, acc21                 // acc[21] a1
    v_cvt_pk_bf16_f32 v28, v29, v30               // pk cvt a0,a1
    v_accvgpr_read_b32 v29, acc22                 // acc[22] a2
    v_accvgpr_read_b32 v30, acc23                 // acc[23] a3
    v_cvt_pk_bf16_f32 v29, v29, v30               // pk cvt a2,a3
    buffer_store_dwordx2 v[28:29], v26, s[64:67], s15, offen offset:32 nt// store D m1_n1 GWVW=4
    v_accvgpr_read_b32 v29, acc36                 // acc[36] a0
    v_accvgpr_read_b32 v30, acc37                 // acc[37] a1
    v_cvt_pk_bf16_f32 v28, v29, v30               // pk cvt a0,a1
    v_accvgpr_read_b32 v29, acc38                 // acc[38] a2
    v_accvgpr_read_b32 v30, acc39                 // acc[39] a3
    v_cvt_pk_bf16_f32 v29, v29, v30               // pk cvt a2,a3
    buffer_store_dwordx2 v[28:29], v26, s[64:67], s15, offen offset:64 nt// store D m2_n1 GWVW=4
    v_accvgpr_read_b32 v29, acc52                 // acc[52] a0
    v_accvgpr_read_b32 v30, acc53                 // acc[53] a1
    v_cvt_pk_bf16_f32 v28, v29, v30               // pk cvt a0,a1
    v_accvgpr_read_b32 v29, acc54                 // acc[54] a2
    v_accvgpr_read_b32 v30, acc55                 // acc[55] a3
    v_cvt_pk_bf16_f32 v29, v29, v30               // pk cvt a2,a3
    buffer_store_dwordx2 v[28:29], v26, s[64:67], s15, offen offset:96 nt// store D m3_n1 GWVW=4
    s_mul_i32 s15, s14, 32                        // soffset = 32 * col_stride (ni=2)
    v_accvgpr_read_b32 v29, acc8                  // acc[8] a0
    v_accvgpr_read_b32 v30, acc9                  // acc[9] a1
    v_cvt_pk_bf16_f32 v28, v29, v30               // pk cvt a0,a1
    v_accvgpr_read_b32 v29, acc10                 // acc[10] a2
    v_accvgpr_read_b32 v30, acc11                 // acc[11] a3
    v_cvt_pk_bf16_f32 v29, v29, v30               // pk cvt a2,a3
    buffer_store_dwordx2 v[28:29], v26, s[64:67], s15, offen nt// store D m0_n2 GWVW=4
    v_accvgpr_read_b32 v29, acc24                 // acc[24] a0
    v_accvgpr_read_b32 v30, acc25                 // acc[25] a1
    v_cvt_pk_bf16_f32 v28, v29, v30               // pk cvt a0,a1
    v_accvgpr_read_b32 v29, acc26                 // acc[26] a2
    v_accvgpr_read_b32 v30, acc27                 // acc[27] a3
    v_cvt_pk_bf16_f32 v29, v29, v30               // pk cvt a2,a3
    buffer_store_dwordx2 v[28:29], v26, s[64:67], s15, offen offset:32 nt// store D m1_n2 GWVW=4
    v_accvgpr_read_b32 v29, acc40                 // acc[40] a0
    v_accvgpr_read_b32 v30, acc41                 // acc[41] a1
    v_cvt_pk_bf16_f32 v28, v29, v30               // pk cvt a0,a1
    v_accvgpr_read_b32 v29, acc42                 // acc[42] a2
    v_accvgpr_read_b32 v30, acc43                 // acc[43] a3
    v_cvt_pk_bf16_f32 v29, v29, v30               // pk cvt a2,a3
    buffer_store_dwordx2 v[28:29], v26, s[64:67], s15, offen offset:64 nt// store D m2_n2 GWVW=4
    v_accvgpr_read_b32 v29, acc56                 // acc[56] a0
    v_accvgpr_read_b32 v30, acc57                 // acc[57] a1
    v_cvt_pk_bf16_f32 v28, v29, v30               // pk cvt a0,a1
    v_accvgpr_read_b32 v29, acc58                 // acc[58] a2
    v_accvgpr_read_b32 v30, acc59                 // acc[59] a3
    v_cvt_pk_bf16_f32 v29, v29, v30               // pk cvt a2,a3
    buffer_store_dwordx2 v[28:29], v26, s[64:67], s15, offen offset:96 nt// store D m3_n2 GWVW=4
    s_mul_i32 s15, s14, 48                        // soffset = 48 * col_stride (ni=3)
    v_accvgpr_read_b32 v29, acc12                 // acc[12] a0
    v_accvgpr_read_b32 v30, acc13                 // acc[13] a1
    v_cvt_pk_bf16_f32 v28, v29, v30               // pk cvt a0,a1
    v_accvgpr_read_b32 v29, acc14                 // acc[14] a2
    v_accvgpr_read_b32 v30, acc15                 // acc[15] a3
    v_cvt_pk_bf16_f32 v29, v29, v30               // pk cvt a2,a3
    buffer_store_dwordx2 v[28:29], v26, s[64:67], s15, offen nt// store D m0_n3 GWVW=4
    v_accvgpr_read_b32 v29, acc28                 // acc[28] a0
    v_accvgpr_read_b32 v30, acc29                 // acc[29] a1
    v_cvt_pk_bf16_f32 v28, v29, v30               // pk cvt a0,a1
    v_accvgpr_read_b32 v29, acc30                 // acc[30] a2
    v_accvgpr_read_b32 v30, acc31                 // acc[31] a3
    v_cvt_pk_bf16_f32 v29, v29, v30               // pk cvt a2,a3
    buffer_store_dwordx2 v[28:29], v26, s[64:67], s15, offen offset:32 nt// store D m1_n3 GWVW=4
    v_accvgpr_read_b32 v29, acc44                 // acc[44] a0
    v_accvgpr_read_b32 v30, acc45                 // acc[45] a1
    v_cvt_pk_bf16_f32 v28, v29, v30               // pk cvt a0,a1
    v_accvgpr_read_b32 v29, acc46                 // acc[46] a2
    v_accvgpr_read_b32 v30, acc47                 // acc[47] a3
    v_cvt_pk_bf16_f32 v29, v29, v30               // pk cvt a2,a3
    buffer_store_dwordx2 v[28:29], v26, s[64:67], s15, offen offset:64 nt// store D m2_n3 GWVW=4
    v_accvgpr_read_b32 v29, acc60                 // acc[60] a0
    v_accvgpr_read_b32 v30, acc61                 // acc[61] a1
    v_cvt_pk_bf16_f32 v28, v29, v30               // pk cvt a0,a1
    v_accvgpr_read_b32 v29, acc62                 // acc[62] a2
    v_accvgpr_read_b32 v30, acc63                 // acc[63] a3
    v_cvt_pk_bf16_f32 v29, v29, v30               // pk cvt a2,a3
    buffer_store_dwordx2 v[28:29], v26, s[64:67], s15, offen offset:96 nt// store D m3_n3 GWVW=4
    s_waitcnt vmcnt(0)                            // wait for stores

    s_endpgm                                      // end of kernel

.rodata
.p2align 6
.amdhsa_kernel Custom_Cijk_Alik_Bljk_F4BS_MXA32_MXB32_MT128x128x256_MI16x16x1_kgen_gfx950
    .amdhsa_group_segment_fixed_size 65536
    .amdhsa_private_segment_fixed_size 0
    .amdhsa_kernarg_size 120
    .amdhsa_user_sgpr_kernarg_segment_ptr 1
    .amdhsa_system_sgpr_workgroup_id_x 1
    .amdhsa_system_sgpr_workgroup_id_y 1
    .amdhsa_system_vgpr_workitem_id 0
    .amdhsa_next_free_vgpr 192
    .amdhsa_next_free_sgpr 68
    .amdhsa_accum_offset 128
    .amdhsa_float_denorm_mode_32 3
    .amdhsa_float_denorm_mode_16_64 3
.end_amdhsa_kernel

.amdgpu_metadata
---
custom.config:
  InternalSupportParams:
    KernArgsVersion: 2
    UseUniversalArgs: false
    SupportUserGSU: false
    SupportCustomWGM: false
    SupportCustomStaggerU: false
  ProblemType:
    OperationType: GEMM
    DataType: F4
    DestDataType: B
    ComputeDataType: S
    HighPrecisionAccumulate: true
    TransposeA: 1
    TransposeB: 0
    UseBeta: true
    Batched: true
    MXBlockA: 32
    MXBlockB: 32
  MatrixInstruction:
  - 16
  - 16
  - 128
  - 1
  MIBlock:
  - 16
  - 16
  - 128
  - 1
  - 1
  - 1
  MIInputPerThread: 32
  MIInputPerThreadA: 32
  MIInputPerThreadB: 32
  MIInputPerThreadMXSA: 1
  MIInputPerThreadMXSB: 1
  WavefrontSize: 64
  WorkGroupMapping: 16
  WorkGroupMappingXCC: 2
  WorkGroupMappingXCCGroup: -1
  StaggerU: 0
  EnableMatrixInstruction: true
  MIWaveGroup:
  - 2
  - 2
  MIWaveTile:
  - 4
  - 4
  DepthU: 256
  MacroTile:
  - 128
  - 128
  DirectToLds: 1
  LocalReadVectorWidth: -1
  GlobalReadVectorWidthA: 32
  GlobalReadVectorWidthB: 32
  GlobalSplitU: 1
  GlobalSplitUAlgorithm: MultipleBuffer
  GlobalSplitUCoalesced: false
  GlobalSplitUWorkGroupMappingRoundRobin: false
  PrefetchGlobalRead: 2
  PrefetchLocalRead: 1
  StreamK: 0
  StreamKAtomic: 0
  StreamKXCCMapping: 0
  TransposeLDS: 0
  PreloadKernArgs: False
  NoReject: true
amdhsa.version: [ 1, 1 ]
amdhsa.kernels:
  - .name:            Custom_Cijk_Alik_Bljk_F4BS_MXA32_MXB32_MT128x128x256_MI16x16x1_kgen_gfx950
    .symbol:          Custom_Cijk_Alik_Bljk_F4BS_MXA32_MXB32_MT128x128x256_MI16x16x1_kgen_gfx950.kd
    .sgpr_count:      68
    .vgpr_count:      192
    .agpr_count:      64
    .kernarg_segment_size: 120
    .kernarg_segment_align: 8
    .group_segment_fixed_size: 65536
    .private_segment_fixed_size: 0
    .wavefront_size:  64
    .max_flat_workgroup_size: 256
    .args:
      - .name:           SizesFree0
        .offset:         0
        .size:           4
        .value_kind:     by_value
      - .name:           SizesFree1
        .offset:         4
        .size:           4
        .value_kind:     by_value
      - .name:           SizesFree2
        .offset:         8
        .size:           4
        .value_kind:     by_value
      - .name:           SizesSum0
        .offset:         12
        .size:           4
        .value_kind:     by_value
      - .name:           D
        .offset:         16
        .size:           8
        .value_kind:     global_buffer
        .address_space:  generic
      - .name:           C
        .offset:         24
        .size:           8
        .value_kind:     global_buffer
        .address_space:  generic
      - .name:           A
        .offset:         32
        .size:           8
        .value_kind:     global_buffer
        .address_space:  generic
      - .name:           MXSA
        .offset:         40
        .size:           8
        .value_kind:     global_buffer
        .address_space:  generic
      - .name:           B
        .offset:         48
        .size:           8
        .value_kind:     global_buffer
        .address_space:  generic
      - .name:           MXSB
        .offset:         56
        .size:           8
        .value_kind:     global_buffer
        .address_space:  generic
      - .name:           strideD0
        .offset:         64
        .size:           4
        .value_kind:     by_value
      - .name:           strideD1
        .offset:         68
        .size:           4
        .value_kind:     by_value
      - .name:           strideC0
        .offset:         72
        .size:           4
        .value_kind:     by_value
      - .name:           strideC1
        .offset:         76
        .size:           4
        .value_kind:     by_value
      - .name:           strideA0
        .offset:         80
        .size:           4
        .value_kind:     by_value
      - .name:           strideA1
        .offset:         84
        .size:           4
        .value_kind:     by_value
      - .name:           strideMXSA0
        .offset:         88
        .size:           4
        .value_kind:     by_value
      - .name:           strideMXSA1
        .offset:         92
        .size:           4
        .value_kind:     by_value
      - .name:           strideB0
        .offset:         96
        .size:           4
        .value_kind:     by_value
      - .name:           strideB1
        .offset:         100
        .size:           4
        .value_kind:     by_value
      - .name:           strideMXSB0
        .offset:         104
        .size:           4
        .value_kind:     by_value
      - .name:           strideMXSB1
        .offset:         108
        .size:           4
        .value_kind:     by_value
      - .name:           alpha
        .offset:         112
        .size:           4
        .value_kind:     by_value
      - .name:           beta
        .offset:         116
        .size:           4
        .value_kind:     by_value
...
.end_amdgpu_metadata
