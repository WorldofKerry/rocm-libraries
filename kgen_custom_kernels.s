.amdgcn_target "amdgcn-amd-amdhsa--gfx950"
.text
.globl wave_mxfp4_128x128x256_kgen
.p2align 8
.type wave_mxfp4_128x128x256_kgen,@function

wave_mxfp4_128x128x256_kgen:
    // === Wave ABI Setup (rocRoller custom kernel) ===
    s_load_dwordx2 s[4:5], s[0:1], 0              // ptr_a (kernel A)
    s_load_dwordx2 s[30:31], s[0:1], 8            // ptr_a_scale
    s_load_dwordx2 s[6:7], s[0:1], 16             // ptr_b (kernel B)
    s_load_dwordx2 s[32:33], s[0:1], 24           // ptr_b_scale
    s_load_dwordx2 s[8:9], s[0:1], 32             // ptr_c (D output)
    s_load_dword s10, s[0:1], 40                  // M = low dword of m (u64)
    s_load_dword s11, s[0:1], 48                  // N = low dword of n (u64)
    s_load_dword s12, s[0:1], 56                  // K = low dword of k (u64)
    s_load_dword s34, s[0:1], 72                  // stride_a_scale_dim0 (low 32)
    s_load_dword s35, s[0:1], 88                  // stride_b_scale_dim0 (low 32)
    s_waitcnt lgkmcnt(0)                          // wait kernargs

    s_lshr_b32 s26, s12, 1                        // s_k_stride = K * 0.5
    v_lshrrev_b32 v1, 6, v0                       // wave_id = tid >> 6
    v_and_b32 v2, 63, v0                          // lane_id = tid & 63
    v_lshrrev_b32 v3, 1, v1                       // wave_m = wave_id >> 1
    v_and_b32 v4, 1, v1                           // wave_n = wave_id & 1

    // DTL offset: 8 threads/row
    v_lshrrev_b32 v29, 3, v0                      // thread_row
    v_and_b32 v30, 7, v0                          // thread_col_group
    v_lshlrev_b32 v30, 4, v30                     // * 16 -> col_bytes
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
    v_lshrrev_b32 v30, 4, v2                      // lane_id / 16
    v_lshlrev_b32 v30, 5, v30                     // * 32
    v_lshlrev_b32 v7, 6, v3                       // wave_m * 64
    v_add_u32 v7, v7, v29                         // + lane_row
    v_lshlrev_b32 v7, 8, v7                       // * 256
    v_add_u32 v7, v7, v30                         // + lane_k
    v_lshrrev_b32 v7, 1, v7                       // * 0.5 (sub-byte)

    v_and_b32 v29, 15, v2                         // lane_row = lane_id % 16 (re-derive)
    v_lshlrev_b32 v8, 6, v4                       // wave_n * 64
    v_add_u32 v8, v8, v29                         // + lane_row
    v_lshlrev_b32 v8, 8, v8                       // * 256
    v_add_u32 v8, v8, v30                         // + lane_k
    v_lshrrev_b32 v8, 1, v8                       // * 0.5 (sub-byte)
    v_add_u32 v8, 16384, v8                       // + lds_b_offset(16384)

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
    // Scale SRD A
    s_mul_i32 s14, s2, 4                          // wg_id_x * 4
    s_mul_i32 s14, s14, s34                       // * stride_scale_a
    s_add_u32 s36, s30, s14                       // SRD_scaleA lo
    s_addc_u32 s37, s31, 0                        // SRD_scaleA hi
    s_mov_b32 s38, 0xFFFFFFFF                     // limit
    s_mov_b32 s39, 0x20000                        // flags

    // Scale SRD B
    s_mul_i32 s14, s3, 4                          // wg_id_y * 4
    s_mul_i32 s14, s14, s35                       // * stride_scale_b
    s_add_u32 s40, s32, s14                       // SRD_scaleB lo
    s_addc_u32 s41, s33, 0                        // SRD_scaleB hi
    s_mov_b32 s42, 0xFFFFFFFF                     // limit
    s_mov_b32 s43, 0x20000                        // flags

    // Scale swizzled voffset: lane_id * 4
    v_lshlrev_b32 v21, 2, v2                      // lane_id * 4 -> swizzled scale voffset
    v_mov_b32 v22, v21                            // scaleB voffset = same

    // Scale group soffsets
    v_lshlrev_b32 v29, 1, v3                      // wave_m * 2
    v_readfirstlane_b32 s14, v29                  // wave_m * 2 -> SGPR
    s_lshl_b32 s46, s14, 8                        // group0 soffset = wave_m*2 * 256
    s_add_u32 s47, s46, 256                       // group1 soffset = (wave_m*2+1) * 256
    v_lshlrev_b32 v29, 1, v4                      // wave_n * 2
    v_readfirstlane_b32 s14, v29                  // wave_n * 2 -> SGPR
    s_lshl_b32 s48, s14, 8                        // group0 soffset B = wave_n*2 * 256
    s_add_u32 s49, s48, 256                       // group1 soffset B = (wave_n*2+1) * 256

    // === DTL Partitioned K-loop ===
    s_lshr_b32 s13, s12, 8                        // k_tiles = K / 256
    s_mov_b32 s50, 32768                          // DB step = 32768

    // Prologue: DTL tile 0
    s_mov_b32 m0, s24                             // m0 = LDS base A
    s_mov_b32 s14, 0                              // cumulative soffset A
    buffer_load_dwordx4 v5, s[16:19], s14, offen offset:0, lds// DTL A[0]
    s_add_u32 m0, m0, 4096                        // m0 += 4096
    s_add_u32 s14, s14, s27                       // soffset += stride
    buffer_load_dwordx4 v5, s[16:19], s14, offen offset:0, lds// DTL A[1]
    s_add_u32 m0, m0, 4096                        // m0 += 4096
    s_add_u32 s14, s14, s27                       // soffset += stride
    buffer_load_dwordx4 v5, s[16:19], s14, offen offset:0, lds// DTL A[2]
    s_add_u32 m0, m0, 4096                        // m0 += 4096
    s_add_u32 s14, s14, s27                       // soffset += stride
    buffer_load_dwordx4 v5, s[16:19], s14, offen offset:0, lds// DTL A[3]
    s_mov_b32 m0, s25                             // m0 = LDS base B
    s_mov_b32 s14, 0                              // cumulative soffset B
    buffer_load_dwordx4 v6, s[20:23], s14, offen offset:0, lds// DTL B[0]
    s_add_u32 m0, m0, 4096                        // m0 += 4096
    s_add_u32 s14, s14, s28                       // soffset += stride
    buffer_load_dwordx4 v6, s[20:23], s14, offen offset:0, lds// DTL B[1]
    s_add_u32 m0, m0, 4096                        // m0 += 4096
    s_add_u32 s14, s14, s28                       // soffset += stride
    buffer_load_dwordx4 v6, s[20:23], s14, offen offset:0, lds// DTL B[2]
    s_add_u32 m0, m0, 4096                        // m0 += 4096
    s_add_u32 s14, s14, s28                       // soffset += stride
    buffer_load_dwordx4 v6, s[20:23], s14, offen offset:0, lds// DTL B[3]
    // Load swizzled scales A (2 groups)
    buffer_load_dword v80, v21, s[36:39], s46, offen// scaleA group0 (mi=0,1)
    buffer_load_dword v81, v21, s[36:39], s47, offen// scaleA group1 (mi=2,3)
    // Load swizzled scales B (2 groups)
    buffer_load_dword v82, v22, s[40:43], s48, offen// scaleB group0 (ni=0,1)
    buffer_load_dword v83, v22, s[40:43], s49, offen// scaleB group1 (ni=2,3)
    s_waitcnt vmcnt(0)                            // wait scale VGPR loads
    s_waitcnt vmcnt(0)                            // wait DTL
    s_barrier                                     // sync

k_loop:

    s_sub_u32 s13, s13, 1                         // k_tiles--
    s_cmp_lg_u32 s13, 0                           // more tiles?
    s_cbranch_scc0 dtl_skip_all                   // skip DTL on last iter
    s_add_u32 s16, s16, 128                       // s_srd_a += 128
    s_addc_u32 s17, s17, 0                        // carry
    s_add_u32 s20, s20, 128                       // s_srd_b += 128
    s_addc_u32 s21, s21, 0                        // carry
    s_add_u32 s36, s36, 256                       // s_srd_scale_a += 256
    s_addc_u32 s37, s37, 0                        // carry
    s_add_u32 s40, s40, 256                       // s_srd_scale_b += 256
    s_addc_u32 s41, s41, 0                        // carry
    s_add_u32 s24, s24, s50                       // wr_a += db
    s_add_u32 s25, s25, s50                       // wr_b += db
    s_mov_b32 m0, s24                             // m0 = LDS base A
    s_mov_b32 s14, 0                              // cumulative soffset A
    buffer_load_dwordx4 v5, s[16:19], s14, offen offset:0, lds// DTL A[0]
    s_add_u32 m0, m0, 4096                        // m0 += 4096
    s_add_u32 s14, s14, s27                       // soffset += stride
    buffer_load_dwordx4 v5, s[16:19], s14, offen offset:0, lds// DTL A[1]
    s_add_u32 m0, m0, 4096                        // m0 += 4096
    s_add_u32 s14, s14, s27                       // soffset += stride
    buffer_load_dwordx4 v5, s[16:19], s14, offen offset:0, lds// DTL A[2]
    s_add_u32 m0, m0, 4096                        // m0 += 4096
    s_add_u32 s14, s14, s27                       // soffset += stride
    buffer_load_dwordx4 v5, s[16:19], s14, offen offset:0, lds// DTL A[3]
    s_mov_b32 m0, s25                             // m0 = LDS base B
    s_mov_b32 s14, 0                              // cumulative soffset B
    buffer_load_dwordx4 v6, s[20:23], s14, offen offset:0, lds// DTL B[0]
    s_add_u32 m0, m0, 4096                        // m0 += 4096
    s_add_u32 s14, s14, s28                       // soffset += stride
    buffer_load_dwordx4 v6, s[20:23], s14, offen offset:0, lds// DTL B[1]
    s_add_u32 m0, m0, 4096                        // m0 += 4096
    s_add_u32 s14, s14, s28                       // soffset += stride
    buffer_load_dwordx4 v6, s[20:23], s14, offen offset:0, lds// DTL B[2]
    s_add_u32 m0, m0, 4096                        // m0 += 4096
    s_add_u32 s14, s14, s28                       // soffset += stride
    buffer_load_dwordx4 v6, s[20:23], s14, offen offset:0, lds// DTL B[3]
    // Load swizzled scales A (2 groups)
    buffer_load_dword v80, v21, s[36:39], s46, offen// scaleA group0 (mi=0,1)
    buffer_load_dword v81, v21, s[36:39], s47, offen// scaleA group1 (mi=2,3)
    // Load swizzled scales B (2 groups)
    buffer_load_dword v82, v22, s[40:43], s48, offen// scaleB group0 (ni=0,1)
    buffer_load_dword v83, v22, s[40:43], s49, offen// scaleB group1 (ni=2,3)
    s_waitcnt vmcnt(0)                            // wait scale VGPR loads

dtl_skip_all:
    s_barrier                                     // sync DTL writes before preamble reads

    // Preamble: scales + B + A[m0]
    ds_read_b128 v[32:35], v8                     // LR B n0k0
    ds_read_b128 v[40:43], v8, offset:2048        // LR B n1k0
    ds_read_b128 v[48:51], v8, offset:4096        // LR B n2k0
    ds_read_b128 v[56:59], v8, offset:6144        // LR B n3k0
    ds_read_b128 v[64:67], v7                     // LR A m0k0 b0
    ds_read_b128 v[36:39], v8, offset:64          // LR B n0k1
    ds_read_b128 v[44:47], v8, offset:2112        // LR B n1k1
    ds_read_b128 v[52:55], v8, offset:4160        // LR B n2k1
    ds_read_b128 v[60:63], v8, offset:6208        // LR B n3k1
    ds_read_b128 v[68:71], v7, offset:64          // LR A m0k1 b0
    s_waitcnt lgkmcnt(10)                         // wait B[ki=0] + A[m0,k0]

    // --- Partition 0 ---
    v_mfma_scale_f32_16x16x128_f8f6f4 acc[0:3], v[64:67], v[32:35], acc[0:3], v80, v82, op_sel:[0,0] op_sel_hi:[0,0] cbsz:4 blgp:4// MFMA m0_n0_k0
    v_mfma_scale_f32_16x16x128_f8f6f4 acc[4:7], v[64:67], v[40:43], acc[4:7], v80, v82, op_sel:[0,1] op_sel_hi:[0,0] cbsz:4 blgp:4// MFMA m0_n1_k0
    ds_read_b128 v[72:75], v7, offset:2048        // LR A m1k0 b1
    v_mfma_scale_f32_16x16x128_f8f6f4 acc[8:11], v[64:67], v[48:51], acc[8:11], v80, v83, op_sel:[0,0] op_sel_hi:[0,0] cbsz:4 blgp:4// MFMA m0_n2_k0
    v_mfma_scale_f32_16x16x128_f8f6f4 acc[12:15], v[64:67], v[56:59], acc[12:15], v80, v83, op_sel:[0,1] op_sel_hi:[0,0] cbsz:4 blgp:4// MFMA m0_n3_k0
    s_waitcnt lgkmcnt(0)                          // wait B[ki=1] + A[m0,k1]
    v_mfma_scale_f32_16x16x128_f8f6f4 acc[0:3], v[68:71], v[36:39], acc[0:3], v80, v82, op_sel:[0,0] op_sel_hi:[1,1] cbsz:4 blgp:4// MFMA m0_n0_k1
    v_mfma_scale_f32_16x16x128_f8f6f4 acc[4:7], v[68:71], v[44:47], acc[4:7], v80, v82, op_sel:[0,1] op_sel_hi:[1,1] cbsz:4 blgp:4// MFMA m0_n1_k1
    v_mfma_scale_f32_16x16x128_f8f6f4 acc[8:11], v[68:71], v[52:55], acc[8:11], v80, v83, op_sel:[0,0] op_sel_hi:[1,1] cbsz:4 blgp:4// MFMA m0_n2_k1
    ds_read_b128 v[76:79], v7, offset:2112        // LR A m1k1 b1
    v_mfma_scale_f32_16x16x128_f8f6f4 acc[12:15], v[68:71], v[60:63], acc[12:15], v80, v83, op_sel:[0,1] op_sel_hi:[1,1] cbsz:4 blgp:4// MFMA m0_n3_k1
    s_waitcnt lgkmcnt(0)                          // wait A[m1]
    v_mfma_scale_f32_16x16x128_f8f6f4 acc[16:19], v[72:75], v[32:35], acc[16:19], v80, v82, op_sel:[1,0] op_sel_hi:[0,0] cbsz:4 blgp:4// MFMA m1_n0_k0
    v_mfma_scale_f32_16x16x128_f8f6f4 acc[20:23], v[72:75], v[40:43], acc[20:23], v80, v82, op_sel:[1,1] op_sel_hi:[0,0] cbsz:4 blgp:4// MFMA m1_n1_k0
    ds_read_b128 v[64:67], v7, offset:4096        // LR A m2k0 b0
    v_mfma_scale_f32_16x16x128_f8f6f4 acc[24:27], v[72:75], v[48:51], acc[24:27], v80, v83, op_sel:[1,0] op_sel_hi:[0,0] cbsz:4 blgp:4// MFMA m1_n2_k0
    v_mfma_scale_f32_16x16x128_f8f6f4 acc[28:31], v[72:75], v[56:59], acc[28:31], v80, v83, op_sel:[1,1] op_sel_hi:[0,0] cbsz:4 blgp:4// MFMA m1_n3_k0
    v_mfma_scale_f32_16x16x128_f8f6f4 acc[16:19], v[76:79], v[36:39], acc[16:19], v80, v82, op_sel:[1,0] op_sel_hi:[1,1] cbsz:4 blgp:4// MFMA m1_n0_k1
    v_mfma_scale_f32_16x16x128_f8f6f4 acc[20:23], v[76:79], v[44:47], acc[20:23], v80, v82, op_sel:[1,1] op_sel_hi:[1,1] cbsz:4 blgp:4// MFMA m1_n1_k1
    v_mfma_scale_f32_16x16x128_f8f6f4 acc[24:27], v[76:79], v[52:55], acc[24:27], v80, v83, op_sel:[1,0] op_sel_hi:[1,1] cbsz:4 blgp:4// MFMA m1_n2_k1
    ds_read_b128 v[68:71], v7, offset:4160        // LR A m2k1 b0
    v_mfma_scale_f32_16x16x128_f8f6f4 acc[28:31], v[76:79], v[60:63], acc[28:31], v80, v83, op_sel:[1,1] op_sel_hi:[1,1] cbsz:4 blgp:4// MFMA m1_n3_k1
    s_waitcnt lgkmcnt(0)                          // wait A[m2]
    // --- Partition 1 ---
    v_mfma_scale_f32_16x16x128_f8f6f4 acc[32:35], v[64:67], v[32:35], acc[32:35], v81, v82, op_sel:[0,0] op_sel_hi:[0,0] cbsz:4 blgp:4// MFMA m2_n0_k0
    v_mfma_scale_f32_16x16x128_f8f6f4 acc[36:39], v[64:67], v[40:43], acc[36:39], v81, v82, op_sel:[0,1] op_sel_hi:[0,0] cbsz:4 blgp:4// MFMA m2_n1_k0
    ds_read_b128 v[72:75], v7, offset:6144        // LR A m3k0 b1
    v_mfma_scale_f32_16x16x128_f8f6f4 acc[40:43], v[64:67], v[48:51], acc[40:43], v81, v83, op_sel:[0,0] op_sel_hi:[0,0] cbsz:4 blgp:4// MFMA m2_n2_k0
    v_mfma_scale_f32_16x16x128_f8f6f4 acc[44:47], v[64:67], v[56:59], acc[44:47], v81, v83, op_sel:[0,1] op_sel_hi:[0,0] cbsz:4 blgp:4// MFMA m2_n3_k0
    v_mfma_scale_f32_16x16x128_f8f6f4 acc[32:35], v[68:71], v[36:39], acc[32:35], v81, v82, op_sel:[0,0] op_sel_hi:[1,1] cbsz:4 blgp:4// MFMA m2_n0_k1
    v_mfma_scale_f32_16x16x128_f8f6f4 acc[36:39], v[68:71], v[44:47], acc[36:39], v81, v82, op_sel:[0,1] op_sel_hi:[1,1] cbsz:4 blgp:4// MFMA m2_n1_k1
    v_mfma_scale_f32_16x16x128_f8f6f4 acc[40:43], v[68:71], v[52:55], acc[40:43], v81, v83, op_sel:[0,0] op_sel_hi:[1,1] cbsz:4 blgp:4// MFMA m2_n2_k1
    ds_read_b128 v[76:79], v7, offset:6208        // LR A m3k1 b1
    v_mfma_scale_f32_16x16x128_f8f6f4 acc[44:47], v[68:71], v[60:63], acc[44:47], v81, v83, op_sel:[0,1] op_sel_hi:[1,1] cbsz:4 blgp:4// MFMA m2_n3_k1
    s_waitcnt lgkmcnt(0)                          // wait A[m3]
    v_mfma_scale_f32_16x16x128_f8f6f4 acc[48:51], v[72:75], v[32:35], acc[48:51], v81, v82, op_sel:[1,0] op_sel_hi:[0,0] cbsz:4 blgp:4// MFMA m3_n0_k0
    v_mfma_scale_f32_16x16x128_f8f6f4 acc[52:55], v[72:75], v[40:43], acc[52:55], v81, v82, op_sel:[1,1] op_sel_hi:[0,0] cbsz:4 blgp:4// MFMA m3_n1_k0
    v_mfma_scale_f32_16x16x128_f8f6f4 acc[56:59], v[72:75], v[48:51], acc[56:59], v81, v83, op_sel:[1,0] op_sel_hi:[0,0] cbsz:4 blgp:4// MFMA m3_n2_k0
    v_mfma_scale_f32_16x16x128_f8f6f4 acc[60:63], v[72:75], v[56:59], acc[60:63], v81, v83, op_sel:[1,1] op_sel_hi:[0,0] cbsz:4 blgp:4// MFMA m3_n3_k0
    s_waitcnt vmcnt(0)                            // wait DTL
    v_add_u32 v7, s50, v7                         // rd_a += db
    v_mfma_scale_f32_16x16x128_f8f6f4 acc[48:51], v[76:79], v[36:39], acc[48:51], v81, v82, op_sel:[1,0] op_sel_hi:[1,1] cbsz:4 blgp:4// MFMA m3_n0_k1
    v_add_u32 v8, s50, v8                         // rd_b += db
    v_add_u32 v23, s50, v23                       // rd_scale_a += db
    v_mfma_scale_f32_16x16x128_f8f6f4 acc[52:55], v[76:79], v[44:47], acc[52:55], v81, v82, op_sel:[1,1] op_sel_hi:[1,1] cbsz:4 blgp:4// MFMA m3_n1_k1
    v_add_u32 v24, s50, v24                       // rd_scale_b += db
    s_sub_u32 s50, 0, s50                         // negate db
    v_mfma_scale_f32_16x16x128_f8f6f4 acc[56:59], v[76:79], v[52:55], acc[56:59], v81, v83, op_sel:[1,0] op_sel_hi:[1,1] cbsz:4 blgp:4// MFMA m3_n2_k1
    v_mfma_scale_f32_16x16x128_f8f6f4 acc[60:63], v[76:79], v[60:63], acc[60:63], v81, v83, op_sel:[1,1] op_sel_hi:[1,1] cbsz:4 blgp:4// MFMA m3_n3_k1
    s_barrier                                     // sync
    s_cmp_lg_u32 s13, 0                           // more?
    s_cbranch_scc1 k_loop                         // loop

    // === Store D via buffer SRD ===
    // SRD for D matrix (raw buffer mode)
    s_mov_b32 s52, s8                             // SRD_D base lo
    s_mov_b32 s53, s9                             // SRD_D base hi
    s_mov_b32 s54, 0xFFFFFFFF                     // SRD_D size (unlimited)
    s_mov_b32 s55, 0x20000                        // SRD_D flags: raw buffer

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
    v_accvgpr_read_b32 v29, acc0                  // acc[0] m0_n0_a0
    v_accvgpr_read_b32 v30, acc1                  // acc[1] m0_n0_a1
    v_cvt_pk_bf16_f32 v28, v29, v30               // pk cvt a0,a1
    buffer_store_dword v28, v26, s[52:55], s15, offen// store D m0_n0_a0a1
    v_accvgpr_read_b32 v29, acc2                  // acc[2] m0_n0_a2
    v_accvgpr_read_b32 v30, acc3                  // acc[3] m0_n0_a3
    v_cvt_pk_bf16_f32 v28, v29, v30               // pk cvt a2,a3
    buffer_store_dword v28, v26, s[52:55], s15, offen offset:4// store D m0_n0_a2a3
    v_accvgpr_read_b32 v29, acc16                 // acc[16] m1_n0_a0
    v_accvgpr_read_b32 v30, acc17                 // acc[17] m1_n0_a1
    v_cvt_pk_bf16_f32 v28, v29, v30               // pk cvt a0,a1
    buffer_store_dword v28, v26, s[52:55], s15, offen offset:32// store D m1_n0_a0a1
    v_accvgpr_read_b32 v29, acc18                 // acc[18] m1_n0_a2
    v_accvgpr_read_b32 v30, acc19                 // acc[19] m1_n0_a3
    v_cvt_pk_bf16_f32 v28, v29, v30               // pk cvt a2,a3
    buffer_store_dword v28, v26, s[52:55], s15, offen offset:36// store D m1_n0_a2a3
    v_accvgpr_read_b32 v29, acc32                 // acc[32] m2_n0_a0
    v_accvgpr_read_b32 v30, acc33                 // acc[33] m2_n0_a1
    v_cvt_pk_bf16_f32 v28, v29, v30               // pk cvt a0,a1
    buffer_store_dword v28, v26, s[52:55], s15, offen offset:64// store D m2_n0_a0a1
    v_accvgpr_read_b32 v29, acc34                 // acc[34] m2_n0_a2
    v_accvgpr_read_b32 v30, acc35                 // acc[35] m2_n0_a3
    v_cvt_pk_bf16_f32 v28, v29, v30               // pk cvt a2,a3
    buffer_store_dword v28, v26, s[52:55], s15, offen offset:68// store D m2_n0_a2a3
    v_accvgpr_read_b32 v29, acc48                 // acc[48] m3_n0_a0
    v_accvgpr_read_b32 v30, acc49                 // acc[49] m3_n0_a1
    v_cvt_pk_bf16_f32 v28, v29, v30               // pk cvt a0,a1
    buffer_store_dword v28, v26, s[52:55], s15, offen offset:96// store D m3_n0_a0a1
    v_accvgpr_read_b32 v29, acc50                 // acc[50] m3_n0_a2
    v_accvgpr_read_b32 v30, acc51                 // acc[51] m3_n0_a3
    v_cvt_pk_bf16_f32 v28, v29, v30               // pk cvt a2,a3
    buffer_store_dword v28, v26, s[52:55], s15, offen offset:100// store D m3_n0_a2a3
    s_mul_i32 s15, s14, 16                        // soffset = 16 * col_stride (ni=1)
    v_accvgpr_read_b32 v29, acc4                  // acc[4] m0_n1_a0
    v_accvgpr_read_b32 v30, acc5                  // acc[5] m0_n1_a1
    v_cvt_pk_bf16_f32 v28, v29, v30               // pk cvt a0,a1
    buffer_store_dword v28, v26, s[52:55], s15, offen// store D m0_n1_a0a1
    v_accvgpr_read_b32 v29, acc6                  // acc[6] m0_n1_a2
    v_accvgpr_read_b32 v30, acc7                  // acc[7] m0_n1_a3
    v_cvt_pk_bf16_f32 v28, v29, v30               // pk cvt a2,a3
    buffer_store_dword v28, v26, s[52:55], s15, offen offset:4// store D m0_n1_a2a3
    v_accvgpr_read_b32 v29, acc20                 // acc[20] m1_n1_a0
    v_accvgpr_read_b32 v30, acc21                 // acc[21] m1_n1_a1
    v_cvt_pk_bf16_f32 v28, v29, v30               // pk cvt a0,a1
    buffer_store_dword v28, v26, s[52:55], s15, offen offset:32// store D m1_n1_a0a1
    v_accvgpr_read_b32 v29, acc22                 // acc[22] m1_n1_a2
    v_accvgpr_read_b32 v30, acc23                 // acc[23] m1_n1_a3
    v_cvt_pk_bf16_f32 v28, v29, v30               // pk cvt a2,a3
    buffer_store_dword v28, v26, s[52:55], s15, offen offset:36// store D m1_n1_a2a3
    v_accvgpr_read_b32 v29, acc36                 // acc[36] m2_n1_a0
    v_accvgpr_read_b32 v30, acc37                 // acc[37] m2_n1_a1
    v_cvt_pk_bf16_f32 v28, v29, v30               // pk cvt a0,a1
    buffer_store_dword v28, v26, s[52:55], s15, offen offset:64// store D m2_n1_a0a1
    v_accvgpr_read_b32 v29, acc38                 // acc[38] m2_n1_a2
    v_accvgpr_read_b32 v30, acc39                 // acc[39] m2_n1_a3
    v_cvt_pk_bf16_f32 v28, v29, v30               // pk cvt a2,a3
    buffer_store_dword v28, v26, s[52:55], s15, offen offset:68// store D m2_n1_a2a3
    v_accvgpr_read_b32 v29, acc52                 // acc[52] m3_n1_a0
    v_accvgpr_read_b32 v30, acc53                 // acc[53] m3_n1_a1
    v_cvt_pk_bf16_f32 v28, v29, v30               // pk cvt a0,a1
    buffer_store_dword v28, v26, s[52:55], s15, offen offset:96// store D m3_n1_a0a1
    v_accvgpr_read_b32 v29, acc54                 // acc[54] m3_n1_a2
    v_accvgpr_read_b32 v30, acc55                 // acc[55] m3_n1_a3
    v_cvt_pk_bf16_f32 v28, v29, v30               // pk cvt a2,a3
    buffer_store_dword v28, v26, s[52:55], s15, offen offset:100// store D m3_n1_a2a3
    s_mul_i32 s15, s14, 32                        // soffset = 32 * col_stride (ni=2)
    v_accvgpr_read_b32 v29, acc8                  // acc[8] m0_n2_a0
    v_accvgpr_read_b32 v30, acc9                  // acc[9] m0_n2_a1
    v_cvt_pk_bf16_f32 v28, v29, v30               // pk cvt a0,a1
    buffer_store_dword v28, v26, s[52:55], s15, offen// store D m0_n2_a0a1
    v_accvgpr_read_b32 v29, acc10                 // acc[10] m0_n2_a2
    v_accvgpr_read_b32 v30, acc11                 // acc[11] m0_n2_a3
    v_cvt_pk_bf16_f32 v28, v29, v30               // pk cvt a2,a3
    buffer_store_dword v28, v26, s[52:55], s15, offen offset:4// store D m0_n2_a2a3
    v_accvgpr_read_b32 v29, acc24                 // acc[24] m1_n2_a0
    v_accvgpr_read_b32 v30, acc25                 // acc[25] m1_n2_a1
    v_cvt_pk_bf16_f32 v28, v29, v30               // pk cvt a0,a1
    buffer_store_dword v28, v26, s[52:55], s15, offen offset:32// store D m1_n2_a0a1
    v_accvgpr_read_b32 v29, acc26                 // acc[26] m1_n2_a2
    v_accvgpr_read_b32 v30, acc27                 // acc[27] m1_n2_a3
    v_cvt_pk_bf16_f32 v28, v29, v30               // pk cvt a2,a3
    buffer_store_dword v28, v26, s[52:55], s15, offen offset:36// store D m1_n2_a2a3
    v_accvgpr_read_b32 v29, acc40                 // acc[40] m2_n2_a0
    v_accvgpr_read_b32 v30, acc41                 // acc[41] m2_n2_a1
    v_cvt_pk_bf16_f32 v28, v29, v30               // pk cvt a0,a1
    buffer_store_dword v28, v26, s[52:55], s15, offen offset:64// store D m2_n2_a0a1
    v_accvgpr_read_b32 v29, acc42                 // acc[42] m2_n2_a2
    v_accvgpr_read_b32 v30, acc43                 // acc[43] m2_n2_a3
    v_cvt_pk_bf16_f32 v28, v29, v30               // pk cvt a2,a3
    buffer_store_dword v28, v26, s[52:55], s15, offen offset:68// store D m2_n2_a2a3
    v_accvgpr_read_b32 v29, acc56                 // acc[56] m3_n2_a0
    v_accvgpr_read_b32 v30, acc57                 // acc[57] m3_n2_a1
    v_cvt_pk_bf16_f32 v28, v29, v30               // pk cvt a0,a1
    buffer_store_dword v28, v26, s[52:55], s15, offen offset:96// store D m3_n2_a0a1
    v_accvgpr_read_b32 v29, acc58                 // acc[58] m3_n2_a2
    v_accvgpr_read_b32 v30, acc59                 // acc[59] m3_n2_a3
    v_cvt_pk_bf16_f32 v28, v29, v30               // pk cvt a2,a3
    buffer_store_dword v28, v26, s[52:55], s15, offen offset:100// store D m3_n2_a2a3
    s_mul_i32 s15, s14, 48                        // soffset = 48 * col_stride (ni=3)
    v_accvgpr_read_b32 v29, acc12                 // acc[12] m0_n3_a0
    v_accvgpr_read_b32 v30, acc13                 // acc[13] m0_n3_a1
    v_cvt_pk_bf16_f32 v28, v29, v30               // pk cvt a0,a1
    buffer_store_dword v28, v26, s[52:55], s15, offen// store D m0_n3_a0a1
    v_accvgpr_read_b32 v29, acc14                 // acc[14] m0_n3_a2
    v_accvgpr_read_b32 v30, acc15                 // acc[15] m0_n3_a3
    v_cvt_pk_bf16_f32 v28, v29, v30               // pk cvt a2,a3
    buffer_store_dword v28, v26, s[52:55], s15, offen offset:4// store D m0_n3_a2a3
    v_accvgpr_read_b32 v29, acc28                 // acc[28] m1_n3_a0
    v_accvgpr_read_b32 v30, acc29                 // acc[29] m1_n3_a1
    v_cvt_pk_bf16_f32 v28, v29, v30               // pk cvt a0,a1
    buffer_store_dword v28, v26, s[52:55], s15, offen offset:32// store D m1_n3_a0a1
    v_accvgpr_read_b32 v29, acc30                 // acc[30] m1_n3_a2
    v_accvgpr_read_b32 v30, acc31                 // acc[31] m1_n3_a3
    v_cvt_pk_bf16_f32 v28, v29, v30               // pk cvt a2,a3
    buffer_store_dword v28, v26, s[52:55], s15, offen offset:36// store D m1_n3_a2a3
    v_accvgpr_read_b32 v29, acc44                 // acc[44] m2_n3_a0
    v_accvgpr_read_b32 v30, acc45                 // acc[45] m2_n3_a1
    v_cvt_pk_bf16_f32 v28, v29, v30               // pk cvt a0,a1
    buffer_store_dword v28, v26, s[52:55], s15, offen offset:64// store D m2_n3_a0a1
    v_accvgpr_read_b32 v29, acc46                 // acc[46] m2_n3_a2
    v_accvgpr_read_b32 v30, acc47                 // acc[47] m2_n3_a3
    v_cvt_pk_bf16_f32 v28, v29, v30               // pk cvt a2,a3
    buffer_store_dword v28, v26, s[52:55], s15, offen offset:68// store D m2_n3_a2a3
    v_accvgpr_read_b32 v29, acc60                 // acc[60] m3_n3_a0
    v_accvgpr_read_b32 v30, acc61                 // acc[61] m3_n3_a1
    v_cvt_pk_bf16_f32 v28, v29, v30               // pk cvt a0,a1
    buffer_store_dword v28, v26, s[52:55], s15, offen offset:96// store D m3_n3_a0a1
    v_accvgpr_read_b32 v29, acc62                 // acc[62] m3_n3_a2
    v_accvgpr_read_b32 v30, acc63                 // acc[63] m3_n3_a3
    v_cvt_pk_bf16_f32 v28, v29, v30               // pk cvt a2,a3
    buffer_store_dword v28, v26, s[52:55], s15, offen offset:100// store D m3_n3_a2a3
    s_waitcnt vmcnt(0)                            // wait for stores

    s_endpgm                                      // end of kernel

.rodata
.p2align 6
.amdhsa_kernel wave_mxfp4_128x128x256_kgen
    .amdhsa_group_segment_fixed_size 65536
    .amdhsa_private_segment_fixed_size 0
    .amdhsa_kernarg_size 136
    .amdhsa_user_sgpr_kernarg_segment_ptr 1
    .amdhsa_system_sgpr_workgroup_id_x 1
    .amdhsa_system_sgpr_workgroup_id_y 1
    .amdhsa_system_vgpr_workitem_id 0
    .amdhsa_next_free_vgpr 148
    .amdhsa_next_free_sgpr 56
    .amdhsa_accum_offset 84
    .amdhsa_float_denorm_mode_32 3
    .amdhsa_float_denorm_mode_16_64 3
.end_amdhsa_kernel

.amdgpu_metadata
---
amdhsa.version: [ 1, 2 ]
amdhsa.kernels:
  - .name:            wave_mxfp4_128x128x256_kgen
    .symbol:          wave_mxfp4_128x128x256_kgen.kd
    .sgpr_count:      56
    .vgpr_count:      148
    .agpr_count:      64
    .kernarg_segment_size: 136
    .kernarg_segment_align: 8
    .group_segment_fixed_size: 65536
    .private_segment_fixed_size: 0
    .wavefront_size:  64
    .max_flat_workgroup_size: 256
    .args:
      - .name:           Gemm info
        .offset:         0
        .size:           4
        .value_kind:     by_value
      - .name:           kernel info0
        .offset:         4
        .size:           4
        .value_kind:     by_value
      - .name:           kernel info1
        .offset:         8
        .size:           4
        .value_kind:     by_value
      - .name:           numWG
        .offset:         12
        .size:           4
        .value_kind:     by_value
      - .name:           SizesFree0
        .offset:         16
        .size:           4
        .value_kind:     by_value
      - .name:           SizesFree1
        .offset:         20
        .size:           4
        .value_kind:     by_value
      - .name:           SizesFree2
        .offset:         24
        .size:           4
        .value_kind:     by_value
      - .name:           SizesSum0
        .offset:         28
        .size:           4
        .value_kind:     by_value
      - .name:           D
        .offset:         32
        .size:           8
        .value_kind:     global_buffer
        .address_space:  generic
      - .name:           C
        .offset:         40
        .size:           8
        .value_kind:     global_buffer
        .address_space:  generic
      - .name:           A
        .offset:         48
        .size:           8
        .value_kind:     global_buffer
        .address_space:  generic
      - .name:           MXSA
        .offset:         56
        .size:           8
        .value_kind:     global_buffer
        .address_space:  generic
      - .name:           B
        .offset:         64
        .size:           8
        .value_kind:     global_buffer
        .address_space:  generic
      - .name:           MXSB
        .offset:         72
        .size:           8
        .value_kind:     global_buffer
        .address_space:  generic
      - .name:           strideD0
        .offset:         80
        .size:           4
        .value_kind:     by_value
      - .name:           strideD1
        .offset:         84
        .size:           4
        .value_kind:     by_value
      - .name:           strideC0
        .offset:         88
        .size:           4
        .value_kind:     by_value
      - .name:           strideC1
        .offset:         92
        .size:           4
        .value_kind:     by_value
      - .name:           strideA0
        .offset:         96
        .size:           4
        .value_kind:     by_value
      - .name:           strideA1
        .offset:         100
        .size:           4
        .value_kind:     by_value
      - .name:           strideMXSA0
        .offset:         104
        .size:           4
        .value_kind:     by_value
      - .name:           strideMXSA1
        .offset:         108
        .size:           4
        .value_kind:     by_value
      - .name:           strideB0
        .offset:         112
        .size:           4
        .value_kind:     by_value
      - .name:           strideB1
        .offset:         116
        .size:           4
        .value_kind:     by_value
      - .name:           strideMXSB0
        .offset:         120
        .size:           4
        .value_kind:     by_value
      - .name:           strideMXSB1
        .offset:         124
        .size:           4
        .value_kind:     by_value
      - .name:           alpha
        .offset:         128
        .size:           4
        .value_kind:     by_value
      - .name:           beta
        .offset:         132
        .size:           4
        .value_kind:     by_value
...
.end_amdgpu_metadata
