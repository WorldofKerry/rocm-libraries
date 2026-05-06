.amdgcn_target "amdgcn-amd-amdhsa--gfx950"
.text
.globl Custom_Cijk_Alik_Bljk_F4BS_MXA32_MXB32_MT256x256x256_MI16x16x1_kgen_gfx950
.p2align 8
.type Custom_Cijk_Alik_Bljk_F4BS_MXA32_MXB32_MT256x256x256_MI16x16x1_kgen_gfx950,@function

Custom_Cijk_Alik_Bljk_F4BS_MXA32_MXB32_MT256x256x256_MI16x16x1_kgen_gfx950:
    // === DTL Interleaved Setup ===
    s_load_dword s10, s[0:1], 16                  // M
    s_load_dword s11, s[0:1], 20                  // N
    s_load_dword s12, s[0:1], 28                  // K
    s_load_dwordx2 s[8:9], s[0:1], 32             // D ptr
    s_load_dwordx2 s[4:5], s[0:1], 48             // A ptr
    s_load_dwordx2 s[6:7], s[0:1], 64             // B ptr
    s_waitcnt lgkmcnt(0)                          // wait kernargs

    // WorkGroupMappingXCC=2: remap for L2 locality
    s_load_dword s46, s[0:1], 12                  // numWG (total workgroups)
    s_waitcnt lgkmcnt(0)                          // wait numWG
    s_lshr_b32 s14, s2, 1                         // old_wg / 2
    s_and_b32 s15, s2, 1                          // old_wg % 2 (XCC lane)
    s_lshr_b32 s46, s46, 1                        // numWG / 2
    s_mul_i32 s15, s15, s46                       // XCC_lane * (numWG / WGMXCC)
    s_add_u32 s2, s14, s15                        // remapped WG serial

    s_mov_b32 s15, s2                             // save wg_serial
    s_add_u32 s14, s10, 255                       // M + 255
    s_lshr_b32 s14, s14, 8                        // numWG_m = ceil(M/256)
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
    s_mul_i32 s14, s2, 256                        // wg_id * 256
    s_mul_i32 s14, s14, s26                       // * K*elem
    s_add_u32 s16, s4, s14                        // SRD_A lo
    s_addc_u32 s17, s5, 0                         // SRD_A hi
    s_mov_b32 s18, 0xFFFFFFFF                     // limit
    s_mov_b32 s19, 0x20000                        // flags

    // SRD B
    s_mul_i32 s14, s3, 256                        // wg_id * 256
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
    s_add_u32 s25, s24, 32768                     // LDS write base B = A + 32768

    // LDS read addresses
    v_and_b32 v29, 15, v2                         // lane_row = lane_id % 16
    v_lshrrev_b32 v30, 4, v2                      // k_group = lane_id / 16
    // LDS read A (paired-row swizzle)
    v_lshlrev_b32 v7, 7, v3                       // wave_m * 128
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
    v_lshlrev_b32 v41, 7, v3                      // m_row_base = wave_m * 128
    v_add_u32 v41, v41, v29                       // + lane_row (persistent)
    v_mov_b32 v42, v30                            // k_group (persistent)

    // LDS read B (paired-row swizzle)
    v_and_b32 v29, 15, v2                         // lane_row = lane_id % 16
    v_lshlrev_b32 v8, 7, v4                       // wave_n * 128
    v_add_u32 v8, v8, v29                         // + lane_row -> n_row_b
    v_lshrrev_b32 v31, 1, v8                      // lds_row = n_row / 2
    v_lshlrev_b32 v31, 8, v31                     // row_base = lds_row * 256
    s_mov_b32 s14, 32768                          // lds_b_off=32768
    v_add_u32 v31, v31, s14                       // + lds_b_offset=32768
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
    v_lshlrev_b32 v43, 7, v4                      // n_row_base = wave_n * 128
    v_and_b32 v29, 15, v2                         // lane_row (re-derive for save)
    v_add_u32 v43, v43, v29                       // + lane_row (persistent)


    // Init 256 accumulators
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
    v_accvgpr_write_b32 acc64, 0
    v_accvgpr_write_b32 acc65, 0
    v_accvgpr_write_b32 acc66, 0
    v_accvgpr_write_b32 acc67, 0
    v_accvgpr_write_b32 acc68, 0
    v_accvgpr_write_b32 acc69, 0
    v_accvgpr_write_b32 acc70, 0
    v_accvgpr_write_b32 acc71, 0
    v_accvgpr_write_b32 acc72, 0
    v_accvgpr_write_b32 acc73, 0
    v_accvgpr_write_b32 acc74, 0
    v_accvgpr_write_b32 acc75, 0
    v_accvgpr_write_b32 acc76, 0
    v_accvgpr_write_b32 acc77, 0
    v_accvgpr_write_b32 acc78, 0
    v_accvgpr_write_b32 acc79, 0
    v_accvgpr_write_b32 acc80, 0
    v_accvgpr_write_b32 acc81, 0
    v_accvgpr_write_b32 acc82, 0
    v_accvgpr_write_b32 acc83, 0
    v_accvgpr_write_b32 acc84, 0
    v_accvgpr_write_b32 acc85, 0
    v_accvgpr_write_b32 acc86, 0
    v_accvgpr_write_b32 acc87, 0
    v_accvgpr_write_b32 acc88, 0
    v_accvgpr_write_b32 acc89, 0
    v_accvgpr_write_b32 acc90, 0
    v_accvgpr_write_b32 acc91, 0
    v_accvgpr_write_b32 acc92, 0
    v_accvgpr_write_b32 acc93, 0
    v_accvgpr_write_b32 acc94, 0
    v_accvgpr_write_b32 acc95, 0
    v_accvgpr_write_b32 acc96, 0
    v_accvgpr_write_b32 acc97, 0
    v_accvgpr_write_b32 acc98, 0
    v_accvgpr_write_b32 acc99, 0
    v_accvgpr_write_b32 acc100, 0
    v_accvgpr_write_b32 acc101, 0
    v_accvgpr_write_b32 acc102, 0
    v_accvgpr_write_b32 acc103, 0
    v_accvgpr_write_b32 acc104, 0
    v_accvgpr_write_b32 acc105, 0
    v_accvgpr_write_b32 acc106, 0
    v_accvgpr_write_b32 acc107, 0
    v_accvgpr_write_b32 acc108, 0
    v_accvgpr_write_b32 acc109, 0
    v_accvgpr_write_b32 acc110, 0
    v_accvgpr_write_b32 acc111, 0
    v_accvgpr_write_b32 acc112, 0
    v_accvgpr_write_b32 acc113, 0
    v_accvgpr_write_b32 acc114, 0
    v_accvgpr_write_b32 acc115, 0
    v_accvgpr_write_b32 acc116, 0
    v_accvgpr_write_b32 acc117, 0
    v_accvgpr_write_b32 acc118, 0
    v_accvgpr_write_b32 acc119, 0
    v_accvgpr_write_b32 acc120, 0
    v_accvgpr_write_b32 acc121, 0
    v_accvgpr_write_b32 acc122, 0
    v_accvgpr_write_b32 acc123, 0
    v_accvgpr_write_b32 acc124, 0
    v_accvgpr_write_b32 acc125, 0
    v_accvgpr_write_b32 acc126, 0
    v_accvgpr_write_b32 acc127, 0
    v_accvgpr_write_b32 acc128, 0
    v_accvgpr_write_b32 acc129, 0
    v_accvgpr_write_b32 acc130, 0
    v_accvgpr_write_b32 acc131, 0
    v_accvgpr_write_b32 acc132, 0
    v_accvgpr_write_b32 acc133, 0
    v_accvgpr_write_b32 acc134, 0
    v_accvgpr_write_b32 acc135, 0
    v_accvgpr_write_b32 acc136, 0
    v_accvgpr_write_b32 acc137, 0
    v_accvgpr_write_b32 acc138, 0
    v_accvgpr_write_b32 acc139, 0
    v_accvgpr_write_b32 acc140, 0
    v_accvgpr_write_b32 acc141, 0
    v_accvgpr_write_b32 acc142, 0
    v_accvgpr_write_b32 acc143, 0
    v_accvgpr_write_b32 acc144, 0
    v_accvgpr_write_b32 acc145, 0
    v_accvgpr_write_b32 acc146, 0
    v_accvgpr_write_b32 acc147, 0
    v_accvgpr_write_b32 acc148, 0
    v_accvgpr_write_b32 acc149, 0
    v_accvgpr_write_b32 acc150, 0
    v_accvgpr_write_b32 acc151, 0
    v_accvgpr_write_b32 acc152, 0
    v_accvgpr_write_b32 acc153, 0
    v_accvgpr_write_b32 acc154, 0
    v_accvgpr_write_b32 acc155, 0
    v_accvgpr_write_b32 acc156, 0
    v_accvgpr_write_b32 acc157, 0
    v_accvgpr_write_b32 acc158, 0
    v_accvgpr_write_b32 acc159, 0
    v_accvgpr_write_b32 acc160, 0
    v_accvgpr_write_b32 acc161, 0
    v_accvgpr_write_b32 acc162, 0
    v_accvgpr_write_b32 acc163, 0
    v_accvgpr_write_b32 acc164, 0
    v_accvgpr_write_b32 acc165, 0
    v_accvgpr_write_b32 acc166, 0
    v_accvgpr_write_b32 acc167, 0
    v_accvgpr_write_b32 acc168, 0
    v_accvgpr_write_b32 acc169, 0
    v_accvgpr_write_b32 acc170, 0
    v_accvgpr_write_b32 acc171, 0
    v_accvgpr_write_b32 acc172, 0
    v_accvgpr_write_b32 acc173, 0
    v_accvgpr_write_b32 acc174, 0
    v_accvgpr_write_b32 acc175, 0
    v_accvgpr_write_b32 acc176, 0
    v_accvgpr_write_b32 acc177, 0
    v_accvgpr_write_b32 acc178, 0
    v_accvgpr_write_b32 acc179, 0
    v_accvgpr_write_b32 acc180, 0
    v_accvgpr_write_b32 acc181, 0
    v_accvgpr_write_b32 acc182, 0
    v_accvgpr_write_b32 acc183, 0
    v_accvgpr_write_b32 acc184, 0
    v_accvgpr_write_b32 acc185, 0
    v_accvgpr_write_b32 acc186, 0
    v_accvgpr_write_b32 acc187, 0
    v_accvgpr_write_b32 acc188, 0
    v_accvgpr_write_b32 acc189, 0
    v_accvgpr_write_b32 acc190, 0
    v_accvgpr_write_b32 acc191, 0
    v_accvgpr_write_b32 acc192, 0
    v_accvgpr_write_b32 acc193, 0
    v_accvgpr_write_b32 acc194, 0
    v_accvgpr_write_b32 acc195, 0
    v_accvgpr_write_b32 acc196, 0
    v_accvgpr_write_b32 acc197, 0
    v_accvgpr_write_b32 acc198, 0
    v_accvgpr_write_b32 acc199, 0
    v_accvgpr_write_b32 acc200, 0
    v_accvgpr_write_b32 acc201, 0
    v_accvgpr_write_b32 acc202, 0
    v_accvgpr_write_b32 acc203, 0
    v_accvgpr_write_b32 acc204, 0
    v_accvgpr_write_b32 acc205, 0
    v_accvgpr_write_b32 acc206, 0
    v_accvgpr_write_b32 acc207, 0
    v_accvgpr_write_b32 acc208, 0
    v_accvgpr_write_b32 acc209, 0
    v_accvgpr_write_b32 acc210, 0
    v_accvgpr_write_b32 acc211, 0
    v_accvgpr_write_b32 acc212, 0
    v_accvgpr_write_b32 acc213, 0
    v_accvgpr_write_b32 acc214, 0
    v_accvgpr_write_b32 acc215, 0
    v_accvgpr_write_b32 acc216, 0
    v_accvgpr_write_b32 acc217, 0
    v_accvgpr_write_b32 acc218, 0
    v_accvgpr_write_b32 acc219, 0
    v_accvgpr_write_b32 acc220, 0
    v_accvgpr_write_b32 acc221, 0
    v_accvgpr_write_b32 acc222, 0
    v_accvgpr_write_b32 acc223, 0
    v_accvgpr_write_b32 acc224, 0
    v_accvgpr_write_b32 acc225, 0
    v_accvgpr_write_b32 acc226, 0
    v_accvgpr_write_b32 acc227, 0
    v_accvgpr_write_b32 acc228, 0
    v_accvgpr_write_b32 acc229, 0
    v_accvgpr_write_b32 acc230, 0
    v_accvgpr_write_b32 acc231, 0
    v_accvgpr_write_b32 acc232, 0
    v_accvgpr_write_b32 acc233, 0
    v_accvgpr_write_b32 acc234, 0
    v_accvgpr_write_b32 acc235, 0
    v_accvgpr_write_b32 acc236, 0
    v_accvgpr_write_b32 acc237, 0
    v_accvgpr_write_b32 acc238, 0
    v_accvgpr_write_b32 acc239, 0
    v_accvgpr_write_b32 acc240, 0
    v_accvgpr_write_b32 acc241, 0
    v_accvgpr_write_b32 acc242, 0
    v_accvgpr_write_b32 acc243, 0
    v_accvgpr_write_b32 acc244, 0
    v_accvgpr_write_b32 acc245, 0
    v_accvgpr_write_b32 acc246, 0
    v_accvgpr_write_b32 acc247, 0
    v_accvgpr_write_b32 acc248, 0
    v_accvgpr_write_b32 acc249, 0
    v_accvgpr_write_b32 acc250, 0
    v_accvgpr_write_b32 acc251, 0
    v_accvgpr_write_b32 acc252, 0
    v_accvgpr_write_b32 acc253, 0
    v_accvgpr_write_b32 acc254, 0
    v_accvgpr_write_b32 acc255, 0

    // Init MX constant scale = 1.0 (E8M0 0x7F)
    v_mov_b32 v20, 0x7F7F7F7F                     // scale = 1.0 for all byte lanes

    // === MX Scale Setup (direct VGPR, no LDS) ===
    s_load_dwordx2 s[30:31], s[0:1], 56           // scale A ptr (MXSA)
    s_load_dwordx2 s[32:33], s[0:1], 72           // scale B ptr (MXSB)
    s_load_dword s34, s[0:1], 104                 // strideMXSA0
    s_load_dword s35, s[0:1], 120                 // strideMXSB0
    s_waitcnt lgkmcnt(0)                          // wait scale kernargs

    // Scale SRD A
    s_mul_i32 s14, s2, 256                        // wg_id_x * 256
    s_mul_i32 s14, s14, s34                       // * stride_scale_a
    s_add_u32 s36, s30, s14                       // SRD_scaleA lo
    s_addc_u32 s37, s31, 0                        // SRD_scaleA hi
    s_mov_b32 s38, 0xFFFFFFFF                     // limit
    s_mov_b32 s39, 0x20000                        // flags

    // Scale SRD B
    s_mul_i32 s14, s3, 256                        // wg_id_y * 256
    s_mul_i32 s14, s14, s35                       // * stride_scale_b
    s_add_u32 s40, s32, s14                       // SRD_scaleB lo
    s_addc_u32 s41, s33, 0                        // SRD_scaleB hi
    s_mov_b32 s42, 0xFFFFFFFF                     // limit
    s_mov_b32 s43, 0x20000                        // flags

    // Scale A wave-level voffset
    v_lshlrev_b32 v29, 7, v3                      // wave_m * 128
    v_mul_lo_u32 v21, s34, v29                    // wave_m_base * stride_scale_a -> voffset_scale_a

    // Scale B wave-level voffset
    v_lshlrev_b32 v29, 7, v4                      // wave_n * 128
    v_mul_lo_u32 v22, s35, v29                    // wave_n_base * stride_scale_b -> voffset_scale_b

    // Scale LDS write bases (within buffer 0)
    s_mov_b32 s48, 65536                          // scale A LDS write base = 65536
    s_mov_b32 s49, 69632                          // scale B LDS write base = 69632

    // Precompute DTL soffsets
    s_mul_i32 s50, s27, 1                         // dtl_soff_a[1] = 1 * soffset_a
    s_mul_i32 s51, s27, 2                         // dtl_soff_a[2] = 2 * soffset_a
    s_mul_i32 s52, s27, 3                         // dtl_soff_a[3] = 3 * soffset_a
    s_mul_i32 s53, s27, 4                         // dtl_soff_a[4] = 4 * soffset_a
    s_mul_i32 s54, s27, 5                         // dtl_soff_a[5] = 5 * soffset_a
    s_mul_i32 s55, s27, 6                         // dtl_soff_a[6] = 6 * soffset_a
    s_mul_i32 s56, s27, 7                         // dtl_soff_a[7] = 7 * soffset_a
    s_mul_i32 s57, s28, 1                         // dtl_soff_b[1] = 1 * soffset_b
    s_mul_i32 s58, s28, 2                         // dtl_soff_b[2] = 2 * soffset_b
    s_mul_i32 s59, s28, 3                         // dtl_soff_b[3] = 3 * soffset_b
    s_mul_i32 s60, s28, 4                         // dtl_soff_b[4] = 4 * soffset_b
    s_mul_i32 s61, s28, 5                         // dtl_soff_b[5] = 5 * soffset_b
    s_mul_i32 s62, s28, 6                         // dtl_soff_b[6] = 6 * soffset_b
    s_mul_i32 s63, s28, 7                         // dtl_soff_b[7] = 7 * soffset_b

    // Scale DTL voffset (strided): (tid%16)*16 + (tid/16)*stride
    v_and_b32 v29, 15, v0                         // tid % 16
    v_lshlrev_b32 v29, 4, v29                     // * 16 -> intra-group byte offset
    v_lshrrev_b32 v30, 4, v0                      // tid / 16 (group index)
    v_mul_lo_u32 v30, s34, v30                    // * stride_scale_a
    v_add_u32 v164, v29, v30                      // scale A voffset
    v_lshrrev_b32 v30, 4, v0                      // tid / 16
    v_mul_lo_u32 v30, s35, v30                    // * stride_scale_b
    v_add_u32 v165, v29, v30                      // scale B voffset

    // Scale ds_read base (pre-swizzled LDS)
    v_and_b32 v29, v2, 63                         // laneId = lane_id & 63
    v_lshlrev_b32 v29, 2, v29                     // laneId * 4
    v_lshlrev_b32 v30, 10, v3                     // wave_m * 1024 (partition offset)
    v_add_u32 v166, v29, v30                      // laneId*4 + partition_m
    s_mov_b32 s14, 65536                          // scale_a_lds_off = 65536
    v_add_u32 v166, v166, s14                     // + lds_base_a
    v_lshlrev_b32 v30, 10, v4                     // wave_n * 1024 (partition offset)
    v_add_u32 v167, v29, v30                      // laneId*4 + partition_n
    s_mov_b32 s14, 69632                          // scale_b_lds_off = 69632
    v_add_u32 v167, v167, s14                     // + lds_base_b

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
    v_add_u32 v31, v41, 64                        // m_row = base + 64 (mi=4)
    v_lshrrev_b32 v32, 1, v31                     // lds_row = m_row / 2
    v_lshlrev_b32 v32, 8, v32                     // row_base = lds_row * 256
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
    s_mov_b32 s14, 80
    v_add_u32 v31, v41, s14                       // m_row = base + 80 (mi=5)
    v_lshrrev_b32 v32, 1, v31                     // lds_row = m_row / 2
    v_lshlrev_b32 v32, 8, v32                     // row_base = lds_row * 256
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
    s_mov_b32 s14, 96
    v_add_u32 v31, v41, s14                       // m_row = base + 96 (mi=6)
    v_lshrrev_b32 v32, 1, v31                     // lds_row = m_row / 2
    v_lshlrev_b32 v32, 8, v32                     // row_base = lds_row * 256
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
    s_mov_b32 s14, 112
    v_add_u32 v31, v41, s14                       // m_row = base + 112 (mi=7)
    v_lshrrev_b32 v32, 1, v31                     // lds_row = m_row / 2
    v_lshlrev_b32 v32, 8, v32                     // row_base = lds_row * 256
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

    // Precompute swizzled B read addresses
    v_mov_b32 v31, v43                            // n_row = base (ni=0)
    v_lshrrev_b32 v32, 1, v31                     // lds_row = n_row / 2
    v_lshlrev_b32 v32, 8, v32                     // row_base = lds_row * 256
    s_mov_b32 s14, 32768                          // lds_b_off=32768
    v_add_u32 v32, v32, s14                       // + lds_b_offset
    v_mov_b32 v35, v42                            // k_col = k_group
    v_and_b32 v37, 1, v31                         // m_row % 2
    v_lshlrev_b32 v37, 3, v37                     // * 8 -> half_offset
    v_lshrrev_b32 v38, 1, v31                     // lds_row = m_row / 2
    v_add_u32 v35, v37, v35                       // half_offset + k_col
    v_add_u32 v35, v35, v38                       // + lds_row (rotation)
    v_and_b32 v35, 15, v35                        // % 16
    v_lshlrev_b32 v60, 4, v35                     // swizzled_col * 16
    v_add_u32 v60, v60, v32                       // + row_base
    v_add_u32 v38, v35, 4                         // k_col + 4 (ki=1)
    v_and_b32 v38, 15, v38                        // % 16
    v_lshlrev_b32 v61, 4, v38                     // swizzled_col * 16
    v_add_u32 v61, v61, v32                       // + row_base
    v_add_u32 v31, v43, 16                        // n_row = base + 16 (ni=1)
    v_lshrrev_b32 v32, 1, v31                     // lds_row = n_row / 2
    v_lshlrev_b32 v32, 8, v32                     // row_base = lds_row * 256
    s_mov_b32 s14, 32768                          // lds_b_off=32768
    v_add_u32 v32, v32, s14                       // + lds_b_offset
    v_mov_b32 v35, v42                            // k_col = k_group
    v_and_b32 v37, 1, v31                         // m_row % 2
    v_lshlrev_b32 v37, 3, v37                     // * 8 -> half_offset
    v_lshrrev_b32 v38, 1, v31                     // lds_row = m_row / 2
    v_add_u32 v35, v37, v35                       // half_offset + k_col
    v_add_u32 v35, v35, v38                       // + lds_row (rotation)
    v_and_b32 v35, 15, v35                        // % 16
    v_lshlrev_b32 v62, 4, v35                     // swizzled_col * 16
    v_add_u32 v62, v62, v32                       // + row_base
    v_add_u32 v38, v35, 4                         // k_col + 4 (ki=1)
    v_and_b32 v38, 15, v38                        // % 16
    v_lshlrev_b32 v63, 4, v38                     // swizzled_col * 16
    v_add_u32 v63, v63, v32                       // + row_base
    v_add_u32 v31, v43, 32                        // n_row = base + 32 (ni=2)
    v_lshrrev_b32 v32, 1, v31                     // lds_row = n_row / 2
    v_lshlrev_b32 v32, 8, v32                     // row_base = lds_row * 256
    s_mov_b32 s14, 32768                          // lds_b_off=32768
    v_add_u32 v32, v32, s14                       // + lds_b_offset
    v_mov_b32 v35, v42                            // k_col = k_group
    v_and_b32 v37, 1, v31                         // m_row % 2
    v_lshlrev_b32 v37, 3, v37                     // * 8 -> half_offset
    v_lshrrev_b32 v38, 1, v31                     // lds_row = m_row / 2
    v_add_u32 v35, v37, v35                       // half_offset + k_col
    v_add_u32 v35, v35, v38                       // + lds_row (rotation)
    v_and_b32 v35, 15, v35                        // % 16
    v_lshlrev_b32 v64, 4, v35                     // swizzled_col * 16
    v_add_u32 v64, v64, v32                       // + row_base
    v_add_u32 v38, v35, 4                         // k_col + 4 (ki=1)
    v_and_b32 v38, 15, v38                        // % 16
    v_lshlrev_b32 v65, 4, v38                     // swizzled_col * 16
    v_add_u32 v65, v65, v32                       // + row_base
    v_add_u32 v31, v43, 48                        // n_row = base + 48 (ni=3)
    v_lshrrev_b32 v32, 1, v31                     // lds_row = n_row / 2
    v_lshlrev_b32 v32, 8, v32                     // row_base = lds_row * 256
    s_mov_b32 s14, 32768                          // lds_b_off=32768
    v_add_u32 v32, v32, s14                       // + lds_b_offset
    v_mov_b32 v35, v42                            // k_col = k_group
    v_and_b32 v37, 1, v31                         // m_row % 2
    v_lshlrev_b32 v37, 3, v37                     // * 8 -> half_offset
    v_lshrrev_b32 v38, 1, v31                     // lds_row = m_row / 2
    v_add_u32 v35, v37, v35                       // half_offset + k_col
    v_add_u32 v35, v35, v38                       // + lds_row (rotation)
    v_and_b32 v35, 15, v35                        // % 16
    v_lshlrev_b32 v66, 4, v35                     // swizzled_col * 16
    v_add_u32 v66, v66, v32                       // + row_base
    v_add_u32 v38, v35, 4                         // k_col + 4 (ki=1)
    v_and_b32 v38, 15, v38                        // % 16
    v_lshlrev_b32 v67, 4, v38                     // swizzled_col * 16
    v_add_u32 v67, v67, v32                       // + row_base
    v_add_u32 v31, v43, 64                        // n_row = base + 64 (ni=4)
    v_lshrrev_b32 v32, 1, v31                     // lds_row = n_row / 2
    v_lshlrev_b32 v32, 8, v32                     // row_base = lds_row * 256
    s_mov_b32 s14, 32768                          // lds_b_off=32768
    v_add_u32 v32, v32, s14                       // + lds_b_offset
    v_mov_b32 v35, v42                            // k_col = k_group
    v_and_b32 v37, 1, v31                         // m_row % 2
    v_lshlrev_b32 v37, 3, v37                     // * 8 -> half_offset
    v_lshrrev_b32 v38, 1, v31                     // lds_row = m_row / 2
    v_add_u32 v35, v37, v35                       // half_offset + k_col
    v_add_u32 v35, v35, v38                       // + lds_row (rotation)
    v_and_b32 v35, 15, v35                        // % 16
    v_lshlrev_b32 v68, 4, v35                     // swizzled_col * 16
    v_add_u32 v68, v68, v32                       // + row_base
    v_add_u32 v38, v35, 4                         // k_col + 4 (ki=1)
    v_and_b32 v38, 15, v38                        // % 16
    v_lshlrev_b32 v69, 4, v38                     // swizzled_col * 16
    v_add_u32 v69, v69, v32                       // + row_base
    s_mov_b32 s14, 80
    v_add_u32 v31, v43, s14                       // n_row = base + 80 (ni=5)
    v_lshrrev_b32 v32, 1, v31                     // lds_row = n_row / 2
    v_lshlrev_b32 v32, 8, v32                     // row_base = lds_row * 256
    s_mov_b32 s14, 32768                          // lds_b_off=32768
    v_add_u32 v32, v32, s14                       // + lds_b_offset
    v_mov_b32 v35, v42                            // k_col = k_group
    v_and_b32 v37, 1, v31                         // m_row % 2
    v_lshlrev_b32 v37, 3, v37                     // * 8 -> half_offset
    v_lshrrev_b32 v38, 1, v31                     // lds_row = m_row / 2
    v_add_u32 v35, v37, v35                       // half_offset + k_col
    v_add_u32 v35, v35, v38                       // + lds_row (rotation)
    v_and_b32 v35, 15, v35                        // % 16
    v_lshlrev_b32 v70, 4, v35                     // swizzled_col * 16
    v_add_u32 v70, v70, v32                       // + row_base
    v_add_u32 v38, v35, 4                         // k_col + 4 (ki=1)
    v_and_b32 v38, 15, v38                        // % 16
    v_lshlrev_b32 v71, 4, v38                     // swizzled_col * 16
    v_add_u32 v71, v71, v32                       // + row_base
    s_mov_b32 s14, 96
    v_add_u32 v31, v43, s14                       // n_row = base + 96 (ni=6)
    v_lshrrev_b32 v32, 1, v31                     // lds_row = n_row / 2
    v_lshlrev_b32 v32, 8, v32                     // row_base = lds_row * 256
    s_mov_b32 s14, 32768                          // lds_b_off=32768
    v_add_u32 v32, v32, s14                       // + lds_b_offset
    v_mov_b32 v35, v42                            // k_col = k_group
    v_and_b32 v37, 1, v31                         // m_row % 2
    v_lshlrev_b32 v37, 3, v37                     // * 8 -> half_offset
    v_lshrrev_b32 v38, 1, v31                     // lds_row = m_row / 2
    v_add_u32 v35, v37, v35                       // half_offset + k_col
    v_add_u32 v35, v35, v38                       // + lds_row (rotation)
    v_and_b32 v35, 15, v35                        // % 16
    v_lshlrev_b32 v72, 4, v35                     // swizzled_col * 16
    v_add_u32 v72, v72, v32                       // + row_base
    v_add_u32 v38, v35, 4                         // k_col + 4 (ki=1)
    v_and_b32 v38, 15, v38                        // % 16
    v_lshlrev_b32 v73, 4, v38                     // swizzled_col * 16
    v_add_u32 v73, v73, v32                       // + row_base
    s_mov_b32 s14, 112
    v_add_u32 v31, v43, s14                       // n_row = base + 112 (ni=7)
    v_lshrrev_b32 v32, 1, v31                     // lds_row = n_row / 2
    v_lshlrev_b32 v32, 8, v32                     // row_base = lds_row * 256
    s_mov_b32 s14, 32768                          // lds_b_off=32768
    v_add_u32 v32, v32, s14                       // + lds_b_offset
    v_mov_b32 v35, v42                            // k_col = k_group
    v_and_b32 v37, 1, v31                         // m_row % 2
    v_lshlrev_b32 v37, 3, v37                     // * 8 -> half_offset
    v_lshrrev_b32 v38, 1, v31                     // lds_row = m_row / 2
    v_add_u32 v35, v37, v35                       // half_offset + k_col
    v_add_u32 v35, v35, v38                       // + lds_row (rotation)
    v_and_b32 v35, 15, v35                        // % 16
    v_lshlrev_b32 v74, 4, v35                     // swizzled_col * 16
    v_add_u32 v74, v74, v32                       // + row_base
    v_add_u32 v38, v35, 4                         // k_col + 4 (ki=1)
    v_and_b32 v38, 15, v38                        // % 16
    v_lshlrev_b32 v75, 4, v38                     // swizzled_col * 16
    v_add_u32 v75, v75, v32                       // + row_base

    s_mov_b32 s65, 0                              // rd_db = 0
    s_mov_b32 s64, 73728                          // DB step = 73728

    s_lshr_b32 s13, s12, 8                        // k_tiles = K / 256

    // Pipeline ramp-up stage 0/2
    buffer_load_dword v29, v164, s[36:39], 0, offen offset:0// scale A dword 0
    buffer_load_dword v30, v164, s[36:39], 0, offen offset:4// scale A dword 1
    buffer_load_dword v31, v164, s[36:39], 0, offen offset:8// scale A dword 2
    buffer_load_dword v32, v164, s[36:39], 0, offen offset:12// scale A dword 3
    s_waitcnt vmcnt(0)                            // wait global loads before ds_write
    v_lshlrev_b32 v37, 4, v0                      // tid * 16
    v_add_u32 v37, v37, s48                       // LDS addr = wr_base_a + tid*16
    ds_write_b32 v37, v29, offset:0               // scale A dw0 -> LDS
    ds_write_b32 v37, v30, offset:4               // scale A dw1 -> LDS
    ds_write_b32 v37, v31, offset:8               // scale A dw2 -> LDS
    ds_write_b32 v37, v32, offset:12              // scale A dw3 -> LDS
    buffer_load_dword v33, v165, s[40:43], 0, offen offset:0// scale B dword 0
    buffer_load_dword v34, v165, s[40:43], 0, offen offset:4// scale B dword 1
    buffer_load_dword v35, v165, s[40:43], 0, offen offset:8// scale B dword 2
    buffer_load_dword v36, v165, s[40:43], 0, offen offset:12// scale B dword 3
    s_waitcnt vmcnt(0)                            // wait global loads before ds_write
    v_lshlrev_b32 v38, 4, v0                      // tid * 16
    v_add_u32 v38, v38, s49                       // LDS addr = wr_base_b + tid*16
    ds_write_b32 v38, v33, offset:0               // scale B dw0 -> LDS
    ds_write_b32 v38, v34, offset:4               // scale B dw1 -> LDS
    ds_write_b32 v38, v35, offset:8               // scale B dw2 -> LDS
    ds_write_b32 v38, v36, offset:12              // scale B dw3 -> LDS
    s_mov_b32 m0, s24                             // m0 = LDS base A
    buffer_load_dwordx4 v5, s[16:19], 0, offen offset:0, lds// DTL A[0]
    s_add_u32 m0, m0, 4096                        // m0 += 4096
    buffer_load_dwordx4 v5, s[16:19], s50, offen offset:0, lds// DTL A[1]
    s_add_u32 m0, m0, 4096                        // m0 += 4096
    buffer_load_dwordx4 v5, s[16:19], s51, offen offset:0, lds// DTL A[2]
    s_add_u32 m0, m0, 4096                        // m0 += 4096
    buffer_load_dwordx4 v5, s[16:19], s52, offen offset:0, lds// DTL A[3]
    s_add_u32 m0, m0, 4096                        // m0 += 4096
    buffer_load_dwordx4 v5, s[16:19], s53, offen offset:0, lds// DTL A[4]
    s_add_u32 m0, m0, 4096                        // m0 += 4096
    buffer_load_dwordx4 v5, s[16:19], s54, offen offset:0, lds// DTL A[5]
    s_add_u32 m0, m0, 4096                        // m0 += 4096
    buffer_load_dwordx4 v5, s[16:19], s55, offen offset:0, lds// DTL A[6]
    s_add_u32 m0, m0, 4096                        // m0 += 4096
    buffer_load_dwordx4 v5, s[16:19], s56, offen offset:0, lds// DTL A[7]
    s_mov_b32 m0, s25                             // m0 = LDS base B
    buffer_load_dwordx4 v6, s[20:23], 0, offen offset:0, lds// DTL B[0]
    s_add_u32 m0, m0, 4096                        // m0 += 4096
    buffer_load_dwordx4 v6, s[20:23], s57, offen offset:0, lds// DTL B[1]
    s_add_u32 m0, m0, 4096                        // m0 += 4096
    buffer_load_dwordx4 v6, s[20:23], s58, offen offset:0, lds// DTL B[2]
    s_add_u32 m0, m0, 4096                        // m0 += 4096
    buffer_load_dwordx4 v6, s[20:23], s59, offen offset:0, lds// DTL B[3]
    s_add_u32 m0, m0, 4096                        // m0 += 4096
    buffer_load_dwordx4 v6, s[20:23], s60, offen offset:0, lds// DTL B[4]
    s_add_u32 m0, m0, 4096                        // m0 += 4096
    buffer_load_dwordx4 v6, s[20:23], s61, offen offset:0, lds// DTL B[5]
    s_add_u32 m0, m0, 4096                        // m0 += 4096
    buffer_load_dwordx4 v6, s[20:23], s62, offen offset:0, lds// DTL B[6]
    s_add_u32 m0, m0, 4096                        // m0 += 4096
    buffer_load_dwordx4 v6, s[20:23], s63, offen offset:0, lds// DTL B[7]
    s_waitcnt vmcnt(0)                            // wait all loads
    s_waitcnt lgkmcnt(0)                          // wait LDS writes
    s_barrier                                     // sync tile 0

    // Pipeline ramp-up stage 1/2
    s_cmp_le_u32 s13, 1                           // skip if k_tiles <= 1
    s_cbranch_scc1 pgr_skip_1                     // not enough tiles for stage 1
    s_add_u32 s36, s36, 256                       // s_srd_scale_a += 256
    s_addc_u32 s37, s37, 0                        // carry
    s_add_u32 s48, s48, s64                       // wr_scale_a += db_step
    buffer_load_dword v29, v164, s[36:39], 0, offen offset:0// scale A dword 0
    buffer_load_dword v30, v164, s[36:39], 0, offen offset:4// scale A dword 1
    buffer_load_dword v31, v164, s[36:39], 0, offen offset:8// scale A dword 2
    buffer_load_dword v32, v164, s[36:39], 0, offen offset:12// scale A dword 3
    s_waitcnt vmcnt(0)                            // wait global loads before ds_write
    v_lshlrev_b32 v37, 4, v0                      // tid * 16
    v_add_u32 v37, v37, s48                       // LDS addr = wr_base_a + tid*16
    ds_write_b32 v37, v29, offset:0               // scale A dw0 -> LDS
    ds_write_b32 v37, v30, offset:4               // scale A dw1 -> LDS
    ds_write_b32 v37, v31, offset:8               // scale A dw2 -> LDS
    ds_write_b32 v37, v32, offset:12              // scale A dw3 -> LDS
    s_add_u32 s40, s40, 256                       // s_srd_scale_b += 256
    s_addc_u32 s41, s41, 0                        // carry
    s_add_u32 s49, s49, s64                       // wr_scale_b += db_step
    buffer_load_dword v33, v165, s[40:43], 0, offen offset:0// scale B dword 0
    buffer_load_dword v34, v165, s[40:43], 0, offen offset:4// scale B dword 1
    buffer_load_dword v35, v165, s[40:43], 0, offen offset:8// scale B dword 2
    buffer_load_dword v36, v165, s[40:43], 0, offen offset:12// scale B dword 3
    s_waitcnt vmcnt(0)                            // wait global loads before ds_write
    v_lshlrev_b32 v38, 4, v0                      // tid * 16
    v_add_u32 v38, v38, s49                       // LDS addr = wr_base_b + tid*16
    ds_write_b32 v38, v33, offset:0               // scale B dw0 -> LDS
    ds_write_b32 v38, v34, offset:4               // scale B dw1 -> LDS
    ds_write_b32 v38, v35, offset:8               // scale B dw2 -> LDS
    ds_write_b32 v38, v36, offset:12              // scale B dw3 -> LDS
    s_add_u32 s16, s16, 128                       // s_srd_a += 128
    s_addc_u32 s17, s17, 0                        // carry
    s_add_u32 s24, s24, s64                       // s_lds_wr_a_sg += db
    s_mov_b32 m0, s24                             // m0 = LDS base A
    buffer_load_dwordx4 v5, s[16:19], 0, offen offset:0, lds// DTL A[0]
    s_add_u32 m0, m0, 4096                        // m0 += 4096
    buffer_load_dwordx4 v5, s[16:19], s50, offen offset:0, lds// DTL A[1]
    s_add_u32 m0, m0, 4096                        // m0 += 4096
    buffer_load_dwordx4 v5, s[16:19], s51, offen offset:0, lds// DTL A[2]
    s_add_u32 m0, m0, 4096                        // m0 += 4096
    buffer_load_dwordx4 v5, s[16:19], s52, offen offset:0, lds// DTL A[3]
    s_add_u32 m0, m0, 4096                        // m0 += 4096
    buffer_load_dwordx4 v5, s[16:19], s53, offen offset:0, lds// DTL A[4]
    s_add_u32 m0, m0, 4096                        // m0 += 4096
    buffer_load_dwordx4 v5, s[16:19], s54, offen offset:0, lds// DTL A[5]
    s_add_u32 m0, m0, 4096                        // m0 += 4096
    buffer_load_dwordx4 v5, s[16:19], s55, offen offset:0, lds// DTL A[6]
    s_add_u32 m0, m0, 4096                        // m0 += 4096
    buffer_load_dwordx4 v5, s[16:19], s56, offen offset:0, lds// DTL A[7]
    s_add_u32 s20, s20, 128                       // s_srd_b += 128
    s_addc_u32 s21, s21, 0                        // carry
    s_add_u32 s25, s25, s64                       // s_lds_wr_b_sg += db
    s_mov_b32 m0, s25                             // m0 = LDS base B
    buffer_load_dwordx4 v6, s[20:23], 0, offen offset:0, lds// DTL B[0]
    s_add_u32 m0, m0, 4096                        // m0 += 4096
    buffer_load_dwordx4 v6, s[20:23], s57, offen offset:0, lds// DTL B[1]
    s_add_u32 m0, m0, 4096                        // m0 += 4096
    buffer_load_dwordx4 v6, s[20:23], s58, offen offset:0, lds// DTL B[2]
    s_add_u32 m0, m0, 4096                        // m0 += 4096
    buffer_load_dwordx4 v6, s[20:23], s59, offen offset:0, lds// DTL B[3]
    s_add_u32 m0, m0, 4096                        // m0 += 4096
    buffer_load_dwordx4 v6, s[20:23], s60, offen offset:0, lds// DTL B[4]
    s_add_u32 m0, m0, 4096                        // m0 += 4096
    buffer_load_dwordx4 v6, s[20:23], s61, offen offset:0, lds// DTL B[5]
    s_add_u32 m0, m0, 4096                        // m0 += 4096
    buffer_load_dwordx4 v6, s[20:23], s62, offen offset:0, lds// DTL B[6]
    s_add_u32 m0, m0, 4096                        // m0 += 4096
    buffer_load_dwordx4 v6, s[20:23], s63, offen offset:0, lds// DTL B[7]
pgr_skip_1:

k_loop:

    s_waitcnt vmcnt(0)                            // wait DTL loads from prev iter
    s_waitcnt lgkmcnt(0)                          // wait LDS writes from prev iter
    s_barrier                                     // sync workgroup
    ds_read_b128 v[140:143], v44                  // LR A m0k0 b0
    ds_read_b128 v[76:79], v60                    // LR B n0k0
    ds_read_b32 v156, v166                        // scale A group0 (mi=0,1)
    ds_read_b32 v160, v167                        // scale B group0 (ni=0,1)
    ds_read_b128 v[84:87], v62                    // LR B n1k0
    ds_read_b128 v[92:95], v64                    // LR B n2k0
    ds_read_b32 v161, v167, offset:256            // scale B group1 (ni=2,3)
    ds_read_b128 v[100:103], v66                  // LR B n3k0
    ds_read_b128 v[108:111], v68                  // LR B n4k0
    ds_read_b32 v162, v167, offset:512            // scale B group2 (ni=4,5)
    ds_read_b128 v[116:119], v70                  // LR B n5k0
    ds_read_b128 v[124:127], v72                  // LR B n6k0
    ds_read_b32 v163, v167, offset:768            // scale B group3 (ni=6,7)
    ds_read_b128 v[132:135], v74                  // LR B n7k0
    ds_read_b128 v[144:147], v45                  // LR A m0k1 b0
    ds_read_b128 v[80:83], v61                    // LR B n0k1
    ds_read_b128 v[88:91], v63                    // LR B n1k1
    ds_read_b128 v[96:99], v65                    // LR B n2k1
    ds_read_b128 v[104:107], v67                  // LR B n3k1
    ds_read_b128 v[112:115], v69                  // LR B n4k1
    ds_read_b128 v[120:123], v71                  // LR B n5k1
    ds_read_b128 v[128:131], v73                  // LR B n6k1
    ds_read_b128 v[136:139], v75                  // LR B n7k1
    ds_read_b128 v[148:151], v46                  // LR A m1k0 b1
    ds_read_b128 v[152:155], v47                  // LR A m1k1 b1
    ds_read_b32 v157, v166, offset:256            // scale A group1 (mi=2,3)
    ds_read_b32 v158, v166, offset:512            // scale A group2 (mi=4,5)
    ds_read_b32 v159, v166, offset:768            // scale A group3 (mi=6,7)
    s_waitcnt lgkmcnt(15)                         // auto-wait at pos 29
    v_mfma_scale_f32_16x16x128_f8f6f4 acc[0:3], v[140:143], v[76:79], acc[0:3], v156, v160, cbsz:4 blgp:4// MFMA m0_n0_k0
    v_mfma_scale_f32_16x16x128_f8f6f4 acc[4:7], v[140:143], v[84:87], acc[4:7], v156, v160, cbsz:4 blgp:4// MFMA m0_n1_k0
    v_mfma_scale_f32_16x16x128_f8f6f4 acc[8:11], v[140:143], v[92:95], acc[8:11], v156, v161, cbsz:4 blgp:4// MFMA m0_n2_k0
    v_mfma_scale_f32_16x16x128_f8f6f4 acc[12:15], v[140:143], v[100:103], acc[12:15], v156, v161, cbsz:4 blgp:4// MFMA m0_n3_k0
    v_mfma_scale_f32_16x16x128_f8f6f4 acc[16:19], v[140:143], v[108:111], acc[16:19], v156, v162, cbsz:4 blgp:4// MFMA m0_n4_k0
    v_mfma_scale_f32_16x16x128_f8f6f4 acc[20:23], v[140:143], v[116:119], acc[20:23], v156, v162, cbsz:4 blgp:4// MFMA m0_n5_k0
    v_mfma_scale_f32_16x16x128_f8f6f4 acc[24:27], v[140:143], v[124:127], acc[24:27], v156, v163, cbsz:4 blgp:4// MFMA m0_n6_k0
    s_waitcnt lgkmcnt(14)                         // auto-wait at pos 36
    v_mfma_scale_f32_16x16x128_f8f6f4 acc[28:31], v[140:143], v[132:135], acc[28:31], v156, v163, cbsz:4 blgp:4// MFMA m0_n7_k0
    s_waitcnt lgkmcnt(12)                         // auto-wait at pos 37
    v_mfma_scale_f32_16x16x128_f8f6f4 acc[0:3], v[144:147], v[80:83], acc[0:3], v156, v160, cbsz:4 blgp:4// MFMA m0_n0_k1
    s_waitcnt lgkmcnt(11)                         // auto-wait at pos 38
    v_mfma_scale_f32_16x16x128_f8f6f4 acc[4:7], v[144:147], v[88:91], acc[4:7], v156, v160, cbsz:4 blgp:4// MFMA m0_n1_k1
    s_waitcnt lgkmcnt(10)                         // auto-wait at pos 39
    v_mfma_scale_f32_16x16x128_f8f6f4 acc[8:11], v[144:147], v[96:99], acc[8:11], v156, v161, cbsz:4 blgp:4// MFMA m0_n2_k1
    s_waitcnt lgkmcnt(9)                          // auto-wait at pos 40
    v_mfma_scale_f32_16x16x128_f8f6f4 acc[12:15], v[144:147], v[104:107], acc[12:15], v156, v161, cbsz:4 blgp:4// MFMA m0_n3_k1
    s_waitcnt lgkmcnt(8)                          // auto-wait at pos 41
    v_mfma_scale_f32_16x16x128_f8f6f4 acc[16:19], v[144:147], v[112:115], acc[16:19], v156, v162, cbsz:4 blgp:4// MFMA m0_n4_k1
    s_waitcnt lgkmcnt(7)                          // auto-wait at pos 42
    v_mfma_scale_f32_16x16x128_f8f6f4 acc[20:23], v[144:147], v[120:123], acc[20:23], v156, v162, cbsz:4 blgp:4// MFMA m0_n5_k1
    s_waitcnt lgkmcnt(6)                          // auto-wait at pos 43
    v_mfma_scale_f32_16x16x128_f8f6f4 acc[24:27], v[144:147], v[128:131], acc[24:27], v156, v163, cbsz:4 blgp:4// MFMA m0_n6_k1
    s_waitcnt lgkmcnt(5)                          // auto-wait at pos 44
    v_mfma_scale_f32_16x16x128_f8f6f4 acc[28:31], v[144:147], v[136:139], acc[28:31], v156, v163, cbsz:4 blgp:4// MFMA m0_n7_k1
    ds_read_b128 v[140:143], v48                  // LR A m2k0 b0
    ds_read_b128 v[144:147], v49                  // LR A m2k1 b0
    s_waitcnt lgkmcnt(6)                          // auto-wait at pos 47
    v_mfma_scale_f32_16x16x128_f8f6f4 acc[32:35], v[148:151], v[76:79], acc[32:35], v156, v160, cbsz:4 blgp:4// MFMA m1_n0_k0
    v_mfma_scale_f32_16x16x128_f8f6f4 acc[36:39], v[148:151], v[84:87], acc[36:39], v156, v160, cbsz:4 blgp:4// MFMA m1_n1_k0
    v_mfma_scale_f32_16x16x128_f8f6f4 acc[40:43], v[148:151], v[92:95], acc[40:43], v156, v161, cbsz:4 blgp:4// MFMA m1_n2_k0
    v_mfma_scale_f32_16x16x128_f8f6f4 acc[44:47], v[148:151], v[100:103], acc[44:47], v156, v161, cbsz:4 blgp:4// MFMA m1_n3_k0
    v_mfma_scale_f32_16x16x128_f8f6f4 acc[48:51], v[148:151], v[108:111], acc[48:51], v156, v162, cbsz:4 blgp:4// MFMA m1_n4_k0
    v_mfma_scale_f32_16x16x128_f8f6f4 acc[52:55], v[148:151], v[116:119], acc[52:55], v156, v162, cbsz:4 blgp:4// MFMA m1_n5_k0
    v_mfma_scale_f32_16x16x128_f8f6f4 acc[56:59], v[148:151], v[124:127], acc[56:59], v156, v163, cbsz:4 blgp:4// MFMA m1_n6_k0
    v_mfma_scale_f32_16x16x128_f8f6f4 acc[60:63], v[148:151], v[132:135], acc[60:63], v156, v163, cbsz:4 blgp:4// MFMA m1_n7_k0
    s_waitcnt lgkmcnt(5)                          // auto-wait at pos 55
    v_mfma_scale_f32_16x16x128_f8f6f4 acc[32:35], v[152:155], v[80:83], acc[32:35], v156, v160, cbsz:4 blgp:4// MFMA m1_n0_k1
    v_mfma_scale_f32_16x16x128_f8f6f4 acc[36:39], v[152:155], v[88:91], acc[36:39], v156, v160, cbsz:4 blgp:4// MFMA m1_n1_k1
    v_mfma_scale_f32_16x16x128_f8f6f4 acc[40:43], v[152:155], v[96:99], acc[40:43], v156, v161, cbsz:4 blgp:4// MFMA m1_n2_k1
    v_mfma_scale_f32_16x16x128_f8f6f4 acc[44:47], v[152:155], v[104:107], acc[44:47], v156, v161, cbsz:4 blgp:4// MFMA m1_n3_k1
    v_mfma_scale_f32_16x16x128_f8f6f4 acc[48:51], v[152:155], v[112:115], acc[48:51], v156, v162, cbsz:4 blgp:4// MFMA m1_n4_k1
    v_mfma_scale_f32_16x16x128_f8f6f4 acc[52:55], v[152:155], v[120:123], acc[52:55], v156, v162, cbsz:4 blgp:4// MFMA m1_n5_k1
    v_mfma_scale_f32_16x16x128_f8f6f4 acc[56:59], v[152:155], v[128:131], acc[56:59], v156, v163, cbsz:4 blgp:4// MFMA m1_n6_k1
    v_mfma_scale_f32_16x16x128_f8f6f4 acc[60:63], v[152:155], v[136:139], acc[60:63], v156, v163, cbsz:4 blgp:4// MFMA m1_n7_k1
    ds_read_b128 v[148:151], v50                  // LR A m3k0 b1
    ds_read_b128 v[152:155], v51                  // LR A m3k1 b1
    s_waitcnt lgkmcnt(3)                          // auto-wait at pos 65
    v_mfma_scale_f32_16x16x128_f8f6f4 acc[64:67], v[140:143], v[76:79], acc[64:67], v157, v160, cbsz:4 blgp:4// MFMA m2_n0_k0
    v_mfma_scale_f32_16x16x128_f8f6f4 acc[68:71], v[140:143], v[84:87], acc[68:71], v157, v160, cbsz:4 blgp:4// MFMA m2_n1_k0
    v_mfma_scale_f32_16x16x128_f8f6f4 acc[72:75], v[140:143], v[92:95], acc[72:75], v157, v161, cbsz:4 blgp:4// MFMA m2_n2_k0
    v_mfma_scale_f32_16x16x128_f8f6f4 acc[76:79], v[140:143], v[100:103], acc[76:79], v157, v161, cbsz:4 blgp:4// MFMA m2_n3_k0
    v_mfma_scale_f32_16x16x128_f8f6f4 acc[80:83], v[140:143], v[108:111], acc[80:83], v157, v162, cbsz:4 blgp:4// MFMA m2_n4_k0
    v_mfma_scale_f32_16x16x128_f8f6f4 acc[84:87], v[140:143], v[116:119], acc[84:87], v157, v162, cbsz:4 blgp:4// MFMA m2_n5_k0
    v_mfma_scale_f32_16x16x128_f8f6f4 acc[88:91], v[140:143], v[124:127], acc[88:91], v157, v163, cbsz:4 blgp:4// MFMA m2_n6_k0
    v_mfma_scale_f32_16x16x128_f8f6f4 acc[92:95], v[140:143], v[132:135], acc[92:95], v157, v163, cbsz:4 blgp:4// MFMA m2_n7_k0
    s_waitcnt lgkmcnt(2)                          // auto-wait at pos 73
    v_mfma_scale_f32_16x16x128_f8f6f4 acc[64:67], v[144:147], v[80:83], acc[64:67], v157, v160, cbsz:4 blgp:4// MFMA m2_n0_k1
    v_mfma_scale_f32_16x16x128_f8f6f4 acc[68:71], v[144:147], v[88:91], acc[68:71], v157, v160, cbsz:4 blgp:4// MFMA m2_n1_k1
    v_mfma_scale_f32_16x16x128_f8f6f4 acc[72:75], v[144:147], v[96:99], acc[72:75], v157, v161, cbsz:4 blgp:4// MFMA m2_n2_k1
    v_mfma_scale_f32_16x16x128_f8f6f4 acc[76:79], v[144:147], v[104:107], acc[76:79], v157, v161, cbsz:4 blgp:4// MFMA m2_n3_k1
    v_mfma_scale_f32_16x16x128_f8f6f4 acc[80:83], v[144:147], v[112:115], acc[80:83], v157, v162, cbsz:4 blgp:4// MFMA m2_n4_k1
    v_mfma_scale_f32_16x16x128_f8f6f4 acc[84:87], v[144:147], v[120:123], acc[84:87], v157, v162, cbsz:4 blgp:4// MFMA m2_n5_k1
    v_mfma_scale_f32_16x16x128_f8f6f4 acc[88:91], v[144:147], v[128:131], acc[88:91], v157, v163, cbsz:4 blgp:4// MFMA m2_n6_k1
    v_mfma_scale_f32_16x16x128_f8f6f4 acc[92:95], v[144:147], v[136:139], acc[92:95], v157, v163, cbsz:4 blgp:4// MFMA m2_n7_k1
    ds_read_b128 v[140:143], v52                  // LR A m4k0 b0
    ds_read_b128 v[144:147], v53                  // LR A m4k1 b0
    s_waitcnt lgkmcnt(3)                          // auto-wait at pos 83
    v_mfma_scale_f32_16x16x128_f8f6f4 acc[96:99], v[148:151], v[76:79], acc[96:99], v157, v160, cbsz:4 blgp:4// MFMA m3_n0_k0
    v_mfma_scale_f32_16x16x128_f8f6f4 acc[100:103], v[148:151], v[84:87], acc[100:103], v157, v160, cbsz:4 blgp:4// MFMA m3_n1_k0
    v_mfma_scale_f32_16x16x128_f8f6f4 acc[104:107], v[148:151], v[92:95], acc[104:107], v157, v161, cbsz:4 blgp:4// MFMA m3_n2_k0
    v_mfma_scale_f32_16x16x128_f8f6f4 acc[108:111], v[148:151], v[100:103], acc[108:111], v157, v161, cbsz:4 blgp:4// MFMA m3_n3_k0
    v_mfma_scale_f32_16x16x128_f8f6f4 acc[112:115], v[148:151], v[108:111], acc[112:115], v157, v162, cbsz:4 blgp:4// MFMA m3_n4_k0
    v_mfma_scale_f32_16x16x128_f8f6f4 acc[116:119], v[148:151], v[116:119], acc[116:119], v157, v162, cbsz:4 blgp:4// MFMA m3_n5_k0
    v_mfma_scale_f32_16x16x128_f8f6f4 acc[120:123], v[148:151], v[124:127], acc[120:123], v157, v163, cbsz:4 blgp:4// MFMA m3_n6_k0
    v_mfma_scale_f32_16x16x128_f8f6f4 acc[124:127], v[148:151], v[132:135], acc[124:127], v157, v163, cbsz:4 blgp:4// MFMA m3_n7_k0
    s_waitcnt lgkmcnt(2)                          // auto-wait at pos 91
    v_mfma_scale_f32_16x16x128_f8f6f4 acc[96:99], v[152:155], v[80:83], acc[96:99], v157, v160, cbsz:4 blgp:4// MFMA m3_n0_k1
    v_mfma_scale_f32_16x16x128_f8f6f4 acc[100:103], v[152:155], v[88:91], acc[100:103], v157, v160, cbsz:4 blgp:4// MFMA m3_n1_k1
    v_mfma_scale_f32_16x16x128_f8f6f4 acc[104:107], v[152:155], v[96:99], acc[104:107], v157, v161, cbsz:4 blgp:4// MFMA m3_n2_k1
    v_mfma_scale_f32_16x16x128_f8f6f4 acc[108:111], v[152:155], v[104:107], acc[108:111], v157, v161, cbsz:4 blgp:4// MFMA m3_n3_k1
    v_mfma_scale_f32_16x16x128_f8f6f4 acc[112:115], v[152:155], v[112:115], acc[112:115], v157, v162, cbsz:4 blgp:4// MFMA m3_n4_k1
    v_mfma_scale_f32_16x16x128_f8f6f4 acc[116:119], v[152:155], v[120:123], acc[116:119], v157, v162, cbsz:4 blgp:4// MFMA m3_n5_k1
    v_mfma_scale_f32_16x16x128_f8f6f4 acc[120:123], v[152:155], v[128:131], acc[120:123], v157, v163, cbsz:4 blgp:4// MFMA m3_n6_k1
    v_mfma_scale_f32_16x16x128_f8f6f4 acc[124:127], v[152:155], v[136:139], acc[124:127], v157, v163, cbsz:4 blgp:4// MFMA m3_n7_k1
    ds_read_b128 v[148:151], v54                  // LR A m5k0 b1
    ds_read_b128 v[152:155], v55                  // LR A m5k1 b1
    s_waitcnt lgkmcnt(3)                          // auto-wait at pos 101
    v_mfma_scale_f32_16x16x128_f8f6f4 acc[128:131], v[140:143], v[76:79], acc[128:131], v158, v160, cbsz:4 blgp:4// MFMA m4_n0_k0
    v_mfma_scale_f32_16x16x128_f8f6f4 acc[132:135], v[140:143], v[84:87], acc[132:135], v158, v160, cbsz:4 blgp:4// MFMA m4_n1_k0
    v_mfma_scale_f32_16x16x128_f8f6f4 acc[136:139], v[140:143], v[92:95], acc[136:139], v158, v161, cbsz:4 blgp:4// MFMA m4_n2_k0
    v_mfma_scale_f32_16x16x128_f8f6f4 acc[140:143], v[140:143], v[100:103], acc[140:143], v158, v161, cbsz:4 blgp:4// MFMA m4_n3_k0
    v_mfma_scale_f32_16x16x128_f8f6f4 acc[144:147], v[140:143], v[108:111], acc[144:147], v158, v162, cbsz:4 blgp:4// MFMA m4_n4_k0
    v_mfma_scale_f32_16x16x128_f8f6f4 acc[148:151], v[140:143], v[116:119], acc[148:151], v158, v162, cbsz:4 blgp:4// MFMA m4_n5_k0
    v_mfma_scale_f32_16x16x128_f8f6f4 acc[152:155], v[140:143], v[124:127], acc[152:155], v158, v163, cbsz:4 blgp:4// MFMA m4_n6_k0
    v_mfma_scale_f32_16x16x128_f8f6f4 acc[156:159], v[140:143], v[132:135], acc[156:159], v158, v163, cbsz:4 blgp:4// MFMA m4_n7_k0
    s_waitcnt lgkmcnt(2)                          // auto-wait at pos 109
    v_mfma_scale_f32_16x16x128_f8f6f4 acc[128:131], v[144:147], v[80:83], acc[128:131], v158, v160, cbsz:4 blgp:4// MFMA m4_n0_k1
    v_mfma_scale_f32_16x16x128_f8f6f4 acc[132:135], v[144:147], v[88:91], acc[132:135], v158, v160, cbsz:4 blgp:4// MFMA m4_n1_k1
    v_mfma_scale_f32_16x16x128_f8f6f4 acc[136:139], v[144:147], v[96:99], acc[136:139], v158, v161, cbsz:4 blgp:4// MFMA m4_n2_k1
    v_mfma_scale_f32_16x16x128_f8f6f4 acc[140:143], v[144:147], v[104:107], acc[140:143], v158, v161, cbsz:4 blgp:4// MFMA m4_n3_k1
    v_mfma_scale_f32_16x16x128_f8f6f4 acc[144:147], v[144:147], v[112:115], acc[144:147], v158, v162, cbsz:4 blgp:4// MFMA m4_n4_k1
    v_mfma_scale_f32_16x16x128_f8f6f4 acc[148:151], v[144:147], v[120:123], acc[148:151], v158, v162, cbsz:4 blgp:4// MFMA m4_n5_k1
    v_mfma_scale_f32_16x16x128_f8f6f4 acc[152:155], v[144:147], v[128:131], acc[152:155], v158, v163, cbsz:4 blgp:4// MFMA m4_n6_k1
    v_mfma_scale_f32_16x16x128_f8f6f4 acc[156:159], v[144:147], v[136:139], acc[156:159], v158, v163, cbsz:4 blgp:4// MFMA m4_n7_k1
    ds_read_b128 v[140:143], v56                  // LR A m6k0 b0
    ds_read_b128 v[144:147], v57                  // LR A m6k1 b0
    s_waitcnt lgkmcnt(3)                          // auto-wait at pos 119
    v_mfma_scale_f32_16x16x128_f8f6f4 acc[160:163], v[148:151], v[76:79], acc[160:163], v158, v160, cbsz:4 blgp:4// MFMA m5_n0_k0
    v_mfma_scale_f32_16x16x128_f8f6f4 acc[164:167], v[148:151], v[84:87], acc[164:167], v158, v160, cbsz:4 blgp:4// MFMA m5_n1_k0
    v_mfma_scale_f32_16x16x128_f8f6f4 acc[168:171], v[148:151], v[92:95], acc[168:171], v158, v161, cbsz:4 blgp:4// MFMA m5_n2_k0
    v_mfma_scale_f32_16x16x128_f8f6f4 acc[172:175], v[148:151], v[100:103], acc[172:175], v158, v161, cbsz:4 blgp:4// MFMA m5_n3_k0
    v_mfma_scale_f32_16x16x128_f8f6f4 acc[176:179], v[148:151], v[108:111], acc[176:179], v158, v162, cbsz:4 blgp:4// MFMA m5_n4_k0
    v_mfma_scale_f32_16x16x128_f8f6f4 acc[180:183], v[148:151], v[116:119], acc[180:183], v158, v162, cbsz:4 blgp:4// MFMA m5_n5_k0
    v_mfma_scale_f32_16x16x128_f8f6f4 acc[184:187], v[148:151], v[124:127], acc[184:187], v158, v163, cbsz:4 blgp:4// MFMA m5_n6_k0
    v_mfma_scale_f32_16x16x128_f8f6f4 acc[188:191], v[148:151], v[132:135], acc[188:191], v158, v163, cbsz:4 blgp:4// MFMA m5_n7_k0
    s_waitcnt lgkmcnt(2)                          // auto-wait at pos 127
    v_mfma_scale_f32_16x16x128_f8f6f4 acc[160:163], v[152:155], v[80:83], acc[160:163], v158, v160, cbsz:4 blgp:4// MFMA m5_n0_k1
    v_mfma_scale_f32_16x16x128_f8f6f4 acc[164:167], v[152:155], v[88:91], acc[164:167], v158, v160, cbsz:4 blgp:4// MFMA m5_n1_k1
    v_mfma_scale_f32_16x16x128_f8f6f4 acc[168:171], v[152:155], v[96:99], acc[168:171], v158, v161, cbsz:4 blgp:4// MFMA m5_n2_k1
    v_mfma_scale_f32_16x16x128_f8f6f4 acc[172:175], v[152:155], v[104:107], acc[172:175], v158, v161, cbsz:4 blgp:4// MFMA m5_n3_k1
    v_mfma_scale_f32_16x16x128_f8f6f4 acc[176:179], v[152:155], v[112:115], acc[176:179], v158, v162, cbsz:4 blgp:4// MFMA m5_n4_k1
    v_mfma_scale_f32_16x16x128_f8f6f4 acc[180:183], v[152:155], v[120:123], acc[180:183], v158, v162, cbsz:4 blgp:4// MFMA m5_n5_k1
    v_mfma_scale_f32_16x16x128_f8f6f4 acc[184:187], v[152:155], v[128:131], acc[184:187], v158, v163, cbsz:4 blgp:4// MFMA m5_n6_k1
    v_mfma_scale_f32_16x16x128_f8f6f4 acc[188:191], v[152:155], v[136:139], acc[188:191], v158, v163, cbsz:4 blgp:4// MFMA m5_n7_k1
    ds_read_b128 v[148:151], v58                  // LR A m7k0 b1
    ds_read_b128 v[152:155], v59                  // LR A m7k1 b1
    s_waitcnt lgkmcnt(3)                          // auto-wait at pos 137
    v_mfma_scale_f32_16x16x128_f8f6f4 acc[192:195], v[140:143], v[76:79], acc[192:195], v159, v160, cbsz:4 blgp:4// MFMA m6_n0_k0
    v_mfma_scale_f32_16x16x128_f8f6f4 acc[196:199], v[140:143], v[84:87], acc[196:199], v159, v160, cbsz:4 blgp:4// MFMA m6_n1_k0
    v_mfma_scale_f32_16x16x128_f8f6f4 acc[200:203], v[140:143], v[92:95], acc[200:203], v159, v161, cbsz:4 blgp:4// MFMA m6_n2_k0
    v_mfma_scale_f32_16x16x128_f8f6f4 acc[204:207], v[140:143], v[100:103], acc[204:207], v159, v161, cbsz:4 blgp:4// MFMA m6_n3_k0
    v_mfma_scale_f32_16x16x128_f8f6f4 acc[208:211], v[140:143], v[108:111], acc[208:211], v159, v162, cbsz:4 blgp:4// MFMA m6_n4_k0
    v_mfma_scale_f32_16x16x128_f8f6f4 acc[212:215], v[140:143], v[116:119], acc[212:215], v159, v162, cbsz:4 blgp:4// MFMA m6_n5_k0
    v_mfma_scale_f32_16x16x128_f8f6f4 acc[216:219], v[140:143], v[124:127], acc[216:219], v159, v163, cbsz:4 blgp:4// MFMA m6_n6_k0
    v_mfma_scale_f32_16x16x128_f8f6f4 acc[220:223], v[140:143], v[132:135], acc[220:223], v159, v163, cbsz:4 blgp:4// MFMA m6_n7_k0
    s_waitcnt lgkmcnt(2)                          // auto-wait at pos 145
    v_mfma_scale_f32_16x16x128_f8f6f4 acc[192:195], v[144:147], v[80:83], acc[192:195], v159, v160, cbsz:4 blgp:4// MFMA m6_n0_k1
    v_mfma_scale_f32_16x16x128_f8f6f4 acc[196:199], v[144:147], v[88:91], acc[196:199], v159, v160, cbsz:4 blgp:4// MFMA m6_n1_k1
    v_mfma_scale_f32_16x16x128_f8f6f4 acc[200:203], v[144:147], v[96:99], acc[200:203], v159, v161, cbsz:4 blgp:4// MFMA m6_n2_k1
    v_mfma_scale_f32_16x16x128_f8f6f4 acc[204:207], v[144:147], v[104:107], acc[204:207], v159, v161, cbsz:4 blgp:4// MFMA m6_n3_k1
    v_mfma_scale_f32_16x16x128_f8f6f4 acc[208:211], v[144:147], v[112:115], acc[208:211], v159, v162, cbsz:4 blgp:4// MFMA m6_n4_k1
    v_mfma_scale_f32_16x16x128_f8f6f4 acc[212:215], v[144:147], v[120:123], acc[212:215], v159, v162, cbsz:4 blgp:4// MFMA m6_n5_k1
    v_mfma_scale_f32_16x16x128_f8f6f4 acc[216:219], v[144:147], v[128:131], acc[216:219], v159, v163, cbsz:4 blgp:4// MFMA m6_n6_k1
    v_mfma_scale_f32_16x16x128_f8f6f4 acc[220:223], v[144:147], v[136:139], acc[220:223], v159, v163, cbsz:4 blgp:4// MFMA m6_n7_k1
    s_waitcnt lgkmcnt(1)                          // auto-wait at pos 153
    v_mfma_scale_f32_16x16x128_f8f6f4 acc[224:227], v[148:151], v[76:79], acc[224:227], v159, v160, cbsz:4 blgp:4// MFMA m7_n0_k0
    v_mfma_scale_f32_16x16x128_f8f6f4 acc[228:231], v[148:151], v[84:87], acc[228:231], v159, v160, cbsz:4 blgp:4// MFMA m7_n1_k0
    v_mfma_scale_f32_16x16x128_f8f6f4 acc[232:235], v[148:151], v[92:95], acc[232:235], v159, v161, cbsz:4 blgp:4// MFMA m7_n2_k0
    v_mfma_scale_f32_16x16x128_f8f6f4 acc[236:239], v[148:151], v[100:103], acc[236:239], v159, v161, cbsz:4 blgp:4// MFMA m7_n3_k0
    v_mfma_scale_f32_16x16x128_f8f6f4 acc[240:243], v[148:151], v[108:111], acc[240:243], v159, v162, cbsz:4 blgp:4// MFMA m7_n4_k0
    v_mfma_scale_f32_16x16x128_f8f6f4 acc[244:247], v[148:151], v[116:119], acc[244:247], v159, v162, cbsz:4 blgp:4// MFMA m7_n5_k0
    v_mfma_scale_f32_16x16x128_f8f6f4 acc[248:251], v[148:151], v[124:127], acc[248:251], v159, v163, cbsz:4 blgp:4// MFMA m7_n6_k0
    v_mfma_scale_f32_16x16x128_f8f6f4 acc[252:255], v[148:151], v[132:135], acc[252:255], v159, v163, cbsz:4 blgp:4// MFMA m7_n7_k0
    s_waitcnt lgkmcnt(0)                          // auto-wait at pos 161
    v_mfma_scale_f32_16x16x128_f8f6f4 acc[224:227], v[152:155], v[80:83], acc[224:227], v159, v160, cbsz:4 blgp:4// MFMA m7_n0_k1
    v_mfma_scale_f32_16x16x128_f8f6f4 acc[228:231], v[152:155], v[88:91], acc[228:231], v159, v160, cbsz:4 blgp:4// MFMA m7_n1_k1
    v_mfma_scale_f32_16x16x128_f8f6f4 acc[232:235], v[152:155], v[96:99], acc[232:235], v159, v161, cbsz:4 blgp:4// MFMA m7_n2_k1
    v_mfma_scale_f32_16x16x128_f8f6f4 acc[236:239], v[152:155], v[104:107], acc[236:239], v159, v161, cbsz:4 blgp:4// MFMA m7_n3_k1
    v_mfma_scale_f32_16x16x128_f8f6f4 acc[240:243], v[152:155], v[112:115], acc[240:243], v159, v162, cbsz:4 blgp:4// MFMA m7_n4_k1
    v_mfma_scale_f32_16x16x128_f8f6f4 acc[244:247], v[152:155], v[120:123], acc[244:247], v159, v162, cbsz:4 blgp:4// MFMA m7_n5_k1
    v_mfma_scale_f32_16x16x128_f8f6f4 acc[248:251], v[152:155], v[128:131], acc[248:251], v159, v163, cbsz:4 blgp:4// MFMA m7_n6_k1
    v_mfma_scale_f32_16x16x128_f8f6f4 acc[252:255], v[152:155], v[136:139], acc[252:255], v159, v163, cbsz:4 blgp:4// MFMA m7_n7_k1
    v_add_u32 v166, v166, s64                     // scale_rd_a += db_step
    v_add_u32 v167, v167, s64                     // scale_rd_b += db_step
    // Toggle all precomputed read addresses (ADD)
    v_add_u32 v44, v44, s64                       // rd_a_m0_k0 += db
    v_add_u32 v45, v45, s64                       // rd_a_m0_k1 += db
    v_add_u32 v46, v46, s64                       // rd_a_m1_k0 += db
    v_add_u32 v47, v47, s64                       // rd_a_m1_k1 += db
    v_add_u32 v48, v48, s64                       // rd_a_m2_k0 += db
    v_add_u32 v49, v49, s64                       // rd_a_m2_k1 += db
    v_add_u32 v50, v50, s64                       // rd_a_m3_k0 += db
    v_add_u32 v51, v51, s64                       // rd_a_m3_k1 += db
    v_add_u32 v52, v52, s64                       // rd_a_m4_k0 += db
    v_add_u32 v53, v53, s64                       // rd_a_m4_k1 += db
    v_add_u32 v54, v54, s64                       // rd_a_m5_k0 += db
    v_add_u32 v55, v55, s64                       // rd_a_m5_k1 += db
    v_add_u32 v56, v56, s64                       // rd_a_m6_k0 += db
    v_add_u32 v57, v57, s64                       // rd_a_m6_k1 += db
    v_add_u32 v58, v58, s64                       // rd_a_m7_k0 += db
    v_add_u32 v59, v59, s64                       // rd_a_m7_k1 += db
    v_add_u32 v60, v60, s64                       // rd_b_n0_k0 += db
    v_add_u32 v61, v61, s64                       // rd_b_n0_k1 += db
    v_add_u32 v62, v62, s64                       // rd_b_n1_k0 += db
    v_add_u32 v63, v63, s64                       // rd_b_n1_k1 += db
    v_add_u32 v64, v64, s64                       // rd_b_n2_k0 += db
    v_add_u32 v65, v65, s64                       // rd_b_n2_k1 += db
    v_add_u32 v66, v66, s64                       // rd_b_n3_k0 += db
    v_add_u32 v67, v67, s64                       // rd_b_n3_k1 += db
    v_add_u32 v68, v68, s64                       // rd_b_n4_k0 += db
    v_add_u32 v69, v69, s64                       // rd_b_n4_k1 += db
    v_add_u32 v70, v70, s64                       // rd_b_n5_k0 += db
    v_add_u32 v71, v71, s64                       // rd_b_n5_k1 += db
    v_add_u32 v72, v72, s64                       // rd_b_n6_k0 += db
    v_add_u32 v73, v73, s64                       // rd_b_n6_k1 += db
    v_add_u32 v74, v74, s64                       // rd_b_n7_k0 += db
    v_add_u32 v75, v75, s64                       // rd_b_n7_k1 += db
    // [TODO] toggle_rd_data_b (scalar)
    s_sub_u32 s13, s13, 1                         // k_tiles--
    s_cmp_gt_u32 s13, 1                           // k_tiles > 1?
    s_cbranch_scc0 load_skip_all                  // skip producers (drain)
    s_sub_u32 s64, 0, s64                         // negate db_step for next toggle
    s_add_u32 s36, s36, 256                       // s_srd_scale_a += 256
    s_addc_u32 s37, s37, 0                        // carry
    s_add_u32 s48, s48, s64                       // wr_scale_a += db_step
    buffer_load_dword v29, v164, s[36:39], 0, offen offset:0// scale A dword 0
    buffer_load_dword v30, v164, s[36:39], 0, offen offset:4// scale A dword 1
    buffer_load_dword v31, v164, s[36:39], 0, offen offset:8// scale A dword 2
    buffer_load_dword v32, v164, s[36:39], 0, offen offset:12// scale A dword 3
    s_waitcnt vmcnt(0)                            // auto-wait at pos 176
    v_lshlrev_b32 v37, 4, v0                      // tid * 16
    v_add_u32 v37, v37, s48                       // LDS addr = wr_base_a + tid*16
    ds_write_b32 v37, v29, offset:0               // scale A dw0 -> LDS
    ds_write_b32 v37, v30, offset:4               // scale A dw1 -> LDS
    ds_write_b32 v37, v31, offset:8               // scale A dw2 -> LDS
    ds_write_b32 v37, v32, offset:12              // scale A dw3 -> LDS
    s_add_u32 s40, s40, 256                       // s_srd_scale_b += 256
    s_addc_u32 s41, s41, 0                        // carry
    s_add_u32 s49, s49, s64                       // wr_scale_b += db_step
    buffer_load_dword v33, v165, s[40:43], 0, offen offset:0// scale B dword 0
    buffer_load_dword v34, v165, s[40:43], 0, offen offset:4// scale B dword 1
    buffer_load_dword v35, v165, s[40:43], 0, offen offset:8// scale B dword 2
    buffer_load_dword v36, v165, s[40:43], 0, offen offset:12// scale B dword 3
    s_waitcnt vmcnt(0)                            // auto-wait at pos 180
    v_lshlrev_b32 v38, 4, v0                      // tid * 16
    v_add_u32 v38, v38, s49                       // LDS addr = wr_base_b + tid*16
    ds_write_b32 v38, v33, offset:0               // scale B dw0 -> LDS
    ds_write_b32 v38, v34, offset:4               // scale B dw1 -> LDS
    ds_write_b32 v38, v35, offset:8               // scale B dw2 -> LDS
    ds_write_b32 v38, v36, offset:12              // scale B dw3 -> LDS
    s_add_u32 s16, s16, 128                       // s_srd_a += 128
    s_addc_u32 s17, s17, 0                        // carry
    s_add_u32 s24, s24, s64                       // s_lds_wr_a_sg += db
    s_mov_b32 m0, s24                             // m0 = LDS base A
    buffer_load_dwordx4 v5, s[16:19], 0, offen offset:0, lds// DTL A[0]
    s_add_u32 m0, m0, 4096                        // m0 += 4096
    buffer_load_dwordx4 v5, s[16:19], s50, offen offset:0, lds// DTL A[1]
    s_add_u32 m0, m0, 4096                        // m0 += 4096
    buffer_load_dwordx4 v5, s[16:19], s51, offen offset:0, lds// DTL A[2]
    s_add_u32 m0, m0, 4096                        // m0 += 4096
    buffer_load_dwordx4 v5, s[16:19], s52, offen offset:0, lds// DTL A[3]
    s_add_u32 m0, m0, 4096                        // m0 += 4096
    buffer_load_dwordx4 v5, s[16:19], s53, offen offset:0, lds// DTL A[4]
    s_add_u32 m0, m0, 4096                        // m0 += 4096
    buffer_load_dwordx4 v5, s[16:19], s54, offen offset:0, lds// DTL A[5]
    s_add_u32 m0, m0, 4096                        // m0 += 4096
    buffer_load_dwordx4 v5, s[16:19], s55, offen offset:0, lds// DTL A[6]
    s_add_u32 m0, m0, 4096                        // m0 += 4096
    buffer_load_dwordx4 v5, s[16:19], s56, offen offset:0, lds// DTL A[7]
    s_add_u32 s20, s20, 128                       // s_srd_b += 128
    s_addc_u32 s21, s21, 0                        // carry
    s_add_u32 s25, s25, s64                       // s_lds_wr_b_sg += db
    s_mov_b32 m0, s25                             // m0 = LDS base B
    buffer_load_dwordx4 v6, s[20:23], 0, offen offset:0, lds// DTL B[0]
    s_add_u32 m0, m0, 4096                        // m0 += 4096
    buffer_load_dwordx4 v6, s[20:23], s57, offen offset:0, lds// DTL B[1]
    s_add_u32 m0, m0, 4096                        // m0 += 4096
    buffer_load_dwordx4 v6, s[20:23], s58, offen offset:0, lds// DTL B[2]
    s_add_u32 m0, m0, 4096                        // m0 += 4096
    buffer_load_dwordx4 v6, s[20:23], s59, offen offset:0, lds// DTL B[3]
    s_add_u32 m0, m0, 4096                        // m0 += 4096
    buffer_load_dwordx4 v6, s[20:23], s60, offen offset:0, lds// DTL B[4]
    s_add_u32 m0, m0, 4096                        // m0 += 4096
    buffer_load_dwordx4 v6, s[20:23], s61, offen offset:0, lds// DTL B[5]
    s_add_u32 m0, m0, 4096                        // m0 += 4096
    buffer_load_dwordx4 v6, s[20:23], s62, offen offset:0, lds// DTL B[6]
    s_add_u32 m0, m0, 4096                        // m0 += 4096
    buffer_load_dwordx4 v6, s[20:23], s63, offen offset:0, lds// DTL B[7]

load_skip_all:
    s_cmp_gt_u32 s13, 1                           // k_tiles > 1? (exit to drain)
    s_cbranch_scc1 k_loop                         // loop

    // Drain stage 0
    s_waitcnt vmcnt(0)                            // wait DTL loads
    s_waitcnt lgkmcnt(0)                          // wait LDS writes
    s_barrier                                     // sync workgroup
    ds_read_b128 v[140:143], v44                  // LR A m0k0 b0
    ds_read_b128 v[76:79], v60                    // LR B n0k0
    ds_read_b32 v156, v166                        // scale A group0 (mi=0,1)
    ds_read_b32 v160, v167                        // scale B group0 (ni=0,1)
    ds_read_b128 v[84:87], v62                    // LR B n1k0
    ds_read_b128 v[92:95], v64                    // LR B n2k0
    ds_read_b32 v161, v167, offset:256            // scale B group1 (ni=2,3)
    ds_read_b128 v[100:103], v66                  // LR B n3k0
    ds_read_b128 v[108:111], v68                  // LR B n4k0
    ds_read_b32 v162, v167, offset:512            // scale B group2 (ni=4,5)
    ds_read_b128 v[116:119], v70                  // LR B n5k0
    ds_read_b128 v[124:127], v72                  // LR B n6k0
    ds_read_b32 v163, v167, offset:768            // scale B group3 (ni=6,7)
    ds_read_b128 v[132:135], v74                  // LR B n7k0
    ds_read_b128 v[144:147], v45                  // LR A m0k1 b0
    ds_read_b128 v[80:83], v61                    // LR B n0k1
    ds_read_b128 v[88:91], v63                    // LR B n1k1
    ds_read_b128 v[96:99], v65                    // LR B n2k1
    ds_read_b128 v[104:107], v67                  // LR B n3k1
    ds_read_b128 v[112:115], v69                  // LR B n4k1
    ds_read_b128 v[120:123], v71                  // LR B n5k1
    ds_read_b128 v[128:131], v73                  // LR B n6k1
    ds_read_b128 v[136:139], v75                  // LR B n7k1
    ds_read_b128 v[148:151], v46                  // LR A m1k0 b1
    ds_read_b128 v[152:155], v47                  // LR A m1k1 b1
    ds_read_b32 v157, v166, offset:256            // scale A group1 (mi=2,3)
    ds_read_b32 v158, v166, offset:512            // scale A group2 (mi=4,5)
    ds_read_b32 v159, v166, offset:768            // scale A group3 (mi=6,7)
    v_mfma_scale_f32_16x16x128_f8f6f4 acc[0:3], v[140:143], v[76:79], acc[0:3], v156, v160, cbsz:4 blgp:4// MFMA m0_n0_k0
    v_mfma_scale_f32_16x16x128_f8f6f4 acc[4:7], v[140:143], v[84:87], acc[4:7], v156, v160, cbsz:4 blgp:4// MFMA m0_n1_k0
    v_mfma_scale_f32_16x16x128_f8f6f4 acc[8:11], v[140:143], v[92:95], acc[8:11], v156, v161, cbsz:4 blgp:4// MFMA m0_n2_k0
    v_mfma_scale_f32_16x16x128_f8f6f4 acc[12:15], v[140:143], v[100:103], acc[12:15], v156, v161, cbsz:4 blgp:4// MFMA m0_n3_k0
    v_mfma_scale_f32_16x16x128_f8f6f4 acc[16:19], v[140:143], v[108:111], acc[16:19], v156, v162, cbsz:4 blgp:4// MFMA m0_n4_k0
    v_mfma_scale_f32_16x16x128_f8f6f4 acc[20:23], v[140:143], v[116:119], acc[20:23], v156, v162, cbsz:4 blgp:4// MFMA m0_n5_k0
    v_mfma_scale_f32_16x16x128_f8f6f4 acc[24:27], v[140:143], v[124:127], acc[24:27], v156, v163, cbsz:4 blgp:4// MFMA m0_n6_k0
    v_mfma_scale_f32_16x16x128_f8f6f4 acc[28:31], v[140:143], v[132:135], acc[28:31], v156, v163, cbsz:4 blgp:4// MFMA m0_n7_k0
    v_mfma_scale_f32_16x16x128_f8f6f4 acc[0:3], v[144:147], v[80:83], acc[0:3], v156, v160, cbsz:4 blgp:4// MFMA m0_n0_k1
    v_mfma_scale_f32_16x16x128_f8f6f4 acc[4:7], v[144:147], v[88:91], acc[4:7], v156, v160, cbsz:4 blgp:4// MFMA m0_n1_k1
    v_mfma_scale_f32_16x16x128_f8f6f4 acc[8:11], v[144:147], v[96:99], acc[8:11], v156, v161, cbsz:4 blgp:4// MFMA m0_n2_k1
    v_mfma_scale_f32_16x16x128_f8f6f4 acc[12:15], v[144:147], v[104:107], acc[12:15], v156, v161, cbsz:4 blgp:4// MFMA m0_n3_k1
    v_mfma_scale_f32_16x16x128_f8f6f4 acc[16:19], v[144:147], v[112:115], acc[16:19], v156, v162, cbsz:4 blgp:4// MFMA m0_n4_k1
    v_mfma_scale_f32_16x16x128_f8f6f4 acc[20:23], v[144:147], v[120:123], acc[20:23], v156, v162, cbsz:4 blgp:4// MFMA m0_n5_k1
    v_mfma_scale_f32_16x16x128_f8f6f4 acc[24:27], v[144:147], v[128:131], acc[24:27], v156, v163, cbsz:4 blgp:4// MFMA m0_n6_k1
    v_mfma_scale_f32_16x16x128_f8f6f4 acc[28:31], v[144:147], v[136:139], acc[28:31], v156, v163, cbsz:4 blgp:4// MFMA m0_n7_k1
    ds_read_b128 v[140:143], v48                  // LR A m2k0 b0
    ds_read_b128 v[144:147], v49                  // LR A m2k1 b0
    v_mfma_scale_f32_16x16x128_f8f6f4 acc[32:35], v[148:151], v[76:79], acc[32:35], v156, v160, cbsz:4 blgp:4// MFMA m1_n0_k0
    v_mfma_scale_f32_16x16x128_f8f6f4 acc[36:39], v[148:151], v[84:87], acc[36:39], v156, v160, cbsz:4 blgp:4// MFMA m1_n1_k0
    v_mfma_scale_f32_16x16x128_f8f6f4 acc[40:43], v[148:151], v[92:95], acc[40:43], v156, v161, cbsz:4 blgp:4// MFMA m1_n2_k0
    v_mfma_scale_f32_16x16x128_f8f6f4 acc[44:47], v[148:151], v[100:103], acc[44:47], v156, v161, cbsz:4 blgp:4// MFMA m1_n3_k0
    v_mfma_scale_f32_16x16x128_f8f6f4 acc[48:51], v[148:151], v[108:111], acc[48:51], v156, v162, cbsz:4 blgp:4// MFMA m1_n4_k0
    v_mfma_scale_f32_16x16x128_f8f6f4 acc[52:55], v[148:151], v[116:119], acc[52:55], v156, v162, cbsz:4 blgp:4// MFMA m1_n5_k0
    v_mfma_scale_f32_16x16x128_f8f6f4 acc[56:59], v[148:151], v[124:127], acc[56:59], v156, v163, cbsz:4 blgp:4// MFMA m1_n6_k0
    v_mfma_scale_f32_16x16x128_f8f6f4 acc[60:63], v[148:151], v[132:135], acc[60:63], v156, v163, cbsz:4 blgp:4// MFMA m1_n7_k0
    v_mfma_scale_f32_16x16x128_f8f6f4 acc[32:35], v[152:155], v[80:83], acc[32:35], v156, v160, cbsz:4 blgp:4// MFMA m1_n0_k1
    v_mfma_scale_f32_16x16x128_f8f6f4 acc[36:39], v[152:155], v[88:91], acc[36:39], v156, v160, cbsz:4 blgp:4// MFMA m1_n1_k1
    v_mfma_scale_f32_16x16x128_f8f6f4 acc[40:43], v[152:155], v[96:99], acc[40:43], v156, v161, cbsz:4 blgp:4// MFMA m1_n2_k1
    v_mfma_scale_f32_16x16x128_f8f6f4 acc[44:47], v[152:155], v[104:107], acc[44:47], v156, v161, cbsz:4 blgp:4// MFMA m1_n3_k1
    v_mfma_scale_f32_16x16x128_f8f6f4 acc[48:51], v[152:155], v[112:115], acc[48:51], v156, v162, cbsz:4 blgp:4// MFMA m1_n4_k1
    v_mfma_scale_f32_16x16x128_f8f6f4 acc[52:55], v[152:155], v[120:123], acc[52:55], v156, v162, cbsz:4 blgp:4// MFMA m1_n5_k1
    v_mfma_scale_f32_16x16x128_f8f6f4 acc[56:59], v[152:155], v[128:131], acc[56:59], v156, v163, cbsz:4 blgp:4// MFMA m1_n6_k1
    v_mfma_scale_f32_16x16x128_f8f6f4 acc[60:63], v[152:155], v[136:139], acc[60:63], v156, v163, cbsz:4 blgp:4// MFMA m1_n7_k1
    ds_read_b128 v[148:151], v50                  // LR A m3k0 b1
    ds_read_b128 v[152:155], v51                  // LR A m3k1 b1
    v_mfma_scale_f32_16x16x128_f8f6f4 acc[64:67], v[140:143], v[76:79], acc[64:67], v157, v160, cbsz:4 blgp:4// MFMA m2_n0_k0
    v_mfma_scale_f32_16x16x128_f8f6f4 acc[68:71], v[140:143], v[84:87], acc[68:71], v157, v160, cbsz:4 blgp:4// MFMA m2_n1_k0
    v_mfma_scale_f32_16x16x128_f8f6f4 acc[72:75], v[140:143], v[92:95], acc[72:75], v157, v161, cbsz:4 blgp:4// MFMA m2_n2_k0
    v_mfma_scale_f32_16x16x128_f8f6f4 acc[76:79], v[140:143], v[100:103], acc[76:79], v157, v161, cbsz:4 blgp:4// MFMA m2_n3_k0
    v_mfma_scale_f32_16x16x128_f8f6f4 acc[80:83], v[140:143], v[108:111], acc[80:83], v157, v162, cbsz:4 blgp:4// MFMA m2_n4_k0
    v_mfma_scale_f32_16x16x128_f8f6f4 acc[84:87], v[140:143], v[116:119], acc[84:87], v157, v162, cbsz:4 blgp:4// MFMA m2_n5_k0
    v_mfma_scale_f32_16x16x128_f8f6f4 acc[88:91], v[140:143], v[124:127], acc[88:91], v157, v163, cbsz:4 blgp:4// MFMA m2_n6_k0
    v_mfma_scale_f32_16x16x128_f8f6f4 acc[92:95], v[140:143], v[132:135], acc[92:95], v157, v163, cbsz:4 blgp:4// MFMA m2_n7_k0
    v_mfma_scale_f32_16x16x128_f8f6f4 acc[64:67], v[144:147], v[80:83], acc[64:67], v157, v160, cbsz:4 blgp:4// MFMA m2_n0_k1
    v_mfma_scale_f32_16x16x128_f8f6f4 acc[68:71], v[144:147], v[88:91], acc[68:71], v157, v160, cbsz:4 blgp:4// MFMA m2_n1_k1
    v_mfma_scale_f32_16x16x128_f8f6f4 acc[72:75], v[144:147], v[96:99], acc[72:75], v157, v161, cbsz:4 blgp:4// MFMA m2_n2_k1
    v_mfma_scale_f32_16x16x128_f8f6f4 acc[76:79], v[144:147], v[104:107], acc[76:79], v157, v161, cbsz:4 blgp:4// MFMA m2_n3_k1
    v_mfma_scale_f32_16x16x128_f8f6f4 acc[80:83], v[144:147], v[112:115], acc[80:83], v157, v162, cbsz:4 blgp:4// MFMA m2_n4_k1
    v_mfma_scale_f32_16x16x128_f8f6f4 acc[84:87], v[144:147], v[120:123], acc[84:87], v157, v162, cbsz:4 blgp:4// MFMA m2_n5_k1
    v_mfma_scale_f32_16x16x128_f8f6f4 acc[88:91], v[144:147], v[128:131], acc[88:91], v157, v163, cbsz:4 blgp:4// MFMA m2_n6_k1
    v_mfma_scale_f32_16x16x128_f8f6f4 acc[92:95], v[144:147], v[136:139], acc[92:95], v157, v163, cbsz:4 blgp:4// MFMA m2_n7_k1
    ds_read_b128 v[140:143], v52                  // LR A m4k0 b0
    ds_read_b128 v[144:147], v53                  // LR A m4k1 b0
    v_mfma_scale_f32_16x16x128_f8f6f4 acc[96:99], v[148:151], v[76:79], acc[96:99], v157, v160, cbsz:4 blgp:4// MFMA m3_n0_k0
    v_mfma_scale_f32_16x16x128_f8f6f4 acc[100:103], v[148:151], v[84:87], acc[100:103], v157, v160, cbsz:4 blgp:4// MFMA m3_n1_k0
    v_mfma_scale_f32_16x16x128_f8f6f4 acc[104:107], v[148:151], v[92:95], acc[104:107], v157, v161, cbsz:4 blgp:4// MFMA m3_n2_k0
    v_mfma_scale_f32_16x16x128_f8f6f4 acc[108:111], v[148:151], v[100:103], acc[108:111], v157, v161, cbsz:4 blgp:4// MFMA m3_n3_k0
    v_mfma_scale_f32_16x16x128_f8f6f4 acc[112:115], v[148:151], v[108:111], acc[112:115], v157, v162, cbsz:4 blgp:4// MFMA m3_n4_k0
    v_mfma_scale_f32_16x16x128_f8f6f4 acc[116:119], v[148:151], v[116:119], acc[116:119], v157, v162, cbsz:4 blgp:4// MFMA m3_n5_k0
    v_mfma_scale_f32_16x16x128_f8f6f4 acc[120:123], v[148:151], v[124:127], acc[120:123], v157, v163, cbsz:4 blgp:4// MFMA m3_n6_k0
    v_mfma_scale_f32_16x16x128_f8f6f4 acc[124:127], v[148:151], v[132:135], acc[124:127], v157, v163, cbsz:4 blgp:4// MFMA m3_n7_k0
    v_mfma_scale_f32_16x16x128_f8f6f4 acc[96:99], v[152:155], v[80:83], acc[96:99], v157, v160, cbsz:4 blgp:4// MFMA m3_n0_k1
    v_mfma_scale_f32_16x16x128_f8f6f4 acc[100:103], v[152:155], v[88:91], acc[100:103], v157, v160, cbsz:4 blgp:4// MFMA m3_n1_k1
    v_mfma_scale_f32_16x16x128_f8f6f4 acc[104:107], v[152:155], v[96:99], acc[104:107], v157, v161, cbsz:4 blgp:4// MFMA m3_n2_k1
    v_mfma_scale_f32_16x16x128_f8f6f4 acc[108:111], v[152:155], v[104:107], acc[108:111], v157, v161, cbsz:4 blgp:4// MFMA m3_n3_k1
    v_mfma_scale_f32_16x16x128_f8f6f4 acc[112:115], v[152:155], v[112:115], acc[112:115], v157, v162, cbsz:4 blgp:4// MFMA m3_n4_k1
    v_mfma_scale_f32_16x16x128_f8f6f4 acc[116:119], v[152:155], v[120:123], acc[116:119], v157, v162, cbsz:4 blgp:4// MFMA m3_n5_k1
    v_mfma_scale_f32_16x16x128_f8f6f4 acc[120:123], v[152:155], v[128:131], acc[120:123], v157, v163, cbsz:4 blgp:4// MFMA m3_n6_k1
    v_mfma_scale_f32_16x16x128_f8f6f4 acc[124:127], v[152:155], v[136:139], acc[124:127], v157, v163, cbsz:4 blgp:4// MFMA m3_n7_k1
    ds_read_b128 v[148:151], v54                  // LR A m5k0 b1
    ds_read_b128 v[152:155], v55                  // LR A m5k1 b1
    v_mfma_scale_f32_16x16x128_f8f6f4 acc[128:131], v[140:143], v[76:79], acc[128:131], v158, v160, cbsz:4 blgp:4// MFMA m4_n0_k0
    v_mfma_scale_f32_16x16x128_f8f6f4 acc[132:135], v[140:143], v[84:87], acc[132:135], v158, v160, cbsz:4 blgp:4// MFMA m4_n1_k0
    v_mfma_scale_f32_16x16x128_f8f6f4 acc[136:139], v[140:143], v[92:95], acc[136:139], v158, v161, cbsz:4 blgp:4// MFMA m4_n2_k0
    v_mfma_scale_f32_16x16x128_f8f6f4 acc[140:143], v[140:143], v[100:103], acc[140:143], v158, v161, cbsz:4 blgp:4// MFMA m4_n3_k0
    v_mfma_scale_f32_16x16x128_f8f6f4 acc[144:147], v[140:143], v[108:111], acc[144:147], v158, v162, cbsz:4 blgp:4// MFMA m4_n4_k0
    v_mfma_scale_f32_16x16x128_f8f6f4 acc[148:151], v[140:143], v[116:119], acc[148:151], v158, v162, cbsz:4 blgp:4// MFMA m4_n5_k0
    v_mfma_scale_f32_16x16x128_f8f6f4 acc[152:155], v[140:143], v[124:127], acc[152:155], v158, v163, cbsz:4 blgp:4// MFMA m4_n6_k0
    v_mfma_scale_f32_16x16x128_f8f6f4 acc[156:159], v[140:143], v[132:135], acc[156:159], v158, v163, cbsz:4 blgp:4// MFMA m4_n7_k0
    v_mfma_scale_f32_16x16x128_f8f6f4 acc[128:131], v[144:147], v[80:83], acc[128:131], v158, v160, cbsz:4 blgp:4// MFMA m4_n0_k1
    v_mfma_scale_f32_16x16x128_f8f6f4 acc[132:135], v[144:147], v[88:91], acc[132:135], v158, v160, cbsz:4 blgp:4// MFMA m4_n1_k1
    v_mfma_scale_f32_16x16x128_f8f6f4 acc[136:139], v[144:147], v[96:99], acc[136:139], v158, v161, cbsz:4 blgp:4// MFMA m4_n2_k1
    v_mfma_scale_f32_16x16x128_f8f6f4 acc[140:143], v[144:147], v[104:107], acc[140:143], v158, v161, cbsz:4 blgp:4// MFMA m4_n3_k1
    v_mfma_scale_f32_16x16x128_f8f6f4 acc[144:147], v[144:147], v[112:115], acc[144:147], v158, v162, cbsz:4 blgp:4// MFMA m4_n4_k1
    v_mfma_scale_f32_16x16x128_f8f6f4 acc[148:151], v[144:147], v[120:123], acc[148:151], v158, v162, cbsz:4 blgp:4// MFMA m4_n5_k1
    v_mfma_scale_f32_16x16x128_f8f6f4 acc[152:155], v[144:147], v[128:131], acc[152:155], v158, v163, cbsz:4 blgp:4// MFMA m4_n6_k1
    v_mfma_scale_f32_16x16x128_f8f6f4 acc[156:159], v[144:147], v[136:139], acc[156:159], v158, v163, cbsz:4 blgp:4// MFMA m4_n7_k1
    ds_read_b128 v[140:143], v56                  // LR A m6k0 b0
    ds_read_b128 v[144:147], v57                  // LR A m6k1 b0
    v_mfma_scale_f32_16x16x128_f8f6f4 acc[160:163], v[148:151], v[76:79], acc[160:163], v158, v160, cbsz:4 blgp:4// MFMA m5_n0_k0
    v_mfma_scale_f32_16x16x128_f8f6f4 acc[164:167], v[148:151], v[84:87], acc[164:167], v158, v160, cbsz:4 blgp:4// MFMA m5_n1_k0
    v_mfma_scale_f32_16x16x128_f8f6f4 acc[168:171], v[148:151], v[92:95], acc[168:171], v158, v161, cbsz:4 blgp:4// MFMA m5_n2_k0
    v_mfma_scale_f32_16x16x128_f8f6f4 acc[172:175], v[148:151], v[100:103], acc[172:175], v158, v161, cbsz:4 blgp:4// MFMA m5_n3_k0
    v_mfma_scale_f32_16x16x128_f8f6f4 acc[176:179], v[148:151], v[108:111], acc[176:179], v158, v162, cbsz:4 blgp:4// MFMA m5_n4_k0
    v_mfma_scale_f32_16x16x128_f8f6f4 acc[180:183], v[148:151], v[116:119], acc[180:183], v158, v162, cbsz:4 blgp:4// MFMA m5_n5_k0
    v_mfma_scale_f32_16x16x128_f8f6f4 acc[184:187], v[148:151], v[124:127], acc[184:187], v158, v163, cbsz:4 blgp:4// MFMA m5_n6_k0
    v_mfma_scale_f32_16x16x128_f8f6f4 acc[188:191], v[148:151], v[132:135], acc[188:191], v158, v163, cbsz:4 blgp:4// MFMA m5_n7_k0
    v_mfma_scale_f32_16x16x128_f8f6f4 acc[160:163], v[152:155], v[80:83], acc[160:163], v158, v160, cbsz:4 blgp:4// MFMA m5_n0_k1
    v_mfma_scale_f32_16x16x128_f8f6f4 acc[164:167], v[152:155], v[88:91], acc[164:167], v158, v160, cbsz:4 blgp:4// MFMA m5_n1_k1
    v_mfma_scale_f32_16x16x128_f8f6f4 acc[168:171], v[152:155], v[96:99], acc[168:171], v158, v161, cbsz:4 blgp:4// MFMA m5_n2_k1
    v_mfma_scale_f32_16x16x128_f8f6f4 acc[172:175], v[152:155], v[104:107], acc[172:175], v158, v161, cbsz:4 blgp:4// MFMA m5_n3_k1
    v_mfma_scale_f32_16x16x128_f8f6f4 acc[176:179], v[152:155], v[112:115], acc[176:179], v158, v162, cbsz:4 blgp:4// MFMA m5_n4_k1
    v_mfma_scale_f32_16x16x128_f8f6f4 acc[180:183], v[152:155], v[120:123], acc[180:183], v158, v162, cbsz:4 blgp:4// MFMA m5_n5_k1
    v_mfma_scale_f32_16x16x128_f8f6f4 acc[184:187], v[152:155], v[128:131], acc[184:187], v158, v163, cbsz:4 blgp:4// MFMA m5_n6_k1
    v_mfma_scale_f32_16x16x128_f8f6f4 acc[188:191], v[152:155], v[136:139], acc[188:191], v158, v163, cbsz:4 blgp:4// MFMA m5_n7_k1
    ds_read_b128 v[148:151], v58                  // LR A m7k0 b1
    ds_read_b128 v[152:155], v59                  // LR A m7k1 b1
    v_mfma_scale_f32_16x16x128_f8f6f4 acc[192:195], v[140:143], v[76:79], acc[192:195], v159, v160, cbsz:4 blgp:4// MFMA m6_n0_k0
    v_mfma_scale_f32_16x16x128_f8f6f4 acc[196:199], v[140:143], v[84:87], acc[196:199], v159, v160, cbsz:4 blgp:4// MFMA m6_n1_k0
    v_mfma_scale_f32_16x16x128_f8f6f4 acc[200:203], v[140:143], v[92:95], acc[200:203], v159, v161, cbsz:4 blgp:4// MFMA m6_n2_k0
    v_mfma_scale_f32_16x16x128_f8f6f4 acc[204:207], v[140:143], v[100:103], acc[204:207], v159, v161, cbsz:4 blgp:4// MFMA m6_n3_k0
    v_mfma_scale_f32_16x16x128_f8f6f4 acc[208:211], v[140:143], v[108:111], acc[208:211], v159, v162, cbsz:4 blgp:4// MFMA m6_n4_k0
    v_mfma_scale_f32_16x16x128_f8f6f4 acc[212:215], v[140:143], v[116:119], acc[212:215], v159, v162, cbsz:4 blgp:4// MFMA m6_n5_k0
    v_mfma_scale_f32_16x16x128_f8f6f4 acc[216:219], v[140:143], v[124:127], acc[216:219], v159, v163, cbsz:4 blgp:4// MFMA m6_n6_k0
    v_mfma_scale_f32_16x16x128_f8f6f4 acc[220:223], v[140:143], v[132:135], acc[220:223], v159, v163, cbsz:4 blgp:4// MFMA m6_n7_k0
    v_mfma_scale_f32_16x16x128_f8f6f4 acc[192:195], v[144:147], v[80:83], acc[192:195], v159, v160, cbsz:4 blgp:4// MFMA m6_n0_k1
    v_mfma_scale_f32_16x16x128_f8f6f4 acc[196:199], v[144:147], v[88:91], acc[196:199], v159, v160, cbsz:4 blgp:4// MFMA m6_n1_k1
    v_mfma_scale_f32_16x16x128_f8f6f4 acc[200:203], v[144:147], v[96:99], acc[200:203], v159, v161, cbsz:4 blgp:4// MFMA m6_n2_k1
    v_mfma_scale_f32_16x16x128_f8f6f4 acc[204:207], v[144:147], v[104:107], acc[204:207], v159, v161, cbsz:4 blgp:4// MFMA m6_n3_k1
    v_mfma_scale_f32_16x16x128_f8f6f4 acc[208:211], v[144:147], v[112:115], acc[208:211], v159, v162, cbsz:4 blgp:4// MFMA m6_n4_k1
    v_mfma_scale_f32_16x16x128_f8f6f4 acc[212:215], v[144:147], v[120:123], acc[212:215], v159, v162, cbsz:4 blgp:4// MFMA m6_n5_k1
    v_mfma_scale_f32_16x16x128_f8f6f4 acc[216:219], v[144:147], v[128:131], acc[216:219], v159, v163, cbsz:4 blgp:4// MFMA m6_n6_k1
    v_mfma_scale_f32_16x16x128_f8f6f4 acc[220:223], v[144:147], v[136:139], acc[220:223], v159, v163, cbsz:4 blgp:4// MFMA m6_n7_k1
    v_mfma_scale_f32_16x16x128_f8f6f4 acc[224:227], v[148:151], v[76:79], acc[224:227], v159, v160, cbsz:4 blgp:4// MFMA m7_n0_k0
    v_mfma_scale_f32_16x16x128_f8f6f4 acc[228:231], v[148:151], v[84:87], acc[228:231], v159, v160, cbsz:4 blgp:4// MFMA m7_n1_k0
    v_mfma_scale_f32_16x16x128_f8f6f4 acc[232:235], v[148:151], v[92:95], acc[232:235], v159, v161, cbsz:4 blgp:4// MFMA m7_n2_k0
    v_mfma_scale_f32_16x16x128_f8f6f4 acc[236:239], v[148:151], v[100:103], acc[236:239], v159, v161, cbsz:4 blgp:4// MFMA m7_n3_k0
    v_mfma_scale_f32_16x16x128_f8f6f4 acc[240:243], v[148:151], v[108:111], acc[240:243], v159, v162, cbsz:4 blgp:4// MFMA m7_n4_k0
    v_mfma_scale_f32_16x16x128_f8f6f4 acc[244:247], v[148:151], v[116:119], acc[244:247], v159, v162, cbsz:4 blgp:4// MFMA m7_n5_k0
    v_mfma_scale_f32_16x16x128_f8f6f4 acc[248:251], v[148:151], v[124:127], acc[248:251], v159, v163, cbsz:4 blgp:4// MFMA m7_n6_k0
    v_mfma_scale_f32_16x16x128_f8f6f4 acc[252:255], v[148:151], v[132:135], acc[252:255], v159, v163, cbsz:4 blgp:4// MFMA m7_n7_k0
    v_mfma_scale_f32_16x16x128_f8f6f4 acc[224:227], v[152:155], v[80:83], acc[224:227], v159, v160, cbsz:4 blgp:4// MFMA m7_n0_k1
    v_mfma_scale_f32_16x16x128_f8f6f4 acc[228:231], v[152:155], v[88:91], acc[228:231], v159, v160, cbsz:4 blgp:4// MFMA m7_n1_k1
    v_mfma_scale_f32_16x16x128_f8f6f4 acc[232:235], v[152:155], v[96:99], acc[232:235], v159, v161, cbsz:4 blgp:4// MFMA m7_n2_k1
    v_mfma_scale_f32_16x16x128_f8f6f4 acc[236:239], v[152:155], v[104:107], acc[236:239], v159, v161, cbsz:4 blgp:4// MFMA m7_n3_k1
    v_mfma_scale_f32_16x16x128_f8f6f4 acc[240:243], v[152:155], v[112:115], acc[240:243], v159, v162, cbsz:4 blgp:4// MFMA m7_n4_k1
    v_mfma_scale_f32_16x16x128_f8f6f4 acc[244:247], v[152:155], v[120:123], acc[244:247], v159, v162, cbsz:4 blgp:4// MFMA m7_n5_k1
    v_mfma_scale_f32_16x16x128_f8f6f4 acc[248:251], v[152:155], v[128:131], acc[248:251], v159, v163, cbsz:4 blgp:4// MFMA m7_n6_k1
    v_mfma_scale_f32_16x16x128_f8f6f4 acc[252:255], v[152:155], v[136:139], acc[252:255], v159, v163, cbsz:4 blgp:4// MFMA m7_n7_k1
    v_add_u32 v166, v166, s64                     // scale_rd_a += db_step
    v_add_u32 v167, v167, s64                     // scale_rd_b += db_step
    // Toggle all precomputed read addresses (ADD)
    v_add_u32 v44, v44, s64                       // rd_a_m0_k0 += db
    v_add_u32 v45, v45, s64                       // rd_a_m0_k1 += db
    v_add_u32 v46, v46, s64                       // rd_a_m1_k0 += db
    v_add_u32 v47, v47, s64                       // rd_a_m1_k1 += db
    v_add_u32 v48, v48, s64                       // rd_a_m2_k0 += db
    v_add_u32 v49, v49, s64                       // rd_a_m2_k1 += db
    v_add_u32 v50, v50, s64                       // rd_a_m3_k0 += db
    v_add_u32 v51, v51, s64                       // rd_a_m3_k1 += db
    v_add_u32 v52, v52, s64                       // rd_a_m4_k0 += db
    v_add_u32 v53, v53, s64                       // rd_a_m4_k1 += db
    v_add_u32 v54, v54, s64                       // rd_a_m5_k0 += db
    v_add_u32 v55, v55, s64                       // rd_a_m5_k1 += db
    v_add_u32 v56, v56, s64                       // rd_a_m6_k0 += db
    v_add_u32 v57, v57, s64                       // rd_a_m6_k1 += db
    v_add_u32 v58, v58, s64                       // rd_a_m7_k0 += db
    v_add_u32 v59, v59, s64                       // rd_a_m7_k1 += db
    v_add_u32 v60, v60, s64                       // rd_b_n0_k0 += db
    v_add_u32 v61, v61, s64                       // rd_b_n0_k1 += db
    v_add_u32 v62, v62, s64                       // rd_b_n1_k0 += db
    v_add_u32 v63, v63, s64                       // rd_b_n1_k1 += db
    v_add_u32 v64, v64, s64                       // rd_b_n2_k0 += db
    v_add_u32 v65, v65, s64                       // rd_b_n2_k1 += db
    v_add_u32 v66, v66, s64                       // rd_b_n3_k0 += db
    v_add_u32 v67, v67, s64                       // rd_b_n3_k1 += db
    v_add_u32 v68, v68, s64                       // rd_b_n4_k0 += db
    v_add_u32 v69, v69, s64                       // rd_b_n4_k1 += db
    v_add_u32 v70, v70, s64                       // rd_b_n5_k0 += db
    v_add_u32 v71, v71, s64                       // rd_b_n5_k1 += db
    v_add_u32 v72, v72, s64                       // rd_b_n6_k0 += db
    v_add_u32 v73, v73, s64                       // rd_b_n6_k1 += db
    v_add_u32 v74, v74, s64                       // rd_b_n7_k0 += db
    v_add_u32 v75, v75, s64                       // rd_b_n7_k1 += db
    // [TODO] toggle_rd_data_b (scalar)

    // === Store D via buffer SRD ===
    // SRD for D matrix (raw buffer mode)
    s_mov_b32 s68, s8                             // SRD_D base lo
    s_mov_b32 s69, s9                             // SRD_D base hi
    s_mov_b32 s70, 0xFFFFFFFF                     // SRD_D size (unlimited)
    s_mov_b32 s71, 0x20000                        // SRD_D flags: raw buffer

    v_and_b32 v29, 15, v2                         // lane_n = lane_id % 16
    v_lshrrev_b32 v30, 4, v2                      // lane_id / 16
    v_lshlrev_b32 v30, 2, v30                     // * 4 -> lane_m_base
    v_lshlrev_b32 v26, 7, v3                      // wave_m * 128
    v_add_u32 v26, v26, v30                       // + lane_m_base
    s_mul_i32 s15, s2, 256                        // wg_id_x * 256
    v_add_u32 v26, s15, v26                       // + wg_base_m -> global_row
    v_lshlrev_b32 v27, 7, v4                      // wave_n * 128
    v_add_u32 v27, v27, v29                       // + lane_n
    s_mul_i32 s15, s3, 256                        // wg_id_y * 256
    v_add_u32 v27, s15, v27                       // + wg_base_n -> global_col
    v_mul_lo_u32 v27, s10, v27                    // global_col * M
    v_add_u32 v26, v26, v27                       // + global_row -> col-major linear index
    v_lshlrev_b32 v26, 1, v26                     // * 2 -> byte offset

    // Store 256 elements (8x8x4) via buffer_store_short
    s_lshl_b32 s14, s10, 1                        // col_stride = M * 2 bytes

    s_mov_b32 s15, 0                              // soffset = 0 (ni=0)
    v_accvgpr_read_b32 v29, acc0                  // acc[0] a0
    v_accvgpr_read_b32 v30, acc1                  // acc[1] a1
    v_cvt_pk_bf16_f32 v28, v29, v30               // pk cvt a0,a1
    v_accvgpr_read_b32 v29, acc2                  // acc[2] a2
    v_accvgpr_read_b32 v30, acc3                  // acc[3] a3
    v_cvt_pk_bf16_f32 v29, v29, v30               // pk cvt a2,a3
    buffer_store_dwordx2 v[28:29], v26, s[68:71], s15, offen nt// store D m0_n0 GWVW=4
    v_accvgpr_read_b32 v29, acc32                 // acc[32] a0
    v_accvgpr_read_b32 v30, acc33                 // acc[33] a1
    v_cvt_pk_bf16_f32 v28, v29, v30               // pk cvt a0,a1
    v_accvgpr_read_b32 v29, acc34                 // acc[34] a2
    v_accvgpr_read_b32 v30, acc35                 // acc[35] a3
    v_cvt_pk_bf16_f32 v29, v29, v30               // pk cvt a2,a3
    buffer_store_dwordx2 v[28:29], v26, s[68:71], s15, offen offset:32 nt// store D m1_n0 GWVW=4
    v_accvgpr_read_b32 v29, acc64                 // acc[64] a0
    v_accvgpr_read_b32 v30, acc65                 // acc[65] a1
    v_cvt_pk_bf16_f32 v28, v29, v30               // pk cvt a0,a1
    v_accvgpr_read_b32 v29, acc66                 // acc[66] a2
    v_accvgpr_read_b32 v30, acc67                 // acc[67] a3
    v_cvt_pk_bf16_f32 v29, v29, v30               // pk cvt a2,a3
    buffer_store_dwordx2 v[28:29], v26, s[68:71], s15, offen offset:64 nt// store D m2_n0 GWVW=4
    v_accvgpr_read_b32 v29, acc96                 // acc[96] a0
    v_accvgpr_read_b32 v30, acc97                 // acc[97] a1
    v_cvt_pk_bf16_f32 v28, v29, v30               // pk cvt a0,a1
    v_accvgpr_read_b32 v29, acc98                 // acc[98] a2
    v_accvgpr_read_b32 v30, acc99                 // acc[99] a3
    v_cvt_pk_bf16_f32 v29, v29, v30               // pk cvt a2,a3
    buffer_store_dwordx2 v[28:29], v26, s[68:71], s15, offen offset:96 nt// store D m3_n0 GWVW=4
    v_accvgpr_read_b32 v29, acc128                // acc[128] a0
    v_accvgpr_read_b32 v30, acc129                // acc[129] a1
    v_cvt_pk_bf16_f32 v28, v29, v30               // pk cvt a0,a1
    v_accvgpr_read_b32 v29, acc130                // acc[130] a2
    v_accvgpr_read_b32 v30, acc131                // acc[131] a3
    v_cvt_pk_bf16_f32 v29, v29, v30               // pk cvt a2,a3
    buffer_store_dwordx2 v[28:29], v26, s[68:71], s15, offen offset:128 nt// store D m4_n0 GWVW=4
    v_accvgpr_read_b32 v29, acc160                // acc[160] a0
    v_accvgpr_read_b32 v30, acc161                // acc[161] a1
    v_cvt_pk_bf16_f32 v28, v29, v30               // pk cvt a0,a1
    v_accvgpr_read_b32 v29, acc162                // acc[162] a2
    v_accvgpr_read_b32 v30, acc163                // acc[163] a3
    v_cvt_pk_bf16_f32 v29, v29, v30               // pk cvt a2,a3
    buffer_store_dwordx2 v[28:29], v26, s[68:71], s15, offen offset:160 nt// store D m5_n0 GWVW=4
    v_accvgpr_read_b32 v29, acc192                // acc[192] a0
    v_accvgpr_read_b32 v30, acc193                // acc[193] a1
    v_cvt_pk_bf16_f32 v28, v29, v30               // pk cvt a0,a1
    v_accvgpr_read_b32 v29, acc194                // acc[194] a2
    v_accvgpr_read_b32 v30, acc195                // acc[195] a3
    v_cvt_pk_bf16_f32 v29, v29, v30               // pk cvt a2,a3
    buffer_store_dwordx2 v[28:29], v26, s[68:71], s15, offen offset:192 nt// store D m6_n0 GWVW=4
    v_accvgpr_read_b32 v29, acc224                // acc[224] a0
    v_accvgpr_read_b32 v30, acc225                // acc[225] a1
    v_cvt_pk_bf16_f32 v28, v29, v30               // pk cvt a0,a1
    v_accvgpr_read_b32 v29, acc226                // acc[226] a2
    v_accvgpr_read_b32 v30, acc227                // acc[227] a3
    v_cvt_pk_bf16_f32 v29, v29, v30               // pk cvt a2,a3
    buffer_store_dwordx2 v[28:29], v26, s[68:71], s15, offen offset:224 nt// store D m7_n0 GWVW=4
    s_mul_i32 s15, s14, 16                        // soffset = 16 * col_stride (ni=1)
    v_accvgpr_read_b32 v29, acc4                  // acc[4] a0
    v_accvgpr_read_b32 v30, acc5                  // acc[5] a1
    v_cvt_pk_bf16_f32 v28, v29, v30               // pk cvt a0,a1
    v_accvgpr_read_b32 v29, acc6                  // acc[6] a2
    v_accvgpr_read_b32 v30, acc7                  // acc[7] a3
    v_cvt_pk_bf16_f32 v29, v29, v30               // pk cvt a2,a3
    buffer_store_dwordx2 v[28:29], v26, s[68:71], s15, offen nt// store D m0_n1 GWVW=4
    v_accvgpr_read_b32 v29, acc36                 // acc[36] a0
    v_accvgpr_read_b32 v30, acc37                 // acc[37] a1
    v_cvt_pk_bf16_f32 v28, v29, v30               // pk cvt a0,a1
    v_accvgpr_read_b32 v29, acc38                 // acc[38] a2
    v_accvgpr_read_b32 v30, acc39                 // acc[39] a3
    v_cvt_pk_bf16_f32 v29, v29, v30               // pk cvt a2,a3
    buffer_store_dwordx2 v[28:29], v26, s[68:71], s15, offen offset:32 nt// store D m1_n1 GWVW=4
    v_accvgpr_read_b32 v29, acc68                 // acc[68] a0
    v_accvgpr_read_b32 v30, acc69                 // acc[69] a1
    v_cvt_pk_bf16_f32 v28, v29, v30               // pk cvt a0,a1
    v_accvgpr_read_b32 v29, acc70                 // acc[70] a2
    v_accvgpr_read_b32 v30, acc71                 // acc[71] a3
    v_cvt_pk_bf16_f32 v29, v29, v30               // pk cvt a2,a3
    buffer_store_dwordx2 v[28:29], v26, s[68:71], s15, offen offset:64 nt// store D m2_n1 GWVW=4
    v_accvgpr_read_b32 v29, acc100                // acc[100] a0
    v_accvgpr_read_b32 v30, acc101                // acc[101] a1
    v_cvt_pk_bf16_f32 v28, v29, v30               // pk cvt a0,a1
    v_accvgpr_read_b32 v29, acc102                // acc[102] a2
    v_accvgpr_read_b32 v30, acc103                // acc[103] a3
    v_cvt_pk_bf16_f32 v29, v29, v30               // pk cvt a2,a3
    buffer_store_dwordx2 v[28:29], v26, s[68:71], s15, offen offset:96 nt// store D m3_n1 GWVW=4
    v_accvgpr_read_b32 v29, acc132                // acc[132] a0
    v_accvgpr_read_b32 v30, acc133                // acc[133] a1
    v_cvt_pk_bf16_f32 v28, v29, v30               // pk cvt a0,a1
    v_accvgpr_read_b32 v29, acc134                // acc[134] a2
    v_accvgpr_read_b32 v30, acc135                // acc[135] a3
    v_cvt_pk_bf16_f32 v29, v29, v30               // pk cvt a2,a3
    buffer_store_dwordx2 v[28:29], v26, s[68:71], s15, offen offset:128 nt// store D m4_n1 GWVW=4
    v_accvgpr_read_b32 v29, acc164                // acc[164] a0
    v_accvgpr_read_b32 v30, acc165                // acc[165] a1
    v_cvt_pk_bf16_f32 v28, v29, v30               // pk cvt a0,a1
    v_accvgpr_read_b32 v29, acc166                // acc[166] a2
    v_accvgpr_read_b32 v30, acc167                // acc[167] a3
    v_cvt_pk_bf16_f32 v29, v29, v30               // pk cvt a2,a3
    buffer_store_dwordx2 v[28:29], v26, s[68:71], s15, offen offset:160 nt// store D m5_n1 GWVW=4
    v_accvgpr_read_b32 v29, acc196                // acc[196] a0
    v_accvgpr_read_b32 v30, acc197                // acc[197] a1
    v_cvt_pk_bf16_f32 v28, v29, v30               // pk cvt a0,a1
    v_accvgpr_read_b32 v29, acc198                // acc[198] a2
    v_accvgpr_read_b32 v30, acc199                // acc[199] a3
    v_cvt_pk_bf16_f32 v29, v29, v30               // pk cvt a2,a3
    buffer_store_dwordx2 v[28:29], v26, s[68:71], s15, offen offset:192 nt// store D m6_n1 GWVW=4
    v_accvgpr_read_b32 v29, acc228                // acc[228] a0
    v_accvgpr_read_b32 v30, acc229                // acc[229] a1
    v_cvt_pk_bf16_f32 v28, v29, v30               // pk cvt a0,a1
    v_accvgpr_read_b32 v29, acc230                // acc[230] a2
    v_accvgpr_read_b32 v30, acc231                // acc[231] a3
    v_cvt_pk_bf16_f32 v29, v29, v30               // pk cvt a2,a3
    buffer_store_dwordx2 v[28:29], v26, s[68:71], s15, offen offset:224 nt// store D m7_n1 GWVW=4
    s_mul_i32 s15, s14, 32                        // soffset = 32 * col_stride (ni=2)
    v_accvgpr_read_b32 v29, acc8                  // acc[8] a0
    v_accvgpr_read_b32 v30, acc9                  // acc[9] a1
    v_cvt_pk_bf16_f32 v28, v29, v30               // pk cvt a0,a1
    v_accvgpr_read_b32 v29, acc10                 // acc[10] a2
    v_accvgpr_read_b32 v30, acc11                 // acc[11] a3
    v_cvt_pk_bf16_f32 v29, v29, v30               // pk cvt a2,a3
    buffer_store_dwordx2 v[28:29], v26, s[68:71], s15, offen nt// store D m0_n2 GWVW=4
    v_accvgpr_read_b32 v29, acc40                 // acc[40] a0
    v_accvgpr_read_b32 v30, acc41                 // acc[41] a1
    v_cvt_pk_bf16_f32 v28, v29, v30               // pk cvt a0,a1
    v_accvgpr_read_b32 v29, acc42                 // acc[42] a2
    v_accvgpr_read_b32 v30, acc43                 // acc[43] a3
    v_cvt_pk_bf16_f32 v29, v29, v30               // pk cvt a2,a3
    buffer_store_dwordx2 v[28:29], v26, s[68:71], s15, offen offset:32 nt// store D m1_n2 GWVW=4
    v_accvgpr_read_b32 v29, acc72                 // acc[72] a0
    v_accvgpr_read_b32 v30, acc73                 // acc[73] a1
    v_cvt_pk_bf16_f32 v28, v29, v30               // pk cvt a0,a1
    v_accvgpr_read_b32 v29, acc74                 // acc[74] a2
    v_accvgpr_read_b32 v30, acc75                 // acc[75] a3
    v_cvt_pk_bf16_f32 v29, v29, v30               // pk cvt a2,a3
    buffer_store_dwordx2 v[28:29], v26, s[68:71], s15, offen offset:64 nt// store D m2_n2 GWVW=4
    v_accvgpr_read_b32 v29, acc104                // acc[104] a0
    v_accvgpr_read_b32 v30, acc105                // acc[105] a1
    v_cvt_pk_bf16_f32 v28, v29, v30               // pk cvt a0,a1
    v_accvgpr_read_b32 v29, acc106                // acc[106] a2
    v_accvgpr_read_b32 v30, acc107                // acc[107] a3
    v_cvt_pk_bf16_f32 v29, v29, v30               // pk cvt a2,a3
    buffer_store_dwordx2 v[28:29], v26, s[68:71], s15, offen offset:96 nt// store D m3_n2 GWVW=4
    v_accvgpr_read_b32 v29, acc136                // acc[136] a0
    v_accvgpr_read_b32 v30, acc137                // acc[137] a1
    v_cvt_pk_bf16_f32 v28, v29, v30               // pk cvt a0,a1
    v_accvgpr_read_b32 v29, acc138                // acc[138] a2
    v_accvgpr_read_b32 v30, acc139                // acc[139] a3
    v_cvt_pk_bf16_f32 v29, v29, v30               // pk cvt a2,a3
    buffer_store_dwordx2 v[28:29], v26, s[68:71], s15, offen offset:128 nt// store D m4_n2 GWVW=4
    v_accvgpr_read_b32 v29, acc168                // acc[168] a0
    v_accvgpr_read_b32 v30, acc169                // acc[169] a1
    v_cvt_pk_bf16_f32 v28, v29, v30               // pk cvt a0,a1
    v_accvgpr_read_b32 v29, acc170                // acc[170] a2
    v_accvgpr_read_b32 v30, acc171                // acc[171] a3
    v_cvt_pk_bf16_f32 v29, v29, v30               // pk cvt a2,a3
    buffer_store_dwordx2 v[28:29], v26, s[68:71], s15, offen offset:160 nt// store D m5_n2 GWVW=4
    v_accvgpr_read_b32 v29, acc200                // acc[200] a0
    v_accvgpr_read_b32 v30, acc201                // acc[201] a1
    v_cvt_pk_bf16_f32 v28, v29, v30               // pk cvt a0,a1
    v_accvgpr_read_b32 v29, acc202                // acc[202] a2
    v_accvgpr_read_b32 v30, acc203                // acc[203] a3
    v_cvt_pk_bf16_f32 v29, v29, v30               // pk cvt a2,a3
    buffer_store_dwordx2 v[28:29], v26, s[68:71], s15, offen offset:192 nt// store D m6_n2 GWVW=4
    v_accvgpr_read_b32 v29, acc232                // acc[232] a0
    v_accvgpr_read_b32 v30, acc233                // acc[233] a1
    v_cvt_pk_bf16_f32 v28, v29, v30               // pk cvt a0,a1
    v_accvgpr_read_b32 v29, acc234                // acc[234] a2
    v_accvgpr_read_b32 v30, acc235                // acc[235] a3
    v_cvt_pk_bf16_f32 v29, v29, v30               // pk cvt a2,a3
    buffer_store_dwordx2 v[28:29], v26, s[68:71], s15, offen offset:224 nt// store D m7_n2 GWVW=4
    s_mul_i32 s15, s14, 48                        // soffset = 48 * col_stride (ni=3)
    v_accvgpr_read_b32 v29, acc12                 // acc[12] a0
    v_accvgpr_read_b32 v30, acc13                 // acc[13] a1
    v_cvt_pk_bf16_f32 v28, v29, v30               // pk cvt a0,a1
    v_accvgpr_read_b32 v29, acc14                 // acc[14] a2
    v_accvgpr_read_b32 v30, acc15                 // acc[15] a3
    v_cvt_pk_bf16_f32 v29, v29, v30               // pk cvt a2,a3
    buffer_store_dwordx2 v[28:29], v26, s[68:71], s15, offen nt// store D m0_n3 GWVW=4
    v_accvgpr_read_b32 v29, acc44                 // acc[44] a0
    v_accvgpr_read_b32 v30, acc45                 // acc[45] a1
    v_cvt_pk_bf16_f32 v28, v29, v30               // pk cvt a0,a1
    v_accvgpr_read_b32 v29, acc46                 // acc[46] a2
    v_accvgpr_read_b32 v30, acc47                 // acc[47] a3
    v_cvt_pk_bf16_f32 v29, v29, v30               // pk cvt a2,a3
    buffer_store_dwordx2 v[28:29], v26, s[68:71], s15, offen offset:32 nt// store D m1_n3 GWVW=4
    v_accvgpr_read_b32 v29, acc76                 // acc[76] a0
    v_accvgpr_read_b32 v30, acc77                 // acc[77] a1
    v_cvt_pk_bf16_f32 v28, v29, v30               // pk cvt a0,a1
    v_accvgpr_read_b32 v29, acc78                 // acc[78] a2
    v_accvgpr_read_b32 v30, acc79                 // acc[79] a3
    v_cvt_pk_bf16_f32 v29, v29, v30               // pk cvt a2,a3
    buffer_store_dwordx2 v[28:29], v26, s[68:71], s15, offen offset:64 nt// store D m2_n3 GWVW=4
    v_accvgpr_read_b32 v29, acc108                // acc[108] a0
    v_accvgpr_read_b32 v30, acc109                // acc[109] a1
    v_cvt_pk_bf16_f32 v28, v29, v30               // pk cvt a0,a1
    v_accvgpr_read_b32 v29, acc110                // acc[110] a2
    v_accvgpr_read_b32 v30, acc111                // acc[111] a3
    v_cvt_pk_bf16_f32 v29, v29, v30               // pk cvt a2,a3
    buffer_store_dwordx2 v[28:29], v26, s[68:71], s15, offen offset:96 nt// store D m3_n3 GWVW=4
    v_accvgpr_read_b32 v29, acc140                // acc[140] a0
    v_accvgpr_read_b32 v30, acc141                // acc[141] a1
    v_cvt_pk_bf16_f32 v28, v29, v30               // pk cvt a0,a1
    v_accvgpr_read_b32 v29, acc142                // acc[142] a2
    v_accvgpr_read_b32 v30, acc143                // acc[143] a3
    v_cvt_pk_bf16_f32 v29, v29, v30               // pk cvt a2,a3
    buffer_store_dwordx2 v[28:29], v26, s[68:71], s15, offen offset:128 nt// store D m4_n3 GWVW=4
    v_accvgpr_read_b32 v29, acc172                // acc[172] a0
    v_accvgpr_read_b32 v30, acc173                // acc[173] a1
    v_cvt_pk_bf16_f32 v28, v29, v30               // pk cvt a0,a1
    v_accvgpr_read_b32 v29, acc174                // acc[174] a2
    v_accvgpr_read_b32 v30, acc175                // acc[175] a3
    v_cvt_pk_bf16_f32 v29, v29, v30               // pk cvt a2,a3
    buffer_store_dwordx2 v[28:29], v26, s[68:71], s15, offen offset:160 nt// store D m5_n3 GWVW=4
    v_accvgpr_read_b32 v29, acc204                // acc[204] a0
    v_accvgpr_read_b32 v30, acc205                // acc[205] a1
    v_cvt_pk_bf16_f32 v28, v29, v30               // pk cvt a0,a1
    v_accvgpr_read_b32 v29, acc206                // acc[206] a2
    v_accvgpr_read_b32 v30, acc207                // acc[207] a3
    v_cvt_pk_bf16_f32 v29, v29, v30               // pk cvt a2,a3
    buffer_store_dwordx2 v[28:29], v26, s[68:71], s15, offen offset:192 nt// store D m6_n3 GWVW=4
    v_accvgpr_read_b32 v29, acc236                // acc[236] a0
    v_accvgpr_read_b32 v30, acc237                // acc[237] a1
    v_cvt_pk_bf16_f32 v28, v29, v30               // pk cvt a0,a1
    v_accvgpr_read_b32 v29, acc238                // acc[238] a2
    v_accvgpr_read_b32 v30, acc239                // acc[239] a3
    v_cvt_pk_bf16_f32 v29, v29, v30               // pk cvt a2,a3
    buffer_store_dwordx2 v[28:29], v26, s[68:71], s15, offen offset:224 nt// store D m7_n3 GWVW=4
    s_mul_i32 s15, s14, 64                        // soffset = 64 * col_stride (ni=4)
    v_accvgpr_read_b32 v29, acc16                 // acc[16] a0
    v_accvgpr_read_b32 v30, acc17                 // acc[17] a1
    v_cvt_pk_bf16_f32 v28, v29, v30               // pk cvt a0,a1
    v_accvgpr_read_b32 v29, acc18                 // acc[18] a2
    v_accvgpr_read_b32 v30, acc19                 // acc[19] a3
    v_cvt_pk_bf16_f32 v29, v29, v30               // pk cvt a2,a3
    buffer_store_dwordx2 v[28:29], v26, s[68:71], s15, offen nt// store D m0_n4 GWVW=4
    v_accvgpr_read_b32 v29, acc48                 // acc[48] a0
    v_accvgpr_read_b32 v30, acc49                 // acc[49] a1
    v_cvt_pk_bf16_f32 v28, v29, v30               // pk cvt a0,a1
    v_accvgpr_read_b32 v29, acc50                 // acc[50] a2
    v_accvgpr_read_b32 v30, acc51                 // acc[51] a3
    v_cvt_pk_bf16_f32 v29, v29, v30               // pk cvt a2,a3
    buffer_store_dwordx2 v[28:29], v26, s[68:71], s15, offen offset:32 nt// store D m1_n4 GWVW=4
    v_accvgpr_read_b32 v29, acc80                 // acc[80] a0
    v_accvgpr_read_b32 v30, acc81                 // acc[81] a1
    v_cvt_pk_bf16_f32 v28, v29, v30               // pk cvt a0,a1
    v_accvgpr_read_b32 v29, acc82                 // acc[82] a2
    v_accvgpr_read_b32 v30, acc83                 // acc[83] a3
    v_cvt_pk_bf16_f32 v29, v29, v30               // pk cvt a2,a3
    buffer_store_dwordx2 v[28:29], v26, s[68:71], s15, offen offset:64 nt// store D m2_n4 GWVW=4
    v_accvgpr_read_b32 v29, acc112                // acc[112] a0
    v_accvgpr_read_b32 v30, acc113                // acc[113] a1
    v_cvt_pk_bf16_f32 v28, v29, v30               // pk cvt a0,a1
    v_accvgpr_read_b32 v29, acc114                // acc[114] a2
    v_accvgpr_read_b32 v30, acc115                // acc[115] a3
    v_cvt_pk_bf16_f32 v29, v29, v30               // pk cvt a2,a3
    buffer_store_dwordx2 v[28:29], v26, s[68:71], s15, offen offset:96 nt// store D m3_n4 GWVW=4
    v_accvgpr_read_b32 v29, acc144                // acc[144] a0
    v_accvgpr_read_b32 v30, acc145                // acc[145] a1
    v_cvt_pk_bf16_f32 v28, v29, v30               // pk cvt a0,a1
    v_accvgpr_read_b32 v29, acc146                // acc[146] a2
    v_accvgpr_read_b32 v30, acc147                // acc[147] a3
    v_cvt_pk_bf16_f32 v29, v29, v30               // pk cvt a2,a3
    buffer_store_dwordx2 v[28:29], v26, s[68:71], s15, offen offset:128 nt// store D m4_n4 GWVW=4
    v_accvgpr_read_b32 v29, acc176                // acc[176] a0
    v_accvgpr_read_b32 v30, acc177                // acc[177] a1
    v_cvt_pk_bf16_f32 v28, v29, v30               // pk cvt a0,a1
    v_accvgpr_read_b32 v29, acc178                // acc[178] a2
    v_accvgpr_read_b32 v30, acc179                // acc[179] a3
    v_cvt_pk_bf16_f32 v29, v29, v30               // pk cvt a2,a3
    buffer_store_dwordx2 v[28:29], v26, s[68:71], s15, offen offset:160 nt// store D m5_n4 GWVW=4
    v_accvgpr_read_b32 v29, acc208                // acc[208] a0
    v_accvgpr_read_b32 v30, acc209                // acc[209] a1
    v_cvt_pk_bf16_f32 v28, v29, v30               // pk cvt a0,a1
    v_accvgpr_read_b32 v29, acc210                // acc[210] a2
    v_accvgpr_read_b32 v30, acc211                // acc[211] a3
    v_cvt_pk_bf16_f32 v29, v29, v30               // pk cvt a2,a3
    buffer_store_dwordx2 v[28:29], v26, s[68:71], s15, offen offset:192 nt// store D m6_n4 GWVW=4
    v_accvgpr_read_b32 v29, acc240                // acc[240] a0
    v_accvgpr_read_b32 v30, acc241                // acc[241] a1
    v_cvt_pk_bf16_f32 v28, v29, v30               // pk cvt a0,a1
    v_accvgpr_read_b32 v29, acc242                // acc[242] a2
    v_accvgpr_read_b32 v30, acc243                // acc[243] a3
    v_cvt_pk_bf16_f32 v29, v29, v30               // pk cvt a2,a3
    buffer_store_dwordx2 v[28:29], v26, s[68:71], s15, offen offset:224 nt// store D m7_n4 GWVW=4
    s_mul_i32 s15, s14, 80                        // soffset = 80 * col_stride (ni=5)
    v_accvgpr_read_b32 v29, acc20                 // acc[20] a0
    v_accvgpr_read_b32 v30, acc21                 // acc[21] a1
    v_cvt_pk_bf16_f32 v28, v29, v30               // pk cvt a0,a1
    v_accvgpr_read_b32 v29, acc22                 // acc[22] a2
    v_accvgpr_read_b32 v30, acc23                 // acc[23] a3
    v_cvt_pk_bf16_f32 v29, v29, v30               // pk cvt a2,a3
    buffer_store_dwordx2 v[28:29], v26, s[68:71], s15, offen nt// store D m0_n5 GWVW=4
    v_accvgpr_read_b32 v29, acc52                 // acc[52] a0
    v_accvgpr_read_b32 v30, acc53                 // acc[53] a1
    v_cvt_pk_bf16_f32 v28, v29, v30               // pk cvt a0,a1
    v_accvgpr_read_b32 v29, acc54                 // acc[54] a2
    v_accvgpr_read_b32 v30, acc55                 // acc[55] a3
    v_cvt_pk_bf16_f32 v29, v29, v30               // pk cvt a2,a3
    buffer_store_dwordx2 v[28:29], v26, s[68:71], s15, offen offset:32 nt// store D m1_n5 GWVW=4
    v_accvgpr_read_b32 v29, acc84                 // acc[84] a0
    v_accvgpr_read_b32 v30, acc85                 // acc[85] a1
    v_cvt_pk_bf16_f32 v28, v29, v30               // pk cvt a0,a1
    v_accvgpr_read_b32 v29, acc86                 // acc[86] a2
    v_accvgpr_read_b32 v30, acc87                 // acc[87] a3
    v_cvt_pk_bf16_f32 v29, v29, v30               // pk cvt a2,a3
    buffer_store_dwordx2 v[28:29], v26, s[68:71], s15, offen offset:64 nt// store D m2_n5 GWVW=4
    v_accvgpr_read_b32 v29, acc116                // acc[116] a0
    v_accvgpr_read_b32 v30, acc117                // acc[117] a1
    v_cvt_pk_bf16_f32 v28, v29, v30               // pk cvt a0,a1
    v_accvgpr_read_b32 v29, acc118                // acc[118] a2
    v_accvgpr_read_b32 v30, acc119                // acc[119] a3
    v_cvt_pk_bf16_f32 v29, v29, v30               // pk cvt a2,a3
    buffer_store_dwordx2 v[28:29], v26, s[68:71], s15, offen offset:96 nt// store D m3_n5 GWVW=4
    v_accvgpr_read_b32 v29, acc148                // acc[148] a0
    v_accvgpr_read_b32 v30, acc149                // acc[149] a1
    v_cvt_pk_bf16_f32 v28, v29, v30               // pk cvt a0,a1
    v_accvgpr_read_b32 v29, acc150                // acc[150] a2
    v_accvgpr_read_b32 v30, acc151                // acc[151] a3
    v_cvt_pk_bf16_f32 v29, v29, v30               // pk cvt a2,a3
    buffer_store_dwordx2 v[28:29], v26, s[68:71], s15, offen offset:128 nt// store D m4_n5 GWVW=4
    v_accvgpr_read_b32 v29, acc180                // acc[180] a0
    v_accvgpr_read_b32 v30, acc181                // acc[181] a1
    v_cvt_pk_bf16_f32 v28, v29, v30               // pk cvt a0,a1
    v_accvgpr_read_b32 v29, acc182                // acc[182] a2
    v_accvgpr_read_b32 v30, acc183                // acc[183] a3
    v_cvt_pk_bf16_f32 v29, v29, v30               // pk cvt a2,a3
    buffer_store_dwordx2 v[28:29], v26, s[68:71], s15, offen offset:160 nt// store D m5_n5 GWVW=4
    v_accvgpr_read_b32 v29, acc212                // acc[212] a0
    v_accvgpr_read_b32 v30, acc213                // acc[213] a1
    v_cvt_pk_bf16_f32 v28, v29, v30               // pk cvt a0,a1
    v_accvgpr_read_b32 v29, acc214                // acc[214] a2
    v_accvgpr_read_b32 v30, acc215                // acc[215] a3
    v_cvt_pk_bf16_f32 v29, v29, v30               // pk cvt a2,a3
    buffer_store_dwordx2 v[28:29], v26, s[68:71], s15, offen offset:192 nt// store D m6_n5 GWVW=4
    v_accvgpr_read_b32 v29, acc244                // acc[244] a0
    v_accvgpr_read_b32 v30, acc245                // acc[245] a1
    v_cvt_pk_bf16_f32 v28, v29, v30               // pk cvt a0,a1
    v_accvgpr_read_b32 v29, acc246                // acc[246] a2
    v_accvgpr_read_b32 v30, acc247                // acc[247] a3
    v_cvt_pk_bf16_f32 v29, v29, v30               // pk cvt a2,a3
    buffer_store_dwordx2 v[28:29], v26, s[68:71], s15, offen offset:224 nt// store D m7_n5 GWVW=4
    s_mul_i32 s15, s14, 96                        // soffset = 96 * col_stride (ni=6)
    v_accvgpr_read_b32 v29, acc24                 // acc[24] a0
    v_accvgpr_read_b32 v30, acc25                 // acc[25] a1
    v_cvt_pk_bf16_f32 v28, v29, v30               // pk cvt a0,a1
    v_accvgpr_read_b32 v29, acc26                 // acc[26] a2
    v_accvgpr_read_b32 v30, acc27                 // acc[27] a3
    v_cvt_pk_bf16_f32 v29, v29, v30               // pk cvt a2,a3
    buffer_store_dwordx2 v[28:29], v26, s[68:71], s15, offen nt// store D m0_n6 GWVW=4
    v_accvgpr_read_b32 v29, acc56                 // acc[56] a0
    v_accvgpr_read_b32 v30, acc57                 // acc[57] a1
    v_cvt_pk_bf16_f32 v28, v29, v30               // pk cvt a0,a1
    v_accvgpr_read_b32 v29, acc58                 // acc[58] a2
    v_accvgpr_read_b32 v30, acc59                 // acc[59] a3
    v_cvt_pk_bf16_f32 v29, v29, v30               // pk cvt a2,a3
    buffer_store_dwordx2 v[28:29], v26, s[68:71], s15, offen offset:32 nt// store D m1_n6 GWVW=4
    v_accvgpr_read_b32 v29, acc88                 // acc[88] a0
    v_accvgpr_read_b32 v30, acc89                 // acc[89] a1
    v_cvt_pk_bf16_f32 v28, v29, v30               // pk cvt a0,a1
    v_accvgpr_read_b32 v29, acc90                 // acc[90] a2
    v_accvgpr_read_b32 v30, acc91                 // acc[91] a3
    v_cvt_pk_bf16_f32 v29, v29, v30               // pk cvt a2,a3
    buffer_store_dwordx2 v[28:29], v26, s[68:71], s15, offen offset:64 nt// store D m2_n6 GWVW=4
    v_accvgpr_read_b32 v29, acc120                // acc[120] a0
    v_accvgpr_read_b32 v30, acc121                // acc[121] a1
    v_cvt_pk_bf16_f32 v28, v29, v30               // pk cvt a0,a1
    v_accvgpr_read_b32 v29, acc122                // acc[122] a2
    v_accvgpr_read_b32 v30, acc123                // acc[123] a3
    v_cvt_pk_bf16_f32 v29, v29, v30               // pk cvt a2,a3
    buffer_store_dwordx2 v[28:29], v26, s[68:71], s15, offen offset:96 nt// store D m3_n6 GWVW=4
    v_accvgpr_read_b32 v29, acc152                // acc[152] a0
    v_accvgpr_read_b32 v30, acc153                // acc[153] a1
    v_cvt_pk_bf16_f32 v28, v29, v30               // pk cvt a0,a1
    v_accvgpr_read_b32 v29, acc154                // acc[154] a2
    v_accvgpr_read_b32 v30, acc155                // acc[155] a3
    v_cvt_pk_bf16_f32 v29, v29, v30               // pk cvt a2,a3
    buffer_store_dwordx2 v[28:29], v26, s[68:71], s15, offen offset:128 nt// store D m4_n6 GWVW=4
    v_accvgpr_read_b32 v29, acc184                // acc[184] a0
    v_accvgpr_read_b32 v30, acc185                // acc[185] a1
    v_cvt_pk_bf16_f32 v28, v29, v30               // pk cvt a0,a1
    v_accvgpr_read_b32 v29, acc186                // acc[186] a2
    v_accvgpr_read_b32 v30, acc187                // acc[187] a3
    v_cvt_pk_bf16_f32 v29, v29, v30               // pk cvt a2,a3
    buffer_store_dwordx2 v[28:29], v26, s[68:71], s15, offen offset:160 nt// store D m5_n6 GWVW=4
    v_accvgpr_read_b32 v29, acc216                // acc[216] a0
    v_accvgpr_read_b32 v30, acc217                // acc[217] a1
    v_cvt_pk_bf16_f32 v28, v29, v30               // pk cvt a0,a1
    v_accvgpr_read_b32 v29, acc218                // acc[218] a2
    v_accvgpr_read_b32 v30, acc219                // acc[219] a3
    v_cvt_pk_bf16_f32 v29, v29, v30               // pk cvt a2,a3
    buffer_store_dwordx2 v[28:29], v26, s[68:71], s15, offen offset:192 nt// store D m6_n6 GWVW=4
    v_accvgpr_read_b32 v29, acc248                // acc[248] a0
    v_accvgpr_read_b32 v30, acc249                // acc[249] a1
    v_cvt_pk_bf16_f32 v28, v29, v30               // pk cvt a0,a1
    v_accvgpr_read_b32 v29, acc250                // acc[250] a2
    v_accvgpr_read_b32 v30, acc251                // acc[251] a3
    v_cvt_pk_bf16_f32 v29, v29, v30               // pk cvt a2,a3
    buffer_store_dwordx2 v[28:29], v26, s[68:71], s15, offen offset:224 nt// store D m7_n6 GWVW=4
    s_mul_i32 s15, s14, 112                       // soffset = 112 * col_stride (ni=7)
    v_accvgpr_read_b32 v29, acc28                 // acc[28] a0
    v_accvgpr_read_b32 v30, acc29                 // acc[29] a1
    v_cvt_pk_bf16_f32 v28, v29, v30               // pk cvt a0,a1
    v_accvgpr_read_b32 v29, acc30                 // acc[30] a2
    v_accvgpr_read_b32 v30, acc31                 // acc[31] a3
    v_cvt_pk_bf16_f32 v29, v29, v30               // pk cvt a2,a3
    buffer_store_dwordx2 v[28:29], v26, s[68:71], s15, offen nt// store D m0_n7 GWVW=4
    v_accvgpr_read_b32 v29, acc60                 // acc[60] a0
    v_accvgpr_read_b32 v30, acc61                 // acc[61] a1
    v_cvt_pk_bf16_f32 v28, v29, v30               // pk cvt a0,a1
    v_accvgpr_read_b32 v29, acc62                 // acc[62] a2
    v_accvgpr_read_b32 v30, acc63                 // acc[63] a3
    v_cvt_pk_bf16_f32 v29, v29, v30               // pk cvt a2,a3
    buffer_store_dwordx2 v[28:29], v26, s[68:71], s15, offen offset:32 nt// store D m1_n7 GWVW=4
    v_accvgpr_read_b32 v29, acc92                 // acc[92] a0
    v_accvgpr_read_b32 v30, acc93                 // acc[93] a1
    v_cvt_pk_bf16_f32 v28, v29, v30               // pk cvt a0,a1
    v_accvgpr_read_b32 v29, acc94                 // acc[94] a2
    v_accvgpr_read_b32 v30, acc95                 // acc[95] a3
    v_cvt_pk_bf16_f32 v29, v29, v30               // pk cvt a2,a3
    buffer_store_dwordx2 v[28:29], v26, s[68:71], s15, offen offset:64 nt// store D m2_n7 GWVW=4
    v_accvgpr_read_b32 v29, acc124                // acc[124] a0
    v_accvgpr_read_b32 v30, acc125                // acc[125] a1
    v_cvt_pk_bf16_f32 v28, v29, v30               // pk cvt a0,a1
    v_accvgpr_read_b32 v29, acc126                // acc[126] a2
    v_accvgpr_read_b32 v30, acc127                // acc[127] a3
    v_cvt_pk_bf16_f32 v29, v29, v30               // pk cvt a2,a3
    buffer_store_dwordx2 v[28:29], v26, s[68:71], s15, offen offset:96 nt// store D m3_n7 GWVW=4
    v_accvgpr_read_b32 v29, acc156                // acc[156] a0
    v_accvgpr_read_b32 v30, acc157                // acc[157] a1
    v_cvt_pk_bf16_f32 v28, v29, v30               // pk cvt a0,a1
    v_accvgpr_read_b32 v29, acc158                // acc[158] a2
    v_accvgpr_read_b32 v30, acc159                // acc[159] a3
    v_cvt_pk_bf16_f32 v29, v29, v30               // pk cvt a2,a3
    buffer_store_dwordx2 v[28:29], v26, s[68:71], s15, offen offset:128 nt// store D m4_n7 GWVW=4
    v_accvgpr_read_b32 v29, acc188                // acc[188] a0
    v_accvgpr_read_b32 v30, acc189                // acc[189] a1
    v_cvt_pk_bf16_f32 v28, v29, v30               // pk cvt a0,a1
    v_accvgpr_read_b32 v29, acc190                // acc[190] a2
    v_accvgpr_read_b32 v30, acc191                // acc[191] a3
    v_cvt_pk_bf16_f32 v29, v29, v30               // pk cvt a2,a3
    buffer_store_dwordx2 v[28:29], v26, s[68:71], s15, offen offset:160 nt// store D m5_n7 GWVW=4
    v_accvgpr_read_b32 v29, acc220                // acc[220] a0
    v_accvgpr_read_b32 v30, acc221                // acc[221] a1
    v_cvt_pk_bf16_f32 v28, v29, v30               // pk cvt a0,a1
    v_accvgpr_read_b32 v29, acc222                // acc[222] a2
    v_accvgpr_read_b32 v30, acc223                // acc[223] a3
    v_cvt_pk_bf16_f32 v29, v29, v30               // pk cvt a2,a3
    buffer_store_dwordx2 v[28:29], v26, s[68:71], s15, offen offset:192 nt// store D m6_n7 GWVW=4
    v_accvgpr_read_b32 v29, acc252                // acc[252] a0
    v_accvgpr_read_b32 v30, acc253                // acc[253] a1
    v_cvt_pk_bf16_f32 v28, v29, v30               // pk cvt a0,a1
    v_accvgpr_read_b32 v29, acc254                // acc[254] a2
    v_accvgpr_read_b32 v30, acc255                // acc[255] a3
    v_cvt_pk_bf16_f32 v29, v29, v30               // pk cvt a2,a3
    buffer_store_dwordx2 v[28:29], v26, s[68:71], s15, offen offset:224 nt// store D m7_n7 GWVW=4
    s_waitcnt vmcnt(0)                            // wait for stores

    s_endpgm                                      // end of kernel

.rodata
.p2align 6
.amdhsa_kernel Custom_Cijk_Alik_Bljk_F4BS_MXA32_MXB32_MT256x256x256_MI16x16x1_kgen_gfx950
    .amdhsa_group_segment_fixed_size 147456
    .amdhsa_private_segment_fixed_size 0
    .amdhsa_kernarg_size 136
    .amdhsa_user_sgpr_kernarg_segment_ptr 1
    .amdhsa_system_sgpr_workgroup_id_x 1
    .amdhsa_system_sgpr_workgroup_id_y 1
    .amdhsa_system_vgpr_workitem_id 0
    .amdhsa_next_free_vgpr 424
    .amdhsa_next_free_sgpr 72
    .amdhsa_accum_offset 168
    .amdhsa_float_denorm_mode_32 3
    .amdhsa_float_denorm_mode_16_64 3
.end_amdhsa_kernel

.amdgpu_metadata
---
custom.config:
  InternalSupportParams:
    KernArgsVersion: 2
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
  - 32
  - 1
  - 1
  - 4
  MIInputPerThread: 32
  MIInputPerThreadA: 32
  MIInputPerThreadB: 32
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
  - 8
  - 8
  DepthU: 256
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
  NoReject: true
amdhsa.version: [ 1, 2 ]
amdhsa.kernels:
  - .name:            Custom_Cijk_Alik_Bljk_F4BS_MXA32_MXB32_MT256x256x256_MI16x16x1_kgen_gfx950
    .symbol:          Custom_Cijk_Alik_Bljk_F4BS_MXA32_MXB32_MT256x256x256_MI16x16x1_kgen_gfx950.kd
    .sgpr_count:      72
    .vgpr_count:      424
    .agpr_count:      256
    .kernarg_segment_size: 136
    .kernarg_segment_align: 8
    .group_segment_fixed_size: 147456
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
