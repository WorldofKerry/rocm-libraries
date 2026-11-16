	.amdgcn_target "amdgcn-amd-amdhsa--gfx950"
	.text
	.globl	gfxjit_gemm_mx_e2m3_e2m3_e8m0b32_f32acc_bf16_rcr_256x256x128_w2x2_16x16x128_s3_4096x4096x65536_c0c1
	.p2align	8
	.type	gfxjit_gemm_mx_e2m3_e2m3_e8m0b32_f32acc_bf16_rcr_256x256x128_w2x2_16x16x128_s3_4096x4096x65536_c0c1,@function
gfxjit_gemm_mx_e2m3_e2m3_e8m0b32_f32acc_bf16_rcr_256x256x128_w2x2_16x16x128_s3_4096x4096x65536_c0c1:
	s_load_dwordx2 s[8:9], s[0:1], 0x0
	s_load_dwordx2 s[10:11], s[0:1], 0x8
	s_load_dwordx2 s[12:13], s[0:1], 0x10
	s_load_dwordx2 s[14:15], s[0:1], 0x18
	s_load_dwordx2 s[16:17], s[0:1], 0x20
	v_and_b32_e32 v1, 63, v0
	v_lshrrev_b32_e32 v2, 6, v0
	s_nop 4
	v_readfirstlane_b32 s20, v2
	s_nop 4
	s_and_b32 s21, s20, 1
	s_lshr_b32 s22, s20, 1
	s_lshr_b32 s24, s2, 5
	s_and_b32 s24, s24, 7
	s_and_b32 s25, s2, 1
	s_lshl_b32 s25, s25, 3
	s_or_b32 s24, s24, s25
	s_lshr_b32 s23, s2, 3
	s_and_b32 s23, s23, 3
	s_lshr_b32 s25, s2, 1
	s_and_b32 s25, s25, 3
	s_lshl_b32 s25, s25, 2
	s_or_b32 s23, s23, s25
	s_waitcnt lgkmcnt(0)
	s_mul_i32 s30, s23, 0xc00000
	s_add_u32 s36, s10, s30
	s_addc_u32 s37, s11, 0
	s_mov_b32 s38, 0x6000
	s_mov_b32 s39, 0x20000
	s_mul_i32 s30, s24, 0xc00000
	s_add_u32 s40, s12, s30
	s_addc_u32 s41, s13, 0
	s_mov_b32 s42, 0x6000
	s_mov_b32 s43, 0x20000
	s_mul_i32 s30, s23, 0x400
	s_add_u32 s44, s14, s30
	s_addc_u32 s45, s15, 0
	s_mov_b32 s46, 0x400
	s_mov_b32 s47, 0x20000
	s_mul_i32 s30, s24, 0x400
	s_add_u32 s48, s16, s30
	s_addc_u32 s49, s17, 0
	s_mov_b32 s50, 0x400
	s_mov_b32 s51, 0x20000
	s_mul_i32 s32, s20, 0x1800
	s_mul_i32 s33, s20, 0x100
	v_lshlrev_b32_e32 v3, 4, v1
	v_add_u32_e32 v3, s32, v3
	v_add_u32_e32 v8, 0xc00, v3
	v_lshlrev_b32_e32 v4, 2, v1
	v_add_u32_e32 v4, s33, v4
	v_lshlrev_b32_e32 v5, 4, v1
	s_mul_i32 s34, s21, 0x2000
	s_mul_i32 s35, s22, 0x2000
	v_lshlrev_b32_e32 v9, 3, v1
	s_mul_i32 s58, s21, 0x1000
	s_mul_i32 s59, s22, 0x1000
	v_lshlrev_b32_e32 v6, 3, v1
	s_mul_i32 s56, s21, 0x200
	s_mul_i32 s57, s22, 0x200
	v_add_u32_e32 v24, s34, v5
	v_add_u32_e32 v24, 0, v24
	v_add_u32_e32 v27, s35, v5
	v_add_u32_e32 v27, 0x6000, v27
	v_add_u32_e32 v30, s58, v9
	v_add_u32_e32 v30, 0x4000, v30
	v_add_u32_e32 v33, s59, v9
	v_add_u32_e32 v33, 0xa000, v33
	v_add_u32_e32 v36, s56, v6
	v_add_u32_e32 v36, 0xc000, v36
	v_add_u32_e32 v39, s57, v6
	v_add_u32_e32 v39, 0xc400, v39
	v_add_u32_e32 v25, s34, v5
	v_add_u32_e32 v25, 0xc800, v25
	v_add_u32_e32 v28, s35, v5
	v_add_u32_e32 v28, 0x12800, v28
	v_add_u32_e32 v31, s58, v9
	v_add_u32_e32 v31, 0x10800, v31
	v_add_u32_e32 v34, s59, v9
	v_add_u32_e32 v34, 0x16800, v34
	v_add_u32_e32 v37, s56, v6
	v_add_u32_e32 v37, 0x18800, v37
	v_add_u32_e32 v40, s57, v6
	v_add_u32_e32 v40, 0x18c00, v40
	v_add_u32_e32 v26, s34, v5
	v_add_u32_e32 v26, 0x19000, v26
	v_add_u32_e32 v29, s35, v5
	v_add_u32_e32 v29, 0x1f000, v29
	v_add_u32_e32 v32, s58, v9
	v_add_u32_e32 v32, 0x1d000, v32
	v_add_u32_e32 v35, s59, v9
	v_add_u32_e32 v35, 0x23000, v35
	v_add_u32_e32 v38, s56, v6
	v_add_u32_e32 v38, 0x25000, v38
	v_add_u32_e32 v41, s57, v6
	v_add_u32_e32 v41, 0x25400, v41
	v_accvgpr_write_b32 a0, 0
	v_accvgpr_write_b32 a1, 0
	v_accvgpr_write_b32 a2, 0
	v_accvgpr_write_b32 a3, 0
	v_accvgpr_write_b32 a4, 0
	v_accvgpr_write_b32 a5, 0
	v_accvgpr_write_b32 a6, 0
	v_accvgpr_write_b32 a7, 0
	v_accvgpr_write_b32 a8, 0
	v_accvgpr_write_b32 a9, 0
	v_accvgpr_write_b32 a10, 0
	v_accvgpr_write_b32 a11, 0
	v_accvgpr_write_b32 a12, 0
	v_accvgpr_write_b32 a13, 0
	v_accvgpr_write_b32 a14, 0
	v_accvgpr_write_b32 a15, 0
	v_accvgpr_write_b32 a16, 0
	v_accvgpr_write_b32 a17, 0
	v_accvgpr_write_b32 a18, 0
	v_accvgpr_write_b32 a19, 0
	v_accvgpr_write_b32 a20, 0
	v_accvgpr_write_b32 a21, 0
	v_accvgpr_write_b32 a22, 0
	v_accvgpr_write_b32 a23, 0
	v_accvgpr_write_b32 a24, 0
	v_accvgpr_write_b32 a25, 0
	v_accvgpr_write_b32 a26, 0
	v_accvgpr_write_b32 a27, 0
	v_accvgpr_write_b32 a28, 0
	v_accvgpr_write_b32 a29, 0
	v_accvgpr_write_b32 a30, 0
	v_accvgpr_write_b32 a31, 0
	v_accvgpr_write_b32 a32, 0
	v_accvgpr_write_b32 a33, 0
	v_accvgpr_write_b32 a34, 0
	v_accvgpr_write_b32 a35, 0
	v_accvgpr_write_b32 a36, 0
	v_accvgpr_write_b32 a37, 0
	v_accvgpr_write_b32 a38, 0
	v_accvgpr_write_b32 a39, 0
	v_accvgpr_write_b32 a40, 0
	v_accvgpr_write_b32 a41, 0
	v_accvgpr_write_b32 a42, 0
	v_accvgpr_write_b32 a43, 0
	v_accvgpr_write_b32 a44, 0
	v_accvgpr_write_b32 a45, 0
	v_accvgpr_write_b32 a46, 0
	v_accvgpr_write_b32 a47, 0
	v_accvgpr_write_b32 a48, 0
	v_accvgpr_write_b32 a49, 0
	v_accvgpr_write_b32 a50, 0
	v_accvgpr_write_b32 a51, 0
	v_accvgpr_write_b32 a52, 0
	v_accvgpr_write_b32 a53, 0
	v_accvgpr_write_b32 a54, 0
	v_accvgpr_write_b32 a55, 0
	v_accvgpr_write_b32 a56, 0
	v_accvgpr_write_b32 a57, 0
	v_accvgpr_write_b32 a58, 0
	v_accvgpr_write_b32 a59, 0
	v_accvgpr_write_b32 a60, 0
	v_accvgpr_write_b32 a61, 0
	v_accvgpr_write_b32 a62, 0
	v_accvgpr_write_b32 a63, 0
	v_accvgpr_write_b32 a64, 0
	v_accvgpr_write_b32 a65, 0
	v_accvgpr_write_b32 a66, 0
	v_accvgpr_write_b32 a67, 0
	v_accvgpr_write_b32 a68, 0
	v_accvgpr_write_b32 a69, 0
	v_accvgpr_write_b32 a70, 0
	v_accvgpr_write_b32 a71, 0
	v_accvgpr_write_b32 a72, 0
	v_accvgpr_write_b32 a73, 0
	v_accvgpr_write_b32 a74, 0
	v_accvgpr_write_b32 a75, 0
	v_accvgpr_write_b32 a76, 0
	v_accvgpr_write_b32 a77, 0
	v_accvgpr_write_b32 a78, 0
	v_accvgpr_write_b32 a79, 0
	v_accvgpr_write_b32 a80, 0
	v_accvgpr_write_b32 a81, 0
	v_accvgpr_write_b32 a82, 0
	v_accvgpr_write_b32 a83, 0
	v_accvgpr_write_b32 a84, 0
	v_accvgpr_write_b32 a85, 0
	v_accvgpr_write_b32 a86, 0
	v_accvgpr_write_b32 a87, 0
	v_accvgpr_write_b32 a88, 0
	v_accvgpr_write_b32 a89, 0
	v_accvgpr_write_b32 a90, 0
	v_accvgpr_write_b32 a91, 0
	v_accvgpr_write_b32 a92, 0
	v_accvgpr_write_b32 a93, 0
	v_accvgpr_write_b32 a94, 0
	v_accvgpr_write_b32 a95, 0
	v_accvgpr_write_b32 a96, 0
	v_accvgpr_write_b32 a97, 0
	v_accvgpr_write_b32 a98, 0
	v_accvgpr_write_b32 a99, 0
	v_accvgpr_write_b32 a100, 0
	v_accvgpr_write_b32 a101, 0
	v_accvgpr_write_b32 a102, 0
	v_accvgpr_write_b32 a103, 0
	v_accvgpr_write_b32 a104, 0
	v_accvgpr_write_b32 a105, 0
	v_accvgpr_write_b32 a106, 0
	v_accvgpr_write_b32 a107, 0
	v_accvgpr_write_b32 a108, 0
	v_accvgpr_write_b32 a109, 0
	v_accvgpr_write_b32 a110, 0
	v_accvgpr_write_b32 a111, 0
	v_accvgpr_write_b32 a112, 0
	v_accvgpr_write_b32 a113, 0
	v_accvgpr_write_b32 a114, 0
	v_accvgpr_write_b32 a115, 0
	v_accvgpr_write_b32 a116, 0
	v_accvgpr_write_b32 a117, 0
	v_accvgpr_write_b32 a118, 0
	v_accvgpr_write_b32 a119, 0
	v_accvgpr_write_b32 a120, 0
	v_accvgpr_write_b32 a121, 0
	v_accvgpr_write_b32 a122, 0
	v_accvgpr_write_b32 a123, 0
	v_accvgpr_write_b32 a124, 0
	v_accvgpr_write_b32 a125, 0
	v_accvgpr_write_b32 a126, 0
	v_accvgpr_write_b32 a127, 0
	v_accvgpr_write_b32 a128, 0
	v_accvgpr_write_b32 a129, 0
	v_accvgpr_write_b32 a130, 0
	v_accvgpr_write_b32 a131, 0
	v_accvgpr_write_b32 a132, 0
	v_accvgpr_write_b32 a133, 0
	v_accvgpr_write_b32 a134, 0
	v_accvgpr_write_b32 a135, 0
	v_accvgpr_write_b32 a136, 0
	v_accvgpr_write_b32 a137, 0
	v_accvgpr_write_b32 a138, 0
	v_accvgpr_write_b32 a139, 0
	v_accvgpr_write_b32 a140, 0
	v_accvgpr_write_b32 a141, 0
	v_accvgpr_write_b32 a142, 0
	v_accvgpr_write_b32 a143, 0
	v_accvgpr_write_b32 a144, 0
	v_accvgpr_write_b32 a145, 0
	v_accvgpr_write_b32 a146, 0
	v_accvgpr_write_b32 a147, 0
	v_accvgpr_write_b32 a148, 0
	v_accvgpr_write_b32 a149, 0
	v_accvgpr_write_b32 a150, 0
	v_accvgpr_write_b32 a151, 0
	v_accvgpr_write_b32 a152, 0
	v_accvgpr_write_b32 a153, 0
	v_accvgpr_write_b32 a154, 0
	v_accvgpr_write_b32 a155, 0
	v_accvgpr_write_b32 a156, 0
	v_accvgpr_write_b32 a157, 0
	v_accvgpr_write_b32 a158, 0
	v_accvgpr_write_b32 a159, 0
	v_accvgpr_write_b32 a160, 0
	v_accvgpr_write_b32 a161, 0
	v_accvgpr_write_b32 a162, 0
	v_accvgpr_write_b32 a163, 0
	v_accvgpr_write_b32 a164, 0
	v_accvgpr_write_b32 a165, 0
	v_accvgpr_write_b32 a166, 0
	v_accvgpr_write_b32 a167, 0
	v_accvgpr_write_b32 a168, 0
	v_accvgpr_write_b32 a169, 0
	v_accvgpr_write_b32 a170, 0
	v_accvgpr_write_b32 a171, 0
	v_accvgpr_write_b32 a172, 0
	v_accvgpr_write_b32 a173, 0
	v_accvgpr_write_b32 a174, 0
	v_accvgpr_write_b32 a175, 0
	v_accvgpr_write_b32 a176, 0
	v_accvgpr_write_b32 a177, 0
	v_accvgpr_write_b32 a178, 0
	v_accvgpr_write_b32 a179, 0
	v_accvgpr_write_b32 a180, 0
	v_accvgpr_write_b32 a181, 0
	v_accvgpr_write_b32 a182, 0
	v_accvgpr_write_b32 a183, 0
	v_accvgpr_write_b32 a184, 0
	v_accvgpr_write_b32 a185, 0
	v_accvgpr_write_b32 a186, 0
	v_accvgpr_write_b32 a187, 0
	v_accvgpr_write_b32 a188, 0
	v_accvgpr_write_b32 a189, 0
	v_accvgpr_write_b32 a190, 0
	v_accvgpr_write_b32 a191, 0
	v_accvgpr_write_b32 a192, 0
	v_accvgpr_write_b32 a193, 0
	v_accvgpr_write_b32 a194, 0
	v_accvgpr_write_b32 a195, 0
	v_accvgpr_write_b32 a196, 0
	v_accvgpr_write_b32 a197, 0
	v_accvgpr_write_b32 a198, 0
	v_accvgpr_write_b32 a199, 0
	v_accvgpr_write_b32 a200, 0
	v_accvgpr_write_b32 a201, 0
	v_accvgpr_write_b32 a202, 0
	v_accvgpr_write_b32 a203, 0
	v_accvgpr_write_b32 a204, 0
	v_accvgpr_write_b32 a205, 0
	v_accvgpr_write_b32 a206, 0
	v_accvgpr_write_b32 a207, 0
	v_accvgpr_write_b32 a208, 0
	v_accvgpr_write_b32 a209, 0
	v_accvgpr_write_b32 a210, 0
	v_accvgpr_write_b32 a211, 0
	v_accvgpr_write_b32 a212, 0
	v_accvgpr_write_b32 a213, 0
	v_accvgpr_write_b32 a214, 0
	v_accvgpr_write_b32 a215, 0
	v_accvgpr_write_b32 a216, 0
	v_accvgpr_write_b32 a217, 0
	v_accvgpr_write_b32 a218, 0
	v_accvgpr_write_b32 a219, 0
	v_accvgpr_write_b32 a220, 0
	v_accvgpr_write_b32 a221, 0
	v_accvgpr_write_b32 a222, 0
	v_accvgpr_write_b32 a223, 0
	v_accvgpr_write_b32 a224, 0
	v_accvgpr_write_b32 a225, 0
	v_accvgpr_write_b32 a226, 0
	v_accvgpr_write_b32 a227, 0
	v_accvgpr_write_b32 a228, 0
	v_accvgpr_write_b32 a229, 0
	v_accvgpr_write_b32 a230, 0
	v_accvgpr_write_b32 a231, 0
	v_accvgpr_write_b32 a232, 0
	v_accvgpr_write_b32 a233, 0
	v_accvgpr_write_b32 a234, 0
	v_accvgpr_write_b32 a235, 0
	v_accvgpr_write_b32 a236, 0
	v_accvgpr_write_b32 a237, 0
	v_accvgpr_write_b32 a238, 0
	v_accvgpr_write_b32 a239, 0
	v_accvgpr_write_b32 a240, 0
	v_accvgpr_write_b32 a241, 0
	v_accvgpr_write_b32 a242, 0
	v_accvgpr_write_b32 a243, 0
	v_accvgpr_write_b32 a244, 0
	v_accvgpr_write_b32 a245, 0
	v_accvgpr_write_b32 a246, 0
	v_accvgpr_write_b32 a247, 0
	v_accvgpr_write_b32 a248, 0
	v_accvgpr_write_b32 a249, 0
	v_accvgpr_write_b32 a250, 0
	v_accvgpr_write_b32 a251, 0
	v_accvgpr_write_b32 a252, 0
	v_accvgpr_write_b32 a253, 0
	v_accvgpr_write_b32 a254, 0
	v_accvgpr_write_b32 a255, 0
	s_add_u32 m0, s32, 0
	buffer_load_dwordx4 v3, s[36:39], 0 offen lds
	buffer_load_dwordx4 v3, s[36:39], 0 offen offset:1024 lds
	buffer_load_dwordx4 v3, s[36:39], 0 offen offset:2048 lds
	s_add_u32 m0, s32, 0xc00
	buffer_load_dwordx4 v8, s[36:39], 0 offen lds
	buffer_load_dwordx4 v8, s[36:39], 0 offen offset:1024 lds
	buffer_load_dwordx4 v8, s[36:39], 0 offen offset:2048 lds
	s_add_u32 s36, s36, 0x6000
	s_addc_u32 s37, s37, 0
	s_add_u32 m0, s32, 0x6000
	buffer_load_dwordx4 v3, s[40:43], 0 offen lds
	buffer_load_dwordx4 v3, s[40:43], 0 offen offset:1024 lds
	buffer_load_dwordx4 v3, s[40:43], 0 offen offset:2048 lds
	s_add_u32 m0, s32, 0x6c00
	buffer_load_dwordx4 v8, s[40:43], 0 offen lds
	buffer_load_dwordx4 v8, s[40:43], 0 offen offset:1024 lds
	buffer_load_dwordx4 v8, s[40:43], 0 offen offset:2048 lds
	s_add_u32 s40, s40, 0x6000
	s_addc_u32 s41, s41, 0
	s_add_u32 m0, s33, 0xc000
	buffer_load_dword v4, s[44:47], 0 offen lds
	s_add_u32 s44, s44, 0x4000
	s_addc_u32 s45, s45, 0
	s_add_u32 m0, s33, 0xc400
	buffer_load_dword v4, s[48:51], 0 offen lds
	s_add_u32 s48, s48, 0x4000
	s_addc_u32 s49, s49, 0
	s_add_u32 m0, s32, 0xc800
	buffer_load_dwordx4 v3, s[36:39], 0 offen lds
	buffer_load_dwordx4 v3, s[36:39], 0 offen offset:1024 lds
	buffer_load_dwordx4 v3, s[36:39], 0 offen offset:2048 lds
	s_add_u32 m0, s32, 0xd400
	buffer_load_dwordx4 v8, s[36:39], 0 offen lds
	buffer_load_dwordx4 v8, s[36:39], 0 offen offset:1024 lds
	buffer_load_dwordx4 v8, s[36:39], 0 offen offset:2048 lds
	s_add_u32 s36, s36, 0x6000
	s_addc_u32 s37, s37, 0
	s_add_u32 m0, s32, 0x12800
	buffer_load_dwordx4 v3, s[40:43], 0 offen lds
	buffer_load_dwordx4 v3, s[40:43], 0 offen offset:1024 lds
	buffer_load_dwordx4 v3, s[40:43], 0 offen offset:2048 lds
	s_add_u32 m0, s32, 0x13400
	buffer_load_dwordx4 v8, s[40:43], 0 offen lds
	buffer_load_dwordx4 v8, s[40:43], 0 offen offset:1024 lds
	buffer_load_dwordx4 v8, s[40:43], 0 offen offset:2048 lds
	s_add_u32 s40, s40, 0x6000
	s_addc_u32 s41, s41, 0
	s_add_u32 m0, s33, 0x18800
	buffer_load_dword v4, s[44:47], 0 offen lds
	s_add_u32 s44, s44, 0x4000
	s_addc_u32 s45, s45, 0
	s_add_u32 m0, s33, 0x18c00
	buffer_load_dword v4, s[48:51], 0 offen lds
	s_add_u32 s48, s48, 0x4000
	s_addc_u32 s49, s49, 0
	s_add_u32 m0, s32, 0x19000
	buffer_load_dwordx4 v3, s[36:39], 0 offen lds
	buffer_load_dwordx4 v3, s[36:39], 0 offen offset:1024 lds
	buffer_load_dwordx4 v3, s[36:39], 0 offen offset:2048 lds
	s_add_u32 m0, s32, 0x19c00
	buffer_load_dwordx4 v8, s[36:39], 0 offen lds
	buffer_load_dwordx4 v8, s[36:39], 0 offen offset:1024 lds
	buffer_load_dwordx4 v8, s[36:39], 0 offen offset:2048 lds
	s_add_u32 s36, s36, 0x6000
	s_addc_u32 s37, s37, 0
	s_add_u32 m0, s32, 0x1f000
	buffer_load_dwordx4 v3, s[40:43], 0 offen lds
	buffer_load_dwordx4 v3, s[40:43], 0 offen offset:1024 lds
	buffer_load_dwordx4 v3, s[40:43], 0 offen offset:2048 lds
	s_add_u32 m0, s32, 0x1fc00
	buffer_load_dwordx4 v8, s[40:43], 0 offen lds
	buffer_load_dwordx4 v8, s[40:43], 0 offen offset:1024 lds
	buffer_load_dwordx4 v8, s[40:43], 0 offen offset:2048 lds
	s_add_u32 s40, s40, 0x6000
	s_addc_u32 s41, s41, 0
	s_add_u32 m0, s33, 0x25000
	buffer_load_dword v4, s[44:47], 0 offen lds
	s_add_u32 s44, s44, 0x4000
	s_addc_u32 s45, s45, 0
	s_add_u32 m0, s33, 0x25400
	buffer_load_dword v4, s[48:51], 0 offen lds
	s_add_u32 s48, s48, 0x4000
	s_addc_u32 s49, s49, 0
	s_waitcnt vmcnt(28)
	s_barrier
	ds_read_b64 v[16:17], v36
	ds_read_b64 v[18:19], v39
	ds_read_b128 v[48:51], v24
	ds_read_b64 v[52:53], v30
	ds_read_b128 v[96:99], v27
	ds_read_b64 v[100:101], v33
	ds_read_b128 v[54:57], v24 offset:1024
	ds_read_b64 v[58:59], v30 offset:512
	ds_read_b128 v[102:105], v27 offset:1024
	ds_read_b64 v[106:107], v33 offset:512
	ds_read_b128 v[60:63], v24 offset:2048
	ds_read_b64 v[64:65], v30 offset:1024
	ds_read_b128 v[108:111], v27 offset:2048
	ds_read_b64 v[112:113], v33 offset:1024
	ds_read_b128 v[66:69], v24 offset:3072
	ds_read_b64 v[70:71], v30 offset:1536
	ds_read_b128 v[114:117], v27 offset:3072
	ds_read_b64 v[118:119], v33 offset:1536
	ds_read_b128 v[72:75], v24 offset:4096
	ds_read_b64 v[76:77], v30 offset:2048
	ds_read_b128 v[120:123], v27 offset:4096
	ds_read_b64 v[124:125], v33 offset:2048
	ds_read_b128 v[78:81], v24 offset:5120
	ds_read_b64 v[82:83], v30 offset:2560
	ds_read_b128 v[126:129], v27 offset:5120
	ds_read_b64 v[130:131], v33 offset:2560
	ds_read_b128 v[84:87], v24 offset:6144
	ds_read_b64 v[88:89], v30 offset:3072
	ds_read_b128 v[132:135], v27 offset:6144
	ds_read_b64 v[136:137], v33 offset:3072
	ds_read_b128 v[90:93], v24 offset:7168
	ds_read_b64 v[94:95], v30 offset:3584
	ds_read_b128 v[138:141], v27 offset:7168
	ds_read_b64 v[142:143], v33 offset:3584
	s_mov_b32 s31, 0x55
	s_waitcnt vmcnt(14)
	s_waitcnt lgkmcnt(0)
	s_barrier
	v_mfma_scale_f32_16x16x128_f8f6f4 a[0:3], v[48:53], v[96:101], a[0:3], v16, v18 op_sel_hi:[0,0,0] cbsz:2 blgp:2
	ds_read_b64 v[20:21], v37
	v_mfma_scale_f32_16x16x128_f8f6f4 a[4:7], v[48:53], v[102:107], a[4:7], v16, v18 op_sel:[0,1,0] op_sel_hi:[0,0,0] cbsz:2 blgp:2
	s_add_u32 m0, s32, 0
	buffer_load_dwordx4 v3, s[36:39], 0 offen lds
	v_mfma_scale_f32_16x16x128_f8f6f4 a[8:11], v[48:53], v[108:113], a[8:11], v16, v18 op_sel_hi:[0,1,0] cbsz:2 blgp:2
	ds_read_b64 v[22:23], v40
	v_mfma_scale_f32_16x16x128_f8f6f4 a[12:15], v[48:53], v[114:119], a[12:15], v16, v18 op_sel:[0,1,0] op_sel_hi:[0,1,0] cbsz:2 blgp:2
	buffer_load_dwordx4 v3, s[36:39], 0 offen offset:1024 lds
	v_mfma_scale_f32_16x16x128_f8f6f4 a[16:19], v[48:53], v[120:125], a[16:19], v16, v19 op_sel_hi:[0,0,0] cbsz:2 blgp:2
	ds_read_b128 v[144:147], v25
	v_mfma_scale_f32_16x16x128_f8f6f4 a[20:23], v[48:53], v[126:131], a[20:23], v16, v19 op_sel:[0,1,0] op_sel_hi:[0,0,0] cbsz:2 blgp:2
	v_mfma_scale_f32_16x16x128_f8f6f4 a[24:27], v[48:53], v[132:137], a[24:27], v16, v19 op_sel_hi:[0,1,0] cbsz:2 blgp:2
	ds_read_b64 v[148:149], v31
	v_mfma_scale_f32_16x16x128_f8f6f4 a[28:31], v[48:53], v[138:143], a[28:31], v16, v19 op_sel:[0,1,0] op_sel_hi:[0,1,0] cbsz:2 blgp:2
	buffer_load_dwordx4 v3, s[36:39], 0 offen offset:2048 lds
	v_mfma_scale_f32_16x16x128_f8f6f4 a[60:63], v[54:59], v[138:143], a[60:63], v16, v19 op_sel:[1,1,0] op_sel_hi:[0,1,0] cbsz:2 blgp:2
	ds_read_b128 v[192:195], v28
	v_mfma_scale_f32_16x16x128_f8f6f4 a[56:59], v[54:59], v[132:137], a[56:59], v16, v19 op_sel:[1,0,0] op_sel_hi:[0,1,0] cbsz:2 blgp:2
	ds_read_b64 v[196:197], v34
	v_mfma_scale_f32_16x16x128_f8f6f4 a[52:55], v[54:59], v[126:131], a[52:55], v16, v19 op_sel:[1,1,0] op_sel_hi:[0,0,0] cbsz:2 blgp:2
	v_mfma_scale_f32_16x16x128_f8f6f4 a[48:51], v[54:59], v[120:125], a[48:51], v16, v19 op_sel:[1,0,0] op_sel_hi:[0,0,0] cbsz:2 blgp:2
	s_add_u32 m0, s32, 0xc00
	buffer_load_dwordx4 v8, s[36:39], 0 offen lds
	v_mfma_scale_f32_16x16x128_f8f6f4 a[44:47], v[54:59], v[114:119], a[44:47], v16, v18 op_sel:[1,1,0] op_sel_hi:[0,1,0] cbsz:2 blgp:2
	ds_read_b128 v[150:153], v25 offset:1024
	v_mfma_scale_f32_16x16x128_f8f6f4 a[40:43], v[54:59], v[108:113], a[40:43], v16, v18 op_sel:[1,0,0] op_sel_hi:[0,1,0] cbsz:2 blgp:2
	ds_read_b64 v[154:155], v31 offset:512
	v_mfma_scale_f32_16x16x128_f8f6f4 a[36:39], v[54:59], v[102:107], a[36:39], v16, v18 op_sel:[1,1,0] op_sel_hi:[0,0,0] cbsz:2 blgp:2
	buffer_load_dwordx4 v8, s[36:39], 0 offen offset:1024 lds
	v_mfma_scale_f32_16x16x128_f8f6f4 a[32:35], v[54:59], v[96:101], a[32:35], v16, v18 op_sel:[1,0,0] op_sel_hi:[0,0,0] cbsz:2 blgp:2
	v_mfma_scale_f32_16x16x128_f8f6f4 a[64:67], v[60:65], v[96:101], a[64:67], v16, v18 op_sel_hi:[1,0,0] cbsz:2 blgp:2
	ds_read_b128 v[198:201], v28 offset:1024
	v_mfma_scale_f32_16x16x128_f8f6f4 a[68:71], v[60:65], v[102:107], a[68:71], v16, v18 op_sel:[0,1,0] op_sel_hi:[1,0,0] cbsz:2 blgp:2
	ds_read_b64 v[202:203], v34 offset:512
	v_mfma_scale_f32_16x16x128_f8f6f4 a[72:75], v[60:65], v[108:113], a[72:75], v16, v18 op_sel_hi:[1,1,0] cbsz:2 blgp:2
	buffer_load_dwordx4 v8, s[36:39], 0 offen offset:2048 lds
	v_mfma_scale_f32_16x16x128_f8f6f4 a[76:79], v[60:65], v[114:119], a[76:79], v16, v18 op_sel:[0,1,0] op_sel_hi:[1,1,0] cbsz:2 blgp:2
	ds_read_b128 v[156:159], v25 offset:2048
	v_mfma_scale_f32_16x16x128_f8f6f4 a[80:83], v[60:65], v[120:125], a[80:83], v16, v19 op_sel_hi:[1,0,0] cbsz:2 blgp:2
	ds_read_b64 v[160:161], v31 offset:1024
	v_mfma_scale_f32_16x16x128_f8f6f4 a[84:87], v[60:65], v[126:131], a[84:87], v16, v19 op_sel:[0,1,0] op_sel_hi:[1,0,0] cbsz:2 blgp:2
	v_mfma_scale_f32_16x16x128_f8f6f4 a[88:91], v[60:65], v[132:137], a[88:91], v16, v19 op_sel_hi:[1,1,0] cbsz:2 blgp:2
	s_add_u32 s36, s36, 0x6000
	s_addc_u32 s37, s37, 0
	v_mfma_scale_f32_16x16x128_f8f6f4 a[92:95], v[60:65], v[138:143], a[92:95], v16, v19 op_sel:[0,1,0] op_sel_hi:[1,1,0] cbsz:2 blgp:2
	ds_read_b128 v[204:207], v28 offset:2048
	v_mfma_scale_f32_16x16x128_f8f6f4 a[124:127], v[66:71], v[138:143], a[124:127], v16, v19 op_sel:[1,1,0] op_sel_hi:[1,1,0] cbsz:2 blgp:2
	ds_read_b64 v[208:209], v34 offset:1024
	v_mfma_scale_f32_16x16x128_f8f6f4 a[120:123], v[66:71], v[132:137], a[120:123], v16, v19 op_sel:[1,0,0] op_sel_hi:[1,1,0] cbsz:2 blgp:2
	s_add_u32 m0, s32, 0x6000
	buffer_load_dwordx4 v3, s[40:43], 0 offen lds
	v_mfma_scale_f32_16x16x128_f8f6f4 a[116:119], v[66:71], v[126:131], a[116:119], v16, v19 op_sel:[1,1,0] op_sel_hi:[1,0,0] cbsz:2 blgp:2
	v_mfma_scale_f32_16x16x128_f8f6f4 a[112:115], v[66:71], v[120:125], a[112:115], v16, v19 op_sel:[1,0,0] op_sel_hi:[1,0,0] cbsz:2 blgp:2
	ds_read_b128 v[162:165], v25 offset:3072
	v_mfma_scale_f32_16x16x128_f8f6f4 a[108:111], v[66:71], v[114:119], a[108:111], v16, v18 op_sel:[1,1,0] op_sel_hi:[1,1,0] cbsz:2 blgp:2
	ds_read_b64 v[166:167], v31 offset:1536
	v_mfma_scale_f32_16x16x128_f8f6f4 a[104:107], v[66:71], v[108:113], a[104:107], v16, v18 op_sel:[1,0,0] op_sel_hi:[1,1,0] cbsz:2 blgp:2
	buffer_load_dwordx4 v3, s[40:43], 0 offen offset:1024 lds
	v_mfma_scale_f32_16x16x128_f8f6f4 a[100:103], v[66:71], v[102:107], a[100:103], v16, v18 op_sel:[1,1,0] op_sel_hi:[1,0,0] cbsz:2 blgp:2
	ds_read_b128 v[210:213], v28 offset:3072
	v_mfma_scale_f32_16x16x128_f8f6f4 a[96:99], v[66:71], v[96:101], a[96:99], v16, v18 op_sel:[1,0,0] op_sel_hi:[1,0,0] cbsz:2 blgp:2
	v_mfma_scale_f32_16x16x128_f8f6f4 a[128:131], v[72:77], v[96:101], a[128:131], v17, v18 op_sel_hi:[0,0,0] cbsz:2 blgp:2
	ds_read_b64 v[214:215], v34 offset:1536
	v_mfma_scale_f32_16x16x128_f8f6f4 a[132:135], v[72:77], v[102:107], a[132:135], v17, v18 op_sel:[0,1,0] op_sel_hi:[0,0,0] cbsz:2 blgp:2
	buffer_load_dwordx4 v3, s[40:43], 0 offen offset:2048 lds
	v_mfma_scale_f32_16x16x128_f8f6f4 a[136:139], v[72:77], v[108:113], a[136:139], v17, v18 op_sel_hi:[0,1,0] cbsz:2 blgp:2
	ds_read_b128 v[168:171], v25 offset:4096
	v_mfma_scale_f32_16x16x128_f8f6f4 a[140:143], v[72:77], v[114:119], a[140:143], v17, v18 op_sel:[0,1,0] op_sel_hi:[0,1,0] cbsz:2 blgp:2
	s_add_u32 m0, s32, 0x6c00
	buffer_load_dwordx4 v8, s[40:43], 0 offen lds
	v_mfma_scale_f32_16x16x128_f8f6f4 a[144:147], v[72:77], v[120:125], a[144:147], v17, v19 op_sel_hi:[0,0,0] cbsz:2 blgp:2
	ds_read_b64 v[172:173], v31 offset:2048
	v_mfma_scale_f32_16x16x128_f8f6f4 a[148:151], v[72:77], v[126:131], a[148:151], v17, v19 op_sel:[0,1,0] op_sel_hi:[0,0,0] cbsz:2 blgp:2
	v_mfma_scale_f32_16x16x128_f8f6f4 a[152:155], v[72:77], v[132:137], a[152:155], v17, v19 op_sel_hi:[0,1,0] cbsz:2 blgp:2
	ds_read_b128 v[216:219], v28 offset:4096
	v_mfma_scale_f32_16x16x128_f8f6f4 a[156:159], v[72:77], v[138:143], a[156:159], v17, v19 op_sel:[0,1,0] op_sel_hi:[0,1,0] cbsz:2 blgp:2
	buffer_load_dwordx4 v8, s[40:43], 0 offen offset:1024 lds
	v_mfma_scale_f32_16x16x128_f8f6f4 a[188:191], v[78:83], v[138:143], a[188:191], v17, v19 op_sel:[1,1,0] op_sel_hi:[0,1,0] cbsz:2 blgp:2
	ds_read_b64 v[220:221], v34 offset:2048
	v_mfma_scale_f32_16x16x128_f8f6f4 a[184:187], v[78:83], v[132:137], a[184:187], v17, v19 op_sel:[1,0,0] op_sel_hi:[0,1,0] cbsz:2 blgp:2
	ds_read_b128 v[174:177], v25 offset:5120
	v_mfma_scale_f32_16x16x128_f8f6f4 a[180:183], v[78:83], v[126:131], a[180:183], v17, v19 op_sel:[1,1,0] op_sel_hi:[0,0,0] cbsz:2 blgp:2
	v_mfma_scale_f32_16x16x128_f8f6f4 a[176:179], v[78:83], v[120:125], a[176:179], v17, v19 op_sel:[1,0,0] op_sel_hi:[0,0,0] cbsz:2 blgp:2
	buffer_load_dwordx4 v8, s[40:43], 0 offen offset:2048 lds
	v_mfma_scale_f32_16x16x128_f8f6f4 a[172:175], v[78:83], v[114:119], a[172:175], v17, v18 op_sel:[1,1,0] op_sel_hi:[0,1,0] cbsz:2 blgp:2
	ds_read_b64 v[178:179], v31 offset:2560
	v_mfma_scale_f32_16x16x128_f8f6f4 a[168:171], v[78:83], v[108:113], a[168:171], v17, v18 op_sel:[1,0,0] op_sel_hi:[0,1,0] cbsz:2 blgp:2
	ds_read_b128 v[222:225], v28 offset:5120
	v_mfma_scale_f32_16x16x128_f8f6f4 a[164:167], v[78:83], v[102:107], a[164:167], v17, v18 op_sel:[1,1,0] op_sel_hi:[0,0,0] cbsz:2 blgp:2
	s_add_u32 s40, s40, 0x6000
	s_addc_u32 s41, s41, 0
	v_mfma_scale_f32_16x16x128_f8f6f4 a[160:163], v[78:83], v[96:101], a[160:163], v17, v18 op_sel:[1,0,0] op_sel_hi:[0,0,0] cbsz:2 blgp:2
	v_mfma_scale_f32_16x16x128_f8f6f4 a[192:195], v[84:89], v[96:101], a[192:195], v17, v18 op_sel_hi:[1,0,0] cbsz:2 blgp:2
	ds_read_b64 v[226:227], v34 offset:2560
	v_mfma_scale_f32_16x16x128_f8f6f4 a[196:199], v[84:89], v[102:107], a[196:199], v17, v18 op_sel:[0,1,0] op_sel_hi:[1,0,0] cbsz:2 blgp:2
	ds_read_b128 v[180:183], v25 offset:6144
	v_mfma_scale_f32_16x16x128_f8f6f4 a[200:203], v[84:89], v[108:113], a[200:203], v17, v18 op_sel_hi:[1,1,0] cbsz:2 blgp:2
	s_add_u32 m0, s33, 0xc000
	buffer_load_dword v4, s[44:47], 0 offen lds
	v_mfma_scale_f32_16x16x128_f8f6f4 a[204:207], v[84:89], v[114:119], a[204:207], v17, v18 op_sel:[0,1,0] op_sel_hi:[1,1,0] cbsz:2 blgp:2
	ds_read_b64 v[184:185], v31 offset:3072
	v_mfma_scale_f32_16x16x128_f8f6f4 a[208:211], v[84:89], v[120:125], a[208:211], v17, v19 op_sel_hi:[1,0,0] cbsz:2 blgp:2
	ds_read_b128 v[228:231], v28 offset:6144
	v_mfma_scale_f32_16x16x128_f8f6f4 a[212:215], v[84:89], v[126:131], a[212:215], v17, v19 op_sel:[0,1,0] op_sel_hi:[1,0,0] cbsz:2 blgp:2
	v_mfma_scale_f32_16x16x128_f8f6f4 a[216:219], v[84:89], v[132:137], a[216:219], v17, v19 op_sel_hi:[1,1,0] cbsz:2 blgp:2
	s_add_u32 s44, s44, 0x4000
	s_addc_u32 s45, s45, 0
	v_mfma_scale_f32_16x16x128_f8f6f4 a[220:223], v[84:89], v[138:143], a[220:223], v17, v19 op_sel:[0,1,0] op_sel_hi:[1,1,0] cbsz:2 blgp:2
	ds_read_b64 v[232:233], v34 offset:3072
	v_mfma_scale_f32_16x16x128_f8f6f4 a[252:255], v[90:95], v[138:143], a[252:255], v17, v19 op_sel:[1,1,0] op_sel_hi:[1,1,0] cbsz:2 blgp:2
	ds_read_b128 v[186:189], v25 offset:7168
	v_mfma_scale_f32_16x16x128_f8f6f4 a[248:251], v[90:95], v[132:137], a[248:251], v17, v19 op_sel:[1,0,0] op_sel_hi:[1,1,0] cbsz:2 blgp:2
	s_add_u32 m0, s33, 0xc400
	buffer_load_dword v4, s[48:51], 0 offen lds
	v_mfma_scale_f32_16x16x128_f8f6f4 a[244:247], v[90:95], v[126:131], a[244:247], v17, v19 op_sel:[1,1,0] op_sel_hi:[1,0,0] cbsz:2 blgp:2
	v_mfma_scale_f32_16x16x128_f8f6f4 a[240:243], v[90:95], v[120:125], a[240:243], v17, v19 op_sel:[1,0,0] op_sel_hi:[1,0,0] cbsz:2 blgp:2
	ds_read_b64 v[190:191], v31 offset:3584
	v_mfma_scale_f32_16x16x128_f8f6f4 a[236:239], v[90:95], v[114:119], a[236:239], v17, v18 op_sel:[1,1,0] op_sel_hi:[1,1,0] cbsz:2 blgp:2
	ds_read_b128 v[234:237], v28 offset:7168
	v_mfma_scale_f32_16x16x128_f8f6f4 a[232:235], v[90:95], v[108:113], a[232:235], v17, v18 op_sel:[1,0,0] op_sel_hi:[1,1,0] cbsz:2 blgp:2
	s_add_u32 s48, s48, 0x4000
	s_addc_u32 s49, s49, 0
	v_mfma_scale_f32_16x16x128_f8f6f4 a[228:231], v[90:95], v[102:107], a[228:231], v17, v18 op_sel:[1,1,0] op_sel_hi:[1,0,0] cbsz:2 blgp:2
	ds_read_b64 v[238:239], v34 offset:3584
	v_mfma_scale_f32_16x16x128_f8f6f4 a[224:227], v[90:95], v[96:101], a[224:227], v17, v18 op_sel:[1,0,0] op_sel_hi:[1,0,0] cbsz:2 blgp:2
	s_waitcnt vmcnt(14)
	s_waitcnt lgkmcnt(0)
	s_barrier
	v_mfma_scale_f32_16x16x128_f8f6f4 a[0:3], v[144:149], v[192:197], a[0:3], v20, v22 op_sel_hi:[0,0,0] cbsz:2 blgp:2
	ds_read_b64 v[16:17], v38
	v_mfma_scale_f32_16x16x128_f8f6f4 a[4:7], v[144:149], v[198:203], a[4:7], v20, v22 op_sel:[0,1,0] op_sel_hi:[0,0,0] cbsz:2 blgp:2
	s_add_u32 m0, s32, 0xc800
	buffer_load_dwordx4 v3, s[36:39], 0 offen lds
	v_mfma_scale_f32_16x16x128_f8f6f4 a[8:11], v[144:149], v[204:209], a[8:11], v20, v22 op_sel_hi:[0,1,0] cbsz:2 blgp:2
	ds_read_b64 v[18:19], v41
	v_mfma_scale_f32_16x16x128_f8f6f4 a[12:15], v[144:149], v[210:215], a[12:15], v20, v22 op_sel:[0,1,0] op_sel_hi:[0,1,0] cbsz:2 blgp:2
	buffer_load_dwordx4 v3, s[36:39], 0 offen offset:1024 lds
	v_mfma_scale_f32_16x16x128_f8f6f4 a[16:19], v[144:149], v[216:221], a[16:19], v20, v23 op_sel_hi:[0,0,0] cbsz:2 blgp:2
	ds_read_b128 v[48:51], v26
	v_mfma_scale_f32_16x16x128_f8f6f4 a[20:23], v[144:149], v[222:227], a[20:23], v20, v23 op_sel:[0,1,0] op_sel_hi:[0,0,0] cbsz:2 blgp:2
	v_mfma_scale_f32_16x16x128_f8f6f4 a[24:27], v[144:149], v[228:233], a[24:27], v20, v23 op_sel_hi:[0,1,0] cbsz:2 blgp:2
	ds_read_b64 v[52:53], v32
	v_mfma_scale_f32_16x16x128_f8f6f4 a[28:31], v[144:149], v[234:239], a[28:31], v20, v23 op_sel:[0,1,0] op_sel_hi:[0,1,0] cbsz:2 blgp:2
	buffer_load_dwordx4 v3, s[36:39], 0 offen offset:2048 lds
	v_mfma_scale_f32_16x16x128_f8f6f4 a[60:63], v[150:155], v[234:239], a[60:63], v20, v23 op_sel:[1,1,0] op_sel_hi:[0,1,0] cbsz:2 blgp:2
	ds_read_b128 v[96:99], v29
	v_mfma_scale_f32_16x16x128_f8f6f4 a[56:59], v[150:155], v[228:233], a[56:59], v20, v23 op_sel:[1,0,0] op_sel_hi:[0,1,0] cbsz:2 blgp:2
	ds_read_b64 v[100:101], v35
	v_mfma_scale_f32_16x16x128_f8f6f4 a[52:55], v[150:155], v[222:227], a[52:55], v20, v23 op_sel:[1,1,0] op_sel_hi:[0,0,0] cbsz:2 blgp:2
	v_mfma_scale_f32_16x16x128_f8f6f4 a[48:51], v[150:155], v[216:221], a[48:51], v20, v23 op_sel:[1,0,0] op_sel_hi:[0,0,0] cbsz:2 blgp:2
	s_add_u32 m0, s32, 0xd400
	buffer_load_dwordx4 v8, s[36:39], 0 offen lds
	v_mfma_scale_f32_16x16x128_f8f6f4 a[44:47], v[150:155], v[210:215], a[44:47], v20, v22 op_sel:[1,1,0] op_sel_hi:[0,1,0] cbsz:2 blgp:2
	ds_read_b128 v[54:57], v26 offset:1024
	v_mfma_scale_f32_16x16x128_f8f6f4 a[40:43], v[150:155], v[204:209], a[40:43], v20, v22 op_sel:[1,0,0] op_sel_hi:[0,1,0] cbsz:2 blgp:2
	ds_read_b64 v[58:59], v32 offset:512
	v_mfma_scale_f32_16x16x128_f8f6f4 a[36:39], v[150:155], v[198:203], a[36:39], v20, v22 op_sel:[1,1,0] op_sel_hi:[0,0,0] cbsz:2 blgp:2
	buffer_load_dwordx4 v8, s[36:39], 0 offen offset:1024 lds
	v_mfma_scale_f32_16x16x128_f8f6f4 a[32:35], v[150:155], v[192:197], a[32:35], v20, v22 op_sel:[1,0,0] op_sel_hi:[0,0,0] cbsz:2 blgp:2
	v_mfma_scale_f32_16x16x128_f8f6f4 a[64:67], v[156:161], v[192:197], a[64:67], v20, v22 op_sel_hi:[1,0,0] cbsz:2 blgp:2
	ds_read_b128 v[102:105], v29 offset:1024
	v_mfma_scale_f32_16x16x128_f8f6f4 a[68:71], v[156:161], v[198:203], a[68:71], v20, v22 op_sel:[0,1,0] op_sel_hi:[1,0,0] cbsz:2 blgp:2
	ds_read_b64 v[106:107], v35 offset:512
	v_mfma_scale_f32_16x16x128_f8f6f4 a[72:75], v[156:161], v[204:209], a[72:75], v20, v22 op_sel_hi:[1,1,0] cbsz:2 blgp:2
	buffer_load_dwordx4 v8, s[36:39], 0 offen offset:2048 lds
	v_mfma_scale_f32_16x16x128_f8f6f4 a[76:79], v[156:161], v[210:215], a[76:79], v20, v22 op_sel:[0,1,0] op_sel_hi:[1,1,0] cbsz:2 blgp:2
	ds_read_b128 v[60:63], v26 offset:2048
	v_mfma_scale_f32_16x16x128_f8f6f4 a[80:83], v[156:161], v[216:221], a[80:83], v20, v23 op_sel_hi:[1,0,0] cbsz:2 blgp:2
	ds_read_b64 v[64:65], v32 offset:1024
	v_mfma_scale_f32_16x16x128_f8f6f4 a[84:87], v[156:161], v[222:227], a[84:87], v20, v23 op_sel:[0,1,0] op_sel_hi:[1,0,0] cbsz:2 blgp:2
	v_mfma_scale_f32_16x16x128_f8f6f4 a[88:91], v[156:161], v[228:233], a[88:91], v20, v23 op_sel_hi:[1,1,0] cbsz:2 blgp:2
	s_add_u32 s36, s36, 0x6000
	s_addc_u32 s37, s37, 0
	v_mfma_scale_f32_16x16x128_f8f6f4 a[92:95], v[156:161], v[234:239], a[92:95], v20, v23 op_sel:[0,1,0] op_sel_hi:[1,1,0] cbsz:2 blgp:2
	ds_read_b128 v[108:111], v29 offset:2048
	v_mfma_scale_f32_16x16x128_f8f6f4 a[124:127], v[162:167], v[234:239], a[124:127], v20, v23 op_sel:[1,1,0] op_sel_hi:[1,1,0] cbsz:2 blgp:2
	ds_read_b64 v[112:113], v35 offset:1024
	v_mfma_scale_f32_16x16x128_f8f6f4 a[120:123], v[162:167], v[228:233], a[120:123], v20, v23 op_sel:[1,0,0] op_sel_hi:[1,1,0] cbsz:2 blgp:2
	s_add_u32 m0, s32, 0x12800
	buffer_load_dwordx4 v3, s[40:43], 0 offen lds
	v_mfma_scale_f32_16x16x128_f8f6f4 a[116:119], v[162:167], v[222:227], a[116:119], v20, v23 op_sel:[1,1,0] op_sel_hi:[1,0,0] cbsz:2 blgp:2
	v_mfma_scale_f32_16x16x128_f8f6f4 a[112:115], v[162:167], v[216:221], a[112:115], v20, v23 op_sel:[1,0,0] op_sel_hi:[1,0,0] cbsz:2 blgp:2
	ds_read_b128 v[66:69], v26 offset:3072
	v_mfma_scale_f32_16x16x128_f8f6f4 a[108:111], v[162:167], v[210:215], a[108:111], v20, v22 op_sel:[1,1,0] op_sel_hi:[1,1,0] cbsz:2 blgp:2
	ds_read_b64 v[70:71], v32 offset:1536
	v_mfma_scale_f32_16x16x128_f8f6f4 a[104:107], v[162:167], v[204:209], a[104:107], v20, v22 op_sel:[1,0,0] op_sel_hi:[1,1,0] cbsz:2 blgp:2
	buffer_load_dwordx4 v3, s[40:43], 0 offen offset:1024 lds
	v_mfma_scale_f32_16x16x128_f8f6f4 a[100:103], v[162:167], v[198:203], a[100:103], v20, v22 op_sel:[1,1,0] op_sel_hi:[1,0,0] cbsz:2 blgp:2
	ds_read_b128 v[114:117], v29 offset:3072
	v_mfma_scale_f32_16x16x128_f8f6f4 a[96:99], v[162:167], v[192:197], a[96:99], v20, v22 op_sel:[1,0,0] op_sel_hi:[1,0,0] cbsz:2 blgp:2
	v_mfma_scale_f32_16x16x128_f8f6f4 a[128:131], v[168:173], v[192:197], a[128:131], v21, v22 op_sel_hi:[0,0,0] cbsz:2 blgp:2
	ds_read_b64 v[118:119], v35 offset:1536
	v_mfma_scale_f32_16x16x128_f8f6f4 a[132:135], v[168:173], v[198:203], a[132:135], v21, v22 op_sel:[0,1,0] op_sel_hi:[0,0,0] cbsz:2 blgp:2
	buffer_load_dwordx4 v3, s[40:43], 0 offen offset:2048 lds
	v_mfma_scale_f32_16x16x128_f8f6f4 a[136:139], v[168:173], v[204:209], a[136:139], v21, v22 op_sel_hi:[0,1,0] cbsz:2 blgp:2
	ds_read_b128 v[72:75], v26 offset:4096
	v_mfma_scale_f32_16x16x128_f8f6f4 a[140:143], v[168:173], v[210:215], a[140:143], v21, v22 op_sel:[0,1,0] op_sel_hi:[0,1,0] cbsz:2 blgp:2
	s_add_u32 m0, s32, 0x13400
	buffer_load_dwordx4 v8, s[40:43], 0 offen lds
	v_mfma_scale_f32_16x16x128_f8f6f4 a[144:147], v[168:173], v[216:221], a[144:147], v21, v23 op_sel_hi:[0,0,0] cbsz:2 blgp:2
	ds_read_b64 v[76:77], v32 offset:2048
	v_mfma_scale_f32_16x16x128_f8f6f4 a[148:151], v[168:173], v[222:227], a[148:151], v21, v23 op_sel:[0,1,0] op_sel_hi:[0,0,0] cbsz:2 blgp:2
	v_mfma_scale_f32_16x16x128_f8f6f4 a[152:155], v[168:173], v[228:233], a[152:155], v21, v23 op_sel_hi:[0,1,0] cbsz:2 blgp:2
	ds_read_b128 v[120:123], v29 offset:4096
	v_mfma_scale_f32_16x16x128_f8f6f4 a[156:159], v[168:173], v[234:239], a[156:159], v21, v23 op_sel:[0,1,0] op_sel_hi:[0,1,0] cbsz:2 blgp:2
	buffer_load_dwordx4 v8, s[40:43], 0 offen offset:1024 lds
	v_mfma_scale_f32_16x16x128_f8f6f4 a[188:191], v[174:179], v[234:239], a[188:191], v21, v23 op_sel:[1,1,0] op_sel_hi:[0,1,0] cbsz:2 blgp:2
	ds_read_b64 v[124:125], v35 offset:2048
	v_mfma_scale_f32_16x16x128_f8f6f4 a[184:187], v[174:179], v[228:233], a[184:187], v21, v23 op_sel:[1,0,0] op_sel_hi:[0,1,0] cbsz:2 blgp:2
	ds_read_b128 v[78:81], v26 offset:5120
	v_mfma_scale_f32_16x16x128_f8f6f4 a[180:183], v[174:179], v[222:227], a[180:183], v21, v23 op_sel:[1,1,0] op_sel_hi:[0,0,0] cbsz:2 blgp:2
	v_mfma_scale_f32_16x16x128_f8f6f4 a[176:179], v[174:179], v[216:221], a[176:179], v21, v23 op_sel:[1,0,0] op_sel_hi:[0,0,0] cbsz:2 blgp:2
	buffer_load_dwordx4 v8, s[40:43], 0 offen offset:2048 lds
	v_mfma_scale_f32_16x16x128_f8f6f4 a[172:175], v[174:179], v[210:215], a[172:175], v21, v22 op_sel:[1,1,0] op_sel_hi:[0,1,0] cbsz:2 blgp:2
	ds_read_b64 v[82:83], v32 offset:2560
	v_mfma_scale_f32_16x16x128_f8f6f4 a[168:171], v[174:179], v[204:209], a[168:171], v21, v22 op_sel:[1,0,0] op_sel_hi:[0,1,0] cbsz:2 blgp:2
	ds_read_b128 v[126:129], v29 offset:5120
	v_mfma_scale_f32_16x16x128_f8f6f4 a[164:167], v[174:179], v[198:203], a[164:167], v21, v22 op_sel:[1,1,0] op_sel_hi:[0,0,0] cbsz:2 blgp:2
	s_add_u32 s40, s40, 0x6000
	s_addc_u32 s41, s41, 0
	v_mfma_scale_f32_16x16x128_f8f6f4 a[160:163], v[174:179], v[192:197], a[160:163], v21, v22 op_sel:[1,0,0] op_sel_hi:[0,0,0] cbsz:2 blgp:2
	v_mfma_scale_f32_16x16x128_f8f6f4 a[192:195], v[180:185], v[192:197], a[192:195], v21, v22 op_sel_hi:[1,0,0] cbsz:2 blgp:2
	ds_read_b64 v[130:131], v35 offset:2560
	v_mfma_scale_f32_16x16x128_f8f6f4 a[196:199], v[180:185], v[198:203], a[196:199], v21, v22 op_sel:[0,1,0] op_sel_hi:[1,0,0] cbsz:2 blgp:2
	ds_read_b128 v[84:87], v26 offset:6144
	v_mfma_scale_f32_16x16x128_f8f6f4 a[200:203], v[180:185], v[204:209], a[200:203], v21, v22 op_sel_hi:[1,1,0] cbsz:2 blgp:2
	s_add_u32 m0, s33, 0x18800
	buffer_load_dword v4, s[44:47], 0 offen lds
	v_mfma_scale_f32_16x16x128_f8f6f4 a[204:207], v[180:185], v[210:215], a[204:207], v21, v22 op_sel:[0,1,0] op_sel_hi:[1,1,0] cbsz:2 blgp:2
	ds_read_b64 v[88:89], v32 offset:3072
	v_mfma_scale_f32_16x16x128_f8f6f4 a[208:211], v[180:185], v[216:221], a[208:211], v21, v23 op_sel_hi:[1,0,0] cbsz:2 blgp:2
	ds_read_b128 v[132:135], v29 offset:6144
	v_mfma_scale_f32_16x16x128_f8f6f4 a[212:215], v[180:185], v[222:227], a[212:215], v21, v23 op_sel:[0,1,0] op_sel_hi:[1,0,0] cbsz:2 blgp:2
	v_mfma_scale_f32_16x16x128_f8f6f4 a[216:219], v[180:185], v[228:233], a[216:219], v21, v23 op_sel_hi:[1,1,0] cbsz:2 blgp:2
	s_add_u32 s44, s44, 0x4000
	s_addc_u32 s45, s45, 0
	v_mfma_scale_f32_16x16x128_f8f6f4 a[220:223], v[180:185], v[234:239], a[220:223], v21, v23 op_sel:[0,1,0] op_sel_hi:[1,1,0] cbsz:2 blgp:2
	ds_read_b64 v[136:137], v35 offset:3072
	v_mfma_scale_f32_16x16x128_f8f6f4 a[252:255], v[186:191], v[234:239], a[252:255], v21, v23 op_sel:[1,1,0] op_sel_hi:[1,1,0] cbsz:2 blgp:2
	ds_read_b128 v[90:93], v26 offset:7168
	v_mfma_scale_f32_16x16x128_f8f6f4 a[248:251], v[186:191], v[228:233], a[248:251], v21, v23 op_sel:[1,0,0] op_sel_hi:[1,1,0] cbsz:2 blgp:2
	s_add_u32 m0, s33, 0x18c00
	buffer_load_dword v4, s[48:51], 0 offen lds
	v_mfma_scale_f32_16x16x128_f8f6f4 a[244:247], v[186:191], v[222:227], a[244:247], v21, v23 op_sel:[1,1,0] op_sel_hi:[1,0,0] cbsz:2 blgp:2
	v_mfma_scale_f32_16x16x128_f8f6f4 a[240:243], v[186:191], v[216:221], a[240:243], v21, v23 op_sel:[1,0,0] op_sel_hi:[1,0,0] cbsz:2 blgp:2
	ds_read_b64 v[94:95], v32 offset:3584
	v_mfma_scale_f32_16x16x128_f8f6f4 a[236:239], v[186:191], v[210:215], a[236:239], v21, v22 op_sel:[1,1,0] op_sel_hi:[1,1,0] cbsz:2 blgp:2
	ds_read_b128 v[138:141], v29 offset:7168
	v_mfma_scale_f32_16x16x128_f8f6f4 a[232:235], v[186:191], v[204:209], a[232:235], v21, v22 op_sel:[1,0,0] op_sel_hi:[1,1,0] cbsz:2 blgp:2
	s_add_u32 s48, s48, 0x4000
	s_addc_u32 s49, s49, 0
	v_mfma_scale_f32_16x16x128_f8f6f4 a[228:231], v[186:191], v[198:203], a[228:231], v21, v22 op_sel:[1,1,0] op_sel_hi:[1,0,0] cbsz:2 blgp:2
	ds_read_b64 v[142:143], v35 offset:3584
	v_mfma_scale_f32_16x16x128_f8f6f4 a[224:227], v[186:191], v[192:197], a[224:227], v21, v22 op_sel:[1,0,0] op_sel_hi:[1,0,0] cbsz:2 blgp:2
	s_waitcnt vmcnt(14)
	s_waitcnt lgkmcnt(0)
	s_barrier
	v_mfma_scale_f32_16x16x128_f8f6f4 a[0:3], v[48:53], v[96:101], a[0:3], v16, v18 op_sel_hi:[0,0,0] cbsz:2 blgp:2
	ds_read_b64 v[20:21], v36
	v_mfma_scale_f32_16x16x128_f8f6f4 a[4:7], v[48:53], v[102:107], a[4:7], v16, v18 op_sel:[0,1,0] op_sel_hi:[0,0,0] cbsz:2 blgp:2
	s_add_u32 m0, s32, 0x19000
	buffer_load_dwordx4 v3, s[36:39], 0 offen lds
	v_mfma_scale_f32_16x16x128_f8f6f4 a[8:11], v[48:53], v[108:113], a[8:11], v16, v18 op_sel_hi:[0,1,0] cbsz:2 blgp:2
	ds_read_b64 v[22:23], v39
	v_mfma_scale_f32_16x16x128_f8f6f4 a[12:15], v[48:53], v[114:119], a[12:15], v16, v18 op_sel:[0,1,0] op_sel_hi:[0,1,0] cbsz:2 blgp:2
	buffer_load_dwordx4 v3, s[36:39], 0 offen offset:1024 lds
	v_mfma_scale_f32_16x16x128_f8f6f4 a[16:19], v[48:53], v[120:125], a[16:19], v16, v19 op_sel_hi:[0,0,0] cbsz:2 blgp:2
	ds_read_b128 v[144:147], v24
	v_mfma_scale_f32_16x16x128_f8f6f4 a[20:23], v[48:53], v[126:131], a[20:23], v16, v19 op_sel:[0,1,0] op_sel_hi:[0,0,0] cbsz:2 blgp:2
	v_mfma_scale_f32_16x16x128_f8f6f4 a[24:27], v[48:53], v[132:137], a[24:27], v16, v19 op_sel_hi:[0,1,0] cbsz:2 blgp:2
	ds_read_b64 v[148:149], v30
	v_mfma_scale_f32_16x16x128_f8f6f4 a[28:31], v[48:53], v[138:143], a[28:31], v16, v19 op_sel:[0,1,0] op_sel_hi:[0,1,0] cbsz:2 blgp:2
	buffer_load_dwordx4 v3, s[36:39], 0 offen offset:2048 lds
	v_mfma_scale_f32_16x16x128_f8f6f4 a[60:63], v[54:59], v[138:143], a[60:63], v16, v19 op_sel:[1,1,0] op_sel_hi:[0,1,0] cbsz:2 blgp:2
	ds_read_b128 v[192:195], v27
	v_mfma_scale_f32_16x16x128_f8f6f4 a[56:59], v[54:59], v[132:137], a[56:59], v16, v19 op_sel:[1,0,0] op_sel_hi:[0,1,0] cbsz:2 blgp:2
	ds_read_b64 v[196:197], v33
	v_mfma_scale_f32_16x16x128_f8f6f4 a[52:55], v[54:59], v[126:131], a[52:55], v16, v19 op_sel:[1,1,0] op_sel_hi:[0,0,0] cbsz:2 blgp:2
	v_mfma_scale_f32_16x16x128_f8f6f4 a[48:51], v[54:59], v[120:125], a[48:51], v16, v19 op_sel:[1,0,0] op_sel_hi:[0,0,0] cbsz:2 blgp:2
	s_add_u32 m0, s32, 0x19c00
	buffer_load_dwordx4 v8, s[36:39], 0 offen lds
	v_mfma_scale_f32_16x16x128_f8f6f4 a[44:47], v[54:59], v[114:119], a[44:47], v16, v18 op_sel:[1,1,0] op_sel_hi:[0,1,0] cbsz:2 blgp:2
	ds_read_b128 v[150:153], v24 offset:1024
	v_mfma_scale_f32_16x16x128_f8f6f4 a[40:43], v[54:59], v[108:113], a[40:43], v16, v18 op_sel:[1,0,0] op_sel_hi:[0,1,0] cbsz:2 blgp:2
	ds_read_b64 v[154:155], v30 offset:512
	v_mfma_scale_f32_16x16x128_f8f6f4 a[36:39], v[54:59], v[102:107], a[36:39], v16, v18 op_sel:[1,1,0] op_sel_hi:[0,0,0] cbsz:2 blgp:2
	buffer_load_dwordx4 v8, s[36:39], 0 offen offset:1024 lds
	v_mfma_scale_f32_16x16x128_f8f6f4 a[32:35], v[54:59], v[96:101], a[32:35], v16, v18 op_sel:[1,0,0] op_sel_hi:[0,0,0] cbsz:2 blgp:2
	v_mfma_scale_f32_16x16x128_f8f6f4 a[64:67], v[60:65], v[96:101], a[64:67], v16, v18 op_sel_hi:[1,0,0] cbsz:2 blgp:2
	ds_read_b128 v[198:201], v27 offset:1024
	v_mfma_scale_f32_16x16x128_f8f6f4 a[68:71], v[60:65], v[102:107], a[68:71], v16, v18 op_sel:[0,1,0] op_sel_hi:[1,0,0] cbsz:2 blgp:2
	ds_read_b64 v[202:203], v33 offset:512
	v_mfma_scale_f32_16x16x128_f8f6f4 a[72:75], v[60:65], v[108:113], a[72:75], v16, v18 op_sel_hi:[1,1,0] cbsz:2 blgp:2
	buffer_load_dwordx4 v8, s[36:39], 0 offen offset:2048 lds
	v_mfma_scale_f32_16x16x128_f8f6f4 a[76:79], v[60:65], v[114:119], a[76:79], v16, v18 op_sel:[0,1,0] op_sel_hi:[1,1,0] cbsz:2 blgp:2
	ds_read_b128 v[156:159], v24 offset:2048
	v_mfma_scale_f32_16x16x128_f8f6f4 a[80:83], v[60:65], v[120:125], a[80:83], v16, v19 op_sel_hi:[1,0,0] cbsz:2 blgp:2
	ds_read_b64 v[160:161], v30 offset:1024
	v_mfma_scale_f32_16x16x128_f8f6f4 a[84:87], v[60:65], v[126:131], a[84:87], v16, v19 op_sel:[0,1,0] op_sel_hi:[1,0,0] cbsz:2 blgp:2
	v_mfma_scale_f32_16x16x128_f8f6f4 a[88:91], v[60:65], v[132:137], a[88:91], v16, v19 op_sel_hi:[1,1,0] cbsz:2 blgp:2
	s_add_u32 s36, s36, 0x6000
	s_addc_u32 s37, s37, 0
	v_mfma_scale_f32_16x16x128_f8f6f4 a[92:95], v[60:65], v[138:143], a[92:95], v16, v19 op_sel:[0,1,0] op_sel_hi:[1,1,0] cbsz:2 blgp:2
	ds_read_b128 v[204:207], v27 offset:2048
	v_mfma_scale_f32_16x16x128_f8f6f4 a[124:127], v[66:71], v[138:143], a[124:127], v16, v19 op_sel:[1,1,0] op_sel_hi:[1,1,0] cbsz:2 blgp:2
	ds_read_b64 v[208:209], v33 offset:1024
	v_mfma_scale_f32_16x16x128_f8f6f4 a[120:123], v[66:71], v[132:137], a[120:123], v16, v19 op_sel:[1,0,0] op_sel_hi:[1,1,0] cbsz:2 blgp:2
	s_add_u32 m0, s32, 0x1f000
	buffer_load_dwordx4 v3, s[40:43], 0 offen lds
	v_mfma_scale_f32_16x16x128_f8f6f4 a[116:119], v[66:71], v[126:131], a[116:119], v16, v19 op_sel:[1,1,0] op_sel_hi:[1,0,0] cbsz:2 blgp:2
	v_mfma_scale_f32_16x16x128_f8f6f4 a[112:115], v[66:71], v[120:125], a[112:115], v16, v19 op_sel:[1,0,0] op_sel_hi:[1,0,0] cbsz:2 blgp:2
	ds_read_b128 v[162:165], v24 offset:3072
	v_mfma_scale_f32_16x16x128_f8f6f4 a[108:111], v[66:71], v[114:119], a[108:111], v16, v18 op_sel:[1,1,0] op_sel_hi:[1,1,0] cbsz:2 blgp:2
	ds_read_b64 v[166:167], v30 offset:1536
	v_mfma_scale_f32_16x16x128_f8f6f4 a[104:107], v[66:71], v[108:113], a[104:107], v16, v18 op_sel:[1,0,0] op_sel_hi:[1,1,0] cbsz:2 blgp:2
	buffer_load_dwordx4 v3, s[40:43], 0 offen offset:1024 lds
	v_mfma_scale_f32_16x16x128_f8f6f4 a[100:103], v[66:71], v[102:107], a[100:103], v16, v18 op_sel:[1,1,0] op_sel_hi:[1,0,0] cbsz:2 blgp:2
	ds_read_b128 v[210:213], v27 offset:3072
	v_mfma_scale_f32_16x16x128_f8f6f4 a[96:99], v[66:71], v[96:101], a[96:99], v16, v18 op_sel:[1,0,0] op_sel_hi:[1,0,0] cbsz:2 blgp:2
	v_mfma_scale_f32_16x16x128_f8f6f4 a[128:131], v[72:77], v[96:101], a[128:131], v17, v18 op_sel_hi:[0,0,0] cbsz:2 blgp:2
	ds_read_b64 v[214:215], v33 offset:1536
	v_mfma_scale_f32_16x16x128_f8f6f4 a[132:135], v[72:77], v[102:107], a[132:135], v17, v18 op_sel:[0,1,0] op_sel_hi:[0,0,0] cbsz:2 blgp:2
	buffer_load_dwordx4 v3, s[40:43], 0 offen offset:2048 lds
	v_mfma_scale_f32_16x16x128_f8f6f4 a[136:139], v[72:77], v[108:113], a[136:139], v17, v18 op_sel_hi:[0,1,0] cbsz:2 blgp:2
	ds_read_b128 v[168:171], v24 offset:4096
	v_mfma_scale_f32_16x16x128_f8f6f4 a[140:143], v[72:77], v[114:119], a[140:143], v17, v18 op_sel:[0,1,0] op_sel_hi:[0,1,0] cbsz:2 blgp:2
	s_add_u32 m0, s32, 0x1fc00
	buffer_load_dwordx4 v8, s[40:43], 0 offen lds
	v_mfma_scale_f32_16x16x128_f8f6f4 a[144:147], v[72:77], v[120:125], a[144:147], v17, v19 op_sel_hi:[0,0,0] cbsz:2 blgp:2
	ds_read_b64 v[172:173], v30 offset:2048
	v_mfma_scale_f32_16x16x128_f8f6f4 a[148:151], v[72:77], v[126:131], a[148:151], v17, v19 op_sel:[0,1,0] op_sel_hi:[0,0,0] cbsz:2 blgp:2
	v_mfma_scale_f32_16x16x128_f8f6f4 a[152:155], v[72:77], v[132:137], a[152:155], v17, v19 op_sel_hi:[0,1,0] cbsz:2 blgp:2
	ds_read_b128 v[216:219], v27 offset:4096
	v_mfma_scale_f32_16x16x128_f8f6f4 a[156:159], v[72:77], v[138:143], a[156:159], v17, v19 op_sel:[0,1,0] op_sel_hi:[0,1,0] cbsz:2 blgp:2
	buffer_load_dwordx4 v8, s[40:43], 0 offen offset:1024 lds
	v_mfma_scale_f32_16x16x128_f8f6f4 a[188:191], v[78:83], v[138:143], a[188:191], v17, v19 op_sel:[1,1,0] op_sel_hi:[0,1,0] cbsz:2 blgp:2
	ds_read_b64 v[220:221], v33 offset:2048
	v_mfma_scale_f32_16x16x128_f8f6f4 a[184:187], v[78:83], v[132:137], a[184:187], v17, v19 op_sel:[1,0,0] op_sel_hi:[0,1,0] cbsz:2 blgp:2
	ds_read_b128 v[174:177], v24 offset:5120
	v_mfma_scale_f32_16x16x128_f8f6f4 a[180:183], v[78:83], v[126:131], a[180:183], v17, v19 op_sel:[1,1,0] op_sel_hi:[0,0,0] cbsz:2 blgp:2
	v_mfma_scale_f32_16x16x128_f8f6f4 a[176:179], v[78:83], v[120:125], a[176:179], v17, v19 op_sel:[1,0,0] op_sel_hi:[0,0,0] cbsz:2 blgp:2
	buffer_load_dwordx4 v8, s[40:43], 0 offen offset:2048 lds
	v_mfma_scale_f32_16x16x128_f8f6f4 a[172:175], v[78:83], v[114:119], a[172:175], v17, v18 op_sel:[1,1,0] op_sel_hi:[0,1,0] cbsz:2 blgp:2
	ds_read_b64 v[178:179], v30 offset:2560
	v_mfma_scale_f32_16x16x128_f8f6f4 a[168:171], v[78:83], v[108:113], a[168:171], v17, v18 op_sel:[1,0,0] op_sel_hi:[0,1,0] cbsz:2 blgp:2
	ds_read_b128 v[222:225], v27 offset:5120
	v_mfma_scale_f32_16x16x128_f8f6f4 a[164:167], v[78:83], v[102:107], a[164:167], v17, v18 op_sel:[1,1,0] op_sel_hi:[0,0,0] cbsz:2 blgp:2
	s_add_u32 s40, s40, 0x6000
	s_addc_u32 s41, s41, 0
	v_mfma_scale_f32_16x16x128_f8f6f4 a[160:163], v[78:83], v[96:101], a[160:163], v17, v18 op_sel:[1,0,0] op_sel_hi:[0,0,0] cbsz:2 blgp:2
	v_mfma_scale_f32_16x16x128_f8f6f4 a[192:195], v[84:89], v[96:101], a[192:195], v17, v18 op_sel_hi:[1,0,0] cbsz:2 blgp:2
	ds_read_b64 v[226:227], v33 offset:2560
	v_mfma_scale_f32_16x16x128_f8f6f4 a[196:199], v[84:89], v[102:107], a[196:199], v17, v18 op_sel:[0,1,0] op_sel_hi:[1,0,0] cbsz:2 blgp:2
	ds_read_b128 v[180:183], v24 offset:6144
	v_mfma_scale_f32_16x16x128_f8f6f4 a[200:203], v[84:89], v[108:113], a[200:203], v17, v18 op_sel_hi:[1,1,0] cbsz:2 blgp:2
	s_add_u32 m0, s33, 0x25000
	buffer_load_dword v4, s[44:47], 0 offen lds
	v_mfma_scale_f32_16x16x128_f8f6f4 a[204:207], v[84:89], v[114:119], a[204:207], v17, v18 op_sel:[0,1,0] op_sel_hi:[1,1,0] cbsz:2 blgp:2
	ds_read_b64 v[184:185], v30 offset:3072
	v_mfma_scale_f32_16x16x128_f8f6f4 a[208:211], v[84:89], v[120:125], a[208:211], v17, v19 op_sel_hi:[1,0,0] cbsz:2 blgp:2
	ds_read_b128 v[228:231], v27 offset:6144
	v_mfma_scale_f32_16x16x128_f8f6f4 a[212:215], v[84:89], v[126:131], a[212:215], v17, v19 op_sel:[0,1,0] op_sel_hi:[1,0,0] cbsz:2 blgp:2
	v_mfma_scale_f32_16x16x128_f8f6f4 a[216:219], v[84:89], v[132:137], a[216:219], v17, v19 op_sel_hi:[1,1,0] cbsz:2 blgp:2
	s_add_u32 s44, s44, 0x4000
	s_addc_u32 s45, s45, 0
	v_mfma_scale_f32_16x16x128_f8f6f4 a[220:223], v[84:89], v[138:143], a[220:223], v17, v19 op_sel:[0,1,0] op_sel_hi:[1,1,0] cbsz:2 blgp:2
	ds_read_b64 v[232:233], v33 offset:3072
	v_mfma_scale_f32_16x16x128_f8f6f4 a[252:255], v[90:95], v[138:143], a[252:255], v17, v19 op_sel:[1,1,0] op_sel_hi:[1,1,0] cbsz:2 blgp:2
	ds_read_b128 v[186:189], v24 offset:7168
	v_mfma_scale_f32_16x16x128_f8f6f4 a[248:251], v[90:95], v[132:137], a[248:251], v17, v19 op_sel:[1,0,0] op_sel_hi:[1,1,0] cbsz:2 blgp:2
	s_add_u32 m0, s33, 0x25400
	buffer_load_dword v4, s[48:51], 0 offen lds
	v_mfma_scale_f32_16x16x128_f8f6f4 a[244:247], v[90:95], v[126:131], a[244:247], v17, v19 op_sel:[1,1,0] op_sel_hi:[1,0,0] cbsz:2 blgp:2
	v_mfma_scale_f32_16x16x128_f8f6f4 a[240:243], v[90:95], v[120:125], a[240:243], v17, v19 op_sel:[1,0,0] op_sel_hi:[1,0,0] cbsz:2 blgp:2
	ds_read_b64 v[190:191], v30 offset:3584
	v_mfma_scale_f32_16x16x128_f8f6f4 a[236:239], v[90:95], v[114:119], a[236:239], v17, v18 op_sel:[1,1,0] op_sel_hi:[1,1,0] cbsz:2 blgp:2
	ds_read_b128 v[234:237], v27 offset:7168
	v_mfma_scale_f32_16x16x128_f8f6f4 a[232:235], v[90:95], v[108:113], a[232:235], v17, v18 op_sel:[1,0,0] op_sel_hi:[1,1,0] cbsz:2 blgp:2
	s_add_u32 s48, s48, 0x4000
	s_addc_u32 s49, s49, 0
	v_mfma_scale_f32_16x16x128_f8f6f4 a[228:231], v[90:95], v[102:107], a[228:231], v17, v18 op_sel:[1,1,0] op_sel_hi:[1,0,0] cbsz:2 blgp:2
	ds_read_b64 v[238:239], v33 offset:3584
	v_mfma_scale_f32_16x16x128_f8f6f4 a[224:227], v[90:95], v[96:101], a[224:227], v17, v18 op_sel:[1,0,0] op_sel_hi:[1,0,0] cbsz:2 blgp:2
	s_waitcnt vmcnt(14)
	s_waitcnt lgkmcnt(0)
	s_barrier
	v_mfma_scale_f32_16x16x128_f8f6f4 a[0:3], v[144:149], v[192:197], a[0:3], v20, v22 op_sel_hi:[0,0,0] cbsz:2 blgp:2
	ds_read_b64 v[16:17], v37
	v_mfma_scale_f32_16x16x128_f8f6f4 a[4:7], v[144:149], v[198:203], a[4:7], v20, v22 op_sel:[0,1,0] op_sel_hi:[0,0,0] cbsz:2 blgp:2
	s_add_u32 m0, s32, 0
	buffer_load_dwordx4 v3, s[36:39], 0 offen lds
	v_mfma_scale_f32_16x16x128_f8f6f4 a[8:11], v[144:149], v[204:209], a[8:11], v20, v22 op_sel_hi:[0,1,0] cbsz:2 blgp:2
	ds_read_b64 v[18:19], v40
	v_mfma_scale_f32_16x16x128_f8f6f4 a[12:15], v[144:149], v[210:215], a[12:15], v20, v22 op_sel:[0,1,0] op_sel_hi:[0,1,0] cbsz:2 blgp:2
	buffer_load_dwordx4 v3, s[36:39], 0 offen offset:1024 lds
	v_mfma_scale_f32_16x16x128_f8f6f4 a[16:19], v[144:149], v[216:221], a[16:19], v20, v23 op_sel_hi:[0,0,0] cbsz:2 blgp:2
	ds_read_b128 v[48:51], v25
	v_mfma_scale_f32_16x16x128_f8f6f4 a[20:23], v[144:149], v[222:227], a[20:23], v20, v23 op_sel:[0,1,0] op_sel_hi:[0,0,0] cbsz:2 blgp:2
	v_mfma_scale_f32_16x16x128_f8f6f4 a[24:27], v[144:149], v[228:233], a[24:27], v20, v23 op_sel_hi:[0,1,0] cbsz:2 blgp:2
	ds_read_b64 v[52:53], v31
	v_mfma_scale_f32_16x16x128_f8f6f4 a[28:31], v[144:149], v[234:239], a[28:31], v20, v23 op_sel:[0,1,0] op_sel_hi:[0,1,0] cbsz:2 blgp:2
	buffer_load_dwordx4 v3, s[36:39], 0 offen offset:2048 lds
	v_mfma_scale_f32_16x16x128_f8f6f4 a[60:63], v[150:155], v[234:239], a[60:63], v20, v23 op_sel:[1,1,0] op_sel_hi:[0,1,0] cbsz:2 blgp:2
	ds_read_b128 v[96:99], v28
	v_mfma_scale_f32_16x16x128_f8f6f4 a[56:59], v[150:155], v[228:233], a[56:59], v20, v23 op_sel:[1,0,0] op_sel_hi:[0,1,0] cbsz:2 blgp:2
	ds_read_b64 v[100:101], v34
	v_mfma_scale_f32_16x16x128_f8f6f4 a[52:55], v[150:155], v[222:227], a[52:55], v20, v23 op_sel:[1,1,0] op_sel_hi:[0,0,0] cbsz:2 blgp:2
	v_mfma_scale_f32_16x16x128_f8f6f4 a[48:51], v[150:155], v[216:221], a[48:51], v20, v23 op_sel:[1,0,0] op_sel_hi:[0,0,0] cbsz:2 blgp:2
	s_add_u32 m0, s32, 0xc00
	buffer_load_dwordx4 v8, s[36:39], 0 offen lds
	v_mfma_scale_f32_16x16x128_f8f6f4 a[44:47], v[150:155], v[210:215], a[44:47], v20, v22 op_sel:[1,1,0] op_sel_hi:[0,1,0] cbsz:2 blgp:2
	ds_read_b128 v[54:57], v25 offset:1024
	v_mfma_scale_f32_16x16x128_f8f6f4 a[40:43], v[150:155], v[204:209], a[40:43], v20, v22 op_sel:[1,0,0] op_sel_hi:[0,1,0] cbsz:2 blgp:2
	ds_read_b64 v[58:59], v31 offset:512
	v_mfma_scale_f32_16x16x128_f8f6f4 a[36:39], v[150:155], v[198:203], a[36:39], v20, v22 op_sel:[1,1,0] op_sel_hi:[0,0,0] cbsz:2 blgp:2
	buffer_load_dwordx4 v8, s[36:39], 0 offen offset:1024 lds
	v_mfma_scale_f32_16x16x128_f8f6f4 a[32:35], v[150:155], v[192:197], a[32:35], v20, v22 op_sel:[1,0,0] op_sel_hi:[0,0,0] cbsz:2 blgp:2
	v_mfma_scale_f32_16x16x128_f8f6f4 a[64:67], v[156:161], v[192:197], a[64:67], v20, v22 op_sel_hi:[1,0,0] cbsz:2 blgp:2
	ds_read_b128 v[102:105], v28 offset:1024
	v_mfma_scale_f32_16x16x128_f8f6f4 a[68:71], v[156:161], v[198:203], a[68:71], v20, v22 op_sel:[0,1,0] op_sel_hi:[1,0,0] cbsz:2 blgp:2
	ds_read_b64 v[106:107], v34 offset:512
	v_mfma_scale_f32_16x16x128_f8f6f4 a[72:75], v[156:161], v[204:209], a[72:75], v20, v22 op_sel_hi:[1,1,0] cbsz:2 blgp:2
	buffer_load_dwordx4 v8, s[36:39], 0 offen offset:2048 lds
	v_mfma_scale_f32_16x16x128_f8f6f4 a[76:79], v[156:161], v[210:215], a[76:79], v20, v22 op_sel:[0,1,0] op_sel_hi:[1,1,0] cbsz:2 blgp:2
	ds_read_b128 v[60:63], v25 offset:2048
	v_mfma_scale_f32_16x16x128_f8f6f4 a[80:83], v[156:161], v[216:221], a[80:83], v20, v23 op_sel_hi:[1,0,0] cbsz:2 blgp:2
	ds_read_b64 v[64:65], v31 offset:1024
	v_mfma_scale_f32_16x16x128_f8f6f4 a[84:87], v[156:161], v[222:227], a[84:87], v20, v23 op_sel:[0,1,0] op_sel_hi:[1,0,0] cbsz:2 blgp:2
	v_mfma_scale_f32_16x16x128_f8f6f4 a[88:91], v[156:161], v[228:233], a[88:91], v20, v23 op_sel_hi:[1,1,0] cbsz:2 blgp:2
	s_add_u32 s36, s36, 0x6000
	s_addc_u32 s37, s37, 0
	v_mfma_scale_f32_16x16x128_f8f6f4 a[92:95], v[156:161], v[234:239], a[92:95], v20, v23 op_sel:[0,1,0] op_sel_hi:[1,1,0] cbsz:2 blgp:2
	ds_read_b128 v[108:111], v28 offset:2048
	v_mfma_scale_f32_16x16x128_f8f6f4 a[124:127], v[162:167], v[234:239], a[124:127], v20, v23 op_sel:[1,1,0] op_sel_hi:[1,1,0] cbsz:2 blgp:2
	ds_read_b64 v[112:113], v34 offset:1024
	v_mfma_scale_f32_16x16x128_f8f6f4 a[120:123], v[162:167], v[228:233], a[120:123], v20, v23 op_sel:[1,0,0] op_sel_hi:[1,1,0] cbsz:2 blgp:2
	s_add_u32 m0, s32, 0x6000
	buffer_load_dwordx4 v3, s[40:43], 0 offen lds
	v_mfma_scale_f32_16x16x128_f8f6f4 a[116:119], v[162:167], v[222:227], a[116:119], v20, v23 op_sel:[1,1,0] op_sel_hi:[1,0,0] cbsz:2 blgp:2
	v_mfma_scale_f32_16x16x128_f8f6f4 a[112:115], v[162:167], v[216:221], a[112:115], v20, v23 op_sel:[1,0,0] op_sel_hi:[1,0,0] cbsz:2 blgp:2
	ds_read_b128 v[66:69], v25 offset:3072
	v_mfma_scale_f32_16x16x128_f8f6f4 a[108:111], v[162:167], v[210:215], a[108:111], v20, v22 op_sel:[1,1,0] op_sel_hi:[1,1,0] cbsz:2 blgp:2
	ds_read_b64 v[70:71], v31 offset:1536
	v_mfma_scale_f32_16x16x128_f8f6f4 a[104:107], v[162:167], v[204:209], a[104:107], v20, v22 op_sel:[1,0,0] op_sel_hi:[1,1,0] cbsz:2 blgp:2
	buffer_load_dwordx4 v3, s[40:43], 0 offen offset:1024 lds
	v_mfma_scale_f32_16x16x128_f8f6f4 a[100:103], v[162:167], v[198:203], a[100:103], v20, v22 op_sel:[1,1,0] op_sel_hi:[1,0,0] cbsz:2 blgp:2
	ds_read_b128 v[114:117], v28 offset:3072
	v_mfma_scale_f32_16x16x128_f8f6f4 a[96:99], v[162:167], v[192:197], a[96:99], v20, v22 op_sel:[1,0,0] op_sel_hi:[1,0,0] cbsz:2 blgp:2
	v_mfma_scale_f32_16x16x128_f8f6f4 a[128:131], v[168:173], v[192:197], a[128:131], v21, v22 op_sel_hi:[0,0,0] cbsz:2 blgp:2
	ds_read_b64 v[118:119], v34 offset:1536
	v_mfma_scale_f32_16x16x128_f8f6f4 a[132:135], v[168:173], v[198:203], a[132:135], v21, v22 op_sel:[0,1,0] op_sel_hi:[0,0,0] cbsz:2 blgp:2
	buffer_load_dwordx4 v3, s[40:43], 0 offen offset:2048 lds
	v_mfma_scale_f32_16x16x128_f8f6f4 a[136:139], v[168:173], v[204:209], a[136:139], v21, v22 op_sel_hi:[0,1,0] cbsz:2 blgp:2
	ds_read_b128 v[72:75], v25 offset:4096
	v_mfma_scale_f32_16x16x128_f8f6f4 a[140:143], v[168:173], v[210:215], a[140:143], v21, v22 op_sel:[0,1,0] op_sel_hi:[0,1,0] cbsz:2 blgp:2
	s_add_u32 m0, s32, 0x6c00
	buffer_load_dwordx4 v8, s[40:43], 0 offen lds
	v_mfma_scale_f32_16x16x128_f8f6f4 a[144:147], v[168:173], v[216:221], a[144:147], v21, v23 op_sel_hi:[0,0,0] cbsz:2 blgp:2
	ds_read_b64 v[76:77], v31 offset:2048
	v_mfma_scale_f32_16x16x128_f8f6f4 a[148:151], v[168:173], v[222:227], a[148:151], v21, v23 op_sel:[0,1,0] op_sel_hi:[0,0,0] cbsz:2 blgp:2
	v_mfma_scale_f32_16x16x128_f8f6f4 a[152:155], v[168:173], v[228:233], a[152:155], v21, v23 op_sel_hi:[0,1,0] cbsz:2 blgp:2
	ds_read_b128 v[120:123], v28 offset:4096
	v_mfma_scale_f32_16x16x128_f8f6f4 a[156:159], v[168:173], v[234:239], a[156:159], v21, v23 op_sel:[0,1,0] op_sel_hi:[0,1,0] cbsz:2 blgp:2
	buffer_load_dwordx4 v8, s[40:43], 0 offen offset:1024 lds
	v_mfma_scale_f32_16x16x128_f8f6f4 a[188:191], v[174:179], v[234:239], a[188:191], v21, v23 op_sel:[1,1,0] op_sel_hi:[0,1,0] cbsz:2 blgp:2
	ds_read_b64 v[124:125], v34 offset:2048
	v_mfma_scale_f32_16x16x128_f8f6f4 a[184:187], v[174:179], v[228:233], a[184:187], v21, v23 op_sel:[1,0,0] op_sel_hi:[0,1,0] cbsz:2 blgp:2
	ds_read_b128 v[78:81], v25 offset:5120
	v_mfma_scale_f32_16x16x128_f8f6f4 a[180:183], v[174:179], v[222:227], a[180:183], v21, v23 op_sel:[1,1,0] op_sel_hi:[0,0,0] cbsz:2 blgp:2
	v_mfma_scale_f32_16x16x128_f8f6f4 a[176:179], v[174:179], v[216:221], a[176:179], v21, v23 op_sel:[1,0,0] op_sel_hi:[0,0,0] cbsz:2 blgp:2
	buffer_load_dwordx4 v8, s[40:43], 0 offen offset:2048 lds
	v_mfma_scale_f32_16x16x128_f8f6f4 a[172:175], v[174:179], v[210:215], a[172:175], v21, v22 op_sel:[1,1,0] op_sel_hi:[0,1,0] cbsz:2 blgp:2
	ds_read_b64 v[82:83], v31 offset:2560
	v_mfma_scale_f32_16x16x128_f8f6f4 a[168:171], v[174:179], v[204:209], a[168:171], v21, v22 op_sel:[1,0,0] op_sel_hi:[0,1,0] cbsz:2 blgp:2
	ds_read_b128 v[126:129], v28 offset:5120
	v_mfma_scale_f32_16x16x128_f8f6f4 a[164:167], v[174:179], v[198:203], a[164:167], v21, v22 op_sel:[1,1,0] op_sel_hi:[0,0,0] cbsz:2 blgp:2
	s_add_u32 s40, s40, 0x6000
	s_addc_u32 s41, s41, 0
	v_mfma_scale_f32_16x16x128_f8f6f4 a[160:163], v[174:179], v[192:197], a[160:163], v21, v22 op_sel:[1,0,0] op_sel_hi:[0,0,0] cbsz:2 blgp:2
	v_mfma_scale_f32_16x16x128_f8f6f4 a[192:195], v[180:185], v[192:197], a[192:195], v21, v22 op_sel_hi:[1,0,0] cbsz:2 blgp:2
	ds_read_b64 v[130:131], v34 offset:2560
	v_mfma_scale_f32_16x16x128_f8f6f4 a[196:199], v[180:185], v[198:203], a[196:199], v21, v22 op_sel:[0,1,0] op_sel_hi:[1,0,0] cbsz:2 blgp:2
	ds_read_b128 v[84:87], v25 offset:6144
	v_mfma_scale_f32_16x16x128_f8f6f4 a[200:203], v[180:185], v[204:209], a[200:203], v21, v22 op_sel_hi:[1,1,0] cbsz:2 blgp:2
	s_add_u32 m0, s33, 0xc000
	buffer_load_dword v4, s[44:47], 0 offen lds
	v_mfma_scale_f32_16x16x128_f8f6f4 a[204:207], v[180:185], v[210:215], a[204:207], v21, v22 op_sel:[0,1,0] op_sel_hi:[1,1,0] cbsz:2 blgp:2
	ds_read_b64 v[88:89], v31 offset:3072
	v_mfma_scale_f32_16x16x128_f8f6f4 a[208:211], v[180:185], v[216:221], a[208:211], v21, v23 op_sel_hi:[1,0,0] cbsz:2 blgp:2
	ds_read_b128 v[132:135], v28 offset:6144
	v_mfma_scale_f32_16x16x128_f8f6f4 a[212:215], v[180:185], v[222:227], a[212:215], v21, v23 op_sel:[0,1,0] op_sel_hi:[1,0,0] cbsz:2 blgp:2
	v_mfma_scale_f32_16x16x128_f8f6f4 a[216:219], v[180:185], v[228:233], a[216:219], v21, v23 op_sel_hi:[1,1,0] cbsz:2 blgp:2
	s_add_u32 s44, s44, 0x4000
	s_addc_u32 s45, s45, 0
	v_mfma_scale_f32_16x16x128_f8f6f4 a[220:223], v[180:185], v[234:239], a[220:223], v21, v23 op_sel:[0,1,0] op_sel_hi:[1,1,0] cbsz:2 blgp:2
	ds_read_b64 v[136:137], v34 offset:3072
	v_mfma_scale_f32_16x16x128_f8f6f4 a[252:255], v[186:191], v[234:239], a[252:255], v21, v23 op_sel:[1,1,0] op_sel_hi:[1,1,0] cbsz:2 blgp:2
	ds_read_b128 v[90:93], v25 offset:7168
	v_mfma_scale_f32_16x16x128_f8f6f4 a[248:251], v[186:191], v[228:233], a[248:251], v21, v23 op_sel:[1,0,0] op_sel_hi:[1,1,0] cbsz:2 blgp:2
	s_add_u32 m0, s33, 0xc400
	buffer_load_dword v4, s[48:51], 0 offen lds
	v_mfma_scale_f32_16x16x128_f8f6f4 a[244:247], v[186:191], v[222:227], a[244:247], v21, v23 op_sel:[1,1,0] op_sel_hi:[1,0,0] cbsz:2 blgp:2
	v_mfma_scale_f32_16x16x128_f8f6f4 a[240:243], v[186:191], v[216:221], a[240:243], v21, v23 op_sel:[1,0,0] op_sel_hi:[1,0,0] cbsz:2 blgp:2
	ds_read_b64 v[94:95], v31 offset:3584
	v_mfma_scale_f32_16x16x128_f8f6f4 a[236:239], v[186:191], v[210:215], a[236:239], v21, v22 op_sel:[1,1,0] op_sel_hi:[1,1,0] cbsz:2 blgp:2
	ds_read_b128 v[138:141], v28 offset:7168
	v_mfma_scale_f32_16x16x128_f8f6f4 a[232:235], v[186:191], v[204:209], a[232:235], v21, v22 op_sel:[1,0,0] op_sel_hi:[1,1,0] cbsz:2 blgp:2
	s_add_u32 s48, s48, 0x4000
	s_addc_u32 s49, s49, 0
	v_mfma_scale_f32_16x16x128_f8f6f4 a[228:231], v[186:191], v[198:203], a[228:231], v21, v22 op_sel:[1,1,0] op_sel_hi:[1,0,0] cbsz:2 blgp:2
	ds_read_b64 v[142:143], v34 offset:3584
	v_mfma_scale_f32_16x16x128_f8f6f4 a[224:227], v[186:191], v[192:197], a[224:227], v21, v22 op_sel:[1,0,0] op_sel_hi:[1,0,0] cbsz:2 blgp:2
	s_waitcnt vmcnt(14)
	s_waitcnt lgkmcnt(0)
	s_barrier
	v_mfma_scale_f32_16x16x128_f8f6f4 a[0:3], v[48:53], v[96:101], a[0:3], v16, v18 op_sel_hi:[0,0,0] cbsz:2 blgp:2
	ds_read_b64 v[20:21], v38
	v_mfma_scale_f32_16x16x128_f8f6f4 a[4:7], v[48:53], v[102:107], a[4:7], v16, v18 op_sel:[0,1,0] op_sel_hi:[0,0,0] cbsz:2 blgp:2
	s_add_u32 m0, s32, 0xc800
	buffer_load_dwordx4 v3, s[36:39], 0 offen lds
	v_mfma_scale_f32_16x16x128_f8f6f4 a[8:11], v[48:53], v[108:113], a[8:11], v16, v18 op_sel_hi:[0,1,0] cbsz:2 blgp:2
	ds_read_b64 v[22:23], v41
	v_mfma_scale_f32_16x16x128_f8f6f4 a[12:15], v[48:53], v[114:119], a[12:15], v16, v18 op_sel:[0,1,0] op_sel_hi:[0,1,0] cbsz:2 blgp:2
	buffer_load_dwordx4 v3, s[36:39], 0 offen offset:1024 lds
	v_mfma_scale_f32_16x16x128_f8f6f4 a[16:19], v[48:53], v[120:125], a[16:19], v16, v19 op_sel_hi:[0,0,0] cbsz:2 blgp:2
	ds_read_b128 v[144:147], v26
	v_mfma_scale_f32_16x16x128_f8f6f4 a[20:23], v[48:53], v[126:131], a[20:23], v16, v19 op_sel:[0,1,0] op_sel_hi:[0,0,0] cbsz:2 blgp:2
	v_mfma_scale_f32_16x16x128_f8f6f4 a[24:27], v[48:53], v[132:137], a[24:27], v16, v19 op_sel_hi:[0,1,0] cbsz:2 blgp:2
	ds_read_b64 v[148:149], v32
	v_mfma_scale_f32_16x16x128_f8f6f4 a[28:31], v[48:53], v[138:143], a[28:31], v16, v19 op_sel:[0,1,0] op_sel_hi:[0,1,0] cbsz:2 blgp:2
	buffer_load_dwordx4 v3, s[36:39], 0 offen offset:2048 lds
	v_mfma_scale_f32_16x16x128_f8f6f4 a[60:63], v[54:59], v[138:143], a[60:63], v16, v19 op_sel:[1,1,0] op_sel_hi:[0,1,0] cbsz:2 blgp:2
	ds_read_b128 v[192:195], v29
	v_mfma_scale_f32_16x16x128_f8f6f4 a[56:59], v[54:59], v[132:137], a[56:59], v16, v19 op_sel:[1,0,0] op_sel_hi:[0,1,0] cbsz:2 blgp:2
	ds_read_b64 v[196:197], v35
	v_mfma_scale_f32_16x16x128_f8f6f4 a[52:55], v[54:59], v[126:131], a[52:55], v16, v19 op_sel:[1,1,0] op_sel_hi:[0,0,0] cbsz:2 blgp:2
	v_mfma_scale_f32_16x16x128_f8f6f4 a[48:51], v[54:59], v[120:125], a[48:51], v16, v19 op_sel:[1,0,0] op_sel_hi:[0,0,0] cbsz:2 blgp:2
	s_add_u32 m0, s32, 0xd400
	buffer_load_dwordx4 v8, s[36:39], 0 offen lds
	v_mfma_scale_f32_16x16x128_f8f6f4 a[44:47], v[54:59], v[114:119], a[44:47], v16, v18 op_sel:[1,1,0] op_sel_hi:[0,1,0] cbsz:2 blgp:2
	ds_read_b128 v[150:153], v26 offset:1024
	v_mfma_scale_f32_16x16x128_f8f6f4 a[40:43], v[54:59], v[108:113], a[40:43], v16, v18 op_sel:[1,0,0] op_sel_hi:[0,1,0] cbsz:2 blgp:2
	ds_read_b64 v[154:155], v32 offset:512
	v_mfma_scale_f32_16x16x128_f8f6f4 a[36:39], v[54:59], v[102:107], a[36:39], v16, v18 op_sel:[1,1,0] op_sel_hi:[0,0,0] cbsz:2 blgp:2
	buffer_load_dwordx4 v8, s[36:39], 0 offen offset:1024 lds
	v_mfma_scale_f32_16x16x128_f8f6f4 a[32:35], v[54:59], v[96:101], a[32:35], v16, v18 op_sel:[1,0,0] op_sel_hi:[0,0,0] cbsz:2 blgp:2
	v_mfma_scale_f32_16x16x128_f8f6f4 a[64:67], v[60:65], v[96:101], a[64:67], v16, v18 op_sel_hi:[1,0,0] cbsz:2 blgp:2
	ds_read_b128 v[198:201], v29 offset:1024
	v_mfma_scale_f32_16x16x128_f8f6f4 a[68:71], v[60:65], v[102:107], a[68:71], v16, v18 op_sel:[0,1,0] op_sel_hi:[1,0,0] cbsz:2 blgp:2
	ds_read_b64 v[202:203], v35 offset:512
	v_mfma_scale_f32_16x16x128_f8f6f4 a[72:75], v[60:65], v[108:113], a[72:75], v16, v18 op_sel_hi:[1,1,0] cbsz:2 blgp:2
	buffer_load_dwordx4 v8, s[36:39], 0 offen offset:2048 lds
	v_mfma_scale_f32_16x16x128_f8f6f4 a[76:79], v[60:65], v[114:119], a[76:79], v16, v18 op_sel:[0,1,0] op_sel_hi:[1,1,0] cbsz:2 blgp:2
	ds_read_b128 v[156:159], v26 offset:2048
	v_mfma_scale_f32_16x16x128_f8f6f4 a[80:83], v[60:65], v[120:125], a[80:83], v16, v19 op_sel_hi:[1,0,0] cbsz:2 blgp:2
	ds_read_b64 v[160:161], v32 offset:1024
	v_mfma_scale_f32_16x16x128_f8f6f4 a[84:87], v[60:65], v[126:131], a[84:87], v16, v19 op_sel:[0,1,0] op_sel_hi:[1,0,0] cbsz:2 blgp:2
	v_mfma_scale_f32_16x16x128_f8f6f4 a[88:91], v[60:65], v[132:137], a[88:91], v16, v19 op_sel_hi:[1,1,0] cbsz:2 blgp:2
	s_add_u32 s36, s36, 0x6000
	s_addc_u32 s37, s37, 0
	v_mfma_scale_f32_16x16x128_f8f6f4 a[92:95], v[60:65], v[138:143], a[92:95], v16, v19 op_sel:[0,1,0] op_sel_hi:[1,1,0] cbsz:2 blgp:2
	ds_read_b128 v[204:207], v29 offset:2048
	v_mfma_scale_f32_16x16x128_f8f6f4 a[124:127], v[66:71], v[138:143], a[124:127], v16, v19 op_sel:[1,1,0] op_sel_hi:[1,1,0] cbsz:2 blgp:2
	ds_read_b64 v[208:209], v35 offset:1024
	v_mfma_scale_f32_16x16x128_f8f6f4 a[120:123], v[66:71], v[132:137], a[120:123], v16, v19 op_sel:[1,0,0] op_sel_hi:[1,1,0] cbsz:2 blgp:2
	s_add_u32 m0, s32, 0x12800
	buffer_load_dwordx4 v3, s[40:43], 0 offen lds
	v_mfma_scale_f32_16x16x128_f8f6f4 a[116:119], v[66:71], v[126:131], a[116:119], v16, v19 op_sel:[1,1,0] op_sel_hi:[1,0,0] cbsz:2 blgp:2
	v_mfma_scale_f32_16x16x128_f8f6f4 a[112:115], v[66:71], v[120:125], a[112:115], v16, v19 op_sel:[1,0,0] op_sel_hi:[1,0,0] cbsz:2 blgp:2
	ds_read_b128 v[162:165], v26 offset:3072
	v_mfma_scale_f32_16x16x128_f8f6f4 a[108:111], v[66:71], v[114:119], a[108:111], v16, v18 op_sel:[1,1,0] op_sel_hi:[1,1,0] cbsz:2 blgp:2
	ds_read_b64 v[166:167], v32 offset:1536
	v_mfma_scale_f32_16x16x128_f8f6f4 a[104:107], v[66:71], v[108:113], a[104:107], v16, v18 op_sel:[1,0,0] op_sel_hi:[1,1,0] cbsz:2 blgp:2
	buffer_load_dwordx4 v3, s[40:43], 0 offen offset:1024 lds
	v_mfma_scale_f32_16x16x128_f8f6f4 a[100:103], v[66:71], v[102:107], a[100:103], v16, v18 op_sel:[1,1,0] op_sel_hi:[1,0,0] cbsz:2 blgp:2
	ds_read_b128 v[210:213], v29 offset:3072
	v_mfma_scale_f32_16x16x128_f8f6f4 a[96:99], v[66:71], v[96:101], a[96:99], v16, v18 op_sel:[1,0,0] op_sel_hi:[1,0,0] cbsz:2 blgp:2
	v_mfma_scale_f32_16x16x128_f8f6f4 a[128:131], v[72:77], v[96:101], a[128:131], v17, v18 op_sel_hi:[0,0,0] cbsz:2 blgp:2
	ds_read_b64 v[214:215], v35 offset:1536
	v_mfma_scale_f32_16x16x128_f8f6f4 a[132:135], v[72:77], v[102:107], a[132:135], v17, v18 op_sel:[0,1,0] op_sel_hi:[0,0,0] cbsz:2 blgp:2
	buffer_load_dwordx4 v3, s[40:43], 0 offen offset:2048 lds
	v_mfma_scale_f32_16x16x128_f8f6f4 a[136:139], v[72:77], v[108:113], a[136:139], v17, v18 op_sel_hi:[0,1,0] cbsz:2 blgp:2
	ds_read_b128 v[168:171], v26 offset:4096
	v_mfma_scale_f32_16x16x128_f8f6f4 a[140:143], v[72:77], v[114:119], a[140:143], v17, v18 op_sel:[0,1,0] op_sel_hi:[0,1,0] cbsz:2 blgp:2
	s_add_u32 m0, s32, 0x13400
	buffer_load_dwordx4 v8, s[40:43], 0 offen lds
	v_mfma_scale_f32_16x16x128_f8f6f4 a[144:147], v[72:77], v[120:125], a[144:147], v17, v19 op_sel_hi:[0,0,0] cbsz:2 blgp:2
	ds_read_b64 v[172:173], v32 offset:2048
	v_mfma_scale_f32_16x16x128_f8f6f4 a[148:151], v[72:77], v[126:131], a[148:151], v17, v19 op_sel:[0,1,0] op_sel_hi:[0,0,0] cbsz:2 blgp:2
	v_mfma_scale_f32_16x16x128_f8f6f4 a[152:155], v[72:77], v[132:137], a[152:155], v17, v19 op_sel_hi:[0,1,0] cbsz:2 blgp:2
	ds_read_b128 v[216:219], v29 offset:4096
	v_mfma_scale_f32_16x16x128_f8f6f4 a[156:159], v[72:77], v[138:143], a[156:159], v17, v19 op_sel:[0,1,0] op_sel_hi:[0,1,0] cbsz:2 blgp:2
	buffer_load_dwordx4 v8, s[40:43], 0 offen offset:1024 lds
	v_mfma_scale_f32_16x16x128_f8f6f4 a[188:191], v[78:83], v[138:143], a[188:191], v17, v19 op_sel:[1,1,0] op_sel_hi:[0,1,0] cbsz:2 blgp:2
	ds_read_b64 v[220:221], v35 offset:2048
	v_mfma_scale_f32_16x16x128_f8f6f4 a[184:187], v[78:83], v[132:137], a[184:187], v17, v19 op_sel:[1,0,0] op_sel_hi:[0,1,0] cbsz:2 blgp:2
	ds_read_b128 v[174:177], v26 offset:5120
	v_mfma_scale_f32_16x16x128_f8f6f4 a[180:183], v[78:83], v[126:131], a[180:183], v17, v19 op_sel:[1,1,0] op_sel_hi:[0,0,0] cbsz:2 blgp:2
	v_mfma_scale_f32_16x16x128_f8f6f4 a[176:179], v[78:83], v[120:125], a[176:179], v17, v19 op_sel:[1,0,0] op_sel_hi:[0,0,0] cbsz:2 blgp:2
	buffer_load_dwordx4 v8, s[40:43], 0 offen offset:2048 lds
	v_mfma_scale_f32_16x16x128_f8f6f4 a[172:175], v[78:83], v[114:119], a[172:175], v17, v18 op_sel:[1,1,0] op_sel_hi:[0,1,0] cbsz:2 blgp:2
	ds_read_b64 v[178:179], v32 offset:2560
	v_mfma_scale_f32_16x16x128_f8f6f4 a[168:171], v[78:83], v[108:113], a[168:171], v17, v18 op_sel:[1,0,0] op_sel_hi:[0,1,0] cbsz:2 blgp:2
	ds_read_b128 v[222:225], v29 offset:5120
	v_mfma_scale_f32_16x16x128_f8f6f4 a[164:167], v[78:83], v[102:107], a[164:167], v17, v18 op_sel:[1,1,0] op_sel_hi:[0,0,0] cbsz:2 blgp:2
	s_add_u32 s40, s40, 0x6000
	s_addc_u32 s41, s41, 0
	v_mfma_scale_f32_16x16x128_f8f6f4 a[160:163], v[78:83], v[96:101], a[160:163], v17, v18 op_sel:[1,0,0] op_sel_hi:[0,0,0] cbsz:2 blgp:2
	v_mfma_scale_f32_16x16x128_f8f6f4 a[192:195], v[84:89], v[96:101], a[192:195], v17, v18 op_sel_hi:[1,0,0] cbsz:2 blgp:2
	ds_read_b64 v[226:227], v35 offset:2560
	v_mfma_scale_f32_16x16x128_f8f6f4 a[196:199], v[84:89], v[102:107], a[196:199], v17, v18 op_sel:[0,1,0] op_sel_hi:[1,0,0] cbsz:2 blgp:2
	ds_read_b128 v[180:183], v26 offset:6144
	v_mfma_scale_f32_16x16x128_f8f6f4 a[200:203], v[84:89], v[108:113], a[200:203], v17, v18 op_sel_hi:[1,1,0] cbsz:2 blgp:2
	s_add_u32 m0, s33, 0x18800
	buffer_load_dword v4, s[44:47], 0 offen lds
	v_mfma_scale_f32_16x16x128_f8f6f4 a[204:207], v[84:89], v[114:119], a[204:207], v17, v18 op_sel:[0,1,0] op_sel_hi:[1,1,0] cbsz:2 blgp:2
	ds_read_b64 v[184:185], v32 offset:3072
	v_mfma_scale_f32_16x16x128_f8f6f4 a[208:211], v[84:89], v[120:125], a[208:211], v17, v19 op_sel_hi:[1,0,0] cbsz:2 blgp:2
	ds_read_b128 v[228:231], v29 offset:6144
	v_mfma_scale_f32_16x16x128_f8f6f4 a[212:215], v[84:89], v[126:131], a[212:215], v17, v19 op_sel:[0,1,0] op_sel_hi:[1,0,0] cbsz:2 blgp:2
	v_mfma_scale_f32_16x16x128_f8f6f4 a[216:219], v[84:89], v[132:137], a[216:219], v17, v19 op_sel_hi:[1,1,0] cbsz:2 blgp:2
	s_add_u32 s44, s44, 0x4000
	s_addc_u32 s45, s45, 0
	v_mfma_scale_f32_16x16x128_f8f6f4 a[220:223], v[84:89], v[138:143], a[220:223], v17, v19 op_sel:[0,1,0] op_sel_hi:[1,1,0] cbsz:2 blgp:2
	ds_read_b64 v[232:233], v35 offset:3072
	v_mfma_scale_f32_16x16x128_f8f6f4 a[252:255], v[90:95], v[138:143], a[252:255], v17, v19 op_sel:[1,1,0] op_sel_hi:[1,1,0] cbsz:2 blgp:2
	ds_read_b128 v[186:189], v26 offset:7168
	v_mfma_scale_f32_16x16x128_f8f6f4 a[248:251], v[90:95], v[132:137], a[248:251], v17, v19 op_sel:[1,0,0] op_sel_hi:[1,1,0] cbsz:2 blgp:2
	s_add_u32 m0, s33, 0x18c00
	buffer_load_dword v4, s[48:51], 0 offen lds
	v_mfma_scale_f32_16x16x128_f8f6f4 a[244:247], v[90:95], v[126:131], a[244:247], v17, v19 op_sel:[1,1,0] op_sel_hi:[1,0,0] cbsz:2 blgp:2
	v_mfma_scale_f32_16x16x128_f8f6f4 a[240:243], v[90:95], v[120:125], a[240:243], v17, v19 op_sel:[1,0,0] op_sel_hi:[1,0,0] cbsz:2 blgp:2
	ds_read_b64 v[190:191], v32 offset:3584
	v_mfma_scale_f32_16x16x128_f8f6f4 a[236:239], v[90:95], v[114:119], a[236:239], v17, v18 op_sel:[1,1,0] op_sel_hi:[1,1,0] cbsz:2 blgp:2
	ds_read_b128 v[234:237], v29 offset:7168
	v_mfma_scale_f32_16x16x128_f8f6f4 a[232:235], v[90:95], v[108:113], a[232:235], v17, v18 op_sel:[1,0,0] op_sel_hi:[1,1,0] cbsz:2 blgp:2
	s_add_u32 s48, s48, 0x4000
	s_addc_u32 s49, s49, 0
	v_mfma_scale_f32_16x16x128_f8f6f4 a[228:231], v[90:95], v[102:107], a[228:231], v17, v18 op_sel:[1,1,0] op_sel_hi:[1,0,0] cbsz:2 blgp:2
	ds_read_b64 v[238:239], v35 offset:3584
	v_mfma_scale_f32_16x16x128_f8f6f4 a[224:227], v[90:95], v[96:101], a[224:227], v17, v18 op_sel:[1,0,0] op_sel_hi:[1,0,0] cbsz:2 blgp:2
	s_waitcnt vmcnt(14)
	s_waitcnt lgkmcnt(0)
	s_barrier
	v_mfma_scale_f32_16x16x128_f8f6f4 a[0:3], v[144:149], v[192:197], a[0:3], v20, v22 op_sel_hi:[0,0,0] cbsz:2 blgp:2
	ds_read_b64 v[16:17], v36
	v_mfma_scale_f32_16x16x128_f8f6f4 a[4:7], v[144:149], v[198:203], a[4:7], v20, v22 op_sel:[0,1,0] op_sel_hi:[0,0,0] cbsz:2 blgp:2
	s_add_u32 m0, s32, 0x19000
	buffer_load_dwordx4 v3, s[36:39], 0 offen lds
	v_mfma_scale_f32_16x16x128_f8f6f4 a[8:11], v[144:149], v[204:209], a[8:11], v20, v22 op_sel_hi:[0,1,0] cbsz:2 blgp:2
	ds_read_b64 v[18:19], v39
	v_mfma_scale_f32_16x16x128_f8f6f4 a[12:15], v[144:149], v[210:215], a[12:15], v20, v22 op_sel:[0,1,0] op_sel_hi:[0,1,0] cbsz:2 blgp:2
	buffer_load_dwordx4 v3, s[36:39], 0 offen offset:1024 lds
	v_mfma_scale_f32_16x16x128_f8f6f4 a[16:19], v[144:149], v[216:221], a[16:19], v20, v23 op_sel_hi:[0,0,0] cbsz:2 blgp:2
	ds_read_b128 v[48:51], v24
	v_mfma_scale_f32_16x16x128_f8f6f4 a[20:23], v[144:149], v[222:227], a[20:23], v20, v23 op_sel:[0,1,0] op_sel_hi:[0,0,0] cbsz:2 blgp:2
	v_mfma_scale_f32_16x16x128_f8f6f4 a[24:27], v[144:149], v[228:233], a[24:27], v20, v23 op_sel_hi:[0,1,0] cbsz:2 blgp:2
	ds_read_b64 v[52:53], v30
	v_mfma_scale_f32_16x16x128_f8f6f4 a[28:31], v[144:149], v[234:239], a[28:31], v20, v23 op_sel:[0,1,0] op_sel_hi:[0,1,0] cbsz:2 blgp:2
	buffer_load_dwordx4 v3, s[36:39], 0 offen offset:2048 lds
	v_mfma_scale_f32_16x16x128_f8f6f4 a[60:63], v[150:155], v[234:239], a[60:63], v20, v23 op_sel:[1,1,0] op_sel_hi:[0,1,0] cbsz:2 blgp:2
	ds_read_b128 v[96:99], v27
	v_mfma_scale_f32_16x16x128_f8f6f4 a[56:59], v[150:155], v[228:233], a[56:59], v20, v23 op_sel:[1,0,0] op_sel_hi:[0,1,0] cbsz:2 blgp:2
	ds_read_b64 v[100:101], v33
	v_mfma_scale_f32_16x16x128_f8f6f4 a[52:55], v[150:155], v[222:227], a[52:55], v20, v23 op_sel:[1,1,0] op_sel_hi:[0,0,0] cbsz:2 blgp:2
	v_mfma_scale_f32_16x16x128_f8f6f4 a[48:51], v[150:155], v[216:221], a[48:51], v20, v23 op_sel:[1,0,0] op_sel_hi:[0,0,0] cbsz:2 blgp:2
	s_add_u32 m0, s32, 0x19c00
	buffer_load_dwordx4 v8, s[36:39], 0 offen lds
	v_mfma_scale_f32_16x16x128_f8f6f4 a[44:47], v[150:155], v[210:215], a[44:47], v20, v22 op_sel:[1,1,0] op_sel_hi:[0,1,0] cbsz:2 blgp:2
	ds_read_b128 v[54:57], v24 offset:1024
	v_mfma_scale_f32_16x16x128_f8f6f4 a[40:43], v[150:155], v[204:209], a[40:43], v20, v22 op_sel:[1,0,0] op_sel_hi:[0,1,0] cbsz:2 blgp:2
	ds_read_b64 v[58:59], v30 offset:512
	v_mfma_scale_f32_16x16x128_f8f6f4 a[36:39], v[150:155], v[198:203], a[36:39], v20, v22 op_sel:[1,1,0] op_sel_hi:[0,0,0] cbsz:2 blgp:2
	buffer_load_dwordx4 v8, s[36:39], 0 offen offset:1024 lds
	v_mfma_scale_f32_16x16x128_f8f6f4 a[32:35], v[150:155], v[192:197], a[32:35], v20, v22 op_sel:[1,0,0] op_sel_hi:[0,0,0] cbsz:2 blgp:2
	v_mfma_scale_f32_16x16x128_f8f6f4 a[64:67], v[156:161], v[192:197], a[64:67], v20, v22 op_sel_hi:[1,0,0] cbsz:2 blgp:2
	ds_read_b128 v[102:105], v27 offset:1024
	v_mfma_scale_f32_16x16x128_f8f6f4 a[68:71], v[156:161], v[198:203], a[68:71], v20, v22 op_sel:[0,1,0] op_sel_hi:[1,0,0] cbsz:2 blgp:2
	ds_read_b64 v[106:107], v33 offset:512
	v_mfma_scale_f32_16x16x128_f8f6f4 a[72:75], v[156:161], v[204:209], a[72:75], v20, v22 op_sel_hi:[1,1,0] cbsz:2 blgp:2
	buffer_load_dwordx4 v8, s[36:39], 0 offen offset:2048 lds
	v_mfma_scale_f32_16x16x128_f8f6f4 a[76:79], v[156:161], v[210:215], a[76:79], v20, v22 op_sel:[0,1,0] op_sel_hi:[1,1,0] cbsz:2 blgp:2
	ds_read_b128 v[60:63], v24 offset:2048
	v_mfma_scale_f32_16x16x128_f8f6f4 a[80:83], v[156:161], v[216:221], a[80:83], v20, v23 op_sel_hi:[1,0,0] cbsz:2 blgp:2
	ds_read_b64 v[64:65], v30 offset:1024
	v_mfma_scale_f32_16x16x128_f8f6f4 a[84:87], v[156:161], v[222:227], a[84:87], v20, v23 op_sel:[0,1,0] op_sel_hi:[1,0,0] cbsz:2 blgp:2
	v_mfma_scale_f32_16x16x128_f8f6f4 a[88:91], v[156:161], v[228:233], a[88:91], v20, v23 op_sel_hi:[1,1,0] cbsz:2 blgp:2
	s_add_u32 s36, s36, 0x6000
	s_addc_u32 s37, s37, 0
	v_mfma_scale_f32_16x16x128_f8f6f4 a[92:95], v[156:161], v[234:239], a[92:95], v20, v23 op_sel:[0,1,0] op_sel_hi:[1,1,0] cbsz:2 blgp:2
	ds_read_b128 v[108:111], v27 offset:2048
	v_mfma_scale_f32_16x16x128_f8f6f4 a[124:127], v[162:167], v[234:239], a[124:127], v20, v23 op_sel:[1,1,0] op_sel_hi:[1,1,0] cbsz:2 blgp:2
	ds_read_b64 v[112:113], v33 offset:1024
	v_mfma_scale_f32_16x16x128_f8f6f4 a[120:123], v[162:167], v[228:233], a[120:123], v20, v23 op_sel:[1,0,0] op_sel_hi:[1,1,0] cbsz:2 blgp:2
	s_add_u32 m0, s32, 0x1f000
	buffer_load_dwordx4 v3, s[40:43], 0 offen lds
	v_mfma_scale_f32_16x16x128_f8f6f4 a[116:119], v[162:167], v[222:227], a[116:119], v20, v23 op_sel:[1,1,0] op_sel_hi:[1,0,0] cbsz:2 blgp:2
	v_mfma_scale_f32_16x16x128_f8f6f4 a[112:115], v[162:167], v[216:221], a[112:115], v20, v23 op_sel:[1,0,0] op_sel_hi:[1,0,0] cbsz:2 blgp:2
	ds_read_b128 v[66:69], v24 offset:3072
	v_mfma_scale_f32_16x16x128_f8f6f4 a[108:111], v[162:167], v[210:215], a[108:111], v20, v22 op_sel:[1,1,0] op_sel_hi:[1,1,0] cbsz:2 blgp:2
	ds_read_b64 v[70:71], v30 offset:1536
	v_mfma_scale_f32_16x16x128_f8f6f4 a[104:107], v[162:167], v[204:209], a[104:107], v20, v22 op_sel:[1,0,0] op_sel_hi:[1,1,0] cbsz:2 blgp:2
	buffer_load_dwordx4 v3, s[40:43], 0 offen offset:1024 lds
	v_mfma_scale_f32_16x16x128_f8f6f4 a[100:103], v[162:167], v[198:203], a[100:103], v20, v22 op_sel:[1,1,0] op_sel_hi:[1,0,0] cbsz:2 blgp:2
	ds_read_b128 v[114:117], v27 offset:3072
	v_mfma_scale_f32_16x16x128_f8f6f4 a[96:99], v[162:167], v[192:197], a[96:99], v20, v22 op_sel:[1,0,0] op_sel_hi:[1,0,0] cbsz:2 blgp:2
	v_mfma_scale_f32_16x16x128_f8f6f4 a[128:131], v[168:173], v[192:197], a[128:131], v21, v22 op_sel_hi:[0,0,0] cbsz:2 blgp:2
	ds_read_b64 v[118:119], v33 offset:1536
	v_mfma_scale_f32_16x16x128_f8f6f4 a[132:135], v[168:173], v[198:203], a[132:135], v21, v22 op_sel:[0,1,0] op_sel_hi:[0,0,0] cbsz:2 blgp:2
	buffer_load_dwordx4 v3, s[40:43], 0 offen offset:2048 lds
	v_mfma_scale_f32_16x16x128_f8f6f4 a[136:139], v[168:173], v[204:209], a[136:139], v21, v22 op_sel_hi:[0,1,0] cbsz:2 blgp:2
	ds_read_b128 v[72:75], v24 offset:4096
	v_mfma_scale_f32_16x16x128_f8f6f4 a[140:143], v[168:173], v[210:215], a[140:143], v21, v22 op_sel:[0,1,0] op_sel_hi:[0,1,0] cbsz:2 blgp:2
	s_add_u32 m0, s32, 0x1fc00
	buffer_load_dwordx4 v8, s[40:43], 0 offen lds
	v_mfma_scale_f32_16x16x128_f8f6f4 a[144:147], v[168:173], v[216:221], a[144:147], v21, v23 op_sel_hi:[0,0,0] cbsz:2 blgp:2
	ds_read_b64 v[76:77], v30 offset:2048
	v_mfma_scale_f32_16x16x128_f8f6f4 a[148:151], v[168:173], v[222:227], a[148:151], v21, v23 op_sel:[0,1,0] op_sel_hi:[0,0,0] cbsz:2 blgp:2
	v_mfma_scale_f32_16x16x128_f8f6f4 a[152:155], v[168:173], v[228:233], a[152:155], v21, v23 op_sel_hi:[0,1,0] cbsz:2 blgp:2
	ds_read_b128 v[120:123], v27 offset:4096
	v_mfma_scale_f32_16x16x128_f8f6f4 a[156:159], v[168:173], v[234:239], a[156:159], v21, v23 op_sel:[0,1,0] op_sel_hi:[0,1,0] cbsz:2 blgp:2
	buffer_load_dwordx4 v8, s[40:43], 0 offen offset:1024 lds
	v_mfma_scale_f32_16x16x128_f8f6f4 a[188:191], v[174:179], v[234:239], a[188:191], v21, v23 op_sel:[1,1,0] op_sel_hi:[0,1,0] cbsz:2 blgp:2
	ds_read_b64 v[124:125], v33 offset:2048
	v_mfma_scale_f32_16x16x128_f8f6f4 a[184:187], v[174:179], v[228:233], a[184:187], v21, v23 op_sel:[1,0,0] op_sel_hi:[0,1,0] cbsz:2 blgp:2
	ds_read_b128 v[78:81], v24 offset:5120
	v_mfma_scale_f32_16x16x128_f8f6f4 a[180:183], v[174:179], v[222:227], a[180:183], v21, v23 op_sel:[1,1,0] op_sel_hi:[0,0,0] cbsz:2 blgp:2
	v_mfma_scale_f32_16x16x128_f8f6f4 a[176:179], v[174:179], v[216:221], a[176:179], v21, v23 op_sel:[1,0,0] op_sel_hi:[0,0,0] cbsz:2 blgp:2
	buffer_load_dwordx4 v8, s[40:43], 0 offen offset:2048 lds
	v_mfma_scale_f32_16x16x128_f8f6f4 a[172:175], v[174:179], v[210:215], a[172:175], v21, v22 op_sel:[1,1,0] op_sel_hi:[0,1,0] cbsz:2 blgp:2
	ds_read_b64 v[82:83], v30 offset:2560
	v_mfma_scale_f32_16x16x128_f8f6f4 a[168:171], v[174:179], v[204:209], a[168:171], v21, v22 op_sel:[1,0,0] op_sel_hi:[0,1,0] cbsz:2 blgp:2
	ds_read_b128 v[126:129], v27 offset:5120
	v_mfma_scale_f32_16x16x128_f8f6f4 a[164:167], v[174:179], v[198:203], a[164:167], v21, v22 op_sel:[1,1,0] op_sel_hi:[0,0,0] cbsz:2 blgp:2
	s_add_u32 s40, s40, 0x6000
	s_addc_u32 s41, s41, 0
	v_mfma_scale_f32_16x16x128_f8f6f4 a[160:163], v[174:179], v[192:197], a[160:163], v21, v22 op_sel:[1,0,0] op_sel_hi:[0,0,0] cbsz:2 blgp:2
	v_mfma_scale_f32_16x16x128_f8f6f4 a[192:195], v[180:185], v[192:197], a[192:195], v21, v22 op_sel_hi:[1,0,0] cbsz:2 blgp:2
	ds_read_b64 v[130:131], v33 offset:2560
	v_mfma_scale_f32_16x16x128_f8f6f4 a[196:199], v[180:185], v[198:203], a[196:199], v21, v22 op_sel:[0,1,0] op_sel_hi:[1,0,0] cbsz:2 blgp:2
	ds_read_b128 v[84:87], v24 offset:6144
	v_mfma_scale_f32_16x16x128_f8f6f4 a[200:203], v[180:185], v[204:209], a[200:203], v21, v22 op_sel_hi:[1,1,0] cbsz:2 blgp:2
	s_add_u32 m0, s33, 0x25000
	buffer_load_dword v4, s[44:47], 0 offen lds
	v_mfma_scale_f32_16x16x128_f8f6f4 a[204:207], v[180:185], v[210:215], a[204:207], v21, v22 op_sel:[0,1,0] op_sel_hi:[1,1,0] cbsz:2 blgp:2
	ds_read_b64 v[88:89], v30 offset:3072
	v_mfma_scale_f32_16x16x128_f8f6f4 a[208:211], v[180:185], v[216:221], a[208:211], v21, v23 op_sel_hi:[1,0,0] cbsz:2 blgp:2
	ds_read_b128 v[132:135], v27 offset:6144
	v_mfma_scale_f32_16x16x128_f8f6f4 a[212:215], v[180:185], v[222:227], a[212:215], v21, v23 op_sel:[0,1,0] op_sel_hi:[1,0,0] cbsz:2 blgp:2
	v_mfma_scale_f32_16x16x128_f8f6f4 a[216:219], v[180:185], v[228:233], a[216:219], v21, v23 op_sel_hi:[1,1,0] cbsz:2 blgp:2
	s_add_u32 s44, s44, 0x4000
	s_addc_u32 s45, s45, 0
	v_mfma_scale_f32_16x16x128_f8f6f4 a[220:223], v[180:185], v[234:239], a[220:223], v21, v23 op_sel:[0,1,0] op_sel_hi:[1,1,0] cbsz:2 blgp:2
	ds_read_b64 v[136:137], v33 offset:3072
	v_mfma_scale_f32_16x16x128_f8f6f4 a[252:255], v[186:191], v[234:239], a[252:255], v21, v23 op_sel:[1,1,0] op_sel_hi:[1,1,0] cbsz:2 blgp:2
	ds_read_b128 v[90:93], v24 offset:7168
	v_mfma_scale_f32_16x16x128_f8f6f4 a[248:251], v[186:191], v[228:233], a[248:251], v21, v23 op_sel:[1,0,0] op_sel_hi:[1,1,0] cbsz:2 blgp:2
	s_add_u32 m0, s33, 0x25400
	buffer_load_dword v4, s[48:51], 0 offen lds
	v_mfma_scale_f32_16x16x128_f8f6f4 a[244:247], v[186:191], v[222:227], a[244:247], v21, v23 op_sel:[1,1,0] op_sel_hi:[1,0,0] cbsz:2 blgp:2
	v_mfma_scale_f32_16x16x128_f8f6f4 a[240:243], v[186:191], v[216:221], a[240:243], v21, v23 op_sel:[1,0,0] op_sel_hi:[1,0,0] cbsz:2 blgp:2
	ds_read_b64 v[94:95], v30 offset:3584
	v_mfma_scale_f32_16x16x128_f8f6f4 a[236:239], v[186:191], v[210:215], a[236:239], v21, v22 op_sel:[1,1,0] op_sel_hi:[1,1,0] cbsz:2 blgp:2
	ds_read_b128 v[138:141], v27 offset:7168
	v_mfma_scale_f32_16x16x128_f8f6f4 a[232:235], v[186:191], v[204:209], a[232:235], v21, v22 op_sel:[1,0,0] op_sel_hi:[1,1,0] cbsz:2 blgp:2
	s_add_u32 s48, s48, 0x4000
	s_addc_u32 s49, s49, 0
	v_mfma_scale_f32_16x16x128_f8f6f4 a[228:231], v[186:191], v[198:203], a[228:231], v21, v22 op_sel:[1,1,0] op_sel_hi:[1,0,0] cbsz:2 blgp:2
	ds_read_b64 v[142:143], v33 offset:3584
	v_mfma_scale_f32_16x16x128_f8f6f4 a[224:227], v[186:191], v[192:197], a[224:227], v21, v22 op_sel:[1,0,0] op_sel_hi:[1,0,0] cbsz:2 blgp:2
	s_sub_u32 s31, s31, 1
	s_cmp_gt_i32 s31, 0
	s_cbranch_scc1 63261
	s_waitcnt vmcnt(14)
	s_waitcnt lgkmcnt(0)
	s_barrier
	v_mfma_scale_f32_16x16x128_f8f6f4 a[0:3], v[48:53], v[96:101], a[0:3], v16, v18 op_sel_hi:[0,0,0] cbsz:2 blgp:2
	ds_read_b64 v[20:21], v37
	v_mfma_scale_f32_16x16x128_f8f6f4 a[4:7], v[48:53], v[102:107], a[4:7], v16, v18 op_sel:[0,1,0] op_sel_hi:[0,0,0] cbsz:2 blgp:2
	s_add_u32 m0, s32, 0
	buffer_load_dwordx4 v3, s[36:39], 0 offen lds
	v_mfma_scale_f32_16x16x128_f8f6f4 a[8:11], v[48:53], v[108:113], a[8:11], v16, v18 op_sel_hi:[0,1,0] cbsz:2 blgp:2
	ds_read_b64 v[22:23], v40
	v_mfma_scale_f32_16x16x128_f8f6f4 a[12:15], v[48:53], v[114:119], a[12:15], v16, v18 op_sel:[0,1,0] op_sel_hi:[0,1,0] cbsz:2 blgp:2
	buffer_load_dwordx4 v3, s[36:39], 0 offen offset:1024 lds
	v_mfma_scale_f32_16x16x128_f8f6f4 a[16:19], v[48:53], v[120:125], a[16:19], v16, v19 op_sel_hi:[0,0,0] cbsz:2 blgp:2
	ds_read_b128 v[144:147], v25
	v_mfma_scale_f32_16x16x128_f8f6f4 a[20:23], v[48:53], v[126:131], a[20:23], v16, v19 op_sel:[0,1,0] op_sel_hi:[0,0,0] cbsz:2 blgp:2
	v_mfma_scale_f32_16x16x128_f8f6f4 a[24:27], v[48:53], v[132:137], a[24:27], v16, v19 op_sel_hi:[0,1,0] cbsz:2 blgp:2
	ds_read_b64 v[148:149], v31
	v_mfma_scale_f32_16x16x128_f8f6f4 a[28:31], v[48:53], v[138:143], a[28:31], v16, v19 op_sel:[0,1,0] op_sel_hi:[0,1,0] cbsz:2 blgp:2
	buffer_load_dwordx4 v3, s[36:39], 0 offen offset:2048 lds
	v_mfma_scale_f32_16x16x128_f8f6f4 a[60:63], v[54:59], v[138:143], a[60:63], v16, v19 op_sel:[1,1,0] op_sel_hi:[0,1,0] cbsz:2 blgp:2
	ds_read_b128 v[192:195], v28
	v_mfma_scale_f32_16x16x128_f8f6f4 a[56:59], v[54:59], v[132:137], a[56:59], v16, v19 op_sel:[1,0,0] op_sel_hi:[0,1,0] cbsz:2 blgp:2
	ds_read_b64 v[196:197], v34
	v_mfma_scale_f32_16x16x128_f8f6f4 a[52:55], v[54:59], v[126:131], a[52:55], v16, v19 op_sel:[1,1,0] op_sel_hi:[0,0,0] cbsz:2 blgp:2
	v_mfma_scale_f32_16x16x128_f8f6f4 a[48:51], v[54:59], v[120:125], a[48:51], v16, v19 op_sel:[1,0,0] op_sel_hi:[0,0,0] cbsz:2 blgp:2
	s_add_u32 m0, s32, 0xc00
	buffer_load_dwordx4 v8, s[36:39], 0 offen lds
	v_mfma_scale_f32_16x16x128_f8f6f4 a[44:47], v[54:59], v[114:119], a[44:47], v16, v18 op_sel:[1,1,0] op_sel_hi:[0,1,0] cbsz:2 blgp:2
	ds_read_b128 v[150:153], v25 offset:1024
	v_mfma_scale_f32_16x16x128_f8f6f4 a[40:43], v[54:59], v[108:113], a[40:43], v16, v18 op_sel:[1,0,0] op_sel_hi:[0,1,0] cbsz:2 blgp:2
	ds_read_b64 v[154:155], v31 offset:512
	v_mfma_scale_f32_16x16x128_f8f6f4 a[36:39], v[54:59], v[102:107], a[36:39], v16, v18 op_sel:[1,1,0] op_sel_hi:[0,0,0] cbsz:2 blgp:2
	buffer_load_dwordx4 v8, s[36:39], 0 offen offset:1024 lds
	v_mfma_scale_f32_16x16x128_f8f6f4 a[32:35], v[54:59], v[96:101], a[32:35], v16, v18 op_sel:[1,0,0] op_sel_hi:[0,0,0] cbsz:2 blgp:2
	v_mfma_scale_f32_16x16x128_f8f6f4 a[64:67], v[60:65], v[96:101], a[64:67], v16, v18 op_sel_hi:[1,0,0] cbsz:2 blgp:2
	ds_read_b128 v[198:201], v28 offset:1024
	v_mfma_scale_f32_16x16x128_f8f6f4 a[68:71], v[60:65], v[102:107], a[68:71], v16, v18 op_sel:[0,1,0] op_sel_hi:[1,0,0] cbsz:2 blgp:2
	ds_read_b64 v[202:203], v34 offset:512
	v_mfma_scale_f32_16x16x128_f8f6f4 a[72:75], v[60:65], v[108:113], a[72:75], v16, v18 op_sel_hi:[1,1,0] cbsz:2 blgp:2
	buffer_load_dwordx4 v8, s[36:39], 0 offen offset:2048 lds
	v_mfma_scale_f32_16x16x128_f8f6f4 a[76:79], v[60:65], v[114:119], a[76:79], v16, v18 op_sel:[0,1,0] op_sel_hi:[1,1,0] cbsz:2 blgp:2
	ds_read_b128 v[156:159], v25 offset:2048
	v_mfma_scale_f32_16x16x128_f8f6f4 a[80:83], v[60:65], v[120:125], a[80:83], v16, v19 op_sel_hi:[1,0,0] cbsz:2 blgp:2
	ds_read_b64 v[160:161], v31 offset:1024
	v_mfma_scale_f32_16x16x128_f8f6f4 a[84:87], v[60:65], v[126:131], a[84:87], v16, v19 op_sel:[0,1,0] op_sel_hi:[1,0,0] cbsz:2 blgp:2
	v_mfma_scale_f32_16x16x128_f8f6f4 a[88:91], v[60:65], v[132:137], a[88:91], v16, v19 op_sel_hi:[1,1,0] cbsz:2 blgp:2
	s_add_u32 s36, s36, 0x6000
	s_addc_u32 s37, s37, 0
	v_mfma_scale_f32_16x16x128_f8f6f4 a[92:95], v[60:65], v[138:143], a[92:95], v16, v19 op_sel:[0,1,0] op_sel_hi:[1,1,0] cbsz:2 blgp:2
	ds_read_b128 v[204:207], v28 offset:2048
	v_mfma_scale_f32_16x16x128_f8f6f4 a[124:127], v[66:71], v[138:143], a[124:127], v16, v19 op_sel:[1,1,0] op_sel_hi:[1,1,0] cbsz:2 blgp:2
	ds_read_b64 v[208:209], v34 offset:1024
	v_mfma_scale_f32_16x16x128_f8f6f4 a[120:123], v[66:71], v[132:137], a[120:123], v16, v19 op_sel:[1,0,0] op_sel_hi:[1,1,0] cbsz:2 blgp:2
	s_add_u32 m0, s32, 0x6000
	buffer_load_dwordx4 v3, s[40:43], 0 offen lds
	v_mfma_scale_f32_16x16x128_f8f6f4 a[116:119], v[66:71], v[126:131], a[116:119], v16, v19 op_sel:[1,1,0] op_sel_hi:[1,0,0] cbsz:2 blgp:2
	v_mfma_scale_f32_16x16x128_f8f6f4 a[112:115], v[66:71], v[120:125], a[112:115], v16, v19 op_sel:[1,0,0] op_sel_hi:[1,0,0] cbsz:2 blgp:2
	ds_read_b128 v[162:165], v25 offset:3072
	v_mfma_scale_f32_16x16x128_f8f6f4 a[108:111], v[66:71], v[114:119], a[108:111], v16, v18 op_sel:[1,1,0] op_sel_hi:[1,1,0] cbsz:2 blgp:2
	ds_read_b64 v[166:167], v31 offset:1536
	v_mfma_scale_f32_16x16x128_f8f6f4 a[104:107], v[66:71], v[108:113], a[104:107], v16, v18 op_sel:[1,0,0] op_sel_hi:[1,1,0] cbsz:2 blgp:2
	buffer_load_dwordx4 v3, s[40:43], 0 offen offset:1024 lds
	v_mfma_scale_f32_16x16x128_f8f6f4 a[100:103], v[66:71], v[102:107], a[100:103], v16, v18 op_sel:[1,1,0] op_sel_hi:[1,0,0] cbsz:2 blgp:2
	ds_read_b128 v[210:213], v28 offset:3072
	v_mfma_scale_f32_16x16x128_f8f6f4 a[96:99], v[66:71], v[96:101], a[96:99], v16, v18 op_sel:[1,0,0] op_sel_hi:[1,0,0] cbsz:2 blgp:2
	v_mfma_scale_f32_16x16x128_f8f6f4 a[128:131], v[72:77], v[96:101], a[128:131], v17, v18 op_sel_hi:[0,0,0] cbsz:2 blgp:2
	ds_read_b64 v[214:215], v34 offset:1536
	v_mfma_scale_f32_16x16x128_f8f6f4 a[132:135], v[72:77], v[102:107], a[132:135], v17, v18 op_sel:[0,1,0] op_sel_hi:[0,0,0] cbsz:2 blgp:2
	buffer_load_dwordx4 v3, s[40:43], 0 offen offset:2048 lds
	v_mfma_scale_f32_16x16x128_f8f6f4 a[136:139], v[72:77], v[108:113], a[136:139], v17, v18 op_sel_hi:[0,1,0] cbsz:2 blgp:2
	ds_read_b128 v[168:171], v25 offset:4096
	v_mfma_scale_f32_16x16x128_f8f6f4 a[140:143], v[72:77], v[114:119], a[140:143], v17, v18 op_sel:[0,1,0] op_sel_hi:[0,1,0] cbsz:2 blgp:2
	s_add_u32 m0, s32, 0x6c00
	buffer_load_dwordx4 v8, s[40:43], 0 offen lds
	v_mfma_scale_f32_16x16x128_f8f6f4 a[144:147], v[72:77], v[120:125], a[144:147], v17, v19 op_sel_hi:[0,0,0] cbsz:2 blgp:2
	ds_read_b64 v[172:173], v31 offset:2048
	v_mfma_scale_f32_16x16x128_f8f6f4 a[148:151], v[72:77], v[126:131], a[148:151], v17, v19 op_sel:[0,1,0] op_sel_hi:[0,0,0] cbsz:2 blgp:2
	v_mfma_scale_f32_16x16x128_f8f6f4 a[152:155], v[72:77], v[132:137], a[152:155], v17, v19 op_sel_hi:[0,1,0] cbsz:2 blgp:2
	ds_read_b128 v[216:219], v28 offset:4096
	v_mfma_scale_f32_16x16x128_f8f6f4 a[156:159], v[72:77], v[138:143], a[156:159], v17, v19 op_sel:[0,1,0] op_sel_hi:[0,1,0] cbsz:2 blgp:2
	buffer_load_dwordx4 v8, s[40:43], 0 offen offset:1024 lds
	v_mfma_scale_f32_16x16x128_f8f6f4 a[188:191], v[78:83], v[138:143], a[188:191], v17, v19 op_sel:[1,1,0] op_sel_hi:[0,1,0] cbsz:2 blgp:2
	ds_read_b64 v[220:221], v34 offset:2048
	v_mfma_scale_f32_16x16x128_f8f6f4 a[184:187], v[78:83], v[132:137], a[184:187], v17, v19 op_sel:[1,0,0] op_sel_hi:[0,1,0] cbsz:2 blgp:2
	ds_read_b128 v[174:177], v25 offset:5120
	v_mfma_scale_f32_16x16x128_f8f6f4 a[180:183], v[78:83], v[126:131], a[180:183], v17, v19 op_sel:[1,1,0] op_sel_hi:[0,0,0] cbsz:2 blgp:2
	v_mfma_scale_f32_16x16x128_f8f6f4 a[176:179], v[78:83], v[120:125], a[176:179], v17, v19 op_sel:[1,0,0] op_sel_hi:[0,0,0] cbsz:2 blgp:2
	buffer_load_dwordx4 v8, s[40:43], 0 offen offset:2048 lds
	v_mfma_scale_f32_16x16x128_f8f6f4 a[172:175], v[78:83], v[114:119], a[172:175], v17, v18 op_sel:[1,1,0] op_sel_hi:[0,1,0] cbsz:2 blgp:2
	ds_read_b64 v[178:179], v31 offset:2560
	v_mfma_scale_f32_16x16x128_f8f6f4 a[168:171], v[78:83], v[108:113], a[168:171], v17, v18 op_sel:[1,0,0] op_sel_hi:[0,1,0] cbsz:2 blgp:2
	ds_read_b128 v[222:225], v28 offset:5120
	v_mfma_scale_f32_16x16x128_f8f6f4 a[164:167], v[78:83], v[102:107], a[164:167], v17, v18 op_sel:[1,1,0] op_sel_hi:[0,0,0] cbsz:2 blgp:2
	s_add_u32 s40, s40, 0x6000
	s_addc_u32 s41, s41, 0
	v_mfma_scale_f32_16x16x128_f8f6f4 a[160:163], v[78:83], v[96:101], a[160:163], v17, v18 op_sel:[1,0,0] op_sel_hi:[0,0,0] cbsz:2 blgp:2
	v_mfma_scale_f32_16x16x128_f8f6f4 a[192:195], v[84:89], v[96:101], a[192:195], v17, v18 op_sel_hi:[1,0,0] cbsz:2 blgp:2
	ds_read_b64 v[226:227], v34 offset:2560
	v_mfma_scale_f32_16x16x128_f8f6f4 a[196:199], v[84:89], v[102:107], a[196:199], v17, v18 op_sel:[0,1,0] op_sel_hi:[1,0,0] cbsz:2 blgp:2
	ds_read_b128 v[180:183], v25 offset:6144
	v_mfma_scale_f32_16x16x128_f8f6f4 a[200:203], v[84:89], v[108:113], a[200:203], v17, v18 op_sel_hi:[1,1,0] cbsz:2 blgp:2
	s_add_u32 m0, s33, 0xc000
	buffer_load_dword v4, s[44:47], 0 offen lds
	v_mfma_scale_f32_16x16x128_f8f6f4 a[204:207], v[84:89], v[114:119], a[204:207], v17, v18 op_sel:[0,1,0] op_sel_hi:[1,1,0] cbsz:2 blgp:2
	ds_read_b64 v[184:185], v31 offset:3072
	v_mfma_scale_f32_16x16x128_f8f6f4 a[208:211], v[84:89], v[120:125], a[208:211], v17, v19 op_sel_hi:[1,0,0] cbsz:2 blgp:2
	ds_read_b128 v[228:231], v28 offset:6144
	v_mfma_scale_f32_16x16x128_f8f6f4 a[212:215], v[84:89], v[126:131], a[212:215], v17, v19 op_sel:[0,1,0] op_sel_hi:[1,0,0] cbsz:2 blgp:2
	v_mfma_scale_f32_16x16x128_f8f6f4 a[216:219], v[84:89], v[132:137], a[216:219], v17, v19 op_sel_hi:[1,1,0] cbsz:2 blgp:2
	s_add_u32 s44, s44, 0x4000
	s_addc_u32 s45, s45, 0
	v_mfma_scale_f32_16x16x128_f8f6f4 a[220:223], v[84:89], v[138:143], a[220:223], v17, v19 op_sel:[0,1,0] op_sel_hi:[1,1,0] cbsz:2 blgp:2
	ds_read_b64 v[232:233], v34 offset:3072
	v_mfma_scale_f32_16x16x128_f8f6f4 a[252:255], v[90:95], v[138:143], a[252:255], v17, v19 op_sel:[1,1,0] op_sel_hi:[1,1,0] cbsz:2 blgp:2
	ds_read_b128 v[186:189], v25 offset:7168
	v_mfma_scale_f32_16x16x128_f8f6f4 a[248:251], v[90:95], v[132:137], a[248:251], v17, v19 op_sel:[1,0,0] op_sel_hi:[1,1,0] cbsz:2 blgp:2
	s_add_u32 m0, s33, 0xc400
	buffer_load_dword v4, s[48:51], 0 offen lds
	v_mfma_scale_f32_16x16x128_f8f6f4 a[244:247], v[90:95], v[126:131], a[244:247], v17, v19 op_sel:[1,1,0] op_sel_hi:[1,0,0] cbsz:2 blgp:2
	v_mfma_scale_f32_16x16x128_f8f6f4 a[240:243], v[90:95], v[120:125], a[240:243], v17, v19 op_sel:[1,0,0] op_sel_hi:[1,0,0] cbsz:2 blgp:2
	ds_read_b64 v[190:191], v31 offset:3584
	v_mfma_scale_f32_16x16x128_f8f6f4 a[236:239], v[90:95], v[114:119], a[236:239], v17, v18 op_sel:[1,1,0] op_sel_hi:[1,1,0] cbsz:2 blgp:2
	ds_read_b128 v[234:237], v28 offset:7168
	v_mfma_scale_f32_16x16x128_f8f6f4 a[232:235], v[90:95], v[108:113], a[232:235], v17, v18 op_sel:[1,0,0] op_sel_hi:[1,1,0] cbsz:2 blgp:2
	s_add_u32 s48, s48, 0x4000
	s_addc_u32 s49, s49, 0
	v_mfma_scale_f32_16x16x128_f8f6f4 a[228:231], v[90:95], v[102:107], a[228:231], v17, v18 op_sel:[1,1,0] op_sel_hi:[1,0,0] cbsz:2 blgp:2
	ds_read_b64 v[238:239], v34 offset:3584
	v_mfma_scale_f32_16x16x128_f8f6f4 a[224:227], v[90:95], v[96:101], a[224:227], v17, v18 op_sel:[1,0,0] op_sel_hi:[1,0,0] cbsz:2 blgp:2
	s_waitcnt vmcnt(14)
	s_waitcnt lgkmcnt(0)
	s_barrier
	v_mfma_scale_f32_16x16x128_f8f6f4 a[0:3], v[144:149], v[192:197], a[0:3], v20, v22 op_sel_hi:[0,0,0] cbsz:2 blgp:2
	ds_read_b64 v[16:17], v38
	v_mfma_scale_f32_16x16x128_f8f6f4 a[4:7], v[144:149], v[198:203], a[4:7], v20, v22 op_sel:[0,1,0] op_sel_hi:[0,0,0] cbsz:2 blgp:2
	s_add_u32 m0, s32, 0xc800
	buffer_load_dwordx4 v3, s[36:39], 0 offen lds
	v_mfma_scale_f32_16x16x128_f8f6f4 a[8:11], v[144:149], v[204:209], a[8:11], v20, v22 op_sel_hi:[0,1,0] cbsz:2 blgp:2
	ds_read_b64 v[18:19], v41
	v_mfma_scale_f32_16x16x128_f8f6f4 a[12:15], v[144:149], v[210:215], a[12:15], v20, v22 op_sel:[0,1,0] op_sel_hi:[0,1,0] cbsz:2 blgp:2
	buffer_load_dwordx4 v3, s[36:39], 0 offen offset:1024 lds
	v_mfma_scale_f32_16x16x128_f8f6f4 a[16:19], v[144:149], v[216:221], a[16:19], v20, v23 op_sel_hi:[0,0,0] cbsz:2 blgp:2
	ds_read_b128 v[48:51], v26
	v_mfma_scale_f32_16x16x128_f8f6f4 a[20:23], v[144:149], v[222:227], a[20:23], v20, v23 op_sel:[0,1,0] op_sel_hi:[0,0,0] cbsz:2 blgp:2
	v_mfma_scale_f32_16x16x128_f8f6f4 a[24:27], v[144:149], v[228:233], a[24:27], v20, v23 op_sel_hi:[0,1,0] cbsz:2 blgp:2
	ds_read_b64 v[52:53], v32
	v_mfma_scale_f32_16x16x128_f8f6f4 a[28:31], v[144:149], v[234:239], a[28:31], v20, v23 op_sel:[0,1,0] op_sel_hi:[0,1,0] cbsz:2 blgp:2
	buffer_load_dwordx4 v3, s[36:39], 0 offen offset:2048 lds
	v_mfma_scale_f32_16x16x128_f8f6f4 a[60:63], v[150:155], v[234:239], a[60:63], v20, v23 op_sel:[1,1,0] op_sel_hi:[0,1,0] cbsz:2 blgp:2
	ds_read_b128 v[96:99], v29
	v_mfma_scale_f32_16x16x128_f8f6f4 a[56:59], v[150:155], v[228:233], a[56:59], v20, v23 op_sel:[1,0,0] op_sel_hi:[0,1,0] cbsz:2 blgp:2
	ds_read_b64 v[100:101], v35
	v_mfma_scale_f32_16x16x128_f8f6f4 a[52:55], v[150:155], v[222:227], a[52:55], v20, v23 op_sel:[1,1,0] op_sel_hi:[0,0,0] cbsz:2 blgp:2
	v_mfma_scale_f32_16x16x128_f8f6f4 a[48:51], v[150:155], v[216:221], a[48:51], v20, v23 op_sel:[1,0,0] op_sel_hi:[0,0,0] cbsz:2 blgp:2
	s_add_u32 m0, s32, 0xd400
	buffer_load_dwordx4 v8, s[36:39], 0 offen lds
	v_mfma_scale_f32_16x16x128_f8f6f4 a[44:47], v[150:155], v[210:215], a[44:47], v20, v22 op_sel:[1,1,0] op_sel_hi:[0,1,0] cbsz:2 blgp:2
	ds_read_b128 v[54:57], v26 offset:1024
	v_mfma_scale_f32_16x16x128_f8f6f4 a[40:43], v[150:155], v[204:209], a[40:43], v20, v22 op_sel:[1,0,0] op_sel_hi:[0,1,0] cbsz:2 blgp:2
	ds_read_b64 v[58:59], v32 offset:512
	v_mfma_scale_f32_16x16x128_f8f6f4 a[36:39], v[150:155], v[198:203], a[36:39], v20, v22 op_sel:[1,1,0] op_sel_hi:[0,0,0] cbsz:2 blgp:2
	buffer_load_dwordx4 v8, s[36:39], 0 offen offset:1024 lds
	v_mfma_scale_f32_16x16x128_f8f6f4 a[32:35], v[150:155], v[192:197], a[32:35], v20, v22 op_sel:[1,0,0] op_sel_hi:[0,0,0] cbsz:2 blgp:2
	v_mfma_scale_f32_16x16x128_f8f6f4 a[64:67], v[156:161], v[192:197], a[64:67], v20, v22 op_sel_hi:[1,0,0] cbsz:2 blgp:2
	ds_read_b128 v[102:105], v29 offset:1024
	v_mfma_scale_f32_16x16x128_f8f6f4 a[68:71], v[156:161], v[198:203], a[68:71], v20, v22 op_sel:[0,1,0] op_sel_hi:[1,0,0] cbsz:2 blgp:2
	ds_read_b64 v[106:107], v35 offset:512
	v_mfma_scale_f32_16x16x128_f8f6f4 a[72:75], v[156:161], v[204:209], a[72:75], v20, v22 op_sel_hi:[1,1,0] cbsz:2 blgp:2
	buffer_load_dwordx4 v8, s[36:39], 0 offen offset:2048 lds
	v_mfma_scale_f32_16x16x128_f8f6f4 a[76:79], v[156:161], v[210:215], a[76:79], v20, v22 op_sel:[0,1,0] op_sel_hi:[1,1,0] cbsz:2 blgp:2
	ds_read_b128 v[60:63], v26 offset:2048
	v_mfma_scale_f32_16x16x128_f8f6f4 a[80:83], v[156:161], v[216:221], a[80:83], v20, v23 op_sel_hi:[1,0,0] cbsz:2 blgp:2
	ds_read_b64 v[64:65], v32 offset:1024
	v_mfma_scale_f32_16x16x128_f8f6f4 a[84:87], v[156:161], v[222:227], a[84:87], v20, v23 op_sel:[0,1,0] op_sel_hi:[1,0,0] cbsz:2 blgp:2
	v_mfma_scale_f32_16x16x128_f8f6f4 a[88:91], v[156:161], v[228:233], a[88:91], v20, v23 op_sel_hi:[1,1,0] cbsz:2 blgp:2
	s_add_u32 s36, s36, 0x6000
	s_addc_u32 s37, s37, 0
	v_mfma_scale_f32_16x16x128_f8f6f4 a[92:95], v[156:161], v[234:239], a[92:95], v20, v23 op_sel:[0,1,0] op_sel_hi:[1,1,0] cbsz:2 blgp:2
	ds_read_b128 v[108:111], v29 offset:2048
	v_mfma_scale_f32_16x16x128_f8f6f4 a[124:127], v[162:167], v[234:239], a[124:127], v20, v23 op_sel:[1,1,0] op_sel_hi:[1,1,0] cbsz:2 blgp:2
	ds_read_b64 v[112:113], v35 offset:1024
	v_mfma_scale_f32_16x16x128_f8f6f4 a[120:123], v[162:167], v[228:233], a[120:123], v20, v23 op_sel:[1,0,0] op_sel_hi:[1,1,0] cbsz:2 blgp:2
	s_add_u32 m0, s32, 0x12800
	buffer_load_dwordx4 v3, s[40:43], 0 offen lds
	v_mfma_scale_f32_16x16x128_f8f6f4 a[116:119], v[162:167], v[222:227], a[116:119], v20, v23 op_sel:[1,1,0] op_sel_hi:[1,0,0] cbsz:2 blgp:2
	v_mfma_scale_f32_16x16x128_f8f6f4 a[112:115], v[162:167], v[216:221], a[112:115], v20, v23 op_sel:[1,0,0] op_sel_hi:[1,0,0] cbsz:2 blgp:2
	ds_read_b128 v[66:69], v26 offset:3072
	v_mfma_scale_f32_16x16x128_f8f6f4 a[108:111], v[162:167], v[210:215], a[108:111], v20, v22 op_sel:[1,1,0] op_sel_hi:[1,1,0] cbsz:2 blgp:2
	ds_read_b64 v[70:71], v32 offset:1536
	v_mfma_scale_f32_16x16x128_f8f6f4 a[104:107], v[162:167], v[204:209], a[104:107], v20, v22 op_sel:[1,0,0] op_sel_hi:[1,1,0] cbsz:2 blgp:2
	buffer_load_dwordx4 v3, s[40:43], 0 offen offset:1024 lds
	v_mfma_scale_f32_16x16x128_f8f6f4 a[100:103], v[162:167], v[198:203], a[100:103], v20, v22 op_sel:[1,1,0] op_sel_hi:[1,0,0] cbsz:2 blgp:2
	ds_read_b128 v[114:117], v29 offset:3072
	v_mfma_scale_f32_16x16x128_f8f6f4 a[96:99], v[162:167], v[192:197], a[96:99], v20, v22 op_sel:[1,0,0] op_sel_hi:[1,0,0] cbsz:2 blgp:2
	v_mfma_scale_f32_16x16x128_f8f6f4 a[128:131], v[168:173], v[192:197], a[128:131], v21, v22 op_sel_hi:[0,0,0] cbsz:2 blgp:2
	ds_read_b64 v[118:119], v35 offset:1536
	v_mfma_scale_f32_16x16x128_f8f6f4 a[132:135], v[168:173], v[198:203], a[132:135], v21, v22 op_sel:[0,1,0] op_sel_hi:[0,0,0] cbsz:2 blgp:2
	buffer_load_dwordx4 v3, s[40:43], 0 offen offset:2048 lds
	v_mfma_scale_f32_16x16x128_f8f6f4 a[136:139], v[168:173], v[204:209], a[136:139], v21, v22 op_sel_hi:[0,1,0] cbsz:2 blgp:2
	ds_read_b128 v[72:75], v26 offset:4096
	v_mfma_scale_f32_16x16x128_f8f6f4 a[140:143], v[168:173], v[210:215], a[140:143], v21, v22 op_sel:[0,1,0] op_sel_hi:[0,1,0] cbsz:2 blgp:2
	s_add_u32 m0, s32, 0x13400
	buffer_load_dwordx4 v8, s[40:43], 0 offen lds
	v_mfma_scale_f32_16x16x128_f8f6f4 a[144:147], v[168:173], v[216:221], a[144:147], v21, v23 op_sel_hi:[0,0,0] cbsz:2 blgp:2
	ds_read_b64 v[76:77], v32 offset:2048
	v_mfma_scale_f32_16x16x128_f8f6f4 a[148:151], v[168:173], v[222:227], a[148:151], v21, v23 op_sel:[0,1,0] op_sel_hi:[0,0,0] cbsz:2 blgp:2
	v_mfma_scale_f32_16x16x128_f8f6f4 a[152:155], v[168:173], v[228:233], a[152:155], v21, v23 op_sel_hi:[0,1,0] cbsz:2 blgp:2
	ds_read_b128 v[120:123], v29 offset:4096
	v_mfma_scale_f32_16x16x128_f8f6f4 a[156:159], v[168:173], v[234:239], a[156:159], v21, v23 op_sel:[0,1,0] op_sel_hi:[0,1,0] cbsz:2 blgp:2
	buffer_load_dwordx4 v8, s[40:43], 0 offen offset:1024 lds
	v_mfma_scale_f32_16x16x128_f8f6f4 a[188:191], v[174:179], v[234:239], a[188:191], v21, v23 op_sel:[1,1,0] op_sel_hi:[0,1,0] cbsz:2 blgp:2
	ds_read_b64 v[124:125], v35 offset:2048
	v_mfma_scale_f32_16x16x128_f8f6f4 a[184:187], v[174:179], v[228:233], a[184:187], v21, v23 op_sel:[1,0,0] op_sel_hi:[0,1,0] cbsz:2 blgp:2
	ds_read_b128 v[78:81], v26 offset:5120
	v_mfma_scale_f32_16x16x128_f8f6f4 a[180:183], v[174:179], v[222:227], a[180:183], v21, v23 op_sel:[1,1,0] op_sel_hi:[0,0,0] cbsz:2 blgp:2
	v_mfma_scale_f32_16x16x128_f8f6f4 a[176:179], v[174:179], v[216:221], a[176:179], v21, v23 op_sel:[1,0,0] op_sel_hi:[0,0,0] cbsz:2 blgp:2
	buffer_load_dwordx4 v8, s[40:43], 0 offen offset:2048 lds
	v_mfma_scale_f32_16x16x128_f8f6f4 a[172:175], v[174:179], v[210:215], a[172:175], v21, v22 op_sel:[1,1,0] op_sel_hi:[0,1,0] cbsz:2 blgp:2
	ds_read_b64 v[82:83], v32 offset:2560
	v_mfma_scale_f32_16x16x128_f8f6f4 a[168:171], v[174:179], v[204:209], a[168:171], v21, v22 op_sel:[1,0,0] op_sel_hi:[0,1,0] cbsz:2 blgp:2
	ds_read_b128 v[126:129], v29 offset:5120
	v_mfma_scale_f32_16x16x128_f8f6f4 a[164:167], v[174:179], v[198:203], a[164:167], v21, v22 op_sel:[1,1,0] op_sel_hi:[0,0,0] cbsz:2 blgp:2
	s_add_u32 s40, s40, 0x6000
	s_addc_u32 s41, s41, 0
	v_mfma_scale_f32_16x16x128_f8f6f4 a[160:163], v[174:179], v[192:197], a[160:163], v21, v22 op_sel:[1,0,0] op_sel_hi:[0,0,0] cbsz:2 blgp:2
	v_mfma_scale_f32_16x16x128_f8f6f4 a[192:195], v[180:185], v[192:197], a[192:195], v21, v22 op_sel_hi:[1,0,0] cbsz:2 blgp:2
	ds_read_b64 v[130:131], v35 offset:2560
	v_mfma_scale_f32_16x16x128_f8f6f4 a[196:199], v[180:185], v[198:203], a[196:199], v21, v22 op_sel:[0,1,0] op_sel_hi:[1,0,0] cbsz:2 blgp:2
	ds_read_b128 v[84:87], v26 offset:6144
	v_mfma_scale_f32_16x16x128_f8f6f4 a[200:203], v[180:185], v[204:209], a[200:203], v21, v22 op_sel_hi:[1,1,0] cbsz:2 blgp:2
	s_add_u32 m0, s33, 0x18800
	buffer_load_dword v4, s[44:47], 0 offen lds
	v_mfma_scale_f32_16x16x128_f8f6f4 a[204:207], v[180:185], v[210:215], a[204:207], v21, v22 op_sel:[0,1,0] op_sel_hi:[1,1,0] cbsz:2 blgp:2
	ds_read_b64 v[88:89], v32 offset:3072
	v_mfma_scale_f32_16x16x128_f8f6f4 a[208:211], v[180:185], v[216:221], a[208:211], v21, v23 op_sel_hi:[1,0,0] cbsz:2 blgp:2
	ds_read_b128 v[132:135], v29 offset:6144
	v_mfma_scale_f32_16x16x128_f8f6f4 a[212:215], v[180:185], v[222:227], a[212:215], v21, v23 op_sel:[0,1,0] op_sel_hi:[1,0,0] cbsz:2 blgp:2
	v_mfma_scale_f32_16x16x128_f8f6f4 a[216:219], v[180:185], v[228:233], a[216:219], v21, v23 op_sel_hi:[1,1,0] cbsz:2 blgp:2
	s_add_u32 s44, s44, 0x4000
	s_addc_u32 s45, s45, 0
	v_mfma_scale_f32_16x16x128_f8f6f4 a[220:223], v[180:185], v[234:239], a[220:223], v21, v23 op_sel:[0,1,0] op_sel_hi:[1,1,0] cbsz:2 blgp:2
	ds_read_b64 v[136:137], v35 offset:3072
	v_mfma_scale_f32_16x16x128_f8f6f4 a[252:255], v[186:191], v[234:239], a[252:255], v21, v23 op_sel:[1,1,0] op_sel_hi:[1,1,0] cbsz:2 blgp:2
	ds_read_b128 v[90:93], v26 offset:7168
	v_mfma_scale_f32_16x16x128_f8f6f4 a[248:251], v[186:191], v[228:233], a[248:251], v21, v23 op_sel:[1,0,0] op_sel_hi:[1,1,0] cbsz:2 blgp:2
	s_add_u32 m0, s33, 0x18c00
	buffer_load_dword v4, s[48:51], 0 offen lds
	v_mfma_scale_f32_16x16x128_f8f6f4 a[244:247], v[186:191], v[222:227], a[244:247], v21, v23 op_sel:[1,1,0] op_sel_hi:[1,0,0] cbsz:2 blgp:2
	v_mfma_scale_f32_16x16x128_f8f6f4 a[240:243], v[186:191], v[216:221], a[240:243], v21, v23 op_sel:[1,0,0] op_sel_hi:[1,0,0] cbsz:2 blgp:2
	ds_read_b64 v[94:95], v32 offset:3584
	v_mfma_scale_f32_16x16x128_f8f6f4 a[236:239], v[186:191], v[210:215], a[236:239], v21, v22 op_sel:[1,1,0] op_sel_hi:[1,1,0] cbsz:2 blgp:2
	ds_read_b128 v[138:141], v29 offset:7168
	v_mfma_scale_f32_16x16x128_f8f6f4 a[232:235], v[186:191], v[204:209], a[232:235], v21, v22 op_sel:[1,0,0] op_sel_hi:[1,1,0] cbsz:2 blgp:2
	s_add_u32 s48, s48, 0x4000
	s_addc_u32 s49, s49, 0
	v_mfma_scale_f32_16x16x128_f8f6f4 a[228:231], v[186:191], v[198:203], a[228:231], v21, v22 op_sel:[1,1,0] op_sel_hi:[1,0,0] cbsz:2 blgp:2
	ds_read_b64 v[142:143], v35 offset:3584
	v_mfma_scale_f32_16x16x128_f8f6f4 a[224:227], v[186:191], v[192:197], a[224:227], v21, v22 op_sel:[1,0,0] op_sel_hi:[1,0,0] cbsz:2 blgp:2
	s_waitcnt lgkmcnt(0)
	s_nop 15
	s_nop 15
	s_nop 15
	s_nop 15
	v_lshrrev_b32_e32 v2, 4, v1
	v_and_b32_e32 v3, 15, v1
	v_mul_u32_u24_e32 v2, 0x8000, v2
	v_lshlrev_b32_e32 v3, 1, v3
	v_add_u32_e32 v7, v2, v3
	s_mul_i32 s30, s23, 0x200000
	s_add_u32 s8, s8, s30
	s_addc_u32 s9, s9, 0
	s_mul_i32 s30, s21, 0x100000
	s_add_u32 s8, s8, s30
	s_addc_u32 s9, s9, 0
	s_mul_i32 s30, s24, 0x200
	s_add_u32 s8, s8, s30
	s_addc_u32 s9, s9, 0
	s_mul_i32 s30, s22, 0x100
	s_add_u32 s8, s8, s30
	s_addc_u32 s9, s9, 0
	v_add_u32_e32 v10, 0, v7
	v_accvgpr_read_b32 v11, a0
	s_nop 2
	v_cvt_pk_bf16_f32 v12, v11, v11
	global_store_short v10, v12, s[8:9]
	v_add_u32_e32 v10, 0x2000, v7
	v_accvgpr_read_b32 v11, a1
	s_nop 2
	v_cvt_pk_bf16_f32 v12, v11, v11
	global_store_short v10, v12, s[8:9]
	v_add_u32_e32 v10, 0x4000, v7
	v_accvgpr_read_b32 v11, a2
	s_nop 2
	v_cvt_pk_bf16_f32 v12, v11, v11
	global_store_short v10, v12, s[8:9]
	v_add_u32_e32 v10, 0x6000, v7
	v_accvgpr_read_b32 v11, a3
	s_nop 2
	v_cvt_pk_bf16_f32 v12, v11, v11
	global_store_short v10, v12, s[8:9]
	v_add_u32_e32 v10, 32, v7
	v_accvgpr_read_b32 v11, a4
	s_nop 2
	v_cvt_pk_bf16_f32 v12, v11, v11
	global_store_short v10, v12, s[8:9]
	v_add_u32_e32 v10, 0x2020, v7
	v_accvgpr_read_b32 v11, a5
	s_nop 2
	v_cvt_pk_bf16_f32 v12, v11, v11
	global_store_short v10, v12, s[8:9]
	v_add_u32_e32 v10, 0x4020, v7
	v_accvgpr_read_b32 v11, a6
	s_nop 2
	v_cvt_pk_bf16_f32 v12, v11, v11
	global_store_short v10, v12, s[8:9]
	v_add_u32_e32 v10, 0x6020, v7
	v_accvgpr_read_b32 v11, a7
	s_nop 2
	v_cvt_pk_bf16_f32 v12, v11, v11
	global_store_short v10, v12, s[8:9]
	v_add_u32_e32 v10, 64, v7
	v_accvgpr_read_b32 v11, a8
	s_nop 2
	v_cvt_pk_bf16_f32 v12, v11, v11
	global_store_short v10, v12, s[8:9]
	v_add_u32_e32 v10, 0x2040, v7
	v_accvgpr_read_b32 v11, a9
	s_nop 2
	v_cvt_pk_bf16_f32 v12, v11, v11
	global_store_short v10, v12, s[8:9]
	v_add_u32_e32 v10, 0x4040, v7
	v_accvgpr_read_b32 v11, a10
	s_nop 2
	v_cvt_pk_bf16_f32 v12, v11, v11
	global_store_short v10, v12, s[8:9]
	v_add_u32_e32 v10, 0x6040, v7
	v_accvgpr_read_b32 v11, a11
	s_nop 2
	v_cvt_pk_bf16_f32 v12, v11, v11
	global_store_short v10, v12, s[8:9]
	v_add_u32_e32 v10, 0x60, v7
	v_accvgpr_read_b32 v11, a12
	s_nop 2
	v_cvt_pk_bf16_f32 v12, v11, v11
	global_store_short v10, v12, s[8:9]
	v_add_u32_e32 v10, 0x2060, v7
	v_accvgpr_read_b32 v11, a13
	s_nop 2
	v_cvt_pk_bf16_f32 v12, v11, v11
	global_store_short v10, v12, s[8:9]
	v_add_u32_e32 v10, 0x4060, v7
	v_accvgpr_read_b32 v11, a14
	s_nop 2
	v_cvt_pk_bf16_f32 v12, v11, v11
	global_store_short v10, v12, s[8:9]
	v_add_u32_e32 v10, 0x6060, v7
	v_accvgpr_read_b32 v11, a15
	s_nop 2
	v_cvt_pk_bf16_f32 v12, v11, v11
	global_store_short v10, v12, s[8:9]
	v_add_u32_e32 v10, 0x80, v7
	v_accvgpr_read_b32 v11, a16
	s_nop 2
	v_cvt_pk_bf16_f32 v12, v11, v11
	global_store_short v10, v12, s[8:9]
	v_add_u32_e32 v10, 0x2080, v7
	v_accvgpr_read_b32 v11, a17
	s_nop 2
	v_cvt_pk_bf16_f32 v12, v11, v11
	global_store_short v10, v12, s[8:9]
	v_add_u32_e32 v10, 0x4080, v7
	v_accvgpr_read_b32 v11, a18
	s_nop 2
	v_cvt_pk_bf16_f32 v12, v11, v11
	global_store_short v10, v12, s[8:9]
	v_add_u32_e32 v10, 0x6080, v7
	v_accvgpr_read_b32 v11, a19
	s_nop 2
	v_cvt_pk_bf16_f32 v12, v11, v11
	global_store_short v10, v12, s[8:9]
	v_add_u32_e32 v10, 0xa0, v7
	v_accvgpr_read_b32 v11, a20
	s_nop 2
	v_cvt_pk_bf16_f32 v12, v11, v11
	global_store_short v10, v12, s[8:9]
	v_add_u32_e32 v10, 0x20a0, v7
	v_accvgpr_read_b32 v11, a21
	s_nop 2
	v_cvt_pk_bf16_f32 v12, v11, v11
	global_store_short v10, v12, s[8:9]
	v_add_u32_e32 v10, 0x40a0, v7
	v_accvgpr_read_b32 v11, a22
	s_nop 2
	v_cvt_pk_bf16_f32 v12, v11, v11
	global_store_short v10, v12, s[8:9]
	v_add_u32_e32 v10, 0x60a0, v7
	v_accvgpr_read_b32 v11, a23
	s_nop 2
	v_cvt_pk_bf16_f32 v12, v11, v11
	global_store_short v10, v12, s[8:9]
	v_add_u32_e32 v10, 0xc0, v7
	v_accvgpr_read_b32 v11, a24
	s_nop 2
	v_cvt_pk_bf16_f32 v12, v11, v11
	global_store_short v10, v12, s[8:9]
	v_add_u32_e32 v10, 0x20c0, v7
	v_accvgpr_read_b32 v11, a25
	s_nop 2
	v_cvt_pk_bf16_f32 v12, v11, v11
	global_store_short v10, v12, s[8:9]
	v_add_u32_e32 v10, 0x40c0, v7
	v_accvgpr_read_b32 v11, a26
	s_nop 2
	v_cvt_pk_bf16_f32 v12, v11, v11
	global_store_short v10, v12, s[8:9]
	v_add_u32_e32 v10, 0x60c0, v7
	v_accvgpr_read_b32 v11, a27
	s_nop 2
	v_cvt_pk_bf16_f32 v12, v11, v11
	global_store_short v10, v12, s[8:9]
	v_add_u32_e32 v10, 0xe0, v7
	v_accvgpr_read_b32 v11, a28
	s_nop 2
	v_cvt_pk_bf16_f32 v12, v11, v11
	global_store_short v10, v12, s[8:9]
	v_add_u32_e32 v10, 0x20e0, v7
	v_accvgpr_read_b32 v11, a29
	s_nop 2
	v_cvt_pk_bf16_f32 v12, v11, v11
	global_store_short v10, v12, s[8:9]
	v_add_u32_e32 v10, 0x40e0, v7
	v_accvgpr_read_b32 v11, a30
	s_nop 2
	v_cvt_pk_bf16_f32 v12, v11, v11
	global_store_short v10, v12, s[8:9]
	v_add_u32_e32 v10, 0x60e0, v7
	v_accvgpr_read_b32 v11, a31
	s_nop 2
	v_cvt_pk_bf16_f32 v12, v11, v11
	global_store_short v10, v12, s[8:9]
	v_add_u32_e32 v10, 0x20000, v7
	v_accvgpr_read_b32 v11, a32
	s_nop 2
	v_cvt_pk_bf16_f32 v12, v11, v11
	global_store_short v10, v12, s[8:9]
	v_add_u32_e32 v10, 0x22000, v7
	v_accvgpr_read_b32 v11, a33
	s_nop 2
	v_cvt_pk_bf16_f32 v12, v11, v11
	global_store_short v10, v12, s[8:9]
	v_add_u32_e32 v10, 0x24000, v7
	v_accvgpr_read_b32 v11, a34
	s_nop 2
	v_cvt_pk_bf16_f32 v12, v11, v11
	global_store_short v10, v12, s[8:9]
	v_add_u32_e32 v10, 0x26000, v7
	v_accvgpr_read_b32 v11, a35
	s_nop 2
	v_cvt_pk_bf16_f32 v12, v11, v11
	global_store_short v10, v12, s[8:9]
	v_add_u32_e32 v10, 0x20020, v7
	v_accvgpr_read_b32 v11, a36
	s_nop 2
	v_cvt_pk_bf16_f32 v12, v11, v11
	global_store_short v10, v12, s[8:9]
	v_add_u32_e32 v10, 0x22020, v7
	v_accvgpr_read_b32 v11, a37
	s_nop 2
	v_cvt_pk_bf16_f32 v12, v11, v11
	global_store_short v10, v12, s[8:9]
	v_add_u32_e32 v10, 0x24020, v7
	v_accvgpr_read_b32 v11, a38
	s_nop 2
	v_cvt_pk_bf16_f32 v12, v11, v11
	global_store_short v10, v12, s[8:9]
	v_add_u32_e32 v10, 0x26020, v7
	v_accvgpr_read_b32 v11, a39
	s_nop 2
	v_cvt_pk_bf16_f32 v12, v11, v11
	global_store_short v10, v12, s[8:9]
	v_add_u32_e32 v10, 0x20040, v7
	v_accvgpr_read_b32 v11, a40
	s_nop 2
	v_cvt_pk_bf16_f32 v12, v11, v11
	global_store_short v10, v12, s[8:9]
	v_add_u32_e32 v10, 0x22040, v7
	v_accvgpr_read_b32 v11, a41
	s_nop 2
	v_cvt_pk_bf16_f32 v12, v11, v11
	global_store_short v10, v12, s[8:9]
	v_add_u32_e32 v10, 0x24040, v7
	v_accvgpr_read_b32 v11, a42
	s_nop 2
	v_cvt_pk_bf16_f32 v12, v11, v11
	global_store_short v10, v12, s[8:9]
	v_add_u32_e32 v10, 0x26040, v7
	v_accvgpr_read_b32 v11, a43
	s_nop 2
	v_cvt_pk_bf16_f32 v12, v11, v11
	global_store_short v10, v12, s[8:9]
	v_add_u32_e32 v10, 0x20060, v7
	v_accvgpr_read_b32 v11, a44
	s_nop 2
	v_cvt_pk_bf16_f32 v12, v11, v11
	global_store_short v10, v12, s[8:9]
	v_add_u32_e32 v10, 0x22060, v7
	v_accvgpr_read_b32 v11, a45
	s_nop 2
	v_cvt_pk_bf16_f32 v12, v11, v11
	global_store_short v10, v12, s[8:9]
	v_add_u32_e32 v10, 0x24060, v7
	v_accvgpr_read_b32 v11, a46
	s_nop 2
	v_cvt_pk_bf16_f32 v12, v11, v11
	global_store_short v10, v12, s[8:9]
	v_add_u32_e32 v10, 0x26060, v7
	v_accvgpr_read_b32 v11, a47
	s_nop 2
	v_cvt_pk_bf16_f32 v12, v11, v11
	global_store_short v10, v12, s[8:9]
	v_add_u32_e32 v10, 0x20080, v7
	v_accvgpr_read_b32 v11, a48
	s_nop 2
	v_cvt_pk_bf16_f32 v12, v11, v11
	global_store_short v10, v12, s[8:9]
	v_add_u32_e32 v10, 0x22080, v7
	v_accvgpr_read_b32 v11, a49
	s_nop 2
	v_cvt_pk_bf16_f32 v12, v11, v11
	global_store_short v10, v12, s[8:9]
	v_add_u32_e32 v10, 0x24080, v7
	v_accvgpr_read_b32 v11, a50
	s_nop 2
	v_cvt_pk_bf16_f32 v12, v11, v11
	global_store_short v10, v12, s[8:9]
	v_add_u32_e32 v10, 0x26080, v7
	v_accvgpr_read_b32 v11, a51
	s_nop 2
	v_cvt_pk_bf16_f32 v12, v11, v11
	global_store_short v10, v12, s[8:9]
	v_add_u32_e32 v10, 0x200a0, v7
	v_accvgpr_read_b32 v11, a52
	s_nop 2
	v_cvt_pk_bf16_f32 v12, v11, v11
	global_store_short v10, v12, s[8:9]
	v_add_u32_e32 v10, 0x220a0, v7
	v_accvgpr_read_b32 v11, a53
	s_nop 2
	v_cvt_pk_bf16_f32 v12, v11, v11
	global_store_short v10, v12, s[8:9]
	v_add_u32_e32 v10, 0x240a0, v7
	v_accvgpr_read_b32 v11, a54
	s_nop 2
	v_cvt_pk_bf16_f32 v12, v11, v11
	global_store_short v10, v12, s[8:9]
	v_add_u32_e32 v10, 0x260a0, v7
	v_accvgpr_read_b32 v11, a55
	s_nop 2
	v_cvt_pk_bf16_f32 v12, v11, v11
	global_store_short v10, v12, s[8:9]
	v_add_u32_e32 v10, 0x200c0, v7
	v_accvgpr_read_b32 v11, a56
	s_nop 2
	v_cvt_pk_bf16_f32 v12, v11, v11
	global_store_short v10, v12, s[8:9]
	v_add_u32_e32 v10, 0x220c0, v7
	v_accvgpr_read_b32 v11, a57
	s_nop 2
	v_cvt_pk_bf16_f32 v12, v11, v11
	global_store_short v10, v12, s[8:9]
	v_add_u32_e32 v10, 0x240c0, v7
	v_accvgpr_read_b32 v11, a58
	s_nop 2
	v_cvt_pk_bf16_f32 v12, v11, v11
	global_store_short v10, v12, s[8:9]
	v_add_u32_e32 v10, 0x260c0, v7
	v_accvgpr_read_b32 v11, a59
	s_nop 2
	v_cvt_pk_bf16_f32 v12, v11, v11
	global_store_short v10, v12, s[8:9]
	v_add_u32_e32 v10, 0x200e0, v7
	v_accvgpr_read_b32 v11, a60
	s_nop 2
	v_cvt_pk_bf16_f32 v12, v11, v11
	global_store_short v10, v12, s[8:9]
	v_add_u32_e32 v10, 0x220e0, v7
	v_accvgpr_read_b32 v11, a61
	s_nop 2
	v_cvt_pk_bf16_f32 v12, v11, v11
	global_store_short v10, v12, s[8:9]
	v_add_u32_e32 v10, 0x240e0, v7
	v_accvgpr_read_b32 v11, a62
	s_nop 2
	v_cvt_pk_bf16_f32 v12, v11, v11
	global_store_short v10, v12, s[8:9]
	v_add_u32_e32 v10, 0x260e0, v7
	v_accvgpr_read_b32 v11, a63
	s_nop 2
	v_cvt_pk_bf16_f32 v12, v11, v11
	global_store_short v10, v12, s[8:9]
	v_add_u32_e32 v10, 0x40000, v7
	v_accvgpr_read_b32 v11, a64
	s_nop 2
	v_cvt_pk_bf16_f32 v12, v11, v11
	global_store_short v10, v12, s[8:9]
	v_add_u32_e32 v10, 0x42000, v7
	v_accvgpr_read_b32 v11, a65
	s_nop 2
	v_cvt_pk_bf16_f32 v12, v11, v11
	global_store_short v10, v12, s[8:9]
	v_add_u32_e32 v10, 0x44000, v7
	v_accvgpr_read_b32 v11, a66
	s_nop 2
	v_cvt_pk_bf16_f32 v12, v11, v11
	global_store_short v10, v12, s[8:9]
	v_add_u32_e32 v10, 0x46000, v7
	v_accvgpr_read_b32 v11, a67
	s_nop 2
	v_cvt_pk_bf16_f32 v12, v11, v11
	global_store_short v10, v12, s[8:9]
	v_add_u32_e32 v10, 0x40020, v7
	v_accvgpr_read_b32 v11, a68
	s_nop 2
	v_cvt_pk_bf16_f32 v12, v11, v11
	global_store_short v10, v12, s[8:9]
	v_add_u32_e32 v10, 0x42020, v7
	v_accvgpr_read_b32 v11, a69
	s_nop 2
	v_cvt_pk_bf16_f32 v12, v11, v11
	global_store_short v10, v12, s[8:9]
	v_add_u32_e32 v10, 0x44020, v7
	v_accvgpr_read_b32 v11, a70
	s_nop 2
	v_cvt_pk_bf16_f32 v12, v11, v11
	global_store_short v10, v12, s[8:9]
	v_add_u32_e32 v10, 0x46020, v7
	v_accvgpr_read_b32 v11, a71
	s_nop 2
	v_cvt_pk_bf16_f32 v12, v11, v11
	global_store_short v10, v12, s[8:9]
	v_add_u32_e32 v10, 0x40040, v7
	v_accvgpr_read_b32 v11, a72
	s_nop 2
	v_cvt_pk_bf16_f32 v12, v11, v11
	global_store_short v10, v12, s[8:9]
	v_add_u32_e32 v10, 0x42040, v7
	v_accvgpr_read_b32 v11, a73
	s_nop 2
	v_cvt_pk_bf16_f32 v12, v11, v11
	global_store_short v10, v12, s[8:9]
	v_add_u32_e32 v10, 0x44040, v7
	v_accvgpr_read_b32 v11, a74
	s_nop 2
	v_cvt_pk_bf16_f32 v12, v11, v11
	global_store_short v10, v12, s[8:9]
	v_add_u32_e32 v10, 0x46040, v7
	v_accvgpr_read_b32 v11, a75
	s_nop 2
	v_cvt_pk_bf16_f32 v12, v11, v11
	global_store_short v10, v12, s[8:9]
	v_add_u32_e32 v10, 0x40060, v7
	v_accvgpr_read_b32 v11, a76
	s_nop 2
	v_cvt_pk_bf16_f32 v12, v11, v11
	global_store_short v10, v12, s[8:9]
	v_add_u32_e32 v10, 0x42060, v7
	v_accvgpr_read_b32 v11, a77
	s_nop 2
	v_cvt_pk_bf16_f32 v12, v11, v11
	global_store_short v10, v12, s[8:9]
	v_add_u32_e32 v10, 0x44060, v7
	v_accvgpr_read_b32 v11, a78
	s_nop 2
	v_cvt_pk_bf16_f32 v12, v11, v11
	global_store_short v10, v12, s[8:9]
	v_add_u32_e32 v10, 0x46060, v7
	v_accvgpr_read_b32 v11, a79
	s_nop 2
	v_cvt_pk_bf16_f32 v12, v11, v11
	global_store_short v10, v12, s[8:9]
	v_add_u32_e32 v10, 0x40080, v7
	v_accvgpr_read_b32 v11, a80
	s_nop 2
	v_cvt_pk_bf16_f32 v12, v11, v11
	global_store_short v10, v12, s[8:9]
	v_add_u32_e32 v10, 0x42080, v7
	v_accvgpr_read_b32 v11, a81
	s_nop 2
	v_cvt_pk_bf16_f32 v12, v11, v11
	global_store_short v10, v12, s[8:9]
	v_add_u32_e32 v10, 0x44080, v7
	v_accvgpr_read_b32 v11, a82
	s_nop 2
	v_cvt_pk_bf16_f32 v12, v11, v11
	global_store_short v10, v12, s[8:9]
	v_add_u32_e32 v10, 0x46080, v7
	v_accvgpr_read_b32 v11, a83
	s_nop 2
	v_cvt_pk_bf16_f32 v12, v11, v11
	global_store_short v10, v12, s[8:9]
	v_add_u32_e32 v10, 0x400a0, v7
	v_accvgpr_read_b32 v11, a84
	s_nop 2
	v_cvt_pk_bf16_f32 v12, v11, v11
	global_store_short v10, v12, s[8:9]
	v_add_u32_e32 v10, 0x420a0, v7
	v_accvgpr_read_b32 v11, a85
	s_nop 2
	v_cvt_pk_bf16_f32 v12, v11, v11
	global_store_short v10, v12, s[8:9]
	v_add_u32_e32 v10, 0x440a0, v7
	v_accvgpr_read_b32 v11, a86
	s_nop 2
	v_cvt_pk_bf16_f32 v12, v11, v11
	global_store_short v10, v12, s[8:9]
	v_add_u32_e32 v10, 0x460a0, v7
	v_accvgpr_read_b32 v11, a87
	s_nop 2
	v_cvt_pk_bf16_f32 v12, v11, v11
	global_store_short v10, v12, s[8:9]
	v_add_u32_e32 v10, 0x400c0, v7
	v_accvgpr_read_b32 v11, a88
	s_nop 2
	v_cvt_pk_bf16_f32 v12, v11, v11
	global_store_short v10, v12, s[8:9]
	v_add_u32_e32 v10, 0x420c0, v7
	v_accvgpr_read_b32 v11, a89
	s_nop 2
	v_cvt_pk_bf16_f32 v12, v11, v11
	global_store_short v10, v12, s[8:9]
	v_add_u32_e32 v10, 0x440c0, v7
	v_accvgpr_read_b32 v11, a90
	s_nop 2
	v_cvt_pk_bf16_f32 v12, v11, v11
	global_store_short v10, v12, s[8:9]
	v_add_u32_e32 v10, 0x460c0, v7
	v_accvgpr_read_b32 v11, a91
	s_nop 2
	v_cvt_pk_bf16_f32 v12, v11, v11
	global_store_short v10, v12, s[8:9]
	v_add_u32_e32 v10, 0x400e0, v7
	v_accvgpr_read_b32 v11, a92
	s_nop 2
	v_cvt_pk_bf16_f32 v12, v11, v11
	global_store_short v10, v12, s[8:9]
	v_add_u32_e32 v10, 0x420e0, v7
	v_accvgpr_read_b32 v11, a93
	s_nop 2
	v_cvt_pk_bf16_f32 v12, v11, v11
	global_store_short v10, v12, s[8:9]
	v_add_u32_e32 v10, 0x440e0, v7
	v_accvgpr_read_b32 v11, a94
	s_nop 2
	v_cvt_pk_bf16_f32 v12, v11, v11
	global_store_short v10, v12, s[8:9]
	v_add_u32_e32 v10, 0x460e0, v7
	v_accvgpr_read_b32 v11, a95
	s_nop 2
	v_cvt_pk_bf16_f32 v12, v11, v11
	global_store_short v10, v12, s[8:9]
	v_add_u32_e32 v10, 0x60000, v7
	v_accvgpr_read_b32 v11, a96
	s_nop 2
	v_cvt_pk_bf16_f32 v12, v11, v11
	global_store_short v10, v12, s[8:9]
	v_add_u32_e32 v10, 0x62000, v7
	v_accvgpr_read_b32 v11, a97
	s_nop 2
	v_cvt_pk_bf16_f32 v12, v11, v11
	global_store_short v10, v12, s[8:9]
	v_add_u32_e32 v10, 0x64000, v7
	v_accvgpr_read_b32 v11, a98
	s_nop 2
	v_cvt_pk_bf16_f32 v12, v11, v11
	global_store_short v10, v12, s[8:9]
	v_add_u32_e32 v10, 0x66000, v7
	v_accvgpr_read_b32 v11, a99
	s_nop 2
	v_cvt_pk_bf16_f32 v12, v11, v11
	global_store_short v10, v12, s[8:9]
	v_add_u32_e32 v10, 0x60020, v7
	v_accvgpr_read_b32 v11, a100
	s_nop 2
	v_cvt_pk_bf16_f32 v12, v11, v11
	global_store_short v10, v12, s[8:9]
	v_add_u32_e32 v10, 0x62020, v7
	v_accvgpr_read_b32 v11, a101
	s_nop 2
	v_cvt_pk_bf16_f32 v12, v11, v11
	global_store_short v10, v12, s[8:9]
	v_add_u32_e32 v10, 0x64020, v7
	v_accvgpr_read_b32 v11, a102
	s_nop 2
	v_cvt_pk_bf16_f32 v12, v11, v11
	global_store_short v10, v12, s[8:9]
	v_add_u32_e32 v10, 0x66020, v7
	v_accvgpr_read_b32 v11, a103
	s_nop 2
	v_cvt_pk_bf16_f32 v12, v11, v11
	global_store_short v10, v12, s[8:9]
	v_add_u32_e32 v10, 0x60040, v7
	v_accvgpr_read_b32 v11, a104
	s_nop 2
	v_cvt_pk_bf16_f32 v12, v11, v11
	global_store_short v10, v12, s[8:9]
	v_add_u32_e32 v10, 0x62040, v7
	v_accvgpr_read_b32 v11, a105
	s_nop 2
	v_cvt_pk_bf16_f32 v12, v11, v11
	global_store_short v10, v12, s[8:9]
	v_add_u32_e32 v10, 0x64040, v7
	v_accvgpr_read_b32 v11, a106
	s_nop 2
	v_cvt_pk_bf16_f32 v12, v11, v11
	global_store_short v10, v12, s[8:9]
	v_add_u32_e32 v10, 0x66040, v7
	v_accvgpr_read_b32 v11, a107
	s_nop 2
	v_cvt_pk_bf16_f32 v12, v11, v11
	global_store_short v10, v12, s[8:9]
	v_add_u32_e32 v10, 0x60060, v7
	v_accvgpr_read_b32 v11, a108
	s_nop 2
	v_cvt_pk_bf16_f32 v12, v11, v11
	global_store_short v10, v12, s[8:9]
	v_add_u32_e32 v10, 0x62060, v7
	v_accvgpr_read_b32 v11, a109
	s_nop 2
	v_cvt_pk_bf16_f32 v12, v11, v11
	global_store_short v10, v12, s[8:9]
	v_add_u32_e32 v10, 0x64060, v7
	v_accvgpr_read_b32 v11, a110
	s_nop 2
	v_cvt_pk_bf16_f32 v12, v11, v11
	global_store_short v10, v12, s[8:9]
	v_add_u32_e32 v10, 0x66060, v7
	v_accvgpr_read_b32 v11, a111
	s_nop 2
	v_cvt_pk_bf16_f32 v12, v11, v11
	global_store_short v10, v12, s[8:9]
	v_add_u32_e32 v10, 0x60080, v7
	v_accvgpr_read_b32 v11, a112
	s_nop 2
	v_cvt_pk_bf16_f32 v12, v11, v11
	global_store_short v10, v12, s[8:9]
	v_add_u32_e32 v10, 0x62080, v7
	v_accvgpr_read_b32 v11, a113
	s_nop 2
	v_cvt_pk_bf16_f32 v12, v11, v11
	global_store_short v10, v12, s[8:9]
	v_add_u32_e32 v10, 0x64080, v7
	v_accvgpr_read_b32 v11, a114
	s_nop 2
	v_cvt_pk_bf16_f32 v12, v11, v11
	global_store_short v10, v12, s[8:9]
	v_add_u32_e32 v10, 0x66080, v7
	v_accvgpr_read_b32 v11, a115
	s_nop 2
	v_cvt_pk_bf16_f32 v12, v11, v11
	global_store_short v10, v12, s[8:9]
	v_add_u32_e32 v10, 0x600a0, v7
	v_accvgpr_read_b32 v11, a116
	s_nop 2
	v_cvt_pk_bf16_f32 v12, v11, v11
	global_store_short v10, v12, s[8:9]
	v_add_u32_e32 v10, 0x620a0, v7
	v_accvgpr_read_b32 v11, a117
	s_nop 2
	v_cvt_pk_bf16_f32 v12, v11, v11
	global_store_short v10, v12, s[8:9]
	v_add_u32_e32 v10, 0x640a0, v7
	v_accvgpr_read_b32 v11, a118
	s_nop 2
	v_cvt_pk_bf16_f32 v12, v11, v11
	global_store_short v10, v12, s[8:9]
	v_add_u32_e32 v10, 0x660a0, v7
	v_accvgpr_read_b32 v11, a119
	s_nop 2
	v_cvt_pk_bf16_f32 v12, v11, v11
	global_store_short v10, v12, s[8:9]
	v_add_u32_e32 v10, 0x600c0, v7
	v_accvgpr_read_b32 v11, a120
	s_nop 2
	v_cvt_pk_bf16_f32 v12, v11, v11
	global_store_short v10, v12, s[8:9]
	v_add_u32_e32 v10, 0x620c0, v7
	v_accvgpr_read_b32 v11, a121
	s_nop 2
	v_cvt_pk_bf16_f32 v12, v11, v11
	global_store_short v10, v12, s[8:9]
	v_add_u32_e32 v10, 0x640c0, v7
	v_accvgpr_read_b32 v11, a122
	s_nop 2
	v_cvt_pk_bf16_f32 v12, v11, v11
	global_store_short v10, v12, s[8:9]
	v_add_u32_e32 v10, 0x660c0, v7
	v_accvgpr_read_b32 v11, a123
	s_nop 2
	v_cvt_pk_bf16_f32 v12, v11, v11
	global_store_short v10, v12, s[8:9]
	v_add_u32_e32 v10, 0x600e0, v7
	v_accvgpr_read_b32 v11, a124
	s_nop 2
	v_cvt_pk_bf16_f32 v12, v11, v11
	global_store_short v10, v12, s[8:9]
	v_add_u32_e32 v10, 0x620e0, v7
	v_accvgpr_read_b32 v11, a125
	s_nop 2
	v_cvt_pk_bf16_f32 v12, v11, v11
	global_store_short v10, v12, s[8:9]
	v_add_u32_e32 v10, 0x640e0, v7
	v_accvgpr_read_b32 v11, a126
	s_nop 2
	v_cvt_pk_bf16_f32 v12, v11, v11
	global_store_short v10, v12, s[8:9]
	v_add_u32_e32 v10, 0x660e0, v7
	v_accvgpr_read_b32 v11, a127
	s_nop 2
	v_cvt_pk_bf16_f32 v12, v11, v11
	global_store_short v10, v12, s[8:9]
	v_add_u32_e32 v10, 0x80000, v7
	v_accvgpr_read_b32 v11, a128
	s_nop 2
	v_cvt_pk_bf16_f32 v12, v11, v11
	global_store_short v10, v12, s[8:9]
	v_add_u32_e32 v10, 0x82000, v7
	v_accvgpr_read_b32 v11, a129
	s_nop 2
	v_cvt_pk_bf16_f32 v12, v11, v11
	global_store_short v10, v12, s[8:9]
	v_add_u32_e32 v10, 0x84000, v7
	v_accvgpr_read_b32 v11, a130
	s_nop 2
	v_cvt_pk_bf16_f32 v12, v11, v11
	global_store_short v10, v12, s[8:9]
	v_add_u32_e32 v10, 0x86000, v7
	v_accvgpr_read_b32 v11, a131
	s_nop 2
	v_cvt_pk_bf16_f32 v12, v11, v11
	global_store_short v10, v12, s[8:9]
	v_add_u32_e32 v10, 0x80020, v7
	v_accvgpr_read_b32 v11, a132
	s_nop 2
	v_cvt_pk_bf16_f32 v12, v11, v11
	global_store_short v10, v12, s[8:9]
	v_add_u32_e32 v10, 0x82020, v7
	v_accvgpr_read_b32 v11, a133
	s_nop 2
	v_cvt_pk_bf16_f32 v12, v11, v11
	global_store_short v10, v12, s[8:9]
	v_add_u32_e32 v10, 0x84020, v7
	v_accvgpr_read_b32 v11, a134
	s_nop 2
	v_cvt_pk_bf16_f32 v12, v11, v11
	global_store_short v10, v12, s[8:9]
	v_add_u32_e32 v10, 0x86020, v7
	v_accvgpr_read_b32 v11, a135
	s_nop 2
	v_cvt_pk_bf16_f32 v12, v11, v11
	global_store_short v10, v12, s[8:9]
	v_add_u32_e32 v10, 0x80040, v7
	v_accvgpr_read_b32 v11, a136
	s_nop 2
	v_cvt_pk_bf16_f32 v12, v11, v11
	global_store_short v10, v12, s[8:9]
	v_add_u32_e32 v10, 0x82040, v7
	v_accvgpr_read_b32 v11, a137
	s_nop 2
	v_cvt_pk_bf16_f32 v12, v11, v11
	global_store_short v10, v12, s[8:9]
	v_add_u32_e32 v10, 0x84040, v7
	v_accvgpr_read_b32 v11, a138
	s_nop 2
	v_cvt_pk_bf16_f32 v12, v11, v11
	global_store_short v10, v12, s[8:9]
	v_add_u32_e32 v10, 0x86040, v7
	v_accvgpr_read_b32 v11, a139
	s_nop 2
	v_cvt_pk_bf16_f32 v12, v11, v11
	global_store_short v10, v12, s[8:9]
	v_add_u32_e32 v10, 0x80060, v7
	v_accvgpr_read_b32 v11, a140
	s_nop 2
	v_cvt_pk_bf16_f32 v12, v11, v11
	global_store_short v10, v12, s[8:9]
	v_add_u32_e32 v10, 0x82060, v7
	v_accvgpr_read_b32 v11, a141
	s_nop 2
	v_cvt_pk_bf16_f32 v12, v11, v11
	global_store_short v10, v12, s[8:9]
	v_add_u32_e32 v10, 0x84060, v7
	v_accvgpr_read_b32 v11, a142
	s_nop 2
	v_cvt_pk_bf16_f32 v12, v11, v11
	global_store_short v10, v12, s[8:9]
	v_add_u32_e32 v10, 0x86060, v7
	v_accvgpr_read_b32 v11, a143
	s_nop 2
	v_cvt_pk_bf16_f32 v12, v11, v11
	global_store_short v10, v12, s[8:9]
	v_add_u32_e32 v10, 0x80080, v7
	v_accvgpr_read_b32 v11, a144
	s_nop 2
	v_cvt_pk_bf16_f32 v12, v11, v11
	global_store_short v10, v12, s[8:9]
	v_add_u32_e32 v10, 0x82080, v7
	v_accvgpr_read_b32 v11, a145
	s_nop 2
	v_cvt_pk_bf16_f32 v12, v11, v11
	global_store_short v10, v12, s[8:9]
	v_add_u32_e32 v10, 0x84080, v7
	v_accvgpr_read_b32 v11, a146
	s_nop 2
	v_cvt_pk_bf16_f32 v12, v11, v11
	global_store_short v10, v12, s[8:9]
	v_add_u32_e32 v10, 0x86080, v7
	v_accvgpr_read_b32 v11, a147
	s_nop 2
	v_cvt_pk_bf16_f32 v12, v11, v11
	global_store_short v10, v12, s[8:9]
	v_add_u32_e32 v10, 0x800a0, v7
	v_accvgpr_read_b32 v11, a148
	s_nop 2
	v_cvt_pk_bf16_f32 v12, v11, v11
	global_store_short v10, v12, s[8:9]
	v_add_u32_e32 v10, 0x820a0, v7
	v_accvgpr_read_b32 v11, a149
	s_nop 2
	v_cvt_pk_bf16_f32 v12, v11, v11
	global_store_short v10, v12, s[8:9]
	v_add_u32_e32 v10, 0x840a0, v7
	v_accvgpr_read_b32 v11, a150
	s_nop 2
	v_cvt_pk_bf16_f32 v12, v11, v11
	global_store_short v10, v12, s[8:9]
	v_add_u32_e32 v10, 0x860a0, v7
	v_accvgpr_read_b32 v11, a151
	s_nop 2
	v_cvt_pk_bf16_f32 v12, v11, v11
	global_store_short v10, v12, s[8:9]
	v_add_u32_e32 v10, 0x800c0, v7
	v_accvgpr_read_b32 v11, a152
	s_nop 2
	v_cvt_pk_bf16_f32 v12, v11, v11
	global_store_short v10, v12, s[8:9]
	v_add_u32_e32 v10, 0x820c0, v7
	v_accvgpr_read_b32 v11, a153
	s_nop 2
	v_cvt_pk_bf16_f32 v12, v11, v11
	global_store_short v10, v12, s[8:9]
	v_add_u32_e32 v10, 0x840c0, v7
	v_accvgpr_read_b32 v11, a154
	s_nop 2
	v_cvt_pk_bf16_f32 v12, v11, v11
	global_store_short v10, v12, s[8:9]
	v_add_u32_e32 v10, 0x860c0, v7
	v_accvgpr_read_b32 v11, a155
	s_nop 2
	v_cvt_pk_bf16_f32 v12, v11, v11
	global_store_short v10, v12, s[8:9]
	v_add_u32_e32 v10, 0x800e0, v7
	v_accvgpr_read_b32 v11, a156
	s_nop 2
	v_cvt_pk_bf16_f32 v12, v11, v11
	global_store_short v10, v12, s[8:9]
	v_add_u32_e32 v10, 0x820e0, v7
	v_accvgpr_read_b32 v11, a157
	s_nop 2
	v_cvt_pk_bf16_f32 v12, v11, v11
	global_store_short v10, v12, s[8:9]
	v_add_u32_e32 v10, 0x840e0, v7
	v_accvgpr_read_b32 v11, a158
	s_nop 2
	v_cvt_pk_bf16_f32 v12, v11, v11
	global_store_short v10, v12, s[8:9]
	v_add_u32_e32 v10, 0x860e0, v7
	v_accvgpr_read_b32 v11, a159
	s_nop 2
	v_cvt_pk_bf16_f32 v12, v11, v11
	global_store_short v10, v12, s[8:9]
	v_add_u32_e32 v10, 0xa0000, v7
	v_accvgpr_read_b32 v11, a160
	s_nop 2
	v_cvt_pk_bf16_f32 v12, v11, v11
	global_store_short v10, v12, s[8:9]
	v_add_u32_e32 v10, 0xa2000, v7
	v_accvgpr_read_b32 v11, a161
	s_nop 2
	v_cvt_pk_bf16_f32 v12, v11, v11
	global_store_short v10, v12, s[8:9]
	v_add_u32_e32 v10, 0xa4000, v7
	v_accvgpr_read_b32 v11, a162
	s_nop 2
	v_cvt_pk_bf16_f32 v12, v11, v11
	global_store_short v10, v12, s[8:9]
	v_add_u32_e32 v10, 0xa6000, v7
	v_accvgpr_read_b32 v11, a163
	s_nop 2
	v_cvt_pk_bf16_f32 v12, v11, v11
	global_store_short v10, v12, s[8:9]
	v_add_u32_e32 v10, 0xa0020, v7
	v_accvgpr_read_b32 v11, a164
	s_nop 2
	v_cvt_pk_bf16_f32 v12, v11, v11
	global_store_short v10, v12, s[8:9]
	v_add_u32_e32 v10, 0xa2020, v7
	v_accvgpr_read_b32 v11, a165
	s_nop 2
	v_cvt_pk_bf16_f32 v12, v11, v11
	global_store_short v10, v12, s[8:9]
	v_add_u32_e32 v10, 0xa4020, v7
	v_accvgpr_read_b32 v11, a166
	s_nop 2
	v_cvt_pk_bf16_f32 v12, v11, v11
	global_store_short v10, v12, s[8:9]
	v_add_u32_e32 v10, 0xa6020, v7
	v_accvgpr_read_b32 v11, a167
	s_nop 2
	v_cvt_pk_bf16_f32 v12, v11, v11
	global_store_short v10, v12, s[8:9]
	v_add_u32_e32 v10, 0xa0040, v7
	v_accvgpr_read_b32 v11, a168
	s_nop 2
	v_cvt_pk_bf16_f32 v12, v11, v11
	global_store_short v10, v12, s[8:9]
	v_add_u32_e32 v10, 0xa2040, v7
	v_accvgpr_read_b32 v11, a169
	s_nop 2
	v_cvt_pk_bf16_f32 v12, v11, v11
	global_store_short v10, v12, s[8:9]
	v_add_u32_e32 v10, 0xa4040, v7
	v_accvgpr_read_b32 v11, a170
	s_nop 2
	v_cvt_pk_bf16_f32 v12, v11, v11
	global_store_short v10, v12, s[8:9]
	v_add_u32_e32 v10, 0xa6040, v7
	v_accvgpr_read_b32 v11, a171
	s_nop 2
	v_cvt_pk_bf16_f32 v12, v11, v11
	global_store_short v10, v12, s[8:9]
	v_add_u32_e32 v10, 0xa0060, v7
	v_accvgpr_read_b32 v11, a172
	s_nop 2
	v_cvt_pk_bf16_f32 v12, v11, v11
	global_store_short v10, v12, s[8:9]
	v_add_u32_e32 v10, 0xa2060, v7
	v_accvgpr_read_b32 v11, a173
	s_nop 2
	v_cvt_pk_bf16_f32 v12, v11, v11
	global_store_short v10, v12, s[8:9]
	v_add_u32_e32 v10, 0xa4060, v7
	v_accvgpr_read_b32 v11, a174
	s_nop 2
	v_cvt_pk_bf16_f32 v12, v11, v11
	global_store_short v10, v12, s[8:9]
	v_add_u32_e32 v10, 0xa6060, v7
	v_accvgpr_read_b32 v11, a175
	s_nop 2
	v_cvt_pk_bf16_f32 v12, v11, v11
	global_store_short v10, v12, s[8:9]
	v_add_u32_e32 v10, 0xa0080, v7
	v_accvgpr_read_b32 v11, a176
	s_nop 2
	v_cvt_pk_bf16_f32 v12, v11, v11
	global_store_short v10, v12, s[8:9]
	v_add_u32_e32 v10, 0xa2080, v7
	v_accvgpr_read_b32 v11, a177
	s_nop 2
	v_cvt_pk_bf16_f32 v12, v11, v11
	global_store_short v10, v12, s[8:9]
	v_add_u32_e32 v10, 0xa4080, v7
	v_accvgpr_read_b32 v11, a178
	s_nop 2
	v_cvt_pk_bf16_f32 v12, v11, v11
	global_store_short v10, v12, s[8:9]
	v_add_u32_e32 v10, 0xa6080, v7
	v_accvgpr_read_b32 v11, a179
	s_nop 2
	v_cvt_pk_bf16_f32 v12, v11, v11
	global_store_short v10, v12, s[8:9]
	v_add_u32_e32 v10, 0xa00a0, v7
	v_accvgpr_read_b32 v11, a180
	s_nop 2
	v_cvt_pk_bf16_f32 v12, v11, v11
	global_store_short v10, v12, s[8:9]
	v_add_u32_e32 v10, 0xa20a0, v7
	v_accvgpr_read_b32 v11, a181
	s_nop 2
	v_cvt_pk_bf16_f32 v12, v11, v11
	global_store_short v10, v12, s[8:9]
	v_add_u32_e32 v10, 0xa40a0, v7
	v_accvgpr_read_b32 v11, a182
	s_nop 2
	v_cvt_pk_bf16_f32 v12, v11, v11
	global_store_short v10, v12, s[8:9]
	v_add_u32_e32 v10, 0xa60a0, v7
	v_accvgpr_read_b32 v11, a183
	s_nop 2
	v_cvt_pk_bf16_f32 v12, v11, v11
	global_store_short v10, v12, s[8:9]
	v_add_u32_e32 v10, 0xa00c0, v7
	v_accvgpr_read_b32 v11, a184
	s_nop 2
	v_cvt_pk_bf16_f32 v12, v11, v11
	global_store_short v10, v12, s[8:9]
	v_add_u32_e32 v10, 0xa20c0, v7
	v_accvgpr_read_b32 v11, a185
	s_nop 2
	v_cvt_pk_bf16_f32 v12, v11, v11
	global_store_short v10, v12, s[8:9]
	v_add_u32_e32 v10, 0xa40c0, v7
	v_accvgpr_read_b32 v11, a186
	s_nop 2
	v_cvt_pk_bf16_f32 v12, v11, v11
	global_store_short v10, v12, s[8:9]
	v_add_u32_e32 v10, 0xa60c0, v7
	v_accvgpr_read_b32 v11, a187
	s_nop 2
	v_cvt_pk_bf16_f32 v12, v11, v11
	global_store_short v10, v12, s[8:9]
	v_add_u32_e32 v10, 0xa00e0, v7
	v_accvgpr_read_b32 v11, a188
	s_nop 2
	v_cvt_pk_bf16_f32 v12, v11, v11
	global_store_short v10, v12, s[8:9]
	v_add_u32_e32 v10, 0xa20e0, v7
	v_accvgpr_read_b32 v11, a189
	s_nop 2
	v_cvt_pk_bf16_f32 v12, v11, v11
	global_store_short v10, v12, s[8:9]
	v_add_u32_e32 v10, 0xa40e0, v7
	v_accvgpr_read_b32 v11, a190
	s_nop 2
	v_cvt_pk_bf16_f32 v12, v11, v11
	global_store_short v10, v12, s[8:9]
	v_add_u32_e32 v10, 0xa60e0, v7
	v_accvgpr_read_b32 v11, a191
	s_nop 2
	v_cvt_pk_bf16_f32 v12, v11, v11
	global_store_short v10, v12, s[8:9]
	v_add_u32_e32 v10, 0xc0000, v7
	v_accvgpr_read_b32 v11, a192
	s_nop 2
	v_cvt_pk_bf16_f32 v12, v11, v11
	global_store_short v10, v12, s[8:9]
	v_add_u32_e32 v10, 0xc2000, v7
	v_accvgpr_read_b32 v11, a193
	s_nop 2
	v_cvt_pk_bf16_f32 v12, v11, v11
	global_store_short v10, v12, s[8:9]
	v_add_u32_e32 v10, 0xc4000, v7
	v_accvgpr_read_b32 v11, a194
	s_nop 2
	v_cvt_pk_bf16_f32 v12, v11, v11
	global_store_short v10, v12, s[8:9]
	v_add_u32_e32 v10, 0xc6000, v7
	v_accvgpr_read_b32 v11, a195
	s_nop 2
	v_cvt_pk_bf16_f32 v12, v11, v11
	global_store_short v10, v12, s[8:9]
	v_add_u32_e32 v10, 0xc0020, v7
	v_accvgpr_read_b32 v11, a196
	s_nop 2
	v_cvt_pk_bf16_f32 v12, v11, v11
	global_store_short v10, v12, s[8:9]
	v_add_u32_e32 v10, 0xc2020, v7
	v_accvgpr_read_b32 v11, a197
	s_nop 2
	v_cvt_pk_bf16_f32 v12, v11, v11
	global_store_short v10, v12, s[8:9]
	v_add_u32_e32 v10, 0xc4020, v7
	v_accvgpr_read_b32 v11, a198
	s_nop 2
	v_cvt_pk_bf16_f32 v12, v11, v11
	global_store_short v10, v12, s[8:9]
	v_add_u32_e32 v10, 0xc6020, v7
	v_accvgpr_read_b32 v11, a199
	s_nop 2
	v_cvt_pk_bf16_f32 v12, v11, v11
	global_store_short v10, v12, s[8:9]
	v_add_u32_e32 v10, 0xc0040, v7
	v_accvgpr_read_b32 v11, a200
	s_nop 2
	v_cvt_pk_bf16_f32 v12, v11, v11
	global_store_short v10, v12, s[8:9]
	v_add_u32_e32 v10, 0xc2040, v7
	v_accvgpr_read_b32 v11, a201
	s_nop 2
	v_cvt_pk_bf16_f32 v12, v11, v11
	global_store_short v10, v12, s[8:9]
	v_add_u32_e32 v10, 0xc4040, v7
	v_accvgpr_read_b32 v11, a202
	s_nop 2
	v_cvt_pk_bf16_f32 v12, v11, v11
	global_store_short v10, v12, s[8:9]
	v_add_u32_e32 v10, 0xc6040, v7
	v_accvgpr_read_b32 v11, a203
	s_nop 2
	v_cvt_pk_bf16_f32 v12, v11, v11
	global_store_short v10, v12, s[8:9]
	v_add_u32_e32 v10, 0xc0060, v7
	v_accvgpr_read_b32 v11, a204
	s_nop 2
	v_cvt_pk_bf16_f32 v12, v11, v11
	global_store_short v10, v12, s[8:9]
	v_add_u32_e32 v10, 0xc2060, v7
	v_accvgpr_read_b32 v11, a205
	s_nop 2
	v_cvt_pk_bf16_f32 v12, v11, v11
	global_store_short v10, v12, s[8:9]
	v_add_u32_e32 v10, 0xc4060, v7
	v_accvgpr_read_b32 v11, a206
	s_nop 2
	v_cvt_pk_bf16_f32 v12, v11, v11
	global_store_short v10, v12, s[8:9]
	v_add_u32_e32 v10, 0xc6060, v7
	v_accvgpr_read_b32 v11, a207
	s_nop 2
	v_cvt_pk_bf16_f32 v12, v11, v11
	global_store_short v10, v12, s[8:9]
	v_add_u32_e32 v10, 0xc0080, v7
	v_accvgpr_read_b32 v11, a208
	s_nop 2
	v_cvt_pk_bf16_f32 v12, v11, v11
	global_store_short v10, v12, s[8:9]
	v_add_u32_e32 v10, 0xc2080, v7
	v_accvgpr_read_b32 v11, a209
	s_nop 2
	v_cvt_pk_bf16_f32 v12, v11, v11
	global_store_short v10, v12, s[8:9]
	v_add_u32_e32 v10, 0xc4080, v7
	v_accvgpr_read_b32 v11, a210
	s_nop 2
	v_cvt_pk_bf16_f32 v12, v11, v11
	global_store_short v10, v12, s[8:9]
	v_add_u32_e32 v10, 0xc6080, v7
	v_accvgpr_read_b32 v11, a211
	s_nop 2
	v_cvt_pk_bf16_f32 v12, v11, v11
	global_store_short v10, v12, s[8:9]
	v_add_u32_e32 v10, 0xc00a0, v7
	v_accvgpr_read_b32 v11, a212
	s_nop 2
	v_cvt_pk_bf16_f32 v12, v11, v11
	global_store_short v10, v12, s[8:9]
	v_add_u32_e32 v10, 0xc20a0, v7
	v_accvgpr_read_b32 v11, a213
	s_nop 2
	v_cvt_pk_bf16_f32 v12, v11, v11
	global_store_short v10, v12, s[8:9]
	v_add_u32_e32 v10, 0xc40a0, v7
	v_accvgpr_read_b32 v11, a214
	s_nop 2
	v_cvt_pk_bf16_f32 v12, v11, v11
	global_store_short v10, v12, s[8:9]
	v_add_u32_e32 v10, 0xc60a0, v7
	v_accvgpr_read_b32 v11, a215
	s_nop 2
	v_cvt_pk_bf16_f32 v12, v11, v11
	global_store_short v10, v12, s[8:9]
	v_add_u32_e32 v10, 0xc00c0, v7
	v_accvgpr_read_b32 v11, a216
	s_nop 2
	v_cvt_pk_bf16_f32 v12, v11, v11
	global_store_short v10, v12, s[8:9]
	v_add_u32_e32 v10, 0xc20c0, v7
	v_accvgpr_read_b32 v11, a217
	s_nop 2
	v_cvt_pk_bf16_f32 v12, v11, v11
	global_store_short v10, v12, s[8:9]
	v_add_u32_e32 v10, 0xc40c0, v7
	v_accvgpr_read_b32 v11, a218
	s_nop 2
	v_cvt_pk_bf16_f32 v12, v11, v11
	global_store_short v10, v12, s[8:9]
	v_add_u32_e32 v10, 0xc60c0, v7
	v_accvgpr_read_b32 v11, a219
	s_nop 2
	v_cvt_pk_bf16_f32 v12, v11, v11
	global_store_short v10, v12, s[8:9]
	v_add_u32_e32 v10, 0xc00e0, v7
	v_accvgpr_read_b32 v11, a220
	s_nop 2
	v_cvt_pk_bf16_f32 v12, v11, v11
	global_store_short v10, v12, s[8:9]
	v_add_u32_e32 v10, 0xc20e0, v7
	v_accvgpr_read_b32 v11, a221
	s_nop 2
	v_cvt_pk_bf16_f32 v12, v11, v11
	global_store_short v10, v12, s[8:9]
	v_add_u32_e32 v10, 0xc40e0, v7
	v_accvgpr_read_b32 v11, a222
	s_nop 2
	v_cvt_pk_bf16_f32 v12, v11, v11
	global_store_short v10, v12, s[8:9]
	v_add_u32_e32 v10, 0xc60e0, v7
	v_accvgpr_read_b32 v11, a223
	s_nop 2
	v_cvt_pk_bf16_f32 v12, v11, v11
	global_store_short v10, v12, s[8:9]
	v_add_u32_e32 v10, 0xe0000, v7
	v_accvgpr_read_b32 v11, a224
	s_nop 2
	v_cvt_pk_bf16_f32 v12, v11, v11
	global_store_short v10, v12, s[8:9]
	v_add_u32_e32 v10, 0xe2000, v7
	v_accvgpr_read_b32 v11, a225
	s_nop 2
	v_cvt_pk_bf16_f32 v12, v11, v11
	global_store_short v10, v12, s[8:9]
	v_add_u32_e32 v10, 0xe4000, v7
	v_accvgpr_read_b32 v11, a226
	s_nop 2
	v_cvt_pk_bf16_f32 v12, v11, v11
	global_store_short v10, v12, s[8:9]
	v_add_u32_e32 v10, 0xe6000, v7
	v_accvgpr_read_b32 v11, a227
	s_nop 2
	v_cvt_pk_bf16_f32 v12, v11, v11
	global_store_short v10, v12, s[8:9]
	v_add_u32_e32 v10, 0xe0020, v7
	v_accvgpr_read_b32 v11, a228
	s_nop 2
	v_cvt_pk_bf16_f32 v12, v11, v11
	global_store_short v10, v12, s[8:9]
	v_add_u32_e32 v10, 0xe2020, v7
	v_accvgpr_read_b32 v11, a229
	s_nop 2
	v_cvt_pk_bf16_f32 v12, v11, v11
	global_store_short v10, v12, s[8:9]
	v_add_u32_e32 v10, 0xe4020, v7
	v_accvgpr_read_b32 v11, a230
	s_nop 2
	v_cvt_pk_bf16_f32 v12, v11, v11
	global_store_short v10, v12, s[8:9]
	v_add_u32_e32 v10, 0xe6020, v7
	v_accvgpr_read_b32 v11, a231
	s_nop 2
	v_cvt_pk_bf16_f32 v12, v11, v11
	global_store_short v10, v12, s[8:9]
	v_add_u32_e32 v10, 0xe0040, v7
	v_accvgpr_read_b32 v11, a232
	s_nop 2
	v_cvt_pk_bf16_f32 v12, v11, v11
	global_store_short v10, v12, s[8:9]
	v_add_u32_e32 v10, 0xe2040, v7
	v_accvgpr_read_b32 v11, a233
	s_nop 2
	v_cvt_pk_bf16_f32 v12, v11, v11
	global_store_short v10, v12, s[8:9]
	v_add_u32_e32 v10, 0xe4040, v7
	v_accvgpr_read_b32 v11, a234
	s_nop 2
	v_cvt_pk_bf16_f32 v12, v11, v11
	global_store_short v10, v12, s[8:9]
	v_add_u32_e32 v10, 0xe6040, v7
	v_accvgpr_read_b32 v11, a235
	s_nop 2
	v_cvt_pk_bf16_f32 v12, v11, v11
	global_store_short v10, v12, s[8:9]
	v_add_u32_e32 v10, 0xe0060, v7
	v_accvgpr_read_b32 v11, a236
	s_nop 2
	v_cvt_pk_bf16_f32 v12, v11, v11
	global_store_short v10, v12, s[8:9]
	v_add_u32_e32 v10, 0xe2060, v7
	v_accvgpr_read_b32 v11, a237
	s_nop 2
	v_cvt_pk_bf16_f32 v12, v11, v11
	global_store_short v10, v12, s[8:9]
	v_add_u32_e32 v10, 0xe4060, v7
	v_accvgpr_read_b32 v11, a238
	s_nop 2
	v_cvt_pk_bf16_f32 v12, v11, v11
	global_store_short v10, v12, s[8:9]
	v_add_u32_e32 v10, 0xe6060, v7
	v_accvgpr_read_b32 v11, a239
	s_nop 2
	v_cvt_pk_bf16_f32 v12, v11, v11
	global_store_short v10, v12, s[8:9]
	v_add_u32_e32 v10, 0xe0080, v7
	v_accvgpr_read_b32 v11, a240
	s_nop 2
	v_cvt_pk_bf16_f32 v12, v11, v11
	global_store_short v10, v12, s[8:9]
	v_add_u32_e32 v10, 0xe2080, v7
	v_accvgpr_read_b32 v11, a241
	s_nop 2
	v_cvt_pk_bf16_f32 v12, v11, v11
	global_store_short v10, v12, s[8:9]
	v_add_u32_e32 v10, 0xe4080, v7
	v_accvgpr_read_b32 v11, a242
	s_nop 2
	v_cvt_pk_bf16_f32 v12, v11, v11
	global_store_short v10, v12, s[8:9]
	v_add_u32_e32 v10, 0xe6080, v7
	v_accvgpr_read_b32 v11, a243
	s_nop 2
	v_cvt_pk_bf16_f32 v12, v11, v11
	global_store_short v10, v12, s[8:9]
	v_add_u32_e32 v10, 0xe00a0, v7
	v_accvgpr_read_b32 v11, a244
	s_nop 2
	v_cvt_pk_bf16_f32 v12, v11, v11
	global_store_short v10, v12, s[8:9]
	v_add_u32_e32 v10, 0xe20a0, v7
	v_accvgpr_read_b32 v11, a245
	s_nop 2
	v_cvt_pk_bf16_f32 v12, v11, v11
	global_store_short v10, v12, s[8:9]
	v_add_u32_e32 v10, 0xe40a0, v7
	v_accvgpr_read_b32 v11, a246
	s_nop 2
	v_cvt_pk_bf16_f32 v12, v11, v11
	global_store_short v10, v12, s[8:9]
	v_add_u32_e32 v10, 0xe60a0, v7
	v_accvgpr_read_b32 v11, a247
	s_nop 2
	v_cvt_pk_bf16_f32 v12, v11, v11
	global_store_short v10, v12, s[8:9]
	v_add_u32_e32 v10, 0xe00c0, v7
	v_accvgpr_read_b32 v11, a248
	s_nop 2
	v_cvt_pk_bf16_f32 v12, v11, v11
	global_store_short v10, v12, s[8:9]
	v_add_u32_e32 v10, 0xe20c0, v7
	v_accvgpr_read_b32 v11, a249
	s_nop 2
	v_cvt_pk_bf16_f32 v12, v11, v11
	global_store_short v10, v12, s[8:9]
	v_add_u32_e32 v10, 0xe40c0, v7
	v_accvgpr_read_b32 v11, a250
	s_nop 2
	v_cvt_pk_bf16_f32 v12, v11, v11
	global_store_short v10, v12, s[8:9]
	v_add_u32_e32 v10, 0xe60c0, v7
	v_accvgpr_read_b32 v11, a251
	s_nop 2
	v_cvt_pk_bf16_f32 v12, v11, v11
	global_store_short v10, v12, s[8:9]
	v_add_u32_e32 v10, 0xe00e0, v7
	v_accvgpr_read_b32 v11, a252
	s_nop 2
	v_cvt_pk_bf16_f32 v12, v11, v11
	global_store_short v10, v12, s[8:9]
	v_add_u32_e32 v10, 0xe20e0, v7
	v_accvgpr_read_b32 v11, a253
	s_nop 2
	v_cvt_pk_bf16_f32 v12, v11, v11
	global_store_short v10, v12, s[8:9]
	v_add_u32_e32 v10, 0xe40e0, v7
	v_accvgpr_read_b32 v11, a254
	s_nop 2
	v_cvt_pk_bf16_f32 v12, v11, v11
	global_store_short v10, v12, s[8:9]
	v_add_u32_e32 v10, 0xe60e0, v7
	v_accvgpr_read_b32 v11, a255
	s_nop 2
	v_cvt_pk_bf16_f32 v12, v11, v11
	global_store_short v10, v12, s[8:9]
	s_waitcnt vmcnt(0)
	s_endpgm

	.section	.rodata,"a",@progbits
	.p2align	6, 0x0
.amdhsa_kernel gfxjit_gemm_mx_e2m3_e2m3_e8m0b32_f32acc_bf16_rcr_256x256x128_w2x2_16x16x128_s3_4096x4096x65536_c0c1
  .amdhsa_group_segment_fixed_size 153600
  .amdhsa_private_segment_fixed_size 0
  .amdhsa_kernarg_size 40
  .amdhsa_user_sgpr_count 2
  .amdhsa_user_sgpr_kernarg_segment_ptr 1
  .amdhsa_system_sgpr_workgroup_id_x 1
  .amdhsa_system_vgpr_workitem_id 0
  .amdhsa_next_free_vgpr 496
  .amdhsa_next_free_sgpr 60
  .amdhsa_reserve_vcc 0
  .amdhsa_accum_offset 240
  .amdhsa_float_denorm_mode_32 3
  .amdhsa_float_denorm_mode_16_64 3
.end_amdhsa_kernel

	.amdgpu_metadata
---
custom.config:
  Source:
    Origin: gfxjit
    Description: MX E2M3 GEMM at 4096x4096x65536 on tiled operands and shuffled scales
  Version: 1.0.0
  Features:
    SupportsUserArgs: false
    SupportsBias: false
    SupportsActivation: false
    SupportsScaleAlpha: false
    SupportsGSU: false
  InternalSupportParams:
    KernArgsVersion: 0
  ProblemType:
    OperationType: GEMM
    DataType: F6
    DestDataType: b
    ComputeDataType: s
    HighPrecisionAccumulate: True
    TransposeA: True
    TransposeB: False
    UseBeta: False
    Batched: True
    UseBias: 0
    Activation: False
    UseScaleAlphaVec: 0
    SwizzleTensorA: True
    SwizzleTensorB: True
    MXBlockA: 32
    MXBlockB: 32
    DataTypeMXSA: E8
    DataTypeMXSB: E8
  CustomKernel:
    # The kernel's a and sa are the GEMM's B side, b and sb its A side: the
    # kernel writes C row-major [M, N], which is the column-major D hipBLASLt
    # wants only if the two operands are exchanged.
    args: [ { type: address, semantic: AddressD },
            { type: address, semantic: AddressB },
            { type: address, semantic: AddressA },
            { type: address, semantic: AddressMXScaleB },
            { type: address, semantic: AddressMXScaleA } ]
    macrotile: [256, 256, 128]
    threads: [256, 1, 1]
    grid: [TilesXY, One, One]
    uniformSummationOrder: true
  MatrixInstruction: [16, 16, 128, 1]
  EnableMatrixInstruction: True
  MIWaveTile: [8, 8]
  WavefrontSize: 64
  StaggerU: 0
  MXScaleFormat: HostPreSwizzle128x4
  AssertSizeEqual: {0: 4096, 1: 4096, 2: 1, 3: 65536}
  AssertStrideAEqual: {1: 65536}
  AssertStrideBEqual: {1: 65536}
  AssertStrideDEqual: {1: 4096}
  AssertAlphaValue: 1
amdhsa.target: amdgcn-amd-amdhsa--gfx950
amdhsa.version: [ 1, 1 ]
amdhsa.kernels:
  - .name: gfxjit_gemm_mx_e2m3_e2m3_e8m0b32_f32acc_bf16_rcr_256x256x128_w2x2_16x16x128_s3_4096x4096x65536_c0c1
    .symbol: gfxjit_gemm_mx_e2m3_e2m3_e8m0b32_f32acc_bf16_rcr_256x256x128_w2x2_16x16x128_s3_4096x4096x65536_c0c1.kd
    .kernarg_segment_size: 40
    .kernarg_segment_align: 8
    .group_segment_fixed_size: 153600
    .private_segment_fixed_size: 0
    .max_flat_workgroup_size: 256
    .wavefront_size: 64
    .sgpr_count: 60
    .vgpr_count: 240
    .agpr_count: 256
    .sgpr_spill_count: 0
    .vgpr_spill_count: 0
    .uniform_work_group_size: 1
    .args:
      - { .name: c,  .size: 8, .offset: 0,  .value_kind: global_buffer, .address_space: global }
      - { .name: a,  .size: 8, .offset: 8,  .value_kind: global_buffer, .address_space: global }
      - { .name: b,  .size: 8, .offset: 16, .value_kind: global_buffer, .address_space: global }
      - { .name: sa, .size: 8, .offset: 24, .value_kind: global_buffer, .address_space: global }
      - { .name: sb, .size: 8, .offset: 32, .value_kind: global_buffer, .address_space: global }
...
	.end_amdgpu_metadata
