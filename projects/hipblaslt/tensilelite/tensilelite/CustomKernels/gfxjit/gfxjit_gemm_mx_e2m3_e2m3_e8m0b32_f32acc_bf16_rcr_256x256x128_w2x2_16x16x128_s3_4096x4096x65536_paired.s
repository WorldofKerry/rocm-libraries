	.amdgcn_target "amdgcn-amd-amdhsa--gfx950"
	.text
	.globl	gfxjit_gemm_mx_e2m3_e2m3_e8m0b32_f32acc_bf16_rcr_256x256x128_w2x2_16x16x128_s3_4096x4096x65536_paired
	.p2align	8
	.type	gfxjit_gemm_mx_e2m3_e2m3_e8m0b32_f32acc_bf16_rcr_256x256x128_w2x2_16x16x128_s3_4096x4096x65536_paired,@function
gfxjit_gemm_mx_e2m3_e2m3_e8m0b32_f32acc_bf16_rcr_256x256x128_w2x2_16x16x128_s3_4096x4096x65536_paired:
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
	s_sub_u32 s36, s36, 0x200
	s_subb_u32 s37, s37, 0
	s_mov_b32 s38, 0x6200
	s_mov_b32 s39, 0x20000
	s_mul_i32 s30, s24, 0xc00000
	s_add_u32 s40, s12, s30
	s_addc_u32 s41, s13, 0
	s_sub_u32 s40, s40, 0x800
	s_subb_u32 s41, s41, 0
	s_mov_b32 s42, 0x6800
	s_mov_b32 s43, 0x20000
	s_mul_i32 s30, s23, 0x400
	s_add_u32 s44, s14, s30
	s_addc_u32 s45, s15, 0
	s_mov_b32 s46, 0x400
	s_mov_b32 s47, 0x20000
	s_mul_i32 s30, s24, 0x400
	s_add_u32 s48, s16, s30
	s_addc_u32 s49, s17, 0
	s_sub_u32 s48, s48, 0x100
	s_subb_u32 s49, s49, 0
	s_mov_b32 s50, 0x500
	s_mov_b32 s51, 0x20000
	s_mul_i32 s32, s20, 0x1800
	s_mul_i32 s33, s20, 0x100
	s_mul_i32 s58, s20, 0x3200
	v_lshlrev_b32_e32 v3, 4, v1
	v_add_u32_e32 v3, s32, v3
	v_add_u32_e32 v8, 0x1200, v3
	v_add_u32_e32 v42, 0x1000, v3
	v_lshlrev_b32_e32 v4, 2, v1
	v_add_u32_e32 v4, s33, v4
	v_mul_u32_u24_e32 v5, 48, v1
	s_mul_i32 s34, s21, 0x6400
	s_mul_i32 s35, s22, 0x6400
	v_lshrrev_b32_e32 v6, 5, v1
	v_mul_u32_u24_e32 v6, 0x3200, v6
	v_and_b32_e32 v2, 31, v1
	v_lshlrev_b32_e32 v2, 3, v2
	v_add_u32_e32 v6, v2, v6
	s_mov_b32 s56, s34
	s_mov_b32 s57, s35
	v_add_u32_e32 v24, s34, v5
	v_add_u32_e32 v24, 0x200, v24
	v_add_u32_e32 v27, s35, v5
	v_add_u32_e32 v27, 0x1a00, v27
	v_add_u32_e32 v36, s56, v6
	v_add_u32_e32 v36, 0, v36
	v_add_u32_e32 v39, s57, v6
	v_add_u32_e32 v39, 0x100, v39
	v_add_u32_e32 v25, s34, v5
	v_add_u32_e32 v25, 0xca00, v25
	v_add_u32_e32 v28, s35, v5
	v_add_u32_e32 v28, 0xe200, v28
	v_add_u32_e32 v37, s56, v6
	v_add_u32_e32 v37, 0xc800, v37
	v_add_u32_e32 v40, s57, v6
	v_add_u32_e32 v40, 0xc900, v40
	v_add_u32_e32 v26, s34, v5
	v_add_u32_e32 v26, 0x19200, v26
	v_add_u32_e32 v29, s35, v5
	v_add_u32_e32 v29, 0x1aa00, v29
	v_add_u32_e32 v38, s56, v6
	v_add_u32_e32 v38, 0x19000, v38
	v_add_u32_e32 v41, s57, v6
	v_add_u32_e32 v41, 0x19100, v41
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
	s_add_u32 m0, s58, 0
	buffer_load_dword v4, s[44:47], 0 offen lds
	s_add_u32 s44, s44, 0x4000
	s_addc_u32 s45, s45, 0
	buffer_load_dword v4, s[48:51], 0 offen offset:256 lds
	s_add_u32 s48, s48, 0x4000
	s_addc_u32 s49, s49, 0
	buffer_load_dwordx4 v3, s[36:39], 0 offen offset:512 lds
	buffer_load_dwordx4 v3, s[36:39], 0 offen offset:1536 lds
	buffer_load_dwordx4 v3, s[36:39], 0 offen offset:2560 lds
	buffer_load_dwordx4 v3, s[36:39], 0 offen offset:3584 lds
	s_add_u32 m0, s58, 0x1200
	buffer_load_dwordx4 v8, s[36:39], 0 offen lds
	buffer_load_dwordx4 v8, s[36:39], 0 offen offset:1024 lds
	s_add_u32 s36, s36, 0x6000
	s_addc_u32 s37, s37, 0
	buffer_load_dwordx4 v3, s[40:43], 0 offen offset:2048 lds
	buffer_load_dwordx4 v3, s[40:43], 0 offen offset:3072 lds
	s_add_u32 m0, s58, 0x2200
	buffer_load_dwordx4 v42, s[40:43], 0 offen lds
	buffer_load_dwordx4 v42, s[40:43], 0 offen offset:1024 lds
	buffer_load_dwordx4 v42, s[40:43], 0 offen offset:2048 lds
	buffer_load_dwordx4 v42, s[40:43], 0 offen offset:3072 lds
	s_add_u32 s40, s40, 0x6000
	s_addc_u32 s41, s41, 0
	s_add_u32 m0, s58, 0xc800
	buffer_load_dword v4, s[44:47], 0 offen lds
	s_add_u32 s44, s44, 0x4000
	s_addc_u32 s45, s45, 0
	buffer_load_dword v4, s[48:51], 0 offen offset:256 lds
	s_add_u32 s48, s48, 0x4000
	s_addc_u32 s49, s49, 0
	buffer_load_dwordx4 v3, s[36:39], 0 offen offset:512 lds
	buffer_load_dwordx4 v3, s[36:39], 0 offen offset:1536 lds
	buffer_load_dwordx4 v3, s[36:39], 0 offen offset:2560 lds
	buffer_load_dwordx4 v3, s[36:39], 0 offen offset:3584 lds
	s_add_u32 m0, s58, 0xda00
	buffer_load_dwordx4 v8, s[36:39], 0 offen lds
	buffer_load_dwordx4 v8, s[36:39], 0 offen offset:1024 lds
	s_add_u32 s36, s36, 0x6000
	s_addc_u32 s37, s37, 0
	buffer_load_dwordx4 v3, s[40:43], 0 offen offset:2048 lds
	buffer_load_dwordx4 v3, s[40:43], 0 offen offset:3072 lds
	s_add_u32 m0, s58, 0xea00
	buffer_load_dwordx4 v42, s[40:43], 0 offen lds
	buffer_load_dwordx4 v42, s[40:43], 0 offen offset:1024 lds
	buffer_load_dwordx4 v42, s[40:43], 0 offen offset:2048 lds
	buffer_load_dwordx4 v42, s[40:43], 0 offen offset:3072 lds
	s_add_u32 s40, s40, 0x6000
	s_addc_u32 s41, s41, 0
	s_add_u32 m0, s58, 0x19000
	buffer_load_dword v4, s[44:47], 0 offen lds
	s_add_u32 s44, s44, 0x4000
	s_addc_u32 s45, s45, 0
	buffer_load_dword v4, s[48:51], 0 offen offset:256 lds
	s_add_u32 s48, s48, 0x4000
	s_addc_u32 s49, s49, 0
	buffer_load_dwordx4 v3, s[36:39], 0 offen offset:512 lds
	buffer_load_dwordx4 v3, s[36:39], 0 offen offset:1536 lds
	buffer_load_dwordx4 v3, s[36:39], 0 offen offset:2560 lds
	buffer_load_dwordx4 v3, s[36:39], 0 offen offset:3584 lds
	s_add_u32 m0, s58, 0x1a200
	buffer_load_dwordx4 v8, s[36:39], 0 offen lds
	buffer_load_dwordx4 v8, s[36:39], 0 offen offset:1024 lds
	s_add_u32 s36, s36, 0x6000
	s_addc_u32 s37, s37, 0
	buffer_load_dwordx4 v3, s[40:43], 0 offen offset:2048 lds
	buffer_load_dwordx4 v3, s[40:43], 0 offen offset:3072 lds
	s_add_u32 m0, s58, 0x1b200
	buffer_load_dwordx4 v42, s[40:43], 0 offen lds
	buffer_load_dwordx4 v42, s[40:43], 0 offen offset:1024 lds
	buffer_load_dwordx4 v42, s[40:43], 0 offen offset:2048 lds
	buffer_load_dwordx4 v42, s[40:43], 0 offen offset:3072 lds
	s_add_u32 s40, s40, 0x6000
	s_addc_u32 s41, s41, 0
	s_waitcnt vmcnt(28)
	s_barrier
	ds_read_b64 v[16:17], v36
	ds_read_b64 v[18:19], v39
	ds_read_b128 v[48:51], v24
	ds_read_b128 v[96:99], v27
	ds_read_b128 v[52:55], v24 offset:16
	ds_read_b128 v[100:103], v27 offset:16
	ds_read_b128 v[56:59], v24 offset:32
	ds_read_b128 v[104:107], v27 offset:32
	ds_read_b128 v[60:63], v24 offset:3072
	ds_read_b128 v[108:111], v27 offset:3072
	ds_read_b128 v[64:67], v24 offset:3088
	ds_read_b128 v[112:115], v27 offset:3088
	ds_read_b128 v[68:71], v24 offset:3104
	ds_read_b128 v[116:119], v27 offset:3104
	ds_read_b128 v[72:75], v24 offset:12800
	ds_read_b128 v[120:123], v27 offset:12800
	ds_read_b128 v[76:79], v24 offset:12816
	ds_read_b128 v[124:127], v27 offset:12816
	ds_read_b128 v[80:83], v24 offset:12832
	ds_read_b128 v[128:131], v27 offset:12832
	ds_read_b128 v[84:87], v24 offset:15872
	ds_read_b128 v[132:135], v27 offset:15872
	ds_read_b128 v[88:91], v24 offset:15888
	ds_read_b128 v[136:139], v27 offset:15888
	ds_read_b128 v[92:95], v24 offset:15904
	ds_read_b128 v[140:143], v27 offset:15904
	s_mov_b32 s31, 0x55
	s_waitcnt vmcnt(14)
	s_waitcnt lgkmcnt(0)
	s_barrier
	ds_read_b64 v[20:21], v37
	v_mfma_scale_f32_16x16x128_f8f6f4 a[0:3], v[96:101], v[48:53], a[0:3], v18, v16 op_sel_hi:[0,0,0] cbsz:2 blgp:2
	s_add_u32 m0, s58, 0
	v_mfma_scale_f32_16x16x128_f8f6f4 a[4:7], v[102:107], v[48:53], a[4:7], v18, v16 op_sel:[1,0,0] op_sel_hi:[0,0,0] cbsz:2 blgp:2
	ds_read_b64 v[22:23], v40
	v_mfma_scale_f32_16x16x128_f8f6f4 a[8:11], v[108:113], v[48:53], a[8:11], v18, v16 op_sel_hi:[1,0,0] cbsz:2 blgp:2
	buffer_load_dword v4, s[44:47], 0 offen lds
	v_mfma_scale_f32_16x16x128_f8f6f4 a[12:15], v[114:119], v[48:53], a[12:15], v18, v16 op_sel:[1,0,0] op_sel_hi:[1,0,0] cbsz:2 blgp:2
	ds_read_b128 v[144:147], v25
	v_mfma_scale_f32_16x16x128_f8f6f4 a[16:19], v[120:125], v[48:53], a[16:19], v19, v16 op_sel_hi:[0,0,0] cbsz:2 blgp:2
	v_mfma_scale_f32_16x16x128_f8f6f4 a[20:23], v[126:131], v[48:53], a[20:23], v19, v16 op_sel:[1,0,0] op_sel_hi:[0,0,0] cbsz:2 blgp:2
	ds_read_b128 v[192:195], v28
	v_mfma_scale_f32_16x16x128_f8f6f4 a[24:27], v[132:137], v[48:53], a[24:27], v19, v16 op_sel_hi:[1,0,0] cbsz:2 blgp:2
	s_add_u32 s44, s44, 0x4000
	s_addc_u32 s45, s45, 0
	v_mfma_scale_f32_16x16x128_f8f6f4 a[28:31], v[138:143], v[48:53], a[28:31], v19, v16 op_sel:[1,0,0] op_sel_hi:[1,0,0] cbsz:2 blgp:2
	ds_read_b128 v[148:151], v25 offset:16
	v_mfma_scale_f32_16x16x128_f8f6f4 a[60:63], v[138:143], v[54:59], a[60:63], v19, v16 op_sel:[1,1,0] op_sel_hi:[1,0,0] cbsz:2 blgp:2
	buffer_load_dword v4, s[48:51], 0 offen offset:256 lds
	v_mfma_scale_f32_16x16x128_f8f6f4 a[56:59], v[132:137], v[54:59], a[56:59], v19, v16 op_sel:[0,1,0] op_sel_hi:[1,0,0] cbsz:2 blgp:2
	ds_read_b128 v[196:199], v28 offset:16
	v_mfma_scale_f32_16x16x128_f8f6f4 a[52:55], v[126:131], v[54:59], a[52:55], v19, v16 op_sel:[1,1,0] op_sel_hi:[0,0,0] cbsz:2 blgp:2
	v_mfma_scale_f32_16x16x128_f8f6f4 a[48:51], v[120:125], v[54:59], a[48:51], v19, v16 op_sel:[0,1,0] op_sel_hi:[0,0,0] cbsz:2 blgp:2
	ds_read_b128 v[152:155], v25 offset:32
	v_mfma_scale_f32_16x16x128_f8f6f4 a[44:47], v[114:119], v[54:59], a[44:47], v18, v16 op_sel:[1,1,0] op_sel_hi:[1,0,0] cbsz:2 blgp:2
	s_add_u32 s48, s48, 0x4000
	s_addc_u32 s49, s49, 0
	v_mfma_scale_f32_16x16x128_f8f6f4 a[40:43], v[108:113], v[54:59], a[40:43], v18, v16 op_sel:[0,1,0] op_sel_hi:[1,0,0] cbsz:2 blgp:2
	v_mfma_scale_f32_16x16x128_f8f6f4 a[36:39], v[102:107], v[54:59], a[36:39], v18, v16 op_sel:[1,1,0] op_sel_hi:[0,0,0] cbsz:2 blgp:2
	ds_read_b128 v[200:203], v28 offset:32
	v_mfma_scale_f32_16x16x128_f8f6f4 a[32:35], v[96:101], v[54:59], a[32:35], v18, v16 op_sel:[0,1,0] op_sel_hi:[0,0,0] cbsz:2 blgp:2
	buffer_load_dwordx4 v3, s[36:39], 0 offen offset:512 lds
	v_mfma_scale_f32_16x16x128_f8f6f4 a[64:67], v[96:101], v[60:65], a[64:67], v18, v16 op_sel_hi:[0,1,0] cbsz:2 blgp:2
	ds_read_b128 v[156:159], v25 offset:3072
	v_mfma_scale_f32_16x16x128_f8f6f4 a[68:71], v[102:107], v[60:65], a[68:71], v18, v16 op_sel:[1,0,0] op_sel_hi:[0,1,0] cbsz:2 blgp:2
	buffer_load_dwordx4 v3, s[36:39], 0 offen offset:1536 lds
	v_mfma_scale_f32_16x16x128_f8f6f4 a[72:75], v[108:113], v[60:65], a[72:75], v18, v16 op_sel_hi:[1,1,0] cbsz:2 blgp:2
	ds_read_b128 v[204:207], v28 offset:3072
	v_mfma_scale_f32_16x16x128_f8f6f4 a[76:79], v[114:119], v[60:65], a[76:79], v18, v16 op_sel:[1,0,0] op_sel_hi:[1,1,0] cbsz:2 blgp:2
	v_mfma_scale_f32_16x16x128_f8f6f4 a[80:83], v[120:125], v[60:65], a[80:83], v19, v16 op_sel_hi:[0,1,0] cbsz:2 blgp:2
	ds_read_b128 v[160:163], v25 offset:3088
	v_mfma_scale_f32_16x16x128_f8f6f4 a[84:87], v[126:131], v[60:65], a[84:87], v19, v16 op_sel:[1,0,0] op_sel_hi:[0,1,0] cbsz:2 blgp:2
	buffer_load_dwordx4 v3, s[36:39], 0 offen offset:2560 lds
	v_mfma_scale_f32_16x16x128_f8f6f4 a[88:91], v[132:137], v[60:65], a[88:91], v19, v16 op_sel_hi:[1,1,0] cbsz:2 blgp:2
	ds_read_b128 v[208:211], v28 offset:3088
	v_mfma_scale_f32_16x16x128_f8f6f4 a[92:95], v[138:143], v[60:65], a[92:95], v19, v16 op_sel:[1,0,0] op_sel_hi:[1,1,0] cbsz:2 blgp:2
	buffer_load_dwordx4 v3, s[36:39], 0 offen offset:3584 lds
	v_mfma_scale_f32_16x16x128_f8f6f4 a[124:127], v[138:143], v[66:71], a[124:127], v19, v16 op_sel:[1,1,0] op_sel_hi:[1,1,0] cbsz:2 blgp:2
	ds_read_b128 v[164:167], v25 offset:3104
	v_mfma_scale_f32_16x16x128_f8f6f4 a[120:123], v[132:137], v[66:71], a[120:123], v19, v16 op_sel:[0,1,0] op_sel_hi:[1,1,0] cbsz:2 blgp:2
	v_mfma_scale_f32_16x16x128_f8f6f4 a[116:119], v[126:131], v[66:71], a[116:119], v19, v16 op_sel:[1,1,0] op_sel_hi:[0,1,0] cbsz:2 blgp:2
	s_add_u32 m0, s58, 0x1200
	v_mfma_scale_f32_16x16x128_f8f6f4 a[112:115], v[120:125], v[66:71], a[112:115], v19, v16 op_sel:[0,1,0] op_sel_hi:[0,1,0] cbsz:2 blgp:2
	ds_read_b128 v[212:215], v28 offset:3104
	v_mfma_scale_f32_16x16x128_f8f6f4 a[108:111], v[114:119], v[66:71], a[108:111], v18, v16 op_sel:[1,1,0] op_sel_hi:[1,1,0] cbsz:2 blgp:2
	v_mfma_scale_f32_16x16x128_f8f6f4 a[104:107], v[108:113], v[66:71], a[104:107], v18, v16 op_sel:[0,1,0] op_sel_hi:[1,1,0] cbsz:2 blgp:2
	ds_read_b128 v[168:171], v25 offset:12800
	v_mfma_scale_f32_16x16x128_f8f6f4 a[100:103], v[102:107], v[66:71], a[100:103], v18, v16 op_sel:[1,1,0] op_sel_hi:[0,1,0] cbsz:2 blgp:2
	buffer_load_dwordx4 v8, s[36:39], 0 offen lds
	v_mfma_scale_f32_16x16x128_f8f6f4 a[96:99], v[96:101], v[66:71], a[96:99], v18, v16 op_sel:[0,1,0] op_sel_hi:[0,1,0] cbsz:2 blgp:2
	ds_read_b128 v[216:219], v28 offset:12800
	v_mfma_scale_f32_16x16x128_f8f6f4 a[128:131], v[96:101], v[72:77], a[128:131], v18, v17 op_sel_hi:[0,0,0] cbsz:2 blgp:2
	buffer_load_dwordx4 v8, s[36:39], 0 offen offset:1024 lds
	v_mfma_scale_f32_16x16x128_f8f6f4 a[132:135], v[102:107], v[72:77], a[132:135], v18, v17 op_sel:[1,0,0] op_sel_hi:[0,0,0] cbsz:2 blgp:2
	ds_read_b128 v[172:175], v25 offset:12816
	v_mfma_scale_f32_16x16x128_f8f6f4 a[136:139], v[108:113], v[72:77], a[136:139], v18, v17 op_sel_hi:[1,0,0] cbsz:2 blgp:2
	v_mfma_scale_f32_16x16x128_f8f6f4 a[140:143], v[114:119], v[72:77], a[140:143], v18, v17 op_sel:[1,0,0] op_sel_hi:[1,0,0] cbsz:2 blgp:2
	ds_read_b128 v[220:223], v28 offset:12816
	v_mfma_scale_f32_16x16x128_f8f6f4 a[144:147], v[120:125], v[72:77], a[144:147], v19, v17 op_sel_hi:[0,0,0] cbsz:2 blgp:2
	s_add_u32 s36, s36, 0x6000
	s_addc_u32 s37, s37, 0
	v_mfma_scale_f32_16x16x128_f8f6f4 a[148:151], v[126:131], v[72:77], a[148:151], v19, v17 op_sel:[1,0,0] op_sel_hi:[0,0,0] cbsz:2 blgp:2
	ds_read_b128 v[176:179], v25 offset:12832
	v_mfma_scale_f32_16x16x128_f8f6f4 a[152:155], v[132:137], v[72:77], a[152:155], v19, v17 op_sel_hi:[1,0,0] cbsz:2 blgp:2
	buffer_load_dwordx4 v3, s[40:43], 0 offen offset:2048 lds
	v_mfma_scale_f32_16x16x128_f8f6f4 a[156:159], v[138:143], v[72:77], a[156:159], v19, v17 op_sel:[1,0,0] op_sel_hi:[1,0,0] cbsz:2 blgp:2
	ds_read_b128 v[224:227], v28 offset:12832
	v_mfma_scale_f32_16x16x128_f8f6f4 a[188:191], v[138:143], v[78:83], a[188:191], v19, v17 op_sel:[1,1,0] op_sel_hi:[1,0,0] cbsz:2 blgp:2
	v_mfma_scale_f32_16x16x128_f8f6f4 a[184:187], v[132:137], v[78:83], a[184:187], v19, v17 op_sel:[0,1,0] op_sel_hi:[1,0,0] cbsz:2 blgp:2
	buffer_load_dwordx4 v3, s[40:43], 0 offen offset:3072 lds
	v_mfma_scale_f32_16x16x128_f8f6f4 a[180:183], v[126:131], v[78:83], a[180:183], v19, v17 op_sel:[1,1,0] op_sel_hi:[0,0,0] cbsz:2 blgp:2
	ds_read_b128 v[180:183], v25 offset:15872
	v_mfma_scale_f32_16x16x128_f8f6f4 a[176:179], v[120:125], v[78:83], a[176:179], v19, v17 op_sel:[0,1,0] op_sel_hi:[0,0,0] cbsz:2 blgp:2
	v_mfma_scale_f32_16x16x128_f8f6f4 a[172:175], v[114:119], v[78:83], a[172:175], v18, v17 op_sel:[1,1,0] op_sel_hi:[1,0,0] cbsz:2 blgp:2
	ds_read_b128 v[228:231], v28 offset:15872
	v_mfma_scale_f32_16x16x128_f8f6f4 a[168:171], v[108:113], v[78:83], a[168:171], v18, v17 op_sel:[0,1,0] op_sel_hi:[1,0,0] cbsz:2 blgp:2
	s_add_u32 m0, s58, 0x2200
	v_mfma_scale_f32_16x16x128_f8f6f4 a[164:167], v[102:107], v[78:83], a[164:167], v18, v17 op_sel:[1,1,0] op_sel_hi:[0,0,0] cbsz:2 blgp:2
	ds_read_b128 v[184:187], v25 offset:15888
	v_mfma_scale_f32_16x16x128_f8f6f4 a[160:163], v[96:101], v[78:83], a[160:163], v18, v17 op_sel:[0,1,0] op_sel_hi:[0,0,0] cbsz:2 blgp:2
	buffer_load_dwordx4 v42, s[40:43], 0 offen lds
	v_mfma_scale_f32_16x16x128_f8f6f4 a[192:195], v[96:101], v[84:89], a[192:195], v18, v17 op_sel_hi:[0,1,0] cbsz:2 blgp:2
	ds_read_b128 v[232:235], v28 offset:15888
	v_mfma_scale_f32_16x16x128_f8f6f4 a[196:199], v[102:107], v[84:89], a[196:199], v18, v17 op_sel:[1,0,0] op_sel_hi:[0,1,0] cbsz:2 blgp:2
	v_mfma_scale_f32_16x16x128_f8f6f4 a[200:203], v[108:113], v[84:89], a[200:203], v18, v17 op_sel_hi:[1,1,0] cbsz:2 blgp:2
	ds_read_b128 v[188:191], v25 offset:15904
	v_mfma_scale_f32_16x16x128_f8f6f4 a[204:207], v[114:119], v[84:89], a[204:207], v18, v17 op_sel:[1,0,0] op_sel_hi:[1,1,0] cbsz:2 blgp:2
	buffer_load_dwordx4 v42, s[40:43], 0 offen offset:1024 lds
	v_mfma_scale_f32_16x16x128_f8f6f4 a[208:211], v[120:125], v[84:89], a[208:211], v19, v17 op_sel_hi:[0,1,0] cbsz:2 blgp:2
	ds_read_b128 v[236:239], v28 offset:15904
	v_mfma_scale_f32_16x16x128_f8f6f4 a[212:215], v[126:131], v[84:89], a[212:215], v19, v17 op_sel:[1,0,0] op_sel_hi:[0,1,0] cbsz:2 blgp:2
	buffer_load_dwordx4 v42, s[40:43], 0 offen offset:2048 lds
	v_mfma_scale_f32_16x16x128_f8f6f4 a[216:219], v[132:137], v[84:89], a[216:219], v19, v17 op_sel_hi:[1,1,0] cbsz:2 blgp:2
	v_mfma_scale_f32_16x16x128_f8f6f4 a[220:223], v[138:143], v[84:89], a[220:223], v19, v17 op_sel:[1,0,0] op_sel_hi:[1,1,0] cbsz:2 blgp:2
	v_mfma_scale_f32_16x16x128_f8f6f4 a[252:255], v[138:143], v[90:95], a[252:255], v19, v17 op_sel:[1,1,0] op_sel_hi:[1,1,0] cbsz:2 blgp:2
	buffer_load_dwordx4 v42, s[40:43], 0 offen offset:3072 lds
	v_mfma_scale_f32_16x16x128_f8f6f4 a[248:251], v[132:137], v[90:95], a[248:251], v19, v17 op_sel:[0,1,0] op_sel_hi:[1,1,0] cbsz:2 blgp:2
	v_mfma_scale_f32_16x16x128_f8f6f4 a[244:247], v[126:131], v[90:95], a[244:247], v19, v17 op_sel:[1,1,0] op_sel_hi:[0,1,0] cbsz:2 blgp:2
	v_mfma_scale_f32_16x16x128_f8f6f4 a[240:243], v[120:125], v[90:95], a[240:243], v19, v17 op_sel:[0,1,0] op_sel_hi:[0,1,0] cbsz:2 blgp:2
	s_add_u32 s40, s40, 0x6000
	s_addc_u32 s41, s41, 0
	v_mfma_scale_f32_16x16x128_f8f6f4 a[236:239], v[114:119], v[90:95], a[236:239], v18, v17 op_sel:[1,1,0] op_sel_hi:[1,1,0] cbsz:2 blgp:2
	v_mfma_scale_f32_16x16x128_f8f6f4 a[232:235], v[108:113], v[90:95], a[232:235], v18, v17 op_sel:[0,1,0] op_sel_hi:[1,1,0] cbsz:2 blgp:2
	v_mfma_scale_f32_16x16x128_f8f6f4 a[228:231], v[102:107], v[90:95], a[228:231], v18, v17 op_sel:[1,1,0] op_sel_hi:[0,1,0] cbsz:2 blgp:2
	v_mfma_scale_f32_16x16x128_f8f6f4 a[224:227], v[96:101], v[90:95], a[224:227], v18, v17 op_sel:[0,1,0] op_sel_hi:[0,1,0] cbsz:2 blgp:2
	s_waitcnt vmcnt(14)
	s_waitcnt lgkmcnt(0)
	s_barrier
	ds_read_b64 v[16:17], v38
	v_mfma_scale_f32_16x16x128_f8f6f4 a[0:3], v[192:197], v[144:149], a[0:3], v22, v20 op_sel_hi:[0,0,0] cbsz:2 blgp:2
	s_add_u32 m0, s58, 0xc800
	v_mfma_scale_f32_16x16x128_f8f6f4 a[4:7], v[198:203], v[144:149], a[4:7], v22, v20 op_sel:[1,0,0] op_sel_hi:[0,0,0] cbsz:2 blgp:2
	ds_read_b64 v[18:19], v41
	v_mfma_scale_f32_16x16x128_f8f6f4 a[8:11], v[204:209], v[144:149], a[8:11], v22, v20 op_sel_hi:[1,0,0] cbsz:2 blgp:2
	buffer_load_dword v4, s[44:47], 0 offen lds
	v_mfma_scale_f32_16x16x128_f8f6f4 a[12:15], v[210:215], v[144:149], a[12:15], v22, v20 op_sel:[1,0,0] op_sel_hi:[1,0,0] cbsz:2 blgp:2
	ds_read_b128 v[48:51], v26
	v_mfma_scale_f32_16x16x128_f8f6f4 a[16:19], v[216:221], v[144:149], a[16:19], v23, v20 op_sel_hi:[0,0,0] cbsz:2 blgp:2
	v_mfma_scale_f32_16x16x128_f8f6f4 a[20:23], v[222:227], v[144:149], a[20:23], v23, v20 op_sel:[1,0,0] op_sel_hi:[0,0,0] cbsz:2 blgp:2
	ds_read_b128 v[96:99], v29
	v_mfma_scale_f32_16x16x128_f8f6f4 a[24:27], v[228:233], v[144:149], a[24:27], v23, v20 op_sel_hi:[1,0,0] cbsz:2 blgp:2
	s_add_u32 s44, s44, 0x4000
	s_addc_u32 s45, s45, 0
	v_mfma_scale_f32_16x16x128_f8f6f4 a[28:31], v[234:239], v[144:149], a[28:31], v23, v20 op_sel:[1,0,0] op_sel_hi:[1,0,0] cbsz:2 blgp:2
	ds_read_b128 v[52:55], v26 offset:16
	v_mfma_scale_f32_16x16x128_f8f6f4 a[60:63], v[234:239], v[150:155], a[60:63], v23, v20 op_sel:[1,1,0] op_sel_hi:[1,0,0] cbsz:2 blgp:2
	buffer_load_dword v4, s[48:51], 0 offen offset:256 lds
	v_mfma_scale_f32_16x16x128_f8f6f4 a[56:59], v[228:233], v[150:155], a[56:59], v23, v20 op_sel:[0,1,0] op_sel_hi:[1,0,0] cbsz:2 blgp:2
	ds_read_b128 v[100:103], v29 offset:16
	v_mfma_scale_f32_16x16x128_f8f6f4 a[52:55], v[222:227], v[150:155], a[52:55], v23, v20 op_sel:[1,1,0] op_sel_hi:[0,0,0] cbsz:2 blgp:2
	v_mfma_scale_f32_16x16x128_f8f6f4 a[48:51], v[216:221], v[150:155], a[48:51], v23, v20 op_sel:[0,1,0] op_sel_hi:[0,0,0] cbsz:2 blgp:2
	ds_read_b128 v[56:59], v26 offset:32
	v_mfma_scale_f32_16x16x128_f8f6f4 a[44:47], v[210:215], v[150:155], a[44:47], v22, v20 op_sel:[1,1,0] op_sel_hi:[1,0,0] cbsz:2 blgp:2
	s_add_u32 s48, s48, 0x4000
	s_addc_u32 s49, s49, 0
	v_mfma_scale_f32_16x16x128_f8f6f4 a[40:43], v[204:209], v[150:155], a[40:43], v22, v20 op_sel:[0,1,0] op_sel_hi:[1,0,0] cbsz:2 blgp:2
	v_mfma_scale_f32_16x16x128_f8f6f4 a[36:39], v[198:203], v[150:155], a[36:39], v22, v20 op_sel:[1,1,0] op_sel_hi:[0,0,0] cbsz:2 blgp:2
	ds_read_b128 v[104:107], v29 offset:32
	v_mfma_scale_f32_16x16x128_f8f6f4 a[32:35], v[192:197], v[150:155], a[32:35], v22, v20 op_sel:[0,1,0] op_sel_hi:[0,0,0] cbsz:2 blgp:2
	buffer_load_dwordx4 v3, s[36:39], 0 offen offset:512 lds
	v_mfma_scale_f32_16x16x128_f8f6f4 a[64:67], v[192:197], v[156:161], a[64:67], v22, v20 op_sel_hi:[0,1,0] cbsz:2 blgp:2
	ds_read_b128 v[60:63], v26 offset:3072
	v_mfma_scale_f32_16x16x128_f8f6f4 a[68:71], v[198:203], v[156:161], a[68:71], v22, v20 op_sel:[1,0,0] op_sel_hi:[0,1,0] cbsz:2 blgp:2
	buffer_load_dwordx4 v3, s[36:39], 0 offen offset:1536 lds
	v_mfma_scale_f32_16x16x128_f8f6f4 a[72:75], v[204:209], v[156:161], a[72:75], v22, v20 op_sel_hi:[1,1,0] cbsz:2 blgp:2
	ds_read_b128 v[108:111], v29 offset:3072
	v_mfma_scale_f32_16x16x128_f8f6f4 a[76:79], v[210:215], v[156:161], a[76:79], v22, v20 op_sel:[1,0,0] op_sel_hi:[1,1,0] cbsz:2 blgp:2
	v_mfma_scale_f32_16x16x128_f8f6f4 a[80:83], v[216:221], v[156:161], a[80:83], v23, v20 op_sel_hi:[0,1,0] cbsz:2 blgp:2
	ds_read_b128 v[64:67], v26 offset:3088
	v_mfma_scale_f32_16x16x128_f8f6f4 a[84:87], v[222:227], v[156:161], a[84:87], v23, v20 op_sel:[1,0,0] op_sel_hi:[0,1,0] cbsz:2 blgp:2
	buffer_load_dwordx4 v3, s[36:39], 0 offen offset:2560 lds
	v_mfma_scale_f32_16x16x128_f8f6f4 a[88:91], v[228:233], v[156:161], a[88:91], v23, v20 op_sel_hi:[1,1,0] cbsz:2 blgp:2
	ds_read_b128 v[112:115], v29 offset:3088
	v_mfma_scale_f32_16x16x128_f8f6f4 a[92:95], v[234:239], v[156:161], a[92:95], v23, v20 op_sel:[1,0,0] op_sel_hi:[1,1,0] cbsz:2 blgp:2
	buffer_load_dwordx4 v3, s[36:39], 0 offen offset:3584 lds
	v_mfma_scale_f32_16x16x128_f8f6f4 a[124:127], v[234:239], v[162:167], a[124:127], v23, v20 op_sel:[1,1,0] op_sel_hi:[1,1,0] cbsz:2 blgp:2
	ds_read_b128 v[68:71], v26 offset:3104
	v_mfma_scale_f32_16x16x128_f8f6f4 a[120:123], v[228:233], v[162:167], a[120:123], v23, v20 op_sel:[0,1,0] op_sel_hi:[1,1,0] cbsz:2 blgp:2
	v_mfma_scale_f32_16x16x128_f8f6f4 a[116:119], v[222:227], v[162:167], a[116:119], v23, v20 op_sel:[1,1,0] op_sel_hi:[0,1,0] cbsz:2 blgp:2
	s_add_u32 m0, s58, 0xda00
	v_mfma_scale_f32_16x16x128_f8f6f4 a[112:115], v[216:221], v[162:167], a[112:115], v23, v20 op_sel:[0,1,0] op_sel_hi:[0,1,0] cbsz:2 blgp:2
	ds_read_b128 v[116:119], v29 offset:3104
	v_mfma_scale_f32_16x16x128_f8f6f4 a[108:111], v[210:215], v[162:167], a[108:111], v22, v20 op_sel:[1,1,0] op_sel_hi:[1,1,0] cbsz:2 blgp:2
	v_mfma_scale_f32_16x16x128_f8f6f4 a[104:107], v[204:209], v[162:167], a[104:107], v22, v20 op_sel:[0,1,0] op_sel_hi:[1,1,0] cbsz:2 blgp:2
	ds_read_b128 v[72:75], v26 offset:12800
	v_mfma_scale_f32_16x16x128_f8f6f4 a[100:103], v[198:203], v[162:167], a[100:103], v22, v20 op_sel:[1,1,0] op_sel_hi:[0,1,0] cbsz:2 blgp:2
	buffer_load_dwordx4 v8, s[36:39], 0 offen lds
	v_mfma_scale_f32_16x16x128_f8f6f4 a[96:99], v[192:197], v[162:167], a[96:99], v22, v20 op_sel:[0,1,0] op_sel_hi:[0,1,0] cbsz:2 blgp:2
	ds_read_b128 v[120:123], v29 offset:12800
	v_mfma_scale_f32_16x16x128_f8f6f4 a[128:131], v[192:197], v[168:173], a[128:131], v22, v21 op_sel_hi:[0,0,0] cbsz:2 blgp:2
	buffer_load_dwordx4 v8, s[36:39], 0 offen offset:1024 lds
	v_mfma_scale_f32_16x16x128_f8f6f4 a[132:135], v[198:203], v[168:173], a[132:135], v22, v21 op_sel:[1,0,0] op_sel_hi:[0,0,0] cbsz:2 blgp:2
	ds_read_b128 v[76:79], v26 offset:12816
	v_mfma_scale_f32_16x16x128_f8f6f4 a[136:139], v[204:209], v[168:173], a[136:139], v22, v21 op_sel_hi:[1,0,0] cbsz:2 blgp:2
	v_mfma_scale_f32_16x16x128_f8f6f4 a[140:143], v[210:215], v[168:173], a[140:143], v22, v21 op_sel:[1,0,0] op_sel_hi:[1,0,0] cbsz:2 blgp:2
	ds_read_b128 v[124:127], v29 offset:12816
	v_mfma_scale_f32_16x16x128_f8f6f4 a[144:147], v[216:221], v[168:173], a[144:147], v23, v21 op_sel_hi:[0,0,0] cbsz:2 blgp:2
	s_add_u32 s36, s36, 0x6000
	s_addc_u32 s37, s37, 0
	v_mfma_scale_f32_16x16x128_f8f6f4 a[148:151], v[222:227], v[168:173], a[148:151], v23, v21 op_sel:[1,0,0] op_sel_hi:[0,0,0] cbsz:2 blgp:2
	ds_read_b128 v[80:83], v26 offset:12832
	v_mfma_scale_f32_16x16x128_f8f6f4 a[152:155], v[228:233], v[168:173], a[152:155], v23, v21 op_sel_hi:[1,0,0] cbsz:2 blgp:2
	buffer_load_dwordx4 v3, s[40:43], 0 offen offset:2048 lds
	v_mfma_scale_f32_16x16x128_f8f6f4 a[156:159], v[234:239], v[168:173], a[156:159], v23, v21 op_sel:[1,0,0] op_sel_hi:[1,0,0] cbsz:2 blgp:2
	ds_read_b128 v[128:131], v29 offset:12832
	v_mfma_scale_f32_16x16x128_f8f6f4 a[188:191], v[234:239], v[174:179], a[188:191], v23, v21 op_sel:[1,1,0] op_sel_hi:[1,0,0] cbsz:2 blgp:2
	v_mfma_scale_f32_16x16x128_f8f6f4 a[184:187], v[228:233], v[174:179], a[184:187], v23, v21 op_sel:[0,1,0] op_sel_hi:[1,0,0] cbsz:2 blgp:2
	buffer_load_dwordx4 v3, s[40:43], 0 offen offset:3072 lds
	v_mfma_scale_f32_16x16x128_f8f6f4 a[180:183], v[222:227], v[174:179], a[180:183], v23, v21 op_sel:[1,1,0] op_sel_hi:[0,0,0] cbsz:2 blgp:2
	ds_read_b128 v[84:87], v26 offset:15872
	v_mfma_scale_f32_16x16x128_f8f6f4 a[176:179], v[216:221], v[174:179], a[176:179], v23, v21 op_sel:[0,1,0] op_sel_hi:[0,0,0] cbsz:2 blgp:2
	v_mfma_scale_f32_16x16x128_f8f6f4 a[172:175], v[210:215], v[174:179], a[172:175], v22, v21 op_sel:[1,1,0] op_sel_hi:[1,0,0] cbsz:2 blgp:2
	ds_read_b128 v[132:135], v29 offset:15872
	v_mfma_scale_f32_16x16x128_f8f6f4 a[168:171], v[204:209], v[174:179], a[168:171], v22, v21 op_sel:[0,1,0] op_sel_hi:[1,0,0] cbsz:2 blgp:2
	s_add_u32 m0, s58, 0xea00
	v_mfma_scale_f32_16x16x128_f8f6f4 a[164:167], v[198:203], v[174:179], a[164:167], v22, v21 op_sel:[1,1,0] op_sel_hi:[0,0,0] cbsz:2 blgp:2
	ds_read_b128 v[88:91], v26 offset:15888
	v_mfma_scale_f32_16x16x128_f8f6f4 a[160:163], v[192:197], v[174:179], a[160:163], v22, v21 op_sel:[0,1,0] op_sel_hi:[0,0,0] cbsz:2 blgp:2
	buffer_load_dwordx4 v42, s[40:43], 0 offen lds
	v_mfma_scale_f32_16x16x128_f8f6f4 a[192:195], v[192:197], v[180:185], a[192:195], v22, v21 op_sel_hi:[0,1,0] cbsz:2 blgp:2
	ds_read_b128 v[136:139], v29 offset:15888
	v_mfma_scale_f32_16x16x128_f8f6f4 a[196:199], v[198:203], v[180:185], a[196:199], v22, v21 op_sel:[1,0,0] op_sel_hi:[0,1,0] cbsz:2 blgp:2
	v_mfma_scale_f32_16x16x128_f8f6f4 a[200:203], v[204:209], v[180:185], a[200:203], v22, v21 op_sel_hi:[1,1,0] cbsz:2 blgp:2
	ds_read_b128 v[92:95], v26 offset:15904
	v_mfma_scale_f32_16x16x128_f8f6f4 a[204:207], v[210:215], v[180:185], a[204:207], v22, v21 op_sel:[1,0,0] op_sel_hi:[1,1,0] cbsz:2 blgp:2
	buffer_load_dwordx4 v42, s[40:43], 0 offen offset:1024 lds
	v_mfma_scale_f32_16x16x128_f8f6f4 a[208:211], v[216:221], v[180:185], a[208:211], v23, v21 op_sel_hi:[0,1,0] cbsz:2 blgp:2
	ds_read_b128 v[140:143], v29 offset:15904
	v_mfma_scale_f32_16x16x128_f8f6f4 a[212:215], v[222:227], v[180:185], a[212:215], v23, v21 op_sel:[1,0,0] op_sel_hi:[0,1,0] cbsz:2 blgp:2
	buffer_load_dwordx4 v42, s[40:43], 0 offen offset:2048 lds
	v_mfma_scale_f32_16x16x128_f8f6f4 a[216:219], v[228:233], v[180:185], a[216:219], v23, v21 op_sel_hi:[1,1,0] cbsz:2 blgp:2
	v_mfma_scale_f32_16x16x128_f8f6f4 a[220:223], v[234:239], v[180:185], a[220:223], v23, v21 op_sel:[1,0,0] op_sel_hi:[1,1,0] cbsz:2 blgp:2
	v_mfma_scale_f32_16x16x128_f8f6f4 a[252:255], v[234:239], v[186:191], a[252:255], v23, v21 op_sel:[1,1,0] op_sel_hi:[1,1,0] cbsz:2 blgp:2
	buffer_load_dwordx4 v42, s[40:43], 0 offen offset:3072 lds
	v_mfma_scale_f32_16x16x128_f8f6f4 a[248:251], v[228:233], v[186:191], a[248:251], v23, v21 op_sel:[0,1,0] op_sel_hi:[1,1,0] cbsz:2 blgp:2
	v_mfma_scale_f32_16x16x128_f8f6f4 a[244:247], v[222:227], v[186:191], a[244:247], v23, v21 op_sel:[1,1,0] op_sel_hi:[0,1,0] cbsz:2 blgp:2
	v_mfma_scale_f32_16x16x128_f8f6f4 a[240:243], v[216:221], v[186:191], a[240:243], v23, v21 op_sel:[0,1,0] op_sel_hi:[0,1,0] cbsz:2 blgp:2
	s_add_u32 s40, s40, 0x6000
	s_addc_u32 s41, s41, 0
	v_mfma_scale_f32_16x16x128_f8f6f4 a[236:239], v[210:215], v[186:191], a[236:239], v22, v21 op_sel:[1,1,0] op_sel_hi:[1,1,0] cbsz:2 blgp:2
	v_mfma_scale_f32_16x16x128_f8f6f4 a[232:235], v[204:209], v[186:191], a[232:235], v22, v21 op_sel:[0,1,0] op_sel_hi:[1,1,0] cbsz:2 blgp:2
	v_mfma_scale_f32_16x16x128_f8f6f4 a[228:231], v[198:203], v[186:191], a[228:231], v22, v21 op_sel:[1,1,0] op_sel_hi:[0,1,0] cbsz:2 blgp:2
	v_mfma_scale_f32_16x16x128_f8f6f4 a[224:227], v[192:197], v[186:191], a[224:227], v22, v21 op_sel:[0,1,0] op_sel_hi:[0,1,0] cbsz:2 blgp:2
	s_waitcnt vmcnt(14)
	s_waitcnt lgkmcnt(0)
	s_barrier
	ds_read_b64 v[20:21], v36
	v_mfma_scale_f32_16x16x128_f8f6f4 a[0:3], v[96:101], v[48:53], a[0:3], v18, v16 op_sel_hi:[0,0,0] cbsz:2 blgp:2
	s_add_u32 m0, s58, 0x19000
	v_mfma_scale_f32_16x16x128_f8f6f4 a[4:7], v[102:107], v[48:53], a[4:7], v18, v16 op_sel:[1,0,0] op_sel_hi:[0,0,0] cbsz:2 blgp:2
	ds_read_b64 v[22:23], v39
	v_mfma_scale_f32_16x16x128_f8f6f4 a[8:11], v[108:113], v[48:53], a[8:11], v18, v16 op_sel_hi:[1,0,0] cbsz:2 blgp:2
	buffer_load_dword v4, s[44:47], 0 offen lds
	v_mfma_scale_f32_16x16x128_f8f6f4 a[12:15], v[114:119], v[48:53], a[12:15], v18, v16 op_sel:[1,0,0] op_sel_hi:[1,0,0] cbsz:2 blgp:2
	ds_read_b128 v[144:147], v24
	v_mfma_scale_f32_16x16x128_f8f6f4 a[16:19], v[120:125], v[48:53], a[16:19], v19, v16 op_sel_hi:[0,0,0] cbsz:2 blgp:2
	v_mfma_scale_f32_16x16x128_f8f6f4 a[20:23], v[126:131], v[48:53], a[20:23], v19, v16 op_sel:[1,0,0] op_sel_hi:[0,0,0] cbsz:2 blgp:2
	ds_read_b128 v[192:195], v27
	v_mfma_scale_f32_16x16x128_f8f6f4 a[24:27], v[132:137], v[48:53], a[24:27], v19, v16 op_sel_hi:[1,0,0] cbsz:2 blgp:2
	s_add_u32 s44, s44, 0x4000
	s_addc_u32 s45, s45, 0
	v_mfma_scale_f32_16x16x128_f8f6f4 a[28:31], v[138:143], v[48:53], a[28:31], v19, v16 op_sel:[1,0,0] op_sel_hi:[1,0,0] cbsz:2 blgp:2
	ds_read_b128 v[148:151], v24 offset:16
	v_mfma_scale_f32_16x16x128_f8f6f4 a[60:63], v[138:143], v[54:59], a[60:63], v19, v16 op_sel:[1,1,0] op_sel_hi:[1,0,0] cbsz:2 blgp:2
	buffer_load_dword v4, s[48:51], 0 offen offset:256 lds
	v_mfma_scale_f32_16x16x128_f8f6f4 a[56:59], v[132:137], v[54:59], a[56:59], v19, v16 op_sel:[0,1,0] op_sel_hi:[1,0,0] cbsz:2 blgp:2
	ds_read_b128 v[196:199], v27 offset:16
	v_mfma_scale_f32_16x16x128_f8f6f4 a[52:55], v[126:131], v[54:59], a[52:55], v19, v16 op_sel:[1,1,0] op_sel_hi:[0,0,0] cbsz:2 blgp:2
	v_mfma_scale_f32_16x16x128_f8f6f4 a[48:51], v[120:125], v[54:59], a[48:51], v19, v16 op_sel:[0,1,0] op_sel_hi:[0,0,0] cbsz:2 blgp:2
	ds_read_b128 v[152:155], v24 offset:32
	v_mfma_scale_f32_16x16x128_f8f6f4 a[44:47], v[114:119], v[54:59], a[44:47], v18, v16 op_sel:[1,1,0] op_sel_hi:[1,0,0] cbsz:2 blgp:2
	s_add_u32 s48, s48, 0x4000
	s_addc_u32 s49, s49, 0
	v_mfma_scale_f32_16x16x128_f8f6f4 a[40:43], v[108:113], v[54:59], a[40:43], v18, v16 op_sel:[0,1,0] op_sel_hi:[1,0,0] cbsz:2 blgp:2
	v_mfma_scale_f32_16x16x128_f8f6f4 a[36:39], v[102:107], v[54:59], a[36:39], v18, v16 op_sel:[1,1,0] op_sel_hi:[0,0,0] cbsz:2 blgp:2
	ds_read_b128 v[200:203], v27 offset:32
	v_mfma_scale_f32_16x16x128_f8f6f4 a[32:35], v[96:101], v[54:59], a[32:35], v18, v16 op_sel:[0,1,0] op_sel_hi:[0,0,0] cbsz:2 blgp:2
	buffer_load_dwordx4 v3, s[36:39], 0 offen offset:512 lds
	v_mfma_scale_f32_16x16x128_f8f6f4 a[64:67], v[96:101], v[60:65], a[64:67], v18, v16 op_sel_hi:[0,1,0] cbsz:2 blgp:2
	ds_read_b128 v[156:159], v24 offset:3072
	v_mfma_scale_f32_16x16x128_f8f6f4 a[68:71], v[102:107], v[60:65], a[68:71], v18, v16 op_sel:[1,0,0] op_sel_hi:[0,1,0] cbsz:2 blgp:2
	buffer_load_dwordx4 v3, s[36:39], 0 offen offset:1536 lds
	v_mfma_scale_f32_16x16x128_f8f6f4 a[72:75], v[108:113], v[60:65], a[72:75], v18, v16 op_sel_hi:[1,1,0] cbsz:2 blgp:2
	ds_read_b128 v[204:207], v27 offset:3072
	v_mfma_scale_f32_16x16x128_f8f6f4 a[76:79], v[114:119], v[60:65], a[76:79], v18, v16 op_sel:[1,0,0] op_sel_hi:[1,1,0] cbsz:2 blgp:2
	v_mfma_scale_f32_16x16x128_f8f6f4 a[80:83], v[120:125], v[60:65], a[80:83], v19, v16 op_sel_hi:[0,1,0] cbsz:2 blgp:2
	ds_read_b128 v[160:163], v24 offset:3088
	v_mfma_scale_f32_16x16x128_f8f6f4 a[84:87], v[126:131], v[60:65], a[84:87], v19, v16 op_sel:[1,0,0] op_sel_hi:[0,1,0] cbsz:2 blgp:2
	buffer_load_dwordx4 v3, s[36:39], 0 offen offset:2560 lds
	v_mfma_scale_f32_16x16x128_f8f6f4 a[88:91], v[132:137], v[60:65], a[88:91], v19, v16 op_sel_hi:[1,1,0] cbsz:2 blgp:2
	ds_read_b128 v[208:211], v27 offset:3088
	v_mfma_scale_f32_16x16x128_f8f6f4 a[92:95], v[138:143], v[60:65], a[92:95], v19, v16 op_sel:[1,0,0] op_sel_hi:[1,1,0] cbsz:2 blgp:2
	buffer_load_dwordx4 v3, s[36:39], 0 offen offset:3584 lds
	v_mfma_scale_f32_16x16x128_f8f6f4 a[124:127], v[138:143], v[66:71], a[124:127], v19, v16 op_sel:[1,1,0] op_sel_hi:[1,1,0] cbsz:2 blgp:2
	ds_read_b128 v[164:167], v24 offset:3104
	v_mfma_scale_f32_16x16x128_f8f6f4 a[120:123], v[132:137], v[66:71], a[120:123], v19, v16 op_sel:[0,1,0] op_sel_hi:[1,1,0] cbsz:2 blgp:2
	v_mfma_scale_f32_16x16x128_f8f6f4 a[116:119], v[126:131], v[66:71], a[116:119], v19, v16 op_sel:[1,1,0] op_sel_hi:[0,1,0] cbsz:2 blgp:2
	s_add_u32 m0, s58, 0x1a200
	v_mfma_scale_f32_16x16x128_f8f6f4 a[112:115], v[120:125], v[66:71], a[112:115], v19, v16 op_sel:[0,1,0] op_sel_hi:[0,1,0] cbsz:2 blgp:2
	ds_read_b128 v[212:215], v27 offset:3104
	v_mfma_scale_f32_16x16x128_f8f6f4 a[108:111], v[114:119], v[66:71], a[108:111], v18, v16 op_sel:[1,1,0] op_sel_hi:[1,1,0] cbsz:2 blgp:2
	v_mfma_scale_f32_16x16x128_f8f6f4 a[104:107], v[108:113], v[66:71], a[104:107], v18, v16 op_sel:[0,1,0] op_sel_hi:[1,1,0] cbsz:2 blgp:2
	ds_read_b128 v[168:171], v24 offset:12800
	v_mfma_scale_f32_16x16x128_f8f6f4 a[100:103], v[102:107], v[66:71], a[100:103], v18, v16 op_sel:[1,1,0] op_sel_hi:[0,1,0] cbsz:2 blgp:2
	buffer_load_dwordx4 v8, s[36:39], 0 offen lds
	v_mfma_scale_f32_16x16x128_f8f6f4 a[96:99], v[96:101], v[66:71], a[96:99], v18, v16 op_sel:[0,1,0] op_sel_hi:[0,1,0] cbsz:2 blgp:2
	ds_read_b128 v[216:219], v27 offset:12800
	v_mfma_scale_f32_16x16x128_f8f6f4 a[128:131], v[96:101], v[72:77], a[128:131], v18, v17 op_sel_hi:[0,0,0] cbsz:2 blgp:2
	buffer_load_dwordx4 v8, s[36:39], 0 offen offset:1024 lds
	v_mfma_scale_f32_16x16x128_f8f6f4 a[132:135], v[102:107], v[72:77], a[132:135], v18, v17 op_sel:[1,0,0] op_sel_hi:[0,0,0] cbsz:2 blgp:2
	ds_read_b128 v[172:175], v24 offset:12816
	v_mfma_scale_f32_16x16x128_f8f6f4 a[136:139], v[108:113], v[72:77], a[136:139], v18, v17 op_sel_hi:[1,0,0] cbsz:2 blgp:2
	v_mfma_scale_f32_16x16x128_f8f6f4 a[140:143], v[114:119], v[72:77], a[140:143], v18, v17 op_sel:[1,0,0] op_sel_hi:[1,0,0] cbsz:2 blgp:2
	ds_read_b128 v[220:223], v27 offset:12816
	v_mfma_scale_f32_16x16x128_f8f6f4 a[144:147], v[120:125], v[72:77], a[144:147], v19, v17 op_sel_hi:[0,0,0] cbsz:2 blgp:2
	s_add_u32 s36, s36, 0x6000
	s_addc_u32 s37, s37, 0
	v_mfma_scale_f32_16x16x128_f8f6f4 a[148:151], v[126:131], v[72:77], a[148:151], v19, v17 op_sel:[1,0,0] op_sel_hi:[0,0,0] cbsz:2 blgp:2
	ds_read_b128 v[176:179], v24 offset:12832
	v_mfma_scale_f32_16x16x128_f8f6f4 a[152:155], v[132:137], v[72:77], a[152:155], v19, v17 op_sel_hi:[1,0,0] cbsz:2 blgp:2
	buffer_load_dwordx4 v3, s[40:43], 0 offen offset:2048 lds
	v_mfma_scale_f32_16x16x128_f8f6f4 a[156:159], v[138:143], v[72:77], a[156:159], v19, v17 op_sel:[1,0,0] op_sel_hi:[1,0,0] cbsz:2 blgp:2
	ds_read_b128 v[224:227], v27 offset:12832
	v_mfma_scale_f32_16x16x128_f8f6f4 a[188:191], v[138:143], v[78:83], a[188:191], v19, v17 op_sel:[1,1,0] op_sel_hi:[1,0,0] cbsz:2 blgp:2
	v_mfma_scale_f32_16x16x128_f8f6f4 a[184:187], v[132:137], v[78:83], a[184:187], v19, v17 op_sel:[0,1,0] op_sel_hi:[1,0,0] cbsz:2 blgp:2
	buffer_load_dwordx4 v3, s[40:43], 0 offen offset:3072 lds
	v_mfma_scale_f32_16x16x128_f8f6f4 a[180:183], v[126:131], v[78:83], a[180:183], v19, v17 op_sel:[1,1,0] op_sel_hi:[0,0,0] cbsz:2 blgp:2
	ds_read_b128 v[180:183], v24 offset:15872
	v_mfma_scale_f32_16x16x128_f8f6f4 a[176:179], v[120:125], v[78:83], a[176:179], v19, v17 op_sel:[0,1,0] op_sel_hi:[0,0,0] cbsz:2 blgp:2
	v_mfma_scale_f32_16x16x128_f8f6f4 a[172:175], v[114:119], v[78:83], a[172:175], v18, v17 op_sel:[1,1,0] op_sel_hi:[1,0,0] cbsz:2 blgp:2
	ds_read_b128 v[228:231], v27 offset:15872
	v_mfma_scale_f32_16x16x128_f8f6f4 a[168:171], v[108:113], v[78:83], a[168:171], v18, v17 op_sel:[0,1,0] op_sel_hi:[1,0,0] cbsz:2 blgp:2
	s_add_u32 m0, s58, 0x1b200
	v_mfma_scale_f32_16x16x128_f8f6f4 a[164:167], v[102:107], v[78:83], a[164:167], v18, v17 op_sel:[1,1,0] op_sel_hi:[0,0,0] cbsz:2 blgp:2
	ds_read_b128 v[184:187], v24 offset:15888
	v_mfma_scale_f32_16x16x128_f8f6f4 a[160:163], v[96:101], v[78:83], a[160:163], v18, v17 op_sel:[0,1,0] op_sel_hi:[0,0,0] cbsz:2 blgp:2
	buffer_load_dwordx4 v42, s[40:43], 0 offen lds
	v_mfma_scale_f32_16x16x128_f8f6f4 a[192:195], v[96:101], v[84:89], a[192:195], v18, v17 op_sel_hi:[0,1,0] cbsz:2 blgp:2
	ds_read_b128 v[232:235], v27 offset:15888
	v_mfma_scale_f32_16x16x128_f8f6f4 a[196:199], v[102:107], v[84:89], a[196:199], v18, v17 op_sel:[1,0,0] op_sel_hi:[0,1,0] cbsz:2 blgp:2
	v_mfma_scale_f32_16x16x128_f8f6f4 a[200:203], v[108:113], v[84:89], a[200:203], v18, v17 op_sel_hi:[1,1,0] cbsz:2 blgp:2
	ds_read_b128 v[188:191], v24 offset:15904
	v_mfma_scale_f32_16x16x128_f8f6f4 a[204:207], v[114:119], v[84:89], a[204:207], v18, v17 op_sel:[1,0,0] op_sel_hi:[1,1,0] cbsz:2 blgp:2
	buffer_load_dwordx4 v42, s[40:43], 0 offen offset:1024 lds
	v_mfma_scale_f32_16x16x128_f8f6f4 a[208:211], v[120:125], v[84:89], a[208:211], v19, v17 op_sel_hi:[0,1,0] cbsz:2 blgp:2
	ds_read_b128 v[236:239], v27 offset:15904
	v_mfma_scale_f32_16x16x128_f8f6f4 a[212:215], v[126:131], v[84:89], a[212:215], v19, v17 op_sel:[1,0,0] op_sel_hi:[0,1,0] cbsz:2 blgp:2
	buffer_load_dwordx4 v42, s[40:43], 0 offen offset:2048 lds
	v_mfma_scale_f32_16x16x128_f8f6f4 a[216:219], v[132:137], v[84:89], a[216:219], v19, v17 op_sel_hi:[1,1,0] cbsz:2 blgp:2
	v_mfma_scale_f32_16x16x128_f8f6f4 a[220:223], v[138:143], v[84:89], a[220:223], v19, v17 op_sel:[1,0,0] op_sel_hi:[1,1,0] cbsz:2 blgp:2
	v_mfma_scale_f32_16x16x128_f8f6f4 a[252:255], v[138:143], v[90:95], a[252:255], v19, v17 op_sel:[1,1,0] op_sel_hi:[1,1,0] cbsz:2 blgp:2
	buffer_load_dwordx4 v42, s[40:43], 0 offen offset:3072 lds
	v_mfma_scale_f32_16x16x128_f8f6f4 a[248:251], v[132:137], v[90:95], a[248:251], v19, v17 op_sel:[0,1,0] op_sel_hi:[1,1,0] cbsz:2 blgp:2
	v_mfma_scale_f32_16x16x128_f8f6f4 a[244:247], v[126:131], v[90:95], a[244:247], v19, v17 op_sel:[1,1,0] op_sel_hi:[0,1,0] cbsz:2 blgp:2
	v_mfma_scale_f32_16x16x128_f8f6f4 a[240:243], v[120:125], v[90:95], a[240:243], v19, v17 op_sel:[0,1,0] op_sel_hi:[0,1,0] cbsz:2 blgp:2
	s_add_u32 s40, s40, 0x6000
	s_addc_u32 s41, s41, 0
	v_mfma_scale_f32_16x16x128_f8f6f4 a[236:239], v[114:119], v[90:95], a[236:239], v18, v17 op_sel:[1,1,0] op_sel_hi:[1,1,0] cbsz:2 blgp:2
	v_mfma_scale_f32_16x16x128_f8f6f4 a[232:235], v[108:113], v[90:95], a[232:235], v18, v17 op_sel:[0,1,0] op_sel_hi:[1,1,0] cbsz:2 blgp:2
	v_mfma_scale_f32_16x16x128_f8f6f4 a[228:231], v[102:107], v[90:95], a[228:231], v18, v17 op_sel:[1,1,0] op_sel_hi:[0,1,0] cbsz:2 blgp:2
	v_mfma_scale_f32_16x16x128_f8f6f4 a[224:227], v[96:101], v[90:95], a[224:227], v18, v17 op_sel:[0,1,0] op_sel_hi:[0,1,0] cbsz:2 blgp:2
	s_waitcnt vmcnt(14)
	s_waitcnt lgkmcnt(0)
	s_barrier
	ds_read_b64 v[16:17], v37
	v_mfma_scale_f32_16x16x128_f8f6f4 a[0:3], v[192:197], v[144:149], a[0:3], v22, v20 op_sel_hi:[0,0,0] cbsz:2 blgp:2
	s_add_u32 m0, s58, 0
	v_mfma_scale_f32_16x16x128_f8f6f4 a[4:7], v[198:203], v[144:149], a[4:7], v22, v20 op_sel:[1,0,0] op_sel_hi:[0,0,0] cbsz:2 blgp:2
	ds_read_b64 v[18:19], v40
	v_mfma_scale_f32_16x16x128_f8f6f4 a[8:11], v[204:209], v[144:149], a[8:11], v22, v20 op_sel_hi:[1,0,0] cbsz:2 blgp:2
	buffer_load_dword v4, s[44:47], 0 offen lds
	v_mfma_scale_f32_16x16x128_f8f6f4 a[12:15], v[210:215], v[144:149], a[12:15], v22, v20 op_sel:[1,0,0] op_sel_hi:[1,0,0] cbsz:2 blgp:2
	ds_read_b128 v[48:51], v25
	v_mfma_scale_f32_16x16x128_f8f6f4 a[16:19], v[216:221], v[144:149], a[16:19], v23, v20 op_sel_hi:[0,0,0] cbsz:2 blgp:2
	v_mfma_scale_f32_16x16x128_f8f6f4 a[20:23], v[222:227], v[144:149], a[20:23], v23, v20 op_sel:[1,0,0] op_sel_hi:[0,0,0] cbsz:2 blgp:2
	ds_read_b128 v[96:99], v28
	v_mfma_scale_f32_16x16x128_f8f6f4 a[24:27], v[228:233], v[144:149], a[24:27], v23, v20 op_sel_hi:[1,0,0] cbsz:2 blgp:2
	s_add_u32 s44, s44, 0x4000
	s_addc_u32 s45, s45, 0
	v_mfma_scale_f32_16x16x128_f8f6f4 a[28:31], v[234:239], v[144:149], a[28:31], v23, v20 op_sel:[1,0,0] op_sel_hi:[1,0,0] cbsz:2 blgp:2
	ds_read_b128 v[52:55], v25 offset:16
	v_mfma_scale_f32_16x16x128_f8f6f4 a[60:63], v[234:239], v[150:155], a[60:63], v23, v20 op_sel:[1,1,0] op_sel_hi:[1,0,0] cbsz:2 blgp:2
	buffer_load_dword v4, s[48:51], 0 offen offset:256 lds
	v_mfma_scale_f32_16x16x128_f8f6f4 a[56:59], v[228:233], v[150:155], a[56:59], v23, v20 op_sel:[0,1,0] op_sel_hi:[1,0,0] cbsz:2 blgp:2
	ds_read_b128 v[100:103], v28 offset:16
	v_mfma_scale_f32_16x16x128_f8f6f4 a[52:55], v[222:227], v[150:155], a[52:55], v23, v20 op_sel:[1,1,0] op_sel_hi:[0,0,0] cbsz:2 blgp:2
	v_mfma_scale_f32_16x16x128_f8f6f4 a[48:51], v[216:221], v[150:155], a[48:51], v23, v20 op_sel:[0,1,0] op_sel_hi:[0,0,0] cbsz:2 blgp:2
	ds_read_b128 v[56:59], v25 offset:32
	v_mfma_scale_f32_16x16x128_f8f6f4 a[44:47], v[210:215], v[150:155], a[44:47], v22, v20 op_sel:[1,1,0] op_sel_hi:[1,0,0] cbsz:2 blgp:2
	s_add_u32 s48, s48, 0x4000
	s_addc_u32 s49, s49, 0
	v_mfma_scale_f32_16x16x128_f8f6f4 a[40:43], v[204:209], v[150:155], a[40:43], v22, v20 op_sel:[0,1,0] op_sel_hi:[1,0,0] cbsz:2 blgp:2
	v_mfma_scale_f32_16x16x128_f8f6f4 a[36:39], v[198:203], v[150:155], a[36:39], v22, v20 op_sel:[1,1,0] op_sel_hi:[0,0,0] cbsz:2 blgp:2
	ds_read_b128 v[104:107], v28 offset:32
	v_mfma_scale_f32_16x16x128_f8f6f4 a[32:35], v[192:197], v[150:155], a[32:35], v22, v20 op_sel:[0,1,0] op_sel_hi:[0,0,0] cbsz:2 blgp:2
	buffer_load_dwordx4 v3, s[36:39], 0 offen offset:512 lds
	v_mfma_scale_f32_16x16x128_f8f6f4 a[64:67], v[192:197], v[156:161], a[64:67], v22, v20 op_sel_hi:[0,1,0] cbsz:2 blgp:2
	ds_read_b128 v[60:63], v25 offset:3072
	v_mfma_scale_f32_16x16x128_f8f6f4 a[68:71], v[198:203], v[156:161], a[68:71], v22, v20 op_sel:[1,0,0] op_sel_hi:[0,1,0] cbsz:2 blgp:2
	buffer_load_dwordx4 v3, s[36:39], 0 offen offset:1536 lds
	v_mfma_scale_f32_16x16x128_f8f6f4 a[72:75], v[204:209], v[156:161], a[72:75], v22, v20 op_sel_hi:[1,1,0] cbsz:2 blgp:2
	ds_read_b128 v[108:111], v28 offset:3072
	v_mfma_scale_f32_16x16x128_f8f6f4 a[76:79], v[210:215], v[156:161], a[76:79], v22, v20 op_sel:[1,0,0] op_sel_hi:[1,1,0] cbsz:2 blgp:2
	v_mfma_scale_f32_16x16x128_f8f6f4 a[80:83], v[216:221], v[156:161], a[80:83], v23, v20 op_sel_hi:[0,1,0] cbsz:2 blgp:2
	ds_read_b128 v[64:67], v25 offset:3088
	v_mfma_scale_f32_16x16x128_f8f6f4 a[84:87], v[222:227], v[156:161], a[84:87], v23, v20 op_sel:[1,0,0] op_sel_hi:[0,1,0] cbsz:2 blgp:2
	buffer_load_dwordx4 v3, s[36:39], 0 offen offset:2560 lds
	v_mfma_scale_f32_16x16x128_f8f6f4 a[88:91], v[228:233], v[156:161], a[88:91], v23, v20 op_sel_hi:[1,1,0] cbsz:2 blgp:2
	ds_read_b128 v[112:115], v28 offset:3088
	v_mfma_scale_f32_16x16x128_f8f6f4 a[92:95], v[234:239], v[156:161], a[92:95], v23, v20 op_sel:[1,0,0] op_sel_hi:[1,1,0] cbsz:2 blgp:2
	buffer_load_dwordx4 v3, s[36:39], 0 offen offset:3584 lds
	v_mfma_scale_f32_16x16x128_f8f6f4 a[124:127], v[234:239], v[162:167], a[124:127], v23, v20 op_sel:[1,1,0] op_sel_hi:[1,1,0] cbsz:2 blgp:2
	ds_read_b128 v[68:71], v25 offset:3104
	v_mfma_scale_f32_16x16x128_f8f6f4 a[120:123], v[228:233], v[162:167], a[120:123], v23, v20 op_sel:[0,1,0] op_sel_hi:[1,1,0] cbsz:2 blgp:2
	v_mfma_scale_f32_16x16x128_f8f6f4 a[116:119], v[222:227], v[162:167], a[116:119], v23, v20 op_sel:[1,1,0] op_sel_hi:[0,1,0] cbsz:2 blgp:2
	s_add_u32 m0, s58, 0x1200
	v_mfma_scale_f32_16x16x128_f8f6f4 a[112:115], v[216:221], v[162:167], a[112:115], v23, v20 op_sel:[0,1,0] op_sel_hi:[0,1,0] cbsz:2 blgp:2
	ds_read_b128 v[116:119], v28 offset:3104
	v_mfma_scale_f32_16x16x128_f8f6f4 a[108:111], v[210:215], v[162:167], a[108:111], v22, v20 op_sel:[1,1,0] op_sel_hi:[1,1,0] cbsz:2 blgp:2
	v_mfma_scale_f32_16x16x128_f8f6f4 a[104:107], v[204:209], v[162:167], a[104:107], v22, v20 op_sel:[0,1,0] op_sel_hi:[1,1,0] cbsz:2 blgp:2
	ds_read_b128 v[72:75], v25 offset:12800
	v_mfma_scale_f32_16x16x128_f8f6f4 a[100:103], v[198:203], v[162:167], a[100:103], v22, v20 op_sel:[1,1,0] op_sel_hi:[0,1,0] cbsz:2 blgp:2
	buffer_load_dwordx4 v8, s[36:39], 0 offen lds
	v_mfma_scale_f32_16x16x128_f8f6f4 a[96:99], v[192:197], v[162:167], a[96:99], v22, v20 op_sel:[0,1,0] op_sel_hi:[0,1,0] cbsz:2 blgp:2
	ds_read_b128 v[120:123], v28 offset:12800
	v_mfma_scale_f32_16x16x128_f8f6f4 a[128:131], v[192:197], v[168:173], a[128:131], v22, v21 op_sel_hi:[0,0,0] cbsz:2 blgp:2
	buffer_load_dwordx4 v8, s[36:39], 0 offen offset:1024 lds
	v_mfma_scale_f32_16x16x128_f8f6f4 a[132:135], v[198:203], v[168:173], a[132:135], v22, v21 op_sel:[1,0,0] op_sel_hi:[0,0,0] cbsz:2 blgp:2
	ds_read_b128 v[76:79], v25 offset:12816
	v_mfma_scale_f32_16x16x128_f8f6f4 a[136:139], v[204:209], v[168:173], a[136:139], v22, v21 op_sel_hi:[1,0,0] cbsz:2 blgp:2
	v_mfma_scale_f32_16x16x128_f8f6f4 a[140:143], v[210:215], v[168:173], a[140:143], v22, v21 op_sel:[1,0,0] op_sel_hi:[1,0,0] cbsz:2 blgp:2
	ds_read_b128 v[124:127], v28 offset:12816
	v_mfma_scale_f32_16x16x128_f8f6f4 a[144:147], v[216:221], v[168:173], a[144:147], v23, v21 op_sel_hi:[0,0,0] cbsz:2 blgp:2
	s_add_u32 s36, s36, 0x6000
	s_addc_u32 s37, s37, 0
	v_mfma_scale_f32_16x16x128_f8f6f4 a[148:151], v[222:227], v[168:173], a[148:151], v23, v21 op_sel:[1,0,0] op_sel_hi:[0,0,0] cbsz:2 blgp:2
	ds_read_b128 v[80:83], v25 offset:12832
	v_mfma_scale_f32_16x16x128_f8f6f4 a[152:155], v[228:233], v[168:173], a[152:155], v23, v21 op_sel_hi:[1,0,0] cbsz:2 blgp:2
	buffer_load_dwordx4 v3, s[40:43], 0 offen offset:2048 lds
	v_mfma_scale_f32_16x16x128_f8f6f4 a[156:159], v[234:239], v[168:173], a[156:159], v23, v21 op_sel:[1,0,0] op_sel_hi:[1,0,0] cbsz:2 blgp:2
	ds_read_b128 v[128:131], v28 offset:12832
	v_mfma_scale_f32_16x16x128_f8f6f4 a[188:191], v[234:239], v[174:179], a[188:191], v23, v21 op_sel:[1,1,0] op_sel_hi:[1,0,0] cbsz:2 blgp:2
	v_mfma_scale_f32_16x16x128_f8f6f4 a[184:187], v[228:233], v[174:179], a[184:187], v23, v21 op_sel:[0,1,0] op_sel_hi:[1,0,0] cbsz:2 blgp:2
	buffer_load_dwordx4 v3, s[40:43], 0 offen offset:3072 lds
	v_mfma_scale_f32_16x16x128_f8f6f4 a[180:183], v[222:227], v[174:179], a[180:183], v23, v21 op_sel:[1,1,0] op_sel_hi:[0,0,0] cbsz:2 blgp:2
	ds_read_b128 v[84:87], v25 offset:15872
	v_mfma_scale_f32_16x16x128_f8f6f4 a[176:179], v[216:221], v[174:179], a[176:179], v23, v21 op_sel:[0,1,0] op_sel_hi:[0,0,0] cbsz:2 blgp:2
	v_mfma_scale_f32_16x16x128_f8f6f4 a[172:175], v[210:215], v[174:179], a[172:175], v22, v21 op_sel:[1,1,0] op_sel_hi:[1,0,0] cbsz:2 blgp:2
	ds_read_b128 v[132:135], v28 offset:15872
	v_mfma_scale_f32_16x16x128_f8f6f4 a[168:171], v[204:209], v[174:179], a[168:171], v22, v21 op_sel:[0,1,0] op_sel_hi:[1,0,0] cbsz:2 blgp:2
	s_add_u32 m0, s58, 0x2200
	v_mfma_scale_f32_16x16x128_f8f6f4 a[164:167], v[198:203], v[174:179], a[164:167], v22, v21 op_sel:[1,1,0] op_sel_hi:[0,0,0] cbsz:2 blgp:2
	ds_read_b128 v[88:91], v25 offset:15888
	v_mfma_scale_f32_16x16x128_f8f6f4 a[160:163], v[192:197], v[174:179], a[160:163], v22, v21 op_sel:[0,1,0] op_sel_hi:[0,0,0] cbsz:2 blgp:2
	buffer_load_dwordx4 v42, s[40:43], 0 offen lds
	v_mfma_scale_f32_16x16x128_f8f6f4 a[192:195], v[192:197], v[180:185], a[192:195], v22, v21 op_sel_hi:[0,1,0] cbsz:2 blgp:2
	ds_read_b128 v[136:139], v28 offset:15888
	v_mfma_scale_f32_16x16x128_f8f6f4 a[196:199], v[198:203], v[180:185], a[196:199], v22, v21 op_sel:[1,0,0] op_sel_hi:[0,1,0] cbsz:2 blgp:2
	v_mfma_scale_f32_16x16x128_f8f6f4 a[200:203], v[204:209], v[180:185], a[200:203], v22, v21 op_sel_hi:[1,1,0] cbsz:2 blgp:2
	ds_read_b128 v[92:95], v25 offset:15904
	v_mfma_scale_f32_16x16x128_f8f6f4 a[204:207], v[210:215], v[180:185], a[204:207], v22, v21 op_sel:[1,0,0] op_sel_hi:[1,1,0] cbsz:2 blgp:2
	buffer_load_dwordx4 v42, s[40:43], 0 offen offset:1024 lds
	v_mfma_scale_f32_16x16x128_f8f6f4 a[208:211], v[216:221], v[180:185], a[208:211], v23, v21 op_sel_hi:[0,1,0] cbsz:2 blgp:2
	ds_read_b128 v[140:143], v28 offset:15904
	v_mfma_scale_f32_16x16x128_f8f6f4 a[212:215], v[222:227], v[180:185], a[212:215], v23, v21 op_sel:[1,0,0] op_sel_hi:[0,1,0] cbsz:2 blgp:2
	buffer_load_dwordx4 v42, s[40:43], 0 offen offset:2048 lds
	v_mfma_scale_f32_16x16x128_f8f6f4 a[216:219], v[228:233], v[180:185], a[216:219], v23, v21 op_sel_hi:[1,1,0] cbsz:2 blgp:2
	v_mfma_scale_f32_16x16x128_f8f6f4 a[220:223], v[234:239], v[180:185], a[220:223], v23, v21 op_sel:[1,0,0] op_sel_hi:[1,1,0] cbsz:2 blgp:2
	v_mfma_scale_f32_16x16x128_f8f6f4 a[252:255], v[234:239], v[186:191], a[252:255], v23, v21 op_sel:[1,1,0] op_sel_hi:[1,1,0] cbsz:2 blgp:2
	buffer_load_dwordx4 v42, s[40:43], 0 offen offset:3072 lds
	v_mfma_scale_f32_16x16x128_f8f6f4 a[248:251], v[228:233], v[186:191], a[248:251], v23, v21 op_sel:[0,1,0] op_sel_hi:[1,1,0] cbsz:2 blgp:2
	v_mfma_scale_f32_16x16x128_f8f6f4 a[244:247], v[222:227], v[186:191], a[244:247], v23, v21 op_sel:[1,1,0] op_sel_hi:[0,1,0] cbsz:2 blgp:2
	v_mfma_scale_f32_16x16x128_f8f6f4 a[240:243], v[216:221], v[186:191], a[240:243], v23, v21 op_sel:[0,1,0] op_sel_hi:[0,1,0] cbsz:2 blgp:2
	s_add_u32 s40, s40, 0x6000
	s_addc_u32 s41, s41, 0
	v_mfma_scale_f32_16x16x128_f8f6f4 a[236:239], v[210:215], v[186:191], a[236:239], v22, v21 op_sel:[1,1,0] op_sel_hi:[1,1,0] cbsz:2 blgp:2
	v_mfma_scale_f32_16x16x128_f8f6f4 a[232:235], v[204:209], v[186:191], a[232:235], v22, v21 op_sel:[0,1,0] op_sel_hi:[1,1,0] cbsz:2 blgp:2
	v_mfma_scale_f32_16x16x128_f8f6f4 a[228:231], v[198:203], v[186:191], a[228:231], v22, v21 op_sel:[1,1,0] op_sel_hi:[0,1,0] cbsz:2 blgp:2
	v_mfma_scale_f32_16x16x128_f8f6f4 a[224:227], v[192:197], v[186:191], a[224:227], v22, v21 op_sel:[0,1,0] op_sel_hi:[0,1,0] cbsz:2 blgp:2
	s_waitcnt vmcnt(14)
	s_waitcnt lgkmcnt(0)
	s_barrier
	ds_read_b64 v[20:21], v38
	v_mfma_scale_f32_16x16x128_f8f6f4 a[0:3], v[96:101], v[48:53], a[0:3], v18, v16 op_sel_hi:[0,0,0] cbsz:2 blgp:2
	s_add_u32 m0, s58, 0xc800
	v_mfma_scale_f32_16x16x128_f8f6f4 a[4:7], v[102:107], v[48:53], a[4:7], v18, v16 op_sel:[1,0,0] op_sel_hi:[0,0,0] cbsz:2 blgp:2
	ds_read_b64 v[22:23], v41
	v_mfma_scale_f32_16x16x128_f8f6f4 a[8:11], v[108:113], v[48:53], a[8:11], v18, v16 op_sel_hi:[1,0,0] cbsz:2 blgp:2
	buffer_load_dword v4, s[44:47], 0 offen lds
	v_mfma_scale_f32_16x16x128_f8f6f4 a[12:15], v[114:119], v[48:53], a[12:15], v18, v16 op_sel:[1,0,0] op_sel_hi:[1,0,0] cbsz:2 blgp:2
	ds_read_b128 v[144:147], v26
	v_mfma_scale_f32_16x16x128_f8f6f4 a[16:19], v[120:125], v[48:53], a[16:19], v19, v16 op_sel_hi:[0,0,0] cbsz:2 blgp:2
	v_mfma_scale_f32_16x16x128_f8f6f4 a[20:23], v[126:131], v[48:53], a[20:23], v19, v16 op_sel:[1,0,0] op_sel_hi:[0,0,0] cbsz:2 blgp:2
	ds_read_b128 v[192:195], v29
	v_mfma_scale_f32_16x16x128_f8f6f4 a[24:27], v[132:137], v[48:53], a[24:27], v19, v16 op_sel_hi:[1,0,0] cbsz:2 blgp:2
	s_add_u32 s44, s44, 0x4000
	s_addc_u32 s45, s45, 0
	v_mfma_scale_f32_16x16x128_f8f6f4 a[28:31], v[138:143], v[48:53], a[28:31], v19, v16 op_sel:[1,0,0] op_sel_hi:[1,0,0] cbsz:2 blgp:2
	ds_read_b128 v[148:151], v26 offset:16
	v_mfma_scale_f32_16x16x128_f8f6f4 a[60:63], v[138:143], v[54:59], a[60:63], v19, v16 op_sel:[1,1,0] op_sel_hi:[1,0,0] cbsz:2 blgp:2
	buffer_load_dword v4, s[48:51], 0 offen offset:256 lds
	v_mfma_scale_f32_16x16x128_f8f6f4 a[56:59], v[132:137], v[54:59], a[56:59], v19, v16 op_sel:[0,1,0] op_sel_hi:[1,0,0] cbsz:2 blgp:2
	ds_read_b128 v[196:199], v29 offset:16
	v_mfma_scale_f32_16x16x128_f8f6f4 a[52:55], v[126:131], v[54:59], a[52:55], v19, v16 op_sel:[1,1,0] op_sel_hi:[0,0,0] cbsz:2 blgp:2
	v_mfma_scale_f32_16x16x128_f8f6f4 a[48:51], v[120:125], v[54:59], a[48:51], v19, v16 op_sel:[0,1,0] op_sel_hi:[0,0,0] cbsz:2 blgp:2
	ds_read_b128 v[152:155], v26 offset:32
	v_mfma_scale_f32_16x16x128_f8f6f4 a[44:47], v[114:119], v[54:59], a[44:47], v18, v16 op_sel:[1,1,0] op_sel_hi:[1,0,0] cbsz:2 blgp:2
	s_add_u32 s48, s48, 0x4000
	s_addc_u32 s49, s49, 0
	v_mfma_scale_f32_16x16x128_f8f6f4 a[40:43], v[108:113], v[54:59], a[40:43], v18, v16 op_sel:[0,1,0] op_sel_hi:[1,0,0] cbsz:2 blgp:2
	v_mfma_scale_f32_16x16x128_f8f6f4 a[36:39], v[102:107], v[54:59], a[36:39], v18, v16 op_sel:[1,1,0] op_sel_hi:[0,0,0] cbsz:2 blgp:2
	ds_read_b128 v[200:203], v29 offset:32
	v_mfma_scale_f32_16x16x128_f8f6f4 a[32:35], v[96:101], v[54:59], a[32:35], v18, v16 op_sel:[0,1,0] op_sel_hi:[0,0,0] cbsz:2 blgp:2
	buffer_load_dwordx4 v3, s[36:39], 0 offen offset:512 lds
	v_mfma_scale_f32_16x16x128_f8f6f4 a[64:67], v[96:101], v[60:65], a[64:67], v18, v16 op_sel_hi:[0,1,0] cbsz:2 blgp:2
	ds_read_b128 v[156:159], v26 offset:3072
	v_mfma_scale_f32_16x16x128_f8f6f4 a[68:71], v[102:107], v[60:65], a[68:71], v18, v16 op_sel:[1,0,0] op_sel_hi:[0,1,0] cbsz:2 blgp:2
	buffer_load_dwordx4 v3, s[36:39], 0 offen offset:1536 lds
	v_mfma_scale_f32_16x16x128_f8f6f4 a[72:75], v[108:113], v[60:65], a[72:75], v18, v16 op_sel_hi:[1,1,0] cbsz:2 blgp:2
	ds_read_b128 v[204:207], v29 offset:3072
	v_mfma_scale_f32_16x16x128_f8f6f4 a[76:79], v[114:119], v[60:65], a[76:79], v18, v16 op_sel:[1,0,0] op_sel_hi:[1,1,0] cbsz:2 blgp:2
	v_mfma_scale_f32_16x16x128_f8f6f4 a[80:83], v[120:125], v[60:65], a[80:83], v19, v16 op_sel_hi:[0,1,0] cbsz:2 blgp:2
	ds_read_b128 v[160:163], v26 offset:3088
	v_mfma_scale_f32_16x16x128_f8f6f4 a[84:87], v[126:131], v[60:65], a[84:87], v19, v16 op_sel:[1,0,0] op_sel_hi:[0,1,0] cbsz:2 blgp:2
	buffer_load_dwordx4 v3, s[36:39], 0 offen offset:2560 lds
	v_mfma_scale_f32_16x16x128_f8f6f4 a[88:91], v[132:137], v[60:65], a[88:91], v19, v16 op_sel_hi:[1,1,0] cbsz:2 blgp:2
	ds_read_b128 v[208:211], v29 offset:3088
	v_mfma_scale_f32_16x16x128_f8f6f4 a[92:95], v[138:143], v[60:65], a[92:95], v19, v16 op_sel:[1,0,0] op_sel_hi:[1,1,0] cbsz:2 blgp:2
	buffer_load_dwordx4 v3, s[36:39], 0 offen offset:3584 lds
	v_mfma_scale_f32_16x16x128_f8f6f4 a[124:127], v[138:143], v[66:71], a[124:127], v19, v16 op_sel:[1,1,0] op_sel_hi:[1,1,0] cbsz:2 blgp:2
	ds_read_b128 v[164:167], v26 offset:3104
	v_mfma_scale_f32_16x16x128_f8f6f4 a[120:123], v[132:137], v[66:71], a[120:123], v19, v16 op_sel:[0,1,0] op_sel_hi:[1,1,0] cbsz:2 blgp:2
	v_mfma_scale_f32_16x16x128_f8f6f4 a[116:119], v[126:131], v[66:71], a[116:119], v19, v16 op_sel:[1,1,0] op_sel_hi:[0,1,0] cbsz:2 blgp:2
	s_add_u32 m0, s58, 0xda00
	v_mfma_scale_f32_16x16x128_f8f6f4 a[112:115], v[120:125], v[66:71], a[112:115], v19, v16 op_sel:[0,1,0] op_sel_hi:[0,1,0] cbsz:2 blgp:2
	ds_read_b128 v[212:215], v29 offset:3104
	v_mfma_scale_f32_16x16x128_f8f6f4 a[108:111], v[114:119], v[66:71], a[108:111], v18, v16 op_sel:[1,1,0] op_sel_hi:[1,1,0] cbsz:2 blgp:2
	v_mfma_scale_f32_16x16x128_f8f6f4 a[104:107], v[108:113], v[66:71], a[104:107], v18, v16 op_sel:[0,1,0] op_sel_hi:[1,1,0] cbsz:2 blgp:2
	ds_read_b128 v[168:171], v26 offset:12800
	v_mfma_scale_f32_16x16x128_f8f6f4 a[100:103], v[102:107], v[66:71], a[100:103], v18, v16 op_sel:[1,1,0] op_sel_hi:[0,1,0] cbsz:2 blgp:2
	buffer_load_dwordx4 v8, s[36:39], 0 offen lds
	v_mfma_scale_f32_16x16x128_f8f6f4 a[96:99], v[96:101], v[66:71], a[96:99], v18, v16 op_sel:[0,1,0] op_sel_hi:[0,1,0] cbsz:2 blgp:2
	ds_read_b128 v[216:219], v29 offset:12800
	v_mfma_scale_f32_16x16x128_f8f6f4 a[128:131], v[96:101], v[72:77], a[128:131], v18, v17 op_sel_hi:[0,0,0] cbsz:2 blgp:2
	buffer_load_dwordx4 v8, s[36:39], 0 offen offset:1024 lds
	v_mfma_scale_f32_16x16x128_f8f6f4 a[132:135], v[102:107], v[72:77], a[132:135], v18, v17 op_sel:[1,0,0] op_sel_hi:[0,0,0] cbsz:2 blgp:2
	ds_read_b128 v[172:175], v26 offset:12816
	v_mfma_scale_f32_16x16x128_f8f6f4 a[136:139], v[108:113], v[72:77], a[136:139], v18, v17 op_sel_hi:[1,0,0] cbsz:2 blgp:2
	v_mfma_scale_f32_16x16x128_f8f6f4 a[140:143], v[114:119], v[72:77], a[140:143], v18, v17 op_sel:[1,0,0] op_sel_hi:[1,0,0] cbsz:2 blgp:2
	ds_read_b128 v[220:223], v29 offset:12816
	v_mfma_scale_f32_16x16x128_f8f6f4 a[144:147], v[120:125], v[72:77], a[144:147], v19, v17 op_sel_hi:[0,0,0] cbsz:2 blgp:2
	s_add_u32 s36, s36, 0x6000
	s_addc_u32 s37, s37, 0
	v_mfma_scale_f32_16x16x128_f8f6f4 a[148:151], v[126:131], v[72:77], a[148:151], v19, v17 op_sel:[1,0,0] op_sel_hi:[0,0,0] cbsz:2 blgp:2
	ds_read_b128 v[176:179], v26 offset:12832
	v_mfma_scale_f32_16x16x128_f8f6f4 a[152:155], v[132:137], v[72:77], a[152:155], v19, v17 op_sel_hi:[1,0,0] cbsz:2 blgp:2
	buffer_load_dwordx4 v3, s[40:43], 0 offen offset:2048 lds
	v_mfma_scale_f32_16x16x128_f8f6f4 a[156:159], v[138:143], v[72:77], a[156:159], v19, v17 op_sel:[1,0,0] op_sel_hi:[1,0,0] cbsz:2 blgp:2
	ds_read_b128 v[224:227], v29 offset:12832
	v_mfma_scale_f32_16x16x128_f8f6f4 a[188:191], v[138:143], v[78:83], a[188:191], v19, v17 op_sel:[1,1,0] op_sel_hi:[1,0,0] cbsz:2 blgp:2
	v_mfma_scale_f32_16x16x128_f8f6f4 a[184:187], v[132:137], v[78:83], a[184:187], v19, v17 op_sel:[0,1,0] op_sel_hi:[1,0,0] cbsz:2 blgp:2
	buffer_load_dwordx4 v3, s[40:43], 0 offen offset:3072 lds
	v_mfma_scale_f32_16x16x128_f8f6f4 a[180:183], v[126:131], v[78:83], a[180:183], v19, v17 op_sel:[1,1,0] op_sel_hi:[0,0,0] cbsz:2 blgp:2
	ds_read_b128 v[180:183], v26 offset:15872
	v_mfma_scale_f32_16x16x128_f8f6f4 a[176:179], v[120:125], v[78:83], a[176:179], v19, v17 op_sel:[0,1,0] op_sel_hi:[0,0,0] cbsz:2 blgp:2
	v_mfma_scale_f32_16x16x128_f8f6f4 a[172:175], v[114:119], v[78:83], a[172:175], v18, v17 op_sel:[1,1,0] op_sel_hi:[1,0,0] cbsz:2 blgp:2
	ds_read_b128 v[228:231], v29 offset:15872
	v_mfma_scale_f32_16x16x128_f8f6f4 a[168:171], v[108:113], v[78:83], a[168:171], v18, v17 op_sel:[0,1,0] op_sel_hi:[1,0,0] cbsz:2 blgp:2
	s_add_u32 m0, s58, 0xea00
	v_mfma_scale_f32_16x16x128_f8f6f4 a[164:167], v[102:107], v[78:83], a[164:167], v18, v17 op_sel:[1,1,0] op_sel_hi:[0,0,0] cbsz:2 blgp:2
	ds_read_b128 v[184:187], v26 offset:15888
	v_mfma_scale_f32_16x16x128_f8f6f4 a[160:163], v[96:101], v[78:83], a[160:163], v18, v17 op_sel:[0,1,0] op_sel_hi:[0,0,0] cbsz:2 blgp:2
	buffer_load_dwordx4 v42, s[40:43], 0 offen lds
	v_mfma_scale_f32_16x16x128_f8f6f4 a[192:195], v[96:101], v[84:89], a[192:195], v18, v17 op_sel_hi:[0,1,0] cbsz:2 blgp:2
	ds_read_b128 v[232:235], v29 offset:15888
	v_mfma_scale_f32_16x16x128_f8f6f4 a[196:199], v[102:107], v[84:89], a[196:199], v18, v17 op_sel:[1,0,0] op_sel_hi:[0,1,0] cbsz:2 blgp:2
	v_mfma_scale_f32_16x16x128_f8f6f4 a[200:203], v[108:113], v[84:89], a[200:203], v18, v17 op_sel_hi:[1,1,0] cbsz:2 blgp:2
	ds_read_b128 v[188:191], v26 offset:15904
	v_mfma_scale_f32_16x16x128_f8f6f4 a[204:207], v[114:119], v[84:89], a[204:207], v18, v17 op_sel:[1,0,0] op_sel_hi:[1,1,0] cbsz:2 blgp:2
	buffer_load_dwordx4 v42, s[40:43], 0 offen offset:1024 lds
	v_mfma_scale_f32_16x16x128_f8f6f4 a[208:211], v[120:125], v[84:89], a[208:211], v19, v17 op_sel_hi:[0,1,0] cbsz:2 blgp:2
	ds_read_b128 v[236:239], v29 offset:15904
	v_mfma_scale_f32_16x16x128_f8f6f4 a[212:215], v[126:131], v[84:89], a[212:215], v19, v17 op_sel:[1,0,0] op_sel_hi:[0,1,0] cbsz:2 blgp:2
	buffer_load_dwordx4 v42, s[40:43], 0 offen offset:2048 lds
	v_mfma_scale_f32_16x16x128_f8f6f4 a[216:219], v[132:137], v[84:89], a[216:219], v19, v17 op_sel_hi:[1,1,0] cbsz:2 blgp:2
	v_mfma_scale_f32_16x16x128_f8f6f4 a[220:223], v[138:143], v[84:89], a[220:223], v19, v17 op_sel:[1,0,0] op_sel_hi:[1,1,0] cbsz:2 blgp:2
	v_mfma_scale_f32_16x16x128_f8f6f4 a[252:255], v[138:143], v[90:95], a[252:255], v19, v17 op_sel:[1,1,0] op_sel_hi:[1,1,0] cbsz:2 blgp:2
	buffer_load_dwordx4 v42, s[40:43], 0 offen offset:3072 lds
	v_mfma_scale_f32_16x16x128_f8f6f4 a[248:251], v[132:137], v[90:95], a[248:251], v19, v17 op_sel:[0,1,0] op_sel_hi:[1,1,0] cbsz:2 blgp:2
	v_mfma_scale_f32_16x16x128_f8f6f4 a[244:247], v[126:131], v[90:95], a[244:247], v19, v17 op_sel:[1,1,0] op_sel_hi:[0,1,0] cbsz:2 blgp:2
	v_mfma_scale_f32_16x16x128_f8f6f4 a[240:243], v[120:125], v[90:95], a[240:243], v19, v17 op_sel:[0,1,0] op_sel_hi:[0,1,0] cbsz:2 blgp:2
	s_add_u32 s40, s40, 0x6000
	s_addc_u32 s41, s41, 0
	v_mfma_scale_f32_16x16x128_f8f6f4 a[236:239], v[114:119], v[90:95], a[236:239], v18, v17 op_sel:[1,1,0] op_sel_hi:[1,1,0] cbsz:2 blgp:2
	v_mfma_scale_f32_16x16x128_f8f6f4 a[232:235], v[108:113], v[90:95], a[232:235], v18, v17 op_sel:[0,1,0] op_sel_hi:[1,1,0] cbsz:2 blgp:2
	v_mfma_scale_f32_16x16x128_f8f6f4 a[228:231], v[102:107], v[90:95], a[228:231], v18, v17 op_sel:[1,1,0] op_sel_hi:[0,1,0] cbsz:2 blgp:2
	v_mfma_scale_f32_16x16x128_f8f6f4 a[224:227], v[96:101], v[90:95], a[224:227], v18, v17 op_sel:[0,1,0] op_sel_hi:[0,1,0] cbsz:2 blgp:2
	s_waitcnt vmcnt(14)
	s_waitcnt lgkmcnt(0)
	s_barrier
	ds_read_b64 v[16:17], v36
	v_mfma_scale_f32_16x16x128_f8f6f4 a[0:3], v[192:197], v[144:149], a[0:3], v22, v20 op_sel_hi:[0,0,0] cbsz:2 blgp:2
	s_add_u32 m0, s58, 0x19000
	v_mfma_scale_f32_16x16x128_f8f6f4 a[4:7], v[198:203], v[144:149], a[4:7], v22, v20 op_sel:[1,0,0] op_sel_hi:[0,0,0] cbsz:2 blgp:2
	ds_read_b64 v[18:19], v39
	v_mfma_scale_f32_16x16x128_f8f6f4 a[8:11], v[204:209], v[144:149], a[8:11], v22, v20 op_sel_hi:[1,0,0] cbsz:2 blgp:2
	buffer_load_dword v4, s[44:47], 0 offen lds
	v_mfma_scale_f32_16x16x128_f8f6f4 a[12:15], v[210:215], v[144:149], a[12:15], v22, v20 op_sel:[1,0,0] op_sel_hi:[1,0,0] cbsz:2 blgp:2
	ds_read_b128 v[48:51], v24
	v_mfma_scale_f32_16x16x128_f8f6f4 a[16:19], v[216:221], v[144:149], a[16:19], v23, v20 op_sel_hi:[0,0,0] cbsz:2 blgp:2
	v_mfma_scale_f32_16x16x128_f8f6f4 a[20:23], v[222:227], v[144:149], a[20:23], v23, v20 op_sel:[1,0,0] op_sel_hi:[0,0,0] cbsz:2 blgp:2
	ds_read_b128 v[96:99], v27
	v_mfma_scale_f32_16x16x128_f8f6f4 a[24:27], v[228:233], v[144:149], a[24:27], v23, v20 op_sel_hi:[1,0,0] cbsz:2 blgp:2
	s_add_u32 s44, s44, 0x4000
	s_addc_u32 s45, s45, 0
	v_mfma_scale_f32_16x16x128_f8f6f4 a[28:31], v[234:239], v[144:149], a[28:31], v23, v20 op_sel:[1,0,0] op_sel_hi:[1,0,0] cbsz:2 blgp:2
	ds_read_b128 v[52:55], v24 offset:16
	v_mfma_scale_f32_16x16x128_f8f6f4 a[60:63], v[234:239], v[150:155], a[60:63], v23, v20 op_sel:[1,1,0] op_sel_hi:[1,0,0] cbsz:2 blgp:2
	buffer_load_dword v4, s[48:51], 0 offen offset:256 lds
	v_mfma_scale_f32_16x16x128_f8f6f4 a[56:59], v[228:233], v[150:155], a[56:59], v23, v20 op_sel:[0,1,0] op_sel_hi:[1,0,0] cbsz:2 blgp:2
	ds_read_b128 v[100:103], v27 offset:16
	v_mfma_scale_f32_16x16x128_f8f6f4 a[52:55], v[222:227], v[150:155], a[52:55], v23, v20 op_sel:[1,1,0] op_sel_hi:[0,0,0] cbsz:2 blgp:2
	v_mfma_scale_f32_16x16x128_f8f6f4 a[48:51], v[216:221], v[150:155], a[48:51], v23, v20 op_sel:[0,1,0] op_sel_hi:[0,0,0] cbsz:2 blgp:2
	ds_read_b128 v[56:59], v24 offset:32
	v_mfma_scale_f32_16x16x128_f8f6f4 a[44:47], v[210:215], v[150:155], a[44:47], v22, v20 op_sel:[1,1,0] op_sel_hi:[1,0,0] cbsz:2 blgp:2
	s_add_u32 s48, s48, 0x4000
	s_addc_u32 s49, s49, 0
	v_mfma_scale_f32_16x16x128_f8f6f4 a[40:43], v[204:209], v[150:155], a[40:43], v22, v20 op_sel:[0,1,0] op_sel_hi:[1,0,0] cbsz:2 blgp:2
	v_mfma_scale_f32_16x16x128_f8f6f4 a[36:39], v[198:203], v[150:155], a[36:39], v22, v20 op_sel:[1,1,0] op_sel_hi:[0,0,0] cbsz:2 blgp:2
	ds_read_b128 v[104:107], v27 offset:32
	v_mfma_scale_f32_16x16x128_f8f6f4 a[32:35], v[192:197], v[150:155], a[32:35], v22, v20 op_sel:[0,1,0] op_sel_hi:[0,0,0] cbsz:2 blgp:2
	buffer_load_dwordx4 v3, s[36:39], 0 offen offset:512 lds
	v_mfma_scale_f32_16x16x128_f8f6f4 a[64:67], v[192:197], v[156:161], a[64:67], v22, v20 op_sel_hi:[0,1,0] cbsz:2 blgp:2
	ds_read_b128 v[60:63], v24 offset:3072
	v_mfma_scale_f32_16x16x128_f8f6f4 a[68:71], v[198:203], v[156:161], a[68:71], v22, v20 op_sel:[1,0,0] op_sel_hi:[0,1,0] cbsz:2 blgp:2
	buffer_load_dwordx4 v3, s[36:39], 0 offen offset:1536 lds
	v_mfma_scale_f32_16x16x128_f8f6f4 a[72:75], v[204:209], v[156:161], a[72:75], v22, v20 op_sel_hi:[1,1,0] cbsz:2 blgp:2
	ds_read_b128 v[108:111], v27 offset:3072
	v_mfma_scale_f32_16x16x128_f8f6f4 a[76:79], v[210:215], v[156:161], a[76:79], v22, v20 op_sel:[1,0,0] op_sel_hi:[1,1,0] cbsz:2 blgp:2
	v_mfma_scale_f32_16x16x128_f8f6f4 a[80:83], v[216:221], v[156:161], a[80:83], v23, v20 op_sel_hi:[0,1,0] cbsz:2 blgp:2
	ds_read_b128 v[64:67], v24 offset:3088
	v_mfma_scale_f32_16x16x128_f8f6f4 a[84:87], v[222:227], v[156:161], a[84:87], v23, v20 op_sel:[1,0,0] op_sel_hi:[0,1,0] cbsz:2 blgp:2
	buffer_load_dwordx4 v3, s[36:39], 0 offen offset:2560 lds
	v_mfma_scale_f32_16x16x128_f8f6f4 a[88:91], v[228:233], v[156:161], a[88:91], v23, v20 op_sel_hi:[1,1,0] cbsz:2 blgp:2
	ds_read_b128 v[112:115], v27 offset:3088
	v_mfma_scale_f32_16x16x128_f8f6f4 a[92:95], v[234:239], v[156:161], a[92:95], v23, v20 op_sel:[1,0,0] op_sel_hi:[1,1,0] cbsz:2 blgp:2
	buffer_load_dwordx4 v3, s[36:39], 0 offen offset:3584 lds
	v_mfma_scale_f32_16x16x128_f8f6f4 a[124:127], v[234:239], v[162:167], a[124:127], v23, v20 op_sel:[1,1,0] op_sel_hi:[1,1,0] cbsz:2 blgp:2
	ds_read_b128 v[68:71], v24 offset:3104
	v_mfma_scale_f32_16x16x128_f8f6f4 a[120:123], v[228:233], v[162:167], a[120:123], v23, v20 op_sel:[0,1,0] op_sel_hi:[1,1,0] cbsz:2 blgp:2
	v_mfma_scale_f32_16x16x128_f8f6f4 a[116:119], v[222:227], v[162:167], a[116:119], v23, v20 op_sel:[1,1,0] op_sel_hi:[0,1,0] cbsz:2 blgp:2
	s_add_u32 m0, s58, 0x1a200
	v_mfma_scale_f32_16x16x128_f8f6f4 a[112:115], v[216:221], v[162:167], a[112:115], v23, v20 op_sel:[0,1,0] op_sel_hi:[0,1,0] cbsz:2 blgp:2
	ds_read_b128 v[116:119], v27 offset:3104
	v_mfma_scale_f32_16x16x128_f8f6f4 a[108:111], v[210:215], v[162:167], a[108:111], v22, v20 op_sel:[1,1,0] op_sel_hi:[1,1,0] cbsz:2 blgp:2
	v_mfma_scale_f32_16x16x128_f8f6f4 a[104:107], v[204:209], v[162:167], a[104:107], v22, v20 op_sel:[0,1,0] op_sel_hi:[1,1,0] cbsz:2 blgp:2
	ds_read_b128 v[72:75], v24 offset:12800
	v_mfma_scale_f32_16x16x128_f8f6f4 a[100:103], v[198:203], v[162:167], a[100:103], v22, v20 op_sel:[1,1,0] op_sel_hi:[0,1,0] cbsz:2 blgp:2
	buffer_load_dwordx4 v8, s[36:39], 0 offen lds
	v_mfma_scale_f32_16x16x128_f8f6f4 a[96:99], v[192:197], v[162:167], a[96:99], v22, v20 op_sel:[0,1,0] op_sel_hi:[0,1,0] cbsz:2 blgp:2
	ds_read_b128 v[120:123], v27 offset:12800
	v_mfma_scale_f32_16x16x128_f8f6f4 a[128:131], v[192:197], v[168:173], a[128:131], v22, v21 op_sel_hi:[0,0,0] cbsz:2 blgp:2
	buffer_load_dwordx4 v8, s[36:39], 0 offen offset:1024 lds
	v_mfma_scale_f32_16x16x128_f8f6f4 a[132:135], v[198:203], v[168:173], a[132:135], v22, v21 op_sel:[1,0,0] op_sel_hi:[0,0,0] cbsz:2 blgp:2
	ds_read_b128 v[76:79], v24 offset:12816
	v_mfma_scale_f32_16x16x128_f8f6f4 a[136:139], v[204:209], v[168:173], a[136:139], v22, v21 op_sel_hi:[1,0,0] cbsz:2 blgp:2
	v_mfma_scale_f32_16x16x128_f8f6f4 a[140:143], v[210:215], v[168:173], a[140:143], v22, v21 op_sel:[1,0,0] op_sel_hi:[1,0,0] cbsz:2 blgp:2
	ds_read_b128 v[124:127], v27 offset:12816
	v_mfma_scale_f32_16x16x128_f8f6f4 a[144:147], v[216:221], v[168:173], a[144:147], v23, v21 op_sel_hi:[0,0,0] cbsz:2 blgp:2
	s_add_u32 s36, s36, 0x6000
	s_addc_u32 s37, s37, 0
	v_mfma_scale_f32_16x16x128_f8f6f4 a[148:151], v[222:227], v[168:173], a[148:151], v23, v21 op_sel:[1,0,0] op_sel_hi:[0,0,0] cbsz:2 blgp:2
	ds_read_b128 v[80:83], v24 offset:12832
	v_mfma_scale_f32_16x16x128_f8f6f4 a[152:155], v[228:233], v[168:173], a[152:155], v23, v21 op_sel_hi:[1,0,0] cbsz:2 blgp:2
	buffer_load_dwordx4 v3, s[40:43], 0 offen offset:2048 lds
	v_mfma_scale_f32_16x16x128_f8f6f4 a[156:159], v[234:239], v[168:173], a[156:159], v23, v21 op_sel:[1,0,0] op_sel_hi:[1,0,0] cbsz:2 blgp:2
	ds_read_b128 v[128:131], v27 offset:12832
	v_mfma_scale_f32_16x16x128_f8f6f4 a[188:191], v[234:239], v[174:179], a[188:191], v23, v21 op_sel:[1,1,0] op_sel_hi:[1,0,0] cbsz:2 blgp:2
	v_mfma_scale_f32_16x16x128_f8f6f4 a[184:187], v[228:233], v[174:179], a[184:187], v23, v21 op_sel:[0,1,0] op_sel_hi:[1,0,0] cbsz:2 blgp:2
	buffer_load_dwordx4 v3, s[40:43], 0 offen offset:3072 lds
	v_mfma_scale_f32_16x16x128_f8f6f4 a[180:183], v[222:227], v[174:179], a[180:183], v23, v21 op_sel:[1,1,0] op_sel_hi:[0,0,0] cbsz:2 blgp:2
	ds_read_b128 v[84:87], v24 offset:15872
	v_mfma_scale_f32_16x16x128_f8f6f4 a[176:179], v[216:221], v[174:179], a[176:179], v23, v21 op_sel:[0,1,0] op_sel_hi:[0,0,0] cbsz:2 blgp:2
	v_mfma_scale_f32_16x16x128_f8f6f4 a[172:175], v[210:215], v[174:179], a[172:175], v22, v21 op_sel:[1,1,0] op_sel_hi:[1,0,0] cbsz:2 blgp:2
	ds_read_b128 v[132:135], v27 offset:15872
	v_mfma_scale_f32_16x16x128_f8f6f4 a[168:171], v[204:209], v[174:179], a[168:171], v22, v21 op_sel:[0,1,0] op_sel_hi:[1,0,0] cbsz:2 blgp:2
	s_add_u32 m0, s58, 0x1b200
	v_mfma_scale_f32_16x16x128_f8f6f4 a[164:167], v[198:203], v[174:179], a[164:167], v22, v21 op_sel:[1,1,0] op_sel_hi:[0,0,0] cbsz:2 blgp:2
	ds_read_b128 v[88:91], v24 offset:15888
	v_mfma_scale_f32_16x16x128_f8f6f4 a[160:163], v[192:197], v[174:179], a[160:163], v22, v21 op_sel:[0,1,0] op_sel_hi:[0,0,0] cbsz:2 blgp:2
	buffer_load_dwordx4 v42, s[40:43], 0 offen lds
	v_mfma_scale_f32_16x16x128_f8f6f4 a[192:195], v[192:197], v[180:185], a[192:195], v22, v21 op_sel_hi:[0,1,0] cbsz:2 blgp:2
	ds_read_b128 v[136:139], v27 offset:15888
	v_mfma_scale_f32_16x16x128_f8f6f4 a[196:199], v[198:203], v[180:185], a[196:199], v22, v21 op_sel:[1,0,0] op_sel_hi:[0,1,0] cbsz:2 blgp:2
	v_mfma_scale_f32_16x16x128_f8f6f4 a[200:203], v[204:209], v[180:185], a[200:203], v22, v21 op_sel_hi:[1,1,0] cbsz:2 blgp:2
	ds_read_b128 v[92:95], v24 offset:15904
	v_mfma_scale_f32_16x16x128_f8f6f4 a[204:207], v[210:215], v[180:185], a[204:207], v22, v21 op_sel:[1,0,0] op_sel_hi:[1,1,0] cbsz:2 blgp:2
	buffer_load_dwordx4 v42, s[40:43], 0 offen offset:1024 lds
	v_mfma_scale_f32_16x16x128_f8f6f4 a[208:211], v[216:221], v[180:185], a[208:211], v23, v21 op_sel_hi:[0,1,0] cbsz:2 blgp:2
	ds_read_b128 v[140:143], v27 offset:15904
	v_mfma_scale_f32_16x16x128_f8f6f4 a[212:215], v[222:227], v[180:185], a[212:215], v23, v21 op_sel:[1,0,0] op_sel_hi:[0,1,0] cbsz:2 blgp:2
	buffer_load_dwordx4 v42, s[40:43], 0 offen offset:2048 lds
	v_mfma_scale_f32_16x16x128_f8f6f4 a[216:219], v[228:233], v[180:185], a[216:219], v23, v21 op_sel_hi:[1,1,0] cbsz:2 blgp:2
	v_mfma_scale_f32_16x16x128_f8f6f4 a[220:223], v[234:239], v[180:185], a[220:223], v23, v21 op_sel:[1,0,0] op_sel_hi:[1,1,0] cbsz:2 blgp:2
	v_mfma_scale_f32_16x16x128_f8f6f4 a[252:255], v[234:239], v[186:191], a[252:255], v23, v21 op_sel:[1,1,0] op_sel_hi:[1,1,0] cbsz:2 blgp:2
	buffer_load_dwordx4 v42, s[40:43], 0 offen offset:3072 lds
	v_mfma_scale_f32_16x16x128_f8f6f4 a[248:251], v[228:233], v[186:191], a[248:251], v23, v21 op_sel:[0,1,0] op_sel_hi:[1,1,0] cbsz:2 blgp:2
	v_mfma_scale_f32_16x16x128_f8f6f4 a[244:247], v[222:227], v[186:191], a[244:247], v23, v21 op_sel:[1,1,0] op_sel_hi:[0,1,0] cbsz:2 blgp:2
	v_mfma_scale_f32_16x16x128_f8f6f4 a[240:243], v[216:221], v[186:191], a[240:243], v23, v21 op_sel:[0,1,0] op_sel_hi:[0,1,0] cbsz:2 blgp:2
	s_add_u32 s40, s40, 0x6000
	s_addc_u32 s41, s41, 0
	v_mfma_scale_f32_16x16x128_f8f6f4 a[236:239], v[210:215], v[186:191], a[236:239], v22, v21 op_sel:[1,1,0] op_sel_hi:[1,1,0] cbsz:2 blgp:2
	v_mfma_scale_f32_16x16x128_f8f6f4 a[232:235], v[204:209], v[186:191], a[232:235], v22, v21 op_sel:[0,1,0] op_sel_hi:[1,1,0] cbsz:2 blgp:2
	v_mfma_scale_f32_16x16x128_f8f6f4 a[228:231], v[198:203], v[186:191], a[228:231], v22, v21 op_sel:[1,1,0] op_sel_hi:[0,1,0] cbsz:2 blgp:2
	v_mfma_scale_f32_16x16x128_f8f6f4 a[224:227], v[192:197], v[186:191], a[224:227], v22, v21 op_sel:[0,1,0] op_sel_hi:[0,1,0] cbsz:2 blgp:2
	s_sub_u32 s31, s31, 1
	s_cmp_gt_i32 s31, 0
	s_cbranch_scc1 63393
	s_waitcnt vmcnt(14)
	s_waitcnt lgkmcnt(0)
	s_barrier
	ds_read_b64 v[20:21], v37
	v_mfma_scale_f32_16x16x128_f8f6f4 a[0:3], v[96:101], v[48:53], a[0:3], v18, v16 op_sel_hi:[0,0,0] cbsz:2 blgp:2
	s_add_u32 m0, s58, 0
	v_mfma_scale_f32_16x16x128_f8f6f4 a[4:7], v[102:107], v[48:53], a[4:7], v18, v16 op_sel:[1,0,0] op_sel_hi:[0,0,0] cbsz:2 blgp:2
	ds_read_b64 v[22:23], v40
	v_mfma_scale_f32_16x16x128_f8f6f4 a[8:11], v[108:113], v[48:53], a[8:11], v18, v16 op_sel_hi:[1,0,0] cbsz:2 blgp:2
	buffer_load_dword v4, s[44:47], 0 offen lds
	v_mfma_scale_f32_16x16x128_f8f6f4 a[12:15], v[114:119], v[48:53], a[12:15], v18, v16 op_sel:[1,0,0] op_sel_hi:[1,0,0] cbsz:2 blgp:2
	ds_read_b128 v[144:147], v25
	v_mfma_scale_f32_16x16x128_f8f6f4 a[16:19], v[120:125], v[48:53], a[16:19], v19, v16 op_sel_hi:[0,0,0] cbsz:2 blgp:2
	v_mfma_scale_f32_16x16x128_f8f6f4 a[20:23], v[126:131], v[48:53], a[20:23], v19, v16 op_sel:[1,0,0] op_sel_hi:[0,0,0] cbsz:2 blgp:2
	ds_read_b128 v[192:195], v28
	v_mfma_scale_f32_16x16x128_f8f6f4 a[24:27], v[132:137], v[48:53], a[24:27], v19, v16 op_sel_hi:[1,0,0] cbsz:2 blgp:2
	s_add_u32 s44, s44, 0x4000
	s_addc_u32 s45, s45, 0
	v_mfma_scale_f32_16x16x128_f8f6f4 a[28:31], v[138:143], v[48:53], a[28:31], v19, v16 op_sel:[1,0,0] op_sel_hi:[1,0,0] cbsz:2 blgp:2
	ds_read_b128 v[148:151], v25 offset:16
	v_mfma_scale_f32_16x16x128_f8f6f4 a[60:63], v[138:143], v[54:59], a[60:63], v19, v16 op_sel:[1,1,0] op_sel_hi:[1,0,0] cbsz:2 blgp:2
	buffer_load_dword v4, s[48:51], 0 offen offset:256 lds
	v_mfma_scale_f32_16x16x128_f8f6f4 a[56:59], v[132:137], v[54:59], a[56:59], v19, v16 op_sel:[0,1,0] op_sel_hi:[1,0,0] cbsz:2 blgp:2
	ds_read_b128 v[196:199], v28 offset:16
	v_mfma_scale_f32_16x16x128_f8f6f4 a[52:55], v[126:131], v[54:59], a[52:55], v19, v16 op_sel:[1,1,0] op_sel_hi:[0,0,0] cbsz:2 blgp:2
	v_mfma_scale_f32_16x16x128_f8f6f4 a[48:51], v[120:125], v[54:59], a[48:51], v19, v16 op_sel:[0,1,0] op_sel_hi:[0,0,0] cbsz:2 blgp:2
	ds_read_b128 v[152:155], v25 offset:32
	v_mfma_scale_f32_16x16x128_f8f6f4 a[44:47], v[114:119], v[54:59], a[44:47], v18, v16 op_sel:[1,1,0] op_sel_hi:[1,0,0] cbsz:2 blgp:2
	s_add_u32 s48, s48, 0x4000
	s_addc_u32 s49, s49, 0
	v_mfma_scale_f32_16x16x128_f8f6f4 a[40:43], v[108:113], v[54:59], a[40:43], v18, v16 op_sel:[0,1,0] op_sel_hi:[1,0,0] cbsz:2 blgp:2
	v_mfma_scale_f32_16x16x128_f8f6f4 a[36:39], v[102:107], v[54:59], a[36:39], v18, v16 op_sel:[1,1,0] op_sel_hi:[0,0,0] cbsz:2 blgp:2
	ds_read_b128 v[200:203], v28 offset:32
	v_mfma_scale_f32_16x16x128_f8f6f4 a[32:35], v[96:101], v[54:59], a[32:35], v18, v16 op_sel:[0,1,0] op_sel_hi:[0,0,0] cbsz:2 blgp:2
	buffer_load_dwordx4 v3, s[36:39], 0 offen offset:512 lds
	v_mfma_scale_f32_16x16x128_f8f6f4 a[64:67], v[96:101], v[60:65], a[64:67], v18, v16 op_sel_hi:[0,1,0] cbsz:2 blgp:2
	ds_read_b128 v[156:159], v25 offset:3072
	v_mfma_scale_f32_16x16x128_f8f6f4 a[68:71], v[102:107], v[60:65], a[68:71], v18, v16 op_sel:[1,0,0] op_sel_hi:[0,1,0] cbsz:2 blgp:2
	buffer_load_dwordx4 v3, s[36:39], 0 offen offset:1536 lds
	v_mfma_scale_f32_16x16x128_f8f6f4 a[72:75], v[108:113], v[60:65], a[72:75], v18, v16 op_sel_hi:[1,1,0] cbsz:2 blgp:2
	ds_read_b128 v[204:207], v28 offset:3072
	v_mfma_scale_f32_16x16x128_f8f6f4 a[76:79], v[114:119], v[60:65], a[76:79], v18, v16 op_sel:[1,0,0] op_sel_hi:[1,1,0] cbsz:2 blgp:2
	v_mfma_scale_f32_16x16x128_f8f6f4 a[80:83], v[120:125], v[60:65], a[80:83], v19, v16 op_sel_hi:[0,1,0] cbsz:2 blgp:2
	ds_read_b128 v[160:163], v25 offset:3088
	v_mfma_scale_f32_16x16x128_f8f6f4 a[84:87], v[126:131], v[60:65], a[84:87], v19, v16 op_sel:[1,0,0] op_sel_hi:[0,1,0] cbsz:2 blgp:2
	buffer_load_dwordx4 v3, s[36:39], 0 offen offset:2560 lds
	v_mfma_scale_f32_16x16x128_f8f6f4 a[88:91], v[132:137], v[60:65], a[88:91], v19, v16 op_sel_hi:[1,1,0] cbsz:2 blgp:2
	ds_read_b128 v[208:211], v28 offset:3088
	v_mfma_scale_f32_16x16x128_f8f6f4 a[92:95], v[138:143], v[60:65], a[92:95], v19, v16 op_sel:[1,0,0] op_sel_hi:[1,1,0] cbsz:2 blgp:2
	buffer_load_dwordx4 v3, s[36:39], 0 offen offset:3584 lds
	v_mfma_scale_f32_16x16x128_f8f6f4 a[124:127], v[138:143], v[66:71], a[124:127], v19, v16 op_sel:[1,1,0] op_sel_hi:[1,1,0] cbsz:2 blgp:2
	ds_read_b128 v[164:167], v25 offset:3104
	v_mfma_scale_f32_16x16x128_f8f6f4 a[120:123], v[132:137], v[66:71], a[120:123], v19, v16 op_sel:[0,1,0] op_sel_hi:[1,1,0] cbsz:2 blgp:2
	v_mfma_scale_f32_16x16x128_f8f6f4 a[116:119], v[126:131], v[66:71], a[116:119], v19, v16 op_sel:[1,1,0] op_sel_hi:[0,1,0] cbsz:2 blgp:2
	s_add_u32 m0, s58, 0x1200
	v_mfma_scale_f32_16x16x128_f8f6f4 a[112:115], v[120:125], v[66:71], a[112:115], v19, v16 op_sel:[0,1,0] op_sel_hi:[0,1,0] cbsz:2 blgp:2
	ds_read_b128 v[212:215], v28 offset:3104
	v_mfma_scale_f32_16x16x128_f8f6f4 a[108:111], v[114:119], v[66:71], a[108:111], v18, v16 op_sel:[1,1,0] op_sel_hi:[1,1,0] cbsz:2 blgp:2
	v_mfma_scale_f32_16x16x128_f8f6f4 a[104:107], v[108:113], v[66:71], a[104:107], v18, v16 op_sel:[0,1,0] op_sel_hi:[1,1,0] cbsz:2 blgp:2
	ds_read_b128 v[168:171], v25 offset:12800
	v_mfma_scale_f32_16x16x128_f8f6f4 a[100:103], v[102:107], v[66:71], a[100:103], v18, v16 op_sel:[1,1,0] op_sel_hi:[0,1,0] cbsz:2 blgp:2
	buffer_load_dwordx4 v8, s[36:39], 0 offen lds
	v_mfma_scale_f32_16x16x128_f8f6f4 a[96:99], v[96:101], v[66:71], a[96:99], v18, v16 op_sel:[0,1,0] op_sel_hi:[0,1,0] cbsz:2 blgp:2
	ds_read_b128 v[216:219], v28 offset:12800
	v_mfma_scale_f32_16x16x128_f8f6f4 a[128:131], v[96:101], v[72:77], a[128:131], v18, v17 op_sel_hi:[0,0,0] cbsz:2 blgp:2
	buffer_load_dwordx4 v8, s[36:39], 0 offen offset:1024 lds
	v_mfma_scale_f32_16x16x128_f8f6f4 a[132:135], v[102:107], v[72:77], a[132:135], v18, v17 op_sel:[1,0,0] op_sel_hi:[0,0,0] cbsz:2 blgp:2
	ds_read_b128 v[172:175], v25 offset:12816
	v_mfma_scale_f32_16x16x128_f8f6f4 a[136:139], v[108:113], v[72:77], a[136:139], v18, v17 op_sel_hi:[1,0,0] cbsz:2 blgp:2
	v_mfma_scale_f32_16x16x128_f8f6f4 a[140:143], v[114:119], v[72:77], a[140:143], v18, v17 op_sel:[1,0,0] op_sel_hi:[1,0,0] cbsz:2 blgp:2
	ds_read_b128 v[220:223], v28 offset:12816
	v_mfma_scale_f32_16x16x128_f8f6f4 a[144:147], v[120:125], v[72:77], a[144:147], v19, v17 op_sel_hi:[0,0,0] cbsz:2 blgp:2
	s_add_u32 s36, s36, 0x6000
	s_addc_u32 s37, s37, 0
	v_mfma_scale_f32_16x16x128_f8f6f4 a[148:151], v[126:131], v[72:77], a[148:151], v19, v17 op_sel:[1,0,0] op_sel_hi:[0,0,0] cbsz:2 blgp:2
	ds_read_b128 v[176:179], v25 offset:12832
	v_mfma_scale_f32_16x16x128_f8f6f4 a[152:155], v[132:137], v[72:77], a[152:155], v19, v17 op_sel_hi:[1,0,0] cbsz:2 blgp:2
	buffer_load_dwordx4 v3, s[40:43], 0 offen offset:2048 lds
	v_mfma_scale_f32_16x16x128_f8f6f4 a[156:159], v[138:143], v[72:77], a[156:159], v19, v17 op_sel:[1,0,0] op_sel_hi:[1,0,0] cbsz:2 blgp:2
	ds_read_b128 v[224:227], v28 offset:12832
	v_mfma_scale_f32_16x16x128_f8f6f4 a[188:191], v[138:143], v[78:83], a[188:191], v19, v17 op_sel:[1,1,0] op_sel_hi:[1,0,0] cbsz:2 blgp:2
	v_mfma_scale_f32_16x16x128_f8f6f4 a[184:187], v[132:137], v[78:83], a[184:187], v19, v17 op_sel:[0,1,0] op_sel_hi:[1,0,0] cbsz:2 blgp:2
	buffer_load_dwordx4 v3, s[40:43], 0 offen offset:3072 lds
	v_mfma_scale_f32_16x16x128_f8f6f4 a[180:183], v[126:131], v[78:83], a[180:183], v19, v17 op_sel:[1,1,0] op_sel_hi:[0,0,0] cbsz:2 blgp:2
	ds_read_b128 v[180:183], v25 offset:15872
	v_mfma_scale_f32_16x16x128_f8f6f4 a[176:179], v[120:125], v[78:83], a[176:179], v19, v17 op_sel:[0,1,0] op_sel_hi:[0,0,0] cbsz:2 blgp:2
	v_mfma_scale_f32_16x16x128_f8f6f4 a[172:175], v[114:119], v[78:83], a[172:175], v18, v17 op_sel:[1,1,0] op_sel_hi:[1,0,0] cbsz:2 blgp:2
	ds_read_b128 v[228:231], v28 offset:15872
	v_mfma_scale_f32_16x16x128_f8f6f4 a[168:171], v[108:113], v[78:83], a[168:171], v18, v17 op_sel:[0,1,0] op_sel_hi:[1,0,0] cbsz:2 blgp:2
	s_add_u32 m0, s58, 0x2200
	v_mfma_scale_f32_16x16x128_f8f6f4 a[164:167], v[102:107], v[78:83], a[164:167], v18, v17 op_sel:[1,1,0] op_sel_hi:[0,0,0] cbsz:2 blgp:2
	ds_read_b128 v[184:187], v25 offset:15888
	v_mfma_scale_f32_16x16x128_f8f6f4 a[160:163], v[96:101], v[78:83], a[160:163], v18, v17 op_sel:[0,1,0] op_sel_hi:[0,0,0] cbsz:2 blgp:2
	buffer_load_dwordx4 v42, s[40:43], 0 offen lds
	v_mfma_scale_f32_16x16x128_f8f6f4 a[192:195], v[96:101], v[84:89], a[192:195], v18, v17 op_sel_hi:[0,1,0] cbsz:2 blgp:2
	ds_read_b128 v[232:235], v28 offset:15888
	v_mfma_scale_f32_16x16x128_f8f6f4 a[196:199], v[102:107], v[84:89], a[196:199], v18, v17 op_sel:[1,0,0] op_sel_hi:[0,1,0] cbsz:2 blgp:2
	v_mfma_scale_f32_16x16x128_f8f6f4 a[200:203], v[108:113], v[84:89], a[200:203], v18, v17 op_sel_hi:[1,1,0] cbsz:2 blgp:2
	ds_read_b128 v[188:191], v25 offset:15904
	v_mfma_scale_f32_16x16x128_f8f6f4 a[204:207], v[114:119], v[84:89], a[204:207], v18, v17 op_sel:[1,0,0] op_sel_hi:[1,1,0] cbsz:2 blgp:2
	buffer_load_dwordx4 v42, s[40:43], 0 offen offset:1024 lds
	v_mfma_scale_f32_16x16x128_f8f6f4 a[208:211], v[120:125], v[84:89], a[208:211], v19, v17 op_sel_hi:[0,1,0] cbsz:2 blgp:2
	ds_read_b128 v[236:239], v28 offset:15904
	v_mfma_scale_f32_16x16x128_f8f6f4 a[212:215], v[126:131], v[84:89], a[212:215], v19, v17 op_sel:[1,0,0] op_sel_hi:[0,1,0] cbsz:2 blgp:2
	buffer_load_dwordx4 v42, s[40:43], 0 offen offset:2048 lds
	v_mfma_scale_f32_16x16x128_f8f6f4 a[216:219], v[132:137], v[84:89], a[216:219], v19, v17 op_sel_hi:[1,1,0] cbsz:2 blgp:2
	v_mfma_scale_f32_16x16x128_f8f6f4 a[220:223], v[138:143], v[84:89], a[220:223], v19, v17 op_sel:[1,0,0] op_sel_hi:[1,1,0] cbsz:2 blgp:2
	v_mfma_scale_f32_16x16x128_f8f6f4 a[252:255], v[138:143], v[90:95], a[252:255], v19, v17 op_sel:[1,1,0] op_sel_hi:[1,1,0] cbsz:2 blgp:2
	buffer_load_dwordx4 v42, s[40:43], 0 offen offset:3072 lds
	v_mfma_scale_f32_16x16x128_f8f6f4 a[248:251], v[132:137], v[90:95], a[248:251], v19, v17 op_sel:[0,1,0] op_sel_hi:[1,1,0] cbsz:2 blgp:2
	v_mfma_scale_f32_16x16x128_f8f6f4 a[244:247], v[126:131], v[90:95], a[244:247], v19, v17 op_sel:[1,1,0] op_sel_hi:[0,1,0] cbsz:2 blgp:2
	v_mfma_scale_f32_16x16x128_f8f6f4 a[240:243], v[120:125], v[90:95], a[240:243], v19, v17 op_sel:[0,1,0] op_sel_hi:[0,1,0] cbsz:2 blgp:2
	s_add_u32 s40, s40, 0x6000
	s_addc_u32 s41, s41, 0
	v_mfma_scale_f32_16x16x128_f8f6f4 a[236:239], v[114:119], v[90:95], a[236:239], v18, v17 op_sel:[1,1,0] op_sel_hi:[1,1,0] cbsz:2 blgp:2
	v_mfma_scale_f32_16x16x128_f8f6f4 a[232:235], v[108:113], v[90:95], a[232:235], v18, v17 op_sel:[0,1,0] op_sel_hi:[1,1,0] cbsz:2 blgp:2
	v_mfma_scale_f32_16x16x128_f8f6f4 a[228:231], v[102:107], v[90:95], a[228:231], v18, v17 op_sel:[1,1,0] op_sel_hi:[0,1,0] cbsz:2 blgp:2
	v_mfma_scale_f32_16x16x128_f8f6f4 a[224:227], v[96:101], v[90:95], a[224:227], v18, v17 op_sel:[0,1,0] op_sel_hi:[0,1,0] cbsz:2 blgp:2
	s_waitcnt vmcnt(14)
	s_waitcnt lgkmcnt(0)
	s_barrier
	ds_read_b64 v[16:17], v38
	v_mfma_scale_f32_16x16x128_f8f6f4 a[0:3], v[192:197], v[144:149], a[0:3], v22, v20 op_sel_hi:[0,0,0] cbsz:2 blgp:2
	s_add_u32 m0, s58, 0xc800
	v_mfma_scale_f32_16x16x128_f8f6f4 a[4:7], v[198:203], v[144:149], a[4:7], v22, v20 op_sel:[1,0,0] op_sel_hi:[0,0,0] cbsz:2 blgp:2
	ds_read_b64 v[18:19], v41
	v_mfma_scale_f32_16x16x128_f8f6f4 a[8:11], v[204:209], v[144:149], a[8:11], v22, v20 op_sel_hi:[1,0,0] cbsz:2 blgp:2
	buffer_load_dword v4, s[44:47], 0 offen lds
	v_mfma_scale_f32_16x16x128_f8f6f4 a[12:15], v[210:215], v[144:149], a[12:15], v22, v20 op_sel:[1,0,0] op_sel_hi:[1,0,0] cbsz:2 blgp:2
	ds_read_b128 v[48:51], v26
	v_mfma_scale_f32_16x16x128_f8f6f4 a[16:19], v[216:221], v[144:149], a[16:19], v23, v20 op_sel_hi:[0,0,0] cbsz:2 blgp:2
	v_mfma_scale_f32_16x16x128_f8f6f4 a[20:23], v[222:227], v[144:149], a[20:23], v23, v20 op_sel:[1,0,0] op_sel_hi:[0,0,0] cbsz:2 blgp:2
	ds_read_b128 v[96:99], v29
	v_mfma_scale_f32_16x16x128_f8f6f4 a[24:27], v[228:233], v[144:149], a[24:27], v23, v20 op_sel_hi:[1,0,0] cbsz:2 blgp:2
	s_add_u32 s44, s44, 0x4000
	s_addc_u32 s45, s45, 0
	v_mfma_scale_f32_16x16x128_f8f6f4 a[28:31], v[234:239], v[144:149], a[28:31], v23, v20 op_sel:[1,0,0] op_sel_hi:[1,0,0] cbsz:2 blgp:2
	ds_read_b128 v[52:55], v26 offset:16
	v_mfma_scale_f32_16x16x128_f8f6f4 a[60:63], v[234:239], v[150:155], a[60:63], v23, v20 op_sel:[1,1,0] op_sel_hi:[1,0,0] cbsz:2 blgp:2
	buffer_load_dword v4, s[48:51], 0 offen offset:256 lds
	v_mfma_scale_f32_16x16x128_f8f6f4 a[56:59], v[228:233], v[150:155], a[56:59], v23, v20 op_sel:[0,1,0] op_sel_hi:[1,0,0] cbsz:2 blgp:2
	ds_read_b128 v[100:103], v29 offset:16
	v_mfma_scale_f32_16x16x128_f8f6f4 a[52:55], v[222:227], v[150:155], a[52:55], v23, v20 op_sel:[1,1,0] op_sel_hi:[0,0,0] cbsz:2 blgp:2
	v_mfma_scale_f32_16x16x128_f8f6f4 a[48:51], v[216:221], v[150:155], a[48:51], v23, v20 op_sel:[0,1,0] op_sel_hi:[0,0,0] cbsz:2 blgp:2
	ds_read_b128 v[56:59], v26 offset:32
	v_mfma_scale_f32_16x16x128_f8f6f4 a[44:47], v[210:215], v[150:155], a[44:47], v22, v20 op_sel:[1,1,0] op_sel_hi:[1,0,0] cbsz:2 blgp:2
	s_add_u32 s48, s48, 0x4000
	s_addc_u32 s49, s49, 0
	v_mfma_scale_f32_16x16x128_f8f6f4 a[40:43], v[204:209], v[150:155], a[40:43], v22, v20 op_sel:[0,1,0] op_sel_hi:[1,0,0] cbsz:2 blgp:2
	v_mfma_scale_f32_16x16x128_f8f6f4 a[36:39], v[198:203], v[150:155], a[36:39], v22, v20 op_sel:[1,1,0] op_sel_hi:[0,0,0] cbsz:2 blgp:2
	ds_read_b128 v[104:107], v29 offset:32
	v_mfma_scale_f32_16x16x128_f8f6f4 a[32:35], v[192:197], v[150:155], a[32:35], v22, v20 op_sel:[0,1,0] op_sel_hi:[0,0,0] cbsz:2 blgp:2
	buffer_load_dwordx4 v3, s[36:39], 0 offen offset:512 lds
	v_mfma_scale_f32_16x16x128_f8f6f4 a[64:67], v[192:197], v[156:161], a[64:67], v22, v20 op_sel_hi:[0,1,0] cbsz:2 blgp:2
	ds_read_b128 v[60:63], v26 offset:3072
	v_mfma_scale_f32_16x16x128_f8f6f4 a[68:71], v[198:203], v[156:161], a[68:71], v22, v20 op_sel:[1,0,0] op_sel_hi:[0,1,0] cbsz:2 blgp:2
	buffer_load_dwordx4 v3, s[36:39], 0 offen offset:1536 lds
	v_mfma_scale_f32_16x16x128_f8f6f4 a[72:75], v[204:209], v[156:161], a[72:75], v22, v20 op_sel_hi:[1,1,0] cbsz:2 blgp:2
	ds_read_b128 v[108:111], v29 offset:3072
	v_mfma_scale_f32_16x16x128_f8f6f4 a[76:79], v[210:215], v[156:161], a[76:79], v22, v20 op_sel:[1,0,0] op_sel_hi:[1,1,0] cbsz:2 blgp:2
	v_mfma_scale_f32_16x16x128_f8f6f4 a[80:83], v[216:221], v[156:161], a[80:83], v23, v20 op_sel_hi:[0,1,0] cbsz:2 blgp:2
	ds_read_b128 v[64:67], v26 offset:3088
	v_mfma_scale_f32_16x16x128_f8f6f4 a[84:87], v[222:227], v[156:161], a[84:87], v23, v20 op_sel:[1,0,0] op_sel_hi:[0,1,0] cbsz:2 blgp:2
	buffer_load_dwordx4 v3, s[36:39], 0 offen offset:2560 lds
	v_mfma_scale_f32_16x16x128_f8f6f4 a[88:91], v[228:233], v[156:161], a[88:91], v23, v20 op_sel_hi:[1,1,0] cbsz:2 blgp:2
	ds_read_b128 v[112:115], v29 offset:3088
	v_mfma_scale_f32_16x16x128_f8f6f4 a[92:95], v[234:239], v[156:161], a[92:95], v23, v20 op_sel:[1,0,0] op_sel_hi:[1,1,0] cbsz:2 blgp:2
	buffer_load_dwordx4 v3, s[36:39], 0 offen offset:3584 lds
	v_mfma_scale_f32_16x16x128_f8f6f4 a[124:127], v[234:239], v[162:167], a[124:127], v23, v20 op_sel:[1,1,0] op_sel_hi:[1,1,0] cbsz:2 blgp:2
	ds_read_b128 v[68:71], v26 offset:3104
	v_mfma_scale_f32_16x16x128_f8f6f4 a[120:123], v[228:233], v[162:167], a[120:123], v23, v20 op_sel:[0,1,0] op_sel_hi:[1,1,0] cbsz:2 blgp:2
	v_mfma_scale_f32_16x16x128_f8f6f4 a[116:119], v[222:227], v[162:167], a[116:119], v23, v20 op_sel:[1,1,0] op_sel_hi:[0,1,0] cbsz:2 blgp:2
	s_add_u32 m0, s58, 0xda00
	v_mfma_scale_f32_16x16x128_f8f6f4 a[112:115], v[216:221], v[162:167], a[112:115], v23, v20 op_sel:[0,1,0] op_sel_hi:[0,1,0] cbsz:2 blgp:2
	ds_read_b128 v[116:119], v29 offset:3104
	v_mfma_scale_f32_16x16x128_f8f6f4 a[108:111], v[210:215], v[162:167], a[108:111], v22, v20 op_sel:[1,1,0] op_sel_hi:[1,1,0] cbsz:2 blgp:2
	v_mfma_scale_f32_16x16x128_f8f6f4 a[104:107], v[204:209], v[162:167], a[104:107], v22, v20 op_sel:[0,1,0] op_sel_hi:[1,1,0] cbsz:2 blgp:2
	ds_read_b128 v[72:75], v26 offset:12800
	v_mfma_scale_f32_16x16x128_f8f6f4 a[100:103], v[198:203], v[162:167], a[100:103], v22, v20 op_sel:[1,1,0] op_sel_hi:[0,1,0] cbsz:2 blgp:2
	buffer_load_dwordx4 v8, s[36:39], 0 offen lds
	v_mfma_scale_f32_16x16x128_f8f6f4 a[96:99], v[192:197], v[162:167], a[96:99], v22, v20 op_sel:[0,1,0] op_sel_hi:[0,1,0] cbsz:2 blgp:2
	ds_read_b128 v[120:123], v29 offset:12800
	v_mfma_scale_f32_16x16x128_f8f6f4 a[128:131], v[192:197], v[168:173], a[128:131], v22, v21 op_sel_hi:[0,0,0] cbsz:2 blgp:2
	buffer_load_dwordx4 v8, s[36:39], 0 offen offset:1024 lds
	v_mfma_scale_f32_16x16x128_f8f6f4 a[132:135], v[198:203], v[168:173], a[132:135], v22, v21 op_sel:[1,0,0] op_sel_hi:[0,0,0] cbsz:2 blgp:2
	ds_read_b128 v[76:79], v26 offset:12816
	v_mfma_scale_f32_16x16x128_f8f6f4 a[136:139], v[204:209], v[168:173], a[136:139], v22, v21 op_sel_hi:[1,0,0] cbsz:2 blgp:2
	v_mfma_scale_f32_16x16x128_f8f6f4 a[140:143], v[210:215], v[168:173], a[140:143], v22, v21 op_sel:[1,0,0] op_sel_hi:[1,0,0] cbsz:2 blgp:2
	ds_read_b128 v[124:127], v29 offset:12816
	v_mfma_scale_f32_16x16x128_f8f6f4 a[144:147], v[216:221], v[168:173], a[144:147], v23, v21 op_sel_hi:[0,0,0] cbsz:2 blgp:2
	s_add_u32 s36, s36, 0x6000
	s_addc_u32 s37, s37, 0
	v_mfma_scale_f32_16x16x128_f8f6f4 a[148:151], v[222:227], v[168:173], a[148:151], v23, v21 op_sel:[1,0,0] op_sel_hi:[0,0,0] cbsz:2 blgp:2
	ds_read_b128 v[80:83], v26 offset:12832
	v_mfma_scale_f32_16x16x128_f8f6f4 a[152:155], v[228:233], v[168:173], a[152:155], v23, v21 op_sel_hi:[1,0,0] cbsz:2 blgp:2
	buffer_load_dwordx4 v3, s[40:43], 0 offen offset:2048 lds
	v_mfma_scale_f32_16x16x128_f8f6f4 a[156:159], v[234:239], v[168:173], a[156:159], v23, v21 op_sel:[1,0,0] op_sel_hi:[1,0,0] cbsz:2 blgp:2
	ds_read_b128 v[128:131], v29 offset:12832
	v_mfma_scale_f32_16x16x128_f8f6f4 a[188:191], v[234:239], v[174:179], a[188:191], v23, v21 op_sel:[1,1,0] op_sel_hi:[1,0,0] cbsz:2 blgp:2
	v_mfma_scale_f32_16x16x128_f8f6f4 a[184:187], v[228:233], v[174:179], a[184:187], v23, v21 op_sel:[0,1,0] op_sel_hi:[1,0,0] cbsz:2 blgp:2
	buffer_load_dwordx4 v3, s[40:43], 0 offen offset:3072 lds
	v_mfma_scale_f32_16x16x128_f8f6f4 a[180:183], v[222:227], v[174:179], a[180:183], v23, v21 op_sel:[1,1,0] op_sel_hi:[0,0,0] cbsz:2 blgp:2
	ds_read_b128 v[84:87], v26 offset:15872
	v_mfma_scale_f32_16x16x128_f8f6f4 a[176:179], v[216:221], v[174:179], a[176:179], v23, v21 op_sel:[0,1,0] op_sel_hi:[0,0,0] cbsz:2 blgp:2
	v_mfma_scale_f32_16x16x128_f8f6f4 a[172:175], v[210:215], v[174:179], a[172:175], v22, v21 op_sel:[1,1,0] op_sel_hi:[1,0,0] cbsz:2 blgp:2
	ds_read_b128 v[132:135], v29 offset:15872
	v_mfma_scale_f32_16x16x128_f8f6f4 a[168:171], v[204:209], v[174:179], a[168:171], v22, v21 op_sel:[0,1,0] op_sel_hi:[1,0,0] cbsz:2 blgp:2
	s_add_u32 m0, s58, 0xea00
	v_mfma_scale_f32_16x16x128_f8f6f4 a[164:167], v[198:203], v[174:179], a[164:167], v22, v21 op_sel:[1,1,0] op_sel_hi:[0,0,0] cbsz:2 blgp:2
	ds_read_b128 v[88:91], v26 offset:15888
	v_mfma_scale_f32_16x16x128_f8f6f4 a[160:163], v[192:197], v[174:179], a[160:163], v22, v21 op_sel:[0,1,0] op_sel_hi:[0,0,0] cbsz:2 blgp:2
	buffer_load_dwordx4 v42, s[40:43], 0 offen lds
	v_mfma_scale_f32_16x16x128_f8f6f4 a[192:195], v[192:197], v[180:185], a[192:195], v22, v21 op_sel_hi:[0,1,0] cbsz:2 blgp:2
	ds_read_b128 v[136:139], v29 offset:15888
	v_mfma_scale_f32_16x16x128_f8f6f4 a[196:199], v[198:203], v[180:185], a[196:199], v22, v21 op_sel:[1,0,0] op_sel_hi:[0,1,0] cbsz:2 blgp:2
	v_mfma_scale_f32_16x16x128_f8f6f4 a[200:203], v[204:209], v[180:185], a[200:203], v22, v21 op_sel_hi:[1,1,0] cbsz:2 blgp:2
	ds_read_b128 v[92:95], v26 offset:15904
	v_mfma_scale_f32_16x16x128_f8f6f4 a[204:207], v[210:215], v[180:185], a[204:207], v22, v21 op_sel:[1,0,0] op_sel_hi:[1,1,0] cbsz:2 blgp:2
	buffer_load_dwordx4 v42, s[40:43], 0 offen offset:1024 lds
	v_mfma_scale_f32_16x16x128_f8f6f4 a[208:211], v[216:221], v[180:185], a[208:211], v23, v21 op_sel_hi:[0,1,0] cbsz:2 blgp:2
	ds_read_b128 v[140:143], v29 offset:15904
	v_mfma_scale_f32_16x16x128_f8f6f4 a[212:215], v[222:227], v[180:185], a[212:215], v23, v21 op_sel:[1,0,0] op_sel_hi:[0,1,0] cbsz:2 blgp:2
	buffer_load_dwordx4 v42, s[40:43], 0 offen offset:2048 lds
	v_mfma_scale_f32_16x16x128_f8f6f4 a[216:219], v[228:233], v[180:185], a[216:219], v23, v21 op_sel_hi:[1,1,0] cbsz:2 blgp:2
	v_mfma_scale_f32_16x16x128_f8f6f4 a[220:223], v[234:239], v[180:185], a[220:223], v23, v21 op_sel:[1,0,0] op_sel_hi:[1,1,0] cbsz:2 blgp:2
	v_mfma_scale_f32_16x16x128_f8f6f4 a[252:255], v[234:239], v[186:191], a[252:255], v23, v21 op_sel:[1,1,0] op_sel_hi:[1,1,0] cbsz:2 blgp:2
	buffer_load_dwordx4 v42, s[40:43], 0 offen offset:3072 lds
	v_mfma_scale_f32_16x16x128_f8f6f4 a[248:251], v[228:233], v[186:191], a[248:251], v23, v21 op_sel:[0,1,0] op_sel_hi:[1,1,0] cbsz:2 blgp:2
	v_mfma_scale_f32_16x16x128_f8f6f4 a[244:247], v[222:227], v[186:191], a[244:247], v23, v21 op_sel:[1,1,0] op_sel_hi:[0,1,0] cbsz:2 blgp:2
	v_mfma_scale_f32_16x16x128_f8f6f4 a[240:243], v[216:221], v[186:191], a[240:243], v23, v21 op_sel:[0,1,0] op_sel_hi:[0,1,0] cbsz:2 blgp:2
	s_add_u32 s40, s40, 0x6000
	s_addc_u32 s41, s41, 0
	v_mfma_scale_f32_16x16x128_f8f6f4 a[236:239], v[210:215], v[186:191], a[236:239], v22, v21 op_sel:[1,1,0] op_sel_hi:[1,1,0] cbsz:2 blgp:2
	v_mfma_scale_f32_16x16x128_f8f6f4 a[232:235], v[204:209], v[186:191], a[232:235], v22, v21 op_sel:[0,1,0] op_sel_hi:[1,1,0] cbsz:2 blgp:2
	v_mfma_scale_f32_16x16x128_f8f6f4 a[228:231], v[198:203], v[186:191], a[228:231], v22, v21 op_sel:[1,1,0] op_sel_hi:[0,1,0] cbsz:2 blgp:2
	v_mfma_scale_f32_16x16x128_f8f6f4 a[224:227], v[192:197], v[186:191], a[224:227], v22, v21 op_sel:[0,1,0] op_sel_hi:[0,1,0] cbsz:2 blgp:2
	s_waitcnt lgkmcnt(0)
	s_nop 15
	s_nop 15
	s_nop 15
	s_nop 15
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
	s_mov_b32 m0, -1
	s_waitcnt vmcnt(0)
	s_barrier
	s_mul_i32 s30, s20, 0x8800
	v_and_b32_e32 v2, 15, v1
	v_mul_u32_u24_e32 v2, 0x110, v2
	v_lshrrev_b32_e32 v3, 4, v1
	v_lshlrev_b32_e32 v3, 3, v3
	v_add_u32_e32 v8, v2, v3
	v_add_u32_e32 v8, s30, v8
	v_lshrrev_b32_e32 v2, 4, v1
	v_mul_u32_u24_e32 v2, 0x110, v2
	v_and_b32_e32 v3, 15, v1
	v_lshlrev_b32_e32 v3, 4, v3
	v_add_u32_e32 v9, v2, v3
	v_add_u32_e32 v9, s30, v9
	v_lshrrev_b32_e32 v2, 4, v1
	v_mul_u32_u24_e32 v2, 0x2000, v2
	v_and_b32_e32 v3, 15, v1
	v_lshlrev_b32_e32 v3, 4, v3
	v_add_u32_e32 v7, v2, v3
	v_accvgpr_read_b32 v16, a0
	v_accvgpr_read_b32 v17, a1
	v_accvgpr_read_b32 v18, a2
	v_accvgpr_read_b32 v19, a3
	s_nop 2
	v_cvt_pk_bf16_f32 v20, v16, v17
	v_cvt_pk_bf16_f32 v21, v18, v19
	ds_write_b64 v8, v[20:21]
	v_accvgpr_read_b32 v16, a4
	v_accvgpr_read_b32 v17, a5
	v_accvgpr_read_b32 v18, a6
	v_accvgpr_read_b32 v19, a7
	s_nop 2
	v_cvt_pk_bf16_f32 v20, v16, v17
	v_cvt_pk_bf16_f32 v21, v18, v19
	ds_write_b64 v8, v[20:21] offset:32
	v_accvgpr_read_b32 v16, a8
	v_accvgpr_read_b32 v17, a9
	v_accvgpr_read_b32 v18, a10
	v_accvgpr_read_b32 v19, a11
	s_nop 2
	v_cvt_pk_bf16_f32 v20, v16, v17
	v_cvt_pk_bf16_f32 v21, v18, v19
	ds_write_b64 v8, v[20:21] offset:64
	v_accvgpr_read_b32 v16, a12
	v_accvgpr_read_b32 v17, a13
	v_accvgpr_read_b32 v18, a14
	v_accvgpr_read_b32 v19, a15
	s_nop 2
	v_cvt_pk_bf16_f32 v20, v16, v17
	v_cvt_pk_bf16_f32 v21, v18, v19
	ds_write_b64 v8, v[20:21] offset:96
	v_accvgpr_read_b32 v16, a16
	v_accvgpr_read_b32 v17, a17
	v_accvgpr_read_b32 v18, a18
	v_accvgpr_read_b32 v19, a19
	s_nop 2
	v_cvt_pk_bf16_f32 v20, v16, v17
	v_cvt_pk_bf16_f32 v21, v18, v19
	ds_write_b64 v8, v[20:21] offset:128
	v_accvgpr_read_b32 v16, a20
	v_accvgpr_read_b32 v17, a21
	v_accvgpr_read_b32 v18, a22
	v_accvgpr_read_b32 v19, a23
	s_nop 2
	v_cvt_pk_bf16_f32 v20, v16, v17
	v_cvt_pk_bf16_f32 v21, v18, v19
	ds_write_b64 v8, v[20:21] offset:160
	v_accvgpr_read_b32 v16, a24
	v_accvgpr_read_b32 v17, a25
	v_accvgpr_read_b32 v18, a26
	v_accvgpr_read_b32 v19, a27
	s_nop 2
	v_cvt_pk_bf16_f32 v20, v16, v17
	v_cvt_pk_bf16_f32 v21, v18, v19
	ds_write_b64 v8, v[20:21] offset:192
	v_accvgpr_read_b32 v16, a28
	v_accvgpr_read_b32 v17, a29
	v_accvgpr_read_b32 v18, a30
	v_accvgpr_read_b32 v19, a31
	s_nop 2
	v_cvt_pk_bf16_f32 v20, v16, v17
	v_cvt_pk_bf16_f32 v21, v18, v19
	ds_write_b64 v8, v[20:21] offset:224
	v_accvgpr_read_b32 v16, a32
	v_accvgpr_read_b32 v17, a33
	v_accvgpr_read_b32 v18, a34
	v_accvgpr_read_b32 v19, a35
	s_nop 2
	v_cvt_pk_bf16_f32 v20, v16, v17
	v_cvt_pk_bf16_f32 v21, v18, v19
	ds_write_b64 v8, v[20:21] offset:4352
	v_accvgpr_read_b32 v16, a36
	v_accvgpr_read_b32 v17, a37
	v_accvgpr_read_b32 v18, a38
	v_accvgpr_read_b32 v19, a39
	s_nop 2
	v_cvt_pk_bf16_f32 v20, v16, v17
	v_cvt_pk_bf16_f32 v21, v18, v19
	ds_write_b64 v8, v[20:21] offset:4384
	v_accvgpr_read_b32 v16, a40
	v_accvgpr_read_b32 v17, a41
	v_accvgpr_read_b32 v18, a42
	v_accvgpr_read_b32 v19, a43
	s_nop 2
	v_cvt_pk_bf16_f32 v20, v16, v17
	v_cvt_pk_bf16_f32 v21, v18, v19
	ds_write_b64 v8, v[20:21] offset:4416
	v_accvgpr_read_b32 v16, a44
	v_accvgpr_read_b32 v17, a45
	v_accvgpr_read_b32 v18, a46
	v_accvgpr_read_b32 v19, a47
	s_nop 2
	v_cvt_pk_bf16_f32 v20, v16, v17
	v_cvt_pk_bf16_f32 v21, v18, v19
	ds_write_b64 v8, v[20:21] offset:4448
	v_accvgpr_read_b32 v16, a48
	v_accvgpr_read_b32 v17, a49
	v_accvgpr_read_b32 v18, a50
	v_accvgpr_read_b32 v19, a51
	s_nop 2
	v_cvt_pk_bf16_f32 v20, v16, v17
	v_cvt_pk_bf16_f32 v21, v18, v19
	ds_write_b64 v8, v[20:21] offset:4480
	v_accvgpr_read_b32 v16, a52
	v_accvgpr_read_b32 v17, a53
	v_accvgpr_read_b32 v18, a54
	v_accvgpr_read_b32 v19, a55
	s_nop 2
	v_cvt_pk_bf16_f32 v20, v16, v17
	v_cvt_pk_bf16_f32 v21, v18, v19
	ds_write_b64 v8, v[20:21] offset:4512
	v_accvgpr_read_b32 v16, a56
	v_accvgpr_read_b32 v17, a57
	v_accvgpr_read_b32 v18, a58
	v_accvgpr_read_b32 v19, a59
	s_nop 2
	v_cvt_pk_bf16_f32 v20, v16, v17
	v_cvt_pk_bf16_f32 v21, v18, v19
	ds_write_b64 v8, v[20:21] offset:4544
	v_accvgpr_read_b32 v16, a60
	v_accvgpr_read_b32 v17, a61
	v_accvgpr_read_b32 v18, a62
	v_accvgpr_read_b32 v19, a63
	s_nop 2
	v_cvt_pk_bf16_f32 v20, v16, v17
	v_cvt_pk_bf16_f32 v21, v18, v19
	ds_write_b64 v8, v[20:21] offset:4576
	v_accvgpr_read_b32 v16, a64
	v_accvgpr_read_b32 v17, a65
	v_accvgpr_read_b32 v18, a66
	v_accvgpr_read_b32 v19, a67
	s_nop 2
	v_cvt_pk_bf16_f32 v20, v16, v17
	v_cvt_pk_bf16_f32 v21, v18, v19
	ds_write_b64 v8, v[20:21] offset:8704
	v_accvgpr_read_b32 v16, a68
	v_accvgpr_read_b32 v17, a69
	v_accvgpr_read_b32 v18, a70
	v_accvgpr_read_b32 v19, a71
	s_nop 2
	v_cvt_pk_bf16_f32 v20, v16, v17
	v_cvt_pk_bf16_f32 v21, v18, v19
	ds_write_b64 v8, v[20:21] offset:8736
	v_accvgpr_read_b32 v16, a72
	v_accvgpr_read_b32 v17, a73
	v_accvgpr_read_b32 v18, a74
	v_accvgpr_read_b32 v19, a75
	s_nop 2
	v_cvt_pk_bf16_f32 v20, v16, v17
	v_cvt_pk_bf16_f32 v21, v18, v19
	ds_write_b64 v8, v[20:21] offset:8768
	v_accvgpr_read_b32 v16, a76
	v_accvgpr_read_b32 v17, a77
	v_accvgpr_read_b32 v18, a78
	v_accvgpr_read_b32 v19, a79
	s_nop 2
	v_cvt_pk_bf16_f32 v20, v16, v17
	v_cvt_pk_bf16_f32 v21, v18, v19
	ds_write_b64 v8, v[20:21] offset:8800
	v_accvgpr_read_b32 v16, a80
	v_accvgpr_read_b32 v17, a81
	v_accvgpr_read_b32 v18, a82
	v_accvgpr_read_b32 v19, a83
	s_nop 2
	v_cvt_pk_bf16_f32 v20, v16, v17
	v_cvt_pk_bf16_f32 v21, v18, v19
	ds_write_b64 v8, v[20:21] offset:8832
	v_accvgpr_read_b32 v16, a84
	v_accvgpr_read_b32 v17, a85
	v_accvgpr_read_b32 v18, a86
	v_accvgpr_read_b32 v19, a87
	s_nop 2
	v_cvt_pk_bf16_f32 v20, v16, v17
	v_cvt_pk_bf16_f32 v21, v18, v19
	ds_write_b64 v8, v[20:21] offset:8864
	v_accvgpr_read_b32 v16, a88
	v_accvgpr_read_b32 v17, a89
	v_accvgpr_read_b32 v18, a90
	v_accvgpr_read_b32 v19, a91
	s_nop 2
	v_cvt_pk_bf16_f32 v20, v16, v17
	v_cvt_pk_bf16_f32 v21, v18, v19
	ds_write_b64 v8, v[20:21] offset:8896
	v_accvgpr_read_b32 v16, a92
	v_accvgpr_read_b32 v17, a93
	v_accvgpr_read_b32 v18, a94
	v_accvgpr_read_b32 v19, a95
	s_nop 2
	v_cvt_pk_bf16_f32 v20, v16, v17
	v_cvt_pk_bf16_f32 v21, v18, v19
	ds_write_b64 v8, v[20:21] offset:8928
	v_accvgpr_read_b32 v16, a96
	v_accvgpr_read_b32 v17, a97
	v_accvgpr_read_b32 v18, a98
	v_accvgpr_read_b32 v19, a99
	s_nop 2
	v_cvt_pk_bf16_f32 v20, v16, v17
	v_cvt_pk_bf16_f32 v21, v18, v19
	ds_write_b64 v8, v[20:21] offset:13056
	v_accvgpr_read_b32 v16, a100
	v_accvgpr_read_b32 v17, a101
	v_accvgpr_read_b32 v18, a102
	v_accvgpr_read_b32 v19, a103
	s_nop 2
	v_cvt_pk_bf16_f32 v20, v16, v17
	v_cvt_pk_bf16_f32 v21, v18, v19
	ds_write_b64 v8, v[20:21] offset:13088
	v_accvgpr_read_b32 v16, a104
	v_accvgpr_read_b32 v17, a105
	v_accvgpr_read_b32 v18, a106
	v_accvgpr_read_b32 v19, a107
	s_nop 2
	v_cvt_pk_bf16_f32 v20, v16, v17
	v_cvt_pk_bf16_f32 v21, v18, v19
	ds_write_b64 v8, v[20:21] offset:13120
	v_accvgpr_read_b32 v16, a108
	v_accvgpr_read_b32 v17, a109
	v_accvgpr_read_b32 v18, a110
	v_accvgpr_read_b32 v19, a111
	s_nop 2
	v_cvt_pk_bf16_f32 v20, v16, v17
	v_cvt_pk_bf16_f32 v21, v18, v19
	ds_write_b64 v8, v[20:21] offset:13152
	v_accvgpr_read_b32 v16, a112
	v_accvgpr_read_b32 v17, a113
	v_accvgpr_read_b32 v18, a114
	v_accvgpr_read_b32 v19, a115
	s_nop 2
	v_cvt_pk_bf16_f32 v20, v16, v17
	v_cvt_pk_bf16_f32 v21, v18, v19
	ds_write_b64 v8, v[20:21] offset:13184
	v_accvgpr_read_b32 v16, a116
	v_accvgpr_read_b32 v17, a117
	v_accvgpr_read_b32 v18, a118
	v_accvgpr_read_b32 v19, a119
	s_nop 2
	v_cvt_pk_bf16_f32 v20, v16, v17
	v_cvt_pk_bf16_f32 v21, v18, v19
	ds_write_b64 v8, v[20:21] offset:13216
	v_accvgpr_read_b32 v16, a120
	v_accvgpr_read_b32 v17, a121
	v_accvgpr_read_b32 v18, a122
	v_accvgpr_read_b32 v19, a123
	s_nop 2
	v_cvt_pk_bf16_f32 v20, v16, v17
	v_cvt_pk_bf16_f32 v21, v18, v19
	ds_write_b64 v8, v[20:21] offset:13248
	v_accvgpr_read_b32 v16, a124
	v_accvgpr_read_b32 v17, a125
	v_accvgpr_read_b32 v18, a126
	v_accvgpr_read_b32 v19, a127
	s_nop 2
	v_cvt_pk_bf16_f32 v20, v16, v17
	v_cvt_pk_bf16_f32 v21, v18, v19
	ds_write_b64 v8, v[20:21] offset:13280
	v_accvgpr_read_b32 v16, a128
	v_accvgpr_read_b32 v17, a129
	v_accvgpr_read_b32 v18, a130
	v_accvgpr_read_b32 v19, a131
	s_nop 2
	v_cvt_pk_bf16_f32 v20, v16, v17
	v_cvt_pk_bf16_f32 v21, v18, v19
	ds_write_b64 v8, v[20:21] offset:17408
	v_accvgpr_read_b32 v16, a132
	v_accvgpr_read_b32 v17, a133
	v_accvgpr_read_b32 v18, a134
	v_accvgpr_read_b32 v19, a135
	s_nop 2
	v_cvt_pk_bf16_f32 v20, v16, v17
	v_cvt_pk_bf16_f32 v21, v18, v19
	ds_write_b64 v8, v[20:21] offset:17440
	v_accvgpr_read_b32 v16, a136
	v_accvgpr_read_b32 v17, a137
	v_accvgpr_read_b32 v18, a138
	v_accvgpr_read_b32 v19, a139
	s_nop 2
	v_cvt_pk_bf16_f32 v20, v16, v17
	v_cvt_pk_bf16_f32 v21, v18, v19
	ds_write_b64 v8, v[20:21] offset:17472
	v_accvgpr_read_b32 v16, a140
	v_accvgpr_read_b32 v17, a141
	v_accvgpr_read_b32 v18, a142
	v_accvgpr_read_b32 v19, a143
	s_nop 2
	v_cvt_pk_bf16_f32 v20, v16, v17
	v_cvt_pk_bf16_f32 v21, v18, v19
	ds_write_b64 v8, v[20:21] offset:17504
	v_accvgpr_read_b32 v16, a144
	v_accvgpr_read_b32 v17, a145
	v_accvgpr_read_b32 v18, a146
	v_accvgpr_read_b32 v19, a147
	s_nop 2
	v_cvt_pk_bf16_f32 v20, v16, v17
	v_cvt_pk_bf16_f32 v21, v18, v19
	ds_write_b64 v8, v[20:21] offset:17536
	v_accvgpr_read_b32 v16, a148
	v_accvgpr_read_b32 v17, a149
	v_accvgpr_read_b32 v18, a150
	v_accvgpr_read_b32 v19, a151
	s_nop 2
	v_cvt_pk_bf16_f32 v20, v16, v17
	v_cvt_pk_bf16_f32 v21, v18, v19
	ds_write_b64 v8, v[20:21] offset:17568
	v_accvgpr_read_b32 v16, a152
	v_accvgpr_read_b32 v17, a153
	v_accvgpr_read_b32 v18, a154
	v_accvgpr_read_b32 v19, a155
	s_nop 2
	v_cvt_pk_bf16_f32 v20, v16, v17
	v_cvt_pk_bf16_f32 v21, v18, v19
	ds_write_b64 v8, v[20:21] offset:17600
	v_accvgpr_read_b32 v16, a156
	v_accvgpr_read_b32 v17, a157
	v_accvgpr_read_b32 v18, a158
	v_accvgpr_read_b32 v19, a159
	s_nop 2
	v_cvt_pk_bf16_f32 v20, v16, v17
	v_cvt_pk_bf16_f32 v21, v18, v19
	ds_write_b64 v8, v[20:21] offset:17632
	v_accvgpr_read_b32 v16, a160
	v_accvgpr_read_b32 v17, a161
	v_accvgpr_read_b32 v18, a162
	v_accvgpr_read_b32 v19, a163
	s_nop 2
	v_cvt_pk_bf16_f32 v20, v16, v17
	v_cvt_pk_bf16_f32 v21, v18, v19
	ds_write_b64 v8, v[20:21] offset:21760
	v_accvgpr_read_b32 v16, a164
	v_accvgpr_read_b32 v17, a165
	v_accvgpr_read_b32 v18, a166
	v_accvgpr_read_b32 v19, a167
	s_nop 2
	v_cvt_pk_bf16_f32 v20, v16, v17
	v_cvt_pk_bf16_f32 v21, v18, v19
	ds_write_b64 v8, v[20:21] offset:21792
	v_accvgpr_read_b32 v16, a168
	v_accvgpr_read_b32 v17, a169
	v_accvgpr_read_b32 v18, a170
	v_accvgpr_read_b32 v19, a171
	s_nop 2
	v_cvt_pk_bf16_f32 v20, v16, v17
	v_cvt_pk_bf16_f32 v21, v18, v19
	ds_write_b64 v8, v[20:21] offset:21824
	v_accvgpr_read_b32 v16, a172
	v_accvgpr_read_b32 v17, a173
	v_accvgpr_read_b32 v18, a174
	v_accvgpr_read_b32 v19, a175
	s_nop 2
	v_cvt_pk_bf16_f32 v20, v16, v17
	v_cvt_pk_bf16_f32 v21, v18, v19
	ds_write_b64 v8, v[20:21] offset:21856
	v_accvgpr_read_b32 v16, a176
	v_accvgpr_read_b32 v17, a177
	v_accvgpr_read_b32 v18, a178
	v_accvgpr_read_b32 v19, a179
	s_nop 2
	v_cvt_pk_bf16_f32 v20, v16, v17
	v_cvt_pk_bf16_f32 v21, v18, v19
	ds_write_b64 v8, v[20:21] offset:21888
	v_accvgpr_read_b32 v16, a180
	v_accvgpr_read_b32 v17, a181
	v_accvgpr_read_b32 v18, a182
	v_accvgpr_read_b32 v19, a183
	s_nop 2
	v_cvt_pk_bf16_f32 v20, v16, v17
	v_cvt_pk_bf16_f32 v21, v18, v19
	ds_write_b64 v8, v[20:21] offset:21920
	v_accvgpr_read_b32 v16, a184
	v_accvgpr_read_b32 v17, a185
	v_accvgpr_read_b32 v18, a186
	v_accvgpr_read_b32 v19, a187
	s_nop 2
	v_cvt_pk_bf16_f32 v20, v16, v17
	v_cvt_pk_bf16_f32 v21, v18, v19
	ds_write_b64 v8, v[20:21] offset:21952
	v_accvgpr_read_b32 v16, a188
	v_accvgpr_read_b32 v17, a189
	v_accvgpr_read_b32 v18, a190
	v_accvgpr_read_b32 v19, a191
	s_nop 2
	v_cvt_pk_bf16_f32 v20, v16, v17
	v_cvt_pk_bf16_f32 v21, v18, v19
	ds_write_b64 v8, v[20:21] offset:21984
	v_accvgpr_read_b32 v16, a192
	v_accvgpr_read_b32 v17, a193
	v_accvgpr_read_b32 v18, a194
	v_accvgpr_read_b32 v19, a195
	s_nop 2
	v_cvt_pk_bf16_f32 v20, v16, v17
	v_cvt_pk_bf16_f32 v21, v18, v19
	ds_write_b64 v8, v[20:21] offset:26112
	v_accvgpr_read_b32 v16, a196
	v_accvgpr_read_b32 v17, a197
	v_accvgpr_read_b32 v18, a198
	v_accvgpr_read_b32 v19, a199
	s_nop 2
	v_cvt_pk_bf16_f32 v20, v16, v17
	v_cvt_pk_bf16_f32 v21, v18, v19
	ds_write_b64 v8, v[20:21] offset:26144
	v_accvgpr_read_b32 v16, a200
	v_accvgpr_read_b32 v17, a201
	v_accvgpr_read_b32 v18, a202
	v_accvgpr_read_b32 v19, a203
	s_nop 2
	v_cvt_pk_bf16_f32 v20, v16, v17
	v_cvt_pk_bf16_f32 v21, v18, v19
	ds_write_b64 v8, v[20:21] offset:26176
	v_accvgpr_read_b32 v16, a204
	v_accvgpr_read_b32 v17, a205
	v_accvgpr_read_b32 v18, a206
	v_accvgpr_read_b32 v19, a207
	s_nop 2
	v_cvt_pk_bf16_f32 v20, v16, v17
	v_cvt_pk_bf16_f32 v21, v18, v19
	ds_write_b64 v8, v[20:21] offset:26208
	v_accvgpr_read_b32 v16, a208
	v_accvgpr_read_b32 v17, a209
	v_accvgpr_read_b32 v18, a210
	v_accvgpr_read_b32 v19, a211
	s_nop 2
	v_cvt_pk_bf16_f32 v20, v16, v17
	v_cvt_pk_bf16_f32 v21, v18, v19
	ds_write_b64 v8, v[20:21] offset:26240
	v_accvgpr_read_b32 v16, a212
	v_accvgpr_read_b32 v17, a213
	v_accvgpr_read_b32 v18, a214
	v_accvgpr_read_b32 v19, a215
	s_nop 2
	v_cvt_pk_bf16_f32 v20, v16, v17
	v_cvt_pk_bf16_f32 v21, v18, v19
	ds_write_b64 v8, v[20:21] offset:26272
	v_accvgpr_read_b32 v16, a216
	v_accvgpr_read_b32 v17, a217
	v_accvgpr_read_b32 v18, a218
	v_accvgpr_read_b32 v19, a219
	s_nop 2
	v_cvt_pk_bf16_f32 v20, v16, v17
	v_cvt_pk_bf16_f32 v21, v18, v19
	ds_write_b64 v8, v[20:21] offset:26304
	v_accvgpr_read_b32 v16, a220
	v_accvgpr_read_b32 v17, a221
	v_accvgpr_read_b32 v18, a222
	v_accvgpr_read_b32 v19, a223
	s_nop 2
	v_cvt_pk_bf16_f32 v20, v16, v17
	v_cvt_pk_bf16_f32 v21, v18, v19
	ds_write_b64 v8, v[20:21] offset:26336
	v_accvgpr_read_b32 v16, a224
	v_accvgpr_read_b32 v17, a225
	v_accvgpr_read_b32 v18, a226
	v_accvgpr_read_b32 v19, a227
	s_nop 2
	v_cvt_pk_bf16_f32 v20, v16, v17
	v_cvt_pk_bf16_f32 v21, v18, v19
	ds_write_b64 v8, v[20:21] offset:30464
	v_accvgpr_read_b32 v16, a228
	v_accvgpr_read_b32 v17, a229
	v_accvgpr_read_b32 v18, a230
	v_accvgpr_read_b32 v19, a231
	s_nop 2
	v_cvt_pk_bf16_f32 v20, v16, v17
	v_cvt_pk_bf16_f32 v21, v18, v19
	ds_write_b64 v8, v[20:21] offset:30496
	v_accvgpr_read_b32 v16, a232
	v_accvgpr_read_b32 v17, a233
	v_accvgpr_read_b32 v18, a234
	v_accvgpr_read_b32 v19, a235
	s_nop 2
	v_cvt_pk_bf16_f32 v20, v16, v17
	v_cvt_pk_bf16_f32 v21, v18, v19
	ds_write_b64 v8, v[20:21] offset:30528
	v_accvgpr_read_b32 v16, a236
	v_accvgpr_read_b32 v17, a237
	v_accvgpr_read_b32 v18, a238
	v_accvgpr_read_b32 v19, a239
	s_nop 2
	v_cvt_pk_bf16_f32 v20, v16, v17
	v_cvt_pk_bf16_f32 v21, v18, v19
	ds_write_b64 v8, v[20:21] offset:30560
	v_accvgpr_read_b32 v16, a240
	v_accvgpr_read_b32 v17, a241
	v_accvgpr_read_b32 v18, a242
	v_accvgpr_read_b32 v19, a243
	s_nop 2
	v_cvt_pk_bf16_f32 v20, v16, v17
	v_cvt_pk_bf16_f32 v21, v18, v19
	ds_write_b64 v8, v[20:21] offset:30592
	v_accvgpr_read_b32 v16, a244
	v_accvgpr_read_b32 v17, a245
	v_accvgpr_read_b32 v18, a246
	v_accvgpr_read_b32 v19, a247
	s_nop 2
	v_cvt_pk_bf16_f32 v20, v16, v17
	v_cvt_pk_bf16_f32 v21, v18, v19
	ds_write_b64 v8, v[20:21] offset:30624
	v_accvgpr_read_b32 v16, a248
	v_accvgpr_read_b32 v17, a249
	v_accvgpr_read_b32 v18, a250
	v_accvgpr_read_b32 v19, a251
	s_nop 2
	v_cvt_pk_bf16_f32 v20, v16, v17
	v_cvt_pk_bf16_f32 v21, v18, v19
	ds_write_b64 v8, v[20:21] offset:30656
	v_accvgpr_read_b32 v16, a252
	v_accvgpr_read_b32 v17, a253
	v_accvgpr_read_b32 v18, a254
	v_accvgpr_read_b32 v19, a255
	s_nop 2
	v_cvt_pk_bf16_f32 v20, v16, v17
	v_cvt_pk_bf16_f32 v21, v18, v19
	ds_write_b64 v8, v[20:21] offset:30688
	s_waitcnt lgkmcnt(0)
	ds_read_b128 v[24:27], v9
	ds_read_b128 v[28:31], v9 offset:1088
	ds_read_b128 v[32:35], v9 offset:2176
	ds_read_b128 v[36:39], v9 offset:3264
	s_waitcnt lgkmcnt(3)
	v_add_u32_e32 v10, 0, v7
	global_store_dwordx4 v10, v[24:27], s[8:9]
	ds_read_b128 v[40:43], v9 offset:4352
	s_waitcnt lgkmcnt(3)
	v_add_u32_e32 v10, 0x8000, v7
	global_store_dwordx4 v10, v[28:31], s[8:9]
	ds_read_b128 v[44:47], v9 offset:5440
	s_waitcnt lgkmcnt(3)
	v_add_u32_e32 v10, 0x10000, v7
	global_store_dwordx4 v10, v[32:35], s[8:9]
	ds_read_b128 v[48:51], v9 offset:6528
	s_waitcnt lgkmcnt(3)
	v_add_u32_e32 v10, 0x18000, v7
	global_store_dwordx4 v10, v[36:39], s[8:9]
	ds_read_b128 v[52:55], v9 offset:7616
	s_waitcnt lgkmcnt(3)
	v_add_u32_e32 v10, 0x20000, v7
	global_store_dwordx4 v10, v[40:43], s[8:9]
	ds_read_b128 v[56:59], v9 offset:8704
	s_waitcnt lgkmcnt(3)
	v_add_u32_e32 v10, 0x28000, v7
	global_store_dwordx4 v10, v[44:47], s[8:9]
	ds_read_b128 v[60:63], v9 offset:9792
	s_waitcnt lgkmcnt(3)
	v_add_u32_e32 v10, 0x30000, v7
	global_store_dwordx4 v10, v[48:51], s[8:9]
	ds_read_b128 v[64:67], v9 offset:10880
	s_waitcnt lgkmcnt(3)
	v_add_u32_e32 v10, 0x38000, v7
	global_store_dwordx4 v10, v[52:55], s[8:9]
	ds_read_b128 v[68:71], v9 offset:11968
	s_waitcnt lgkmcnt(3)
	v_add_u32_e32 v10, 0x40000, v7
	global_store_dwordx4 v10, v[56:59], s[8:9]
	ds_read_b128 v[72:75], v9 offset:13056
	s_waitcnt lgkmcnt(3)
	v_add_u32_e32 v10, 0x48000, v7
	global_store_dwordx4 v10, v[60:63], s[8:9]
	ds_read_b128 v[76:79], v9 offset:14144
	s_waitcnt lgkmcnt(3)
	v_add_u32_e32 v10, 0x50000, v7
	global_store_dwordx4 v10, v[64:67], s[8:9]
	ds_read_b128 v[80:83], v9 offset:15232
	s_waitcnt lgkmcnt(3)
	v_add_u32_e32 v10, 0x58000, v7
	global_store_dwordx4 v10, v[68:71], s[8:9]
	ds_read_b128 v[84:87], v9 offset:16320
	s_waitcnt lgkmcnt(3)
	v_add_u32_e32 v10, 0x60000, v7
	global_store_dwordx4 v10, v[72:75], s[8:9]
	ds_read_b128 v[88:91], v9 offset:17408
	s_waitcnt lgkmcnt(3)
	v_add_u32_e32 v10, 0x68000, v7
	global_store_dwordx4 v10, v[76:79], s[8:9]
	ds_read_b128 v[92:95], v9 offset:18496
	s_waitcnt lgkmcnt(3)
	v_add_u32_e32 v10, 0x70000, v7
	global_store_dwordx4 v10, v[80:83], s[8:9]
	ds_read_b128 v[96:99], v9 offset:19584
	s_waitcnt lgkmcnt(3)
	v_add_u32_e32 v10, 0x78000, v7
	global_store_dwordx4 v10, v[84:87], s[8:9]
	ds_read_b128 v[100:103], v9 offset:20672
	s_waitcnt lgkmcnt(3)
	v_add_u32_e32 v10, 0x80000, v7
	global_store_dwordx4 v10, v[88:91], s[8:9]
	ds_read_b128 v[104:107], v9 offset:21760
	s_waitcnt lgkmcnt(3)
	v_add_u32_e32 v10, 0x88000, v7
	global_store_dwordx4 v10, v[92:95], s[8:9]
	ds_read_b128 v[108:111], v9 offset:22848
	s_waitcnt lgkmcnt(3)
	v_add_u32_e32 v10, 0x90000, v7
	global_store_dwordx4 v10, v[96:99], s[8:9]
	ds_read_b128 v[112:115], v9 offset:23936
	s_waitcnt lgkmcnt(3)
	v_add_u32_e32 v10, 0x98000, v7
	global_store_dwordx4 v10, v[100:103], s[8:9]
	ds_read_b128 v[116:119], v9 offset:25024
	s_waitcnt lgkmcnt(3)
	v_add_u32_e32 v10, 0xa0000, v7
	global_store_dwordx4 v10, v[104:107], s[8:9]
	ds_read_b128 v[120:123], v9 offset:26112
	s_waitcnt lgkmcnt(3)
	v_add_u32_e32 v10, 0xa8000, v7
	global_store_dwordx4 v10, v[108:111], s[8:9]
	ds_read_b128 v[124:127], v9 offset:27200
	s_waitcnt lgkmcnt(3)
	v_add_u32_e32 v10, 0xb0000, v7
	global_store_dwordx4 v10, v[112:115], s[8:9]
	ds_read_b128 v[128:131], v9 offset:28288
	s_waitcnt lgkmcnt(3)
	v_add_u32_e32 v10, 0xb8000, v7
	global_store_dwordx4 v10, v[116:119], s[8:9]
	ds_read_b128 v[132:135], v9 offset:29376
	s_waitcnt lgkmcnt(3)
	v_add_u32_e32 v10, 0xc0000, v7
	global_store_dwordx4 v10, v[120:123], s[8:9]
	ds_read_b128 v[136:139], v9 offset:30464
	s_waitcnt lgkmcnt(3)
	v_add_u32_e32 v10, 0xc8000, v7
	global_store_dwordx4 v10, v[124:127], s[8:9]
	ds_read_b128 v[140:143], v9 offset:31552
	s_waitcnt lgkmcnt(3)
	v_add_u32_e32 v10, 0xd0000, v7
	global_store_dwordx4 v10, v[128:131], s[8:9]
	ds_read_b128 v[144:147], v9 offset:32640
	s_waitcnt lgkmcnt(3)
	v_add_u32_e32 v10, 0xd8000, v7
	global_store_dwordx4 v10, v[132:135], s[8:9]
	ds_read_b128 v[148:151], v9 offset:33728
	s_waitcnt lgkmcnt(3)
	v_add_u32_e32 v10, 0xe0000, v7
	global_store_dwordx4 v10, v[136:139], s[8:9]
	s_waitcnt lgkmcnt(2)
	v_add_u32_e32 v10, 0xe8000, v7
	global_store_dwordx4 v10, v[140:143], s[8:9]
	s_waitcnt lgkmcnt(1)
	v_add_u32_e32 v10, 0xf0000, v7
	global_store_dwordx4 v10, v[144:147], s[8:9]
	s_waitcnt lgkmcnt(0)
	v_add_u32_e32 v10, 0xf8000, v7
	global_store_dwordx4 v10, v[148:151], s[8:9]
	s_waitcnt vmcnt(0)
	s_endpgm

	.section	.rodata,"a",@progbits
	.p2align	6, 0x0
.amdhsa_kernel gfxjit_gemm_mx_e2m3_e2m3_e8m0b32_f32acc_bf16_rcr_256x256x128_w2x2_16x16x128_s3_4096x4096x65536_paired
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
    # SwizzleTensorA/B are plain booleans: they say THAT the operands are swizzled, not
    # WHICH layout. This kernel requires the PAIRED layout -- a 24-byte group stored as 24
    # contiguous bytes at 3072*(t/2) + 48*lane + 24*(t%2) -- and NOT the published
    # HIPBLASLT_ORDER_COL256_R128_EXT ("c0c1") two-plane layout. Given c0c1 data it is
    # silently wrong at full speed. The matching host packer is tileFp6Data in
    # clients/common/include/mx_tile128x4.hpp, which this worktree replaced in place;
    # the two must be changed together.
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
  - .name: gfxjit_gemm_mx_e2m3_e2m3_e8m0b32_f32acc_bf16_rcr_256x256x128_w2x2_16x16x128_s3_4096x4096x65536_paired
    .symbol: gfxjit_gemm_mx_e2m3_e2m3_e8m0b32_f32acc_bf16_rcr_256x256x128_w2x2_16x16x128_s3_4096x4096x65536_paired.kd
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
