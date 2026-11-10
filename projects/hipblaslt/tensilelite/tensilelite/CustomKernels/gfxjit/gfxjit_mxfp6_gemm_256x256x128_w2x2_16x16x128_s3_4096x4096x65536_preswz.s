	.amdgcn_target "amdgcn-amd-amdhsa--gfx950"
	.text
	.globl	gfxjit_mxfp6_gemm_256x256x128_w2x2_16x16x128_s3_4096x4096x65536_preswz
	.p2align	8
	.type	gfxjit_mxfp6_gemm_256x256x128_w2x2_16x16x128_s3_4096x4096x65536_preswz,@function
gfxjit_mxfp6_gemm_256x256x128_w2x2_16x16x128_s3_4096x4096x65536_preswz:
	s_load_dwordx8 s[8:15], s[0:1], 0x0
	s_load_dwordx2 s[16:17], s[0:1], 0x20
	v_lshrrev_b32_e32 v2, 6, v0
	v_and_b32_e32 v1, 63, v0
	s_nop 4
	v_readfirstlane_b32 s18, v2
	s_nop 4
	s_lshr_b32 s19, s18, 1
	s_and_b32 s20, s18, 1
	s_and_b32 s23, s2, 7
	s_lshr_b32 s24, s2, 3
	s_and_b32 s25, s23, 1
	s_lshl_b32 s25, s25, 3
	s_and_b32 s21, s24, 7
	s_add_u32 s21, s21, s25
	s_lshr_b32 s25, s23, 1
	s_lshl_b32 s25, s25, 2
	s_lshr_b32 s22, s24, 3
	s_add_u32 s22, s22, s25
	v_lshlrev_b32_e32 v3, 4, v1
	s_mul_i32 s23, s18, 0x1800
	v_add_u32_e32 v3, s23, v3
	v_add_u32_e32 v4, 0xc00, v3
	v_mul_u32_u24_e32 v2, 24, v1
	s_mul_i32 s24, s19, 0x3000
	s_add_u32 s24, s24, 0
	v_add_u32_e32 v240, s24, v2
	s_mul_i32 s24, s20, 0x3000
	s_add_u32 s24, s24, 0x6000
	v_add_u32_e32 v243, s24, v2
	s_mul_i32 s24, s19, 0x3000
	s_add_u32 s24, s24, 0xc800
	v_add_u32_e32 v241, s24, v2
	s_mul_i32 s24, s20, 0x3000
	s_add_u32 s24, s24, 0x12800
	v_add_u32_e32 v244, s24, v2
	s_mul_i32 s24, s19, 0x3000
	s_add_u32 s24, s24, 0x19000
	v_add_u32_e32 v242, s24, v2
	s_mul_i32 s24, s20, 0x3000
	s_add_u32 s24, s24, 0x1f000
	v_add_u32_e32 v245, s24, v2
	s_mul_i32 s24, s19, 0x200
	s_add_u32 s24, s24, 0xc000
	v_add_u32_e32 v246, s24, v1
	s_mul_i32 s24, s20, 0x200
	s_add_u32 s24, s24, 0xc400
	v_add_u32_e32 v249, s24, v1
	s_mul_i32 s24, s19, 0x200
	s_add_u32 s24, s24, 0x18800
	v_add_u32_e32 v247, s24, v1
	s_mul_i32 s24, s20, 0x200
	s_add_u32 s24, s24, 0x18c00
	v_add_u32_e32 v250, s24, v1
	s_mul_i32 s24, s19, 0x200
	s_add_u32 s24, s24, 0x25000
	v_add_u32_e32 v248, s24, v1
	s_mul_i32 s24, s20, 0x200
	s_add_u32 s24, s24, 0x25400
	v_add_u32_e32 v251, s24, v1
	v_lshlrev_b32_e32 v13, 2, v1
	s_mul_i32 s24, s18, 0x100
	v_add_u32_e32 v13, s24, v13
	s_mul_i32 s32, s18, 0x1800
	s_mul_i32 s33, s18, 0x100
	s_waitcnt lgkmcnt(0)
	s_mov_b32 s36, s10
	s_mov_b32 s37, s11
	.long	0xBEA600FF
	.long	0xFFFFFFFF
	s_mov_b32 s39, 0x20000
	s_mov_b32 s40, s12
	s_mov_b32 s41, s13
	.long	0xBEAA00FF
	.long	0xFFFFFFFF
	s_mov_b32 s43, 0x20000
	s_mov_b32 s44, s14
	s_mov_b32 s45, s15
	.long	0xBEAE00FF
	.long	0xFFFFFFFF
	s_mov_b32 s47, 0x20000
	s_mov_b32 s48, s16
	s_mov_b32 s49, s17
	.long	0xBEB200FF
	.long	0xFFFFFFFF
	s_mov_b32 s51, 0x20000
	s_mul_i32 s23, s21, 0x6000
	s_add_u32 s36, s36, s23
	s_addc_u32 s37, s37, 0
	s_mul_i32 s23, s22, 0x6000
	s_add_u32 s40, s40, s23
	s_addc_u32 s41, s41, 0
	s_mul_i32 s23, s21, 0x400
	s_add_u32 s44, s44, s23
	s_addc_u32 s45, s45, 0
	s_mul_i32 s23, s22, 0x400
	s_add_u32 s48, s48, s23
	s_addc_u32 s49, s49, 0
	s_lshl_b32 s23, s21, 8
	s_mul_i32 s24, s19, 0x80
	s_add_u32 s23, s23, s24
	s_mul_i32 s23, s23, 0x2000
	s_lshl_b32 s24, s22, 8
	s_mul_i32 s25, s20, 0x80
	s_add_u32 s24, s24, s25
	s_lshl_b32 s24, s24, 1
	s_add_u32 s23, s23, s24
	s_add_u32 s26, s8, s23
	s_addc_u32 s27, s9, 0
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
	buffer_load_dwordx4 v4, s[36:39], 0 offen lds
	buffer_load_dwordx4 v4, s[36:39], 0 offen offset:1024 lds
	buffer_load_dwordx4 v4, s[36:39], 0 offen offset:2048 lds
	s_add_u32 m0, s32, 0x6000
	buffer_load_dwordx4 v3, s[40:43], 0 offen lds
	buffer_load_dwordx4 v3, s[40:43], 0 offen offset:1024 lds
	buffer_load_dwordx4 v3, s[40:43], 0 offen offset:2048 lds
	s_add_u32 m0, s32, 0x6c00
	buffer_load_dwordx4 v4, s[40:43], 0 offen lds
	buffer_load_dwordx4 v4, s[40:43], 0 offen offset:1024 lds
	buffer_load_dwordx4 v4, s[40:43], 0 offen offset:2048 lds
	s_add_u32 m0, s33, 0xc000
	buffer_load_dword v13, s[44:47], 0 offen lds
	s_add_u32 m0, s33, 0xc400
	buffer_load_dword v13, s[48:51], 0 offen lds
	s_add_u32 s36, s36, 0x60000
	s_addc_u32 s37, s37, 0
	s_add_u32 s40, s40, 0x60000
	s_addc_u32 s41, s41, 0
	s_add_u32 s44, s44, 0x4000
	s_addc_u32 s45, s45, 0
	s_add_u32 s48, s48, 0x4000
	s_addc_u32 s49, s49, 0
	s_add_u32 m0, s32, 0xc800
	buffer_load_dwordx4 v3, s[36:39], 0 offen lds
	buffer_load_dwordx4 v3, s[36:39], 0 offen offset:1024 lds
	buffer_load_dwordx4 v3, s[36:39], 0 offen offset:2048 lds
	s_add_u32 m0, s32, 0xd400
	buffer_load_dwordx4 v4, s[36:39], 0 offen lds
	buffer_load_dwordx4 v4, s[36:39], 0 offen offset:1024 lds
	buffer_load_dwordx4 v4, s[36:39], 0 offen offset:2048 lds
	s_add_u32 m0, s32, 0x12800
	buffer_load_dwordx4 v3, s[40:43], 0 offen lds
	buffer_load_dwordx4 v3, s[40:43], 0 offen offset:1024 lds
	buffer_load_dwordx4 v3, s[40:43], 0 offen offset:2048 lds
	s_add_u32 m0, s32, 0x13400
	buffer_load_dwordx4 v4, s[40:43], 0 offen lds
	buffer_load_dwordx4 v4, s[40:43], 0 offen offset:1024 lds
	buffer_load_dwordx4 v4, s[40:43], 0 offen offset:2048 lds
	s_add_u32 m0, s33, 0x18800
	buffer_load_dword v13, s[44:47], 0 offen lds
	s_add_u32 m0, s33, 0x18c00
	buffer_load_dword v13, s[48:51], 0 offen lds
	s_add_u32 s36, s36, 0x60000
	s_addc_u32 s37, s37, 0
	s_add_u32 s40, s40, 0x60000
	s_addc_u32 s41, s41, 0
	s_add_u32 s44, s44, 0x4000
	s_addc_u32 s45, s45, 0
	s_add_u32 s48, s48, 0x4000
	s_addc_u32 s49, s49, 0
	s_add_u32 m0, s32, 0x19000
	buffer_load_dwordx4 v3, s[36:39], 0 offen lds
	buffer_load_dwordx4 v3, s[36:39], 0 offen offset:1024 lds
	buffer_load_dwordx4 v3, s[36:39], 0 offen offset:2048 lds
	s_add_u32 m0, s32, 0x19c00
	buffer_load_dwordx4 v4, s[36:39], 0 offen lds
	buffer_load_dwordx4 v4, s[36:39], 0 offen offset:1024 lds
	buffer_load_dwordx4 v4, s[36:39], 0 offen offset:2048 lds
	s_add_u32 m0, s32, 0x1f000
	buffer_load_dwordx4 v3, s[40:43], 0 offen lds
	buffer_load_dwordx4 v3, s[40:43], 0 offen offset:1024 lds
	buffer_load_dwordx4 v3, s[40:43], 0 offen offset:2048 lds
	s_add_u32 m0, s32, 0x1fc00
	buffer_load_dwordx4 v4, s[40:43], 0 offen lds
	buffer_load_dwordx4 v4, s[40:43], 0 offen offset:1024 lds
	buffer_load_dwordx4 v4, s[40:43], 0 offen offset:2048 lds
	s_add_u32 m0, s33, 0x25000
	buffer_load_dword v13, s[44:47], 0 offen lds
	s_add_u32 m0, s33, 0x25400
	buffer_load_dword v13, s[48:51], 0 offen lds
	s_add_u32 s36, s36, 0x60000
	s_addc_u32 s37, s37, 0
	s_add_u32 s40, s40, 0x60000
	s_addc_u32 s41, s41, 0
	s_add_u32 s44, s44, 0x4000
	s_addc_u32 s45, s45, 0
	s_add_u32 s48, s48, 0x4000
	s_addc_u32 s49, s49, 0
	s_waitcnt vmcnt(0)
	s_barrier
	ds_read_u8 v208, v246
	ds_read_u8 v209, v246 offset:64
	ds_read_u8 v210, v246 offset:128
	ds_read_u8 v211, v246 offset:192
	ds_read_u8 v212, v246 offset:256
	ds_read_u8 v213, v246 offset:320
	ds_read_u8 v214, v246 offset:384
	ds_read_u8 v215, v246 offset:448
	ds_read_u8 v224, v249
	ds_read_u8 v225, v249 offset:64
	ds_read_u8 v226, v249 offset:128
	ds_read_u8 v227, v249 offset:192
	ds_read_u8 v228, v249 offset:256
	ds_read_u8 v229, v249 offset:320
	ds_read_u8 v230, v249 offset:384
	ds_read_u8 v231, v249 offset:448
	ds_read_b64 v[16:17], v240
	ds_read_b64 v[18:19], v240 offset:8
	ds_read_b64 v[20:21], v240 offset:16
	ds_read_b64 v[22:23], v240 offset:1536
	ds_read_b64 v[24:25], v240 offset:1544
	ds_read_b64 v[26:27], v240 offset:1552
	ds_read_b64 v[28:29], v240 offset:3072
	ds_read_b64 v[30:31], v240 offset:3080
	ds_read_b64 v[32:33], v240 offset:3088
	ds_read_b64 v[34:35], v240 offset:4608
	ds_read_b64 v[36:37], v240 offset:4616
	ds_read_b64 v[38:39], v240 offset:4624
	ds_read_b64 v[40:41], v240 offset:6144
	ds_read_b64 v[42:43], v240 offset:6152
	ds_read_b64 v[44:45], v240 offset:6160
	ds_read_b64 v[46:47], v240 offset:7680
	ds_read_b64 v[48:49], v240 offset:7688
	ds_read_b64 v[50:51], v240 offset:7696
	ds_read_b64 v[52:53], v240 offset:9216
	ds_read_b64 v[54:55], v240 offset:9224
	ds_read_b64 v[56:57], v240 offset:9232
	ds_read_b64 v[58:59], v240 offset:10752
	ds_read_b64 v[60:61], v240 offset:10760
	ds_read_b64 v[62:63], v240 offset:10768
	ds_read_b64 v[112:113], v243
	ds_read_b64 v[114:115], v243 offset:8
	ds_read_b64 v[116:117], v243 offset:16
	ds_read_b64 v[118:119], v243 offset:1536
	ds_read_b64 v[120:121], v243 offset:1544
	ds_read_b64 v[122:123], v243 offset:1552
	ds_read_b64 v[124:125], v243 offset:3072
	ds_read_b64 v[126:127], v243 offset:3080
	ds_read_b64 v[128:129], v243 offset:3088
	ds_read_b64 v[130:131], v243 offset:4608
	ds_read_b64 v[132:133], v243 offset:4616
	ds_read_b64 v[134:135], v243 offset:4624
	ds_read_b64 v[136:137], v243 offset:6144
	ds_read_b64 v[138:139], v243 offset:6152
	ds_read_b64 v[140:141], v243 offset:6160
	ds_read_b64 v[142:143], v243 offset:7680
	ds_read_b64 v[144:145], v243 offset:7688
	ds_read_b64 v[146:147], v243 offset:7696
	ds_read_b64 v[148:149], v243 offset:9216
	ds_read_b64 v[150:151], v243 offset:9224
	ds_read_b64 v[152:153], v243 offset:9232
	ds_read_b64 v[154:155], v243 offset:10752
	ds_read_b64 v[156:157], v243 offset:10760
	ds_read_b64 v[158:159], v243 offset:10768
	s_waitcnt vmcnt(0) lgkmcnt(0)
	s_mov_b32 s30, 0x56
L_ed8:
	s_waitcnt vmcnt(14)
	s_barrier
	s_waitcnt lgkmcnt(0)
	v_mfma_scale_f32_16x16x128_f8f6f4 a[0:3], v[16:21], v[112:117], a[0:3], v208, v224 op_sel_hi:[0,0,0] cbsz:2 blgp:2
	s_add_u32 m0, s32, 0
	buffer_load_dwordx4 v3, s[36:39], 0 offen lds
	v_mfma_scale_f32_16x16x128_f8f6f4 a[4:7], v[16:21], v[118:123], a[4:7], v208, v225 op_sel_hi:[0,0,0] cbsz:2 blgp:2
	ds_read_u8 v216, v247
	ds_read_b64 v[180:181], v244 offset:4616
	v_mfma_scale_f32_16x16x128_f8f6f4 a[8:11], v[16:21], v[124:129], a[8:11], v208, v226 op_sel_hi:[0,0,0] cbsz:2 blgp:2
	ds_read_u8 v217, v247 offset:64
	ds_read_b64 v[182:183], v244 offset:4624
	v_mfma_scale_f32_16x16x128_f8f6f4 a[12:15], v[16:21], v[130:135], a[12:15], v208, v227 op_sel_hi:[0,0,0] cbsz:2 blgp:2
	ds_read_u8 v218, v247 offset:128
	ds_read_b64 v[184:185], v244 offset:6144
	v_mfma_scale_f32_16x16x128_f8f6f4 a[16:19], v[16:21], v[136:141], a[16:19], v208, v228 op_sel_hi:[0,0,0] cbsz:2 blgp:2
	buffer_load_dwordx4 v3, s[36:39], 0 offen offset:1024 lds
	v_mfma_scale_f32_16x16x128_f8f6f4 a[20:23], v[16:21], v[142:147], a[20:23], v208, v229 op_sel_hi:[0,0,0] cbsz:2 blgp:2
	ds_read_u8 v219, v247 offset:192
	ds_read_b64 v[186:187], v244 offset:6152
	v_mfma_scale_f32_16x16x128_f8f6f4 a[24:27], v[16:21], v[148:153], a[24:27], v208, v230 op_sel_hi:[0,0,0] cbsz:2 blgp:2
	ds_read_u8 v220, v247 offset:256
	ds_read_b64 v[188:189], v244 offset:6160
	v_mfma_scale_f32_16x16x128_f8f6f4 a[28:31], v[16:21], v[154:159], a[28:31], v208, v231 op_sel_hi:[0,0,0] cbsz:2 blgp:2
	ds_read_u8 v221, v247 offset:320
	ds_read_b64 v[190:191], v244 offset:7680
	v_mfma_scale_f32_16x16x128_f8f6f4 a[32:35], v[22:27], v[112:117], a[32:35], v209, v224 op_sel_hi:[0,0,0] cbsz:2 blgp:2
	ds_read_u8 v222, v247 offset:384
	ds_read_b64 v[192:193], v244 offset:7688
	v_mfma_scale_f32_16x16x128_f8f6f4 a[36:39], v[22:27], v[118:123], a[36:39], v209, v225 op_sel_hi:[0,0,0] cbsz:2 blgp:2
	buffer_load_dwordx4 v3, s[36:39], 0 offen offset:2048 lds
	v_mfma_scale_f32_16x16x128_f8f6f4 a[40:43], v[22:27], v[124:129], a[40:43], v209, v226 op_sel_hi:[0,0,0] cbsz:2 blgp:2
	ds_read_u8 v223, v247 offset:448
	ds_read_b64 v[194:195], v244 offset:7696
	v_mfma_scale_f32_16x16x128_f8f6f4 a[44:47], v[22:27], v[130:135], a[44:47], v209, v227 op_sel_hi:[0,0,0] cbsz:2 blgp:2
	ds_read_u8 v232, v250
	ds_read_b64 v[196:197], v244 offset:9216
	v_mfma_scale_f32_16x16x128_f8f6f4 a[48:51], v[22:27], v[136:141], a[48:51], v209, v228 op_sel_hi:[0,0,0] cbsz:2 blgp:2
	ds_read_u8 v233, v250 offset:64
	ds_read_b64 v[198:199], v244 offset:9224
	v_mfma_scale_f32_16x16x128_f8f6f4 a[52:55], v[22:27], v[142:147], a[52:55], v209, v229 op_sel_hi:[0,0,0] cbsz:2 blgp:2
	s_add_u32 m0, s32, 0xc00
	buffer_load_dwordx4 v4, s[36:39], 0 offen lds
	v_mfma_scale_f32_16x16x128_f8f6f4 a[56:59], v[22:27], v[148:153], a[56:59], v209, v230 op_sel_hi:[0,0,0] cbsz:2 blgp:2
	ds_read_u8 v234, v250 offset:128
	ds_read_b64 v[200:201], v244 offset:9232
	v_mfma_scale_f32_16x16x128_f8f6f4 a[60:63], v[22:27], v[154:159], a[60:63], v209, v231 op_sel_hi:[0,0,0] cbsz:2 blgp:2
	ds_read_u8 v235, v250 offset:192
	ds_read_b64 v[202:203], v244 offset:10752
	v_mfma_scale_f32_16x16x128_f8f6f4 a[64:67], v[28:33], v[112:117], a[64:67], v210, v224 op_sel_hi:[0,0,0] cbsz:2 blgp:2
	ds_read_u8 v236, v250 offset:256
	ds_read_b64 v[204:205], v244 offset:10760
	v_mfma_scale_f32_16x16x128_f8f6f4 a[68:71], v[28:33], v[118:123], a[68:71], v210, v225 op_sel_hi:[0,0,0] cbsz:2 blgp:2
	ds_read_u8 v237, v250 offset:320
	ds_read_b64 v[206:207], v244 offset:10768
	v_mfma_scale_f32_16x16x128_f8f6f4 a[72:75], v[28:33], v[124:129], a[72:75], v210, v226 op_sel_hi:[0,0,0] cbsz:2 blgp:2
	buffer_load_dwordx4 v4, s[36:39], 0 offen offset:1024 lds
	v_mfma_scale_f32_16x16x128_f8f6f4 a[76:79], v[28:33], v[130:135], a[76:79], v210, v227 op_sel_hi:[0,0,0] cbsz:2 blgp:2
	ds_read_u8 v238, v250 offset:384
	v_mfma_scale_f32_16x16x128_f8f6f4 a[80:83], v[28:33], v[136:141], a[80:83], v210, v228 op_sel_hi:[0,0,0] cbsz:2 blgp:2
	ds_read_u8 v239, v250 offset:448
	v_mfma_scale_f32_16x16x128_f8f6f4 a[84:87], v[28:33], v[142:147], a[84:87], v210, v229 op_sel_hi:[0,0,0] cbsz:2 blgp:2
	ds_read_b64 v[64:65], v241
	v_mfma_scale_f32_16x16x128_f8f6f4 a[88:91], v[28:33], v[148:153], a[88:91], v210, v230 op_sel_hi:[0,0,0] cbsz:2 blgp:2
	buffer_load_dwordx4 v4, s[36:39], 0 offen offset:2048 lds
	v_mfma_scale_f32_16x16x128_f8f6f4 a[92:95], v[28:33], v[154:159], a[92:95], v210, v231 op_sel_hi:[0,0,0] cbsz:2 blgp:2
	ds_read_b64 v[66:67], v241 offset:8
	v_mfma_scale_f32_16x16x128_f8f6f4 a[96:99], v[34:39], v[112:117], a[96:99], v211, v224 op_sel_hi:[0,0,0] cbsz:2 blgp:2
	ds_read_b64 v[68:69], v241 offset:16
	v_mfma_scale_f32_16x16x128_f8f6f4 a[100:103], v[34:39], v[118:123], a[100:103], v211, v225 op_sel_hi:[0,0,0] cbsz:2 blgp:2
	ds_read_b64 v[70:71], v241 offset:1536
	v_mfma_scale_f32_16x16x128_f8f6f4 a[104:107], v[34:39], v[124:129], a[104:107], v211, v226 op_sel_hi:[0,0,0] cbsz:2 blgp:2
	ds_read_b64 v[72:73], v241 offset:1544
	v_mfma_scale_f32_16x16x128_f8f6f4 a[108:111], v[34:39], v[130:135], a[108:111], v211, v227 op_sel_hi:[0,0,0] cbsz:2 blgp:2
	s_add_u32 m0, s32, 0x6000
	buffer_load_dwordx4 v3, s[40:43], 0 offen lds
	v_mfma_scale_f32_16x16x128_f8f6f4 a[112:115], v[34:39], v[136:141], a[112:115], v211, v228 op_sel_hi:[0,0,0] cbsz:2 blgp:2
	ds_read_b64 v[74:75], v241 offset:1552
	v_mfma_scale_f32_16x16x128_f8f6f4 a[116:119], v[34:39], v[142:147], a[116:119], v211, v229 op_sel_hi:[0,0,0] cbsz:2 blgp:2
	ds_read_b64 v[76:77], v241 offset:3072
	v_mfma_scale_f32_16x16x128_f8f6f4 a[120:123], v[34:39], v[148:153], a[120:123], v211, v230 op_sel_hi:[0,0,0] cbsz:2 blgp:2
	ds_read_b64 v[78:79], v241 offset:3080
	v_mfma_scale_f32_16x16x128_f8f6f4 a[124:127], v[34:39], v[154:159], a[124:127], v211, v231 op_sel_hi:[0,0,0] cbsz:2 blgp:2
	ds_read_b64 v[80:81], v241 offset:3088
	v_mfma_scale_f32_16x16x128_f8f6f4 a[128:131], v[40:45], v[112:117], a[128:131], v212, v224 op_sel_hi:[0,0,0] cbsz:2 blgp:2
	buffer_load_dwordx4 v3, s[40:43], 0 offen offset:1024 lds
	v_mfma_scale_f32_16x16x128_f8f6f4 a[132:135], v[40:45], v[118:123], a[132:135], v212, v225 op_sel_hi:[0,0,0] cbsz:2 blgp:2
	ds_read_b64 v[82:83], v241 offset:4608
	v_mfma_scale_f32_16x16x128_f8f6f4 a[136:139], v[40:45], v[124:129], a[136:139], v212, v226 op_sel_hi:[0,0,0] cbsz:2 blgp:2
	ds_read_b64 v[84:85], v241 offset:4616
	v_mfma_scale_f32_16x16x128_f8f6f4 a[140:143], v[40:45], v[130:135], a[140:143], v212, v227 op_sel_hi:[0,0,0] cbsz:2 blgp:2
	ds_read_b64 v[86:87], v241 offset:4624
	v_mfma_scale_f32_16x16x128_f8f6f4 a[144:147], v[40:45], v[136:141], a[144:147], v212, v228 op_sel_hi:[0,0,0] cbsz:2 blgp:2
	buffer_load_dwordx4 v3, s[40:43], 0 offen offset:2048 lds
	v_mfma_scale_f32_16x16x128_f8f6f4 a[148:151], v[40:45], v[142:147], a[148:151], v212, v229 op_sel_hi:[0,0,0] cbsz:2 blgp:2
	ds_read_b64 v[88:89], v241 offset:6144
	v_mfma_scale_f32_16x16x128_f8f6f4 a[152:155], v[40:45], v[148:153], a[152:155], v212, v230 op_sel_hi:[0,0,0] cbsz:2 blgp:2
	ds_read_b64 v[90:91], v241 offset:6152
	v_mfma_scale_f32_16x16x128_f8f6f4 a[156:159], v[40:45], v[154:159], a[156:159], v212, v231 op_sel_hi:[0,0,0] cbsz:2 blgp:2
	ds_read_b64 v[92:93], v241 offset:6160
	v_mfma_scale_f32_16x16x128_f8f6f4 a[160:163], v[46:51], v[112:117], a[160:163], v213, v224 op_sel_hi:[0,0,0] cbsz:2 blgp:2
	ds_read_b64 v[94:95], v241 offset:7680
	v_mfma_scale_f32_16x16x128_f8f6f4 a[164:167], v[46:51], v[118:123], a[164:167], v213, v225 op_sel_hi:[0,0,0] cbsz:2 blgp:2
	s_add_u32 m0, s32, 0x6c00
	buffer_load_dwordx4 v4, s[40:43], 0 offen lds
	v_mfma_scale_f32_16x16x128_f8f6f4 a[168:171], v[46:51], v[124:129], a[168:171], v213, v226 op_sel_hi:[0,0,0] cbsz:2 blgp:2
	ds_read_b64 v[96:97], v241 offset:7688
	v_mfma_scale_f32_16x16x128_f8f6f4 a[172:175], v[46:51], v[130:135], a[172:175], v213, v227 op_sel_hi:[0,0,0] cbsz:2 blgp:2
	ds_read_b64 v[98:99], v241 offset:7696
	v_mfma_scale_f32_16x16x128_f8f6f4 a[176:179], v[46:51], v[136:141], a[176:179], v213, v228 op_sel_hi:[0,0,0] cbsz:2 blgp:2
	ds_read_b64 v[100:101], v241 offset:9216
	v_mfma_scale_f32_16x16x128_f8f6f4 a[180:183], v[46:51], v[142:147], a[180:183], v213, v229 op_sel_hi:[0,0,0] cbsz:2 blgp:2
	buffer_load_dwordx4 v4, s[40:43], 0 offen offset:1024 lds
	v_mfma_scale_f32_16x16x128_f8f6f4 a[184:187], v[46:51], v[148:153], a[184:187], v213, v230 op_sel_hi:[0,0,0] cbsz:2 blgp:2
	ds_read_b64 v[102:103], v241 offset:9224
	v_mfma_scale_f32_16x16x128_f8f6f4 a[188:191], v[46:51], v[154:159], a[188:191], v213, v231 op_sel_hi:[0,0,0] cbsz:2 blgp:2
	ds_read_b64 v[104:105], v241 offset:9232
	v_mfma_scale_f32_16x16x128_f8f6f4 a[192:195], v[52:57], v[112:117], a[192:195], v214, v224 op_sel_hi:[0,0,0] cbsz:2 blgp:2
	ds_read_b64 v[106:107], v241 offset:10752
	v_mfma_scale_f32_16x16x128_f8f6f4 a[196:199], v[52:57], v[118:123], a[196:199], v214, v225 op_sel_hi:[0,0,0] cbsz:2 blgp:2
	ds_read_b64 v[108:109], v241 offset:10760
	v_mfma_scale_f32_16x16x128_f8f6f4 a[200:203], v[52:57], v[124:129], a[200:203], v214, v226 op_sel_hi:[0,0,0] cbsz:2 blgp:2
	buffer_load_dwordx4 v4, s[40:43], 0 offen offset:2048 lds
	v_mfma_scale_f32_16x16x128_f8f6f4 a[204:207], v[52:57], v[130:135], a[204:207], v214, v227 op_sel_hi:[0,0,0] cbsz:2 blgp:2
	ds_read_b64 v[110:111], v241 offset:10768
	v_mfma_scale_f32_16x16x128_f8f6f4 a[208:211], v[52:57], v[136:141], a[208:211], v214, v228 op_sel_hi:[0,0,0] cbsz:2 blgp:2
	ds_read_b64 v[160:161], v244
	v_mfma_scale_f32_16x16x128_f8f6f4 a[212:215], v[52:57], v[142:147], a[212:215], v214, v229 op_sel_hi:[0,0,0] cbsz:2 blgp:2
	ds_read_b64 v[162:163], v244 offset:8
	v_mfma_scale_f32_16x16x128_f8f6f4 a[216:219], v[52:57], v[148:153], a[216:219], v214, v230 op_sel_hi:[0,0,0] cbsz:2 blgp:2
	s_add_u32 m0, s33, 0xc000
	buffer_load_dword v13, s[44:47], 0 offen lds
	v_mfma_scale_f32_16x16x128_f8f6f4 a[220:223], v[52:57], v[154:159], a[220:223], v214, v231 op_sel_hi:[0,0,0] cbsz:2 blgp:2
	ds_read_b64 v[164:165], v244 offset:16
	v_mfma_scale_f32_16x16x128_f8f6f4 a[224:227], v[58:63], v[112:117], a[224:227], v215, v224 op_sel_hi:[0,0,0] cbsz:2 blgp:2
	ds_read_b64 v[166:167], v244 offset:1536
	v_mfma_scale_f32_16x16x128_f8f6f4 a[228:231], v[58:63], v[118:123], a[228:231], v215, v225 op_sel_hi:[0,0,0] cbsz:2 blgp:2
	ds_read_b64 v[168:169], v244 offset:1544
	v_mfma_scale_f32_16x16x128_f8f6f4 a[232:235], v[58:63], v[124:129], a[232:235], v215, v226 op_sel_hi:[0,0,0] cbsz:2 blgp:2
	ds_read_b64 v[170:171], v244 offset:1552
	v_mfma_scale_f32_16x16x128_f8f6f4 a[236:239], v[58:63], v[130:135], a[236:239], v215, v227 op_sel_hi:[0,0,0] cbsz:2 blgp:2
	s_add_u32 m0, s33, 0xc400
	buffer_load_dword v13, s[48:51], 0 offen lds
	v_mfma_scale_f32_16x16x128_f8f6f4 a[240:243], v[58:63], v[136:141], a[240:243], v215, v228 op_sel_hi:[0,0,0] cbsz:2 blgp:2
	ds_read_b64 v[172:173], v244 offset:3072
	v_mfma_scale_f32_16x16x128_f8f6f4 a[244:247], v[58:63], v[142:147], a[244:247], v215, v229 op_sel_hi:[0,0,0] cbsz:2 blgp:2
	ds_read_b64 v[174:175], v244 offset:3080
	v_mfma_scale_f32_16x16x128_f8f6f4 a[248:251], v[58:63], v[148:153], a[248:251], v215, v230 op_sel_hi:[0,0,0] cbsz:2 blgp:2
	ds_read_b64 v[176:177], v244 offset:3088
	v_mfma_scale_f32_16x16x128_f8f6f4 a[252:255], v[58:63], v[154:159], a[252:255], v215, v231 op_sel_hi:[0,0,0] cbsz:2 blgp:2
	ds_read_b64 v[178:179], v244 offset:4608
	s_add_u32 s36, s36, 0x60000
	s_addc_u32 s37, s37, 0
	s_add_u32 s40, s40, 0x60000
	s_addc_u32 s41, s41, 0
	s_add_u32 s44, s44, 0x4000
	s_addc_u32 s45, s45, 0
	s_add_u32 s48, s48, 0x4000
	s_addc_u32 s49, s49, 0
	s_waitcnt vmcnt(14)
	s_barrier
	s_waitcnt lgkmcnt(0)
	v_mfma_scale_f32_16x16x128_f8f6f4 a[0:3], v[64:69], v[160:165], a[0:3], v216, v232 op_sel_hi:[0,0,0] cbsz:2 blgp:2
	s_add_u32 m0, s32, 0xc800
	buffer_load_dwordx4 v3, s[36:39], 0 offen lds
	v_mfma_scale_f32_16x16x128_f8f6f4 a[4:7], v[64:69], v[166:171], a[4:7], v216, v233 op_sel_hi:[0,0,0] cbsz:2 blgp:2
	ds_read_u8 v208, v248
	ds_read_b64 v[132:133], v245 offset:4616
	v_mfma_scale_f32_16x16x128_f8f6f4 a[8:11], v[64:69], v[172:177], a[8:11], v216, v234 op_sel_hi:[0,0,0] cbsz:2 blgp:2
	ds_read_u8 v209, v248 offset:64
	ds_read_b64 v[134:135], v245 offset:4624
	v_mfma_scale_f32_16x16x128_f8f6f4 a[12:15], v[64:69], v[178:183], a[12:15], v216, v235 op_sel_hi:[0,0,0] cbsz:2 blgp:2
	ds_read_u8 v210, v248 offset:128
	ds_read_b64 v[136:137], v245 offset:6144
	v_mfma_scale_f32_16x16x128_f8f6f4 a[16:19], v[64:69], v[184:189], a[16:19], v216, v236 op_sel_hi:[0,0,0] cbsz:2 blgp:2
	buffer_load_dwordx4 v3, s[36:39], 0 offen offset:1024 lds
	v_mfma_scale_f32_16x16x128_f8f6f4 a[20:23], v[64:69], v[190:195], a[20:23], v216, v237 op_sel_hi:[0,0,0] cbsz:2 blgp:2
	ds_read_u8 v211, v248 offset:192
	ds_read_b64 v[138:139], v245 offset:6152
	v_mfma_scale_f32_16x16x128_f8f6f4 a[24:27], v[64:69], v[196:201], a[24:27], v216, v238 op_sel_hi:[0,0,0] cbsz:2 blgp:2
	ds_read_u8 v212, v248 offset:256
	ds_read_b64 v[140:141], v245 offset:6160
	v_mfma_scale_f32_16x16x128_f8f6f4 a[28:31], v[64:69], v[202:207], a[28:31], v216, v239 op_sel_hi:[0,0,0] cbsz:2 blgp:2
	ds_read_u8 v213, v248 offset:320
	ds_read_b64 v[142:143], v245 offset:7680
	v_mfma_scale_f32_16x16x128_f8f6f4 a[32:35], v[70:75], v[160:165], a[32:35], v217, v232 op_sel_hi:[0,0,0] cbsz:2 blgp:2
	ds_read_u8 v214, v248 offset:384
	ds_read_b64 v[144:145], v245 offset:7688
	v_mfma_scale_f32_16x16x128_f8f6f4 a[36:39], v[70:75], v[166:171], a[36:39], v217, v233 op_sel_hi:[0,0,0] cbsz:2 blgp:2
	buffer_load_dwordx4 v3, s[36:39], 0 offen offset:2048 lds
	v_mfma_scale_f32_16x16x128_f8f6f4 a[40:43], v[70:75], v[172:177], a[40:43], v217, v234 op_sel_hi:[0,0,0] cbsz:2 blgp:2
	ds_read_u8 v215, v248 offset:448
	ds_read_b64 v[146:147], v245 offset:7696
	v_mfma_scale_f32_16x16x128_f8f6f4 a[44:47], v[70:75], v[178:183], a[44:47], v217, v235 op_sel_hi:[0,0,0] cbsz:2 blgp:2
	ds_read_u8 v224, v251
	ds_read_b64 v[148:149], v245 offset:9216
	v_mfma_scale_f32_16x16x128_f8f6f4 a[48:51], v[70:75], v[184:189], a[48:51], v217, v236 op_sel_hi:[0,0,0] cbsz:2 blgp:2
	ds_read_u8 v225, v251 offset:64
	ds_read_b64 v[150:151], v245 offset:9224
	v_mfma_scale_f32_16x16x128_f8f6f4 a[52:55], v[70:75], v[190:195], a[52:55], v217, v237 op_sel_hi:[0,0,0] cbsz:2 blgp:2
	s_add_u32 m0, s32, 0xd400
	buffer_load_dwordx4 v4, s[36:39], 0 offen lds
	v_mfma_scale_f32_16x16x128_f8f6f4 a[56:59], v[70:75], v[196:201], a[56:59], v217, v238 op_sel_hi:[0,0,0] cbsz:2 blgp:2
	ds_read_u8 v226, v251 offset:128
	ds_read_b64 v[152:153], v245 offset:9232
	v_mfma_scale_f32_16x16x128_f8f6f4 a[60:63], v[70:75], v[202:207], a[60:63], v217, v239 op_sel_hi:[0,0,0] cbsz:2 blgp:2
	ds_read_u8 v227, v251 offset:192
	ds_read_b64 v[154:155], v245 offset:10752
	v_mfma_scale_f32_16x16x128_f8f6f4 a[64:67], v[76:81], v[160:165], a[64:67], v218, v232 op_sel_hi:[0,0,0] cbsz:2 blgp:2
	ds_read_u8 v228, v251 offset:256
	ds_read_b64 v[156:157], v245 offset:10760
	v_mfma_scale_f32_16x16x128_f8f6f4 a[68:71], v[76:81], v[166:171], a[68:71], v218, v233 op_sel_hi:[0,0,0] cbsz:2 blgp:2
	ds_read_u8 v229, v251 offset:320
	ds_read_b64 v[158:159], v245 offset:10768
	v_mfma_scale_f32_16x16x128_f8f6f4 a[72:75], v[76:81], v[172:177], a[72:75], v218, v234 op_sel_hi:[0,0,0] cbsz:2 blgp:2
	buffer_load_dwordx4 v4, s[36:39], 0 offen offset:1024 lds
	v_mfma_scale_f32_16x16x128_f8f6f4 a[76:79], v[76:81], v[178:183], a[76:79], v218, v235 op_sel_hi:[0,0,0] cbsz:2 blgp:2
	ds_read_u8 v230, v251 offset:384
	v_mfma_scale_f32_16x16x128_f8f6f4 a[80:83], v[76:81], v[184:189], a[80:83], v218, v236 op_sel_hi:[0,0,0] cbsz:2 blgp:2
	ds_read_u8 v231, v251 offset:448
	v_mfma_scale_f32_16x16x128_f8f6f4 a[84:87], v[76:81], v[190:195], a[84:87], v218, v237 op_sel_hi:[0,0,0] cbsz:2 blgp:2
	ds_read_b64 v[16:17], v242
	v_mfma_scale_f32_16x16x128_f8f6f4 a[88:91], v[76:81], v[196:201], a[88:91], v218, v238 op_sel_hi:[0,0,0] cbsz:2 blgp:2
	buffer_load_dwordx4 v4, s[36:39], 0 offen offset:2048 lds
	v_mfma_scale_f32_16x16x128_f8f6f4 a[92:95], v[76:81], v[202:207], a[92:95], v218, v239 op_sel_hi:[0,0,0] cbsz:2 blgp:2
	ds_read_b64 v[18:19], v242 offset:8
	v_mfma_scale_f32_16x16x128_f8f6f4 a[96:99], v[82:87], v[160:165], a[96:99], v219, v232 op_sel_hi:[0,0,0] cbsz:2 blgp:2
	ds_read_b64 v[20:21], v242 offset:16
	v_mfma_scale_f32_16x16x128_f8f6f4 a[100:103], v[82:87], v[166:171], a[100:103], v219, v233 op_sel_hi:[0,0,0] cbsz:2 blgp:2
	ds_read_b64 v[22:23], v242 offset:1536
	v_mfma_scale_f32_16x16x128_f8f6f4 a[104:107], v[82:87], v[172:177], a[104:107], v219, v234 op_sel_hi:[0,0,0] cbsz:2 blgp:2
	ds_read_b64 v[24:25], v242 offset:1544
	v_mfma_scale_f32_16x16x128_f8f6f4 a[108:111], v[82:87], v[178:183], a[108:111], v219, v235 op_sel_hi:[0,0,0] cbsz:2 blgp:2
	s_add_u32 m0, s32, 0x12800
	buffer_load_dwordx4 v3, s[40:43], 0 offen lds
	v_mfma_scale_f32_16x16x128_f8f6f4 a[112:115], v[82:87], v[184:189], a[112:115], v219, v236 op_sel_hi:[0,0,0] cbsz:2 blgp:2
	ds_read_b64 v[26:27], v242 offset:1552
	v_mfma_scale_f32_16x16x128_f8f6f4 a[116:119], v[82:87], v[190:195], a[116:119], v219, v237 op_sel_hi:[0,0,0] cbsz:2 blgp:2
	ds_read_b64 v[28:29], v242 offset:3072
	v_mfma_scale_f32_16x16x128_f8f6f4 a[120:123], v[82:87], v[196:201], a[120:123], v219, v238 op_sel_hi:[0,0,0] cbsz:2 blgp:2
	ds_read_b64 v[30:31], v242 offset:3080
	v_mfma_scale_f32_16x16x128_f8f6f4 a[124:127], v[82:87], v[202:207], a[124:127], v219, v239 op_sel_hi:[0,0,0] cbsz:2 blgp:2
	ds_read_b64 v[32:33], v242 offset:3088
	v_mfma_scale_f32_16x16x128_f8f6f4 a[128:131], v[88:93], v[160:165], a[128:131], v220, v232 op_sel_hi:[0,0,0] cbsz:2 blgp:2
	buffer_load_dwordx4 v3, s[40:43], 0 offen offset:1024 lds
	v_mfma_scale_f32_16x16x128_f8f6f4 a[132:135], v[88:93], v[166:171], a[132:135], v220, v233 op_sel_hi:[0,0,0] cbsz:2 blgp:2
	ds_read_b64 v[34:35], v242 offset:4608
	v_mfma_scale_f32_16x16x128_f8f6f4 a[136:139], v[88:93], v[172:177], a[136:139], v220, v234 op_sel_hi:[0,0,0] cbsz:2 blgp:2
	ds_read_b64 v[36:37], v242 offset:4616
	v_mfma_scale_f32_16x16x128_f8f6f4 a[140:143], v[88:93], v[178:183], a[140:143], v220, v235 op_sel_hi:[0,0,0] cbsz:2 blgp:2
	ds_read_b64 v[38:39], v242 offset:4624
	v_mfma_scale_f32_16x16x128_f8f6f4 a[144:147], v[88:93], v[184:189], a[144:147], v220, v236 op_sel_hi:[0,0,0] cbsz:2 blgp:2
	buffer_load_dwordx4 v3, s[40:43], 0 offen offset:2048 lds
	v_mfma_scale_f32_16x16x128_f8f6f4 a[148:151], v[88:93], v[190:195], a[148:151], v220, v237 op_sel_hi:[0,0,0] cbsz:2 blgp:2
	ds_read_b64 v[40:41], v242 offset:6144
	v_mfma_scale_f32_16x16x128_f8f6f4 a[152:155], v[88:93], v[196:201], a[152:155], v220, v238 op_sel_hi:[0,0,0] cbsz:2 blgp:2
	ds_read_b64 v[42:43], v242 offset:6152
	v_mfma_scale_f32_16x16x128_f8f6f4 a[156:159], v[88:93], v[202:207], a[156:159], v220, v239 op_sel_hi:[0,0,0] cbsz:2 blgp:2
	ds_read_b64 v[44:45], v242 offset:6160
	v_mfma_scale_f32_16x16x128_f8f6f4 a[160:163], v[94:99], v[160:165], a[160:163], v221, v232 op_sel_hi:[0,0,0] cbsz:2 blgp:2
	ds_read_b64 v[46:47], v242 offset:7680
	v_mfma_scale_f32_16x16x128_f8f6f4 a[164:167], v[94:99], v[166:171], a[164:167], v221, v233 op_sel_hi:[0,0,0] cbsz:2 blgp:2
	s_add_u32 m0, s32, 0x13400
	buffer_load_dwordx4 v4, s[40:43], 0 offen lds
	v_mfma_scale_f32_16x16x128_f8f6f4 a[168:171], v[94:99], v[172:177], a[168:171], v221, v234 op_sel_hi:[0,0,0] cbsz:2 blgp:2
	ds_read_b64 v[48:49], v242 offset:7688
	v_mfma_scale_f32_16x16x128_f8f6f4 a[172:175], v[94:99], v[178:183], a[172:175], v221, v235 op_sel_hi:[0,0,0] cbsz:2 blgp:2
	ds_read_b64 v[50:51], v242 offset:7696
	v_mfma_scale_f32_16x16x128_f8f6f4 a[176:179], v[94:99], v[184:189], a[176:179], v221, v236 op_sel_hi:[0,0,0] cbsz:2 blgp:2
	ds_read_b64 v[52:53], v242 offset:9216
	v_mfma_scale_f32_16x16x128_f8f6f4 a[180:183], v[94:99], v[190:195], a[180:183], v221, v237 op_sel_hi:[0,0,0] cbsz:2 blgp:2
	buffer_load_dwordx4 v4, s[40:43], 0 offen offset:1024 lds
	v_mfma_scale_f32_16x16x128_f8f6f4 a[184:187], v[94:99], v[196:201], a[184:187], v221, v238 op_sel_hi:[0,0,0] cbsz:2 blgp:2
	ds_read_b64 v[54:55], v242 offset:9224
	v_mfma_scale_f32_16x16x128_f8f6f4 a[188:191], v[94:99], v[202:207], a[188:191], v221, v239 op_sel_hi:[0,0,0] cbsz:2 blgp:2
	ds_read_b64 v[56:57], v242 offset:9232
	v_mfma_scale_f32_16x16x128_f8f6f4 a[192:195], v[100:105], v[160:165], a[192:195], v222, v232 op_sel_hi:[0,0,0] cbsz:2 blgp:2
	ds_read_b64 v[58:59], v242 offset:10752
	v_mfma_scale_f32_16x16x128_f8f6f4 a[196:199], v[100:105], v[166:171], a[196:199], v222, v233 op_sel_hi:[0,0,0] cbsz:2 blgp:2
	ds_read_b64 v[60:61], v242 offset:10760
	v_mfma_scale_f32_16x16x128_f8f6f4 a[200:203], v[100:105], v[172:177], a[200:203], v222, v234 op_sel_hi:[0,0,0] cbsz:2 blgp:2
	buffer_load_dwordx4 v4, s[40:43], 0 offen offset:2048 lds
	v_mfma_scale_f32_16x16x128_f8f6f4 a[204:207], v[100:105], v[178:183], a[204:207], v222, v235 op_sel_hi:[0,0,0] cbsz:2 blgp:2
	ds_read_b64 v[62:63], v242 offset:10768
	v_mfma_scale_f32_16x16x128_f8f6f4 a[208:211], v[100:105], v[184:189], a[208:211], v222, v236 op_sel_hi:[0,0,0] cbsz:2 blgp:2
	ds_read_b64 v[112:113], v245
	v_mfma_scale_f32_16x16x128_f8f6f4 a[212:215], v[100:105], v[190:195], a[212:215], v222, v237 op_sel_hi:[0,0,0] cbsz:2 blgp:2
	ds_read_b64 v[114:115], v245 offset:8
	v_mfma_scale_f32_16x16x128_f8f6f4 a[216:219], v[100:105], v[196:201], a[216:219], v222, v238 op_sel_hi:[0,0,0] cbsz:2 blgp:2
	s_add_u32 m0, s33, 0x18800
	buffer_load_dword v13, s[44:47], 0 offen lds
	v_mfma_scale_f32_16x16x128_f8f6f4 a[220:223], v[100:105], v[202:207], a[220:223], v222, v239 op_sel_hi:[0,0,0] cbsz:2 blgp:2
	ds_read_b64 v[116:117], v245 offset:16
	v_mfma_scale_f32_16x16x128_f8f6f4 a[224:227], v[106:111], v[160:165], a[224:227], v223, v232 op_sel_hi:[0,0,0] cbsz:2 blgp:2
	ds_read_b64 v[118:119], v245 offset:1536
	v_mfma_scale_f32_16x16x128_f8f6f4 a[228:231], v[106:111], v[166:171], a[228:231], v223, v233 op_sel_hi:[0,0,0] cbsz:2 blgp:2
	ds_read_b64 v[120:121], v245 offset:1544
	v_mfma_scale_f32_16x16x128_f8f6f4 a[232:235], v[106:111], v[172:177], a[232:235], v223, v234 op_sel_hi:[0,0,0] cbsz:2 blgp:2
	ds_read_b64 v[122:123], v245 offset:1552
	v_mfma_scale_f32_16x16x128_f8f6f4 a[236:239], v[106:111], v[178:183], a[236:239], v223, v235 op_sel_hi:[0,0,0] cbsz:2 blgp:2
	s_add_u32 m0, s33, 0x18c00
	buffer_load_dword v13, s[48:51], 0 offen lds
	v_mfma_scale_f32_16x16x128_f8f6f4 a[240:243], v[106:111], v[184:189], a[240:243], v223, v236 op_sel_hi:[0,0,0] cbsz:2 blgp:2
	ds_read_b64 v[124:125], v245 offset:3072
	v_mfma_scale_f32_16x16x128_f8f6f4 a[244:247], v[106:111], v[190:195], a[244:247], v223, v237 op_sel_hi:[0,0,0] cbsz:2 blgp:2
	ds_read_b64 v[126:127], v245 offset:3080
	v_mfma_scale_f32_16x16x128_f8f6f4 a[248:251], v[106:111], v[196:201], a[248:251], v223, v238 op_sel_hi:[0,0,0] cbsz:2 blgp:2
	ds_read_b64 v[128:129], v245 offset:3088
	v_mfma_scale_f32_16x16x128_f8f6f4 a[252:255], v[106:111], v[202:207], a[252:255], v223, v239 op_sel_hi:[0,0,0] cbsz:2 blgp:2
	ds_read_b64 v[130:131], v245 offset:4608
	s_add_u32 s36, s36, 0x60000
	s_addc_u32 s37, s37, 0
	s_add_u32 s40, s40, 0x60000
	s_addc_u32 s41, s41, 0
	s_add_u32 s44, s44, 0x4000
	s_addc_u32 s45, s45, 0
	s_add_u32 s48, s48, 0x4000
	s_addc_u32 s49, s49, 0
	s_waitcnt vmcnt(14)
	s_barrier
	s_waitcnt lgkmcnt(0)
	v_mfma_scale_f32_16x16x128_f8f6f4 a[0:3], v[16:21], v[112:117], a[0:3], v208, v224 op_sel_hi:[0,0,0] cbsz:2 blgp:2
	s_add_u32 m0, s32, 0x19000
	buffer_load_dwordx4 v3, s[36:39], 0 offen lds
	v_mfma_scale_f32_16x16x128_f8f6f4 a[4:7], v[16:21], v[118:123], a[4:7], v208, v225 op_sel_hi:[0,0,0] cbsz:2 blgp:2
	ds_read_u8 v216, v246
	ds_read_b64 v[180:181], v243 offset:4616
	v_mfma_scale_f32_16x16x128_f8f6f4 a[8:11], v[16:21], v[124:129], a[8:11], v208, v226 op_sel_hi:[0,0,0] cbsz:2 blgp:2
	ds_read_u8 v217, v246 offset:64
	ds_read_b64 v[182:183], v243 offset:4624
	v_mfma_scale_f32_16x16x128_f8f6f4 a[12:15], v[16:21], v[130:135], a[12:15], v208, v227 op_sel_hi:[0,0,0] cbsz:2 blgp:2
	ds_read_u8 v218, v246 offset:128
	ds_read_b64 v[184:185], v243 offset:6144
	v_mfma_scale_f32_16x16x128_f8f6f4 a[16:19], v[16:21], v[136:141], a[16:19], v208, v228 op_sel_hi:[0,0,0] cbsz:2 blgp:2
	buffer_load_dwordx4 v3, s[36:39], 0 offen offset:1024 lds
	v_mfma_scale_f32_16x16x128_f8f6f4 a[20:23], v[16:21], v[142:147], a[20:23], v208, v229 op_sel_hi:[0,0,0] cbsz:2 blgp:2
	ds_read_u8 v219, v246 offset:192
	ds_read_b64 v[186:187], v243 offset:6152
	v_mfma_scale_f32_16x16x128_f8f6f4 a[24:27], v[16:21], v[148:153], a[24:27], v208, v230 op_sel_hi:[0,0,0] cbsz:2 blgp:2
	ds_read_u8 v220, v246 offset:256
	ds_read_b64 v[188:189], v243 offset:6160
	v_mfma_scale_f32_16x16x128_f8f6f4 a[28:31], v[16:21], v[154:159], a[28:31], v208, v231 op_sel_hi:[0,0,0] cbsz:2 blgp:2
	ds_read_u8 v221, v246 offset:320
	ds_read_b64 v[190:191], v243 offset:7680
	v_mfma_scale_f32_16x16x128_f8f6f4 a[32:35], v[22:27], v[112:117], a[32:35], v209, v224 op_sel_hi:[0,0,0] cbsz:2 blgp:2
	ds_read_u8 v222, v246 offset:384
	ds_read_b64 v[192:193], v243 offset:7688
	v_mfma_scale_f32_16x16x128_f8f6f4 a[36:39], v[22:27], v[118:123], a[36:39], v209, v225 op_sel_hi:[0,0,0] cbsz:2 blgp:2
	buffer_load_dwordx4 v3, s[36:39], 0 offen offset:2048 lds
	v_mfma_scale_f32_16x16x128_f8f6f4 a[40:43], v[22:27], v[124:129], a[40:43], v209, v226 op_sel_hi:[0,0,0] cbsz:2 blgp:2
	ds_read_u8 v223, v246 offset:448
	ds_read_b64 v[194:195], v243 offset:7696
	v_mfma_scale_f32_16x16x128_f8f6f4 a[44:47], v[22:27], v[130:135], a[44:47], v209, v227 op_sel_hi:[0,0,0] cbsz:2 blgp:2
	ds_read_u8 v232, v249
	ds_read_b64 v[196:197], v243 offset:9216
	v_mfma_scale_f32_16x16x128_f8f6f4 a[48:51], v[22:27], v[136:141], a[48:51], v209, v228 op_sel_hi:[0,0,0] cbsz:2 blgp:2
	ds_read_u8 v233, v249 offset:64
	ds_read_b64 v[198:199], v243 offset:9224
	v_mfma_scale_f32_16x16x128_f8f6f4 a[52:55], v[22:27], v[142:147], a[52:55], v209, v229 op_sel_hi:[0,0,0] cbsz:2 blgp:2
	s_add_u32 m0, s32, 0x19c00
	buffer_load_dwordx4 v4, s[36:39], 0 offen lds
	v_mfma_scale_f32_16x16x128_f8f6f4 a[56:59], v[22:27], v[148:153], a[56:59], v209, v230 op_sel_hi:[0,0,0] cbsz:2 blgp:2
	ds_read_u8 v234, v249 offset:128
	ds_read_b64 v[200:201], v243 offset:9232
	v_mfma_scale_f32_16x16x128_f8f6f4 a[60:63], v[22:27], v[154:159], a[60:63], v209, v231 op_sel_hi:[0,0,0] cbsz:2 blgp:2
	ds_read_u8 v235, v249 offset:192
	ds_read_b64 v[202:203], v243 offset:10752
	v_mfma_scale_f32_16x16x128_f8f6f4 a[64:67], v[28:33], v[112:117], a[64:67], v210, v224 op_sel_hi:[0,0,0] cbsz:2 blgp:2
	ds_read_u8 v236, v249 offset:256
	ds_read_b64 v[204:205], v243 offset:10760
	v_mfma_scale_f32_16x16x128_f8f6f4 a[68:71], v[28:33], v[118:123], a[68:71], v210, v225 op_sel_hi:[0,0,0] cbsz:2 blgp:2
	ds_read_u8 v237, v249 offset:320
	ds_read_b64 v[206:207], v243 offset:10768
	v_mfma_scale_f32_16x16x128_f8f6f4 a[72:75], v[28:33], v[124:129], a[72:75], v210, v226 op_sel_hi:[0,0,0] cbsz:2 blgp:2
	buffer_load_dwordx4 v4, s[36:39], 0 offen offset:1024 lds
	v_mfma_scale_f32_16x16x128_f8f6f4 a[76:79], v[28:33], v[130:135], a[76:79], v210, v227 op_sel_hi:[0,0,0] cbsz:2 blgp:2
	ds_read_u8 v238, v249 offset:384
	v_mfma_scale_f32_16x16x128_f8f6f4 a[80:83], v[28:33], v[136:141], a[80:83], v210, v228 op_sel_hi:[0,0,0] cbsz:2 blgp:2
	ds_read_u8 v239, v249 offset:448
	v_mfma_scale_f32_16x16x128_f8f6f4 a[84:87], v[28:33], v[142:147], a[84:87], v210, v229 op_sel_hi:[0,0,0] cbsz:2 blgp:2
	ds_read_b64 v[64:65], v240
	v_mfma_scale_f32_16x16x128_f8f6f4 a[88:91], v[28:33], v[148:153], a[88:91], v210, v230 op_sel_hi:[0,0,0] cbsz:2 blgp:2
	buffer_load_dwordx4 v4, s[36:39], 0 offen offset:2048 lds
	v_mfma_scale_f32_16x16x128_f8f6f4 a[92:95], v[28:33], v[154:159], a[92:95], v210, v231 op_sel_hi:[0,0,0] cbsz:2 blgp:2
	ds_read_b64 v[66:67], v240 offset:8
	v_mfma_scale_f32_16x16x128_f8f6f4 a[96:99], v[34:39], v[112:117], a[96:99], v211, v224 op_sel_hi:[0,0,0] cbsz:2 blgp:2
	ds_read_b64 v[68:69], v240 offset:16
	v_mfma_scale_f32_16x16x128_f8f6f4 a[100:103], v[34:39], v[118:123], a[100:103], v211, v225 op_sel_hi:[0,0,0] cbsz:2 blgp:2
	ds_read_b64 v[70:71], v240 offset:1536
	v_mfma_scale_f32_16x16x128_f8f6f4 a[104:107], v[34:39], v[124:129], a[104:107], v211, v226 op_sel_hi:[0,0,0] cbsz:2 blgp:2
	ds_read_b64 v[72:73], v240 offset:1544
	v_mfma_scale_f32_16x16x128_f8f6f4 a[108:111], v[34:39], v[130:135], a[108:111], v211, v227 op_sel_hi:[0,0,0] cbsz:2 blgp:2
	s_add_u32 m0, s32, 0x1f000
	buffer_load_dwordx4 v3, s[40:43], 0 offen lds
	v_mfma_scale_f32_16x16x128_f8f6f4 a[112:115], v[34:39], v[136:141], a[112:115], v211, v228 op_sel_hi:[0,0,0] cbsz:2 blgp:2
	ds_read_b64 v[74:75], v240 offset:1552
	v_mfma_scale_f32_16x16x128_f8f6f4 a[116:119], v[34:39], v[142:147], a[116:119], v211, v229 op_sel_hi:[0,0,0] cbsz:2 blgp:2
	ds_read_b64 v[76:77], v240 offset:3072
	v_mfma_scale_f32_16x16x128_f8f6f4 a[120:123], v[34:39], v[148:153], a[120:123], v211, v230 op_sel_hi:[0,0,0] cbsz:2 blgp:2
	ds_read_b64 v[78:79], v240 offset:3080
	v_mfma_scale_f32_16x16x128_f8f6f4 a[124:127], v[34:39], v[154:159], a[124:127], v211, v231 op_sel_hi:[0,0,0] cbsz:2 blgp:2
	ds_read_b64 v[80:81], v240 offset:3088
	v_mfma_scale_f32_16x16x128_f8f6f4 a[128:131], v[40:45], v[112:117], a[128:131], v212, v224 op_sel_hi:[0,0,0] cbsz:2 blgp:2
	buffer_load_dwordx4 v3, s[40:43], 0 offen offset:1024 lds
	v_mfma_scale_f32_16x16x128_f8f6f4 a[132:135], v[40:45], v[118:123], a[132:135], v212, v225 op_sel_hi:[0,0,0] cbsz:2 blgp:2
	ds_read_b64 v[82:83], v240 offset:4608
	v_mfma_scale_f32_16x16x128_f8f6f4 a[136:139], v[40:45], v[124:129], a[136:139], v212, v226 op_sel_hi:[0,0,0] cbsz:2 blgp:2
	ds_read_b64 v[84:85], v240 offset:4616
	v_mfma_scale_f32_16x16x128_f8f6f4 a[140:143], v[40:45], v[130:135], a[140:143], v212, v227 op_sel_hi:[0,0,0] cbsz:2 blgp:2
	ds_read_b64 v[86:87], v240 offset:4624
	v_mfma_scale_f32_16x16x128_f8f6f4 a[144:147], v[40:45], v[136:141], a[144:147], v212, v228 op_sel_hi:[0,0,0] cbsz:2 blgp:2
	buffer_load_dwordx4 v3, s[40:43], 0 offen offset:2048 lds
	v_mfma_scale_f32_16x16x128_f8f6f4 a[148:151], v[40:45], v[142:147], a[148:151], v212, v229 op_sel_hi:[0,0,0] cbsz:2 blgp:2
	ds_read_b64 v[88:89], v240 offset:6144
	v_mfma_scale_f32_16x16x128_f8f6f4 a[152:155], v[40:45], v[148:153], a[152:155], v212, v230 op_sel_hi:[0,0,0] cbsz:2 blgp:2
	ds_read_b64 v[90:91], v240 offset:6152
	v_mfma_scale_f32_16x16x128_f8f6f4 a[156:159], v[40:45], v[154:159], a[156:159], v212, v231 op_sel_hi:[0,0,0] cbsz:2 blgp:2
	ds_read_b64 v[92:93], v240 offset:6160
	v_mfma_scale_f32_16x16x128_f8f6f4 a[160:163], v[46:51], v[112:117], a[160:163], v213, v224 op_sel_hi:[0,0,0] cbsz:2 blgp:2
	ds_read_b64 v[94:95], v240 offset:7680
	v_mfma_scale_f32_16x16x128_f8f6f4 a[164:167], v[46:51], v[118:123], a[164:167], v213, v225 op_sel_hi:[0,0,0] cbsz:2 blgp:2
	s_add_u32 m0, s32, 0x1fc00
	buffer_load_dwordx4 v4, s[40:43], 0 offen lds
	v_mfma_scale_f32_16x16x128_f8f6f4 a[168:171], v[46:51], v[124:129], a[168:171], v213, v226 op_sel_hi:[0,0,0] cbsz:2 blgp:2
	ds_read_b64 v[96:97], v240 offset:7688
	v_mfma_scale_f32_16x16x128_f8f6f4 a[172:175], v[46:51], v[130:135], a[172:175], v213, v227 op_sel_hi:[0,0,0] cbsz:2 blgp:2
	ds_read_b64 v[98:99], v240 offset:7696
	v_mfma_scale_f32_16x16x128_f8f6f4 a[176:179], v[46:51], v[136:141], a[176:179], v213, v228 op_sel_hi:[0,0,0] cbsz:2 blgp:2
	ds_read_b64 v[100:101], v240 offset:9216
	v_mfma_scale_f32_16x16x128_f8f6f4 a[180:183], v[46:51], v[142:147], a[180:183], v213, v229 op_sel_hi:[0,0,0] cbsz:2 blgp:2
	buffer_load_dwordx4 v4, s[40:43], 0 offen offset:1024 lds
	v_mfma_scale_f32_16x16x128_f8f6f4 a[184:187], v[46:51], v[148:153], a[184:187], v213, v230 op_sel_hi:[0,0,0] cbsz:2 blgp:2
	ds_read_b64 v[102:103], v240 offset:9224
	v_mfma_scale_f32_16x16x128_f8f6f4 a[188:191], v[46:51], v[154:159], a[188:191], v213, v231 op_sel_hi:[0,0,0] cbsz:2 blgp:2
	ds_read_b64 v[104:105], v240 offset:9232
	v_mfma_scale_f32_16x16x128_f8f6f4 a[192:195], v[52:57], v[112:117], a[192:195], v214, v224 op_sel_hi:[0,0,0] cbsz:2 blgp:2
	ds_read_b64 v[106:107], v240 offset:10752
	v_mfma_scale_f32_16x16x128_f8f6f4 a[196:199], v[52:57], v[118:123], a[196:199], v214, v225 op_sel_hi:[0,0,0] cbsz:2 blgp:2
	ds_read_b64 v[108:109], v240 offset:10760
	v_mfma_scale_f32_16x16x128_f8f6f4 a[200:203], v[52:57], v[124:129], a[200:203], v214, v226 op_sel_hi:[0,0,0] cbsz:2 blgp:2
	buffer_load_dwordx4 v4, s[40:43], 0 offen offset:2048 lds
	v_mfma_scale_f32_16x16x128_f8f6f4 a[204:207], v[52:57], v[130:135], a[204:207], v214, v227 op_sel_hi:[0,0,0] cbsz:2 blgp:2
	ds_read_b64 v[110:111], v240 offset:10768
	v_mfma_scale_f32_16x16x128_f8f6f4 a[208:211], v[52:57], v[136:141], a[208:211], v214, v228 op_sel_hi:[0,0,0] cbsz:2 blgp:2
	ds_read_b64 v[160:161], v243
	v_mfma_scale_f32_16x16x128_f8f6f4 a[212:215], v[52:57], v[142:147], a[212:215], v214, v229 op_sel_hi:[0,0,0] cbsz:2 blgp:2
	ds_read_b64 v[162:163], v243 offset:8
	v_mfma_scale_f32_16x16x128_f8f6f4 a[216:219], v[52:57], v[148:153], a[216:219], v214, v230 op_sel_hi:[0,0,0] cbsz:2 blgp:2
	s_add_u32 m0, s33, 0x25000
	buffer_load_dword v13, s[44:47], 0 offen lds
	v_mfma_scale_f32_16x16x128_f8f6f4 a[220:223], v[52:57], v[154:159], a[220:223], v214, v231 op_sel_hi:[0,0,0] cbsz:2 blgp:2
	ds_read_b64 v[164:165], v243 offset:16
	v_mfma_scale_f32_16x16x128_f8f6f4 a[224:227], v[58:63], v[112:117], a[224:227], v215, v224 op_sel_hi:[0,0,0] cbsz:2 blgp:2
	ds_read_b64 v[166:167], v243 offset:1536
	v_mfma_scale_f32_16x16x128_f8f6f4 a[228:231], v[58:63], v[118:123], a[228:231], v215, v225 op_sel_hi:[0,0,0] cbsz:2 blgp:2
	ds_read_b64 v[168:169], v243 offset:1544
	v_mfma_scale_f32_16x16x128_f8f6f4 a[232:235], v[58:63], v[124:129], a[232:235], v215, v226 op_sel_hi:[0,0,0] cbsz:2 blgp:2
	ds_read_b64 v[170:171], v243 offset:1552
	v_mfma_scale_f32_16x16x128_f8f6f4 a[236:239], v[58:63], v[130:135], a[236:239], v215, v227 op_sel_hi:[0,0,0] cbsz:2 blgp:2
	s_add_u32 m0, s33, 0x25400
	buffer_load_dword v13, s[48:51], 0 offen lds
	v_mfma_scale_f32_16x16x128_f8f6f4 a[240:243], v[58:63], v[136:141], a[240:243], v215, v228 op_sel_hi:[0,0,0] cbsz:2 blgp:2
	ds_read_b64 v[172:173], v243 offset:3072
	v_mfma_scale_f32_16x16x128_f8f6f4 a[244:247], v[58:63], v[142:147], a[244:247], v215, v229 op_sel_hi:[0,0,0] cbsz:2 blgp:2
	ds_read_b64 v[174:175], v243 offset:3080
	v_mfma_scale_f32_16x16x128_f8f6f4 a[248:251], v[58:63], v[148:153], a[248:251], v215, v230 op_sel_hi:[0,0,0] cbsz:2 blgp:2
	ds_read_b64 v[176:177], v243 offset:3088
	v_mfma_scale_f32_16x16x128_f8f6f4 a[252:255], v[58:63], v[154:159], a[252:255], v215, v231 op_sel_hi:[0,0,0] cbsz:2 blgp:2
	ds_read_b64 v[178:179], v243 offset:4608
	s_add_u32 s36, s36, 0x60000
	s_addc_u32 s37, s37, 0
	s_add_u32 s40, s40, 0x60000
	s_addc_u32 s41, s41, 0
	s_add_u32 s44, s44, 0x4000
	s_addc_u32 s45, s45, 0
	s_add_u32 s48, s48, 0x4000
	s_addc_u32 s49, s49, 0
	s_waitcnt vmcnt(14)
	s_barrier
	s_waitcnt lgkmcnt(0)
	v_mfma_scale_f32_16x16x128_f8f6f4 a[0:3], v[64:69], v[160:165], a[0:3], v216, v232 op_sel_hi:[0,0,0] cbsz:2 blgp:2
	s_add_u32 m0, s32, 0
	buffer_load_dwordx4 v3, s[36:39], 0 offen lds
	v_mfma_scale_f32_16x16x128_f8f6f4 a[4:7], v[64:69], v[166:171], a[4:7], v216, v233 op_sel_hi:[0,0,0] cbsz:2 blgp:2
	ds_read_u8 v208, v247
	ds_read_b64 v[132:133], v244 offset:4616
	v_mfma_scale_f32_16x16x128_f8f6f4 a[8:11], v[64:69], v[172:177], a[8:11], v216, v234 op_sel_hi:[0,0,0] cbsz:2 blgp:2
	ds_read_u8 v209, v247 offset:64
	ds_read_b64 v[134:135], v244 offset:4624
	v_mfma_scale_f32_16x16x128_f8f6f4 a[12:15], v[64:69], v[178:183], a[12:15], v216, v235 op_sel_hi:[0,0,0] cbsz:2 blgp:2
	ds_read_u8 v210, v247 offset:128
	ds_read_b64 v[136:137], v244 offset:6144
	v_mfma_scale_f32_16x16x128_f8f6f4 a[16:19], v[64:69], v[184:189], a[16:19], v216, v236 op_sel_hi:[0,0,0] cbsz:2 blgp:2
	buffer_load_dwordx4 v3, s[36:39], 0 offen offset:1024 lds
	v_mfma_scale_f32_16x16x128_f8f6f4 a[20:23], v[64:69], v[190:195], a[20:23], v216, v237 op_sel_hi:[0,0,0] cbsz:2 blgp:2
	ds_read_u8 v211, v247 offset:192
	ds_read_b64 v[138:139], v244 offset:6152
	v_mfma_scale_f32_16x16x128_f8f6f4 a[24:27], v[64:69], v[196:201], a[24:27], v216, v238 op_sel_hi:[0,0,0] cbsz:2 blgp:2
	ds_read_u8 v212, v247 offset:256
	ds_read_b64 v[140:141], v244 offset:6160
	v_mfma_scale_f32_16x16x128_f8f6f4 a[28:31], v[64:69], v[202:207], a[28:31], v216, v239 op_sel_hi:[0,0,0] cbsz:2 blgp:2
	ds_read_u8 v213, v247 offset:320
	ds_read_b64 v[142:143], v244 offset:7680
	v_mfma_scale_f32_16x16x128_f8f6f4 a[32:35], v[70:75], v[160:165], a[32:35], v217, v232 op_sel_hi:[0,0,0] cbsz:2 blgp:2
	ds_read_u8 v214, v247 offset:384
	ds_read_b64 v[144:145], v244 offset:7688
	v_mfma_scale_f32_16x16x128_f8f6f4 a[36:39], v[70:75], v[166:171], a[36:39], v217, v233 op_sel_hi:[0,0,0] cbsz:2 blgp:2
	buffer_load_dwordx4 v3, s[36:39], 0 offen offset:2048 lds
	v_mfma_scale_f32_16x16x128_f8f6f4 a[40:43], v[70:75], v[172:177], a[40:43], v217, v234 op_sel_hi:[0,0,0] cbsz:2 blgp:2
	ds_read_u8 v215, v247 offset:448
	ds_read_b64 v[146:147], v244 offset:7696
	v_mfma_scale_f32_16x16x128_f8f6f4 a[44:47], v[70:75], v[178:183], a[44:47], v217, v235 op_sel_hi:[0,0,0] cbsz:2 blgp:2
	ds_read_u8 v224, v250
	ds_read_b64 v[148:149], v244 offset:9216
	v_mfma_scale_f32_16x16x128_f8f6f4 a[48:51], v[70:75], v[184:189], a[48:51], v217, v236 op_sel_hi:[0,0,0] cbsz:2 blgp:2
	ds_read_u8 v225, v250 offset:64
	ds_read_b64 v[150:151], v244 offset:9224
	v_mfma_scale_f32_16x16x128_f8f6f4 a[52:55], v[70:75], v[190:195], a[52:55], v217, v237 op_sel_hi:[0,0,0] cbsz:2 blgp:2
	s_add_u32 m0, s32, 0xc00
	buffer_load_dwordx4 v4, s[36:39], 0 offen lds
	v_mfma_scale_f32_16x16x128_f8f6f4 a[56:59], v[70:75], v[196:201], a[56:59], v217, v238 op_sel_hi:[0,0,0] cbsz:2 blgp:2
	ds_read_u8 v226, v250 offset:128
	ds_read_b64 v[152:153], v244 offset:9232
	v_mfma_scale_f32_16x16x128_f8f6f4 a[60:63], v[70:75], v[202:207], a[60:63], v217, v239 op_sel_hi:[0,0,0] cbsz:2 blgp:2
	ds_read_u8 v227, v250 offset:192
	ds_read_b64 v[154:155], v244 offset:10752
	v_mfma_scale_f32_16x16x128_f8f6f4 a[64:67], v[76:81], v[160:165], a[64:67], v218, v232 op_sel_hi:[0,0,0] cbsz:2 blgp:2
	ds_read_u8 v228, v250 offset:256
	ds_read_b64 v[156:157], v244 offset:10760
	v_mfma_scale_f32_16x16x128_f8f6f4 a[68:71], v[76:81], v[166:171], a[68:71], v218, v233 op_sel_hi:[0,0,0] cbsz:2 blgp:2
	ds_read_u8 v229, v250 offset:320
	ds_read_b64 v[158:159], v244 offset:10768
	v_mfma_scale_f32_16x16x128_f8f6f4 a[72:75], v[76:81], v[172:177], a[72:75], v218, v234 op_sel_hi:[0,0,0] cbsz:2 blgp:2
	buffer_load_dwordx4 v4, s[36:39], 0 offen offset:1024 lds
	v_mfma_scale_f32_16x16x128_f8f6f4 a[76:79], v[76:81], v[178:183], a[76:79], v218, v235 op_sel_hi:[0,0,0] cbsz:2 blgp:2
	ds_read_u8 v230, v250 offset:384
	v_mfma_scale_f32_16x16x128_f8f6f4 a[80:83], v[76:81], v[184:189], a[80:83], v218, v236 op_sel_hi:[0,0,0] cbsz:2 blgp:2
	ds_read_u8 v231, v250 offset:448
	v_mfma_scale_f32_16x16x128_f8f6f4 a[84:87], v[76:81], v[190:195], a[84:87], v218, v237 op_sel_hi:[0,0,0] cbsz:2 blgp:2
	ds_read_b64 v[16:17], v241
	v_mfma_scale_f32_16x16x128_f8f6f4 a[88:91], v[76:81], v[196:201], a[88:91], v218, v238 op_sel_hi:[0,0,0] cbsz:2 blgp:2
	buffer_load_dwordx4 v4, s[36:39], 0 offen offset:2048 lds
	v_mfma_scale_f32_16x16x128_f8f6f4 a[92:95], v[76:81], v[202:207], a[92:95], v218, v239 op_sel_hi:[0,0,0] cbsz:2 blgp:2
	ds_read_b64 v[18:19], v241 offset:8
	v_mfma_scale_f32_16x16x128_f8f6f4 a[96:99], v[82:87], v[160:165], a[96:99], v219, v232 op_sel_hi:[0,0,0] cbsz:2 blgp:2
	ds_read_b64 v[20:21], v241 offset:16
	v_mfma_scale_f32_16x16x128_f8f6f4 a[100:103], v[82:87], v[166:171], a[100:103], v219, v233 op_sel_hi:[0,0,0] cbsz:2 blgp:2
	ds_read_b64 v[22:23], v241 offset:1536
	v_mfma_scale_f32_16x16x128_f8f6f4 a[104:107], v[82:87], v[172:177], a[104:107], v219, v234 op_sel_hi:[0,0,0] cbsz:2 blgp:2
	ds_read_b64 v[24:25], v241 offset:1544
	v_mfma_scale_f32_16x16x128_f8f6f4 a[108:111], v[82:87], v[178:183], a[108:111], v219, v235 op_sel_hi:[0,0,0] cbsz:2 blgp:2
	s_add_u32 m0, s32, 0x6000
	buffer_load_dwordx4 v3, s[40:43], 0 offen lds
	v_mfma_scale_f32_16x16x128_f8f6f4 a[112:115], v[82:87], v[184:189], a[112:115], v219, v236 op_sel_hi:[0,0,0] cbsz:2 blgp:2
	ds_read_b64 v[26:27], v241 offset:1552
	v_mfma_scale_f32_16x16x128_f8f6f4 a[116:119], v[82:87], v[190:195], a[116:119], v219, v237 op_sel_hi:[0,0,0] cbsz:2 blgp:2
	ds_read_b64 v[28:29], v241 offset:3072
	v_mfma_scale_f32_16x16x128_f8f6f4 a[120:123], v[82:87], v[196:201], a[120:123], v219, v238 op_sel_hi:[0,0,0] cbsz:2 blgp:2
	ds_read_b64 v[30:31], v241 offset:3080
	v_mfma_scale_f32_16x16x128_f8f6f4 a[124:127], v[82:87], v[202:207], a[124:127], v219, v239 op_sel_hi:[0,0,0] cbsz:2 blgp:2
	ds_read_b64 v[32:33], v241 offset:3088
	v_mfma_scale_f32_16x16x128_f8f6f4 a[128:131], v[88:93], v[160:165], a[128:131], v220, v232 op_sel_hi:[0,0,0] cbsz:2 blgp:2
	buffer_load_dwordx4 v3, s[40:43], 0 offen offset:1024 lds
	v_mfma_scale_f32_16x16x128_f8f6f4 a[132:135], v[88:93], v[166:171], a[132:135], v220, v233 op_sel_hi:[0,0,0] cbsz:2 blgp:2
	ds_read_b64 v[34:35], v241 offset:4608
	v_mfma_scale_f32_16x16x128_f8f6f4 a[136:139], v[88:93], v[172:177], a[136:139], v220, v234 op_sel_hi:[0,0,0] cbsz:2 blgp:2
	ds_read_b64 v[36:37], v241 offset:4616
	v_mfma_scale_f32_16x16x128_f8f6f4 a[140:143], v[88:93], v[178:183], a[140:143], v220, v235 op_sel_hi:[0,0,0] cbsz:2 blgp:2
	ds_read_b64 v[38:39], v241 offset:4624
	v_mfma_scale_f32_16x16x128_f8f6f4 a[144:147], v[88:93], v[184:189], a[144:147], v220, v236 op_sel_hi:[0,0,0] cbsz:2 blgp:2
	buffer_load_dwordx4 v3, s[40:43], 0 offen offset:2048 lds
	v_mfma_scale_f32_16x16x128_f8f6f4 a[148:151], v[88:93], v[190:195], a[148:151], v220, v237 op_sel_hi:[0,0,0] cbsz:2 blgp:2
	ds_read_b64 v[40:41], v241 offset:6144
	v_mfma_scale_f32_16x16x128_f8f6f4 a[152:155], v[88:93], v[196:201], a[152:155], v220, v238 op_sel_hi:[0,0,0] cbsz:2 blgp:2
	ds_read_b64 v[42:43], v241 offset:6152
	v_mfma_scale_f32_16x16x128_f8f6f4 a[156:159], v[88:93], v[202:207], a[156:159], v220, v239 op_sel_hi:[0,0,0] cbsz:2 blgp:2
	ds_read_b64 v[44:45], v241 offset:6160
	v_mfma_scale_f32_16x16x128_f8f6f4 a[160:163], v[94:99], v[160:165], a[160:163], v221, v232 op_sel_hi:[0,0,0] cbsz:2 blgp:2
	ds_read_b64 v[46:47], v241 offset:7680
	v_mfma_scale_f32_16x16x128_f8f6f4 a[164:167], v[94:99], v[166:171], a[164:167], v221, v233 op_sel_hi:[0,0,0] cbsz:2 blgp:2
	s_add_u32 m0, s32, 0x6c00
	buffer_load_dwordx4 v4, s[40:43], 0 offen lds
	v_mfma_scale_f32_16x16x128_f8f6f4 a[168:171], v[94:99], v[172:177], a[168:171], v221, v234 op_sel_hi:[0,0,0] cbsz:2 blgp:2
	ds_read_b64 v[48:49], v241 offset:7688
	v_mfma_scale_f32_16x16x128_f8f6f4 a[172:175], v[94:99], v[178:183], a[172:175], v221, v235 op_sel_hi:[0,0,0] cbsz:2 blgp:2
	ds_read_b64 v[50:51], v241 offset:7696
	v_mfma_scale_f32_16x16x128_f8f6f4 a[176:179], v[94:99], v[184:189], a[176:179], v221, v236 op_sel_hi:[0,0,0] cbsz:2 blgp:2
	ds_read_b64 v[52:53], v241 offset:9216
	v_mfma_scale_f32_16x16x128_f8f6f4 a[180:183], v[94:99], v[190:195], a[180:183], v221, v237 op_sel_hi:[0,0,0] cbsz:2 blgp:2
	buffer_load_dwordx4 v4, s[40:43], 0 offen offset:1024 lds
	v_mfma_scale_f32_16x16x128_f8f6f4 a[184:187], v[94:99], v[196:201], a[184:187], v221, v238 op_sel_hi:[0,0,0] cbsz:2 blgp:2
	ds_read_b64 v[54:55], v241 offset:9224
	v_mfma_scale_f32_16x16x128_f8f6f4 a[188:191], v[94:99], v[202:207], a[188:191], v221, v239 op_sel_hi:[0,0,0] cbsz:2 blgp:2
	ds_read_b64 v[56:57], v241 offset:9232
	v_mfma_scale_f32_16x16x128_f8f6f4 a[192:195], v[100:105], v[160:165], a[192:195], v222, v232 op_sel_hi:[0,0,0] cbsz:2 blgp:2
	ds_read_b64 v[58:59], v241 offset:10752
	v_mfma_scale_f32_16x16x128_f8f6f4 a[196:199], v[100:105], v[166:171], a[196:199], v222, v233 op_sel_hi:[0,0,0] cbsz:2 blgp:2
	ds_read_b64 v[60:61], v241 offset:10760
	v_mfma_scale_f32_16x16x128_f8f6f4 a[200:203], v[100:105], v[172:177], a[200:203], v222, v234 op_sel_hi:[0,0,0] cbsz:2 blgp:2
	buffer_load_dwordx4 v4, s[40:43], 0 offen offset:2048 lds
	v_mfma_scale_f32_16x16x128_f8f6f4 a[204:207], v[100:105], v[178:183], a[204:207], v222, v235 op_sel_hi:[0,0,0] cbsz:2 blgp:2
	ds_read_b64 v[62:63], v241 offset:10768
	v_mfma_scale_f32_16x16x128_f8f6f4 a[208:211], v[100:105], v[184:189], a[208:211], v222, v236 op_sel_hi:[0,0,0] cbsz:2 blgp:2
	ds_read_b64 v[112:113], v244
	v_mfma_scale_f32_16x16x128_f8f6f4 a[212:215], v[100:105], v[190:195], a[212:215], v222, v237 op_sel_hi:[0,0,0] cbsz:2 blgp:2
	ds_read_b64 v[114:115], v244 offset:8
	v_mfma_scale_f32_16x16x128_f8f6f4 a[216:219], v[100:105], v[196:201], a[216:219], v222, v238 op_sel_hi:[0,0,0] cbsz:2 blgp:2
	s_add_u32 m0, s33, 0xc000
	buffer_load_dword v13, s[44:47], 0 offen lds
	v_mfma_scale_f32_16x16x128_f8f6f4 a[220:223], v[100:105], v[202:207], a[220:223], v222, v239 op_sel_hi:[0,0,0] cbsz:2 blgp:2
	ds_read_b64 v[116:117], v244 offset:16
	v_mfma_scale_f32_16x16x128_f8f6f4 a[224:227], v[106:111], v[160:165], a[224:227], v223, v232 op_sel_hi:[0,0,0] cbsz:2 blgp:2
	ds_read_b64 v[118:119], v244 offset:1536
	v_mfma_scale_f32_16x16x128_f8f6f4 a[228:231], v[106:111], v[166:171], a[228:231], v223, v233 op_sel_hi:[0,0,0] cbsz:2 blgp:2
	ds_read_b64 v[120:121], v244 offset:1544
	v_mfma_scale_f32_16x16x128_f8f6f4 a[232:235], v[106:111], v[172:177], a[232:235], v223, v234 op_sel_hi:[0,0,0] cbsz:2 blgp:2
	ds_read_b64 v[122:123], v244 offset:1552
	v_mfma_scale_f32_16x16x128_f8f6f4 a[236:239], v[106:111], v[178:183], a[236:239], v223, v235 op_sel_hi:[0,0,0] cbsz:2 blgp:2
	s_add_u32 m0, s33, 0xc400
	buffer_load_dword v13, s[48:51], 0 offen lds
	v_mfma_scale_f32_16x16x128_f8f6f4 a[240:243], v[106:111], v[184:189], a[240:243], v223, v236 op_sel_hi:[0,0,0] cbsz:2 blgp:2
	ds_read_b64 v[124:125], v244 offset:3072
	v_mfma_scale_f32_16x16x128_f8f6f4 a[244:247], v[106:111], v[190:195], a[244:247], v223, v237 op_sel_hi:[0,0,0] cbsz:2 blgp:2
	ds_read_b64 v[126:127], v244 offset:3080
	v_mfma_scale_f32_16x16x128_f8f6f4 a[248:251], v[106:111], v[196:201], a[248:251], v223, v238 op_sel_hi:[0,0,0] cbsz:2 blgp:2
	ds_read_b64 v[128:129], v244 offset:3088
	v_mfma_scale_f32_16x16x128_f8f6f4 a[252:255], v[106:111], v[202:207], a[252:255], v223, v239 op_sel_hi:[0,0,0] cbsz:2 blgp:2
	ds_read_b64 v[130:131], v244 offset:4608
	s_add_u32 s36, s36, 0x60000
	s_addc_u32 s37, s37, 0
	s_add_u32 s40, s40, 0x60000
	s_addc_u32 s41, s41, 0
	s_add_u32 s44, s44, 0x4000
	s_addc_u32 s45, s45, 0
	s_add_u32 s48, s48, 0x4000
	s_addc_u32 s49, s49, 0
	s_waitcnt vmcnt(14)
	s_barrier
	s_waitcnt lgkmcnt(0)
	v_mfma_scale_f32_16x16x128_f8f6f4 a[0:3], v[16:21], v[112:117], a[0:3], v208, v224 op_sel_hi:[0,0,0] cbsz:2 blgp:2
	s_add_u32 m0, s32, 0xc800
	buffer_load_dwordx4 v3, s[36:39], 0 offen lds
	v_mfma_scale_f32_16x16x128_f8f6f4 a[4:7], v[16:21], v[118:123], a[4:7], v208, v225 op_sel_hi:[0,0,0] cbsz:2 blgp:2
	ds_read_u8 v216, v248
	ds_read_b64 v[180:181], v245 offset:4616
	v_mfma_scale_f32_16x16x128_f8f6f4 a[8:11], v[16:21], v[124:129], a[8:11], v208, v226 op_sel_hi:[0,0,0] cbsz:2 blgp:2
	ds_read_u8 v217, v248 offset:64
	ds_read_b64 v[182:183], v245 offset:4624
	v_mfma_scale_f32_16x16x128_f8f6f4 a[12:15], v[16:21], v[130:135], a[12:15], v208, v227 op_sel_hi:[0,0,0] cbsz:2 blgp:2
	ds_read_u8 v218, v248 offset:128
	ds_read_b64 v[184:185], v245 offset:6144
	v_mfma_scale_f32_16x16x128_f8f6f4 a[16:19], v[16:21], v[136:141], a[16:19], v208, v228 op_sel_hi:[0,0,0] cbsz:2 blgp:2
	buffer_load_dwordx4 v3, s[36:39], 0 offen offset:1024 lds
	v_mfma_scale_f32_16x16x128_f8f6f4 a[20:23], v[16:21], v[142:147], a[20:23], v208, v229 op_sel_hi:[0,0,0] cbsz:2 blgp:2
	ds_read_u8 v219, v248 offset:192
	ds_read_b64 v[186:187], v245 offset:6152
	v_mfma_scale_f32_16x16x128_f8f6f4 a[24:27], v[16:21], v[148:153], a[24:27], v208, v230 op_sel_hi:[0,0,0] cbsz:2 blgp:2
	ds_read_u8 v220, v248 offset:256
	ds_read_b64 v[188:189], v245 offset:6160
	v_mfma_scale_f32_16x16x128_f8f6f4 a[28:31], v[16:21], v[154:159], a[28:31], v208, v231 op_sel_hi:[0,0,0] cbsz:2 blgp:2
	ds_read_u8 v221, v248 offset:320
	ds_read_b64 v[190:191], v245 offset:7680
	v_mfma_scale_f32_16x16x128_f8f6f4 a[32:35], v[22:27], v[112:117], a[32:35], v209, v224 op_sel_hi:[0,0,0] cbsz:2 blgp:2
	ds_read_u8 v222, v248 offset:384
	ds_read_b64 v[192:193], v245 offset:7688
	v_mfma_scale_f32_16x16x128_f8f6f4 a[36:39], v[22:27], v[118:123], a[36:39], v209, v225 op_sel_hi:[0,0,0] cbsz:2 blgp:2
	buffer_load_dwordx4 v3, s[36:39], 0 offen offset:2048 lds
	v_mfma_scale_f32_16x16x128_f8f6f4 a[40:43], v[22:27], v[124:129], a[40:43], v209, v226 op_sel_hi:[0,0,0] cbsz:2 blgp:2
	ds_read_u8 v223, v248 offset:448
	ds_read_b64 v[194:195], v245 offset:7696
	v_mfma_scale_f32_16x16x128_f8f6f4 a[44:47], v[22:27], v[130:135], a[44:47], v209, v227 op_sel_hi:[0,0,0] cbsz:2 blgp:2
	ds_read_u8 v232, v251
	ds_read_b64 v[196:197], v245 offset:9216
	v_mfma_scale_f32_16x16x128_f8f6f4 a[48:51], v[22:27], v[136:141], a[48:51], v209, v228 op_sel_hi:[0,0,0] cbsz:2 blgp:2
	ds_read_u8 v233, v251 offset:64
	ds_read_b64 v[198:199], v245 offset:9224
	v_mfma_scale_f32_16x16x128_f8f6f4 a[52:55], v[22:27], v[142:147], a[52:55], v209, v229 op_sel_hi:[0,0,0] cbsz:2 blgp:2
	s_add_u32 m0, s32, 0xd400
	buffer_load_dwordx4 v4, s[36:39], 0 offen lds
	v_mfma_scale_f32_16x16x128_f8f6f4 a[56:59], v[22:27], v[148:153], a[56:59], v209, v230 op_sel_hi:[0,0,0] cbsz:2 blgp:2
	ds_read_u8 v234, v251 offset:128
	ds_read_b64 v[200:201], v245 offset:9232
	v_mfma_scale_f32_16x16x128_f8f6f4 a[60:63], v[22:27], v[154:159], a[60:63], v209, v231 op_sel_hi:[0,0,0] cbsz:2 blgp:2
	ds_read_u8 v235, v251 offset:192
	ds_read_b64 v[202:203], v245 offset:10752
	v_mfma_scale_f32_16x16x128_f8f6f4 a[64:67], v[28:33], v[112:117], a[64:67], v210, v224 op_sel_hi:[0,0,0] cbsz:2 blgp:2
	ds_read_u8 v236, v251 offset:256
	ds_read_b64 v[204:205], v245 offset:10760
	v_mfma_scale_f32_16x16x128_f8f6f4 a[68:71], v[28:33], v[118:123], a[68:71], v210, v225 op_sel_hi:[0,0,0] cbsz:2 blgp:2
	ds_read_u8 v237, v251 offset:320
	ds_read_b64 v[206:207], v245 offset:10768
	v_mfma_scale_f32_16x16x128_f8f6f4 a[72:75], v[28:33], v[124:129], a[72:75], v210, v226 op_sel_hi:[0,0,0] cbsz:2 blgp:2
	buffer_load_dwordx4 v4, s[36:39], 0 offen offset:1024 lds
	v_mfma_scale_f32_16x16x128_f8f6f4 a[76:79], v[28:33], v[130:135], a[76:79], v210, v227 op_sel_hi:[0,0,0] cbsz:2 blgp:2
	ds_read_u8 v238, v251 offset:384
	v_mfma_scale_f32_16x16x128_f8f6f4 a[80:83], v[28:33], v[136:141], a[80:83], v210, v228 op_sel_hi:[0,0,0] cbsz:2 blgp:2
	ds_read_u8 v239, v251 offset:448
	v_mfma_scale_f32_16x16x128_f8f6f4 a[84:87], v[28:33], v[142:147], a[84:87], v210, v229 op_sel_hi:[0,0,0] cbsz:2 blgp:2
	ds_read_b64 v[64:65], v242
	v_mfma_scale_f32_16x16x128_f8f6f4 a[88:91], v[28:33], v[148:153], a[88:91], v210, v230 op_sel_hi:[0,0,0] cbsz:2 blgp:2
	buffer_load_dwordx4 v4, s[36:39], 0 offen offset:2048 lds
	v_mfma_scale_f32_16x16x128_f8f6f4 a[92:95], v[28:33], v[154:159], a[92:95], v210, v231 op_sel_hi:[0,0,0] cbsz:2 blgp:2
	ds_read_b64 v[66:67], v242 offset:8
	v_mfma_scale_f32_16x16x128_f8f6f4 a[96:99], v[34:39], v[112:117], a[96:99], v211, v224 op_sel_hi:[0,0,0] cbsz:2 blgp:2
	ds_read_b64 v[68:69], v242 offset:16
	v_mfma_scale_f32_16x16x128_f8f6f4 a[100:103], v[34:39], v[118:123], a[100:103], v211, v225 op_sel_hi:[0,0,0] cbsz:2 blgp:2
	ds_read_b64 v[70:71], v242 offset:1536
	v_mfma_scale_f32_16x16x128_f8f6f4 a[104:107], v[34:39], v[124:129], a[104:107], v211, v226 op_sel_hi:[0,0,0] cbsz:2 blgp:2
	ds_read_b64 v[72:73], v242 offset:1544
	v_mfma_scale_f32_16x16x128_f8f6f4 a[108:111], v[34:39], v[130:135], a[108:111], v211, v227 op_sel_hi:[0,0,0] cbsz:2 blgp:2
	s_add_u32 m0, s32, 0x12800
	buffer_load_dwordx4 v3, s[40:43], 0 offen lds
	v_mfma_scale_f32_16x16x128_f8f6f4 a[112:115], v[34:39], v[136:141], a[112:115], v211, v228 op_sel_hi:[0,0,0] cbsz:2 blgp:2
	ds_read_b64 v[74:75], v242 offset:1552
	v_mfma_scale_f32_16x16x128_f8f6f4 a[116:119], v[34:39], v[142:147], a[116:119], v211, v229 op_sel_hi:[0,0,0] cbsz:2 blgp:2
	ds_read_b64 v[76:77], v242 offset:3072
	v_mfma_scale_f32_16x16x128_f8f6f4 a[120:123], v[34:39], v[148:153], a[120:123], v211, v230 op_sel_hi:[0,0,0] cbsz:2 blgp:2
	ds_read_b64 v[78:79], v242 offset:3080
	v_mfma_scale_f32_16x16x128_f8f6f4 a[124:127], v[34:39], v[154:159], a[124:127], v211, v231 op_sel_hi:[0,0,0] cbsz:2 blgp:2
	ds_read_b64 v[80:81], v242 offset:3088
	v_mfma_scale_f32_16x16x128_f8f6f4 a[128:131], v[40:45], v[112:117], a[128:131], v212, v224 op_sel_hi:[0,0,0] cbsz:2 blgp:2
	buffer_load_dwordx4 v3, s[40:43], 0 offen offset:1024 lds
	v_mfma_scale_f32_16x16x128_f8f6f4 a[132:135], v[40:45], v[118:123], a[132:135], v212, v225 op_sel_hi:[0,0,0] cbsz:2 blgp:2
	ds_read_b64 v[82:83], v242 offset:4608
	v_mfma_scale_f32_16x16x128_f8f6f4 a[136:139], v[40:45], v[124:129], a[136:139], v212, v226 op_sel_hi:[0,0,0] cbsz:2 blgp:2
	ds_read_b64 v[84:85], v242 offset:4616
	v_mfma_scale_f32_16x16x128_f8f6f4 a[140:143], v[40:45], v[130:135], a[140:143], v212, v227 op_sel_hi:[0,0,0] cbsz:2 blgp:2
	ds_read_b64 v[86:87], v242 offset:4624
	v_mfma_scale_f32_16x16x128_f8f6f4 a[144:147], v[40:45], v[136:141], a[144:147], v212, v228 op_sel_hi:[0,0,0] cbsz:2 blgp:2
	buffer_load_dwordx4 v3, s[40:43], 0 offen offset:2048 lds
	v_mfma_scale_f32_16x16x128_f8f6f4 a[148:151], v[40:45], v[142:147], a[148:151], v212, v229 op_sel_hi:[0,0,0] cbsz:2 blgp:2
	ds_read_b64 v[88:89], v242 offset:6144
	v_mfma_scale_f32_16x16x128_f8f6f4 a[152:155], v[40:45], v[148:153], a[152:155], v212, v230 op_sel_hi:[0,0,0] cbsz:2 blgp:2
	ds_read_b64 v[90:91], v242 offset:6152
	v_mfma_scale_f32_16x16x128_f8f6f4 a[156:159], v[40:45], v[154:159], a[156:159], v212, v231 op_sel_hi:[0,0,0] cbsz:2 blgp:2
	ds_read_b64 v[92:93], v242 offset:6160
	v_mfma_scale_f32_16x16x128_f8f6f4 a[160:163], v[46:51], v[112:117], a[160:163], v213, v224 op_sel_hi:[0,0,0] cbsz:2 blgp:2
	ds_read_b64 v[94:95], v242 offset:7680
	v_mfma_scale_f32_16x16x128_f8f6f4 a[164:167], v[46:51], v[118:123], a[164:167], v213, v225 op_sel_hi:[0,0,0] cbsz:2 blgp:2
	s_add_u32 m0, s32, 0x13400
	buffer_load_dwordx4 v4, s[40:43], 0 offen lds
	v_mfma_scale_f32_16x16x128_f8f6f4 a[168:171], v[46:51], v[124:129], a[168:171], v213, v226 op_sel_hi:[0,0,0] cbsz:2 blgp:2
	ds_read_b64 v[96:97], v242 offset:7688
	v_mfma_scale_f32_16x16x128_f8f6f4 a[172:175], v[46:51], v[130:135], a[172:175], v213, v227 op_sel_hi:[0,0,0] cbsz:2 blgp:2
	ds_read_b64 v[98:99], v242 offset:7696
	v_mfma_scale_f32_16x16x128_f8f6f4 a[176:179], v[46:51], v[136:141], a[176:179], v213, v228 op_sel_hi:[0,0,0] cbsz:2 blgp:2
	ds_read_b64 v[100:101], v242 offset:9216
	v_mfma_scale_f32_16x16x128_f8f6f4 a[180:183], v[46:51], v[142:147], a[180:183], v213, v229 op_sel_hi:[0,0,0] cbsz:2 blgp:2
	buffer_load_dwordx4 v4, s[40:43], 0 offen offset:1024 lds
	v_mfma_scale_f32_16x16x128_f8f6f4 a[184:187], v[46:51], v[148:153], a[184:187], v213, v230 op_sel_hi:[0,0,0] cbsz:2 blgp:2
	ds_read_b64 v[102:103], v242 offset:9224
	v_mfma_scale_f32_16x16x128_f8f6f4 a[188:191], v[46:51], v[154:159], a[188:191], v213, v231 op_sel_hi:[0,0,0] cbsz:2 blgp:2
	ds_read_b64 v[104:105], v242 offset:9232
	v_mfma_scale_f32_16x16x128_f8f6f4 a[192:195], v[52:57], v[112:117], a[192:195], v214, v224 op_sel_hi:[0,0,0] cbsz:2 blgp:2
	ds_read_b64 v[106:107], v242 offset:10752
	v_mfma_scale_f32_16x16x128_f8f6f4 a[196:199], v[52:57], v[118:123], a[196:199], v214, v225 op_sel_hi:[0,0,0] cbsz:2 blgp:2
	ds_read_b64 v[108:109], v242 offset:10760
	v_mfma_scale_f32_16x16x128_f8f6f4 a[200:203], v[52:57], v[124:129], a[200:203], v214, v226 op_sel_hi:[0,0,0] cbsz:2 blgp:2
	buffer_load_dwordx4 v4, s[40:43], 0 offen offset:2048 lds
	v_mfma_scale_f32_16x16x128_f8f6f4 a[204:207], v[52:57], v[130:135], a[204:207], v214, v227 op_sel_hi:[0,0,0] cbsz:2 blgp:2
	ds_read_b64 v[110:111], v242 offset:10768
	v_mfma_scale_f32_16x16x128_f8f6f4 a[208:211], v[52:57], v[136:141], a[208:211], v214, v228 op_sel_hi:[0,0,0] cbsz:2 blgp:2
	ds_read_b64 v[160:161], v245
	v_mfma_scale_f32_16x16x128_f8f6f4 a[212:215], v[52:57], v[142:147], a[212:215], v214, v229 op_sel_hi:[0,0,0] cbsz:2 blgp:2
	ds_read_b64 v[162:163], v245 offset:8
	v_mfma_scale_f32_16x16x128_f8f6f4 a[216:219], v[52:57], v[148:153], a[216:219], v214, v230 op_sel_hi:[0,0,0] cbsz:2 blgp:2
	s_add_u32 m0, s33, 0x18800
	buffer_load_dword v13, s[44:47], 0 offen lds
	v_mfma_scale_f32_16x16x128_f8f6f4 a[220:223], v[52:57], v[154:159], a[220:223], v214, v231 op_sel_hi:[0,0,0] cbsz:2 blgp:2
	ds_read_b64 v[164:165], v245 offset:16
	v_mfma_scale_f32_16x16x128_f8f6f4 a[224:227], v[58:63], v[112:117], a[224:227], v215, v224 op_sel_hi:[0,0,0] cbsz:2 blgp:2
	ds_read_b64 v[166:167], v245 offset:1536
	v_mfma_scale_f32_16x16x128_f8f6f4 a[228:231], v[58:63], v[118:123], a[228:231], v215, v225 op_sel_hi:[0,0,0] cbsz:2 blgp:2
	ds_read_b64 v[168:169], v245 offset:1544
	v_mfma_scale_f32_16x16x128_f8f6f4 a[232:235], v[58:63], v[124:129], a[232:235], v215, v226 op_sel_hi:[0,0,0] cbsz:2 blgp:2
	ds_read_b64 v[170:171], v245 offset:1552
	v_mfma_scale_f32_16x16x128_f8f6f4 a[236:239], v[58:63], v[130:135], a[236:239], v215, v227 op_sel_hi:[0,0,0] cbsz:2 blgp:2
	s_add_u32 m0, s33, 0x18c00
	buffer_load_dword v13, s[48:51], 0 offen lds
	v_mfma_scale_f32_16x16x128_f8f6f4 a[240:243], v[58:63], v[136:141], a[240:243], v215, v228 op_sel_hi:[0,0,0] cbsz:2 blgp:2
	ds_read_b64 v[172:173], v245 offset:3072
	v_mfma_scale_f32_16x16x128_f8f6f4 a[244:247], v[58:63], v[142:147], a[244:247], v215, v229 op_sel_hi:[0,0,0] cbsz:2 blgp:2
	ds_read_b64 v[174:175], v245 offset:3080
	v_mfma_scale_f32_16x16x128_f8f6f4 a[248:251], v[58:63], v[148:153], a[248:251], v215, v230 op_sel_hi:[0,0,0] cbsz:2 blgp:2
	ds_read_b64 v[176:177], v245 offset:3088
	v_mfma_scale_f32_16x16x128_f8f6f4 a[252:255], v[58:63], v[154:159], a[252:255], v215, v231 op_sel_hi:[0,0,0] cbsz:2 blgp:2
	ds_read_b64 v[178:179], v245 offset:4608
	s_add_u32 s36, s36, 0x60000
	s_addc_u32 s37, s37, 0
	s_add_u32 s40, s40, 0x60000
	s_addc_u32 s41, s41, 0
	s_add_u32 s44, s44, 0x4000
	s_addc_u32 s45, s45, 0
	s_add_u32 s48, s48, 0x4000
	s_addc_u32 s49, s49, 0
	s_waitcnt vmcnt(14)
	s_barrier
	s_waitcnt lgkmcnt(0)
	v_mfma_scale_f32_16x16x128_f8f6f4 a[0:3], v[64:69], v[160:165], a[0:3], v216, v232 op_sel_hi:[0,0,0] cbsz:2 blgp:2
	s_add_u32 m0, s32, 0x19000
	buffer_load_dwordx4 v3, s[36:39], 0 offen lds
	v_mfma_scale_f32_16x16x128_f8f6f4 a[4:7], v[64:69], v[166:171], a[4:7], v216, v233 op_sel_hi:[0,0,0] cbsz:2 blgp:2
	ds_read_u8 v208, v246
	ds_read_b64 v[132:133], v243 offset:4616
	v_mfma_scale_f32_16x16x128_f8f6f4 a[8:11], v[64:69], v[172:177], a[8:11], v216, v234 op_sel_hi:[0,0,0] cbsz:2 blgp:2
	ds_read_u8 v209, v246 offset:64
	ds_read_b64 v[134:135], v243 offset:4624
	v_mfma_scale_f32_16x16x128_f8f6f4 a[12:15], v[64:69], v[178:183], a[12:15], v216, v235 op_sel_hi:[0,0,0] cbsz:2 blgp:2
	ds_read_u8 v210, v246 offset:128
	ds_read_b64 v[136:137], v243 offset:6144
	v_mfma_scale_f32_16x16x128_f8f6f4 a[16:19], v[64:69], v[184:189], a[16:19], v216, v236 op_sel_hi:[0,0,0] cbsz:2 blgp:2
	buffer_load_dwordx4 v3, s[36:39], 0 offen offset:1024 lds
	v_mfma_scale_f32_16x16x128_f8f6f4 a[20:23], v[64:69], v[190:195], a[20:23], v216, v237 op_sel_hi:[0,0,0] cbsz:2 blgp:2
	ds_read_u8 v211, v246 offset:192
	ds_read_b64 v[138:139], v243 offset:6152
	v_mfma_scale_f32_16x16x128_f8f6f4 a[24:27], v[64:69], v[196:201], a[24:27], v216, v238 op_sel_hi:[0,0,0] cbsz:2 blgp:2
	ds_read_u8 v212, v246 offset:256
	ds_read_b64 v[140:141], v243 offset:6160
	v_mfma_scale_f32_16x16x128_f8f6f4 a[28:31], v[64:69], v[202:207], a[28:31], v216, v239 op_sel_hi:[0,0,0] cbsz:2 blgp:2
	ds_read_u8 v213, v246 offset:320
	ds_read_b64 v[142:143], v243 offset:7680
	v_mfma_scale_f32_16x16x128_f8f6f4 a[32:35], v[70:75], v[160:165], a[32:35], v217, v232 op_sel_hi:[0,0,0] cbsz:2 blgp:2
	ds_read_u8 v214, v246 offset:384
	ds_read_b64 v[144:145], v243 offset:7688
	v_mfma_scale_f32_16x16x128_f8f6f4 a[36:39], v[70:75], v[166:171], a[36:39], v217, v233 op_sel_hi:[0,0,0] cbsz:2 blgp:2
	buffer_load_dwordx4 v3, s[36:39], 0 offen offset:2048 lds
	v_mfma_scale_f32_16x16x128_f8f6f4 a[40:43], v[70:75], v[172:177], a[40:43], v217, v234 op_sel_hi:[0,0,0] cbsz:2 blgp:2
	ds_read_u8 v215, v246 offset:448
	ds_read_b64 v[146:147], v243 offset:7696
	v_mfma_scale_f32_16x16x128_f8f6f4 a[44:47], v[70:75], v[178:183], a[44:47], v217, v235 op_sel_hi:[0,0,0] cbsz:2 blgp:2
	ds_read_u8 v224, v249
	ds_read_b64 v[148:149], v243 offset:9216
	v_mfma_scale_f32_16x16x128_f8f6f4 a[48:51], v[70:75], v[184:189], a[48:51], v217, v236 op_sel_hi:[0,0,0] cbsz:2 blgp:2
	ds_read_u8 v225, v249 offset:64
	ds_read_b64 v[150:151], v243 offset:9224
	v_mfma_scale_f32_16x16x128_f8f6f4 a[52:55], v[70:75], v[190:195], a[52:55], v217, v237 op_sel_hi:[0,0,0] cbsz:2 blgp:2
	s_add_u32 m0, s32, 0x19c00
	buffer_load_dwordx4 v4, s[36:39], 0 offen lds
	v_mfma_scale_f32_16x16x128_f8f6f4 a[56:59], v[70:75], v[196:201], a[56:59], v217, v238 op_sel_hi:[0,0,0] cbsz:2 blgp:2
	ds_read_u8 v226, v249 offset:128
	ds_read_b64 v[152:153], v243 offset:9232
	v_mfma_scale_f32_16x16x128_f8f6f4 a[60:63], v[70:75], v[202:207], a[60:63], v217, v239 op_sel_hi:[0,0,0] cbsz:2 blgp:2
	ds_read_u8 v227, v249 offset:192
	ds_read_b64 v[154:155], v243 offset:10752
	v_mfma_scale_f32_16x16x128_f8f6f4 a[64:67], v[76:81], v[160:165], a[64:67], v218, v232 op_sel_hi:[0,0,0] cbsz:2 blgp:2
	ds_read_u8 v228, v249 offset:256
	ds_read_b64 v[156:157], v243 offset:10760
	v_mfma_scale_f32_16x16x128_f8f6f4 a[68:71], v[76:81], v[166:171], a[68:71], v218, v233 op_sel_hi:[0,0,0] cbsz:2 blgp:2
	ds_read_u8 v229, v249 offset:320
	ds_read_b64 v[158:159], v243 offset:10768
	v_mfma_scale_f32_16x16x128_f8f6f4 a[72:75], v[76:81], v[172:177], a[72:75], v218, v234 op_sel_hi:[0,0,0] cbsz:2 blgp:2
	buffer_load_dwordx4 v4, s[36:39], 0 offen offset:1024 lds
	v_mfma_scale_f32_16x16x128_f8f6f4 a[76:79], v[76:81], v[178:183], a[76:79], v218, v235 op_sel_hi:[0,0,0] cbsz:2 blgp:2
	ds_read_u8 v230, v249 offset:384
	v_mfma_scale_f32_16x16x128_f8f6f4 a[80:83], v[76:81], v[184:189], a[80:83], v218, v236 op_sel_hi:[0,0,0] cbsz:2 blgp:2
	ds_read_u8 v231, v249 offset:448
	v_mfma_scale_f32_16x16x128_f8f6f4 a[84:87], v[76:81], v[190:195], a[84:87], v218, v237 op_sel_hi:[0,0,0] cbsz:2 blgp:2
	ds_read_b64 v[16:17], v240
	v_mfma_scale_f32_16x16x128_f8f6f4 a[88:91], v[76:81], v[196:201], a[88:91], v218, v238 op_sel_hi:[0,0,0] cbsz:2 blgp:2
	buffer_load_dwordx4 v4, s[36:39], 0 offen offset:2048 lds
	v_mfma_scale_f32_16x16x128_f8f6f4 a[92:95], v[76:81], v[202:207], a[92:95], v218, v239 op_sel_hi:[0,0,0] cbsz:2 blgp:2
	ds_read_b64 v[18:19], v240 offset:8
	v_mfma_scale_f32_16x16x128_f8f6f4 a[96:99], v[82:87], v[160:165], a[96:99], v219, v232 op_sel_hi:[0,0,0] cbsz:2 blgp:2
	ds_read_b64 v[20:21], v240 offset:16
	v_mfma_scale_f32_16x16x128_f8f6f4 a[100:103], v[82:87], v[166:171], a[100:103], v219, v233 op_sel_hi:[0,0,0] cbsz:2 blgp:2
	ds_read_b64 v[22:23], v240 offset:1536
	v_mfma_scale_f32_16x16x128_f8f6f4 a[104:107], v[82:87], v[172:177], a[104:107], v219, v234 op_sel_hi:[0,0,0] cbsz:2 blgp:2
	ds_read_b64 v[24:25], v240 offset:1544
	v_mfma_scale_f32_16x16x128_f8f6f4 a[108:111], v[82:87], v[178:183], a[108:111], v219, v235 op_sel_hi:[0,0,0] cbsz:2 blgp:2
	s_add_u32 m0, s32, 0x1f000
	buffer_load_dwordx4 v3, s[40:43], 0 offen lds
	v_mfma_scale_f32_16x16x128_f8f6f4 a[112:115], v[82:87], v[184:189], a[112:115], v219, v236 op_sel_hi:[0,0,0] cbsz:2 blgp:2
	ds_read_b64 v[26:27], v240 offset:1552
	v_mfma_scale_f32_16x16x128_f8f6f4 a[116:119], v[82:87], v[190:195], a[116:119], v219, v237 op_sel_hi:[0,0,0] cbsz:2 blgp:2
	ds_read_b64 v[28:29], v240 offset:3072
	v_mfma_scale_f32_16x16x128_f8f6f4 a[120:123], v[82:87], v[196:201], a[120:123], v219, v238 op_sel_hi:[0,0,0] cbsz:2 blgp:2
	ds_read_b64 v[30:31], v240 offset:3080
	v_mfma_scale_f32_16x16x128_f8f6f4 a[124:127], v[82:87], v[202:207], a[124:127], v219, v239 op_sel_hi:[0,0,0] cbsz:2 blgp:2
	ds_read_b64 v[32:33], v240 offset:3088
	v_mfma_scale_f32_16x16x128_f8f6f4 a[128:131], v[88:93], v[160:165], a[128:131], v220, v232 op_sel_hi:[0,0,0] cbsz:2 blgp:2
	buffer_load_dwordx4 v3, s[40:43], 0 offen offset:1024 lds
	v_mfma_scale_f32_16x16x128_f8f6f4 a[132:135], v[88:93], v[166:171], a[132:135], v220, v233 op_sel_hi:[0,0,0] cbsz:2 blgp:2
	ds_read_b64 v[34:35], v240 offset:4608
	v_mfma_scale_f32_16x16x128_f8f6f4 a[136:139], v[88:93], v[172:177], a[136:139], v220, v234 op_sel_hi:[0,0,0] cbsz:2 blgp:2
	ds_read_b64 v[36:37], v240 offset:4616
	v_mfma_scale_f32_16x16x128_f8f6f4 a[140:143], v[88:93], v[178:183], a[140:143], v220, v235 op_sel_hi:[0,0,0] cbsz:2 blgp:2
	ds_read_b64 v[38:39], v240 offset:4624
	v_mfma_scale_f32_16x16x128_f8f6f4 a[144:147], v[88:93], v[184:189], a[144:147], v220, v236 op_sel_hi:[0,0,0] cbsz:2 blgp:2
	buffer_load_dwordx4 v3, s[40:43], 0 offen offset:2048 lds
	v_mfma_scale_f32_16x16x128_f8f6f4 a[148:151], v[88:93], v[190:195], a[148:151], v220, v237 op_sel_hi:[0,0,0] cbsz:2 blgp:2
	ds_read_b64 v[40:41], v240 offset:6144
	v_mfma_scale_f32_16x16x128_f8f6f4 a[152:155], v[88:93], v[196:201], a[152:155], v220, v238 op_sel_hi:[0,0,0] cbsz:2 blgp:2
	ds_read_b64 v[42:43], v240 offset:6152
	v_mfma_scale_f32_16x16x128_f8f6f4 a[156:159], v[88:93], v[202:207], a[156:159], v220, v239 op_sel_hi:[0,0,0] cbsz:2 blgp:2
	ds_read_b64 v[44:45], v240 offset:6160
	v_mfma_scale_f32_16x16x128_f8f6f4 a[160:163], v[94:99], v[160:165], a[160:163], v221, v232 op_sel_hi:[0,0,0] cbsz:2 blgp:2
	ds_read_b64 v[46:47], v240 offset:7680
	v_mfma_scale_f32_16x16x128_f8f6f4 a[164:167], v[94:99], v[166:171], a[164:167], v221, v233 op_sel_hi:[0,0,0] cbsz:2 blgp:2
	s_add_u32 m0, s32, 0x1fc00
	buffer_load_dwordx4 v4, s[40:43], 0 offen lds
	v_mfma_scale_f32_16x16x128_f8f6f4 a[168:171], v[94:99], v[172:177], a[168:171], v221, v234 op_sel_hi:[0,0,0] cbsz:2 blgp:2
	ds_read_b64 v[48:49], v240 offset:7688
	v_mfma_scale_f32_16x16x128_f8f6f4 a[172:175], v[94:99], v[178:183], a[172:175], v221, v235 op_sel_hi:[0,0,0] cbsz:2 blgp:2
	ds_read_b64 v[50:51], v240 offset:7696
	v_mfma_scale_f32_16x16x128_f8f6f4 a[176:179], v[94:99], v[184:189], a[176:179], v221, v236 op_sel_hi:[0,0,0] cbsz:2 blgp:2
	ds_read_b64 v[52:53], v240 offset:9216
	v_mfma_scale_f32_16x16x128_f8f6f4 a[180:183], v[94:99], v[190:195], a[180:183], v221, v237 op_sel_hi:[0,0,0] cbsz:2 blgp:2
	buffer_load_dwordx4 v4, s[40:43], 0 offen offset:1024 lds
	v_mfma_scale_f32_16x16x128_f8f6f4 a[184:187], v[94:99], v[196:201], a[184:187], v221, v238 op_sel_hi:[0,0,0] cbsz:2 blgp:2
	ds_read_b64 v[54:55], v240 offset:9224
	v_mfma_scale_f32_16x16x128_f8f6f4 a[188:191], v[94:99], v[202:207], a[188:191], v221, v239 op_sel_hi:[0,0,0] cbsz:2 blgp:2
	ds_read_b64 v[56:57], v240 offset:9232
	v_mfma_scale_f32_16x16x128_f8f6f4 a[192:195], v[100:105], v[160:165], a[192:195], v222, v232 op_sel_hi:[0,0,0] cbsz:2 blgp:2
	ds_read_b64 v[58:59], v240 offset:10752
	v_mfma_scale_f32_16x16x128_f8f6f4 a[196:199], v[100:105], v[166:171], a[196:199], v222, v233 op_sel_hi:[0,0,0] cbsz:2 blgp:2
	ds_read_b64 v[60:61], v240 offset:10760
	v_mfma_scale_f32_16x16x128_f8f6f4 a[200:203], v[100:105], v[172:177], a[200:203], v222, v234 op_sel_hi:[0,0,0] cbsz:2 blgp:2
	buffer_load_dwordx4 v4, s[40:43], 0 offen offset:2048 lds
	v_mfma_scale_f32_16x16x128_f8f6f4 a[204:207], v[100:105], v[178:183], a[204:207], v222, v235 op_sel_hi:[0,0,0] cbsz:2 blgp:2
	ds_read_b64 v[62:63], v240 offset:10768
	v_mfma_scale_f32_16x16x128_f8f6f4 a[208:211], v[100:105], v[184:189], a[208:211], v222, v236 op_sel_hi:[0,0,0] cbsz:2 blgp:2
	ds_read_b64 v[112:113], v243
	v_mfma_scale_f32_16x16x128_f8f6f4 a[212:215], v[100:105], v[190:195], a[212:215], v222, v237 op_sel_hi:[0,0,0] cbsz:2 blgp:2
	ds_read_b64 v[114:115], v243 offset:8
	v_mfma_scale_f32_16x16x128_f8f6f4 a[216:219], v[100:105], v[196:201], a[216:219], v222, v238 op_sel_hi:[0,0,0] cbsz:2 blgp:2
	s_add_u32 m0, s33, 0x25000
	buffer_load_dword v13, s[44:47], 0 offen lds
	v_mfma_scale_f32_16x16x128_f8f6f4 a[220:223], v[100:105], v[202:207], a[220:223], v222, v239 op_sel_hi:[0,0,0] cbsz:2 blgp:2
	ds_read_b64 v[116:117], v243 offset:16
	v_mfma_scale_f32_16x16x128_f8f6f4 a[224:227], v[106:111], v[160:165], a[224:227], v223, v232 op_sel_hi:[0,0,0] cbsz:2 blgp:2
	ds_read_b64 v[118:119], v243 offset:1536
	v_mfma_scale_f32_16x16x128_f8f6f4 a[228:231], v[106:111], v[166:171], a[228:231], v223, v233 op_sel_hi:[0,0,0] cbsz:2 blgp:2
	ds_read_b64 v[120:121], v243 offset:1544
	v_mfma_scale_f32_16x16x128_f8f6f4 a[232:235], v[106:111], v[172:177], a[232:235], v223, v234 op_sel_hi:[0,0,0] cbsz:2 blgp:2
	ds_read_b64 v[122:123], v243 offset:1552
	v_mfma_scale_f32_16x16x128_f8f6f4 a[236:239], v[106:111], v[178:183], a[236:239], v223, v235 op_sel_hi:[0,0,0] cbsz:2 blgp:2
	s_add_u32 m0, s33, 0x25400
	buffer_load_dword v13, s[48:51], 0 offen lds
	v_mfma_scale_f32_16x16x128_f8f6f4 a[240:243], v[106:111], v[184:189], a[240:243], v223, v236 op_sel_hi:[0,0,0] cbsz:2 blgp:2
	ds_read_b64 v[124:125], v243 offset:3072
	v_mfma_scale_f32_16x16x128_f8f6f4 a[244:247], v[106:111], v[190:195], a[244:247], v223, v237 op_sel_hi:[0,0,0] cbsz:2 blgp:2
	ds_read_b64 v[126:127], v243 offset:3080
	v_mfma_scale_f32_16x16x128_f8f6f4 a[248:251], v[106:111], v[196:201], a[248:251], v223, v238 op_sel_hi:[0,0,0] cbsz:2 blgp:2
	ds_read_b64 v[128:129], v243 offset:3088
	v_mfma_scale_f32_16x16x128_f8f6f4 a[252:255], v[106:111], v[202:207], a[252:255], v223, v239 op_sel_hi:[0,0,0] cbsz:2 blgp:2
	ds_read_b64 v[130:131], v243 offset:4608
	s_add_u32 s36, s36, 0x60000
	s_addc_u32 s37, s37, 0
	s_add_u32 s40, s40, 0x60000
	s_addc_u32 s41, s41, 0
	s_add_u32 s44, s44, 0x4000
	s_addc_u32 s45, s45, 0
	s_add_u32 s48, s48, 0x4000
	s_addc_u32 s49, s49, 0
	s_sub_u32 s30, s30, 1
	s_cmp_lg_u32 s30, 0
	s_cbranch_scc1 L_ed8
	v_lshrrev_b32_e32 v2, 4, v1
	v_lshlrev_b32_e32 v2, 2, v2
	s_mov_b32 s25, 0x2000
	v_mul_lo_u32 v2, s25, v2
	v_and_b32_e32 v9, 15, v1
	v_lshlrev_b32_e32 v9, 1, v9
	v_add_u32_e32 v9, v2, v9
	v_add_u32_e32 v10, 0x2000, v9
	v_add_u32_e32 v11, 0x4000, v9
	v_add_u32_e32 v12, 0x6000, v9
	s_nop 15
	s_nop 15
	s_add_u32 s28, s26, 0
	s_addc_u32 s29, s27, 0
	v_accvgpr_read_b32 v2, a0
	s_nop 4
	v_cvt_pk_bf16_f32 v2, v2, 0
	global_store_short v9, v2, s[28:29]
	v_accvgpr_read_b32 v2, a1
	s_nop 4
	v_cvt_pk_bf16_f32 v2, v2, 0
	global_store_short v10, v2, s[28:29]
	v_accvgpr_read_b32 v2, a2
	s_nop 4
	v_cvt_pk_bf16_f32 v2, v2, 0
	global_store_short v11, v2, s[28:29]
	v_accvgpr_read_b32 v2, a3
	s_nop 4
	v_cvt_pk_bf16_f32 v2, v2, 0
	global_store_short v12, v2, s[28:29]
	s_add_u32 s28, s26, 32
	s_addc_u32 s29, s27, 0
	v_accvgpr_read_b32 v2, a4
	s_nop 4
	v_cvt_pk_bf16_f32 v2, v2, 0
	global_store_short v9, v2, s[28:29]
	v_accvgpr_read_b32 v2, a5
	s_nop 4
	v_cvt_pk_bf16_f32 v2, v2, 0
	global_store_short v10, v2, s[28:29]
	v_accvgpr_read_b32 v2, a6
	s_nop 4
	v_cvt_pk_bf16_f32 v2, v2, 0
	global_store_short v11, v2, s[28:29]
	v_accvgpr_read_b32 v2, a7
	s_nop 4
	v_cvt_pk_bf16_f32 v2, v2, 0
	global_store_short v12, v2, s[28:29]
	s_add_u32 s28, s26, 64
	s_addc_u32 s29, s27, 0
	v_accvgpr_read_b32 v2, a8
	s_nop 4
	v_cvt_pk_bf16_f32 v2, v2, 0
	global_store_short v9, v2, s[28:29]
	v_accvgpr_read_b32 v2, a9
	s_nop 4
	v_cvt_pk_bf16_f32 v2, v2, 0
	global_store_short v10, v2, s[28:29]
	v_accvgpr_read_b32 v2, a10
	s_nop 4
	v_cvt_pk_bf16_f32 v2, v2, 0
	global_store_short v11, v2, s[28:29]
	v_accvgpr_read_b32 v2, a11
	s_nop 4
	v_cvt_pk_bf16_f32 v2, v2, 0
	global_store_short v12, v2, s[28:29]
	s_add_u32 s28, s26, 0x60
	s_addc_u32 s29, s27, 0
	v_accvgpr_read_b32 v2, a12
	s_nop 4
	v_cvt_pk_bf16_f32 v2, v2, 0
	global_store_short v9, v2, s[28:29]
	v_accvgpr_read_b32 v2, a13
	s_nop 4
	v_cvt_pk_bf16_f32 v2, v2, 0
	global_store_short v10, v2, s[28:29]
	v_accvgpr_read_b32 v2, a14
	s_nop 4
	v_cvt_pk_bf16_f32 v2, v2, 0
	global_store_short v11, v2, s[28:29]
	v_accvgpr_read_b32 v2, a15
	s_nop 4
	v_cvt_pk_bf16_f32 v2, v2, 0
	global_store_short v12, v2, s[28:29]
	s_add_u32 s28, s26, 0x80
	s_addc_u32 s29, s27, 0
	v_accvgpr_read_b32 v2, a16
	s_nop 4
	v_cvt_pk_bf16_f32 v2, v2, 0
	global_store_short v9, v2, s[28:29]
	v_accvgpr_read_b32 v2, a17
	s_nop 4
	v_cvt_pk_bf16_f32 v2, v2, 0
	global_store_short v10, v2, s[28:29]
	v_accvgpr_read_b32 v2, a18
	s_nop 4
	v_cvt_pk_bf16_f32 v2, v2, 0
	global_store_short v11, v2, s[28:29]
	v_accvgpr_read_b32 v2, a19
	s_nop 4
	v_cvt_pk_bf16_f32 v2, v2, 0
	global_store_short v12, v2, s[28:29]
	s_add_u32 s28, s26, 0xa0
	s_addc_u32 s29, s27, 0
	v_accvgpr_read_b32 v2, a20
	s_nop 4
	v_cvt_pk_bf16_f32 v2, v2, 0
	global_store_short v9, v2, s[28:29]
	v_accvgpr_read_b32 v2, a21
	s_nop 4
	v_cvt_pk_bf16_f32 v2, v2, 0
	global_store_short v10, v2, s[28:29]
	v_accvgpr_read_b32 v2, a22
	s_nop 4
	v_cvt_pk_bf16_f32 v2, v2, 0
	global_store_short v11, v2, s[28:29]
	v_accvgpr_read_b32 v2, a23
	s_nop 4
	v_cvt_pk_bf16_f32 v2, v2, 0
	global_store_short v12, v2, s[28:29]
	s_add_u32 s28, s26, 0xc0
	s_addc_u32 s29, s27, 0
	v_accvgpr_read_b32 v2, a24
	s_nop 4
	v_cvt_pk_bf16_f32 v2, v2, 0
	global_store_short v9, v2, s[28:29]
	v_accvgpr_read_b32 v2, a25
	s_nop 4
	v_cvt_pk_bf16_f32 v2, v2, 0
	global_store_short v10, v2, s[28:29]
	v_accvgpr_read_b32 v2, a26
	s_nop 4
	v_cvt_pk_bf16_f32 v2, v2, 0
	global_store_short v11, v2, s[28:29]
	v_accvgpr_read_b32 v2, a27
	s_nop 4
	v_cvt_pk_bf16_f32 v2, v2, 0
	global_store_short v12, v2, s[28:29]
	s_add_u32 s28, s26, 0xe0
	s_addc_u32 s29, s27, 0
	v_accvgpr_read_b32 v2, a28
	s_nop 4
	v_cvt_pk_bf16_f32 v2, v2, 0
	global_store_short v9, v2, s[28:29]
	v_accvgpr_read_b32 v2, a29
	s_nop 4
	v_cvt_pk_bf16_f32 v2, v2, 0
	global_store_short v10, v2, s[28:29]
	v_accvgpr_read_b32 v2, a30
	s_nop 4
	v_cvt_pk_bf16_f32 v2, v2, 0
	global_store_short v11, v2, s[28:29]
	v_accvgpr_read_b32 v2, a31
	s_nop 4
	v_cvt_pk_bf16_f32 v2, v2, 0
	global_store_short v12, v2, s[28:29]
	s_add_u32 s28, s26, 0x20000
	s_addc_u32 s29, s27, 0
	v_accvgpr_read_b32 v2, a32
	s_nop 4
	v_cvt_pk_bf16_f32 v2, v2, 0
	global_store_short v9, v2, s[28:29]
	v_accvgpr_read_b32 v2, a33
	s_nop 4
	v_cvt_pk_bf16_f32 v2, v2, 0
	global_store_short v10, v2, s[28:29]
	v_accvgpr_read_b32 v2, a34
	s_nop 4
	v_cvt_pk_bf16_f32 v2, v2, 0
	global_store_short v11, v2, s[28:29]
	v_accvgpr_read_b32 v2, a35
	s_nop 4
	v_cvt_pk_bf16_f32 v2, v2, 0
	global_store_short v12, v2, s[28:29]
	s_add_u32 s28, s26, 0x20020
	s_addc_u32 s29, s27, 0
	v_accvgpr_read_b32 v2, a36
	s_nop 4
	v_cvt_pk_bf16_f32 v2, v2, 0
	global_store_short v9, v2, s[28:29]
	v_accvgpr_read_b32 v2, a37
	s_nop 4
	v_cvt_pk_bf16_f32 v2, v2, 0
	global_store_short v10, v2, s[28:29]
	v_accvgpr_read_b32 v2, a38
	s_nop 4
	v_cvt_pk_bf16_f32 v2, v2, 0
	global_store_short v11, v2, s[28:29]
	v_accvgpr_read_b32 v2, a39
	s_nop 4
	v_cvt_pk_bf16_f32 v2, v2, 0
	global_store_short v12, v2, s[28:29]
	s_add_u32 s28, s26, 0x20040
	s_addc_u32 s29, s27, 0
	v_accvgpr_read_b32 v2, a40
	s_nop 4
	v_cvt_pk_bf16_f32 v2, v2, 0
	global_store_short v9, v2, s[28:29]
	v_accvgpr_read_b32 v2, a41
	s_nop 4
	v_cvt_pk_bf16_f32 v2, v2, 0
	global_store_short v10, v2, s[28:29]
	v_accvgpr_read_b32 v2, a42
	s_nop 4
	v_cvt_pk_bf16_f32 v2, v2, 0
	global_store_short v11, v2, s[28:29]
	v_accvgpr_read_b32 v2, a43
	s_nop 4
	v_cvt_pk_bf16_f32 v2, v2, 0
	global_store_short v12, v2, s[28:29]
	s_add_u32 s28, s26, 0x20060
	s_addc_u32 s29, s27, 0
	v_accvgpr_read_b32 v2, a44
	s_nop 4
	v_cvt_pk_bf16_f32 v2, v2, 0
	global_store_short v9, v2, s[28:29]
	v_accvgpr_read_b32 v2, a45
	s_nop 4
	v_cvt_pk_bf16_f32 v2, v2, 0
	global_store_short v10, v2, s[28:29]
	v_accvgpr_read_b32 v2, a46
	s_nop 4
	v_cvt_pk_bf16_f32 v2, v2, 0
	global_store_short v11, v2, s[28:29]
	v_accvgpr_read_b32 v2, a47
	s_nop 4
	v_cvt_pk_bf16_f32 v2, v2, 0
	global_store_short v12, v2, s[28:29]
	s_add_u32 s28, s26, 0x20080
	s_addc_u32 s29, s27, 0
	v_accvgpr_read_b32 v2, a48
	s_nop 4
	v_cvt_pk_bf16_f32 v2, v2, 0
	global_store_short v9, v2, s[28:29]
	v_accvgpr_read_b32 v2, a49
	s_nop 4
	v_cvt_pk_bf16_f32 v2, v2, 0
	global_store_short v10, v2, s[28:29]
	v_accvgpr_read_b32 v2, a50
	s_nop 4
	v_cvt_pk_bf16_f32 v2, v2, 0
	global_store_short v11, v2, s[28:29]
	v_accvgpr_read_b32 v2, a51
	s_nop 4
	v_cvt_pk_bf16_f32 v2, v2, 0
	global_store_short v12, v2, s[28:29]
	s_add_u32 s28, s26, 0x200a0
	s_addc_u32 s29, s27, 0
	v_accvgpr_read_b32 v2, a52
	s_nop 4
	v_cvt_pk_bf16_f32 v2, v2, 0
	global_store_short v9, v2, s[28:29]
	v_accvgpr_read_b32 v2, a53
	s_nop 4
	v_cvt_pk_bf16_f32 v2, v2, 0
	global_store_short v10, v2, s[28:29]
	v_accvgpr_read_b32 v2, a54
	s_nop 4
	v_cvt_pk_bf16_f32 v2, v2, 0
	global_store_short v11, v2, s[28:29]
	v_accvgpr_read_b32 v2, a55
	s_nop 4
	v_cvt_pk_bf16_f32 v2, v2, 0
	global_store_short v12, v2, s[28:29]
	s_add_u32 s28, s26, 0x200c0
	s_addc_u32 s29, s27, 0
	v_accvgpr_read_b32 v2, a56
	s_nop 4
	v_cvt_pk_bf16_f32 v2, v2, 0
	global_store_short v9, v2, s[28:29]
	v_accvgpr_read_b32 v2, a57
	s_nop 4
	v_cvt_pk_bf16_f32 v2, v2, 0
	global_store_short v10, v2, s[28:29]
	v_accvgpr_read_b32 v2, a58
	s_nop 4
	v_cvt_pk_bf16_f32 v2, v2, 0
	global_store_short v11, v2, s[28:29]
	v_accvgpr_read_b32 v2, a59
	s_nop 4
	v_cvt_pk_bf16_f32 v2, v2, 0
	global_store_short v12, v2, s[28:29]
	s_add_u32 s28, s26, 0x200e0
	s_addc_u32 s29, s27, 0
	v_accvgpr_read_b32 v2, a60
	s_nop 4
	v_cvt_pk_bf16_f32 v2, v2, 0
	global_store_short v9, v2, s[28:29]
	v_accvgpr_read_b32 v2, a61
	s_nop 4
	v_cvt_pk_bf16_f32 v2, v2, 0
	global_store_short v10, v2, s[28:29]
	v_accvgpr_read_b32 v2, a62
	s_nop 4
	v_cvt_pk_bf16_f32 v2, v2, 0
	global_store_short v11, v2, s[28:29]
	v_accvgpr_read_b32 v2, a63
	s_nop 4
	v_cvt_pk_bf16_f32 v2, v2, 0
	global_store_short v12, v2, s[28:29]
	s_add_u32 s28, s26, 0x40000
	s_addc_u32 s29, s27, 0
	v_accvgpr_read_b32 v2, a64
	s_nop 4
	v_cvt_pk_bf16_f32 v2, v2, 0
	global_store_short v9, v2, s[28:29]
	v_accvgpr_read_b32 v2, a65
	s_nop 4
	v_cvt_pk_bf16_f32 v2, v2, 0
	global_store_short v10, v2, s[28:29]
	v_accvgpr_read_b32 v2, a66
	s_nop 4
	v_cvt_pk_bf16_f32 v2, v2, 0
	global_store_short v11, v2, s[28:29]
	v_accvgpr_read_b32 v2, a67
	s_nop 4
	v_cvt_pk_bf16_f32 v2, v2, 0
	global_store_short v12, v2, s[28:29]
	s_add_u32 s28, s26, 0x40020
	s_addc_u32 s29, s27, 0
	v_accvgpr_read_b32 v2, a68
	s_nop 4
	v_cvt_pk_bf16_f32 v2, v2, 0
	global_store_short v9, v2, s[28:29]
	v_accvgpr_read_b32 v2, a69
	s_nop 4
	v_cvt_pk_bf16_f32 v2, v2, 0
	global_store_short v10, v2, s[28:29]
	v_accvgpr_read_b32 v2, a70
	s_nop 4
	v_cvt_pk_bf16_f32 v2, v2, 0
	global_store_short v11, v2, s[28:29]
	v_accvgpr_read_b32 v2, a71
	s_nop 4
	v_cvt_pk_bf16_f32 v2, v2, 0
	global_store_short v12, v2, s[28:29]
	s_add_u32 s28, s26, 0x40040
	s_addc_u32 s29, s27, 0
	v_accvgpr_read_b32 v2, a72
	s_nop 4
	v_cvt_pk_bf16_f32 v2, v2, 0
	global_store_short v9, v2, s[28:29]
	v_accvgpr_read_b32 v2, a73
	s_nop 4
	v_cvt_pk_bf16_f32 v2, v2, 0
	global_store_short v10, v2, s[28:29]
	v_accvgpr_read_b32 v2, a74
	s_nop 4
	v_cvt_pk_bf16_f32 v2, v2, 0
	global_store_short v11, v2, s[28:29]
	v_accvgpr_read_b32 v2, a75
	s_nop 4
	v_cvt_pk_bf16_f32 v2, v2, 0
	global_store_short v12, v2, s[28:29]
	s_add_u32 s28, s26, 0x40060
	s_addc_u32 s29, s27, 0
	v_accvgpr_read_b32 v2, a76
	s_nop 4
	v_cvt_pk_bf16_f32 v2, v2, 0
	global_store_short v9, v2, s[28:29]
	v_accvgpr_read_b32 v2, a77
	s_nop 4
	v_cvt_pk_bf16_f32 v2, v2, 0
	global_store_short v10, v2, s[28:29]
	v_accvgpr_read_b32 v2, a78
	s_nop 4
	v_cvt_pk_bf16_f32 v2, v2, 0
	global_store_short v11, v2, s[28:29]
	v_accvgpr_read_b32 v2, a79
	s_nop 4
	v_cvt_pk_bf16_f32 v2, v2, 0
	global_store_short v12, v2, s[28:29]
	s_add_u32 s28, s26, 0x40080
	s_addc_u32 s29, s27, 0
	v_accvgpr_read_b32 v2, a80
	s_nop 4
	v_cvt_pk_bf16_f32 v2, v2, 0
	global_store_short v9, v2, s[28:29]
	v_accvgpr_read_b32 v2, a81
	s_nop 4
	v_cvt_pk_bf16_f32 v2, v2, 0
	global_store_short v10, v2, s[28:29]
	v_accvgpr_read_b32 v2, a82
	s_nop 4
	v_cvt_pk_bf16_f32 v2, v2, 0
	global_store_short v11, v2, s[28:29]
	v_accvgpr_read_b32 v2, a83
	s_nop 4
	v_cvt_pk_bf16_f32 v2, v2, 0
	global_store_short v12, v2, s[28:29]
	s_add_u32 s28, s26, 0x400a0
	s_addc_u32 s29, s27, 0
	v_accvgpr_read_b32 v2, a84
	s_nop 4
	v_cvt_pk_bf16_f32 v2, v2, 0
	global_store_short v9, v2, s[28:29]
	v_accvgpr_read_b32 v2, a85
	s_nop 4
	v_cvt_pk_bf16_f32 v2, v2, 0
	global_store_short v10, v2, s[28:29]
	v_accvgpr_read_b32 v2, a86
	s_nop 4
	v_cvt_pk_bf16_f32 v2, v2, 0
	global_store_short v11, v2, s[28:29]
	v_accvgpr_read_b32 v2, a87
	s_nop 4
	v_cvt_pk_bf16_f32 v2, v2, 0
	global_store_short v12, v2, s[28:29]
	s_add_u32 s28, s26, 0x400c0
	s_addc_u32 s29, s27, 0
	v_accvgpr_read_b32 v2, a88
	s_nop 4
	v_cvt_pk_bf16_f32 v2, v2, 0
	global_store_short v9, v2, s[28:29]
	v_accvgpr_read_b32 v2, a89
	s_nop 4
	v_cvt_pk_bf16_f32 v2, v2, 0
	global_store_short v10, v2, s[28:29]
	v_accvgpr_read_b32 v2, a90
	s_nop 4
	v_cvt_pk_bf16_f32 v2, v2, 0
	global_store_short v11, v2, s[28:29]
	v_accvgpr_read_b32 v2, a91
	s_nop 4
	v_cvt_pk_bf16_f32 v2, v2, 0
	global_store_short v12, v2, s[28:29]
	s_add_u32 s28, s26, 0x400e0
	s_addc_u32 s29, s27, 0
	v_accvgpr_read_b32 v2, a92
	s_nop 4
	v_cvt_pk_bf16_f32 v2, v2, 0
	global_store_short v9, v2, s[28:29]
	v_accvgpr_read_b32 v2, a93
	s_nop 4
	v_cvt_pk_bf16_f32 v2, v2, 0
	global_store_short v10, v2, s[28:29]
	v_accvgpr_read_b32 v2, a94
	s_nop 4
	v_cvt_pk_bf16_f32 v2, v2, 0
	global_store_short v11, v2, s[28:29]
	v_accvgpr_read_b32 v2, a95
	s_nop 4
	v_cvt_pk_bf16_f32 v2, v2, 0
	global_store_short v12, v2, s[28:29]
	s_add_u32 s28, s26, 0x60000
	s_addc_u32 s29, s27, 0
	v_accvgpr_read_b32 v2, a96
	s_nop 4
	v_cvt_pk_bf16_f32 v2, v2, 0
	global_store_short v9, v2, s[28:29]
	v_accvgpr_read_b32 v2, a97
	s_nop 4
	v_cvt_pk_bf16_f32 v2, v2, 0
	global_store_short v10, v2, s[28:29]
	v_accvgpr_read_b32 v2, a98
	s_nop 4
	v_cvt_pk_bf16_f32 v2, v2, 0
	global_store_short v11, v2, s[28:29]
	v_accvgpr_read_b32 v2, a99
	s_nop 4
	v_cvt_pk_bf16_f32 v2, v2, 0
	global_store_short v12, v2, s[28:29]
	s_add_u32 s28, s26, 0x60020
	s_addc_u32 s29, s27, 0
	v_accvgpr_read_b32 v2, a100
	s_nop 4
	v_cvt_pk_bf16_f32 v2, v2, 0
	global_store_short v9, v2, s[28:29]
	v_accvgpr_read_b32 v2, a101
	s_nop 4
	v_cvt_pk_bf16_f32 v2, v2, 0
	global_store_short v10, v2, s[28:29]
	v_accvgpr_read_b32 v2, a102
	s_nop 4
	v_cvt_pk_bf16_f32 v2, v2, 0
	global_store_short v11, v2, s[28:29]
	v_accvgpr_read_b32 v2, a103
	s_nop 4
	v_cvt_pk_bf16_f32 v2, v2, 0
	global_store_short v12, v2, s[28:29]
	s_add_u32 s28, s26, 0x60040
	s_addc_u32 s29, s27, 0
	v_accvgpr_read_b32 v2, a104
	s_nop 4
	v_cvt_pk_bf16_f32 v2, v2, 0
	global_store_short v9, v2, s[28:29]
	v_accvgpr_read_b32 v2, a105
	s_nop 4
	v_cvt_pk_bf16_f32 v2, v2, 0
	global_store_short v10, v2, s[28:29]
	v_accvgpr_read_b32 v2, a106
	s_nop 4
	v_cvt_pk_bf16_f32 v2, v2, 0
	global_store_short v11, v2, s[28:29]
	v_accvgpr_read_b32 v2, a107
	s_nop 4
	v_cvt_pk_bf16_f32 v2, v2, 0
	global_store_short v12, v2, s[28:29]
	s_add_u32 s28, s26, 0x60060
	s_addc_u32 s29, s27, 0
	v_accvgpr_read_b32 v2, a108
	s_nop 4
	v_cvt_pk_bf16_f32 v2, v2, 0
	global_store_short v9, v2, s[28:29]
	v_accvgpr_read_b32 v2, a109
	s_nop 4
	v_cvt_pk_bf16_f32 v2, v2, 0
	global_store_short v10, v2, s[28:29]
	v_accvgpr_read_b32 v2, a110
	s_nop 4
	v_cvt_pk_bf16_f32 v2, v2, 0
	global_store_short v11, v2, s[28:29]
	v_accvgpr_read_b32 v2, a111
	s_nop 4
	v_cvt_pk_bf16_f32 v2, v2, 0
	global_store_short v12, v2, s[28:29]
	s_add_u32 s28, s26, 0x60080
	s_addc_u32 s29, s27, 0
	v_accvgpr_read_b32 v2, a112
	s_nop 4
	v_cvt_pk_bf16_f32 v2, v2, 0
	global_store_short v9, v2, s[28:29]
	v_accvgpr_read_b32 v2, a113
	s_nop 4
	v_cvt_pk_bf16_f32 v2, v2, 0
	global_store_short v10, v2, s[28:29]
	v_accvgpr_read_b32 v2, a114
	s_nop 4
	v_cvt_pk_bf16_f32 v2, v2, 0
	global_store_short v11, v2, s[28:29]
	v_accvgpr_read_b32 v2, a115
	s_nop 4
	v_cvt_pk_bf16_f32 v2, v2, 0
	global_store_short v12, v2, s[28:29]
	s_add_u32 s28, s26, 0x600a0
	s_addc_u32 s29, s27, 0
	v_accvgpr_read_b32 v2, a116
	s_nop 4
	v_cvt_pk_bf16_f32 v2, v2, 0
	global_store_short v9, v2, s[28:29]
	v_accvgpr_read_b32 v2, a117
	s_nop 4
	v_cvt_pk_bf16_f32 v2, v2, 0
	global_store_short v10, v2, s[28:29]
	v_accvgpr_read_b32 v2, a118
	s_nop 4
	v_cvt_pk_bf16_f32 v2, v2, 0
	global_store_short v11, v2, s[28:29]
	v_accvgpr_read_b32 v2, a119
	s_nop 4
	v_cvt_pk_bf16_f32 v2, v2, 0
	global_store_short v12, v2, s[28:29]
	s_add_u32 s28, s26, 0x600c0
	s_addc_u32 s29, s27, 0
	v_accvgpr_read_b32 v2, a120
	s_nop 4
	v_cvt_pk_bf16_f32 v2, v2, 0
	global_store_short v9, v2, s[28:29]
	v_accvgpr_read_b32 v2, a121
	s_nop 4
	v_cvt_pk_bf16_f32 v2, v2, 0
	global_store_short v10, v2, s[28:29]
	v_accvgpr_read_b32 v2, a122
	s_nop 4
	v_cvt_pk_bf16_f32 v2, v2, 0
	global_store_short v11, v2, s[28:29]
	v_accvgpr_read_b32 v2, a123
	s_nop 4
	v_cvt_pk_bf16_f32 v2, v2, 0
	global_store_short v12, v2, s[28:29]
	s_add_u32 s28, s26, 0x600e0
	s_addc_u32 s29, s27, 0
	v_accvgpr_read_b32 v2, a124
	s_nop 4
	v_cvt_pk_bf16_f32 v2, v2, 0
	global_store_short v9, v2, s[28:29]
	v_accvgpr_read_b32 v2, a125
	s_nop 4
	v_cvt_pk_bf16_f32 v2, v2, 0
	global_store_short v10, v2, s[28:29]
	v_accvgpr_read_b32 v2, a126
	s_nop 4
	v_cvt_pk_bf16_f32 v2, v2, 0
	global_store_short v11, v2, s[28:29]
	v_accvgpr_read_b32 v2, a127
	s_nop 4
	v_cvt_pk_bf16_f32 v2, v2, 0
	global_store_short v12, v2, s[28:29]
	s_add_u32 s28, s26, 0x80000
	s_addc_u32 s29, s27, 0
	v_accvgpr_read_b32 v2, a128
	s_nop 4
	v_cvt_pk_bf16_f32 v2, v2, 0
	global_store_short v9, v2, s[28:29]
	v_accvgpr_read_b32 v2, a129
	s_nop 4
	v_cvt_pk_bf16_f32 v2, v2, 0
	global_store_short v10, v2, s[28:29]
	v_accvgpr_read_b32 v2, a130
	s_nop 4
	v_cvt_pk_bf16_f32 v2, v2, 0
	global_store_short v11, v2, s[28:29]
	v_accvgpr_read_b32 v2, a131
	s_nop 4
	v_cvt_pk_bf16_f32 v2, v2, 0
	global_store_short v12, v2, s[28:29]
	s_add_u32 s28, s26, 0x80020
	s_addc_u32 s29, s27, 0
	v_accvgpr_read_b32 v2, a132
	s_nop 4
	v_cvt_pk_bf16_f32 v2, v2, 0
	global_store_short v9, v2, s[28:29]
	v_accvgpr_read_b32 v2, a133
	s_nop 4
	v_cvt_pk_bf16_f32 v2, v2, 0
	global_store_short v10, v2, s[28:29]
	v_accvgpr_read_b32 v2, a134
	s_nop 4
	v_cvt_pk_bf16_f32 v2, v2, 0
	global_store_short v11, v2, s[28:29]
	v_accvgpr_read_b32 v2, a135
	s_nop 4
	v_cvt_pk_bf16_f32 v2, v2, 0
	global_store_short v12, v2, s[28:29]
	s_add_u32 s28, s26, 0x80040
	s_addc_u32 s29, s27, 0
	v_accvgpr_read_b32 v2, a136
	s_nop 4
	v_cvt_pk_bf16_f32 v2, v2, 0
	global_store_short v9, v2, s[28:29]
	v_accvgpr_read_b32 v2, a137
	s_nop 4
	v_cvt_pk_bf16_f32 v2, v2, 0
	global_store_short v10, v2, s[28:29]
	v_accvgpr_read_b32 v2, a138
	s_nop 4
	v_cvt_pk_bf16_f32 v2, v2, 0
	global_store_short v11, v2, s[28:29]
	v_accvgpr_read_b32 v2, a139
	s_nop 4
	v_cvt_pk_bf16_f32 v2, v2, 0
	global_store_short v12, v2, s[28:29]
	s_add_u32 s28, s26, 0x80060
	s_addc_u32 s29, s27, 0
	v_accvgpr_read_b32 v2, a140
	s_nop 4
	v_cvt_pk_bf16_f32 v2, v2, 0
	global_store_short v9, v2, s[28:29]
	v_accvgpr_read_b32 v2, a141
	s_nop 4
	v_cvt_pk_bf16_f32 v2, v2, 0
	global_store_short v10, v2, s[28:29]
	v_accvgpr_read_b32 v2, a142
	s_nop 4
	v_cvt_pk_bf16_f32 v2, v2, 0
	global_store_short v11, v2, s[28:29]
	v_accvgpr_read_b32 v2, a143
	s_nop 4
	v_cvt_pk_bf16_f32 v2, v2, 0
	global_store_short v12, v2, s[28:29]
	s_add_u32 s28, s26, 0x80080
	s_addc_u32 s29, s27, 0
	v_accvgpr_read_b32 v2, a144
	s_nop 4
	v_cvt_pk_bf16_f32 v2, v2, 0
	global_store_short v9, v2, s[28:29]
	v_accvgpr_read_b32 v2, a145
	s_nop 4
	v_cvt_pk_bf16_f32 v2, v2, 0
	global_store_short v10, v2, s[28:29]
	v_accvgpr_read_b32 v2, a146
	s_nop 4
	v_cvt_pk_bf16_f32 v2, v2, 0
	global_store_short v11, v2, s[28:29]
	v_accvgpr_read_b32 v2, a147
	s_nop 4
	v_cvt_pk_bf16_f32 v2, v2, 0
	global_store_short v12, v2, s[28:29]
	s_add_u32 s28, s26, 0x800a0
	s_addc_u32 s29, s27, 0
	v_accvgpr_read_b32 v2, a148
	s_nop 4
	v_cvt_pk_bf16_f32 v2, v2, 0
	global_store_short v9, v2, s[28:29]
	v_accvgpr_read_b32 v2, a149
	s_nop 4
	v_cvt_pk_bf16_f32 v2, v2, 0
	global_store_short v10, v2, s[28:29]
	v_accvgpr_read_b32 v2, a150
	s_nop 4
	v_cvt_pk_bf16_f32 v2, v2, 0
	global_store_short v11, v2, s[28:29]
	v_accvgpr_read_b32 v2, a151
	s_nop 4
	v_cvt_pk_bf16_f32 v2, v2, 0
	global_store_short v12, v2, s[28:29]
	s_add_u32 s28, s26, 0x800c0
	s_addc_u32 s29, s27, 0
	v_accvgpr_read_b32 v2, a152
	s_nop 4
	v_cvt_pk_bf16_f32 v2, v2, 0
	global_store_short v9, v2, s[28:29]
	v_accvgpr_read_b32 v2, a153
	s_nop 4
	v_cvt_pk_bf16_f32 v2, v2, 0
	global_store_short v10, v2, s[28:29]
	v_accvgpr_read_b32 v2, a154
	s_nop 4
	v_cvt_pk_bf16_f32 v2, v2, 0
	global_store_short v11, v2, s[28:29]
	v_accvgpr_read_b32 v2, a155
	s_nop 4
	v_cvt_pk_bf16_f32 v2, v2, 0
	global_store_short v12, v2, s[28:29]
	s_add_u32 s28, s26, 0x800e0
	s_addc_u32 s29, s27, 0
	v_accvgpr_read_b32 v2, a156
	s_nop 4
	v_cvt_pk_bf16_f32 v2, v2, 0
	global_store_short v9, v2, s[28:29]
	v_accvgpr_read_b32 v2, a157
	s_nop 4
	v_cvt_pk_bf16_f32 v2, v2, 0
	global_store_short v10, v2, s[28:29]
	v_accvgpr_read_b32 v2, a158
	s_nop 4
	v_cvt_pk_bf16_f32 v2, v2, 0
	global_store_short v11, v2, s[28:29]
	v_accvgpr_read_b32 v2, a159
	s_nop 4
	v_cvt_pk_bf16_f32 v2, v2, 0
	global_store_short v12, v2, s[28:29]
	s_add_u32 s28, s26, 0xa0000
	s_addc_u32 s29, s27, 0
	v_accvgpr_read_b32 v2, a160
	s_nop 4
	v_cvt_pk_bf16_f32 v2, v2, 0
	global_store_short v9, v2, s[28:29]
	v_accvgpr_read_b32 v2, a161
	s_nop 4
	v_cvt_pk_bf16_f32 v2, v2, 0
	global_store_short v10, v2, s[28:29]
	v_accvgpr_read_b32 v2, a162
	s_nop 4
	v_cvt_pk_bf16_f32 v2, v2, 0
	global_store_short v11, v2, s[28:29]
	v_accvgpr_read_b32 v2, a163
	s_nop 4
	v_cvt_pk_bf16_f32 v2, v2, 0
	global_store_short v12, v2, s[28:29]
	s_add_u32 s28, s26, 0xa0020
	s_addc_u32 s29, s27, 0
	v_accvgpr_read_b32 v2, a164
	s_nop 4
	v_cvt_pk_bf16_f32 v2, v2, 0
	global_store_short v9, v2, s[28:29]
	v_accvgpr_read_b32 v2, a165
	s_nop 4
	v_cvt_pk_bf16_f32 v2, v2, 0
	global_store_short v10, v2, s[28:29]
	v_accvgpr_read_b32 v2, a166
	s_nop 4
	v_cvt_pk_bf16_f32 v2, v2, 0
	global_store_short v11, v2, s[28:29]
	v_accvgpr_read_b32 v2, a167
	s_nop 4
	v_cvt_pk_bf16_f32 v2, v2, 0
	global_store_short v12, v2, s[28:29]
	s_add_u32 s28, s26, 0xa0040
	s_addc_u32 s29, s27, 0
	v_accvgpr_read_b32 v2, a168
	s_nop 4
	v_cvt_pk_bf16_f32 v2, v2, 0
	global_store_short v9, v2, s[28:29]
	v_accvgpr_read_b32 v2, a169
	s_nop 4
	v_cvt_pk_bf16_f32 v2, v2, 0
	global_store_short v10, v2, s[28:29]
	v_accvgpr_read_b32 v2, a170
	s_nop 4
	v_cvt_pk_bf16_f32 v2, v2, 0
	global_store_short v11, v2, s[28:29]
	v_accvgpr_read_b32 v2, a171
	s_nop 4
	v_cvt_pk_bf16_f32 v2, v2, 0
	global_store_short v12, v2, s[28:29]
	s_add_u32 s28, s26, 0xa0060
	s_addc_u32 s29, s27, 0
	v_accvgpr_read_b32 v2, a172
	s_nop 4
	v_cvt_pk_bf16_f32 v2, v2, 0
	global_store_short v9, v2, s[28:29]
	v_accvgpr_read_b32 v2, a173
	s_nop 4
	v_cvt_pk_bf16_f32 v2, v2, 0
	global_store_short v10, v2, s[28:29]
	v_accvgpr_read_b32 v2, a174
	s_nop 4
	v_cvt_pk_bf16_f32 v2, v2, 0
	global_store_short v11, v2, s[28:29]
	v_accvgpr_read_b32 v2, a175
	s_nop 4
	v_cvt_pk_bf16_f32 v2, v2, 0
	global_store_short v12, v2, s[28:29]
	s_add_u32 s28, s26, 0xa0080
	s_addc_u32 s29, s27, 0
	v_accvgpr_read_b32 v2, a176
	s_nop 4
	v_cvt_pk_bf16_f32 v2, v2, 0
	global_store_short v9, v2, s[28:29]
	v_accvgpr_read_b32 v2, a177
	s_nop 4
	v_cvt_pk_bf16_f32 v2, v2, 0
	global_store_short v10, v2, s[28:29]
	v_accvgpr_read_b32 v2, a178
	s_nop 4
	v_cvt_pk_bf16_f32 v2, v2, 0
	global_store_short v11, v2, s[28:29]
	v_accvgpr_read_b32 v2, a179
	s_nop 4
	v_cvt_pk_bf16_f32 v2, v2, 0
	global_store_short v12, v2, s[28:29]
	s_add_u32 s28, s26, 0xa00a0
	s_addc_u32 s29, s27, 0
	v_accvgpr_read_b32 v2, a180
	s_nop 4
	v_cvt_pk_bf16_f32 v2, v2, 0
	global_store_short v9, v2, s[28:29]
	v_accvgpr_read_b32 v2, a181
	s_nop 4
	v_cvt_pk_bf16_f32 v2, v2, 0
	global_store_short v10, v2, s[28:29]
	v_accvgpr_read_b32 v2, a182
	s_nop 4
	v_cvt_pk_bf16_f32 v2, v2, 0
	global_store_short v11, v2, s[28:29]
	v_accvgpr_read_b32 v2, a183
	s_nop 4
	v_cvt_pk_bf16_f32 v2, v2, 0
	global_store_short v12, v2, s[28:29]
	s_add_u32 s28, s26, 0xa00c0
	s_addc_u32 s29, s27, 0
	v_accvgpr_read_b32 v2, a184
	s_nop 4
	v_cvt_pk_bf16_f32 v2, v2, 0
	global_store_short v9, v2, s[28:29]
	v_accvgpr_read_b32 v2, a185
	s_nop 4
	v_cvt_pk_bf16_f32 v2, v2, 0
	global_store_short v10, v2, s[28:29]
	v_accvgpr_read_b32 v2, a186
	s_nop 4
	v_cvt_pk_bf16_f32 v2, v2, 0
	global_store_short v11, v2, s[28:29]
	v_accvgpr_read_b32 v2, a187
	s_nop 4
	v_cvt_pk_bf16_f32 v2, v2, 0
	global_store_short v12, v2, s[28:29]
	s_add_u32 s28, s26, 0xa00e0
	s_addc_u32 s29, s27, 0
	v_accvgpr_read_b32 v2, a188
	s_nop 4
	v_cvt_pk_bf16_f32 v2, v2, 0
	global_store_short v9, v2, s[28:29]
	v_accvgpr_read_b32 v2, a189
	s_nop 4
	v_cvt_pk_bf16_f32 v2, v2, 0
	global_store_short v10, v2, s[28:29]
	v_accvgpr_read_b32 v2, a190
	s_nop 4
	v_cvt_pk_bf16_f32 v2, v2, 0
	global_store_short v11, v2, s[28:29]
	v_accvgpr_read_b32 v2, a191
	s_nop 4
	v_cvt_pk_bf16_f32 v2, v2, 0
	global_store_short v12, v2, s[28:29]
	s_add_u32 s28, s26, 0xc0000
	s_addc_u32 s29, s27, 0
	v_accvgpr_read_b32 v2, a192
	s_nop 4
	v_cvt_pk_bf16_f32 v2, v2, 0
	global_store_short v9, v2, s[28:29]
	v_accvgpr_read_b32 v2, a193
	s_nop 4
	v_cvt_pk_bf16_f32 v2, v2, 0
	global_store_short v10, v2, s[28:29]
	v_accvgpr_read_b32 v2, a194
	s_nop 4
	v_cvt_pk_bf16_f32 v2, v2, 0
	global_store_short v11, v2, s[28:29]
	v_accvgpr_read_b32 v2, a195
	s_nop 4
	v_cvt_pk_bf16_f32 v2, v2, 0
	global_store_short v12, v2, s[28:29]
	s_add_u32 s28, s26, 0xc0020
	s_addc_u32 s29, s27, 0
	v_accvgpr_read_b32 v2, a196
	s_nop 4
	v_cvt_pk_bf16_f32 v2, v2, 0
	global_store_short v9, v2, s[28:29]
	v_accvgpr_read_b32 v2, a197
	s_nop 4
	v_cvt_pk_bf16_f32 v2, v2, 0
	global_store_short v10, v2, s[28:29]
	v_accvgpr_read_b32 v2, a198
	s_nop 4
	v_cvt_pk_bf16_f32 v2, v2, 0
	global_store_short v11, v2, s[28:29]
	v_accvgpr_read_b32 v2, a199
	s_nop 4
	v_cvt_pk_bf16_f32 v2, v2, 0
	global_store_short v12, v2, s[28:29]
	s_add_u32 s28, s26, 0xc0040
	s_addc_u32 s29, s27, 0
	v_accvgpr_read_b32 v2, a200
	s_nop 4
	v_cvt_pk_bf16_f32 v2, v2, 0
	global_store_short v9, v2, s[28:29]
	v_accvgpr_read_b32 v2, a201
	s_nop 4
	v_cvt_pk_bf16_f32 v2, v2, 0
	global_store_short v10, v2, s[28:29]
	v_accvgpr_read_b32 v2, a202
	s_nop 4
	v_cvt_pk_bf16_f32 v2, v2, 0
	global_store_short v11, v2, s[28:29]
	v_accvgpr_read_b32 v2, a203
	s_nop 4
	v_cvt_pk_bf16_f32 v2, v2, 0
	global_store_short v12, v2, s[28:29]
	s_add_u32 s28, s26, 0xc0060
	s_addc_u32 s29, s27, 0
	v_accvgpr_read_b32 v2, a204
	s_nop 4
	v_cvt_pk_bf16_f32 v2, v2, 0
	global_store_short v9, v2, s[28:29]
	v_accvgpr_read_b32 v2, a205
	s_nop 4
	v_cvt_pk_bf16_f32 v2, v2, 0
	global_store_short v10, v2, s[28:29]
	v_accvgpr_read_b32 v2, a206
	s_nop 4
	v_cvt_pk_bf16_f32 v2, v2, 0
	global_store_short v11, v2, s[28:29]
	v_accvgpr_read_b32 v2, a207
	s_nop 4
	v_cvt_pk_bf16_f32 v2, v2, 0
	global_store_short v12, v2, s[28:29]
	s_add_u32 s28, s26, 0xc0080
	s_addc_u32 s29, s27, 0
	v_accvgpr_read_b32 v2, a208
	s_nop 4
	v_cvt_pk_bf16_f32 v2, v2, 0
	global_store_short v9, v2, s[28:29]
	v_accvgpr_read_b32 v2, a209
	s_nop 4
	v_cvt_pk_bf16_f32 v2, v2, 0
	global_store_short v10, v2, s[28:29]
	v_accvgpr_read_b32 v2, a210
	s_nop 4
	v_cvt_pk_bf16_f32 v2, v2, 0
	global_store_short v11, v2, s[28:29]
	v_accvgpr_read_b32 v2, a211
	s_nop 4
	v_cvt_pk_bf16_f32 v2, v2, 0
	global_store_short v12, v2, s[28:29]
	s_add_u32 s28, s26, 0xc00a0
	s_addc_u32 s29, s27, 0
	v_accvgpr_read_b32 v2, a212
	s_nop 4
	v_cvt_pk_bf16_f32 v2, v2, 0
	global_store_short v9, v2, s[28:29]
	v_accvgpr_read_b32 v2, a213
	s_nop 4
	v_cvt_pk_bf16_f32 v2, v2, 0
	global_store_short v10, v2, s[28:29]
	v_accvgpr_read_b32 v2, a214
	s_nop 4
	v_cvt_pk_bf16_f32 v2, v2, 0
	global_store_short v11, v2, s[28:29]
	v_accvgpr_read_b32 v2, a215
	s_nop 4
	v_cvt_pk_bf16_f32 v2, v2, 0
	global_store_short v12, v2, s[28:29]
	s_add_u32 s28, s26, 0xc00c0
	s_addc_u32 s29, s27, 0
	v_accvgpr_read_b32 v2, a216
	s_nop 4
	v_cvt_pk_bf16_f32 v2, v2, 0
	global_store_short v9, v2, s[28:29]
	v_accvgpr_read_b32 v2, a217
	s_nop 4
	v_cvt_pk_bf16_f32 v2, v2, 0
	global_store_short v10, v2, s[28:29]
	v_accvgpr_read_b32 v2, a218
	s_nop 4
	v_cvt_pk_bf16_f32 v2, v2, 0
	global_store_short v11, v2, s[28:29]
	v_accvgpr_read_b32 v2, a219
	s_nop 4
	v_cvt_pk_bf16_f32 v2, v2, 0
	global_store_short v12, v2, s[28:29]
	s_add_u32 s28, s26, 0xc00e0
	s_addc_u32 s29, s27, 0
	v_accvgpr_read_b32 v2, a220
	s_nop 4
	v_cvt_pk_bf16_f32 v2, v2, 0
	global_store_short v9, v2, s[28:29]
	v_accvgpr_read_b32 v2, a221
	s_nop 4
	v_cvt_pk_bf16_f32 v2, v2, 0
	global_store_short v10, v2, s[28:29]
	v_accvgpr_read_b32 v2, a222
	s_nop 4
	v_cvt_pk_bf16_f32 v2, v2, 0
	global_store_short v11, v2, s[28:29]
	v_accvgpr_read_b32 v2, a223
	s_nop 4
	v_cvt_pk_bf16_f32 v2, v2, 0
	global_store_short v12, v2, s[28:29]
	s_add_u32 s28, s26, 0xe0000
	s_addc_u32 s29, s27, 0
	v_accvgpr_read_b32 v2, a224
	s_nop 4
	v_cvt_pk_bf16_f32 v2, v2, 0
	global_store_short v9, v2, s[28:29]
	v_accvgpr_read_b32 v2, a225
	s_nop 4
	v_cvt_pk_bf16_f32 v2, v2, 0
	global_store_short v10, v2, s[28:29]
	v_accvgpr_read_b32 v2, a226
	s_nop 4
	v_cvt_pk_bf16_f32 v2, v2, 0
	global_store_short v11, v2, s[28:29]
	v_accvgpr_read_b32 v2, a227
	s_nop 4
	v_cvt_pk_bf16_f32 v2, v2, 0
	global_store_short v12, v2, s[28:29]
	s_add_u32 s28, s26, 0xe0020
	s_addc_u32 s29, s27, 0
	v_accvgpr_read_b32 v2, a228
	s_nop 4
	v_cvt_pk_bf16_f32 v2, v2, 0
	global_store_short v9, v2, s[28:29]
	v_accvgpr_read_b32 v2, a229
	s_nop 4
	v_cvt_pk_bf16_f32 v2, v2, 0
	global_store_short v10, v2, s[28:29]
	v_accvgpr_read_b32 v2, a230
	s_nop 4
	v_cvt_pk_bf16_f32 v2, v2, 0
	global_store_short v11, v2, s[28:29]
	v_accvgpr_read_b32 v2, a231
	s_nop 4
	v_cvt_pk_bf16_f32 v2, v2, 0
	global_store_short v12, v2, s[28:29]
	s_add_u32 s28, s26, 0xe0040
	s_addc_u32 s29, s27, 0
	v_accvgpr_read_b32 v2, a232
	s_nop 4
	v_cvt_pk_bf16_f32 v2, v2, 0
	global_store_short v9, v2, s[28:29]
	v_accvgpr_read_b32 v2, a233
	s_nop 4
	v_cvt_pk_bf16_f32 v2, v2, 0
	global_store_short v10, v2, s[28:29]
	v_accvgpr_read_b32 v2, a234
	s_nop 4
	v_cvt_pk_bf16_f32 v2, v2, 0
	global_store_short v11, v2, s[28:29]
	v_accvgpr_read_b32 v2, a235
	s_nop 4
	v_cvt_pk_bf16_f32 v2, v2, 0
	global_store_short v12, v2, s[28:29]
	s_add_u32 s28, s26, 0xe0060
	s_addc_u32 s29, s27, 0
	v_accvgpr_read_b32 v2, a236
	s_nop 4
	v_cvt_pk_bf16_f32 v2, v2, 0
	global_store_short v9, v2, s[28:29]
	v_accvgpr_read_b32 v2, a237
	s_nop 4
	v_cvt_pk_bf16_f32 v2, v2, 0
	global_store_short v10, v2, s[28:29]
	v_accvgpr_read_b32 v2, a238
	s_nop 4
	v_cvt_pk_bf16_f32 v2, v2, 0
	global_store_short v11, v2, s[28:29]
	v_accvgpr_read_b32 v2, a239
	s_nop 4
	v_cvt_pk_bf16_f32 v2, v2, 0
	global_store_short v12, v2, s[28:29]
	s_add_u32 s28, s26, 0xe0080
	s_addc_u32 s29, s27, 0
	v_accvgpr_read_b32 v2, a240
	s_nop 4
	v_cvt_pk_bf16_f32 v2, v2, 0
	global_store_short v9, v2, s[28:29]
	v_accvgpr_read_b32 v2, a241
	s_nop 4
	v_cvt_pk_bf16_f32 v2, v2, 0
	global_store_short v10, v2, s[28:29]
	v_accvgpr_read_b32 v2, a242
	s_nop 4
	v_cvt_pk_bf16_f32 v2, v2, 0
	global_store_short v11, v2, s[28:29]
	v_accvgpr_read_b32 v2, a243
	s_nop 4
	v_cvt_pk_bf16_f32 v2, v2, 0
	global_store_short v12, v2, s[28:29]
	s_add_u32 s28, s26, 0xe00a0
	s_addc_u32 s29, s27, 0
	v_accvgpr_read_b32 v2, a244
	s_nop 4
	v_cvt_pk_bf16_f32 v2, v2, 0
	global_store_short v9, v2, s[28:29]
	v_accvgpr_read_b32 v2, a245
	s_nop 4
	v_cvt_pk_bf16_f32 v2, v2, 0
	global_store_short v10, v2, s[28:29]
	v_accvgpr_read_b32 v2, a246
	s_nop 4
	v_cvt_pk_bf16_f32 v2, v2, 0
	global_store_short v11, v2, s[28:29]
	v_accvgpr_read_b32 v2, a247
	s_nop 4
	v_cvt_pk_bf16_f32 v2, v2, 0
	global_store_short v12, v2, s[28:29]
	s_add_u32 s28, s26, 0xe00c0
	s_addc_u32 s29, s27, 0
	v_accvgpr_read_b32 v2, a248
	s_nop 4
	v_cvt_pk_bf16_f32 v2, v2, 0
	global_store_short v9, v2, s[28:29]
	v_accvgpr_read_b32 v2, a249
	s_nop 4
	v_cvt_pk_bf16_f32 v2, v2, 0
	global_store_short v10, v2, s[28:29]
	v_accvgpr_read_b32 v2, a250
	s_nop 4
	v_cvt_pk_bf16_f32 v2, v2, 0
	global_store_short v11, v2, s[28:29]
	v_accvgpr_read_b32 v2, a251
	s_nop 4
	v_cvt_pk_bf16_f32 v2, v2, 0
	global_store_short v12, v2, s[28:29]
	s_add_u32 s28, s26, 0xe00e0
	s_addc_u32 s29, s27, 0
	v_accvgpr_read_b32 v2, a252
	s_nop 4
	v_cvt_pk_bf16_f32 v2, v2, 0
	global_store_short v9, v2, s[28:29]
	v_accvgpr_read_b32 v2, a253
	s_nop 4
	v_cvt_pk_bf16_f32 v2, v2, 0
	global_store_short v10, v2, s[28:29]
	v_accvgpr_read_b32 v2, a254
	s_nop 4
	v_cvt_pk_bf16_f32 v2, v2, 0
	global_store_short v11, v2, s[28:29]
	v_accvgpr_read_b32 v2, a255
	s_nop 4
	v_cvt_pk_bf16_f32 v2, v2, 0
	global_store_short v12, v2, s[28:29]
	s_waitcnt vmcnt(0)
	s_endpgm

	.section	.rodata,"a",@progbits
	.p2align	6, 0x0
.amdhsa_kernel gfxjit_mxfp6_gemm_256x256x128_w2x2_16x16x128_s3_4096x4096x65536_preswz
	.amdhsa_group_segment_fixed_size 153600
	.amdhsa_private_segment_fixed_size 0
	.amdhsa_kernarg_size 40
	.amdhsa_accum_offset 252
	.amdhsa_tg_split 0
	.amdhsa_next_free_vgpr 512
	.amdhsa_reserve_vcc 0
	.amdhsa_next_free_sgpr 64
	.amdhsa_float_round_mode_32 0
	.amdhsa_float_round_mode_16_64 0
	.amdhsa_float_denorm_mode_32 0
	.amdhsa_float_denorm_mode_16_64 3
	.amdhsa_dx10_clamp 1
	.amdhsa_ieee_mode 1
	.amdhsa_fp16_overflow 0
	.amdhsa_enable_private_segment 0
	.amdhsa_system_sgpr_workgroup_id_x 1
	.amdhsa_system_sgpr_workgroup_id_y 0
	.amdhsa_system_sgpr_workgroup_id_z 0
	.amdhsa_system_sgpr_workgroup_info 0
	.amdhsa_system_vgpr_workitem_id 0
	.amdhsa_exception_fp_ieee_invalid_op 0
	.amdhsa_exception_fp_denorm_src 0
	.amdhsa_exception_fp_ieee_div_zero 0
	.amdhsa_exception_fp_ieee_overflow 0
	.amdhsa_exception_fp_ieee_underflow 0
	.amdhsa_exception_fp_ieee_inexact 0
	.amdhsa_exception_int_div_zero 0
	.amdhsa_user_sgpr_dispatch_ptr 0
	.amdhsa_user_sgpr_queue_ptr 0
	.amdhsa_user_sgpr_kernarg_segment_ptr 1
	.amdhsa_user_sgpr_dispatch_id 0
	.amdhsa_user_sgpr_private_segment_size 0
	.amdhsa_uses_dynamic_stack 0
.end_amdhsa_kernel

	.amdgpu_metadata
---
custom.config:
  Source:
    Origin: gfxjit
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
    DataTypeA: F6
    DataTypeB: F6
    DestDataType: b
    ComputeDataType: s
    HighPrecisionAccumulate: True
    TransposeA: True
    TransposeB: False
    UseBeta: True
    Batched: True
    UseBias: 0
    Activation: False
    UseScaleAlphaVec: 0
    MXBlockA: 32
    MXBlockB: 32
  CustomKernel:
    args: [ { type: address, semantic: AddressD },
            { type: address, semantic: AddressB },
            { type: address, semantic: AddressA },
            { type: address, semantic: AddressMXScaleB },
            { type: address, semantic: AddressMXScaleA } ]
    macrotile: [256, 256, 128]
    threads: [256, 1, 1]
    grid: [TilesXY, One, One]
  MatrixInstruction: [16, 16, 128, 1]
  StaggerU: 0
  EnableMatrixInstruction: True
  MIWaveTile: [8, 8]
  WavefrontSize: 64
amdhsa.kernels:
    - .agpr_count: 256
      .args:
          - .address_space: global
            .name: c
            .offset: 0
            .size: 8
            .value_kind: global_buffer
          - .address_space: global
            .name: a
            .offset: 8
            .size: 8
            .value_kind: global_buffer
          - .address_space: global
            .name: b
            .offset: 16
            .size: 8
            .value_kind: global_buffer
          - .address_space: global
            .name: sa
            .offset: 24
            .size: 8
            .value_kind: global_buffer
          - .address_space: global
            .name: sb
            .offset: 32
            .size: 8
            .value_kind: global_buffer
      .group_segment_fixed_size: 153600
      .kernarg_segment_align: 8
      .kernarg_segment_size: 40
      .max_flat_workgroup_size: 256
      .name: gfxjit_mxfp6_gemm_256x256x128_w2x2_16x16x128_s3_4096x4096x65536_preswz
      .private_segment_fixed_size: 0
      .sgpr_count: 51
      .sgpr_spill_count: 0
      .symbol: gfxjit_mxfp6_gemm_256x256x128_w2x2_16x16x128_s3_4096x4096x65536_preswz.kd
      .uniform_work_group_size: 1
      .uses_dynamic_stack: false
      .vgpr_count: 252
      .vgpr_spill_count: 0
      .wavefront_size: 64
amdhsa.target: 'amdgcn-amd-amdhsa--gfx950'
amdhsa.version:
    - 1
    - 2
...
	.end_amdgpu_metadata

