; Function Attrs: nounwind
define void @tanh_1(i8* nocapture readonly dereferenceable(16777216) %arg4.raw, i8* nocapture dereferenceable(16777216) %tanh.1.raw, i8* noalias nocapture readnone dereferenceable(16777216) %temp_buffer) local_unnamed_addr #0 {
entry:
  %tanh.1.raw4 = addrspacecast i8* %tanh.1.raw to i8 addrspace(1)*
  %arg4.raw2 = addrspacecast i8* %arg4.raw to i8 addrspace(1)*
  %0 = tail call i32 @llvm.nvvm.read.ptx.sreg.ctaid.x(), !range !20
  %1 = tail call i32 @llvm.nvvm.read.ptx.sreg.tid.x(), !range !21
  %2 = shl nuw nsw i32 %0, 6
  %linear_index1 = or i32 %2, %1
  %linear_index = zext i32 %linear_index1 to i64
  %3 = bitcast i8 addrspace(1)* %arg4.raw2 to float addrspace(1)*
  %4 = getelementptr inbounds float, float addrspace(1)* %3, i64 %linear_index
  %5 = load float, float addrspace(1)* %4, align 4, !invariant.load !22, !noalias !23
  %6 = tail call float @llvm.fabs.f32(float %5) #0
  %cmp.i = fcmp ult float %6, 0x3FE19999A0000000
  br i1 %cmp.i, label %if.else.i, label %if.then.i

if.then.i:                                        ; preds = %entry
  %add.i = fadd float %6, %6
  %mul.i.i.i = fmul float %add.i, 0x3FF7154760000000
  %7 = tail call float @llvm.trunc.f32(float %mul.i.i.i) #0
  %8 = tail call float @llvm.fma.f32(float %7, float 0xBFE62E4000000000, float %add.i) #0
  %9 = tail call float @llvm.fma.f32(float %7, float 0xBEB7F7D1C0000000, float %8) #0
  %mul8.i.i.i = fmul float %9, 0x3FF7154760000000
  %10 = tail call float @llvm.nvvm.ex2.approx.ftz.f(float %mul8.i.i.i) #0
  %11 = tail call float @llvm.nvvm.ex2.approx.f(float %7) #0
  %12 = tail call float @llvm.fma.f32(float %10, float %11, float 1.000000e+00) #0
  %13 = tail call float asm "rcp.approx.ftz.f32 $0,$1;", "=f,f"(float %12) #0
  %14 = tail call float @llvm.fma.f32(float %13, float -2.000000e+00, float 1.000000e+00) #0
  %cmp8.i = fcmp oge float %6, 8.800000e+01
  %15 = bitcast float %14 to i32
  %16 = select i1 %cmp8.i, i32 1065353216, i32 %15
  %17 = bitcast float %5 to i32
  %and.i.i = and i32 %17, -2147483648
  %or.i.i = or i32 %16, %and.i.i
  %18 = bitcast i32 %or.i.i to float
  br label %__nv_tanhf.exit

if.else.i:                                        ; preds = %entry
  %mul.i = fmul float %5, %5
  %19 = tail call float @llvm.fma.f32(float %mul.i, float 0x3F90D50360000000, float 0xBFAAF7CCC0000000) #0
  %20 = tail call float @llvm.fma.f32(float %19, float %mul.i, float 0x3FC10CEF60000000) #0
  %21 = tail call float @llvm.fma.f32(float %20, float %mul.i, float 0xBFD5554520000000) #0
  %mul26.i = fmul float %mul.i, %21
  %22 = tail call float @llvm.fma.f32(float %mul26.i, float %5, float %5) #0
  %cmp32.i = fcmp oeq float %5, 0.000000e+00
  %add36.i = fadd float %5, %5
  %s.1.i = select i1 %cmp32.i, float %add36.i, float %22
  br label %__nv_tanhf.exit

__nv_tanhf.exit:                                  ; preds = %if.then.i, %if.else.i
  %s.2.i = phi float [ %18, %if.then.i ], [ %s.1.i, %if.else.i ]
  %23 = bitcast i8 addrspace(1)* %tanh.1.raw4 to float addrspace(1)*
  %24 = getelementptr inbounds float, float addrspace(1)* %23, i64 %linear_index
  store float %s.2.i, float addrspace(1)* %24, align 4, !alias.scope !29, !noalias !30
  ret void
}