; Function Attrs: nounwind
define void @fusion_5(i8* nocapture readonly dereferenceable(16777216) %arg3.raw, i8* noalias nocapture dereferenceable(16777216) %temp_buffer) local_unnamed_addr #0 {
entry:
  %temp_buffer4 = addrspacecast i8* %temp_buffer to i8 addrspace(1)*
  %arg3.raw2 = addrspacecast i8* %arg3.raw to i8 addrspace(1)*
  %0 = tail call i32 @llvm.nvvm.read.ptx.sreg.ctaid.x(), !range !20
  %1 = tail call i32 @llvm.nvvm.read.ptx.sreg.tid.x(), !range !21
  %2 = shl nuw nsw i32 %0, 6
  %linear_index1 = or i32 %2, %1
  %linear_index = zext i32 %linear_index1 to i64
  %3 = bitcast i8 addrspace(1)* %arg3.raw2 to float addrspace(1)*
  %4 = getelementptr inbounds float, float addrspace(1)* %3, i64 %linear_index
  %5 = load float, float addrspace(1)* %4, align 4, !invariant.load !22, !noalias !23
  %6 = fmul fast float %5, 5.000000e-01
  %7 = tail call float @llvm.fabs.f32(float %6) #0
  %cmp.i = fcmp ult float %7, 0x3FE19999A0000000
  br i1 %cmp.i, label %if.else.i, label %if.then.i

if.then.i:                                        ; preds = %entry
  %add.i = fadd float %7, %7
  %mul.i.i.i = fmul float %add.i, 0x3FF7154760000000
  %8 = tail call float @llvm.trunc.f32(float %mul.i.i.i) #0
  %9 = tail call float @llvm.fma.f32(float %8, float 0xBFE62E4000000000, float %add.i) #0
  %10 = tail call float @llvm.fma.f32(float %8, float 0xBEB7F7D1C0000000, float %9) #0
  %mul8.i.i.i = fmul float %10, 0x3FF7154760000000
  %11 = tail call float @llvm.nvvm.ex2.approx.ftz.f(float %mul8.i.i.i) #0
  %12 = tail call float @llvm.nvvm.ex2.approx.f(float %8) #0
  %13 = tail call float @llvm.fma.f32(float %11, float %12, float 1.000000e+00) #0
  %14 = tail call float asm "rcp.approx.ftz.f32 $0,$1;", "=f,f"(float %13) #0
  %15 = tail call float @llvm.fma.f32(float %14, float -2.000000e+00, float 1.000000e+00) #0
  %cmp8.i = fcmp oge float %7, 8.800000e+01
  %16 = bitcast float %15 to i32
  %17 = select i1 %cmp8.i, i32 1065353216, i32 %16
  %18 = bitcast float %6 to i32
  %and.i.i = and i32 %18, -2147483648
  %or.i.i = or i32 %17, %and.i.i
  %19 = bitcast i32 %or.i.i to float
  br label %__nv_tanhf.exit

if.else.i:                                        ; preds = %entry
  %mul.i = fmul float %6, %6
  %20 = tail call float @llvm.fma.f32(float %mul.i, float 0x3F90D50360000000, float 0xBFAAF7CCC0000000) #0
  %21 = tail call float @llvm.fma.f32(float %20, float %mul.i, float 0x3FC10CEF60000000) #0
  %22 = tail call float @llvm.fma.f32(float %21, float %mul.i, float 0xBFD5554520000000) #0
  %mul26.i = fmul float %mul.i, %22
  %23 = tail call float @llvm.fma.f32(float %mul26.i, float %6, float %6) #0
  %cmp32.i = fcmp oeq float %6, 0.000000e+00
  %add36.i = fadd float %6, %6
  %s.1.i = select i1 %cmp32.i, float %add36.i, float %23
  br label %__nv_tanhf.exit

__nv_tanhf.exit:                                  ; preds = %if.then.i, %if.else.i
  %s.2.i = phi float [ %19, %if.then.i ], [ %s.1.i, %if.else.i ]
  %24 = fmul fast float %s.2.i, 5.000000e-01
  %25 = fadd fast float %24, 5.000000e-01
  %26 = bitcast i8 addrspace(1)* %temp_buffer4 to float addrspace(1)*
  %27 = getelementptr inbounds float, float addrspace(1)* %26, i64 %linear_index
  store float %25, float addrspace(1)* %27, align 4, !alias.scope !23, !noalias !26
  ret void
}