; Function Attrs: nounwind
define void @fusion(i8* nocapture readonly dereferenceable(16777216) %tanh.1.raw, i8* nocapture readonly dereferenceable(16777216) %fusion.2.raw, i8* nocapture readonly dereferenceable(8192) %arg2.raw, i8* nocapture readonly dereferenceable(8192) %arg1.raw, i8* nocapture dereferenceable(16777216) %fusion.raw, i8* noalias nocapture readnone dereferenceable(16777216) %temp_buffer) local_unnamed_addr #0 {
entry:
  %fusion.raw10 = addrspacecast i8* %fusion.raw to i8 addrspace(1)*
  %arg1.raw8 = addrspacecast i8* %arg1.raw to i8 addrspace(1)*
  %arg2.raw6 = addrspacecast i8* %arg2.raw to i8 addrspace(1)*
  %fusion.2.raw4 = addrspacecast i8* %fusion.2.raw to i8 addrspace(1)*
  %tanh.1.raw2 = addrspacecast i8* %tanh.1.raw to i8 addrspace(1)*
  %0 = tail call i32 @llvm.nvvm.read.ptx.sreg.ctaid.x(), !range !20
  %1 = tail call i32 @llvm.nvvm.read.ptx.sreg.tid.x(), !range !21
  %2 = shl nuw nsw i32 %0, 6
  %linear_index1 = or i32 %2, %1
  %linear_index = zext i32 %linear_index1 to i64
  %arg1.typed = bitcast i8 addrspace(1)* %arg1.raw8 to [2048 x float] addrspace(1)*
  %arg2.typed = bitcast i8 addrspace(1)* %arg2.raw6 to [2048 x float] addrspace(1)*
  %3 = lshr i64 %linear_index, 11
  %4 = getelementptr inbounds [2048 x float], [2048 x float] addrspace(1)* %arg1.typed, i64 0, i64 %3
  %5 = load float, float addrspace(1)* %4, align 4, !invariant.load !22, !noalias !23
  %6 = fcmp fast oeq float %5, 1.000000e+00
  %7 = select i1 %6, float 1.000000e+00, float 0.000000e+00
  %8 = bitcast i8 addrspace(1)* %fusion.2.raw4 to float addrspace(1)*
  %9 = getelementptr inbounds float, float addrspace(1)* %8, i64 %linear_index
  %10 = load float, float addrspace(1)* %9, align 4, !invariant.load !22, !alias.scope !34, !noalias !35
  %11 = getelementptr inbounds [2048 x float], [2048 x float] addrspace(1)* %arg2.typed, i64 0, i64 %3
  %12 = load float, float addrspace(1)* %11, align 4, !invariant.load !22, !noalias !23
  %13 = fcmp fast oeq float %12, 0.000000e+00
  %14 = or i1 %6, %13
  %15 = select i1 %14, float 0.000000e+00, float 1.000000e+00
  %16 = fadd fast float %15, %7
  %17 = bitcast i8 addrspace(1)* %tanh.1.raw2 to float addrspace(1)*
  %18 = getelementptr inbounds float, float addrspace(1)* %17, i64 %linear_index
  %19 = load float, float addrspace(1)* %18, align 4, !invariant.load !22, !alias.scope !29, !noalias !30
  %20 = fmul fast float %19, %19
  %21 = fsub fast float 1.000000e+00, %20
  %22 = fmul fast float %21, %10
  %23 = fmul fast float %22, %16
  %24 = bitcast i8 addrspace(1)* %fusion.raw10 to float addrspace(1)*
  %25 = getelementptr inbounds float, float addrspace(1)* %24, i64 %linear_index
  store float %23, float addrspace(1)* %25, align 4, !alias.scope !38, !noalias !39
  ret void
}