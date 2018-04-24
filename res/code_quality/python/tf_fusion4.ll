; Function Attrs: nounwind
define void @fusion_4(i8* nocapture readonly dereferenceable(8192) %arg2.raw, i8* nocapture readonly dereferenceable(8192) %arg1.raw, i8* nocapture dereferenceable(16777216) %fusion.4.raw, i8* noalias nocapture readonly dereferenceable(16777216) %temp_buffer) local_unnamed_addr #0 {
entry:
  %temp_buffer8 = addrspacecast i8* %temp_buffer to i8 addrspace(1)*
  %fusion.4.raw6 = addrspacecast i8* %fusion.4.raw to i8 addrspace(1)*
  %arg1.raw4 = addrspacecast i8* %arg1.raw to i8 addrspace(1)*
  %arg2.raw2 = addrspacecast i8* %arg2.raw to i8 addrspace(1)*
  %0 = tail call i32 @llvm.nvvm.read.ptx.sreg.ctaid.x(), !range !20
  %1 = tail call i32 @llvm.nvvm.read.ptx.sreg.tid.x(), !range !21
  %2 = shl nuw nsw i32 %0, 6
  %linear_index1 = or i32 %2, %1
  %linear_index = zext i32 %linear_index1 to i64
  %arg1.typed = bitcast i8 addrspace(1)* %arg1.raw4 to [2048 x float] addrspace(1)*
  %arg2.typed = bitcast i8 addrspace(1)* %arg2.raw2 to [2048 x float] addrspace(1)*
  %3 = lshr i64 %linear_index, 11
  %4 = getelementptr inbounds [2048 x float], [2048 x float] addrspace(1)* %arg2.typed, i64 0, i64 %3
  %5 = load float, float addrspace(1)* %4, align 4, !invariant.load !22, !noalias !23
  %6 = fcmp fast oeq float %5, 0.000000e+00
  %7 = getelementptr inbounds [2048 x float], [2048 x float] addrspace(1)* %arg1.typed, i64 0, i64 %3
  %8 = load float, float addrspace(1)* %7, align 4, !invariant.load !22, !noalias !23
  %9 = fcmp fast oeq float %8, 1.000000e+00
  %10 = select i1 %9, float 0.000000e+00, float 1.000000e+00
  %11 = select i1 %6, float %10, float 0.000000e+00
  %12 = select i1 %6, float 0.000000e+00, float %10
  %13 = bitcast i8 addrspace(1)* %temp_buffer8 to float addrspace(1)*
  %14 = getelementptr inbounds float, float addrspace(1)* %13, i64 %linear_index
  %15 = load float, float addrspace(1)* %14, align 4, !invariant.load !22, !alias.scope !23, !noalias !26
  %16 = fmul fast float %12, %15
  %17 = fadd fast float %16, %11
  %18 = bitcast i8 addrspace(1)* %fusion.4.raw6 to float addrspace(1)*
  %19 = getelementptr inbounds float, float addrspace(1)* %18, i64 %linear_index
  store float %17, float addrspace(1)* %19, align 4, !alias.scope !29, !noalias !30
  ret void
}