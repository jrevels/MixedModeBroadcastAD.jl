; Function Attrs: nounwind
define void @fusion_3(i8* nocapture readonly dereferenceable(16777216) %arg0.raw, i8* nocapture readonly dereferenceable(8192) %arg2.raw, i8* nocapture readonly dereferenceable(8192) %arg1.raw, i8* nocapture dereferenceable(16777216) %fusion.3.raw, i8* noalias nocapture readonly dereferenceable(16777216) %temp_buffer) local_unnamed_addr #0 {
entry:
  %temp_buffer10 = addrspacecast i8* %temp_buffer to i8 addrspace(1)*
  %fusion.3.raw8 = addrspacecast i8* %fusion.3.raw to i8 addrspace(1)*
  %arg1.raw6 = addrspacecast i8* %arg1.raw to i8 addrspace(1)*
  %arg2.raw4 = addrspacecast i8* %arg2.raw to i8 addrspace(1)*
  %arg0.raw2 = addrspacecast i8* %arg0.raw to i8 addrspace(1)*
  %0 = tail call i32 @llvm.nvvm.read.ptx.sreg.ctaid.x(), !range !20
  %1 = tail call i32 @llvm.nvvm.read.ptx.sreg.tid.x(), !range !21
  %2 = shl nuw nsw i32 %0, 6
  %linear_index1 = or i32 %2, %1
  %linear_index = zext i32 %linear_index1 to i64
  %arg1.typed = bitcast i8 addrspace(1)* %arg1.raw6 to [2048 x float] addrspace(1)*
  %arg2.typed = bitcast i8 addrspace(1)* %arg2.raw4 to [2048 x float] addrspace(1)*
  %3 = lshr i64 %linear_index, 11
  %4 = getelementptr inbounds [2048 x float], [2048 x float] addrspace(1)* %arg2.typed, i64 0, i64 %3
  %5 = load float, float addrspace(1)* %4, align 4, !invariant.load !22, !noalias !23
  %6 = fcmp fast oeq float %5, 0.000000e+00
  %7 = getelementptr inbounds [2048 x float], [2048 x float] addrspace(1)* %arg1.typed, i64 0, i64 %3
  %8 = load float, float addrspace(1)* %7, align 4, !invariant.load !22, !noalias !23
  %9 = fcmp fast oeq float %8, 1.000000e+00
  %10 = or i1 %6, %9
  %11 = select i1 %10, float 0.000000e+00, float 1.000000e+00
  %12 = bitcast i8 addrspace(1)* %arg0.raw2 to float addrspace(1)*
  %13 = getelementptr inbounds float, float addrspace(1)* %12, i64 %linear_index
  %14 = load float, float addrspace(1)* %13, align 4, !invariant.load !22, !noalias !23
  %15 = bitcast i8 addrspace(1)* %temp_buffer10 to float addrspace(1)*
  %16 = getelementptr inbounds float, float addrspace(1)* %15, i64 %linear_index
  %17 = load float, float addrspace(1)* %16, align 4, !invariant.load !22, !alias.scope !23, !noalias !26
  %18 = fsub fast float 1.000000e+00, %17
  %19 = fmul fast float %17, %14
  %20 = fmul fast float %19, %11
  %21 = fmul fast float %20, %18
  %22 = bitcast i8 addrspace(1)* %fusion.3.raw8 to float addrspace(1)*
  %23 = getelementptr inbounds float, float addrspace(1)* %22, i64 %linear_index
  store float %21, float addrspace(1)* %23, align 4, !alias.scope !34, !noalias !35
  ret void
}