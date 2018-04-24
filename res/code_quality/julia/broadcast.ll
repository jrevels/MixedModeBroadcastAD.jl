; ModuleID = 'Kernel'
source_filename = "Kernel"
target datalayout = "e-p:64:64:64-i1:8:8-i8:8:8-i16:16:16-i32:32:32-i64:64:64-f32:32:32-f64:64:64-v16:16:16-v32:32:32-v64:64:64-v128:128:128-n16:32:64"
target triple = "nvptx64-nvidia-cuda"

; Function Attrs: nounwind readnone
declare i32 @llvm.nvvm.read.ptx.sreg.ctaid.x() #0

; Function Attrs: nounwind readnone
declare i32 @llvm.nvvm.read.ptx.sreg.ntid.x() #0

; Function Attrs: nounwind readnone
declare i32 @llvm.nvvm.read.ptx.sreg.tid.x() #0

; Function Attrs: noreturn nounwind
declare void @llvm.trap() #1

declare float @__nv_tanhf(float) local_unnamed_addr

define void @ptxcall_Kernel_3({ [2 x i64], { i64 } }, [1 x { [2 x i64], { i64 } }], [1 x [2 x i8]], [1 x [2 x i64]], [2 x { i64 }]) local_unnamed_addr {
entry:
  %.elt19 = extractvalue [2 x { i64 }] %4, 0
  %5 = extractvalue { i64 } %.elt19, 0
  %.elt21 = extractvalue [2 x { i64 }] %4, 1
  %6 = extractvalue { i64 } %.elt21, 0
  %7 = call i32 @llvm.nvvm.read.ptx.sreg.ctaid.x(), !dbg !5, !range !25
  %8 = zext i32 %7 to i64, !dbg !26
  %9 = call i32 @llvm.nvvm.read.ptx.sreg.ntid.x(), !dbg !31, !range !41
  %10 = zext i32 %9 to i64, !dbg !42
  %11 = mul nuw nsw i64 %10, %8, !dbg !46
  %12 = call i32 @llvm.nvvm.read.ptx.sreg.tid.x(), !dbg !49, !range !59
  %13 = zext i32 %12 to i64, !dbg !60
  %14 = add nuw nsw i64 %13, 1, !dbg !64
  %15 = add nuw nsw i64 %14, %11, !dbg !66
  %16 = mul i64 %5, %6, !dbg !68
  %17 = icmp slt i64 %16, %15, !dbg !74
  br i1 %17, label %julia_Kernel_34868.exit, label %L210.i, !dbg !74

L210.i:                                           ; preds = %entry
  %18 = icmp eq i64 %5, 0, !dbg !75
  br i1 %18, label %fail.i, label %pass.i, !dbg !75

fail.i:                                           ; preds = %L210.i
  call void @llvm.trap(), !dbg !23
  unreachable, !dbg !75

pass.i:                                           ; preds = %L210.i
  %.elt = extractvalue { [2 x i64], { i64 } } %0, 0
  %.elt.elt = extractvalue [2 x i64] %.elt, 0
  %.elt2 = extractvalue { [2 x i64], { i64 } } %0, 1
  %19 = extractvalue { i64 } %.elt2, 0
  %20 = extractvalue [1 x { [2 x i64], { i64 } }] %1, 0
  %.elt5 = extractvalue { [2 x i64], { i64 } } %20, 0
  %.elt5.elt = extractvalue [2 x i64] %.elt5, 0
  %.elt7 = extractvalue { [2 x i64], { i64 } } %20, 1
  %21 = extractvalue { i64 } %.elt7, 0
  %22 = extractvalue [1 x [2 x i8]] %2, 0
  %.elt11 = extractvalue [2 x i8] %22, 0
  %.elt13 = extractvalue [2 x i8] %22, 1
  %23 = extractvalue [1 x [2 x i64]] %3, 0
  %.elt15 = extractvalue [2 x i64] %23, 0
  %.elt17 = extractvalue [2 x i64] %23, 1
  %24 = add nuw nsw i64 %11, %13, !dbg !95
  %25 = sdiv i64 %24, %5, !dbg !75
  %26 = mul i64 %25, %5, !dbg !97
  %27 = sub i64 %24, %26, !dbg !100
  %28 = add i64 %27, 1, !dbg !102
  %29 = add i64 %25, 1, !dbg !104
  %30 = and i8 %.elt11, 1, !dbg !110
  %31 = icmp eq i8 %30, 0, !dbg !110
  %32 = select i1 %31, i64 %.elt15, i64 %28, !dbg !110
  %33 = and i8 %.elt13, 1, !dbg !119
  %34 = icmp eq i8 %33, 0, !dbg !119
  %35 = select i1 %34, i64 %.elt17, i64 %29, !dbg !119
  %36 = icmp slt i64 %.elt5.elt, 0, !dbg !123
  %37 = select i1 %36, i64 0, i64 %.elt5.elt, !dbg !123
  %38 = add i64 %35, -1, !dbg !149
  %39 = mul i64 %38, %37, !dbg !159
  %40 = add i64 %39, -1, !dbg !161
  %41 = add i64 %40, %32, !dbg !163
  %42 = inttoptr i64 %21 to float*, !dbg !165
  %43 = getelementptr float, float* %42, i64 %41, !dbg !165
  %44 = addrspacecast float* %43 to float addrspace(1)*, !dbg !165
  %45 = load float, float addrspace(1)* %44, align 4, !dbg !165
  %46 = call float @__nv_tanhf(float %45), !dbg !175
  %47 = icmp slt i64 %.elt.elt, 0, !dbg !185
  %48 = select i1 %47, i64 0, i64 %.elt.elt, !dbg !185
  %49 = mul i64 %25, %48, !dbg !203
  %50 = add i64 %27, %49, !dbg !211
  %51 = inttoptr i64 %19 to float*, !dbg !213
  %52 = getelementptr float, float* %51, i64 %50, !dbg !213
  %53 = addrspacecast float* %52 to float addrspace(1)*, !dbg !213
  store float %46, float addrspace(1)* %53, align 4, !dbg !213
  br label %julia_Kernel_34868.exit, !dbg !221

julia_Kernel_34868.exit:                          ; preds = %entry, %pass.i
  ret void
}

attributes #0 = { nounwind readnone }
attributes #1 = { noreturn nounwind }

!llvm.module.flags = !{!0}
!llvm.dbg.cu = !{!1}
!nvvm.annotations = !{!4}

!0 = !{i32 1, !"Debug Info Version", i32 3}
!1 = distinct !DICompileUnit(language: DW_LANG_C89, file: !2, producer: "julia", isOptimized: true, runtimeVersion: 0, emissionKind: FullDebug, enums: !3)
!2 = !DIFile(filename: "/home/maleadt/Julia/CUDAnative/src/execution.jl", directory: ".")
!3 = !{}
!4 = !{void ({ [2 x i64], { i64 } }, [1 x { [2 x i64], { i64 } }], [1 x [2 x i8]], [1 x [2 x i64]], [2 x { i64 }])* @ptxcall_Kernel_3, !"kernel", i32 1}
!5 = !DILocation(line: 42, scope: !6, inlinedAt: !9)
!6 = distinct !DISubprogram(name: "macro expansion;", linkageName: "macro expansion", scope: !7, file: !7, type: !8, isLocal: false, isDefinition: true, isOptimized: true, unit: !1, variables: !3)
!7 = !DIFile(filename: "/home/maleadt/Julia/LLVM/src/interop/base.jl", directory: ".")
!8 = !DISubroutineType(types: !3)
!9 = !DILocation(line: 8, scope: !10, inlinedAt: !12)
!10 = distinct !DISubprogram(name: "@generated body;", linkageName: "@generated body", scope: !11, file: !11, type: !8, isLocal: false, isDefinition: true, isOptimized: true, unit: !1, variables: !3)
!11 = !DIFile(filename: "/home/maleadt/Julia/CUDAnative/src/device/intrinsics/indexing.jl", directory: ".")
!12 = !DILocation(line: 8, scope: !13, inlinedAt: !14)
!13 = distinct !DISubprogram(name: "_index;", linkageName: "_index", scope: !11, file: !11, line: 8, type: !8, isLocal: false, isDefinition: true, scopeLine: 8, isOptimized: true, unit: !1, variables: !3)
!14 = !DILocation(line: 55, scope: !15, inlinedAt: !16)
!15 = distinct !DISubprogram(name: "blockIdx_x;", linkageName: "blockIdx_x", scope: !11, file: !11, line: 55, type: !8, isLocal: false, isDefinition: true, scopeLine: 55, isOptimized: true, unit: !1, variables: !3)
!16 = !DILocation(line: 75, scope: !17, inlinedAt: !18)
!17 = distinct !DISubprogram(name: "blockIdx;", linkageName: "blockIdx", scope: !11, file: !11, line: 75, type: !8, isLocal: false, isDefinition: true, scopeLine: 75, isOptimized: true, unit: !1, variables: !3)
!18 = !DILocation(line: 22, scope: !19, inlinedAt: !21)
!19 = distinct !DISubprogram(name: "@generated body;", linkageName: "@generated body", scope: !20, file: !20, type: !8, isLocal: false, isDefinition: true, isOptimized: true, unit: !1, variables: !3)
!20 = !DIFile(filename: "/home/maleadt/Julia/MixedModeBroadcastAD/src/MixedModeBroadcastAD.jl", directory: ".")
!21 = !DILocation(line: 75, scope: !22, inlinedAt: !23)
!22 = distinct !DISubprogram(name: "_cuda_broadcast_kernel!;", linkageName: "_cuda_broadcast_kernel!", scope: !20, file: !20, line: 75, type: !8, isLocal: false, isDefinition: true, scopeLine: 75, isOptimized: true, unit: !1, variables: !3)
!23 = !DILocation(line: 53, scope: !24)
!24 = distinct !DISubprogram(name: "Kernel", linkageName: "julia_Kernel_34868", scope: null, file: !2, line: 53, type: !8, isLocal: false, isDefinition: true, scopeLine: 53, isOptimized: true, unit: !1, variables: !3)
!25 = !{i32 0, i32 2147483646}
!26 = !DILocation(line: 645, scope: !27, inlinedAt: !29)
!27 = distinct !DISubprogram(name: "toInt64;", linkageName: "toInt64", scope: !28, file: !28, line: 645, type: !8, isLocal: false, isDefinition: true, scopeLine: 645, isOptimized: true, unit: !1, variables: !3)
!28 = !DIFile(filename: "boot.jl", directory: ".")
!29 = !DILocation(line: 721, scope: !30, inlinedAt: !14)
!30 = distinct !DISubprogram(name: "Type;", linkageName: "Type", scope: !28, file: !28, line: 721, type: !8, isLocal: false, isDefinition: true, scopeLine: 721, isOptimized: true, unit: !1, variables: !3)
!31 = !DILocation(line: 42, scope: !32, inlinedAt: !33)
!32 = distinct !DISubprogram(name: "macro expansion;", linkageName: "macro expansion", scope: !7, file: !7, type: !8, isLocal: false, isDefinition: true, isOptimized: true, unit: !1, variables: !3)
!33 = !DILocation(line: 8, scope: !34, inlinedAt: !35)
!34 = distinct !DISubprogram(name: "@generated body;", linkageName: "@generated body", scope: !11, file: !11, type: !8, isLocal: false, isDefinition: true, isOptimized: true, unit: !1, variables: !3)
!35 = !DILocation(line: 8, scope: !36, inlinedAt: !37)
!36 = distinct !DISubprogram(name: "_index;", linkageName: "_index", scope: !11, file: !11, line: 8, type: !8, isLocal: false, isDefinition: true, scopeLine: 8, isOptimized: true, unit: !1, variables: !3)
!37 = !DILocation(line: 50, scope: !38, inlinedAt: !39)
!38 = distinct !DISubprogram(name: "blockDim_x;", linkageName: "blockDim_x", scope: !11, file: !11, line: 50, type: !8, isLocal: false, isDefinition: true, scopeLine: 50, isOptimized: true, unit: !1, variables: !3)
!39 = !DILocation(line: 82, scope: !40, inlinedAt: !18)
!40 = distinct !DISubprogram(name: "blockDim;", linkageName: "blockDim", scope: !11, file: !11, line: 82, type: !8, isLocal: false, isDefinition: true, scopeLine: 82, isOptimized: true, unit: !1, variables: !3)
!41 = !{i32 1, i32 1024}
!42 = !DILocation(line: 645, scope: !43, inlinedAt: !44)
!43 = distinct !DISubprogram(name: "toInt64;", linkageName: "toInt64", scope: !28, file: !28, line: 645, type: !8, isLocal: false, isDefinition: true, scopeLine: 645, isOptimized: true, unit: !1, variables: !3)
!44 = !DILocation(line: 721, scope: !45, inlinedAt: !37)
!45 = distinct !DISubprogram(name: "Type;", linkageName: "Type", scope: !28, file: !28, line: 721, type: !8, isLocal: false, isDefinition: true, scopeLine: 721, isOptimized: true, unit: !1, variables: !3)
!46 = !DILocation(line: 54, scope: !47, inlinedAt: !18)
!47 = distinct !DISubprogram(name: "*;", linkageName: "*", scope: !48, file: !48, line: 54, type: !8, isLocal: false, isDefinition: true, scopeLine: 54, isOptimized: true, unit: !1, variables: !3)
!48 = !DIFile(filename: "int.jl", directory: ".")
!49 = !DILocation(line: 42, scope: !50, inlinedAt: !51)
!50 = distinct !DISubprogram(name: "macro expansion;", linkageName: "macro expansion", scope: !7, file: !7, type: !8, isLocal: false, isDefinition: true, isOptimized: true, unit: !1, variables: !3)
!51 = !DILocation(line: 8, scope: !52, inlinedAt: !53)
!52 = distinct !DISubprogram(name: "@generated body;", linkageName: "@generated body", scope: !11, file: !11, type: !8, isLocal: false, isDefinition: true, isOptimized: true, unit: !1, variables: !3)
!53 = !DILocation(line: 8, scope: !54, inlinedAt: !55)
!54 = distinct !DISubprogram(name: "_index;", linkageName: "_index", scope: !11, file: !11, line: 8, type: !8, isLocal: false, isDefinition: true, scopeLine: 8, isOptimized: true, unit: !1, variables: !3)
!55 = !DILocation(line: 45, scope: !56, inlinedAt: !57)
!56 = distinct !DISubprogram(name: "threadIdx_x;", linkageName: "threadIdx_x", scope: !11, file: !11, line: 45, type: !8, isLocal: false, isDefinition: true, scopeLine: 45, isOptimized: true, unit: !1, variables: !3)
!57 = !DILocation(line: 89, scope: !58, inlinedAt: !18)
!58 = distinct !DISubprogram(name: "threadIdx;", linkageName: "threadIdx", scope: !11, file: !11, line: 89, type: !8, isLocal: false, isDefinition: true, scopeLine: 89, isOptimized: true, unit: !1, variables: !3)
!59 = !{i32 0, i32 1023}
!60 = !DILocation(line: 645, scope: !61, inlinedAt: !62)
!61 = distinct !DISubprogram(name: "toInt64;", linkageName: "toInt64", scope: !28, file: !28, line: 645, type: !8, isLocal: false, isDefinition: true, scopeLine: 645, isOptimized: true, unit: !1, variables: !3)
!62 = !DILocation(line: 721, scope: !63, inlinedAt: !55)
!63 = distinct !DISubprogram(name: "Type;", linkageName: "Type", scope: !28, file: !28, line: 721, type: !8, isLocal: false, isDefinition: true, scopeLine: 721, isOptimized: true, unit: !1, variables: !3)
!64 = !DILocation(line: 53, scope: !65, inlinedAt: !55)
!65 = distinct !DISubprogram(name: "+;", linkageName: "+", scope: !48, file: !48, line: 53, type: !8, isLocal: false, isDefinition: true, scopeLine: 53, isOptimized: true, unit: !1, variables: !3)
!66 = !DILocation(line: 53, scope: !67, inlinedAt: !18)
!67 = distinct !DISubprogram(name: "+;", linkageName: "+", scope: !48, file: !48, line: 53, type: !8, isLocal: false, isDefinition: true, scopeLine: 53, isOptimized: true, unit: !1, variables: !3)
!68 = !DILocation(line: 54, scope: !69, inlinedAt: !70)
!69 = distinct !DISubprogram(name: "*;", linkageName: "*", scope: !48, file: !48, line: 54, type: !8, isLocal: false, isDefinition: true, scopeLine: 54, isOptimized: true, unit: !1, variables: !3)
!70 = !DILocation(line: 40, scope: !71, inlinedAt: !72)
!71 = distinct !DISubprogram(name: "@generated body;", linkageName: "@generated body", scope: !20, file: !20, type: !8, isLocal: false, isDefinition: true, isOptimized: true, unit: !1, variables: !3)
!72 = !DILocation(line: 31, scope: !73, inlinedAt: !74)
!73 = distinct !DISubprogram(name: "lengthproduct;", linkageName: "lengthproduct", scope: !20, file: !20, line: 31, type: !8, isLocal: false, isDefinition: true, scopeLine: 31, isOptimized: true, unit: !1, variables: !3)
!74 = !DILocation(line: 23, scope: !19, inlinedAt: !21)
!75 = !DILocation(line: 229, scope: !76, inlinedAt: !77)
!76 = distinct !DISubprogram(name: "div;", linkageName: "div", scope: !48, file: !48, line: 229, type: !8, isLocal: false, isDefinition: true, scopeLine: 229, isOptimized: true, unit: !1, variables: !3)
!77 = !DILocation(line: 1767, scope: !78, inlinedAt: !80)
!78 = distinct !DISubprogram(name: "_div;", linkageName: "_div", scope: !79, file: !79, line: 1767, type: !8, isLocal: false, isDefinition: true, scopeLine: 1767, isOptimized: true, unit: !1, variables: !3)
!79 = !DIFile(filename: "abstractarray.jl", directory: ".")
!80 = !DILocation(line: 1760, scope: !81, inlinedAt: !82)
!81 = distinct !DISubprogram(name: "_ind2sub_recurse;", linkageName: "_ind2sub_recurse", scope: !79, file: !79, line: 1759, type: !8, isLocal: false, isDefinition: true, scopeLine: 1759, isOptimized: true, unit: !1, variables: !3)
!82 = !DILocation(line: 1747, scope: !83, inlinedAt: !84)
!83 = distinct !DISubprogram(name: "_ind2sub;", linkageName: "_ind2sub", scope: !79, file: !79, line: 1747, type: !8, isLocal: false, isDefinition: true, scopeLine: 1747, isOptimized: true, unit: !1, variables: !3)
!84 = !DILocation(line: 1710, scope: !85, inlinedAt: !86)
!85 = distinct !DISubprogram(name: "_ind2sub;", linkageName: "_ind2sub", scope: !79, file: !79, line: 1710, type: !8, isLocal: false, isDefinition: true, scopeLine: 1710, isOptimized: true, unit: !1, variables: !3)
!86 = !DILocation(line: 1006, scope: !87, inlinedAt: !88)
!87 = distinct !DISubprogram(name: "_unsafe_ind2sub;", linkageName: "_unsafe_ind2sub", scope: !79, file: !79, line: 1006, type: !8, isLocal: false, isDefinition: true, scopeLine: 1006, isOptimized: true, unit: !1, variables: !3)
!88 = !DILocation(line: 984, scope: !89, inlinedAt: !90)
!89 = distinct !DISubprogram(name: "_to_subscript_indices;", linkageName: "_to_subscript_indices", scope: !79, file: !79, line: 984, type: !8, isLocal: false, isDefinition: true, scopeLine: 984, isOptimized: true, unit: !1, variables: !3)
!90 = !DILocation(line: 977, scope: !91, inlinedAt: !92)
!91 = distinct !DISubprogram(name: "_getindex;", linkageName: "_getindex", scope: !79, file: !79, line: 976, type: !8, isLocal: false, isDefinition: true, scopeLine: 976, isOptimized: true, unit: !1, variables: !3)
!92 = !DILocation(line: 942, scope: !93, inlinedAt: !94)
!93 = distinct !DISubprogram(name: "getindex;", linkageName: "getindex", scope: !79, file: !79, line: 942, type: !8, isLocal: false, isDefinition: true, scopeLine: 942, isOptimized: true, unit: !1, variables: !3)
!94 = !DILocation(line: 47, scope: !19, inlinedAt: !21)
!95 = !DILocation(line: 52, scope: !96, inlinedAt: !82)
!96 = distinct !DISubprogram(name: "-;", linkageName: "-", scope: !48, file: !48, line: 52, type: !8, isLocal: false, isDefinition: true, scopeLine: 52, isOptimized: true, unit: !1, variables: !3)
!97 = !DILocation(line: 54, scope: !98, inlinedAt: !99)
!98 = distinct !DISubprogram(name: "*;", linkageName: "*", scope: !48, file: !48, line: 54, type: !8, isLocal: false, isDefinition: true, scopeLine: 54, isOptimized: true, unit: !1, variables: !3)
!99 = !DILocation(line: 1761, scope: !81, inlinedAt: !82)
!100 = !DILocation(line: 52, scope: !101, inlinedAt: !99)
!101 = distinct !DISubprogram(name: "-;", linkageName: "-", scope: !48, file: !48, line: 52, type: !8, isLocal: false, isDefinition: true, scopeLine: 52, isOptimized: true, unit: !1, variables: !3)
!102 = !DILocation(line: 53, scope: !103, inlinedAt: !99)
!103 = distinct !DISubprogram(name: "+;", linkageName: "+", scope: !48, file: !48, line: 53, type: !8, isLocal: false, isDefinition: true, scopeLine: 53, isOptimized: true, unit: !1, variables: !3)
!104 = !DILocation(line: 53, scope: !105, inlinedAt: !106)
!105 = distinct !DISubprogram(name: "+;", linkageName: "+", scope: !48, file: !48, line: 53, type: !8, isLocal: false, isDefinition: true, scopeLine: 53, isOptimized: true, unit: !1, variables: !3)
!106 = !DILocation(line: 1765, scope: !107, inlinedAt: !108)
!107 = distinct !DISubprogram(name: "_lookup;", linkageName: "_lookup", scope: !79, file: !79, line: 1765, type: !8, isLocal: false, isDefinition: true, scopeLine: 1765, isOptimized: true, unit: !1, variables: !3)
!108 = !DILocation(line: 1755, scope: !109, inlinedAt: !99)
!109 = distinct !DISubprogram(name: "_ind2sub_recurse;", linkageName: "_ind2sub_recurse", scope: !79, file: !79, line: 1755, type: !8, isLocal: false, isDefinition: true, scopeLine: 1755, isOptimized: true, unit: !1, variables: !3)
!110 = !DILocation(line: 351, scope: !111, inlinedAt: !113)
!111 = distinct !DISubprogram(name: "ifelse;", linkageName: "ifelse", scope: !112, file: !112, line: 351, type: !8, isLocal: false, isDefinition: true, scopeLine: 351, isOptimized: true, unit: !1, variables: !3)
!112 = !DIFile(filename: "operators.jl", directory: ".")
!113 = !DILocation(line: 287, scope: !114, inlinedAt: !116)
!114 = distinct !DISubprogram(name: "_newindex;", linkageName: "_newindex", scope: !115, file: !115, line: 287, type: !8, isLocal: false, isDefinition: true, scopeLine: 287, isOptimized: true, unit: !1, variables: !3)
!115 = !DIFile(filename: "broadcast.jl", directory: ".")
!116 = !DILocation(line: 286, scope: !117, inlinedAt: !118)
!117 = distinct !DISubprogram(name: "newindex;", linkageName: "newindex", scope: !115, file: !115, line: 286, type: !8, isLocal: false, isDefinition: true, scopeLine: 286, isOptimized: true, unit: !1, variables: !3)
!118 = !DILocation(line: 81, scope: !19, inlinedAt: !21)
!119 = !DILocation(line: 351, scope: !120, inlinedAt: !121)
!120 = distinct !DISubprogram(name: "ifelse;", linkageName: "ifelse", scope: !112, file: !112, line: 351, type: !8, isLocal: false, isDefinition: true, scopeLine: 351, isOptimized: true, unit: !1, variables: !3)
!121 = !DILocation(line: 287, scope: !122, inlinedAt: !113)
!122 = distinct !DISubprogram(name: "_newindex;", linkageName: "_newindex", scope: !115, file: !115, line: 287, type: !8, isLocal: false, isDefinition: true, scopeLine: 287, isOptimized: true, unit: !1, variables: !3)
!123 = !DILocation(line: 435, scope: !124, inlinedAt: !126)
!124 = distinct !DISubprogram(name: "max;", linkageName: "max", scope: !125, file: !125, line: 435, type: !8, isLocal: false, isDefinition: true, scopeLine: 435, isOptimized: true, unit: !1, variables: !3)
!125 = !DIFile(filename: "promotion.jl", directory: ".")
!126 = !DILocation(line: 195, scope: !127, inlinedAt: !129)
!127 = distinct !DISubprogram(name: "Type;", linkageName: "Type", scope: !128, file: !128, line: 195, type: !8, isLocal: false, isDefinition: true, scopeLine: 195, isOptimized: true, unit: !1, variables: !3)
!128 = !DIFile(filename: "range.jl", directory: ".")
!129 = !DILocation(line: 197, scope: !130, inlinedAt: !131)
!130 = distinct !DISubprogram(name: "Type;", linkageName: "Type", scope: !128, file: !128, line: 197, type: !8, isLocal: false, isDefinition: true, scopeLine: 197, isOptimized: true, unit: !1, variables: !3)
!131 = !DILocation(line: 152, scope: !132, inlinedAt: !134)
!132 = distinct !DISubprogram(name: "map;", linkageName: "map", scope: !133, file: !133, line: 152, type: !8, isLocal: false, isDefinition: true, scopeLine: 152, isOptimized: true, unit: !1, variables: !3)
!133 = !DIFile(filename: "tuple.jl", directory: ".")
!134 = !DILocation(line: 83, scope: !135, inlinedAt: !136)
!135 = distinct !DISubprogram(name: "axes;", linkageName: "axes", scope: !79, file: !79, line: 83, type: !8, isLocal: false, isDefinition: true, scopeLine: 83, isOptimized: true, unit: !1, variables: !3)
!136 = !DILocation(line: 1705, scope: !137, inlinedAt: !138)
!137 = distinct !DISubprogram(name: "_sub2ind;", linkageName: "_sub2ind", scope: !79, file: !79, line: 1705, type: !8, isLocal: false, isDefinition: true, scopeLine: 1705, isOptimized: true, unit: !1, variables: !3)
!138 = !DILocation(line: 971, scope: !139, inlinedAt: !140)
!139 = distinct !DISubprogram(name: "_to_linear_index;", linkageName: "_to_linear_index", scope: !79, file: !79, line: 971, type: !8, isLocal: false, isDefinition: true, scopeLine: 971, isOptimized: true, unit: !1, variables: !3)
!140 = !DILocation(line: 965, scope: !141, inlinedAt: !142)
!141 = distinct !DISubprogram(name: "_getindex;", linkageName: "_getindex", scope: !79, file: !79, line: 964, type: !8, isLocal: false, isDefinition: true, scopeLine: 964, isOptimized: true, unit: !1, variables: !3)
!142 = !DILocation(line: 942, scope: !143, inlinedAt: !144)
!143 = distinct !DISubprogram(name: "getindex;", linkageName: "getindex", scope: !79, file: !79, line: 942, type: !8, isLocal: false, isDefinition: true, scopeLine: 942, isOptimized: true, unit: !1, variables: !3)
!144 = !DILocation(line: 320, scope: !145, inlinedAt: !146)
!145 = distinct !DISubprogram(name: "_broadcast_getindex;", linkageName: "_broadcast_getindex", scope: !115, file: !115, line: 320, type: !8, isLocal: false, isDefinition: true, scopeLine: 320, isOptimized: true, unit: !1, variables: !3)
!146 = !DILocation(line: 318, scope: !147, inlinedAt: !148)
!147 = distinct !DISubprogram(name: "_broadcast_getindex;", linkageName: "_broadcast_getindex", scope: !115, file: !115, line: 318, type: !8, isLocal: false, isDefinition: true, scopeLine: 318, isOptimized: true, unit: !1, variables: !3)
!148 = !DILocation(line: 82, scope: !19, inlinedAt: !21)
!149 = !DILocation(line: 52, scope: !150, inlinedAt: !151)
!150 = distinct !DISubprogram(name: "-;", linkageName: "-", scope: !48, file: !48, line: 52, type: !8, isLocal: false, isDefinition: true, scopeLine: 52, isOptimized: true, unit: !1, variables: !3)
!151 = !DILocation(line: 1743, scope: !152, inlinedAt: !153)
!152 = distinct !DISubprogram(name: "offsetin;", linkageName: "offsetin", scope: !79, file: !79, line: 1743, type: !8, isLocal: false, isDefinition: true, scopeLine: 1743, isOptimized: true, unit: !1, variables: !3)
!153 = !DILocation(line: 1737, scope: !154, inlinedAt: !155)
!154 = distinct !DISubprogram(name: "_sub2ind_recurse;", linkageName: "_sub2ind_recurse", scope: !79, file: !79, line: 1736, type: !8, isLocal: false, isDefinition: true, scopeLine: 1736, isOptimized: true, unit: !1, variables: !3)
!155 = !DILocation(line: 1737, scope: !156, inlinedAt: !157)
!156 = distinct !DISubprogram(name: "_sub2ind_recurse;", linkageName: "_sub2ind_recurse", scope: !79, file: !79, line: 1736, type: !8, isLocal: false, isDefinition: true, scopeLine: 1736, isOptimized: true, unit: !1, variables: !3)
!157 = !DILocation(line: 1721, scope: !158, inlinedAt: !136)
!158 = distinct !DISubprogram(name: "_sub2ind;", linkageName: "_sub2ind", scope: !79, file: !79, line: 1721, type: !8, isLocal: false, isDefinition: true, scopeLine: 1721, isOptimized: true, unit: !1, variables: !3)
!159 = !DILocation(line: 54, scope: !160, inlinedAt: !153)
!160 = distinct !DISubprogram(name: "*;", linkageName: "*", scope: !48, file: !48, line: 54, type: !8, isLocal: false, isDefinition: true, scopeLine: 54, isOptimized: true, unit: !1, variables: !3)
!161 = !DILocation(line: 53, scope: !162, inlinedAt: !153)
!162 = distinct !DISubprogram(name: "+;", linkageName: "+", scope: !48, file: !48, line: 53, type: !8, isLocal: false, isDefinition: true, scopeLine: 53, isOptimized: true, unit: !1, variables: !3)
!163 = !DILocation(line: 52, scope: !164, inlinedAt: !165)
!164 = distinct !DISubprogram(name: "-;", linkageName: "-", scope: !48, file: !48, line: 52, type: !8, isLocal: false, isDefinition: true, scopeLine: 52, isOptimized: true, unit: !1, variables: !3)
!165 = !DILocation(line: 42, scope: !166, inlinedAt: !167)
!166 = distinct !DISubprogram(name: "macro expansion;", linkageName: "macro expansion", scope: !7, file: !7, type: !8, isLocal: false, isDefinition: true, isOptimized: true, unit: !1, variables: !3)
!167 = !DILocation(line: 103, scope: !168, inlinedAt: !170)
!168 = distinct !DISubprogram(name: "@generated body;", linkageName: "@generated body", scope: !169, file: !169, type: !8, isLocal: false, isDefinition: true, isOptimized: true, unit: !1, variables: !3)
!169 = !DIFile(filename: "/home/maleadt/Julia/CUDAnative/src/pointer.jl", directory: ".")
!170 = !DILocation(line: 103, scope: !171, inlinedAt: !172)
!171 = distinct !DISubprogram(name: "unsafe_load;", linkageName: "unsafe_load", scope: !169, file: !169, line: 103, type: !8, isLocal: false, isDefinition: true, scopeLine: 103, isOptimized: true, unit: !1, variables: !3)
!172 = !DILocation(line: 78, scope: !173, inlinedAt: !140)
!173 = distinct !DISubprogram(name: "getindex;", linkageName: "getindex", scope: !174, file: !174, line: 76, type: !8, isLocal: false, isDefinition: true, scopeLine: 76, isOptimized: true, unit: !1, variables: !3)
!174 = !DIFile(filename: "/home/maleadt/Julia/CUDAnative/src/device/array.jl", directory: ".")
!175 = !DILocation(line: 114, scope: !176, inlinedAt: !178)
!176 = distinct !DISubprogram(name: "macro expansion;", linkageName: "macro expansion", scope: !177, file: !177, type: !8, isLocal: false, isDefinition: true, isOptimized: true, unit: !1, variables: !3)
!177 = !DIFile(filename: "/home/maleadt/Julia/CUDAnative/src/cgutils.jl", directory: ".")
!178 = !DILocation(line: 48, scope: !179, inlinedAt: !181)
!179 = distinct !DISubprogram(name: "tanh;", linkageName: "tanh", scope: !180, file: !180, line: 48, type: !8, isLocal: false, isDefinition: true, scopeLine: 48, isOptimized: true, unit: !1, variables: !3)
!180 = !DIFile(filename: "/home/maleadt/Julia/CUDAnative/src/device/libdevice.jl", directory: ".")
!181 = !DILocation(line: 6, scope: !182, inlinedAt: !184)
!182 = distinct !DISubprogram(name: "cuda_tanh;", linkageName: "cuda_tanh", scope: !183, file: !183, line: 6, type: !8, isLocal: false, isDefinition: true, scopeLine: 6, isOptimized: true, unit: !1, variables: !3)
!183 = !DIFile(filename: "/home/maleadt/Julia/MixedModeBroadcastAD/test/kernels.jl", directory: ".")
!184 = !DILocation(line: 83, scope: !19, inlinedAt: !21)
!185 = !DILocation(line: 435, scope: !186, inlinedAt: !187)
!186 = distinct !DISubprogram(name: "max;", linkageName: "max", scope: !125, file: !125, line: 435, type: !8, isLocal: false, isDefinition: true, scopeLine: 435, isOptimized: true, unit: !1, variables: !3)
!187 = !DILocation(line: 195, scope: !188, inlinedAt: !189)
!188 = distinct !DISubprogram(name: "Type;", linkageName: "Type", scope: !128, file: !128, line: 195, type: !8, isLocal: false, isDefinition: true, scopeLine: 195, isOptimized: true, unit: !1, variables: !3)
!189 = !DILocation(line: 197, scope: !190, inlinedAt: !191)
!190 = distinct !DISubprogram(name: "Type;", linkageName: "Type", scope: !128, file: !128, line: 197, type: !8, isLocal: false, isDefinition: true, scopeLine: 197, isOptimized: true, unit: !1, variables: !3)
!191 = !DILocation(line: 152, scope: !192, inlinedAt: !193)
!192 = distinct !DISubprogram(name: "map;", linkageName: "map", scope: !133, file: !133, line: 152, type: !8, isLocal: false, isDefinition: true, scopeLine: 152, isOptimized: true, unit: !1, variables: !3)
!193 = !DILocation(line: 83, scope: !194, inlinedAt: !195)
!194 = distinct !DISubprogram(name: "axes;", linkageName: "axes", scope: !79, file: !79, line: 83, type: !8, isLocal: false, isDefinition: true, scopeLine: 83, isOptimized: true, unit: !1, variables: !3)
!195 = !DILocation(line: 1705, scope: !196, inlinedAt: !197)
!196 = distinct !DISubprogram(name: "_sub2ind;", linkageName: "_sub2ind", scope: !79, file: !79, line: 1705, type: !8, isLocal: false, isDefinition: true, scopeLine: 1705, isOptimized: true, unit: !1, variables: !3)
!197 = !DILocation(line: 971, scope: !198, inlinedAt: !199)
!198 = distinct !DISubprogram(name: "_to_linear_index;", linkageName: "_to_linear_index", scope: !79, file: !79, line: 971, type: !8, isLocal: false, isDefinition: true, scopeLine: 971, isOptimized: true, unit: !1, variables: !3)
!199 = !DILocation(line: 1042, scope: !200, inlinedAt: !201)
!200 = distinct !DISubprogram(name: "_setindex!;", linkageName: "_setindex!", scope: !79, file: !79, line: 1041, type: !8, isLocal: false, isDefinition: true, scopeLine: 1041, isOptimized: true, unit: !1, variables: !3)
!201 = !DILocation(line: 1019, scope: !202, inlinedAt: !184)
!202 = distinct !DISubprogram(name: "setindex!;", linkageName: "setindex!", scope: !79, file: !79, line: 1019, type: !8, isLocal: false, isDefinition: true, scopeLine: 1019, isOptimized: true, unit: !1, variables: !3)
!203 = !DILocation(line: 54, scope: !204, inlinedAt: !205)
!204 = distinct !DISubprogram(name: "*;", linkageName: "*", scope: !48, file: !48, line: 54, type: !8, isLocal: false, isDefinition: true, scopeLine: 54, isOptimized: true, unit: !1, variables: !3)
!205 = !DILocation(line: 1737, scope: !206, inlinedAt: !207)
!206 = distinct !DISubprogram(name: "_sub2ind_recurse;", linkageName: "_sub2ind_recurse", scope: !79, file: !79, line: 1736, type: !8, isLocal: false, isDefinition: true, scopeLine: 1736, isOptimized: true, unit: !1, variables: !3)
!207 = !DILocation(line: 1737, scope: !208, inlinedAt: !209)
!208 = distinct !DISubprogram(name: "_sub2ind_recurse;", linkageName: "_sub2ind_recurse", scope: !79, file: !79, line: 1736, type: !8, isLocal: false, isDefinition: true, scopeLine: 1736, isOptimized: true, unit: !1, variables: !3)
!209 = !DILocation(line: 1721, scope: !210, inlinedAt: !195)
!210 = distinct !DISubprogram(name: "_sub2ind;", linkageName: "_sub2ind", scope: !79, file: !79, line: 1721, type: !8, isLocal: false, isDefinition: true, scopeLine: 1721, isOptimized: true, unit: !1, variables: !3)
!211 = !DILocation(line: 52, scope: !212, inlinedAt: !213)
!212 = distinct !DISubprogram(name: "-;", linkageName: "-", scope: !48, file: !48, line: 52, type: !8, isLocal: false, isDefinition: true, scopeLine: 52, isOptimized: true, unit: !1, variables: !3)
!213 = !DILocation(line: 42, scope: !214, inlinedAt: !215)
!214 = distinct !DISubprogram(name: "macro expansion;", linkageName: "macro expansion", scope: !7, file: !7, type: !8, isLocal: false, isDefinition: true, isOptimized: true, unit: !1, variables: !3)
!215 = !DILocation(line: 133, scope: !216, inlinedAt: !217)
!216 = distinct !DISubprogram(name: "@generated body;", linkageName: "@generated body", scope: !169, file: !169, type: !8, isLocal: false, isDefinition: true, isOptimized: true, unit: !1, variables: !3)
!217 = !DILocation(line: 133, scope: !218, inlinedAt: !219)
!218 = distinct !DISubprogram(name: "unsafe_store!;", linkageName: "unsafe_store!", scope: !169, file: !169, line: 133, type: !8, isLocal: false, isDefinition: true, scopeLine: 133, isOptimized: true, unit: !1, variables: !3)
!219 = !DILocation(line: 84, scope: !220, inlinedAt: !199)
!220 = distinct !DISubprogram(name: "setindex!;", linkageName: "setindex!", scope: !174, file: !174, line: 82, type: !8, isLocal: false, isDefinition: true, scopeLine: 82, isOptimized: true, unit: !1, variables: !3)
!221 = !DILocation(line: 85, scope: !19, inlinedAt: !21)
