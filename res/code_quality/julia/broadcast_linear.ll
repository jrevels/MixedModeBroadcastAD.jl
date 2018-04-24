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

declare float @__nv_tanhf(float) local_unnamed_addr

define void @ptxcall_Kernel_3({ [2 x i64], { i64 } }, [1 x { [2 x i64], { i64 } }], [1 x [2 x i8]], [1 x [2 x i64]], [2 x { i64 }]) local_unnamed_addr {
entry:
  %.elt11 = extractvalue [2 x { i64 }] %4, 0
  %5 = extractvalue { i64 } %.elt11, 0
  %.elt13 = extractvalue [2 x { i64 }] %4, 1
  %6 = extractvalue { i64 } %.elt13, 0
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
  br i1 %17, label %julia_Kernel_34871.exit, label %L202.i, !dbg !74

L202.i:                                           ; preds = %entry
  %18 = extractvalue [1 x { [2 x i64], { i64 } }] %1, 0
  %.elt7 = extractvalue { [2 x i64], { i64 } } %18, 1
  %19 = extractvalue { i64 } %.elt7, 0
  %.elt2 = extractvalue { [2 x i64], { i64 } } %0, 1
  %20 = extractvalue { i64 } %.elt2, 0
  %21 = add nuw nsw i64 %11, %13, !dbg !75
  %22 = inttoptr i64 %19 to float*, !dbg !77
  %23 = getelementptr float, float* %22, i64 %21, !dbg !77
  %24 = addrspacecast float* %23 to float addrspace(1)*, !dbg !77
  %25 = load float, float addrspace(1)* %24, align 4, !dbg !77
  %26 = call float @__nv_tanhf(float %25), !dbg !88
  %27 = inttoptr i64 %20 to float*, !dbg !98
  %28 = getelementptr float, float* %27, i64 %21, !dbg !98
  %29 = addrspacecast float* %28 to float addrspace(1)*, !dbg !98
  store float %26, float addrspace(1)* %29, align 4, !dbg !98
  br label %julia_Kernel_34871.exit, !dbg !106

julia_Kernel_34871.exit:                          ; preds = %entry, %L202.i
  ret void
}

attributes #0 = { nounwind readnone }

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
!18 = !DILocation(line: 32, scope: !19, inlinedAt: !21)
!19 = distinct !DISubprogram(name: "@generated body;", linkageName: "@generated body", scope: !20, file: !20, type: !8, isLocal: false, isDefinition: true, isOptimized: true, unit: !1, variables: !3)
!20 = !DIFile(filename: "/home/maleadt/Julia/MixedModeBroadcastAD/src/MixedModeBroadcastAD.jl", directory: ".")
!21 = !DILocation(line: 89, scope: !22, inlinedAt: !23)
!22 = distinct !DISubprogram(name: "_cuda_broadcast_kernel!;", linkageName: "_cuda_broadcast_kernel!", scope: !20, file: !20, line: 89, type: !8, isLocal: false, isDefinition: true, scopeLine: 89, isOptimized: true, unit: !1, variables: !3)
!23 = !DILocation(line: 53, scope: !24)
!24 = distinct !DISubprogram(name: "Kernel", linkageName: "julia_Kernel_34871", scope: null, file: !2, line: 53, type: !8, isLocal: false, isDefinition: true, scopeLine: 53, isOptimized: true, unit: !1, variables: !3)
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
!70 = !DILocation(line: 50, scope: !71, inlinedAt: !72)
!71 = distinct !DISubprogram(name: "@generated body;", linkageName: "@generated body", scope: !20, file: !20, type: !8, isLocal: false, isDefinition: true, isOptimized: true, unit: !1, variables: !3)
!72 = !DILocation(line: 41, scope: !73, inlinedAt: !74)
!73 = distinct !DISubprogram(name: "lengthproduct;", linkageName: "lengthproduct", scope: !20, file: !20, line: 41, type: !8, isLocal: false, isDefinition: true, scopeLine: 41, isOptimized: true, unit: !1, variables: !3)
!74 = !DILocation(line: 33, scope: !19, inlinedAt: !21)
!75 = !DILocation(line: 52, scope: !76, inlinedAt: !77)
!76 = distinct !DISubprogram(name: "-;", linkageName: "-", scope: !48, file: !48, line: 52, type: !8, isLocal: false, isDefinition: true, scopeLine: 52, isOptimized: true, unit: !1, variables: !3)
!77 = !DILocation(line: 42, scope: !78, inlinedAt: !79)
!78 = distinct !DISubprogram(name: "macro expansion;", linkageName: "macro expansion", scope: !7, file: !7, type: !8, isLocal: false, isDefinition: true, isOptimized: true, unit: !1, variables: !3)
!79 = !DILocation(line: 103, scope: !80, inlinedAt: !82)
!80 = distinct !DISubprogram(name: "@generated body;", linkageName: "@generated body", scope: !81, file: !81, type: !8, isLocal: false, isDefinition: true, isOptimized: true, unit: !1, variables: !3)
!81 = !DIFile(filename: "/home/maleadt/Julia/CUDAnative/src/pointer.jl", directory: ".")
!82 = !DILocation(line: 103, scope: !83, inlinedAt: !84)
!83 = distinct !DISubprogram(name: "unsafe_load;", linkageName: "unsafe_load", scope: !81, file: !81, line: 103, type: !8, isLocal: false, isDefinition: true, scopeLine: 103, isOptimized: true, unit: !1, variables: !3)
!84 = !DILocation(line: 78, scope: !85, inlinedAt: !87)
!85 = distinct !DISubprogram(name: "getindex;", linkageName: "getindex", scope: !86, file: !86, line: 76, type: !8, isLocal: false, isDefinition: true, scopeLine: 76, isOptimized: true, unit: !1, variables: !3)
!86 = !DIFile(filename: "/home/maleadt/Julia/CUDAnative/src/device/array.jl", directory: ".")
!87 = !DILocation(line: 93, scope: !19, inlinedAt: !21)
!88 = !DILocation(line: 114, scope: !89, inlinedAt: !91)
!89 = distinct !DISubprogram(name: "macro expansion;", linkageName: "macro expansion", scope: !90, file: !90, type: !8, isLocal: false, isDefinition: true, isOptimized: true, unit: !1, variables: !3)
!90 = !DIFile(filename: "/home/maleadt/Julia/CUDAnative/src/cgutils.jl", directory: ".")
!91 = !DILocation(line: 48, scope: !92, inlinedAt: !94)
!92 = distinct !DISubprogram(name: "tanh;", linkageName: "tanh", scope: !93, file: !93, line: 48, type: !8, isLocal: false, isDefinition: true, scopeLine: 48, isOptimized: true, unit: !1, variables: !3)
!93 = !DIFile(filename: "/home/maleadt/Julia/CUDAnative/src/device/libdevice.jl", directory: ".")
!94 = !DILocation(line: 6, scope: !95, inlinedAt: !97)
!95 = distinct !DISubprogram(name: "cuda_tanh;", linkageName: "cuda_tanh", scope: !96, file: !96, line: 6, type: !8, isLocal: false, isDefinition: true, scopeLine: 6, isOptimized: true, unit: !1, variables: !3)
!96 = !DIFile(filename: "/home/maleadt/Julia/MixedModeBroadcastAD/test/kernels.jl", directory: ".")
!97 = !DILocation(line: 94, scope: !19, inlinedAt: !21)
!98 = !DILocation(line: 42, scope: !99, inlinedAt: !100)
!99 = distinct !DISubprogram(name: "macro expansion;", linkageName: "macro expansion", scope: !7, file: !7, type: !8, isLocal: false, isDefinition: true, isOptimized: true, unit: !1, variables: !3)
!100 = !DILocation(line: 133, scope: !101, inlinedAt: !102)
!101 = distinct !DISubprogram(name: "@generated body;", linkageName: "@generated body", scope: !81, file: !81, type: !8, isLocal: false, isDefinition: true, isOptimized: true, unit: !1, variables: !3)
!102 = !DILocation(line: 133, scope: !103, inlinedAt: !104)
!103 = distinct !DISubprogram(name: "unsafe_store!;", linkageName: "unsafe_store!", scope: !81, file: !81, line: 133, type: !8, isLocal: false, isDefinition: true, scopeLine: 133, isOptimized: true, unit: !1, variables: !3)
!104 = !DILocation(line: 84, scope: !105, inlinedAt: !97)
!105 = distinct !DISubprogram(name: "setindex!;", linkageName: "setindex!", scope: !86, file: !86, line: 82, type: !8, isLocal: false, isDefinition: true, scopeLine: 82, isOptimized: true, unit: !1, variables: !3)
!106 = !DILocation(line: 96, scope: !19, inlinedAt: !21)