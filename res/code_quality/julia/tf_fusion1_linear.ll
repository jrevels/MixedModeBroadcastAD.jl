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

define void @ptxcall_Kernel_7({ [2 x i64], { i64 } }, { { [2 x i64], { i64 } }, { [2 x i64], { i64 } }, float, float, { [1 x i64], { i64 } }, { [1 x i64], { i64 } } }, { [2 x i8], [2 x i8], [1 x i8], [1 x i8] }, { [2 x i64], [2 x i64], [1 x i64], [1 x i64] }, [2 x { i64 }]) local_unnamed_addr {
entry:
  %.elt29 = extractvalue [2 x { i64 }] %4, 0
  %5 = extractvalue { i64 } %.elt29, 0
  %.elt31 = extractvalue [2 x { i64 }] %4, 1
  %6 = extractvalue { i64 } %.elt31, 0
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
  br i1 %17, label %julia_Kernel_34888.exit, label %L217.i, !dbg !74

L217.i:                                           ; preds = %entry
  %.elt9 = extractvalue { { [2 x i64], { i64 } }, { [2 x i64], { i64 } }, float, float, { [1 x i64], { i64 } }, { [1 x i64], { i64 } } } %1, 2
  %.elt11 = extractvalue { { [2 x i64], { i64 } }, { [2 x i64], { i64 } }, float, float, { [1 x i64], { i64 } }, { [1 x i64], { i64 } } } %1, 3
  %.elt15 = extractvalue { { [2 x i64], { i64 } }, { [2 x i64], { i64 } }, float, float, { [1 x i64], { i64 } }, { [1 x i64], { i64 } } } %1, 5
  %.elt15.elt17 = extractvalue { [1 x i64], { i64 } } %.elt15, 1
  %18 = extractvalue { i64 } %.elt15.elt17, 0
  %.elt13 = extractvalue { { [2 x i64], { i64 } }, { [2 x i64], { i64 } }, float, float, { [1 x i64], { i64 } }, { [1 x i64], { i64 } } } %1, 4
  %.elt13.elt19 = extractvalue { [1 x i64], { i64 } } %.elt13, 1
  %19 = extractvalue { i64 } %.elt13.elt19, 0
  %.elt7 = extractvalue { { [2 x i64], { i64 } }, { [2 x i64], { i64 } }, float, float, { [1 x i64], { i64 } }, { [1 x i64], { i64 } } } %1, 1
  %.elt7.elt21 = extractvalue { [2 x i64], { i64 } } %.elt7, 1
  %20 = extractvalue { i64 } %.elt7.elt21, 0
  %.elt5 = extractvalue { { [2 x i64], { i64 } }, { [2 x i64], { i64 } }, float, float, { [1 x i64], { i64 } }, { [1 x i64], { i64 } } } %1, 0
  %.elt5.elt26 = extractvalue { [2 x i64], { i64 } } %.elt5, 1
  %21 = extractvalue { i64 } %.elt5.elt26, 0
  %.elt2 = extractvalue { [2 x i64], { i64 } } %0, 1
  %22 = extractvalue { i64 } %.elt2, 0
  %23 = add nuw nsw i64 %11, %13, !dbg !75
  %24 = inttoptr i64 %21 to float*, !dbg !77
  %25 = getelementptr float, float* %24, i64 %23, !dbg !77
  %26 = addrspacecast float* %25 to float addrspace(1)*, !dbg !77
  %27 = load float, float addrspace(1)* %26, align 4, !dbg !77
  %28 = inttoptr i64 %20 to float*, !dbg !88
  %29 = getelementptr float, float* %28, i64 %23, !dbg !88
  %30 = addrspacecast float* %29 to float addrspace(1)*, !dbg !88
  %31 = load float, float addrspace(1)* %30, align 4, !dbg !88
  %32 = inttoptr i64 %19 to float*, !dbg !96
  %33 = getelementptr float, float* %32, i64 %23, !dbg !96
  %34 = addrspacecast float* %33 to float addrspace(1)*, !dbg !96
  %35 = load float, float addrspace(1)* %34, align 4, !dbg !96
  %36 = inttoptr i64 %18 to float*, !dbg !104
  %37 = getelementptr float, float* %36, i64 %23, !dbg !104
  %38 = addrspacecast float* %37 to float addrspace(1)*, !dbg !104
  %39 = load float, float addrspace(1)* %38, align 4, !dbg !104
  %40 = fcmp fast oeq float %.elt9, %39, !dbg !112
  %41 = select i1 %40, float %.elt9, float %.elt11, !dbg !119
  %42 = select i1 %40, float %.elt11, float %.elt9, !dbg !123
  %43 = fcmp une float %.elt11, %35, !dbg !126
  %44 = select i1 %43, float %42, float %.elt11, !dbg !126
  %45 = fadd fast float %44, %41, !dbg !129
  %46 = fsub fast float %.elt9, %27, !dbg !132
  %47 = fmul fast float %31, %27, !dbg !129
  %48 = fmul fast float %47, %46, !dbg !135
  %49 = fmul fast float %48, %45, !dbg !138
  %50 = inttoptr i64 %22 to float*, !dbg !141
  %51 = getelementptr float, float* %50, i64 %23, !dbg !141
  %52 = addrspacecast float* %51 to float addrspace(1)*, !dbg !141
  store float %49, float addrspace(1)* %52, align 4, !dbg !141
  br label %julia_Kernel_34888.exit, !dbg !149

julia_Kernel_34888.exit:                          ; preds = %entry, %L217.i
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
!4 = !{void ({ [2 x i64], { i64 } }, { { [2 x i64], { i64 } }, { [2 x i64], { i64 } }, float, float, { [1 x i64], { i64 } }, { [1 x i64], { i64 } } }, { [2 x i8], [2 x i8], [1 x i8], [1 x i8] }, { [2 x i64], [2 x i64], [1 x i64], [1 x i64] }, [2 x { i64 }])* @ptxcall_Kernel_7, !"kernel", i32 1}
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
!24 = distinct !DISubprogram(name: "Kernel", linkageName: "julia_Kernel_34888", scope: null, file: !2, line: 53, type: !8, isLocal: false, isDefinition: true, scopeLine: 53, isOptimized: true, unit: !1, variables: !3)
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
!88 = !DILocation(line: 42, scope: !89, inlinedAt: !90)
!89 = distinct !DISubprogram(name: "macro expansion;", linkageName: "macro expansion", scope: !7, file: !7, type: !8, isLocal: false, isDefinition: true, isOptimized: true, unit: !1, variables: !3)
!90 = !DILocation(line: 103, scope: !91, inlinedAt: !92)
!91 = distinct !DISubprogram(name: "@generated body;", linkageName: "@generated body", scope: !81, file: !81, type: !8, isLocal: false, isDefinition: true, isOptimized: true, unit: !1, variables: !3)
!92 = !DILocation(line: 103, scope: !93, inlinedAt: !94)
!93 = distinct !DISubprogram(name: "unsafe_load;", linkageName: "unsafe_load", scope: !81, file: !81, line: 103, type: !8, isLocal: false, isDefinition: true, scopeLine: 103, isOptimized: true, unit: !1, variables: !3)
!94 = !DILocation(line: 78, scope: !95, inlinedAt: !87)
!95 = distinct !DISubprogram(name: "getindex;", linkageName: "getindex", scope: !86, file: !86, line: 76, type: !8, isLocal: false, isDefinition: true, scopeLine: 76, isOptimized: true, unit: !1, variables: !3)
!96 = !DILocation(line: 42, scope: !97, inlinedAt: !98)
!97 = distinct !DISubprogram(name: "macro expansion;", linkageName: "macro expansion", scope: !7, file: !7, type: !8, isLocal: false, isDefinition: true, isOptimized: true, unit: !1, variables: !3)
!98 = !DILocation(line: 103, scope: !99, inlinedAt: !100)
!99 = distinct !DISubprogram(name: "@generated body;", linkageName: "@generated body", scope: !81, file: !81, type: !8, isLocal: false, isDefinition: true, isOptimized: true, unit: !1, variables: !3)
!100 = !DILocation(line: 103, scope: !101, inlinedAt: !102)
!101 = distinct !DISubprogram(name: "unsafe_load;", linkageName: "unsafe_load", scope: !81, file: !81, line: 103, type: !8, isLocal: false, isDefinition: true, scopeLine: 103, isOptimized: true, unit: !1, variables: !3)
!102 = !DILocation(line: 78, scope: !103, inlinedAt: !87)
!103 = distinct !DISubprogram(name: "getindex;", linkageName: "getindex", scope: !86, file: !86, line: 76, type: !8, isLocal: false, isDefinition: true, scopeLine: 76, isOptimized: true, unit: !1, variables: !3)
!104 = !DILocation(line: 42, scope: !105, inlinedAt: !106)
!105 = distinct !DISubprogram(name: "macro expansion;", linkageName: "macro expansion", scope: !7, file: !7, type: !8, isLocal: false, isDefinition: true, isOptimized: true, unit: !1, variables: !3)
!106 = !DILocation(line: 103, scope: !107, inlinedAt: !108)
!107 = distinct !DISubprogram(name: "@generated body;", linkageName: "@generated body", scope: !81, file: !81, type: !8, isLocal: false, isDefinition: true, isOptimized: true, unit: !1, variables: !3)
!108 = !DILocation(line: 103, scope: !109, inlinedAt: !110)
!109 = distinct !DISubprogram(name: "unsafe_load;", linkageName: "unsafe_load", scope: !81, file: !81, line: 103, type: !8, isLocal: false, isDefinition: true, scopeLine: 103, isOptimized: true, unit: !1, variables: !3)
!110 = !DILocation(line: 78, scope: !111, inlinedAt: !87)
!111 = distinct !DISubprogram(name: "getindex;", linkageName: "getindex", scope: !86, file: !86, line: 76, type: !8, isLocal: false, isDefinition: true, scopeLine: 76, isOptimized: true, unit: !1, variables: !3)
!112 = !DILocation(line: 179, scope: !113, inlinedAt: !115)
!113 = distinct !DISubprogram(name: "eq_fast;", linkageName: "eq_fast", scope: !114, file: !114, line: 179, type: !8, isLocal: false, isDefinition: true, scopeLine: 179, isOptimized: true, unit: !1, variables: !3)
!114 = !DIFile(filename: "fastmath.jl", directory: ".")
!115 = !DILocation(line: 148, scope: !116, inlinedAt: !118)
!116 = distinct !DISubprogram(name: "#16;", linkageName: "#16", scope: !117, file: !117, line: 147, type: !8, isLocal: false, isDefinition: true, scopeLine: 147, isOptimized: true, unit: !1, variables: !3)
!117 = !DIFile(filename: "/home/maleadt/Julia/MixedModeBroadcastAD/test/kernels.jl", directory: ".")
!118 = !DILocation(line: 94, scope: !19, inlinedAt: !21)
!119 = !DILocation(line: 351, scope: !120, inlinedAt: !122)
!120 = distinct !DISubprogram(name: "ifelse;", linkageName: "ifelse", scope: !121, file: !121, line: 351, type: !8, isLocal: false, isDefinition: true, scopeLine: 351, isOptimized: true, unit: !1, variables: !3)
!121 = !DIFile(filename: "operators.jl", directory: ".")
!122 = !DILocation(line: 149, scope: !116, inlinedAt: !118)
!123 = !DILocation(line: 351, scope: !124, inlinedAt: !125)
!124 = distinct !DISubprogram(name: "ifelse;", linkageName: "ifelse", scope: !121, file: !121, line: 351, type: !8, isLocal: false, isDefinition: true, scopeLine: 351, isOptimized: true, unit: !1, variables: !3)
!125 = !DILocation(line: 150, scope: !116, inlinedAt: !118)
!126 = !DILocation(line: 351, scope: !127, inlinedAt: !128)
!127 = distinct !DISubprogram(name: "ifelse;", linkageName: "ifelse", scope: !121, file: !121, line: 351, type: !8, isLocal: false, isDefinition: true, scopeLine: 351, isOptimized: true, unit: !1, variables: !3)
!128 = !DILocation(line: 151, scope: !116, inlinedAt: !118)
!129 = !DILocation(line: 163, scope: !130, inlinedAt: !131)
!130 = distinct !DISubprogram(name: "add_fast;", linkageName: "add_fast", scope: !114, file: !114, line: 163, type: !8, isLocal: false, isDefinition: true, scopeLine: 163, isOptimized: true, unit: !1, variables: !3)
!131 = !DILocation(line: 154, scope: !116, inlinedAt: !118)
!132 = !DILocation(line: 164, scope: !133, inlinedAt: !134)
!133 = distinct !DISubprogram(name: "sub_fast;", linkageName: "sub_fast", scope: !114, file: !114, line: 164, type: !8, isLocal: false, isDefinition: true, scopeLine: 164, isOptimized: true, unit: !1, variables: !3)
!134 = !DILocation(line: 156, scope: !116, inlinedAt: !118)
!135 = !DILocation(line: 165, scope: !136, inlinedAt: !137)
!136 = distinct !DISubprogram(name: "mul_fast;", linkageName: "mul_fast", scope: !114, file: !114, line: 165, type: !8, isLocal: false, isDefinition: true, scopeLine: 165, isOptimized: true, unit: !1, variables: !3)
!137 = !DILocation(line: 155, scope: !116, inlinedAt: !118)
!138 = !DILocation(line: 165, scope: !139, inlinedAt: !140)
!139 = distinct !DISubprogram(name: "mul_fast;", linkageName: "mul_fast", scope: !114, file: !114, line: 165, type: !8, isLocal: false, isDefinition: true, scopeLine: 165, isOptimized: true, unit: !1, variables: !3)
!140 = !DILocation(line: 157, scope: !116, inlinedAt: !118)
!141 = !DILocation(line: 42, scope: !142, inlinedAt: !143)
!142 = distinct !DISubprogram(name: "macro expansion;", linkageName: "macro expansion", scope: !7, file: !7, type: !8, isLocal: false, isDefinition: true, isOptimized: true, unit: !1, variables: !3)
!143 = !DILocation(line: 133, scope: !144, inlinedAt: !145)
!144 = distinct !DISubprogram(name: "@generated body;", linkageName: "@generated body", scope: !81, file: !81, type: !8, isLocal: false, isDefinition: true, isOptimized: true, unit: !1, variables: !3)
!145 = !DILocation(line: 133, scope: !146, inlinedAt: !147)
!146 = distinct !DISubprogram(name: "unsafe_store!;", linkageName: "unsafe_store!", scope: !81, file: !81, line: 133, type: !8, isLocal: false, isDefinition: true, scopeLine: 133, isOptimized: true, unit: !1, variables: !3)
!147 = !DILocation(line: 84, scope: !148, inlinedAt: !118)
!148 = distinct !DISubprogram(name: "setindex!;", linkageName: "setindex!", scope: !86, file: !86, line: 82, type: !8, isLocal: false, isDefinition: true, scopeLine: 82, isOptimized: true, unit: !1, variables: !3)
!149 = !DILocation(line: 96, scope: !19, inlinedAt: !21)
ERROR: LoadError: CUDA error: an illegal memory access was encountered (code #700, ERROR_ILLEGAL_ADDRESS)
Stacktrace:
 [1] macro expansion at /home/maleadt/Julia/CUDAdrv/src/base.jl:145 [inlined]
 [2] CuModule(::String, ::Dict{CUDAdrv.CUjit_option,Any}) at /home/maleadt/Julia/CUDAdrv/src/module.jl:36
 [3] #cufunction#94(::Base.Iterators.Pairs{Union{},Union{},Tuple{},NamedTuple{(),Tuple{}}}, ::Function, ::CuDevice, ::Function, ::Function, ::Type{T} where T<:Tuple) at /home/maleadt/Julia/CUDAnative/src/jit.jl:550
 [4] cufunction at /home/maleadt/Julia/CUDAnative/src/jit.jl:527 [inlined]
 [5] @generated body at /home/maleadt/Julia/CUDAnative/src/execution.jl:182 [inlined]
 [6] #_cuda#112(::Base.Iterators.Pairs{Symbol,Int64,Tuple{Symbol,Symbol},NamedTuple{(:blocks, :threads),Tuple{Int64,Int64}}}, ::typeof(CUDAnative._cuda), ::CUDAnative.Kernel{typeof(MixedModeBroadcastAD._cuda_broadcast_kernel!)}, ::typeof(MixedModeBroadcastAD._cuda_broadcast_kernel!), ::getfield(Main, Symbol("##16#17")), ::CuDeviceArray{Float32,2,CUDAnative.AS.Global}, ::Tuple{CuDeviceArray{Float32,2,CUDAnative.AS.Global},CuDeviceArray{Float32,2,CUDAnative.AS.Global},Float32,Float32,CuDeviceArray{Float32,1,CUDAnative.AS.Global},CuDeviceArray{Float32,1,CUDAnative.AS.Global}}, ::Tuple{Tuple{Bool,Bool},Tuple{Bool,Bool},Tuple{},Tuple{},Tuple{Bool},Tuple{Bool}}, ::Tuple{Tuple{Int64,Int64},Tuple{Int64,Int64},Tuple{},Tuple{},Tuple{Int64},Tuple{Int64}}, ::Tuple{Base.OneTo{Int64},Base.OneTo{Int64}}) at /home/maleadt/Julia/CUDAnative/src/execution.jl:137
 [7] (::getfield(CUDAnative, Symbol("#kw##_cuda")))(::NamedTuple{(:blocks, :threads),Tuple{Int64,Int64}}, ::typeof(CUDAnative._cuda), ::Function, ::Function, ::Function, ::Vararg{Any,N} where N) at ./<missing>:0
 [8] macro expansion at ./gcutils.jl:82 [inlined]
 [9] broadcast!(::Function, ::CuArray{Float32,2}, ::Nothing, ::CuArray{Float32,2}, ::CuArray{Float32,2}, ::Float32, ::Float32, ::CuArray{Float32,1}, ::CuArray{Float32,1}) at /home/maleadt/Julia/MixedModeBroadcastAD/src/MixedModeBroadcastAD.jl:77
 [10] broadcast! at ./broadcast.jl:434 [inlined]
 [11] broadcast! at ./broadcast.jl:432 [inlined]
 [12] tf_fusion1! at /home/maleadt/Julia/MixedModeBroadcastAD/test/kernels.jl:146 [inlined]
 [13] tf_hmlstm_update_c_gradients!(::Tuple{CuArray{Float32,1},CuArray{Float32,1},CuArray{Float32,2},CuArray{Float32,2},CuArray{Float32,2},CuArray{Float32,2}}, ::NTuple{4,CuArray{Float32,2}}, ::Tuple{CuArray{Float32,2},CuArray{Float32,2},CuArray{Float32,2}}) at /home/maleadt/Julia/MixedModeBroadcastAD/test/kernels.jl:111
 [14] macro expansion at /home/maleadt/Julia/NVTX/src/ranges.jl:60 [inlined]
 [15] benchmark(::typeof(tf_hmlstm_update_c_gradients!), ::Tuple{CuArray{Float32,1},CuArray{Float32,1},CuArray{Float32,2},CuArray{Float32,2},CuArray{Float32,2},CuArray{Float32,2}}, ::NTuple{4,CuArray{Float32,2}}, ::Tuple{CuArray{Float32,2},CuArray{Float32,2},CuArray{Float32,2}}) at /home/maleadt/Julia/MixedModeBroadcastAD/test/runprofile.jl:22
 [16] top-level scope at ./<missing>:148
 [17] include at ./boot.jl:314 [inlined]
 [18] include_relative(::Module, ::String) at ./loading.jl:1067
 [19] include(::Module, ::String) at ./sysimg.jl:29
 [20] exec_options(::Base.JLOptions) at ./client.jl:327
 [21] _start() at ./client.jl:455
in expression starting at /home/maleadt/Julia/MixedModeBroadcastAD/test/runprofile.jl:25
