##################################################
# Hetero-cc routines
##################################################
set(HPVM_DECLS_FILE /home/xiaoboh2/hpvm/hpvm/build/tools/hpvm/projects/hetero-c++/lib/HPVMCFunctionDeclarations/HPVMCFunctionDeclarations.bc)
set(HPVM_BUILD /home/xiaoboh2/hpvm/hpvm/build/)
set(HETERO_DC_LIB /home/xiaoboh2/hpvm/hpvm/projects/guided_deep_copy/src/)

list(APPEND HPVM_INCLUDE
    ${HETERO_DC_LIB} # The run-time DC lib must be at the front of EVERYTHING
    ${HPVM_BUILD}/include
    ${HPVM_BUILD}/tools/hpvm
    ${HPVM_BUILD}/tools/hpvm/projects/hpvm-tensor-rt/tensor_runtime/include
    /home/xiaoboh2/hpvm/hpvm/benchmarks/include
    /home/xiaoboh2/hpvm/hpvm/llvm/include
    /home/xiaoboh2/hpvm/hpvm/llvm/tools/hpvm/./include
    /home/xiaoboh2/hpvm/hpvm/llvm/tools/hpvm/projects/hpvm-tensor-rt/./tensor_runtime/include
    /usr/local/cuda/targets/x86_64-linux/include
    /usr/include
)
list(TRANSFORM HPVM_INCLUDE PREPEND -I)

function(hetero_compile_obj source_files hetero_obj)
  # First compile to hetero-cc LLs
  foreach(src ${source_files})
      get_filename_component(src_name ${src} NAME)
      MESSAGE(STATUS "Compile as Hetero-cc: ${src_name}")
      add_custom_command(
        OUTPUT ${src_name}.hetero.ll
        WORKING_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}

        #COMMAND clang++-9 -emit-llvm -c ${src} ${PROJECT_INCLUDES} -o ${src_name}.hetero.ll # Regular compiler for debug purpose

        COMMAND ${HPVM_BUILD}/bin/clang++ -O1 ${HPVM_INCLUDE} ${PROJECT_INCLUDES} -std=c++17 -DDEVICE=CPU_TARGET -emit-llvm -fno-exceptions -Xclang -disable-lifetime-markers -E ${src} -o ${src_name}.deepcopy.cc 
        #COMMAND ${HPVM_BUILD}/bin/hetero-dc ${src_name}.deepcopy.cc # deep copy disabled for now, nothing use it
        COMMAND ${HPVM_BUILD}/bin/clang++ -O1 ${HPVM_INCLUDE} ${PROJECT_INCLUDES} -std=c++17 -DDEVICE=CPU_TARGET -emit-llvm -fno-exceptions -Xclang -disable-lifetime-markers -S ${src_name}.deepcopy.cc -o ${src_name}.hetero.ll
        DEPENDS ${src}
        )
      list(APPEND hetero_ll_files ${src_name}.hetero.ll)
  endforeach()

  # Link heterocc ll modules together
  set(combined_name hetero_combined)
  add_custom_command(
      OUTPUT ${combined_name}.hetero.ll
      WORKING_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}
      COMMAND ${HPVM_BUILD}/bin/llvm-link -S ${hetero_ll_files} -o ${combined_name}.hetero.ll
      DEPENDS ${hetero_ll_files}
    )

  # Then lower to hpvm-c -> lower to llvm host -> lower to object code
  add_custom_command(
    OUTPUT ${hetero_obj}
    WORKING_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}
    COMMAND ${HPVM_BUILD}/bin/hcc -declsfile ${HPVM_DECLS_FILE} -sanitise-funcs -S ${combined_name}.hetero.ll -o ${combined_name}.ll
    COMMAND ${HPVM_BUILD}/bin/opt -enable-new-pm=0 -load HPVMGenHPVM.so -genhpvm -globaldce -S ${combined_name}.ll -o ${combined_name}.hpvm.ll -dot-callgraph
    COMMAND ${HPVM_BUILD}/bin/opt -enable-new-pm=0 -load HPVMBuildDFG.so -load HPVMLocalMem.so -load HPVMDFG2LLVM_CPU.so -load HPVMClearDFG.so -buildDFG -localmem -dfg2llvm-cpu -clearDFG -S ${combined_name}.hpvm.ll -o ${combined_name}.hpvm.ll
    COMMAND ${HPVM_BUILD}/bin/llvm-link ${HPVM_BUILD}/tools/hpvm/projects/hpvm-rt/hpvm-rt.bc -S ${combined_name}.hpvm.ll -o ${combined_name}.linked.bc
    COMMAND ${HPVM_BUILD}/bin/clang++ -c ${combined_name}.linked.bc -o ${hetero_obj} -fPIC
    DEPENDS ${combined_name}.hetero.ll
    )
endfunction()


##################################################
# Make hetero-cc objects
##################################################
#list(APPEND hetero_source_files # everything involved from __hpvm__init
    #state/StateHelper_hetero.cpp
    #update/UpdaterSLAM.cpp
    #update/UpdaterMSCKF.cpp
    #core/VioManager.cpp
    #state/Propagator.cpp
    #run_illixr_msckf_hetero.cpp
#)
#list(TRANSFORM hetero_source_files PREPEND ${CMAKE_CURRENT_SOURCE_DIR}/src/)

#set(hetero_obj ${CMAKE_CURRENT_BINARY_DIR}/hetero_archive.o)

#get_target_property(PROJECT_INCLUDS ov_msckf_lib INCLUDE_DIRECTORIES)
#list(TRANSFORM PROJECT_INCLUDS PREPEND -I)

#hetero_compile_obj("${hetero_source_files}" ${hetero_obj})

#add_library(HETERO_DEP OBJECT ${hetero_obj}) # dummy dependence lib

#add_library(HETERO_OBJS OBJECT IMPORTED DEPENDS ${hetero_obj})
#set_property(TARGET HETERO_OBJS PROPERTY 
    #IMPORTED_OBJECTS ${hetero_obj}
#)

