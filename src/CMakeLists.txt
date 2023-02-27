include(${CMAKE_CURRENT_SOURCE_DIR}/../cmake/hpvm.cmake)

# Collect other dependencies
set(THREADS_PREFER_PTHREAD_FLAG ON)
find_package(Threads REQUIRED)
find_package(OpenCL REQUIRED)

list(APPEND thirdparty_libraries
    pthread
    OpenCL::OpenCL
    )

##################################################
# Make hetero-cc objects
##################################################
list(APPEND hetero_source_files # everything involved from __hpvm__init
    run_onnx.cpp
    )
list(TRANSFORM hetero_source_files PREPEND ${CMAKE_CURRENT_SOURCE_DIR}/)

set(hetero_obj ${CMAKE_CURRENT_BINARY_DIR}/hetero_archive.o)

#get_target_property(MSCKF_INCLUDES ov_msckf_lib INCLUDE_DIRECTORIES)
#list(TRANSFORM MSCKF_INCLUDES PREPEND -I)
get_property(PROJECT_INCLUDES DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR} PROPERTY INCLUDE_DIRECTORIES)
list(TRANSFORM PROJECT_INCLUDES PREPEND -I)

get_property(dirs DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR} PROPERTY INCLUDE_DIRECTORIES)
foreach(dir ${dirs})
    message(STATUS "dir='${dir}'")
endforeach()

hetero_compile_obj("${hetero_source_files}" ${hetero_obj})

add_library(HETERO_DEP OBJECT ${hetero_obj}) # dummy dependence lib

add_library(HETERO_OBJS OBJECT IMPORTED DEPENDS ${hetero_obj})
set_property(TARGET HETERO_OBJS PROPERTY
    IMPORTED_OBJECTS ${hetero_obj}
    )


add_library(model_inference_lib SHARED model.cpp)
target_link_libraries(model_inference_lib PRIVATE onnxruntime)

add_executable(run_onnx /dev/null)
set_target_properties(run_onnx PROPERTIES LINKER_LANGUAGE CXX)
target_link_libraries(run_onnx PRIVATE $<TARGET_OBJECTS:HETERO_OBJS>)
target_link_libraries(run_onnx PRIVATE model_inference_lib)
target_link_libraries(run_onnx PRIVATE ${thirdparty_libraries})
add_dependencies(run_onnx HETERO_DEP)

