add_custom_target(dvm-resource-dynamic-include
  WORKING_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}
  DEPENDS ${CMAKE_CURRENT_SOURCE_DIR}/include/dvmhlib_f2c.h
          ${CMAKE_CURRENT_SOURCE_DIR}/include/dvmhlib_block_red.h
          ${CMAKE_CURRENT_SOURCE_DIR}/include/dvmhlib_warp_red.h
          ${CMAKE_CURRENT_SOURCE_DIR}/include/dvmhlib_device.h
  COMMAND dvmh_conv_include "static const std::string dvmhlib_f2c = " <${CMAKE_CURRENT_SOURCE_DIR}/include/dvmhlib_f2c.h >dynamic_include.h
  COMMAND dvmh_conv_include "static const std::string dvmhlib_block_red = " <${CMAKE_CURRENT_SOURCE_DIR}/include/dvmhlib_block_red.h >>dynamic_include.h
  COMMAND dvmh_conv_include "static const std::string dvmhlib_warp_red = " <${CMAKE_CURRENT_SOURCE_DIR}/include/dvmhlib_warp_red.h >>dynamic_include.h
  COMMAND dvmh_conv_include "static const std::string dvmhlib_device = " <${CMAKE_CURRENT_SOURCE_DIR}/include/dvmhlib_device.h >>dynamic_include.h
  COMMAND ${CMAKE_COMMAND} -E echo "static const std::string pgi_include;" >>dynamic_include.h
  COMMENT "Generate ${CMAKE_CURRENT_BINARY_DIR}/dynamic_include.h..."
)
add_dependencies(dvm-resource-dynamic-include dvmh_conv_include)
set_target_properties(dvm-resource-dynamic-include PROPERTIES
  FOLDER "${DVM_RESOURCE_FOLDER}")

set(DVMH_CUDA_SOURCES
  cuda_copying.cu cuda_device.cu cuda_reduction.cu cuda_transform.cu)

set(DVMH_HOST_SOURCES
  distrib.cpp dvmh.cpp dvmh2.cpp dvmh_async.cpp dvmh_buffer.cpp dvmh_copying.cpp
  dvmh_data.cpp dvmh_debug.cpp dvmh_device.cpp dvmh_log.cpp dvmh_pieces.cpp
  dvmh_predictor.cpp dvmh_rtc.cpp dvmh_rts.cpp dvmh_stat.cpp dvmh_stdio.cpp
  dvmh_types.cpp dvmlib_adapter.cpp loop.cpp loop_distribution.cpp mps.cpp
  region.cpp settings.cpp util.cpp)

if(MSVC_IDE)
  file(GLOB_RECURSE DVMH_INTERNAL_HEADERS RELATIVE
    ${CMAKE_CURRENT_SOURCE_DIR} *.h)
  file(GLOB_RECURSE DVMH_INCLUDE RELATIVE ${CMAKE_CURRENT_SOURCE_DIR}
    "include/*.h")
  source_group(include FILES ${DVMH_INCLUDE})
  file(GLOB_RECURSE DVMH_WIN_HEADERS RELATIVE ${CMAKE_CURRENT_SOURCE_DIR}
    "Win/*.h")
  source_group(Win FILES ${DVMH_WIN_HEADERS})
endif()

if (NOT DVM_ENABLE_CUDA)
  set_source_files_properties(${DVMH_CUDA_SOURCES} PROPERTIES LANGUAGE CXX)
endif()

add_library(dvmh ${DVMH_CUDA_SOURCES} ${DVMH_HOST_SOURCES}
  ${DVMH_INTERNAL_HEADERS} ${DVMH_INCLUDE} ${DVMH_WIN_HEADERS})

set(DVM_HOST_CXX_FLAGS "${DVM_CXX_COMPILE_FLAGS} ${DVM_OpenMP_CXX_FLAGS}")
if (DVM_HOST_CXX_FLAGS)
  string(REPLACE " " ";" COMPILE_OPTIONS "${DVM_HOST_CXX_FLAGS}")
  target_compile_options(dvmh PRIVATE
    $<$<COMPILE_LANGUAGE:CXX>:${COMPILE_OPTIONS}>
    $<$<COMPILE_LANGUAGE:CUDA>:-Xcompiler=${DVM_HOST_CXX_OPTIONS}>
  )
endif()
if (MSVC)
  target_compile_options(dvmh PRIVATE $<$<COMPILE_LANGUAGE:CXX>:/EHsc>)
endif()

if (DVM_CUDA_FLAGS)
  string(REPLACE " " ";" COMPILE_OPTIONS "${DVM_CUDA_FLAGS}")
  target_compile_options(dvmh PRIVATE $<$<COMPILE_LANGUAGE:CUDA>:${COMPILE_OPTIONS}>)
endif()

add_dependencies(dvmh dvm zlib dvm-resource-dynamic-include)
target_link_libraries(dvmh PRIVATE zlib)
target_include_directories(dvmh PRIVATE
  ${DVM_INCLUDE_DIR} ${CMAKE_CURRENT_BINARY_DIR} ${MPI_CXX_INCLUDE_DIRS})
target_compile_definitions(dvmh PRIVATE _DVM_MPI_ ${DVM_Fortran_NAMING_MACROS})
if (DVM_ENABLE_CUDA)
  target_compile_definitions(dvmh PRIVATE HAVE_CUDA)
  set(DVM_CUDA_LIBS ${CUDA_cudart_LIBRARY})
  if (TARGET CUDA::nvrtc)
    target_compile_definitions(dvmh PRIVATE HAVE_NVRTC)
    list(APPEND DVM_CUDA_LIBS ${CUDA_cuda_driver_LIBRARY} ${CUDA_nvrtc_LIBRARY})
  endif()
endif()
if (WIN32)
  target_include_directories(dvmh PRIVATE Win/include)
  set_target_properties(dvmh PROPERTIES
    STATIC_LIBRARY_OPTIONS "${CMAKE_CURRENT_SOURCE_DIR}/Win/lib/${DVM_MACHINE_STRING}/pthreadVC2.lib;${DVM_CUDA_LIBS}")
endif()
if (MSVC)
  target_compile_definitions(dvmh PRIVATE _CRT_SECURE_NO_WARNINGS)
endif()
set_target_properties(dvmh PROPERTIES
  FOLDER "${DVM_LIBRARY_FOLDER}"
  ARCHIVE_OUTPUT_DIRECTORY ${DVM_ARCHIVE_DIR}
)

add_custom_command(TARGET dvmh POST_BUILD
  WORKING_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}
  COMMAND ${CMAKE_COMMAND} -E copy_directory include "${DVM_INCLUDE_DIR}"
  COMMENT "Copy DVMH runtime headers to ${DVM_INCLUDE_DIR}...\n")

if (WIN32)
  add_custom_target(dvm-resource-pthread ALL
    SOURCES ${CMAKE_CURRENT_SOURCE_DIR}/Win/dll/${DVM_MACHINE_STRING}/pthreadVC2.dll
    WORKING_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}/Win/dll/${DVM_MACHINE_STRING}
    COMMAND ${CMAKE_COMMAND} -E make_directory "${DVM_BIN_DIR}"
    COMMAND ${CMAKE_COMMAND} -E copy_if_different pthreadVC2.dll "${DVM_BIN_DIR}"
    COMMENT "Copy PThread dynamic library to ${DVM_BIN_DIR}..."
  )
  set_target_properties(dvm-resource-pthread PROPERTIES FOLDER "${DVM_RESOURCE_FOLDER}")
endif()
