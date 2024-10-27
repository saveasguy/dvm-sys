file(GLOB_RECURSE DVM_EXEC_VARIANTS RELATIVE ${CMAKE_CURRENT_SOURCE_DIR} *.run)
if (NOT DVM_EXEC_VARIANTS)
  message(FATAL_ERROR "No DVM schedulers are available.")
endif()

set(DVM_EXEC "" CACHE STRING "DVM job scheduler.")
set_property(CACHE DVM_EXEC PROPERTY STRINGS ${DVM_EXEC_VARIANTS})
if (NOT DVM_EXEC)
  message(SEND_ERROR "DVM job scheduler must be specified (see DVM_EXEC variable).")
endif()

add_executable(dvmdiff dvmdiff.c)
set_target_properties(dvmdiff PROPERTIES
  FOLDER "${DVM_TOOL_FOLDER}"
  RUNTIME_OUTPUT_DIRECTORY ${DVM_BIN_DIR}
  COMPILE_PDB_OUTPUT_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}/dvmdiff.dir/$<CONFIG>
  PDB_OUTPUT_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}/dvmdiff.dir/$<CONFIG>
)

add_executable(analyzePtxasCuda analyzePtxasCuda.cpp)
set_target_properties(analyzePtxasCuda PROPERTIES
  FOLDER "${DVM_TOOL_FOLDER}"
  RUNTIME_OUTPUT_DIRECTORY ${DVM_BIN_DIR}
  COMPILE_PDB_OUTPUT_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}/analyzePtxasCuda.dir/$<CONFIG>
  PDB_OUTPUT_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}/analyzePtxasCuda.dir/$<CONFIG>
)

add_executable(dvmh_conv_include dvmh_conv_include.cpp)
set_target_properties(dvmh_conv_include PROPERTIES
  FOLDER "${DVM_TOOL_FOLDER}"
  RUNTIME_OUTPUT_DIRECTORY ${DVM_BIN_DIR}
  COMPILE_PDB_OUTPUT_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}/dvmh_conv_include.dir/$<CONFIG>
  PDB_OUTPUT_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}/dvmh_conv_include.dir/$<CONFIG>
)

set(DVM_INTERNAL_SCRIPTS
  dvm_drv dvm_utils.sh dvm_help dvm_comp dvm_link dvm_ver dvm_help.txt
  # Revision needed:
  dvm_test dvm_pred dvm_doc
)

if (WIN32)
  set(DVM_INTERNAL_SCRIPTS ${DVM_INTERNAL_SCRIPTS} dvm_drv.bat setvars.bat)
  if (MSVC_IDE)
    source_group("Source Files" FILES ${DVM_INTERNAL_SCRIPTS} ${DVM_RUN} ${DVM_EXEC})
  endif()
endif()

add_custom_target(dvm-resource-driver ALL
  SOURCES ${DVM_INTERNAL_SCRIPTS} ${DVM_RUN} ${DVM_EXEC}
  WORKING_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}
  COMMAND ${CMAKE_COMMAND} -E make_directory "${DVM_BIN_DIR}"
  COMMAND ${CMAKE_COMMAND} -E copy_if_different ${DVM_INTERNAL_SCRIPTS} "${DVM_BIN_DIR}"
  COMMAND ${CMAKE_COMMAND} -E copy_if_different "${DVM_RUN}" "${DVM_BIN_DIR}/dvm_run"
  COMMAND ${CMAKE_COMMAND} -E copy_if_different "${DVM_EXEC}" "${DVM_BIN_DIR}/${DVM_EXEC}"
  COMMENT "Copy DVM driver executables to ${DVM_BIN_DIR}..."
)
add_dependencies(dvm-resource-driver dvmdiff analyzePtxasCuda dvmh_conv_include)
set_target_properties(dvm-resource-driver PROPERTIES FOLDER "${DVM_RESOURCE_FOLDER}")

if (WIN32)
  file(TO_NATIVE_PATH "${DVM_DIR}" DVM_DIR_NATIVE)
  string(FIND "${DVM_DIR}" ":" POS)
  if (POS EQUAL -1)
    message(FATAL_ERROR "Unable to convert ${DVM_DIR} to MSYS style.")
  endif()
  string(SUBSTRING ${DVM_DIR} 0 ${POS} DVM_DIR_VOLUME)
  math(EXPR POS "${POS} + 1")
  string(SUBSTRING ${DVM_DIR} ${POS} -1 DVM_DIR_MSYS)
  string(CONCAT DVM_DIR_MSYS "/" "${DVM_DIR_VOLUME}" "${DVM_DIR_MSYS}")

  if (DVM_ENABLE_OPENMP)
    set(DVMH_NUM_THREADS_IF_ENABLED "
    set \"DVMH_NUM_THREADS=${DVMH_NUM_THREADS}\" :: Number of CPU threads per process")
  else()
    set(DVMH_NUM_THREADS_IF_ENABLED "")
  endif()

  if (DVM_ENABLE_CUDA)
    set(DVM_CUDA_FLAGS_IF_ENABLED
"if [ -z \"$NVCC\" ]; then export NVCC='\"${CUDAToolkit_NVCC_EXECUTABLE}\" -use-local-env -Xcompiler=\"${DVM_CXX_COMPILE_FLAGS} ${DVM_OpenMP_CXX_FLAGS}\"'; fi
export NVCC=\"$NVCC ${DVM_Fortran_NAMING_MACROS}\"")
    file(TO_NATIVE_PATH "${CUDAToolkit_TARGET_DIR}" DVM_CUDA_PATH_NATIVE)
    set(DVM_CUDA_PATH_IF_ENABLED "set CUDA_PATH=${DVM_CUDA_PATH_NATIVE}")
    set(DVMH_NUM_CUDAS_IF_ENABLED "
    set \"DVMH_NUM_CUDAS=${DVMH_NUM_CUDAS}\" :: Number of GPUs per process")
  else()
    set(DVM_CUDA_FLAGS_IF_ENABLED "")
    set(DVM_CUDA_PATH_IF_ENABLED "")
    set(DVMH_NUM_CUDAS_IF_ENABLED "")
  endif()

  configure_file(dvm_settings.sh.win.in ${CMAKE_CURRENT_BINARY_DIR}/dvm_settings.sh.gen.in)
  file(GENERATE OUTPUT ${DVM_BIN_DIR}/dvm_settings.sh INPUT ${CMAKE_CURRENT_BINARY_DIR}/dvm_settings.sh.gen.in)

  configure_file(dvm_settings.bat.in ${CMAKE_CURRENT_BINARY_DIR}/dvm_settings.bat.gen.in)
  file(GENERATE OUTPUT ${DVM_BIN_DIR}/dvm_settings.bat INPUT ${CMAKE_CURRENT_BINARY_DIR}/dvm_settings.bat.gen.in)

  configure_file(dvm.bat.in ${CMAKE_CURRENT_BINARY_DIR}/dvm.bat.gen.in)
  file(GENERATE OUTPUT ${DVM_USER_DIR}/dvm.bat INPUT ${CMAKE_CURRENT_BINARY_DIR}/dvm.bat.gen.in)
endif()
