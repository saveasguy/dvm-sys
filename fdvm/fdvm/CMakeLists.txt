set(FDVM_SOURCES acc.cpp acc_across.cpp acc_across_analyzer.cpp acc_analyzer.cpp
  acc_data.cpp acc_f2c.cpp acc_f2c_handlers.cpp acc_rtc.cpp acc_utilities.cpp
  aks_analyzeLoops.cpp aks_structs.cpp calls.cpp checkpoint.cpp debug.cpp
  dvm.cpp funcall.cpp help.cpp hpf.cpp io.cpp omp.cpp ompdebug.cpp parloop.cpp
  stmt.cpp)

if(MSVC_IDE)
  file(GLOB_RECURSE FDVM_HEADERS RELATIVE
    ${CMAKE_CURRENT_SOURCE_DIR} *.h)
  foreach(DIR ${DVM_FORTRAN_INCLUDE_DIRS})
    file(GLOB_RECURSE FILES RELATIVE ${CMAKE_CURRENT_SOURCE_DIR} "${DIR}/*.h")
    set(FDVM_HEADERS ${FDVM_HEADERS} ${FILES})
  endforeach()
endif()

add_executable(f_dvm ${FDVM_SOURCES} ${FDVM_HEADERS})

add_dependencies(f_dvm db sage sage++)
target_link_libraries(f_dvm db sage sage++)
target_compile_definitions(f_dvm PRIVATE SYS5)
target_include_directories(f_dvm PRIVATE "${DVM_FORTRAN_INCLUDE_DIRS}")
set_target_properties(f_dvm PROPERTIES
  FOLDER "${DVM_TOOL_FOLDER}"
  RUNTIME_OUTPUT_DIRECTORY ${DVM_BIN_DIR}
  COMPILE_PDB_OUTPUT_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}/$<CONFIG>
  PDB_OUTPUT_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}/$<CONFIG>
)
