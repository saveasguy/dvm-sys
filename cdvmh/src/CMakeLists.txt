include_directories("${CMAKE_CURRENT_BINARY_DIR}")

set(CDVMH_SOURCES across.cpp aux_visitors.cpp c_dvmh.cpp cdvmh_log.cpp
  converter_aux.cpp converter_debug.cpp converter_main.cpp file_ctx.cpp
  handler_gen.cpp pass_ctx.cpp pragma_parser.cpp pragmas.cpp
  project_ctx.cpp utils.cpp external_preprocessor.cpp)
if(MSVC_IDE)
  file(GLOB_RECURSE CDVMH_INTERNAL_HEADERS RELATIVE
    ${CMAKE_CURRENT_SOURCE_DIR} *.h)
  file(GLOB_RECURSE CDVMH_INCLUDE RELATIVE ${CMAKE_CURRENT_SOURCE_DIR}
    "../include/*.h")
  source_group(include FILES ${CDVMH_INCLUDE})
endif()

add_executable(c_dvmh ${CDVMH_SOURCES} ${CDVMH_INTERNAL_HEADERS} ${CDVMH_INCLUDE})
target_link_libraries(c_dvmh ${CLANG_LIBS} ${LLVM_LIBS} ${SYSTEM_LIBS})
target_compile_options(c_dvmh PRIVATE ${CDVMH_MSVC_RUNTIME_FLAGS})

if (CDVMH_IN_DVM_PROJECT)
  set_target_properties(c_dvmh PROPERTIES
    RUNTIME_OUTPUT_DIRECTORY ${DVM_BIN_DIR}
    COMPILE_PDB_OUTPUT_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}/$<CONFIG>
    PDB_OUTPUT_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}/$<CONFIG>
  )
  add_custom_command(TARGET c_dvmh POST_BUILD
    WORKING_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}
    COMMAND ${CMAKE_COMMAND} -E copy_directory ../include "${DVM_INCLUDE_DIR}"
    COMMENT "Copy CDVMH headers to ${DVM_INCLUDE_DIR}...\n"
  )
else()
  install(TARGETS c_dvmh RUNTIME DESTINATION bin)
endif()

if (MSVC)
  set_target_properties(c_dvmh PROPERTIES FOLDER "${DVM_TOOL_FOLDER}")
endif()

