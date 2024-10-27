set(INLINE_SOURCES inl_exp.cpp inliner.cpp hlp.cpp)

if(MSVC_IDE)
  file(GLOB_RECURSE INLINE_HEADERS RELATIVE
    ${CMAKE_CURRENT_SOURCE_DIR} *.h)
  foreach(DIR ${DVM_FORTRAN_INCLUDE_DIRS})
    file(GLOB_RECURSE FILES RELATIVE ${CMAKE_CURRENT_SOURCE_DIR} "${DIR}/*.h")
    set(INLINE_HEADERS ${INLINE_HEADERS} ${FILES})
  endforeach()
endif()

add_executable(inl_exp ${INLINE_SOURCES} ${INLINE_HEADERS})

add_dependencies(inl_exp db sage sage++)
target_link_libraries(inl_exp db sage sage++)

target_include_directories(inl_exp PRIVATE "${DVM_FORTRAN_INCLUDE_DIRS}")
set_target_properties(inl_exp PROPERTIES
  FOLDER "${DVM_TOOL_FOLDER}"
  RUNTIME_OUTPUT_DIRECTORY ${DVM_BIN_DIR}
  COMPILE_PDB_OUTPUT_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}/$<CONFIG>
  PDB_OUTPUT_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}/$<CONFIG>
)
