set(PARSE_SOURCES cftn.c errors.c gram1.tab.c hash.c init.c lexfdvm.c lists.c
  low_hpf.c misc.c stat.c sym.c types.c unparse_hpf.c)

set_source_files_properties(gram1.tab.c PROPERTIES COMPILE_DEFINITIONS YYDEBUG)

if(MSVC_IDE)
  file(GLOB_RECURSE PARSE_HEADERS RELATIVE
    ${CMAKE_CURRENT_SOURCE_DIR} *.h)
  set(PARSE_HEADERS ${PARSE_HEADERS} head tag tokens)
  source_group("Header Files" FILES ${PARSE_HEADERS})
  foreach(DIR ${DVM_FORTRAN_INCLUDE_DIRS})
    file(GLOB_RECURSE FILES RELATIVE ${CMAKE_CURRENT_SOURCE_DIR} "${DIR}/*.h")
    set(PARSE_HEADERS ${PARSE_HEADERS} ${FILES})
  endforeach()
  file(GLOB_RECURSE PARSE_GRAMMERS RELATIVE
    ${CMAKE_CURRENT_SOURCE_DIR} *.gram *.y)
  source_group("Grammer Files" FILES ${PARSE_GRAMMERS})
endif()

add_executable(parse ${PARSE_SOURCES} ${PARSE_HEADERS} ${PARSE_GRAMMERS})

add_dependencies(parse db sage sage++)
target_link_libraries(parse db sage sage++)
target_compile_definitions(parse PRIVATE SYS5)
if (WIN32)
  target_compile_definitions(parse PRIVATE WIN)
endif()
target_include_directories(parse PRIVATE "${DVM_FORTRAN_INCLUDE_DIRS}")
set_target_properties(parse PROPERTIES FOLDER "${DVM_TOOL_FOLDER}"
  RUNTIME_OUTPUT_DIRECTORY ${DVM_BIN_DIR}
  COMPILE_PDB_OUTPUT_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}/$<CONFIG>
  PDB_OUTPUT_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}/$<CONFIG>
)
