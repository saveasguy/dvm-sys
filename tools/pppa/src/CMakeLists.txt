configure_file(dvmvers.h.in dvmvers.h)

set(PPPA_SOURCES inter.cpp potensyn.cpp statfile.cpp statread.cpp treeinter.cpp
  statprintf.cpp)

if(MSVC_IDE)
  file(GLOB_RECURSE PPPA_HEADERS RELATIVE ${CMAKE_CURRENT_SOURCE_DIR} *.h)
endif()

add_executable(dvmstf ${PPPA_SOURCES} ${PPPA_HEADERS})

add_dependencies(dvmstf zlib)
target_link_libraries(dvmstf zlib)
target_include_directories(dvmstf PRIVATE ${CMAKE_CURRENT_BINARY_DIR})
set_target_properties(dvmstf PROPERTIES
  FOLDER "${DVM_TOOL_FOLDER}"
  RUNTIME_OUTPUT_DIRECTORY ${DVM_BIN_DIR}
  COMPILE_PDB_OUTPUT_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}/$<CONFIG>
  PDB_OUTPUT_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}/$<CONFIG>
)
