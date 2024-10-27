set(SAGE_SOURCES low_level.c unparse.c)

if(MSVC_IDE)
  foreach(DIR ${DVM_SAGE_INCLUDE_DIRS})
    file(GLOB_RECURSE FILES RELATIVE ${CMAKE_CURRENT_SOURCE_DIR}
      "${DIR}/*.h" "${DIR}/*.def" "${DIR}/head" "${DIR}/tag")
    set(SAGE_HEADERS ${SAGE_HEADERS} ${FILES})
  endforeach()
  source_group("Header Files" FILES ${SAGE_HEADERS})
endif()

add_library(sage ${SAGE_SOURCES} ${SAGE_HEADERS})

target_compile_definitions(sage PRIVATE SYS5)
target_include_directories(sage PUBLIC "${DVM_SAGE_INCLUDE_DIRS}")
set_target_properties(sage PROPERTIES FOLDER "${DVM_LIBRARY_FOLDER}")
