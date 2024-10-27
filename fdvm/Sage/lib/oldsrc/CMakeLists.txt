set(DB_SOURCES anal_ind.c db.c db_unp.c db_unp_vpc.c dbutils.c
  garb_coll.c glob_anal.c ker_fun.c list.c make_nodes.c mod_ref.c ndeps.c
  readnodes.c sets.c setutils.c symb_alg.c writenodes.c)

if(MSVC_IDE)
  foreach(DIR ${DVM_SAGE_INCLUDE_DIRS})
    file(GLOB_RECURSE FILES RELATIVE ${CMAKE_CURRENT_SOURCE_DIR}
      "${DIR}/*.h" "${DIR}/*.def" "${DIR}/head" "${DIR}/tag")
    set(DB_HEADERS ${DB_HEADERS} ${FILES})
  endforeach()
  source_group("Header Files" FILES ${DB_HEADERS})
endif()

add_library(db ${DB_SOURCES} ${DB_HEADERS})

target_compile_definitions(db PRIVATE SYS5)
target_include_directories(db PUBLIC "${DVM_SAGE_INCLUDE_DIRS}")
set_target_properties(db PROPERTIES FOLDER "${DVM_LIBRARY_FOLDER}")
