set(SAGEP_SOURCES libSage++.cpp)

if(MSVC_IDE)
  foreach(DIR ${DVM_SAGE_INCLUDE_DIRS})
    file(GLOB_RECURSE FILES RELATIVE ${CMAKE_CURRENT_SOURCE_DIR}
      "${DIR}/*.h" "${DIR}/*.def" "${DIR}/head" "${DIR}/tag")
    set(SAGEP_HEADERS ${SAGEP_HEADERS} ${FILES})
  endforeach()
  source_group("Header Files" FILES ${SAGEP_HEADERS})
endif()
add_library(sage++ ${SAGEP_SOURCES} ${SAGEP_HEADERS})

target_include_directories(sage++ PUBLIC "${DVM_SAGE_INCLUDE_DIRS}")
set_target_properties(sage++ PROPERTIES FOLDER "${DVM_LIBRARY_FOLDER}")
