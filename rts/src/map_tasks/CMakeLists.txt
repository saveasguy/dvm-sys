set(MAP_TASKS_SOURCES Distribution.cpp NaiveDistribution.cpp ProcessorLoad.cpp
  Task.cpp TaskData.cpp TransposedDistribution.cpp Muzzle.cpp)

if(MSVC_IDE)
  file(GLOB_RECURSE MAP_TASKS_INTERNAL_HEADERS RELATIVE
    ${CMAKE_CURRENT_SOURCE_DIR} *.h)
endif()

add_library(map_tasks ${MAP_TASKS_SOURCES} ${MAP_TASKS_INTERNAL_HEADERS})

string(REPLACE " " ";" COMPILE_OPTIONS "${DVM_CXX_COMPILE_FLAGS} ${DVM_OpenMP_CXX_FLAGS}")
set_target_properties(map_tasks PROPERTIES
  FOLDER "${DVM_LIBRARY_FOLDER}"
  ARCHIVE_OUTPUT_DIRECTORY ${DVM_ARCHIVE_DIR}
  COMPILE_OPTIONS "${COMPILE_OPTIONS}"
)
