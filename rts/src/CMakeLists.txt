configure_file(../include/dvmvers.h.in dvmvers.h)

if (WIN32)
  set(EXT .cpp)
  set(DVM_DVMLIB_COMPILE_FLAGS "${DVM_CXX_COMPILE_FLAGS} ${DVM_OpenMP_CXX_FLAGS}")
else()
  set(EXT .c)
  set(DVM_DVMLIB_COMPILE_FLAGS "${DVM_C_COMPILE_FLAGS} ${DVM_OpenMP_C_FLAGS}")
endif()

set(DVMLIB_SOURCES _adler32${EXT} _compress${EXT} _crc32${EXT} _deflate${EXT}
  _gzio${EXT} _infblock${EXT} _infcodes${EXT} _inffast${EXT} _inflate${EXT}
  _inftrees${EXT} _infutil${EXT} _trees${EXT} _uncompr${EXT} _zutil${EXT}
  1${EXT} 2${EXT} 3${EXT} 4${EXT} 5${EXT} 6${EXT} 7${EXT} 8${EXT} 9${EXT}
  linitstd${EXT})

if(MSVC_IDE)
  file(GLOB_RECURSE DVMLIB_INTERNAL_HEADERS RELATIVE
    ${CMAKE_CURRENT_SOURCE_DIR} *.h)
  file(GLOB_RECURSE DVMLIB_INCLUDE RELATIVE
    ${CMAKE_CURRENT_SOURCE_DIR} ../include/*.h)
  source_group(include FILES ${DVMLIB_INCLUDE})
endif()

add_library(dvm ${DVMLIB_SOURCES} ${DVMLIB_INTERNAL_HEADERS} ${DVMLIB_INCLUDE})
add_dependencies(dvm zlib)

target_include_directories(dvm PRIVATE ../include
  ${CMAKE_CURRENT_BINARY_DIR} ${MPI_CXX_INCLUDE_DIRS})
target_link_libraries(dvm PRIVATE zlib)
target_compile_definitions(dvm PRIVATE _F_TIME_ ${DVM_Fortran_NAMING_MACROS})
if (WIN32)
  set(MPSOPT_DEFINITIONS _WIN_MPI_ _NT_MPI_ _MPI_PROF_EXT_ _DVM_STAT_
    MPICH_IGNORE_CXX_SEEK _COPY_FILE_NAME_ _DVM_MPI_ _DVM_MSC_)
  target_compile_definitions(dvm PRIVATE ${MPSOPT_DEFINITIONS})
  if (MSVC)
    target_compile_options(dvm PRIVATE /EHsc)
    target_compile_definitions(dvm PRIVATE _CRT_SECURE_NO_WARNINGS)
  endif()
endif()
string(REPLACE " " ";" COMPILE_OPTIONS ${DVM_DVMLIB_COMPILE_FLAGS})
set_target_properties(dvm PROPERTIES
  FOLDER "${DVM_LIBRARY_FOLDER}"
  ARCHIVE_OUTPUT_DIRECTORY ${DVM_ARCHIVE_DIR}
  COMPILE_OPTIONS "${COMPILE_OPTIONS}"
)

add_custom_command(TARGET dvm POST_BUILD
  WORKING_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}
  COMMAND ${CMAKE_COMMAND} -E copy_directory ../include ${DVM_INCLUDE_DIR}
  COMMAND ${CMAKE_COMMAND} -E copy_if_different ${CMAKE_CURRENT_BINARY_DIR}/dvmvers.h ${DVM_INCLUDE_DIR}
  COMMENT "Copy DVM runtime headers to ${DVM_INCLUDE_DIR}...")

if (DVM_USE_Fortran_WRAPPER)
  add_library(fwrapper getarg.f nargs.f clfdvm.f)
  string(REPLACE " " ";" COMPILE_OPTIONS "${DVM_Fortran_COMPILE_FLAGS} ${DVM_OpenMP_Fortran_FLAGS}")
  set_target_properties(fwrapper PROPERTIES
    FOLDER "${DVM_LIBRARY_FOLDER}"
    COMPILE_OPTIONS "${COMPILE_OPTIONS}"
  )
  add_dependencies(dvm fwrapper)
  set_target_properties(dvm PROPERTIES
    STATIC_LIBRARY_OPTIONS "${MPI_CXX_LIBRARIES};$<TARGET_LINKER_FILE:fwrapper>")
endif()

add_subdirectory(map_tasks)
