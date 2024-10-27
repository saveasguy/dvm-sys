set(ZLIB_SOURCES adler32.c compress.c crc32.c gzio.c uncompr.c deflate.c
 trees.c zutil.c inflate.c infblock.c inftrees.c infcodes.c infutil.c inffast.c)

if(MSVC_IDE)
  file(GLOB_RECURSE FILES RELATIVE ${CMAKE_CURRENT_SOURCE_DIR} *.h ../include/*.h)
endif()

add_library(zlib ${ZLIB_SOURCES} ${ZLIB_HEADERS})

target_include_directories(zlib
  PUBLIC ${CMAKE_INCLUDE_CURRENT_DIR}../include ${CMAKE_CURRENT_SOURCE_DIR})
set_target_properties(zlib PROPERTIES FOLDER "${DVM_LIBRARY_FOLDER}")
