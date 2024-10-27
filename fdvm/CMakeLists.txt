set(DVM_FORTRAN_INCLUDE_DIRS ${CMAKE_CURRENT_SOURCE_DIR}/include)

add_subdirectory(Sage)
add_subdirectory(parser)
add_subdirectory(fdvm)
add_subdirectory(InlineExpansion)

