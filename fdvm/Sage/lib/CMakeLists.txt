set(DVM_SAGE_INCLUDE_DIRS ${DVM_SAGE_INCLUDE_DIRS}
  ${CMAKE_CURRENT_SOURCE_DIR}/include)
set(DVM_SAGE_INCLUDE_DIRS ${DVM_SAGE_INCLUDE_DIRS} PARENT_SCOPE)

add_subdirectory(newsrc)
add_subdirectory(oldsrc)