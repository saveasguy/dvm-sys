add_subdirectory(src)

if (MSVC_IDE)
  file(GLOB_RECURSE DVM_PARAMETERS RELATIVE ${CMAKE_CURRENT_SOURCE_DIR}
    par/*.par par/*.rel par/*.txt)
  source_group("Parameter Files" FILES ${DVM_PARAMETERS})
endif()

add_custom_target(dvm-resource-parameters ALL
  SOURCES ${DVM_PARAMETERS}
  WORKING_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}/par
  COMMAND ${CMAKE_COMMAND} -E copy_directory "${CMAKE_CURRENT_SOURCE_DIR}/par"
                                             "${DVM_PARAMETER_DIR}"
  COMMAND ${CMAKE_COMMAND} -E make_directory "${DVM_USER_DIR}"
  COMMAND ${CMAKE_COMMAND} -E copy_if_different usr.par "${DVM_USER_DIR}"
  COMMENT "Copy DVM runtime parameters to ${DVM_PARAMETER_DIR}...")
set_target_properties(dvm-resource-parameters PROPERTIES FOLDER "${DVM_RESOURCE_FOLDER}")

