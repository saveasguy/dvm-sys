cmake_minimum_required(VERSION 3.4.3)

project(CDVMH VERSION 0.0.1 LANGUAGES C CXX)

set(LLVM_VERSION CACHE STRING "Version of installed LLVM package")
set(DVM_TOOL_FOLDER "" CACHE STRING "Specify IDE folder for DVM executables.")

set(CMAKE_MODULE_PATH "${PROJECT_SOURCE_DIR}/cmake")
include(CheckCXXCompilerFlag)
include(utility)

string(REGEX REPLACE "[A-Za-z].*" "" LLVM_VERSION_TO_SEARCH "${LLVM_VERSION}")

find_package(LLVM ${LLVM_VERSION_TO_SEARCH} REQUIRED CONFIG)
# TODO (kaniandr@gmail.com): use targets exported from LLVM and Clang packages
# instead of manually specified libraries.

string(REGEX REPLACE "[A-Za-z].*" "" LLVM_VERSION_TO_SEARCH ${LLVM_PACKAGE_VERSION})

if (${LLVM_VERSION_TO_SEARCH} VERSION_LESS "3.4" OR
    ${LLVM_VERSION_TO_SEARCH} VERSION_GREATER_EQUAL "18.0" )
  message(FATAL_ERROR "Unsupported LLVM version ${LLVM_PACKAGE_VERSION}, the only supported versions are in range [3.4, 18.x]")
endif()

if (MSVC AND (NOT "${CMAKE_PROJECT_NAME}" STREQUAL SAPFOR OR PACKAGE_LLVM))
  if (TARGET LLVMCore)
    get_target_property(CDVMH_LLVM_CONFIG LLVMCore IMPORTED_CONFIGURATIONS)
    list(LENGTH CDVMH_LLVM_CONFIG CDVMH_LLVM_CONFIG_NUM)
    if (${CDVMH_LLVM_CONFIG_NUM} EQUAL 1)
      if(LLVM_USE_CRT_${CDVMH_LLVM_CONFIG})
        unset(CDVMH_MSVC_RUNTIME_FLAGS CACHE)
        set(CDVMH_MSVC_RUNTIME_FLAGS "/${LLVM_USE_CRT_${CDVMH_LLVM_CONFIG}}")
      endif()
    endif()
  endif()
  if (NOT CDVMH_MSVC_RUNTIME_FLAGS)
    set(CDVMH_MSVC_RUNTIME_FLAGS "" CACHE STRING "The MSVC runtime library to build CDVMH converter.")
  endif()
  if (NOT CDVMH_MSVC_RUNTIME_FLAGS)
    message(SEND_ERROR "The MSVC runtime library must be specified (see CDVMH_MSVC_RUNTIME_FLAGS variable)")
  endif()
endif()

if ("${CMAKE_PROJECT_NAME}" STREQUAL SAPFOR)
  if (NOT PACKAGE_LLVM)
    set(CLANG_SOURCE_DIR "${LLVM_PROJECT_DIR}/clang")
    set(CLANG_BINARY_DIR "${LLVM_BINARY_DIR}/tools/clang")
    set(LLVM_INCLUDE_DIRS ${LLVM_INCLUDE_DIRS}
      ${CLANG_SOURCE_DIR}/include ${CLANG_BINARY_DIR}/include)
  endif()
elseif ("${CMAKE_PROJECT_NAME}" STREQUAL "DVM")
  set(CDVMH_IN_DVM_PROJECT TRUE)
endif()


message(STATUS "LLVM version: ${LLVM_PACKAGE_VERSION}")
message(STATUS "LLVM include directories: ${LLVM_INCLUDE_DIRS}")
message(STATUS "LLVM library directories: ${LLVM_LIBRARY_DIRS}")

set(LLVM_COMPONENTS bitreader core option bitreader mcparser transformutils)
if (${LLVM_PACKAGE_VERSION} VERSION_GREATER_EQUAL "10.0")
  set(LLVM_COMPONENTS ${LLVM_COMPONENTS} frontendopenmp)
endif()
if (${LLVM_PACKAGE_VERSION} VERSION_GREATER_EQUAL "15.0")
  set(LLVM_COMPONENTS ${LLVM_COMPONENTS} windowsdriver)
endif()

if(${LLVM_PACKAGE_VERSION} VERSION_LESS "3.5")
  llvm_map_components_to_libraries(LLVM_LIBS ${LLVM_COMPONENTS})
else()
  llvm_map_components_to_libnames(LLVM_LIBS ${LLVM_COMPONENTS})
endif()

set(CLANG_LIBS clangTooling clangCodeGen clangFrontend clangDriver
  clangSerialization clangParse clangSema clangAnalysis clangEdit clangAST
  clangLex clangBasic)


if (${LLVM_PACKAGE_VERSION} VERSION_EQUAL "3.4")
  set(CLANG_LIBS ${CLANG_LIBS} clangRewriteCore)
else()
  set(CLANG_LIBS ${CLANG_LIBS} clangRewrite)
  if (${LLVM_PACKAGE_VERSION} VERSION_GREATER_EQUAL "15.0")
    set(CLANG_LIBS ${CLANG_LIBS} clangSupport)
  endif()
  if (${LLVM_PACKAGE_VERSION} VERSION_GREATER_EQUAL "16")
    set(CLANG_LIBS ${CLANG_LIBS} clangASTMatchers)
  endif()
  if (MSVC AND ${LLVM_PACKAGE_VERSION} VERSION_GREATER_EQUAL "3.9")
    set(SYSTEM_LIBS version.lib)
  endif()
endif()

if(CMAKE_COMPILER_IS_GNUCXX)
  set(CDVMH_COMPILER_IS_GCC_COMPATIBLE ON)
elseif(MSVC)
  set(CDVMH_COMPILER_IS_GCC_COMPATIBLE OFF)
elseif("${CMAKE_CXX_COMPILER_ID}" MATCHES "Clang")
  set(CDVMH_COMPILER_IS_GCC_COMPATIBLE ON)
endif()

if(${LLVM_PACKAGE_VERSION} VERSION_LESS "3.5" OR NOT LLVM_ENABLE_RTTI)
  if(CDVMH_COMPILER_IS_GCC_COMPATIBLE)
    cdvmh_replace_compiler_option(CMAKE_CXX_FLAGS "-frtti" "-fno-rtti")
  elseif(MSVC)
    cdvmh_replace_compiler_option(CMAKE_CXX_FLAGS "/GR" "/GR-")
  endif()
endif()

if (MSVC)
  add_compile_options("/bigobj")
  if (MSVC_VERSION GREATER_EQUAL 1920)
    add_compile_options("/Zc:__cplusplus")
  endif()
endif()

include_directories(${LLVM_INCLUDE_DIRS})
link_directories(${LLVM_LIBRARY_DIRS})

add_subdirectory(src)

if (TARGET intrinsics_gen)
  set_target_properties(intrinsics_gen PROPERTIES FOLDER "${DVM_RESOURCE_FOLDER}")
endif()
if (TARGET omp_gen)
  set_target_properties(omp_gen PROPERTIES FOLDER "${DVM_RESOURCE_FOLDER}")
endif()
if (TARGET acc_gen)
  set_target_properties(acc_gen PROPERTIES FOLDER "${DVM_RESOURCE_FOLDER}")
endif()
