cmake_minimum_required(VERSION 3.16)
project(carnot_pack LANGUAGES CXX)
include(ExternalProject)
find_package(Threads REQUIRED)
find_program(MAKE_TOOL NAMES gmake make REQUIRED)
set(prefix ${CMAKE_CURRENT_BINARY_DIR}/gmp)
file(MAKE_DIRECTORY ${prefix}/include)
set(m4_prefix ${CMAKE_CURRENT_BINARY_DIR}/m4)
ExternalProject_Add(m4_source URL ${CMAKE_CURRENT_SOURCE_DIR}/m4-1.4.19.tar.xz
  URL_HASH SHA256=63aede5c6d33b6d9b13511cd0be2cac046f2e70fd0a07aa9573a04a82783af96
  CONFIGURE_COMMAND ${CMAKE_COMMAND} -E env FORCE_UNSAFE_CONFIGURE=1 <SOURCE_DIR>/configure --prefix=${m4_prefix}
  BUILD_COMMAND ${MAKE_TOOL} -j4 INSTALL_COMMAND ${MAKE_TOOL} install)
ExternalProject_Add(gmp_source DEPENDS m4_source URL ${CMAKE_CURRENT_SOURCE_DIR}/gmp-6.3.0.tar.xz
  URL_HASH SHA256=a3c2b80201b89e68616f4ad30bc66aee4927c3ce50e33929ca819d5c43538898
  CONFIGURE_COMMAND ${CMAKE_COMMAND} -E env M4=${m4_prefix}/bin/m4 <SOURCE_DIR>/configure --prefix=${prefix} --disable-shared --enable-static
  BUILD_COMMAND ${MAKE_TOOL} -j4 INSTALL_COMMAND ${MAKE_TOOL} install
  BUILD_BYPRODUCTS ${prefix}/lib/libgmp.a)
add_executable(solution main.cpp solve.cpp)
add_dependencies(solution gmp_source)
target_compile_features(solution PRIVATE cxx_std_17)
target_compile_options(solution PRIVATE -O3)
target_include_directories(solution PRIVATE ${prefix}/include)
target_link_libraries(solution PRIVATE ${prefix}/lib/libgmp.a Threads::Threads ${CMAKE_DL_LIBS})
