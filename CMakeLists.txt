cmake_minimum_required(VERSION 3.9.6)

project(dynawo-mpi CXX)

if(NOT DEFINED DYNAWO_INSTALL_DIR)
  message(FATAL_ERROR "You need to define DYNAWO_INSTALL_DIR variable.")
endif()
file(TO_CMAKE_PATH "${DYNAWO_INSTALL_DIR}" DYNAWO_INSTALL_DIR)
get_filename_component(DYNAWO_INSTALL_DIR ${DYNAWO_INSTALL_DIR} ABSOLUTE)
set(CMAKE_MODULE_PATH "${DYNAWO_INSTALL_DIR}/share/cmake")
set(Dynawo_DIR ${DYNAWO_INSTALL_DIR}/share)
set(CMAKE_PREFIX_PATH ${DYNAWO_INSTALL_DIR}/share/cmake)

# If C++11 is enabled, check flags availability
set(CXX11_ENABLED ON CACHE BOOL "")

if(CXX11_ENABLED)
  # Test if a C++11 flag is supported
  include(CheckCXXCompilerFlag)
  check_cxx_compiler_flag("-std=c++11" COMPILER_SUPPORTS_CXX11)
  check_cxx_compiler_flag("-std=c++0x" COMPILER_SUPPORTS_CXX0X)
  if(COMPILER_SUPPORTS_CXX11)
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++11 -DLANG_CXX11")
  elseif(COMPILER_SUPPORTS_CXX0X)
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++0x -DLANG_CXX0X")
  else()
    message(STATUS "The compiler ${CMAKE_CXX_COMPILER} has no C++11 support. Disabling C++11.")
    set(CXX11_ENABLED OFF CACHE BOOL "" FORCE)
  endif()
else()
  set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++98")
endif()

set(CMAKE_BUILD_TYPE "Release" CACHE STRING "Choose the type of build, options are: Debug Release RelWithDebInfo MinSizeRel.")

set(CMAKE_CXX_FLAGS_DEBUG "${CMAKE_CXX_FLAGS_DEBUG} -O0")

if(NOT CXX11_ENABLED)
  set(USE_LEGACY_IIDM ON CACHE BOOL "Force use of legacy IIDM library" FORCE)
endif()

if(NOT USE_LEGACY_IIDM)
  add_definitions(-DUSE_POWSYBL)
endif()

set(ADEPT_HOME ${DYNAWO_INSTALL_DIR})
set(XERCESC_HOME ${DYNAWO_INSTALL_DIR})
set(DYNAWO_LIBXML_HOME ${DYNAWO_INSTALL_DIR})
set(SUITESPARSE_HOME ${DYNAWO_INSTALL_DIR})
set(NICSLU_HOME ${DYNAWO_INSTALL_DIR})
set(SUNDIALS_HOME ${DYNAWO_INSTALL_DIR})
set(BOOST_ROOT ${DYNAWO_INSTALL_DIR})
set(LIBIIDM_HOME ${DYNAWO_INSTALL_DIR})
set(LIBZIP_HOME ${DYNAWO_INSTALL_DIR})
set(LIBXML2_HOME ${DYNAWO_INSTALL_DIR})
list(APPEND CMAKE_PREFIX_PATH "${LIBXML2_HOME}")

# Boost
set(BUILD_SHARED_LIBS ON)
set(Boost_USE_STATIC_RUNTIME False)
add_definitions(-DBOOST_ALL_DYN_LINK)
# Tell BOOST to disable auto linking
add_definitions(-DBOOST_ALL_NO_LIB)

if(CXX11_ENABLED)
  set(Boost_RequiredVersion  "1.70")
else()
  set(Boost_RequiredVersion  "1.64")
endif()

if(BUILD_TESTS OR BUILD_TESTS_COVERAGE)
  find_package(Boost "${Boost_RequiredVersion}" REQUIRED
          COMPONENTS
          filesystem
          program_options
          serialization
          system
          log
          log_setup
          iostreams
          thread
          regex
          chrono
          atomic
          )
else()
  find_package(Boost "${Boost_RequiredVersion}" REQUIRED
    COMPONENTS filesystem program_options serialization system log log_setup thread regex chrono atomic
  )
endif()

# Python Interpreter
find_package(PythonInterp REQUIRED)

find_package(SuiteSparse REQUIRED)

find_package(NICSLU)

if(${NICSLU_FOUND})
  add_definitions(-DWITH_NICSLU)
endif()

find_package(Sundials REQUIRED)

set(USE_ADEPT "YES")
find_package(Adept REQUIRED)

find_package(XercesC 3.2.2 REQUIRED)

find_package(ZLIB 1.2.3 REQUIRED)
set_property(TARGET ZLIB::ZLIB APPEND PROPERTY IMPORTED_LOCATION "${ZLIB_LIBRARY}")

find_package(LibArchive 2.8.0 REQUIRED)

find_package(libZIP 1.3.0 REQUIRED)

find_package(libXML 0.2.4 REQUIRED PATHS ${DYNAWO_LIBXML_HOME})

if(NOT USE_LEGACY_IIDM)
  set(DYNAWO_USE_LIBXML2 "Y")
  find_package(LibXml2 "2.9" REQUIRED)
  find_package(LibIIDM "1.4.0" REQUIRED PATHS ${LIBIIDM_HOME})
else()
  set(DYNAWO_USE_LIBXML2 "N")
  find_package(libIIDM "0.2.8" REQUIRED PATHS ${LIBIIDM_HOME}
    COMPONENTS
    core
    xml
    ext-activeseason
    ext-activeseason-xml
    ext-activepowercontrol
    ext-activepowercontrol-xml
    ext-busbarsectionposition
    ext-busbarsectionposition-xml
    ext-connectableposition
    ext-connectableposition-xml
    ext-currentlimitsperseason
    ext-currentlimitsperseason-xml
    ext-generatoractivepowercontrol
    ext-generatoractivepowercontrol-xml
    ext-hvdcangledroopactivepowercontrol
    ext-hvdcangledroopactivepowercontrol-xml
    ext-hvdcoperatoractivepowerrange
    ext-hvdcoperatoractivepowerrange-xml
    ext-generatorentsoecategory
    ext-generatorentsoecategory-xml
    ext-generatorstartup
    ext-generatorstartup-xml
    ext-loaddetail
    ext-loaddetail-xml
    ext-standbyautomaton
    ext-standbyautomaton-xml
    )
endif()

find_package(Dynawo CONFIG REQUIRED PATHS ${DYNAWO_INSTALL_DIR} PATH_SUFFIXES share)

find_package(MPI REQUIRED)

set(runtime_LD_LIBRARY_PATH "LD_LIBRARY_PATH=${DYNAWO_INSTALL_DIR}/lib:${Boost_INCLUDE_DIR}/../lib:$ENV{LD_LIBRARY_PATH}")
set(runtime_PATH "PATH=$ENV{PATH}")

set(DYNAWOMPI_SOURCES
  main_mpi.cpp
  )

add_executable(dynawo-mpi ${DYNAWOMPI_SOURCES})

target_link_libraries(dynawo-mpi
  Dynawo::dynawo_Common
  Dynawo::dynawo_SimulationCommon
  Dynawo::dynawo_Simulation
  Dynawo::dynawo_API_JOB
  XMLSAXParser${LibXML_LINK_SUFFIX}
  $<$<NOT:$<BOOL:${USE_LEGACY_IIDM}>>:LibXml2::LibXml2>
  Boost::filesystem
  Boost::program_options
  MPI::MPI_CXX
  )

find_package(OpenMP)

#set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${OpenMP_CXX_FLAGS} -DWITH_OPENMP")


set(DYNAWOOPENMP_SOURCES
  main_openmp.cpp
  )

add_executable(dynawo-openmp ${DYNAWOOPENMP_SOURCES})

target_compile_definitions(dynawo-openmp PUBLIC WITH_OPENMP)

target_compile_options(dynawo-openmp PUBLIC ${OpenMP_CXX_FLAGS})

target_link_libraries(dynawo-openmp
  Dynawo::dynawo_Common
  Dynawo::dynawo_SimulationCommon
  Dynawo::dynawo_Simulation
  Dynawo::dynawo_API_JOB
  XMLSAXParser${LibXML_LINK_SUFFIX}
  $<$<NOT:$<BOOL:${USE_LEGACY_IIDM}>>:LibXml2::LibXml2>
  Boost::filesystem
  Boost::program_options
  ${OpenMP_CXX_FLAGS}
  )
