cmake_minimum_required(VERSION 3.6)

# Use lb-dev command line search path, if defined.
if(EXISTS ${CMAKE_CURRENT_SOURCE_DIR}/searchPath.cmake)
  include(${CMAKE_CURRENT_SOURCE_DIR}/searchPath.cmake)
endif()

if(CMAKE_PREFIX_PATH)
  list(REMOVE_DUPLICATES CMAKE_PREFIX_PATH)
endif()

include(/cvmfs/lhcb.cern.ch/lib/var/lib/LbEnv/3588/stable/linux-64/lib/python3.12/site-packages/LbDevTools/data/toolchain.cmake)
