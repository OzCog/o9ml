#!/bin/bash
# build_and_install_atomspace.sh - Helper script to build AtomSpace with config files

set -e

echo "Building and installing AtomSpace..."

cd orc-as/atomspace

# Create missing lib directory if it doesn't exist
if [ ! -d "lib" ]; then
  mkdir -p lib
  echo "# Empty lib directory for build compatibility" > lib/CMakeLists.txt
fi

if [ ! -d "build" ]; then
  mkdir -p build
fi

cd build
cmake .. -DCMAKE_BUILD_TYPE=Release
make -j$(nproc)
sudo make install
sudo ldconfig

# Create AtomSpace CMake config files for dependent projects
sudo mkdir -p /usr/local/lib/cmake/AtomSpace

sudo tee /usr/local/lib/cmake/AtomSpace/AtomSpaceConfig.cmake > /dev/null <<'EOF'
# AtomSpaceConfig.cmake - Minimal config file for AtomSpace

# Set version information
set(PACKAGE_VERSION "5.0.3")
set(AtomSpace_VERSION "5.0.3")
set(ATOMSPACE_VERSION "5.0.3")

# Version compatibility check
set(PACKAGE_VERSION_EXACT FALSE)
set(PACKAGE_VERSION_COMPATIBLE TRUE)
set(PACKAGE_VERSION_UNSUITABLE FALSE)

# Set basic variables
set(ATOMSPACE_FOUND TRUE)
set(AtomSpace_FOUND TRUE)

# Set include directories
set(ATOMSPACE_INCLUDE_DIRS "/usr/local/include")
set(AtomSpace_INCLUDE_DIRS "/usr/local/include")

# Set library directories and libraries
set(ATOMSPACE_LIBRARY_DIRS "/usr/local/lib/opencog")
set(AtomSpace_LIBRARY_DIRS "/usr/local/lib/opencog")

# Find the atomspace library
find_library(ATOMSPACE_LIBRARIES
    NAMES atomspace
    PATHS /usr/local/lib/opencog
    NO_DEFAULT_PATH
)

set(AtomSpace_LIBRARIES ${ATOMSPACE_LIBRARIES})

# Set other common variables
set(ATOMSPACE_DATA_DIR "/usr/local/share/opencog")
set(AtomSpace_DATA_DIR "/usr/local/share/opencog")

# Mark as found
set(ATOMSPACE_FOUND TRUE)
set(AtomSpace_FOUND TRUE)

# Export targets (minimal)
if(NOT TARGET atomspace::atomspace)
    add_library(atomspace::atomspace SHARED IMPORTED)
    set_target_properties(atomspace::atomspace PROPERTIES
        IMPORTED_LOCATION "${ATOMSPACE_LIBRARIES}"
        INTERFACE_INCLUDE_DIRECTORIES "${ATOMSPACE_INCLUDE_DIRS}"
    )
endif()

message(STATUS "Found AtomSpace: ${ATOMSPACE_LIBRARIES}")
EOF

sudo tee /usr/local/lib/cmake/AtomSpace/AtomSpaceConfigVersion.cmake > /dev/null <<'EOF'
# AtomSpaceConfigVersion.cmake - Version file for AtomSpace

set(PACKAGE_VERSION "5.0.3")

# Check whether the requested PACKAGE_FIND_VERSION is compatible
if("${PACKAGE_VERSION}" VERSION_LESS "${PACKAGE_FIND_VERSION}")
  set(PACKAGE_VERSION_COMPATIBLE FALSE)
else()
  set(PACKAGE_VERSION_COMPATIBLE TRUE)
  if ("${PACKAGE_VERSION}" VERSION_EQUAL "${PACKAGE_FIND_VERSION}")
    set(PACKAGE_VERSION_EXACT TRUE)
  endif()
endif()
EOF

echo "AtomSpace built and installed with config files"