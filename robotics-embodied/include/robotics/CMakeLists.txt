# Robotics Embodied Cognition Headers
SET(ROBOTICS_HEADERS
    EmbodiedCognition.hpp
)

# Install headers
INSTALL(FILES ${ROBOTICS_HEADERS}
    DESTINATION include/robotics
    COMPONENT "Development"
)

# Add library sources (will be added when we create .cpp files)
ADD_LIBRARY(robotics-embodied SHARED
    EmbodiedCognition.cpp
)

# Link libraries
TARGET_LINK_LIBRARIES(robotics-embodied
    ${ATOMSPACE_LIBRARIES}
    ${COGUTIL_LIBRARIES}
)

# Install library
INSTALL(TARGETS robotics-embodied
    LIBRARY DESTINATION lib
    COMPONENT "Runtime"
)