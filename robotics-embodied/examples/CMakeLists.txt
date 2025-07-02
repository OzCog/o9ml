# Embodied Cognition Examples
ADD_EXECUTABLE(embodied_cognition_demo
    embodied_cognition_demo.cpp
)

TARGET_LINK_LIBRARIES(embodied_cognition_demo
    robotics-embodied
    ${ATOMSPACE_LIBRARIES}
    ${COGUTIL_LIBRARIES}
)

# Install demo
INSTALL(TARGETS embodied_cognition_demo
    RUNTIME DESTINATION bin
    COMPONENT "Development"
)