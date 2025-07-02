# Embodied Cognition Tests
ADD_EXECUTABLE(test_embodied_cognition
    test_embodied_cognition.cpp
)

TARGET_LINK_LIBRARIES(test_embodied_cognition
    robotics-embodied
    ${ATOMSPACE_LIBRARIES}
    ${COGUTIL_LIBRARIES}
)

# Test for sensory-motor validation
ADD_EXECUTABLE(test_sensory_motor_validation
    test_sensory_motor_validation.cpp
)

TARGET_LINK_LIBRARIES(test_sensory_motor_validation
    robotics-embodied
    ${ATOMSPACE_LIBRARIES}
    ${COGUTIL_LIBRARIES}
)

# Install tests
INSTALL(TARGETS test_embodied_cognition test_sensory_motor_validation
    RUNTIME DESTINATION bin
    COMPONENT "Development"
)