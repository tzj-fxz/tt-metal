add_executable(test_distributed test_distributed.cpp)

# Set up properties for the target
setup_ttnn_test_target(test_distributed)

# Add test to CTest
add_test(NAME test_distributed COMMAND test_distributed)
