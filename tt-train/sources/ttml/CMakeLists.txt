project(ttml)

file(
    GLOB_RECURSE SOURCES
    LIST_DIRECTORIES true
    *.hpp
    *.cpp
) # I am a bad person

# Check if Metalium::Metal target exists
# If it does not exist, assume that we are building with tt-train as top level project
if(NOT TARGET Metalium::Metal)
    if("$ENV{TT_METAL_HOME}" STREQUAL "")
        message(FATAL_ERROR "TT_METAL_HOME is not set")
    endif()

    set(METALIUM_INCLUDE_DIRS
        # Metalium
        "$ENV{TT_METAL_HOME}"
        "$ENV{TT_METAL_HOME}/tt_metal"
        "$ENV{TT_METAL_HOME}/tt_metal/third_party/umd"
        "$ENV{TT_METAL_HOME}/tt_metal/hw/inc/wormhole"
        "$ENV{TT_METAL_HOME}/tt_metal/hw/inc/wormhole/wormhole_b0_defines"
        "$ENV{TT_METAL_HOME}/tt_metal/hw/inc/"
        "$ENV{TT_METAL_HOME}/tt_metal/third_party/umd/src/firmware/riscv/wormhole"
        "$ENV{TT_METAL_HOME}/tt_metal/third_party/umd/device"
        # TTNN
        "$ENV{TT_METAL_HOME}/ttnn/cpp"
        "$ENV{TT_METAL_HOME}/ttnn/cpp/ttnn/deprecated"
        "${reflect_SOURCE_DIR}"
    )

    message(STATUS "Metalium not found, attempting to locate")

    # Define the path to look for the library
    set(METALIUM_LIB_PATH "$ENV{TT_METAL_HOME}/build/lib")

    # Try to find the library
    find_library(TT_METAL_LIBRARY NAMES "tt_metal" PATHS "${METALIUM_LIB_PATH}" NO_DEFAULT_PATH)
    find_library(TTNN_LIBRARY NAMES "_ttnn.so" PATHS "${METALIUM_LIB_PATH}" NO_DEFAULT_PATH)

    if(TT_METAL_LIBRARY)
        add_library(Metalium::Metal SHARED IMPORTED)
        set_target_properties(
            Metalium::Metal
            PROPERTIES
                IMPORTED_LOCATION
                    "${TT_METAL_LIBRARY}"
                INTERFACE_INCLUDE_DIRECTORIES
                    "${METALIUM_INCLUDE_DIRS}"
        )
        message(STATUS "Successfully found libtt_metal.so at ${TT_METAL_LIBRARY}")
    else()
        message(FATAL_ERROR "libtt_metal.so not found in ${METALIUM_LIB_PATH}")
    endif()
    if(TTNN_LIBRARY)
        add_library(Metalium::TTNN SHARED IMPORTED)
        set_target_properties(
            Metalium::TTNN
            PROPERTIES
                IMPORTED_LOCATION
                    "${TTNN_LIBRARY}"
                INTERFACE_INCLUDE_DIRECTORIES
                    "${METALIUM_INCLUDE_DIRS}"
        )
        message(STATUS "Successfully found _ttnn.so at ${TTNN_LIBRARY}")
    else()
        message(FATAL_ERROR "_ttnn.so not found in ${METALIUM_LIB_PATH}")
    endif()
else()
    message(STATUS "Metalium targets already exists")
endif()

add_library(ttml STATIC ${SOURCES})

target_include_directories(ttml PUBLIC ${PROJECT_SOURCE_DIR})

find_package(Python REQUIRED Development)

foreach(lib ${BoostPackages})
    target_include_directories(ttml SYSTEM PUBLIC ${Boost${lib}_SOURCE_DIR}/include)
endforeach()

target_link_libraries(
    ttml
    PUBLIC
        dl
        z
        pthread
        atomic
        Metalium::Metal
        Metalium::TTNN
        Python::Python
        fmt::fmt-header-only
        magic_enum
        yaml-cpp::yaml-cpp
        xtensor
        xtl
        tokenizers_cpp
        wandbcpp
        Boost::core
        Boost::container
)

message(STATUS "xtensor_SOURCE_DIR: ${xtensor_SOURCE_DIR}")
message(STATUS "xtl_SOURCE_DIR: ${xtl_SOURCE_DIR}")

message(STATUS "msgpack_SOURCE_DIR: ${msgpack_SOURCE_DIR}")
target_include_directories(ttml PUBLIC ${msgpack_SOURCE_DIR}/include)
message(STATUS "cli11_SOURCE_DIR: ${CLI11_SOURCE_DIR}")
target_include_directories(ttml PUBLIC ${CLI11_SOURCE_DIR}/include)

target_include_directories(ttml PUBLIC ${TOKENZIER_CPP_PATH}/include)

target_link_libraries(
    tokenizers_cpp
    PUBLIC
        ${LIBC++}
        ${LIBC++ABI}
)
target_compile_options(tokenizers_cpp PUBLIC -stdlib=libc++)

target_link_libraries(
    wandbcpp
    PUBLIC
        ${LIBC++}
        ${LIBC++ABI}
)
target_compile_options(wandbcpp PUBLIC -stdlib=libc++)

add_definitions(-DTOKENIZERS_DATA_PATH="${CMAKE_CURRENT_SOURCE_DIR}/data/tokenizers")
