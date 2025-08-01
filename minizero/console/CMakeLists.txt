file(GLOB SRCS *.cpp)

find_package(pybind11 REQUIRED)

add_library(console ${SRCS})
target_include_directories(console PUBLIC ${CMAKE_CURRENT_SOURCE_DIR})
target_link_libraries(
    console
    actor
    config
    environment
    network
    utils
    zero
    ${TORCH_LIBRARIES}
    pybind11::embed
)