set(VULKAN_SDK "/Users/tmc/VulkanSDK/1.3.261.1/macOS")
cmake_minimum_required(VERSION 3.25)
project(LearnVulkan)
set(CMAKE_CXX_STANDARD 20)
find_package(glfw3 3.3 REQUIRED)
find_package(glm REQUIRED)
find_package(Vulkan REQUIRED)
if(APPLE)
    add_definitions(-DAPPLE)
endif()
add_definitions(-DCMAKE_SOURCE_DIR="${CMAKE_SOURCE_DIR}")
add_executable(${PROJECT_NAME} src/main.cpp
        src/stb_image.h
        src/tiny_obj_loader.h)

target_include_directories(${PROJECT_NAME} PRIVATE ${VULKAN_SDK}/include)
target_link_directories(${PROJECT_NAME} PRIVATE ${VULKAN_SDK}/lib)

# Link against the Vulkan library
target_link_libraries(${PROJECT_NAME} PRIVATE Vulkan::Vulkan)
target_link_libraries(${PROJECT_NAME} PRIVATE glfw)
