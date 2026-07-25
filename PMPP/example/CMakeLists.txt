cmake_minimum_required(VERSION 3.18)
set(CMAKE_BUILD_TYPE Debug)

project(pp LANGUAGES CXX CUDA)
find_package(OpenMP REQUIRED)

# Set C++ Standard
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)


# Set CUDA Standard
set(CMAKE_CUDA_STANDARD 17)
set(CMAKE_CUDA_STANDARD_REQUIRED ON)
set(CMAKE_CUDA_ARCHITECTURES 89)

# Set CUDA Flags
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -O1 -fopenmp -Wno-return-type")
set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} -O1 -fopenmp --extended-lambda -lineinfo")

# Set Searching Path for Header Files
include_directories(${PROJECT_SOURCE_DIR}/include)

# Collecting source codes
file(GLOB_RECURSE SRC_FILES CONFIGURE_DEPENDS
	"${PROJECT_SOURCE_DIR}/src/*.c"
	"${PROJECT_SOURCE_DIR}/src/*.cpp"
	"${PROJECT_SOURCE_DIR}/src/*.cu"
)

#add_executable(${PROJECT_NAME} ${SRC_FILES})

# # Add Core Library
add_library(${PROJECT_NAME} STATIC ${SRC_FILES})

# #add_subdirectory(depends/libsnark)

# # 编译测试程序                          
file(GLOB TEST_SOURCES CONFIGURE_DEPENDS "${PROJECT_SOURCE_DIR}/test/*.cu" "${PROJECT_SOURCE_DIR}/test/*.cpp")
foreach(test_src ${TEST_SOURCES})
    get_filename_component(test_name ${test_src} NAME_WE)
    add_executable(${test_name} ${test_src})
    set_target_properties(${test_name} PROPERTIES 	
		CUDA_SEPARABLE_COMPILATION ON
    	POSITION_INDEPENDENT_CODE ON
	)
    target_link_libraries(${test_name} PRIVATE 
		${PROJECT_NAME}
		OpenMP::OpenMP_CXX
	)
    target_include_directories(${test_name} PRIVATE 
		${PROJECT_SOURCE_DIR}/include
	)
endforeach()