##
## CMAKE VERSION
##
cmake_minimum_required(VERSION 3.5)

##
## PROJECT NAME
##
project(SMPL++)

##
## COMPILER SETTINGS
##
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

set(CMAKE_BINARY_DIR ${CMAKE_SOURCE_DIR}/build)
set(EXECUTABLE_OUTPUT_PATH ${CMAKE_BINARY_DIR})
set(LIBRARY_OUTPUT_PATH ${CMAKE_BINARY_DIR})

if(MSVC)
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} /EHsc /MP /bigobj")
    set(CMAKE_EXE_LINKER_FLAGS /MANIFEST:NO)
    set(CMAKE_C_FLAGS "/EHsc")
    set(CMAKE_CXX_FLAGS "/EHsc")
endif()

# We need zlib for reading numpy .npz files
add_subdirectory(zlib)
include_directories("${CMAKE_SOURCE_DIR}/./zlib")

add_subdirectory(freeglut-3.0.0)
SET(CMAKE_CXX_FLAGS_DEBUG  "-MDd -Zi -Ob0 -Od -RTC1  -_Debug") 
FIND_PACKAGE(OpenGL)
INCLUDE_DIRECTORIES("glm")
INCLUDE_DIRECTORIES("freeglut-3.0.0/include")
IF ( MSVC )
  SET(M_LIB)
ELSE ( MSVC )
  find_library(M_LIB m)
ENDIF ( MSVC )


find_package(OpenCV REQUIRED)

include_directories(${OpenCV_INCLUDE_DIRS})
INCLUDE_DIRECTORIES("OpenGL_renderer")
INCLUDE_DIRECTORIES("glew/include")
INCLUDE_DIRECTORIES("glm")
include_directories("${CMAKE_SOURCE_DIR}")
# include_directories("${CMAKE_SOURCE_DIR}/../Eigen")
include_directories("${CMAKE_SOURCE_DIR}/zlib")
INCLUDE_DIRECTORIES("${CMAKE_SOURCE_DIR}/numpyLib")
    
##
## GATHER FILES
##
file(GLOB HEADER
    "SMPL++/include/*.h"
    "SMPL++/include/*.hpp"
    "SMPL++/include/definition/*.h"
    "SMPL++/include/definition/*.hpp"
    "SMPL++/include/toolbox/*.h"
    "SMPL++/include/toolbox/*.hpp"
    "SMPL++/include/smpl/*.h"
    "SMPL++/include/smpl/*.hpp"
    "OpenGL_renderer/OpenGL_renderer.h"
	"OpenGL_renderer/GLSLShader.h"
    "OpenGL_renderer/model.h"
)

file(GLOB SOURCE
    "SMPL++/src/*.c"
    "SMPL++/src/*.cpp"
    "SMPL++/src/definition/*.c"
    "SMPL++/src/definition/*.cpp"
    "SMPL++/src/toolbox/*.c"
    "SMPL++/src/toolbox/*.cpp"
    "SMPL++/src/smpl/*.c"
    "SMPL++/src/smpl/*.cpp"
    "OpenGL_renderer/OpenGL_renderer.cpp"
	"OpenGL_renderer/GLSLShader.cpp"
	"OpenGL_renderer/model.cpp"
    "numpyLib/cnpy.cpp"
)

##
## INCLUDE DIRECTORIES
##
set(SMPLXX_INCLUDE_DIRS
    "SMPL++/include"
)

set(LibTorchReleaseRoot "F:/Documents/Visual Studio 2019/projects/libtorch_release")
set(LibTorchDebugRoot "F:/Documents/Visual Studio 2019/projects/libtorch_debug")

set( TORCH_INCLUDE_DIRS
"${LibTorchReleaseRoot}/include"
"${LibTorchReleaseRoot}/include/torch/csrc/api/include"
)
    
set(PROJECT_INCLUDE_DIRS
    ${SMPLXX_INCLUDE_DIRS}    
    ${TORCH_INCLUDE_DIRS}
)

##
## LIBRARIES
##
# set(TORCH_LIBRARIES
#     optimized "${LibTorchReleaseRoot}/lib/c10.lib"
#     optimized "${LibTorchReleaseRoot}/lib/torch.lib"
#     debug "f${LibTorchDebugRoot}/libtorch-debug/lib/c10.lib"
#     debug "f${LibTorchDebugRoot}/libtorch-debug/lib/torch.lib"
# 
# )

set(TORCH_LIBRARIES
    optimized "${LibTorchReleaseRoot}/lib/c10.lib"
    optimized "${LibTorchReleaseRoot}/lib/c10_cuda.lib"
    optimized "${LibTorchReleaseRoot}/lib/torch.lib"
    optimized "${LibTorchReleaseRoot}/lib/torch_cpu.lib"
    optimized "${LibTorchReleaseRoot}/lib/torch_cuda.lib"
    optimized "${LibTorchReleaseRoot}/lib/clog.lib"
    optimized "${LibTorchReleaseRoot}/lib/cpuinfo.lib"    
    optimized "${LibTorchReleaseRoot}/lib/asmjit.lib"
    optimized "${LibTorchReleaseRoot}/lib/fbgemm.lib"
    optimized "${LibTorchReleaseRoot}/lib/libprotobuf.lib"
    optimized "${LibTorchReleaseRoot}/lib/libprotobuf-lite.lib"
    optimized "${LibTorchReleaseRoot}/lib/libprotoc.lib"
    optimized "${LibTorchReleaseRoot}/lib/mkldnn.lib"


    debug "${LibTorchDebugRoot}/lib/c10.lib"
    debug "${LibTorchDebugRoot}/lib/c10_cuda.lib"
    debug "${LibTorchDebugRoot}/lib/torch.lib"
    debug "${LibTorchDebugRoot}/lib/torch_cpu.lib"
    debug "${LibTorchDebugRoot}/lib/torch_cuda.lib"
    debug "${LibTorchDebugRoot}/lib/clog.lib"
    debug "${LibTorchDebugRoot}/lib/cpuinfo.lib"    
    debug "${LibTorchDebugRoot}/lib/asmjit.lib"
    debug "${LibTorchDebugRoot}/lib/fbgemm.lib"
    debug "${LibTorchDebugRoot}/lib/libprotobufd.lib"
    debug "${LibTorchDebugRoot}/lib/libprotobuf-lited.lib"
    debug "${LibTorchDebugRoot}/lib/libprotocd.lib"
    debug "${LibTorchDebugRoot}/lib/mkldnn.lib"

)

set(PROJECT_LIBRARIES
    zlib
    ${OpenCV_LIBS}
    freeglut
    "../glew/lib/Release/x64/glew32"    
    ${TORCH_LIBRARIES}
)
##
## OUTPUTS
##
add_executable(smplpp ${HEADER} ${SOURCE})
target_include_directories(smplpp PRIVATE ${PROJECT_INCLUDE_DIRS})
target_link_libraries(smplpp ${PROJECT_LIBRARIES})
if(MSVC)
set_target_properties( smplpp PROPERTIES  VS_DEBUGGER_WORKING_DIRECTORY "$<TARGET_FILE_DIR:smplpp>")
set_property(DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR} PROPERTY VS_STARTUP_PROJECT smplpp)
endif()