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

set(LibTorchReleaseRoot "F:/Documents/Visual Studio 2019/projects/libtorch_release")
set(LibTorchDebugRoot "F:/Documents/Visual Studio 2019/projects/libtorch_debug")

find_package(Torch REQUIRED PATHS "${LibTorchReleaseRoot}/share/cmake/Torch")
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${TORCH_CXX_FLAGS}")
include_directories(${TORCH_INCLUDE_DIRS})
include_directories("SMPL++/include")
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
  file(GLOB TORCH_DLLS "${TORCH_INSTALL_PREFIX}/lib/*.dll")
  add_custom_command(TARGET smplpp
                     POST_BUILD
                     COMMAND ${CMAKE_COMMAND} -E copy_if_different
                     ${TORCH_DLLS}
                     $<TARGET_FILE_DIR:smplpp>)
set_target_properties( smplpp PROPERTIES  VS_DEBUGGER_WORKING_DIRECTORY "$<TARGET_FILE_DIR:smplpp>")
set_property(DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR} PROPERTY VS_STARTUP_PROJECT smplpp)
endif()