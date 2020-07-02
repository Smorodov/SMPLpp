#define GL_GLEXT_PROTOTYPES
#include <windows.h>

#include <numeric>
#include <vector>
#include <limits>
#include <iostream>
#include <opencv2/opencv.hpp>
#include "model.h"
#include "GLSLShader.h"
#include <GL/gl.h>			
#include <GL/glu.h>	

#include <glm/glm.hpp>
#include <glm/gtc/matrix_transform.hpp>
#include <glm/gtx/transform.hpp>
#include <glm/gtx/transform2.hpp>
#include <glm/gtc/type_ptr.hpp>

#include "Eigen/Eigen"

#define ARRAY_COUNT( array ) (sizeof( array ) / (sizeof( array[0] ) * (sizeof( array ) != sizeof(void*) || sizeof( array[0] ) <= sizeof(void*))))

#pragma once

class OpenGL_Renderer
{
public:

	// Fields

	static glm::mat4 Projection; // Projection matrix
	static glm::mat4 Viewport; // model to shader space matrix (to cube with side 2 centered at origin)
	static glm::mat4 ModelView; // Viewport matrix
	static glm::mat4x3 vp; // 3D to screen 2d matrix 

	static GLSLShader shader; // Shader helper class instance
	static GLuint theProgram; // shader program handle
	static GLuint vertexBufferObject; // vbo handle
	static GLuint vao; // vertex array object handle
	static GLuint pbo_id; // vertex array object handle
	static size_t pbo_size;
	static GLuint BG_textureID; // Backgroun image quad texture
	static GLuint Face_textureID; // Face texture

	static int width;
	static int height;
	static cv::Mat BG; // Background image
	static float* buf;
	static glm::vec4 viewport; // viewport vector (x,y,w,h)
	static GLModel* model; // model for rendering
	static MSG msg;
	static HWND hwnd;

	// Methods

	OpenGL_Renderer(cv::Mat& Background);
	~OpenGL_Renderer();
	static void Render(GLModel* mdl);
	// Viewport
	static void setViewport(float l, float r, float b, float t, float n, float f);
	static glm::vec4 getViewport(void);
	// Projection matrix
	static void setProjectionMatrix(glm::mat4 proj);
	static void setProjectionMatrix(cv::Mat& proj);
	static glm::mat4 getProjectionMatrix(void);
	// ModelView Matrix
	static void setModelViewMatrix(glm::mat4 mv);
	static void setModelViewMatrix(cv::Mat& mv);
	static glm::mat4 getModelViewMatrix(void);
	// Affine camera matrix
	static glm::mat4x3 getAffineCameraMatrix(void);
	// Project points from model space to screen
	static void projectPoints(std::vector<Eigen::Vector3f> vertCoords, std::vector<Eigen::Vector2f>& vertProjected);

	static void InitializeVertexBuffer(GLModel* mdl);
	static void setTexture(cv::Mat& image, GLuint& _textureID, int tex_size = 0);
	static void setTextureA(cv::Mat& image, cv::Mat& mask, GLuint& _textureID, int tex_size = 0);
	static void getImage(cv::Mat& im);
	static void AdjustSize();
	static void init();
	static int SetWindowPixelFormat(HDC hDC);
	static LRESULT CALLBACK WindowFunc(HWND hWnd, UINT msg, WPARAM wParam, LPARAM lParam);
	static HWND createGLWindow(int w,int h);
};


