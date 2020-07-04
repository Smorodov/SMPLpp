#include "OpenGL_Renderer.h"

using namespace std;
using namespace cv;
using namespace Eigen;

// 3d transformation matrices
glm::mat4 OpenGL_Renderer::Projection;
glm::mat4 OpenGL_Renderer::Viewport;
glm::mat4 OpenGL_Renderer::ModelView;

glm::mat4x3 OpenGL_Renderer::vp(0);

GLSLShader OpenGL_Renderer::shader;
GLuint OpenGL_Renderer::theProgram;
GLuint OpenGL_Renderer::vertexBufferObject;
GLuint OpenGL_Renderer::vao;
GLuint OpenGL_Renderer::pbo_id;
size_t OpenGL_Renderer::pbo_size;

GLuint OpenGL_Renderer::BG_textureID;
//GLuint OpenGL_Renderer::Face_textureID;

int OpenGL_Renderer::width;
int OpenGL_Renderer::height;
cv::Mat OpenGL_Renderer::BG;
float* OpenGL_Renderer::buf = NULL; // Buffer, (see InitializeVertexBuffer method)
glm::vec4 OpenGL_Renderer::viewport;
GLModel* OpenGL_Renderer::model = NULL;
MSG OpenGL_Renderer::msg;
HWND OpenGL_Renderer::hwnd;


int OpenGL_Renderer::SetWindowPixelFormat(HDC hDC)
{
	int m_GLPixelIndex;
	PIXELFORMATDESCRIPTOR pfd;
	pfd.nSize = sizeof(PIXELFORMATDESCRIPTOR);
	pfd.nVersion = 1;
	pfd.dwFlags = PFD_DRAW_TO_WINDOW |
		PFD_SUPPORT_OPENGL | PFD_DOUBLEBUFFER;
	pfd.iPixelType = PFD_TYPE_RGBA;
	pfd.cColorBits = 32;
	pfd.cRedBits = 8;
	pfd.cRedShift = 16;
	pfd.cGreenBits = 8;
	pfd.cGreenShift = 8;
	pfd.cBlueBits = 8;
	pfd.cBlueShift = 0;
	pfd.cAlphaBits = 0;
	pfd.cAlphaShift = 0;
	pfd.cAccumBits = 64;
	pfd.cAccumRedBits = 16;
	pfd.cAccumGreenBits = 16;
	pfd.cAccumBlueBits = 16;
	pfd.cAccumAlphaBits = 0;
	pfd.cDepthBits = 32;
	pfd.cStencilBits = 8;
	pfd.cAuxBuffers = 0;
	pfd.iLayerType = PFD_MAIN_PLANE;
	pfd.bReserved = 0;
	pfd.dwLayerMask = 0;
	pfd.dwVisibleMask = 0;
	pfd.dwDamageMask = 0;
	m_GLPixelIndex = ChoosePixelFormat(hDC, &pfd);
	if (m_GLPixelIndex == 0) // Let's choose a default index.
	{
		m_GLPixelIndex = 1;
		if (DescribePixelFormat(hDC, m_GLPixelIndex, sizeof(PIXELFORMATDESCRIPTOR), &pfd) == 0)
			return 0;
	}
	if (SetPixelFormat(hDC, m_GLPixelIndex, &pfd) == FALSE)
		return 0;
	return 1;
}

void OpenGL_Renderer::setTexture(cv::Mat& image, GLuint& _textureID, int tex_size)
{
	if (tex_size != 0)
	{
		cv::resize(image, image, cv::Size(tex_size, tex_size));
	}
	cv::Mat image_rgb;
	if (image.channels() == 3)
	{
		cvtColor(image, image_rgb, cv::COLOR_BGR2RGB);
	}

	if (image.channels() == 1)
	{
		cvtColor(image, image_rgb, cv::COLOR_GRAY2RGB);
	}

	glGenTextures(1, &_textureID);
	glBindTexture(GL_TEXTURE_2D, _textureID);
	glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_NEAREST);
	//use fast 4-byte alignment (default anyway) if possible
	glPixelStorei(GL_UNPACK_ALIGNMENT, (image_rgb.step & 3) ? 1 : 4);
	//set length of one complete row in data (doesn't need to equal image.cols)
	glPixelStorei(GL_UNPACK_ROW_LENGTH, image_rgb.step / image_rgb.elemSize());
	glTexImage2D(GL_TEXTURE_2D, 0, 3, image_rgb.cols, image_rgb.rows, 0, GL_RGB, GL_UNSIGNED_BYTE, image_rgb.data);
}

// ----------------------------------------------------------------
//
// ----------------------------------------------------------------

void OpenGL_Renderer::setTextureA(cv::Mat& image, cv::Mat& mask, GLuint& _textureID, int tex_size)
{
	cv::resize(image, image, cv::Size(tex_size, tex_size));
	cv::resize(mask, mask, cv::Size(tex_size, tex_size));
	cv::Mat image_rgb;
	if (image.channels() == 3)
	{
		vector<cv::Mat> ch_src;
		split(image, ch_src);

		image_rgb = cv::Mat::zeros(image.rows, image.cols, CV_8UC4);
		vector<cv::Mat> ch;
		split(image_rgb, ch);
		ch[0] = ch_src[2];
		ch[1] = ch_src[1];
		ch[2] = ch_src[0];
		ch[3] = mask.clone();
		merge(ch, image_rgb);
	}

	if (image.channels() == 1)
	{
		image_rgb = cv::Mat::zeros(image.rows, image.cols, CV_8UC4);
		vector<cv::Mat> ch;
		split(image_rgb, ch);
		ch[0] = image.clone();
		ch[1] = image.clone();
		ch[2] = image.clone();
		ch[3] = mask.clone();
		merge(ch, image_rgb);
	}

	glGenTextures(1, &_textureID);
	glBindTexture(GL_TEXTURE_2D, _textureID);
	glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_NEAREST);
	//use fast 4-byte alignment (default anyway) if possible
	glPixelStorei(GL_UNPACK_ALIGNMENT, (image_rgb.step & 3) ? 1 : 4);
	//set length of one complete row in data (doesn't need to equal image.cols)
	glPixelStorei(GL_UNPACK_ROW_LENGTH, image_rgb.step / image_rgb.elemSize());
	glTexImage2D(GL_TEXTURE_2D, 0, 4, image_rgb.cols, image_rgb.rows, 0, GL_RGBA, GL_UNSIGNED_BYTE, image_rgb.data);
}

// ----------------------------------------------------------------
//
// ----------------------------------------------------------------

void OpenGL_Renderer::getImage(cv::Mat& im)
{
	int width, height;
	width = abs(viewport[2]);
	height = abs(viewport[3]);

	im = cv::Mat(height, width, CV_8UC4);

	glReadBuffer(GL_FRONT);

	// https://vec.io/posts/faster-alternatives-to-glreadpixels-and-glteximage2d-in-opengl-es  
	// Faster method to read framebuffer

	glReadBuffer(GL_COLOR_ATTACHMENT0);
	glBindBuffer(GL_PIXEL_PACK_BUFFER, pbo_id);

	glReadPixels(0, 0, width, height, GL_RGBA, GL_UNSIGNED_BYTE, 0);
	GLubyte *ptr = (GLubyte *)glMapBufferRange(GL_PIXEL_PACK_BUFFER, 0, pbo_size, GL_MAP_READ_BIT);
	memcpy(im.data, ptr, pbo_size);
	cv::cvtColor(im, im, cv::COLOR_RGBA2BGR);
	cv::flip(im, im, 0);

	glUnmapBuffer(GL_PIXEL_PACK_BUFFER);
	glBindBuffer(GL_PIXEL_PACK_BUFFER, 0);



}


void OpenGL_Renderer::AdjustSize()
{
	RECT r;
	GetClientRect(hwnd, &r);
	setViewport(r.left, r.right, r.bottom, r.top, -1000, 1000);
}


void OpenGL_Renderer::init()
{
	float pos[4] = { 3,3,3,1 };
	float dir[3] = { -1,-1,-1 };
	glEnable(GL_ALPHA_TEST);
	glEnable(GL_DEPTH_TEST);
	glEnable(GL_COLOR_MATERIAL);
	glEnable(GL_LIGHTING);
	glEnable(GL_LIGHT0);
	glEnable(GL_BLEND);
	glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA);
	glLightfv(GL_LIGHT0, GL_POSITION, pos);
	glLightfv(GL_LIGHT0, GL_SPOT_DIRECTION, dir);
}

LRESULT CALLBACK OpenGL_Renderer::WindowFunc(HWND hWnd, UINT msg, WPARAM wParam, LPARAM lParam)
{
	static HGLRC hGLRC;
	static HDC hDC;
	float pos[4] = { 3,3,3,1 };
	float dir[3] = { -1,-1,-1 };
	PAINTSTRUCT ps;
	int width, height;
	GLint dims[4] = { 0 };
	switch (msg)
	{
		// сообщение WM_CREATE приходит
		// один раз при создании окна
	case WM_CREATE:
		// получаем контекст устройства нашего окна
		hDC = GetDC(hWnd);
		// устанавливаем параметры контекста воспроизведения OpenGL
		SetWindowPixelFormat(hDC);
		// создаем контекст воспроизведения OpenGL
		hGLRC = wglCreateContext(hDC);
		// делаем его текущим
		wglMakeCurrent(hDC, hGLRC);
		// далее см. предыдущий раздел
		init();

		//	glGetIntegerv(GL_VIEWPORT, dims);
		//	width = dims[2];
		//	height = dims[3];
		//	resize();


		break;
		// это сообщение приходит при уничтожении окна
	case WM_DESTROY:
		// удаляем созданный выше
		// контекст воспроизведения OpenGL
		if (hGLRC)
		{
			wglMakeCurrent(NULL, NULL);
			wglDeleteContext(hGLRC);
		}
		// освобождаем контекст устройства нашего окна
		ReleaseDC(hWnd, hDC);
		//delete renderer;
		PostQuitMessage(0);
		break;
		// это сообщение приходит всякий раз,
		// когда нужно перерисовать окно
	case WM_PAINT:
		BeginPaint(hWnd, &ps);
		Render(model);
		EndPaint(hWnd, &ps);
		break;
	case WM_SIZE:
		AdjustSize();
		break;
	default:
		return DefWindowProc(hWnd, msg, wParam, lParam);
	}
	return 0;
}

HWND OpenGL_Renderer::createGLWindow(int w, int h)
{
	HWND hWnd;
	DWORD dwExWindowStyle;
	DWORD dwWindowStyle;

	// Register a window-class:
	WNDCLASS wcWindowClass;
	wcWindowClass.style = CS_HREDRAW | CS_VREDRAW | CS_OWNDC;
	wcWindowClass.lpfnWndProc = WindowFunc;
	wcWindowClass.cbClsExtra = 0;
	wcWindowClass.cbWndExtra = 0;
	wcWindowClass.hInstance = GetModuleHandle(NULL);
	wcWindowClass.hIcon = LoadIcon(NULL, IDI_APPLICATION);
	wcWindowClass.hCursor = LoadCursor(NULL, IDC_ARROW);
	wcWindowClass.hbrBackground = NULL;
	wcWindowClass.lpszMenuName = NULL;
	wcWindowClass.lpszClassName = "Teste";
	RegisterClass(&wcWindowClass);
	std::cout << "Registered window-class" << std::endl;

	// Set the window-style:
	dwExWindowStyle = WS_EX_APPWINDOW | WS_EX_WINDOWEDGE;
	dwWindowStyle = WS_OVERLAPPEDWINDOW | WS_CLIPSIBLINGS | WS_CLIPCHILDREN;

	RECT r;
	r.left = 0;
	r.right = w;
	r.top = 0;
	r.bottom = h;

	AdjustWindowRect(&r, dwWindowStyle, false);

	// Create a rendering-window:
	hWnd = CreateWindowEx(dwExWindowStyle, "Teste", "System", dwWindowStyle, 0, 0, r.right - r.left, r.bottom - r.top, NULL, NULL, GetModuleHandle(NULL), NULL);

	//updateWindow("Teste");

	std::cout << "Created window" << std::endl;
	return hWnd;
}


OpenGL_Renderer::OpenGL_Renderer(cv::Mat& Background)
{
	BG = Background.clone();
	width = BG.cols;
	height = BG.rows;

	hwnd = createGLWindow(width, height); // Hidden!!!
	glewInit();

	AdjustSize();
	// --------------------------------------
	// Загрузка и активация шейдерных программ
	// --------------------------------------
	shader.LoadFromFile(GL_VERTEX_SHADER, "../../shaders/VertexColors.vert");
	shader.LoadFromFile(GL_FRAGMENT_SHADER, "../../shaders/VertexColors.frag");
	shader.CreateAndLinkProgram();

	shader.AddUniform("MVP");
	shader.AddUniform("COL");
	shader.AddUniform("MVP_normals");
	glGenBuffers(1, &vertexBufferObject);
	glGenVertexArrays(1, &vao);
	glBindVertexArray(vao);

	pbo_size = width*height * 4;
	glGenBuffers(1, &pbo_id);
	glBindBuffer(GL_PIXEL_PACK_BUFFER, pbo_id);
	glBufferData(GL_PIXEL_PACK_BUFFER, pbo_size, 0, GL_DYNAMIC_READ);
	glBindBuffer(GL_PIXEL_PACK_BUFFER, 0);

	setTexture(BG, BG_textureID);
}

void OpenGL_Renderer::InitializeVertexBuffer(GLModel* mdl)
{
	if (buf == NULL)
	{
		buf = new float[mdl->nverts() * 8]; // 4 vertex coords, 2 texture coords, 4 normals (deleted in destructor)
	}

	memcpy((char*)buf, (char*)mdl->vertices.data(), mdl->nverts() * 3 * sizeof(float));
	memcpy((char*)(buf + mdl->nverts() * 3), (char*)mdl->tex_coords.data(), mdl->nverts() * 2 * sizeof(float));
	memcpy((char*)(buf + mdl->nverts() * 5), (char*)mdl->vertex_normals.data(), mdl->nverts() * 3 * sizeof(float));

	glBindBuffer(GL_ARRAY_BUFFER, vertexBufferObject);
	glBufferData(GL_ARRAY_BUFFER, mdl->nverts() * 8 * sizeof(float), buf, GL_STATIC_DRAW);
	glBindBuffer(GL_ARRAY_BUFFER, 0);

}

void OpenGL_Renderer::setProjectionMatrix(cv::Mat& proj)
{
#pragma omp parallel for
	for (int i = 0; i < 4; ++i)
	{
		for (int j = 0; j < 4; ++j)
		{
			Projection[i][j] = proj.at<float>(i, j);
		}
	}
}

void OpenGL_Renderer::setModelViewMatrix(cv::Mat& mv)
{
#pragma omp parallel for
	for (int i = 0; i < 4; ++i)
	{
		for (int j = 0; j < 4; ++j)
		{
			ModelView[i][j] = mv.at<float>(i, j);
		}
	}
}

void OpenGL_Renderer::setProjectionMatrix(glm::mat4 proj)
{
	Projection = proj;
}

void OpenGL_Renderer::setModelViewMatrix(glm::mat4 mv)
{
	ModelView = mv;
}

glm::mat4 OpenGL_Renderer::getProjectionMatrix(void)
{
	return Projection;
}

glm::mat4 OpenGL_Renderer::getModelViewMatrix(void)
{
	return ModelView;
}

void OpenGL_Renderer::setViewport(float l, float r, float b, float t, float n, float f)
{
	viewport[0] = l;
	viewport[1] = t;
	viewport[2] = r - l;
	viewport[3] = b - t;

	Viewport = glm::mat4(1);
	Viewport[3][0] = -(r + l) / (r - l);
	Viewport[3][1] = -(b + t) / (b - t);
	Viewport[3][2] = -(f + n) / (f - n);

	Viewport[0][0] = 2.0f / (r - l);
	Viewport[1][1] = 2.0f / (b - t);
	Viewport[2][2] = -2.0 / (f - n);

	vp[0][0] = 1;
	vp[1][1] = -1;
	vp[2][2] = 0;
	vp[3][0] = viewport[2];
	vp[3][1] = viewport[3];
	vp[3][0] = 1;

}

glm::vec4 OpenGL_Renderer::getViewport(void)
{
	return viewport;
}

void OpenGL_Renderer::Render(GLModel* mdl)
{
	glLoadIdentity();
	model = mdl;
	glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);

	glDisable(GL_POLYGON_OFFSET_FILL);

	glColor3f(1, 1, 1);
	// Background quad
	glDisable(GL_LIGHTING);
	glEnable(GL_TEXTURE_2D);
	glPolygonMode(GL_FRONT_AND_BACK, GL_FILL);
	glBindTexture(GL_TEXTURE_2D, BG_textureID);
	glBegin(GL_QUADS);                      // Draw A Quad
	glTexCoord2f(0.0f, 0.0f); glVertex3f(-1.0f, 1.0f, 0.5f);              // Top Left
	glTexCoord2f(1.0f, 0.0f); glVertex3f(1.0f, 1.0f, 0.5f);              // Top Right
	glTexCoord2f(1.0f, 1.0f); glVertex3f(1.0f, -1.0f, 0.5f);              // Bottom Right
	glTexCoord2f(0.0f, 1.0f); glVertex3f(-1.0f, -1.0f, 0.5f);              // Bottom Left
	glEnd();                            // Done Drawing The Quad

	glEnable(GL_LIGHTING);
	glBindTexture(GL_TEXTURE_2D, 0);
	glDisable(GL_TEXTURE_2D);
	// -------------------
	InitializeVertexBuffer(model);

	glEnable(GL_TEXTURE_2D);
	//glBindTexture(GL_TEXTURE_2D, Face_textureID);


	
	glm::mat4 MVP = Viewport * Projection * ModelView;
	glm::mat4 MVP_normals = glm::transpose(glm::inverse(MVP));
	glm::vec4 col= glm::vec4(241 / 255.0, 194 / 255.0, 125 / 255.0, 0.8);

	glEnable(GL_DEPTH_TEST);
	//glDepthMask(GL_FALSE);
	//glDepthFunc(GL_LEQUAL);
	glDisable(GL_CULL_FACE);
	//glEnable(GL_MULTISAMPLE);

	glEnable(GL_BLEND);

	//glBlendFunci(0, GL_ONE, GL_ONE);
	//glBlendEquationi(0, GL_FUNC_ADD);

	glBlendFunci(1, GL_DST_COLOR, GL_ZERO);
	glBlendEquationi(1, GL_FUNC_ADD);

	glEnable(GL_POLYGON_OFFSET_FILL);
	glPolygonOffset(1, -1);
	glPolygonMode(GL_FRONT_AND_BACK, GL_FILL);
	glBindBuffer(GL_ARRAY_BUFFER, vertexBufferObject);
	glEnableVertexAttribArray(0);
	glEnableVertexAttribArray(1);
	glEnableVertexAttribArray(2);
	glVertexAttribPointer(0, 3, GL_FLOAT, GL_FALSE, 0, 0);
	glVertexAttribPointer(1, 2, GL_FLOAT, GL_FALSE, 0, (void*)(model->nverts() * 3 * sizeof(float)));
	glVertexAttribPointer(2, 3, GL_FLOAT, GL_FALSE, 0, (void*)(model->nverts() * 5 * sizeof(float)));
	shader.Use();
	glUniform4fv(shader("COL"), 1, glm::value_ptr(col));
	glUniformMatrix4fv(shader("MVP"), 1, GL_FALSE, glm::value_ptr(MVP));
	glUniformMatrix4fv(shader("MVP_normals"), 1, GL_FALSE, glm::value_ptr(MVP_normals));
	glDrawArrays(GL_TRIANGLES, 0, model->nverts());
	shader.UnUse();


	glLoadMatrixf(glm::value_ptr(MVP));
	glDisable(GL_LIGHTING);
	glColor3f(0, 0, 0);
	glPolygonMode(GL_FRONT_AND_BACK, GL_LINE);
	glDrawArrays(GL_TRIANGLES, 0, model->nverts());

	glDisableVertexAttribArray(0);
	glDisableVertexAttribArray(1);
	glDisableVertexAttribArray(2);
	
	glDisable(GL_DEPTH_TEST);
	glPointSize(4);
	glPolygonMode(GL_FRONT_AND_BACK, GL_POINT);
	
	glBegin(GL_POINTS);
	for (int i = 0; i < mdl->jx.size(); ++i)
	{
		if (i == 17)
		{
			glColor3f(1, 0, 0);
		}
		else
		{
			glColor3f(1, 1, 1);
		}
		glVertex3f(mdl->jx[i], mdl->jy[i], mdl->jz[i]);
	}
	glEnd();

	glColor3f(1, 1, 1);
	glLineWidth(3);
	glBegin(GL_LINES);
	for (int i = 0; i < mdl->l1.size(); ++i)
	{
		if (mdl->l1[i] == -1 || mdl->l2[i] == -1)
		{
			continue;
		}
		glVertex3f(mdl->jx[mdl->l1[i]], mdl->jy[mdl->l1[i]], mdl->jz[mdl->l1[i]]);
		glVertex3f(mdl->jx[mdl->l2[i]], mdl->jy[mdl->l2[i]], mdl->jz[mdl->l2[i]]);
	}
	glEnd();
	glLineWidth(1);

	//glDepthMask(GL_TRUE);
	glDisable(GL_BLEND);

	glBindTexture(GL_TEXTURE_2D, 0);
	glDisable(GL_TEXTURE_2D);


	SwapBuffers(wglGetCurrentDC());
}



glm::vec3 mul(glm::mat4x3 m1, glm::vec4 v)
{
	glm::vec3 mult(0);
	for (int i = 0; i < 4; ++i)
	{
		for (int k = 0; k < 3; ++k)
		{
			mult[k] += m1[i][k] * v[i];
		}
	}
	return mult;
}

glm::mat4x3 mul(glm::mat4 m1, glm::mat4x3 m2)
{
	glm::mat4x3 mult(0);
	for (int i = 0; i < 4; ++i)
		for (int j = 0; j < 3; ++j)
			for (int k = 0; k < 4; ++k)
			{
				mult[i][j] += m1[i][k] * m2[k][j];
			}
	return mult;
}


glm::mat4x3& operator*(glm::mat4 m1, const glm::mat4x3& m2)
{
	return mul(m1, m2);
}

glm::vec3& operator*(glm::mat4x3 m, const glm::vec4& v)
{
	return mul(m, v);
}

glm::mat4x3 OpenGL_Renderer::getAffineCameraMatrix(void)
{
	return Projection * ModelView * vp;
}

void OpenGL_Renderer::projectPoints(std::vector<glm::vec3> vertCoords, std::vector<glm::vec2>& vertProjected)
{
	vertProjected.clear();
	// Transformation to screen space
	glm::mat4x3 MVP = Projection * ModelView * vp;
	glm::vec3 p_screen;
	size_t sz = vertCoords.size();
	for (int i = 0; i < sz; ++i)
	{
		p_screen = MVP*glm::vec4(vertCoords[i][0], vertCoords[i][1], vertCoords[i][2], 1.0);
		vertProjected.push_back(glm::vec2(p_screen[0], p_screen[1]));
	}
}


OpenGL_Renderer::~OpenGL_Renderer()
{
	// Free GPU buffers
	glDeleteBuffers(1, &vao);
	glDeleteBuffers(1, &pbo_id);
	glDeleteBuffers(1, &vertexBufferObject);
	// Kill window:
	DestroyWindow(hwnd);
	UnregisterClass("Teste", GetModuleHandle(NULL));
	delete buf;
}
