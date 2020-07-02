#define SINGLE_SMPL smpl::Singleton<smpl::SMPL>
#include <chrono>
#include <torch/torch.h>
#include "definition/def.h"
#include "toolbox/Singleton.hpp"
#include "smpl/SMPL.h"

#include "opencv2/opencv.hpp"
#include <limits>
#include "OpenGL_Renderer.h"
#include "GLSLShader.h"

// ----------------------------------------------------------------------
// Resizes image to given size, preserving sides ratio
// ----------------------------------------------------------------------
float rescale(cv::Mat& image, int maxw, int maxh)
{
	float scale = 1;
	if (image.cols > image.rows)
	{
		if (image.cols > maxw)
		{
			scale = float(maxw) / image.cols;
		}
	}
	else
	{
		if (image.rows > maxh)
		{
			scale = float(maxh) / image.rows;
		}
	}
	cv::resize(image, image, cv::Size(image.cols * scale, image.rows * scale));
	return scale;
}
// ----------------------------------------------------------------------
//
// ----------------------------------------------------------------------
void setTexture(Eigen::MatrixXf& image, GLuint& _textureID, int tex_size)
{
	glGenTextures(1, &_textureID);
	glBindTexture(GL_TEXTURE_2D, _textureID);
	glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_NEAREST);
	glPixelStorei(GL_UNPACK_ROW_LENGTH, tex_size);
	glTexImage2D(GL_TEXTURE_2D, 0, 3, tex_size, tex_size, 0, GL_RGB, GL_FLOAT, image.data());
}


using ms = std::chrono::milliseconds;
using clk = std::chrono::system_clock;

int main(int argc, char const* argv[])
{
	std::string modelPath = "MANO_left.npz";
	torch::Device cuda(torch::kCPU);
	cuda.set_index(0);

	cv::Mat face;
	GLModel model;
	OpenGL_Renderer* renderer;
	int max_width = 800;
	int max_height = 800;
	cv::Mat bg = cv::imread("F:/ImagesForTest/lena.jpg", 1);
	//bg = cv::Scalar::all(0);
	//cv::flip(bg, bg, 1);
	//cv::resize(bg, bg, cv::Size(512, 512));
	float scale = rescale(bg, max_width, max_height);
	srand((unsigned int)time(0));
	renderer = new OpenGL_Renderer(bg);
	glDeleteTextures(1, &renderer->Face_textureID);
	int k = 0;


	auto begin = clk::now();
	SINGLE_SMPL::get()->setDevice(cuda);
	SINGLE_SMPL::get()->setModelPath(modelPath);
	SINGLE_SMPL::get()->init();
	
	torch::Tensor vertices;
	torch::Tensor beta;
	torch::Tensor theta;

	beta = 0.3 * torch::rand({ BATCH_SIZE, SHAPE_BASIS_DIM });

	while (k != 27)
	{
		auto end = clk::now();
		auto duration = std::chrono::duration_cast<ms>(end - begin);
		std::cout
 << "Time duration to load SMPL: " << (double)duration.count() / 1000 << " s" << std::endl;


		theta = 0.6 * torch::rand({ BATCH_SIZE, JOINT_NUM, 3 })- 0.3 * torch::ones({ BATCH_SIZE, JOINT_NUM, 3 });
				
		theta.data<float>()[0] = 0;
		theta.data<float>()[1] = 0;
		theta.data<float>()[2] = 0;
		try
		{
			const int64_t LOOPS = 1;
			duration = std::chrono::duration_cast<ms>(end - end);// reset duration

			begin = clk::now();
			SINGLE_SMPL::get()->launch(beta, theta);

			end = clk::now();
			duration += std::chrono::duration_cast<ms>(end - begin);
			std::cout << "Time duration to run SMPL: " << (double)duration.count() / LOOPS << " ms" << std::endl;

			vertices = SINGLE_SMPL::get()->getVertex();
		}
		catch (std::exception& e)
		{
			std::cerr << e.what() << std::endl;
		}

		std::vector<float> vx;
		std::vector<float> vy;
		std::vector<float> vz;
		std::vector<size_t> f1;
		std::vector<size_t> f2;
		std::vector<size_t> f3;
		SINGLE_SMPL::get()->getVandF(0, vx, vy, vz, f1, f2, f3);

		



		double t = (double)cv::getTickCount();
		
		glm::mat4 Projection = glm::mat4(1.0f);
		glm::mat4 ModelView = glm::mat4(1.0f);

		ModelView = glm::translate(ModelView, glm::vec3(bg.cols / 2, bg.rows / 2, 0));
		float scl = 6;
		float tx = 0;
		float ty = 0;
		float tz = 0;
		ModelView = glm::scale(ModelView, glm::vec3(bg.cols / 2, bg.cols / 2, bg.cols / 2));

		renderer->setModelViewMatrix(ModelView);
		renderer->setProjectionMatrix(Projection);
		model.clearMesh();
		for (int i = 0; i < f1.size(); i++)
		{
			int vi1 = f1[i] - 1;
			int vi2 = f2[i] - 1;
			int vi3 = f3[i] - 1;
			//std::cout << vx[vi1] << " " << vy[vi1] << " " << vz[vi1] << std::endl;
			model.addFace(vi1, vi2, vi3);
			model.addVertex((tx + vx[vi1]) * scl, (ty + vy[vi1]) * scl, (tz + vz[vi1]) * scl);
			model.addVertex((tx + vx[vi2]) * scl, (ty + vy[vi2]) * scl, (tz + vz[vi2]) * scl);
			model.addVertex((tx + vx[vi3]) * scl, (ty + vy[vi3]) * scl, (tz + vz[vi3]) * scl);

			Eigen::Vector3f a, b, c, v1, v2, N;
			a << vx[vi1], vy[vi1], vz[vi1];
			b << vx[vi2], vy[vi2], vz[vi2];
			c << vx[vi3], vy[vi3], vz[vi3];
			// compute normals
			v1 = b - a;
			v2 = b - c;
			N << v1[1] * v2[2] - v1[2] * v2[1],
				v1[2] * v2[0] - v1[0] * v2[2],
				v1[0] * v2[1] - v1[1] * v2[0];
			N = N.normalized().eval();

			model.addNormal(N(0), N(1), N(2));
			model.addNormal(N(0), N(1), N(2));
			model.addNormal(N(0), N(1), N(2));

			model.addTexCoord(0, 0);
			model.addTexCoord(0, 1);
			model.addTexCoord(1, 1);
		}
		renderer->Render(&model);
		renderer->getImage(face);
		t = ((double)cv::getTickCount() - t) / cv::getTickFrequency();
		std::cout << "Elapsed time (seconds) :" << t;
		cv::imshow("face", face);
		k = cv::waitKey();
	}
	SINGLE_SMPL::destroy();
	delete renderer;

    return 0;
}


//===== CLEAN AFTERWARD =======================================================

#undef SINGLE_SMPL

//=============================================================================
