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

using ms = std::chrono::milliseconds;
using clk = std::chrono::system_clock;

int main(int argc, char const* argv[])
{
	std::string modelPath = "female_model.npz";
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
	//glDeleteTextures(1, &renderer->Face_textureID);
	int k = 0;


	auto begin = clk::now();
	SINGLE_SMPL::get()->setDevice(cuda);
	SINGLE_SMPL::get()->setModelPath(modelPath);
	SINGLE_SMPL::get()->init();
	
	torch::Tensor vertices;
	torch::Tensor beta;
	torch::Tensor theta;

	beta = 0.3 * torch::rand({ BATCH_SIZE, SHAPE_BASIS_DIM });

	float pose_rand_amplitude = 1;
	while (k != 27)
	{
		auto end = clk::now();
		auto duration = std::chrono::duration_cast<ms>(end - begin);
		std::cout
 << "Time duration to load SMPL: " << (double)duration.count() / 1000 << " s" << std::endl;


		theta = pose_rand_amplitude * torch::rand({ BATCH_SIZE, JOINT_NUM, 3 })- pose_rand_amplitude/2 * torch::ones({ BATCH_SIZE, JOINT_NUM, 3 });
				
		theta.data<float>()[0] = 0;
		theta.data<float>()[1] = 0;
		theta.data<float>()[2] = 0;

		for (int i = 0; i < JOINT_NUM; ++i)
		{
			//theta.data<float>()[i*3+0] = 0; // rx
			theta.data<float>()[i*3+1] = 0; // ry
			theta.data<float>()[i*3+2] = 0; // rz
		}

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
			SINGLE_SMPL::get()->setVertPath("model.obj");
			SINGLE_SMPL::get()->out(0);
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

		
		std::vector<float> jx;
		std::vector<float> jy;
		std::vector<float> jz;

		std::vector<int64_t> l1;
		std::vector<int64_t> l2;

		SINGLE_SMPL::get()->getSkeleton(0,
			l1,
			l2,
			jx,
			jy,
			jz);



		double t = (double)cv::getTickCount();
		
		glm::mat4 Projection = glm::mat4(1.0f);
		glm::mat4 ModelView = glm::mat4(1.0f);

		float scl = 1;
		float tx = 0;
		float ty = 0.35;
		float tz = 0;
		float rx = 0;
		float ry = 0;
		float rz = 0;

		ModelView = glm::translate(ModelView, glm::vec3(bg.cols / 2, bg.rows / 2, 0));
		ModelView = glm::rotate(ModelView, float(rx*CV_PI/180.0), glm::vec3(1, 0, 0));
		ModelView = glm::rotate(ModelView, float(ry), glm::vec3(0, 1, 0));
		ModelView = glm::rotate(ModelView, float(rz), glm::vec3(0, 0, 1));

		ModelView = glm::scale(ModelView, glm::vec3(bg.cols / 2*scl, bg.cols / 2 * scl, bg.cols / 2 * scl));
		
		ModelView = glm::translate(ModelView, glm::vec3(tx, ty, tz));
		
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
			model.addVertex((vx[vi1]), (vy[vi1]), (vz[vi1]));
			model.addVertex((vx[vi2]), (vy[vi2]), (vz[vi2]));
			model.addVertex((vx[vi3]), (vy[vi3]), (vz[vi3]));

			glm::vec3 a, b, c, v1, v2, N;
			a = glm::vec3( vx[vi1], vy[vi1], vz[vi1]);
			b = glm::vec3(vx[vi2], vy[vi2], vz[vi2]);
			c = glm::vec3(vx[vi3], vy[vi3], vz[vi3]);
			// compute normals
			v1 = b - a;
			v2 = b - c;
			N[0] = v1[1] * v2[2] - v1[2] * v2[1];
			N[1] = v1[2] * v2[0] - v1[0] * v2[2];
			N[2] = v1[0] * v2[1] - v1[1] * v2[0];
			N[0] = N[0]/N.length();
			N[1] = N[1] / N.length();
			N[2] = N[2] / N.length();
			model.addNormal(N[0], N[1], N[2]);
			model.addNormal(N[0], N[1], N[2]);
			model.addNormal(N[0], N[1], N[2]);

			model.addTexCoord(0, 0);
			model.addTexCoord(0, 1);
			model.addTexCoord(1, 1);
		}
		model.jx = jx;
		model.jy = jy;
		model.jz = jz;
		model.l1 = l1;
		model.l2 = l2;
		renderer->Render(&model);
		renderer->getImage(face);
		/*
		std::vector<glm::vec3> vert3d;
		std::vector<glm::vec2> vert2d;
		for (int i = 0; i < vx.size(); ++i)
		{
			vert3d.push_back(glm::vec3(vx[i], vy[i], vz[i]));
		}
		renderer->projectPoints(vert3d, vert2d);

		for (int i = 0; i < vx.size(); ++i)
		{
			cv::Point p = cv::Point(vert2d[i][0], vert2d[i][1]);
			std::cout << p << std::endl;
			//cv::circle(face,p, 2, cv::Scalar(0, 255, 0), -1);

			cv::putText(face, //target image
				std::to_string(i), //text
				p, //top-left position
				cv::FONT_HERSHEY_SIMPLEX,
				0.3,
				CV_RGB(255, 255, 0), //font color
				1);

		}
		*/
		t = ((double)cv::getTickCount() - t) / cv::getTickFrequency();
		std::cout << "Elapsed time (seconds) :" << t;
		cv::imshow("face", face);
		k = cv::waitKey();
	}
	SINGLE_SMPL::destroy();
	delete renderer;
	cv::imwrite("result.jpg", face);
    return 0;
}


//===== CLEAN AFTERWARD =======================================================

#undef SINGLE_SMPL

//=============================================================================
