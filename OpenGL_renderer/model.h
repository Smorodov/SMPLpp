#ifndef __MODEL_H__
#define __MODEL_H__
#include <vector>
#include <string>
#include <glm/vec3.hpp> // glm::vec3
#include <glm/vec4.hpp> // glm::vec4
#include <glm/mat4x4.hpp> // glm::mat4
#include <glm/gtc/matrix_transform.hpp> // glm::translate, glm::rotate, glm::scale, glm::perspective
#include "Eigen/Eigen"

class GLModel
{
	private:
	public:
		GLModel();
		~GLModel();
		void addVertex(float vx, float vy, float vz, float ty, float tx);
		void addVertex(float vx, float vy, float vz);
		void addNormal(float nx, float ny, float nz);
		void addFace(int v1, int v2, int v3);
		void addTexCoord(float  tx, float  ty);
		void clearMesh(void);
		void clearVertices(void);
		void clearTexCoords(void);
		int nverts();
		int nfaces();
		glm::vec3 fetch_vertex_coord(int i);
		glm::vec2 fetch_vertex_tex_coord(int i);
		glm::vec3 fetch_vertex_coord(int iface, int nthvert);
		glm::ivec3 face(int idx);
		std::vector<glm::ivec3> faces;
		std::vector<glm::vec3> faces_normals;  // TODO compute faces normals and vertices normals
		std::vector<glm::vec3> vertices;
		std::vector<glm::vec3> vertex_normals;
		std::vector<glm::vec2> tex_coords;
		Eigen::MatrixXf texture;
};
#endif //__MODEL_H__

