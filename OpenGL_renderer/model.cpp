#include <iostream>
#include <fstream>
#include <sstream>
#include "model.h"
#include <opencv2/opencv.hpp>

GLModel::GLModel() : vertices(), faces(), tex_coords()
{

}

void GLModel::addVertex(float vx, float vy, float vz, float tx, float ty)
{
	vertices.push_back(glm::vec3(vx,vy,vz));
	tex_coords.push_back(glm::vec2(tx,ty));
}

void GLModel::addFace(int v1, int v2, int v3)
{
	faces.push_back(glm::ivec3(v1, v2, v3));
}

void GLModel::clearMesh(void)
{
	jx.clear();
	jy.clear();
	jz.clear();
	l1.clear();
	l2.clear();
	
	vertices.clear();
	tex_coords.clear();
	vertex_normals.clear();
	faces_normals.clear();
	faces.clear();
}


void GLModel::addVertex(float vx, float vy, float vz)
{
	vertices.push_back(glm::vec3(vx, vy, vz));
}

void GLModel::addNormal(float nx, float ny, float nz)
{
	vertex_normals.push_back(glm::vec3(nx, ny, nz));
}

void GLModel::addTexCoord(float  tx, float  ty)
{
	tex_coords.push_back(glm::vec2(tx, ty));
}

void GLModel::clearVertices(void)
{
	vertices.clear();
}

void GLModel::clearTexCoords(void)
{
	tex_coords.clear();
}


GLModel::~GLModel() {}

int GLModel::nverts()
{
	return (int)vertices.size();
}

int GLModel::nfaces()
{
	return (int)faces.size();
}

glm::ivec3 GLModel::face(int idx)
{
	return faces[idx];
}

glm::vec3 GLModel::fetch_vertex_coord(int i)
{
	return vertices[i];
}

glm::vec2 GLModel::fetch_vertex_tex_coord(int i)
{
	return tex_coords[i];
}

glm::vec3 GLModel::fetch_vertex_coord(int iface, int nthvert)
{
	return vertices[faces[iface][nthvert]];
}

