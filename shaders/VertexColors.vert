#version 330

layout (location = 0) in vec3 position;
layout (location = 1) in vec2 texCoord;
layout (location = 2) in vec3 Normals;

uniform mat4 MVP;
uniform mat4 MVP_normals;
uniform vec4 COL;
smooth out vec4 col;
smooth out vec2 tex_coord;
smooth out vec3 normals;
smooth out vec3 pos;

void main()
{
pos=vec3(MVP * vec4(position,1.0));
normals =  vec3(MVP_normals*vec4(Normals,1.0));
gl_Position=vec4(pos,1.0);
tex_coord = texCoord;
col=COL;
}
