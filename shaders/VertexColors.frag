#version 330
uniform sampler2D sampler;

smooth in vec4 col;
smooth in vec2 tex_coord;
smooth in vec3 normals;
smooth in vec3 pos;

out vec4 outputColor;

float DistToLine(vec2 pt1, vec2 pt2, vec2 testPt)
{
  vec2 lineDir = pt2 - pt1;
  vec2 perpDir = vec2(lineDir.y, -lineDir.x);
  vec2 dirToPt1 = pt1 - testPt;
  return abs(dot(normalize(perpDir), dirToPt1));
}

void main()
{
   float Shininess=16;
   float AmbientIntensity=0.8;
   vec3 SpecularColor=vec3(0.5,0.5,0.5);
   vec3 LightPosition=vec3(30,-30,100);

   vec3 L = normalize(LightPosition - pos);  

   vec4 Idiff = vec4(SpecularColor,1.0) * max(pow(dot(normalize(normals),L),Shininess), 0.0);
  /* 
   vec2 p1=vec2(0.0,0.0);
   vec2 p2=vec2(0.0,1.0);
   vec2 p3=vec2(1.0,1.0);
   vec2 p=tex_coord;
   float d1=DistToLine(p1, p2, p);
   float d2=DistToLine(p2, p3, p);
   float d3=DistToLine(p3, p1, p);
   float d=min(min(d1,d2),d3);
   float k=8*d/sqrt(2);
	if(k>1){k=1;}
   if(k<0){k=0;}
   vec4 kv=vec4(sqrt(k),sqrt(k),sqrt(k),1);
   */
   outputColor = vec4( vec3(clamp( col*AmbientIntensity+Idiff, 0.0, 1.0) ) ,col[3]);

   //outputColor = vec4(vec3(clamp( texture2D(sampler, tex_coord)*AmbientIntensity+Idiff, 0.0, 1.0)),1);
   
}
