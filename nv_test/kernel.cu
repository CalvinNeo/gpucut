#define UNICODE
#include <Windows.h>
#include <iostream>
#include "stdio.h"
#include <cctype>
#include <vector>
#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include "bitmap.h"

#define uint unsigned int 
#define ulong unsigned long long
#define ufloat float

using namespace std;

__global__  
void inKernel(char* result,uint points_count,uint poly_count,uint circle_count,uint line_count,uint width,uint height
	,uint* vcount_poly,ufloat* polys, ufloat *circles, ufloat *lines)
{
	uint i = blockIdx.x*blockDim.x+threadIdx.x;
	bool is_in=false;
	while(i<points_count){
		bool is_shape = false;
		uint px = i % width,py = i / width;
		for (uint i = 0; i < circle_count; i++)
		{
			ufloat x=circles[3*i],y=circles[3*i+1],r=circles[3*i+2];
			ufloat dist = (px-x)*(px-x)+(py-y)*(py-y)-r*r;
			if (dist<5e2 && dist>-5e2)
			{
				is_shape = true;
				goto POINT_CONSIST_SHAPE;
			}
		}
		for (uint i = 0; i < line_count; i++)
		{
			ufloat x1=lines[4*i],y1=lines[4*i+1],x2=lines[4*i+2],y2=lines[4*i+3];
			ufloat dist = ((y1-y2)*px-(x1-x2)*py-(y1-y2)*x1+y1*(x1-x2))/((y1-y2)*(y1-y2)+(x1-x2)*(x1-x2));
			if (dist<1e-3 && dist>-1e-3)
			{
				is_shape = true;
				goto POINT_CONSIST_SHAPE;
			}
		}
		result[(py*width+px)*3] = 0xff;
		result[(py*width+px)*3+1] = 0xff;
		result[(py*width+px)*3+2] = 0xff;
	POINT_CONSIST_SHAPE:
		for (int poly_index = 0; poly_index < poly_count; poly_index++)
		{
			uint pi,pj = vcount_poly[poly_index]-1;
			is_in = false;
			for(pi=0; pi<vcount_poly[poly_index]; pi++) {
				ufloat polyxi=polys[2*pi],polyyi=polys[2*pi+1],polyxj=polys[2*pj],polyyj=polys[2*pj+1];
				if(polyyi<py && polyyj>=py || polyyj<py && polyyi>=py) {//两点不同侧
					ufloat dist_p = (py-polyyi)*(polyxj-polyxi)-(px-polyxi)*(polyyj-polyyi);
					ufloat dist = (py-polyyi)*(polyxj-polyxi)/(polyyj-polyyi)-px+polyxi;
					if (dist_p<5e2 && dist_p>-5e2)
					{
						result[(py*width+px)*3] = 0xff;
						result[(py*width+px)*3+1] = 0x00;
						result[(py*width+px)*3+2] = 0x00;
						goto END;
					}
					if(dist<0) {
						is_in=!is_in;
					}
				}
				pj=pi;
			}
		}
		if (is_in && is_shape)
		{
			result[(py*width+px)*3] = 0;
			result[(py*width+px)*3+1] = 0;
			result[(py*width+px)*3+2] = 0;
		}else{
			result[(py*width+px)*3] = 0x00;
			result[(py*width+px)*3+1] = 0x00;
			result[(py*width+px)*3+2] = 0xff;
		}
	END:
		i+=blockDim.x*gridDim.x;
		__nop;
	}
}
int main(){
	int err = 0;
	HANDLE hfile = CreateFile(L"F:\\Codes\\C++\\cnsoftbei_cut\\cache\\20911C78012C0A690C4A6B34F63FC0A9\\case2.dat",GENERIC_READ,0,NULL,OPEN_ALWAYS,FILE_ATTRIBUTE_READONLY,NULL);
	HANDLE hmapping = CreateFileMapping(hfile,0,PAGE_READONLY,0,0,0);
	LPVOID buffer_void = MapViewOfFile(hmapping,FILE_MAP_READ,0,0,0);
	uint* buffer_int = (uint*)buffer_void;
	ufloat* buffer_float = (ufloat*)buffer_void;

	uint poly_count=*(buffer_int++),circle_count=*(buffer_int++),line_count=*(buffer_int++),polyvertex_count = *(buffer_int++);
	uint* vcount_poly = new uint[poly_count];//每个多边形的点数/边数
	ufloat* poly_vertexs = new ufloat[polyvertex_count*2];//所有多边形的点坐标,排列为x1,y1,x2,y2
	ufloat* circles = new ufloat[circle_count*3];//所有圆的坐标,排列为x1,y1,r1,x2,y2,r2
	ufloat* lines = new ufloat[line_count*4];//所有线的坐标,排列为l1p1x,l1p1y,l1p2x,l1p2y,l2p1x,l2p1y,l2p2x,l2p2y
	
	for (int i = 0; i < poly_count; i++)
	{
		vcount_poly[i] = *(buffer_int++);
	}
	buffer_float = (ufloat*)buffer_int;
	for (int i = 0,k = 0; i < poly_count; i++)
	{
		for (int j = 0; j < vcount_poly[i]; j++)
		{
			for (int data = 0; data < 2; data++){
				ufloat f = *(buffer_float++);
				poly_vertexs[k++] = f;
			}
		}
	}
	for (int i = 0; i < circle_count; i++)
	{
		for (int data = 0; data < 3; data++)circles[i+data] = *(buffer_float++);
	}
	for (int i = 0; i < line_count; i++)
	{
		for (int data = 0; data < 4; data++){
			ufloat f = *(buffer_float++);
			lines[i+data] = f;
		}
	}
	//points
	char* result = new char[1440*900*3];
	uint width=1440,height=900;
	uint point_count=width*height;

	//gpu
	char* gpu_result;
	uint* gpu_vcount_poly;
	ufloat* gpu_poly_vertexs;
	ufloat* gpu_circles;
	ufloat* gpu_lines;

	err = cudaDeviceReset();
	err = cudaSetDevice(0);
	if(cudaMalloc((void**)&gpu_result,1440*900*3)==cudaSuccess){
		 err = cudaMemcpy(gpu_result, result, 1440*900*3, cudaMemcpyHostToDevice);
	}
	if(cudaMalloc((void**)&gpu_vcount_poly, poly_count*sizeof(uint))==cudaSuccess){
		 err = cudaMemcpy(gpu_vcount_poly, vcount_poly, poly_count*sizeof(uint), cudaMemcpyHostToDevice);
	}
	if(cudaMalloc((void**)&gpu_poly_vertexs,2*polyvertex_count*sizeof(ufloat))==cudaSuccess){
		 err = cudaMemcpy(gpu_poly_vertexs, poly_vertexs, 2*polyvertex_count*sizeof(ufloat), cudaMemcpyHostToDevice);
	}
	if(cudaMalloc((void**)&gpu_circles, 3*circle_count*sizeof(ufloat))==cudaSuccess){
		 err = cudaMemcpy(gpu_circles, circles, 3*circle_count*sizeof(ufloat), cudaMemcpyHostToDevice);
	}
	if(cudaMalloc((void**)&gpu_lines, 4*line_count*sizeof(ufloat))==cudaSuccess){
		 err = cudaMemcpy(gpu_lines, lines, 4*line_count*sizeof(ufloat), cudaMemcpyHostToDevice);
	}
	uint numBlock=128,threadsPerBlock=128;

	inKernel <<<numBlock,threadsPerBlock>>>(gpu_result,point_count,poly_count,circle_count,line_count,width,height
		,gpu_vcount_poly,gpu_poly_vertexs,gpu_circles,gpu_lines);

	err = cudaDeviceSynchronize();

	cudaMemcpy(result, gpu_result, 1440*900*3, cudaMemcpyDeviceToHost);
	tobitmap(width,height,result);

	//gc
    cudaFree(gpu_vcount_poly);cudaFree(gpu_poly_vertexs);cudaFree(gpu_circles);cudaFree(gpu_lines);
	system("pause");
}