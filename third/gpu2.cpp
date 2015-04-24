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
				if(polyyi<py && polyyj>=py || polyyj<py && polyyi>=py) {
					ufloat dist1 = (py-polyyi)*(polyxj-polyxi)-(px-polyxi)*(polyyj-polyyi);
					ufloat dist = (py-polyyi)*(polyxj-polyxi)/(polyyj-polyyi)-px+polyxi;
					if (dist1<5e2 && dist1>-5e2)
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
		i+=16384;
	}
}
