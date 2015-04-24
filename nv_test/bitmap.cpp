#include <stdio.h>
#include <stdlib.h>
#include <conio.h>
#include <windows.h>
#include <string>
#include <minwindef.h>
#include <iostream>
#include "bitmap.h"

using namespace std;

BITMAPFILEHEADER getbitmapfileheader(int size,int bitcnt)
{
	BITMAPFILEHEADER bmpfileheader;
	bmpfileheader.bfOffBits=54+ ((bitcnt==8)?1024:0);
	bmpfileheader.bfSize=bmpfileheader.bfOffBits+size;
	bmpfileheader.bfReserved1=0;
	bmpfileheader.bfReserved2=0;
	bmpfileheader.bfType=0x4D42;
	return bmpfileheader;
}
BITMAPINFOHEADER getbitmapinfoheader(int size,int bitcnt,int w,int h)
{
	BITMAPINFOHEADER bmpinfoheader;
	bmpinfoheader.biBitCount=bitcnt;
	bmpinfoheader.biClrImportant=0;
	bmpinfoheader.biClrUsed=0;
	bmpinfoheader.biCompression=0;
	bmpinfoheader.biHeight=h;
	bmpinfoheader.biPlanes=1;
	bmpinfoheader.biSize=40;
	bmpinfoheader.biSizeImage=size;
	bmpinfoheader.biWidth=w;
	bmpinfoheader.biXPelsPerMeter=0;
	bmpinfoheader.biYPelsPerMeter=0;
	return bmpinfoheader;
}

int tobitmap(int width,int height,char* srcbmp){
	FILE * fp;
	int bitcount = 24;
	int size = ((width*bitcount/8)+3)*height;
	BITMAPFILEHEADER strHead = getbitmapfileheader(size,24);  
	RGBQUAD strPla[256]; 
	BITMAPINFOHEADER strInfo = getbitmapinfoheader(size,24,width,height);  

	if((fp=fopen("b.bmp","wb"))==NULL)  
    {  
        cout<<"create the bmp file error!"<<endl;  
        return NULL;  
    }  
	BITMAPFILEHEADER bmpfileheader=getbitmapfileheader(size,24);
	fwrite(&bmpfileheader,sizeof(BITMAPFILEHEADER),1,fp);
	BITMAPINFOHEADER bmpinfoheader=	getbitmapinfoheader(size,24,width,height);
	fwrite(&bmpinfoheader,sizeof(BITMAPINFOHEADER),1,fp);
	fwrite(srcbmp,size,1,fp);
	return 0;
}