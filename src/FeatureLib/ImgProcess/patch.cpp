#include "patch.h"

using namespace arma;
void getPatch(const Image * img, int patch_w, int patch_h, 
		mat& patches, int&w, int&h)
{
	int img_w = img->width;
	int img_h = img->height;

	int patch_size = patch_w * patch_h;

	w = img_w - patch_w;
	h = img_h - patch_h;
	int s = w*h;

	fmat fm(grayImg2Float(img),img_h,img_w);
	mat m = conv_to<mat>::from(fm);
	int pid = 0;

	patches.set_size(patch_size,s);
	for(int c = 0;c<s;c++){
		int sr = c/w;
		int sc = c%w;
		for(int r = 0,tr=0,tc=0;r<patch_size;r++,tc++){
			if(tc == patch_w){
				tc = 0;
				tr ++;
			}
			patches.at(r,c)=m.at(sr+tr, sc+tc);
		}
	}
//	normalise(patches);
}


void nms(arma::mat& m){}

