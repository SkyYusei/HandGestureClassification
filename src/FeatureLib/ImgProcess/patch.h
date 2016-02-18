#pragma once
#include "codelib.h"
#include <armadillo>
// patches must be empty
// add each patch of the img to the matrix patches
// normalise the patch
// return size of the patch area
void getPatch(const Image * img, int patch_w, int patch_h, 
	arma::mat& patches,
	int& w, int& h);

void nms(arma::mat& m);

