#include "pca.h"
#include <assert.h>

using namespace arma;
void pca(const arma::mat& data, arma::mat& eigen, int num)
{
	mat U,V;
	vec s;
	svd_econ(U, s, V, data,"left");
	assert(num<=U.n_cols);
	if(num<=0) num = U.n_cols;
	eigen = U.cols(0,num-1);
}

void pca_feature(const arma::mat& data, const arma::mat& eigen, arma::mat& feature, bool err_term)
{
	feature = eigen.t()*data;
	if(err_term){
		mat reconstruct = eigen*feature;
		mat diff = reconstruct-data;
		rowvec err = sqrt(sum(square(diff)));
		feature.insert_rows(feature.n_rows,err);
	}
}

