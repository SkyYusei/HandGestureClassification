#include "NearClassifier.h"
#include <armadillo>
using namespace arma;

void NearClassifier::train(const arma::mat& positive,const arma::mat& negative){
	pos = trans((positive));
	neg = trans((negative));

	threshold	= sum(sum(square(pos))) / pos.n_rows
				- sum(sum(square(negative))) / neg.n_rows;
	threshold /= 2 ;
}


void NearClassifier::test(const arma::mat& data, std::vector<int>& prediction)
{
	prediction.clear();
	rowvec c0 = sum(neg * data) / neg.n_rows;
	rowvec c1 = sum(pos * data) / pos.n_rows;
	int size = data.n_cols;
	for(int i = 0;i<size;i++){
		prediction.push_back(c1(i)-c0(i) > threshold);
	}
}

NearClassifier::NearClassifier() {
}

NearClassifier::~NearClassifier() {
}

bool NearClassifier::save(const char* file){
	return true;
}

bool NearClassifier::load(const char* file){
	return true;
}




