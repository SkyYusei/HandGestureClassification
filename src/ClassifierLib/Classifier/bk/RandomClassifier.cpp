#include "RandomClassifier.h"

RandomClassifier::RandomClassifier() {
}

RandomClassifier::~RandomClassifier() {
}

void RandomClassifier::train(const arma::mat& positive,const arma::mat& negative){
	return;
}

void RandomClassifier::test(const arma::mat& data, std::vector<int>& prediction)
{
	int size = data.n_cols;
	prediction.clear();
	for(int i = 0;i<size;i++){
		prediction.push_back(rand()%2);
		//prediction.push_back(1);
	}
}




