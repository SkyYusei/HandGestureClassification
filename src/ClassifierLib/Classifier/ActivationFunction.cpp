#include "ANNClassifier.h"

arma::mat sigmoid(const arma::mat & x){
	return 1.0 / (1.0 + arma::exp(-x));	
}

arma::mat sigmoid_d(const arma::mat & x)
{
	return  sigmoid(x) * (1-sigmoid(x));
}

