#pragma once
#include "BaseClassifier.h"

class RandomClassifier:public BaseClassifier {
public:
	RandomClassifier() ;
	~RandomClassifier() ;

    void train(const arma::mat& positive,const arma::mat& negative);
    void test(const arma::mat& data, std::vector<int>& prediction) ;
};



