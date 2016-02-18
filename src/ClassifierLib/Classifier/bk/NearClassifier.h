#pragma once
#include "BaseClassifier.h"

class NearClassifier:public BaseClassifier {
protected:
	arma::mat pos;
	arma::mat neg;
	double threshold;
public:
	NearClassifier() ;
	~NearClassifier() ;

    void train(const arma::mat& positive,const arma::mat& negative);
    void test(const arma::mat& data, std::vector<int>& prediction) ;
    bool save(const char* file);
    bool load(const char* file);
};

