#pragma once


#include <armadillo>
#include <vector>
#include "codelib.h"
#include "Base_M_Classifier.h"

class Base_2_Classifier: public Base_M_Classifier{
public:
	Base_2_Classifier() {};
	virtual ~Base_2_Classifier() {};

    virtual void train2(const arma::mat& positive,const arma::mat& negative) = 0;
	// 1 column each sample
	// label is a vertical vector
	virtual void train(const arma::mat& data, const arma::vec& label){
		arma::mat positive;
		arma::mat negative;
		for(int i=0;i<label.n_cols;i++){
			if(label[i]==0)
				negative.insert_cols(negative.n_cols, data.col(i));
			else			
				positive.insert_cols(positive.n_cols, data.col(i));
		}
		train(positive,negative);
		train_flag = true;
	}
};



