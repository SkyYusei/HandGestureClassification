#pragma once
#include "Base_2_Classifier.h"

class Naive_2_WClassifier:public Base_2_Classifier {
	protected:
		double threshold;
		int direction;
		int dimension;
		int num_segment;
		virtual void getPredictionWeight(const arma::mat& data, arma::mat& prediction);

	public:
		Naive_2_WClassifier() ;
		~Naive_2_WClassifier() ;

		virtual void train2(const arma::mat& positive,const arma::mat& negative);
//		void test(const arma::mat& data, std::vector<int>& prediction) ;

		virtual bool save( FILE * fptr);
		virtual bool load( FILE * fptr);
		virtual bool setParams (const char * params);
		virtual void printParams () ;

};

