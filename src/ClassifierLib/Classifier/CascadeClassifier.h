#pragma once
#include "Base_M_Classifier.h"
#include <vector>

class CascadeClassifier:public Base_M_Classifier {
	protected:
		std::vector<Base_M_Classifier*> stage_classifiers;
		virtual void getPredictionWeight(const arma::mat& data, arma::mat& prediction);
		
	public:
		void addClassifier(const char * classifier_name="navie",
			const char * params="", int num = 1);

	public:
		CascadeClassifier() ;
		~CascadeClassifier() ;

		virtual bool setParams (const char * params);
		virtual void printParams () ;

		virtual bool save(FILE * fptr);
		virtual bool load(FILE * fptr);
		
};



