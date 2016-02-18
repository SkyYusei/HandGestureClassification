#pragma once
#include "Base_2_Classifier.h"
#include <vector>

class Adaboost_2_Classifier:public Base_2_Classifier {
	protected:
		std::string weak_classifier_name;
		std::string weak_classifier_params;

		int max_iteration;
		double error_threshold;

		std::vector<Base_M_Classifier*> weak_classifiers;
		arma::Row<double> weight;
		arma::Row<double> estimate_sum;
		//std::vector<double> alpha;
		arma::vec alpha;
		virtual void getPredictionWeight(const arma::mat& data, arma::mat& prediction);
		
	public:

		void addWeakClassifier(const char * classifier_name="navie",
			const char * params="", int num = 1);
		void clearWeakClassifier();
	public:
		Adaboost_2_Classifier() ;
		~Adaboost_2_Classifier() ;

		virtual bool setParams (const char * params);
		virtual void printParams () ;
		virtual void train2(const arma::mat& positive,const arma::mat& negative);
		//virtual void test(const arma::mat& data, std::vector<int>& prediction) ;

		virtual bool save(FILE * fptr);
		virtual bool load(FILE * fptr);
		
};



