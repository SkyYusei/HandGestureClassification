#pragma once
#include "Base_2_Classifier.h"
#include <armadillo>
#include <vector>

class SVMClassifier:public Base_2_Classifier {
	protected:
        int kernel_type;
        double degree;
        double gamma;
        double coef0;
        double C;
		double tolerance;
		double eps;
        int max_iteration;
        
        int sample_num, sv_num, dimension;
        arma::mat X;
        arma::vec Y;
        arma::vec w;
        arma::vec alpha;
        arma::mat K;
        arma::vec error_cache;
        double b, delta_b;
	public:
		SVMClassifier() ;
		~SVMClassifier() ;

        double kernel(const arma::vec& v1, const arma::vec& v2); 
        void initialize_data(const arma::mat& positive, const arma::mat& negative);
        int check_alpha(int i1);
        bool optimize_alpha(int i1, int i2); 
        double predict_function(int k);
        double calculate_err();

		virtual bool setParams (const char * params);
		virtual void printParams () ;
		virtual void train2(const arma::mat& positive, const arma::mat& negative);
		//virtual void test(const arma::mat& data, std::vector<int>& prediction) ;
		virtual void getPredictionWeight(const arma::mat& data, arma::mat& prediction) ;
		virtual bool save(FILE * fptr);
		virtual bool load(FILE * fptr);
};



