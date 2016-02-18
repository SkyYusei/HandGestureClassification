#pragma once

#include <armadillo>
#include <vector>
#include "codelib.h"

class Base_M_Classifier: public BaseDebug{
protected:
	bool train_flag;
	arma::Row<double> * weight;
    virtual void getPredictionWeight(const arma::mat& data, arma::mat& prediction)=0;
public:
	Base_M_Classifier() {train_flag=false;weight=NULL;};
	virtual ~Base_M_Classifier() {};

	// give a params string to intialize the classifier
	virtual bool setParams (const char * params) {return true;}
	virtual void printParams () {printf("No any parameters!\n");};
    //virtual void train(const arma::mat& positive,const arma::mat& negative) = 0;
	// 1 column each sample
	// label is a vertical vector
	virtual void train(const arma::mat& data, const arma::vec& label) = 0;
    virtual void train2(const arma::mat& positive,const arma::mat& negative){
		arma::vec combine_label = arma::ones(positive.n_cols);
		combine_label.insert_cols(combine_label.n_cols, arma::zeros(negative.n_cols) );
		arma::mat combine_data;
		combine_data.insert_cols(combine_data.n_cols, positive);
		combine_data.insert_cols(combine_data.n_cols, negative);
		train(combine_data, combine_label);
	}

    //virtual void test(const arma::mat& data, std::vector<int>& prediction) = 0;
	// 1 column each sample
	// prediction is a sample_num(row) * class_num(col) matrix
	// prediction[i][j] = the probability of the i sample belong to the class j
	// sum of each row of prediction = 1
    virtual void getPredictionProb(const arma::mat& data, arma::mat& prediction){
		assert(train_flag);
		getPredictionWeight(data, prediction);
		arma::vec s = sum(prediction,1);
		s+=1e-10;
		prediction.each_col() /= s;
		//normalise(prediction, 1, 1);
	};

    virtual void getPredictionLabel(const arma::mat& data, arma::vec& prediction){
		assert(train_flag);

		arma::mat prediction_prob;
		getPredictionProb(data,prediction_prob);
		prediction.set_size(data.n_cols);

		for (int i = 0;i<prediction_prob.n_rows;i++){
			arma::uword max_index;
			prediction_prob.row(i).max(max_index);
			prediction[i] = max_index;
		}
	};

    virtual bool save(const char* file) {
		FILE * fptr = fopen(file,"w");
		if(fptr==NULL) return false;
		bool flag = save(fptr);
		fclose(fptr);return flag;
	}
		
    virtual bool load(const char* file) {
		FILE * fptr = fopen(file,"r");
		if(fptr==NULL) return false;
		bool flag = load(fptr);
		train_flag = true;
		fclose(fptr);return flag;
	};
    virtual bool save(FILE* fptr) {return true;};
    virtual bool load(FILE* fptr) {return true;};
	
	virtual void setWeight(arma::Row<double> * weight){
		this->weight = weight;
	}
};



