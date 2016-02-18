#pragma once
#include <armadillo>
#include "Classifier.h"
#include "BaseProcess.h"

class BaseTrainer:public BaseProcess
{
private:
	void reportPosNeg();
protected:
	virtual void train();
	virtual void test();
	virtual void result();
public:
	BaseTrainer();
	~BaseTrainer();

	arma::mat trainPositiveFeatureSet;
	arma::mat trainNegativeFeatureSet;
	arma::mat testPositiveFeatureSet;
	arma::mat testNegativeFeatureSet;
	
	Base_M_Classifier * classifier;
	void setClassifier(Base_M_Classifier* classifier);
	Base_M_Classifier * getClassifier();
	void loadData(const char * data_dir);
	virtual void run();

	int true_pos;
	int false_pos;
	int true_neg;
	int false_neg;
	arma::vec positive_result;
	arma::vec negative_result;
};

