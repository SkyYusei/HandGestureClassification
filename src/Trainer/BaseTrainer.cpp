#include "BaseTrainer.h"
#include "codelib.h"
#include <stdio.h>
#include <iostream>
using namespace std;
using namespace arma;

void BaseTrainer::run(){
	assert(classifier);
	showProcess("Classifier");
	classifier->printParams();
	train();
	test();
	result();
}

void BaseTrainer::train(){
	double t = startTimer();
	showProcess("Training");
	assert(classifier);
	((Base_2_Classifier*)classifier)->train2(trainPositiveFeatureSet, trainNegativeFeatureSet);	
	if(DEBUG_LEVEL>=SHOW_PHASE)endTimer(t);
}

void BaseTrainer::test(){
	double t = startTimer();
	showProcess("Testing");
	//classifier->test(testPositiveFeatureSet, positive_result); 
	//classifier->test(testNegativeFeatureSet, negative_result);
	classifier->getPredictionLabel(testPositiveFeatureSet, positive_result); 
	classifier->getPredictionLabel(testNegativeFeatureSet, negative_result);
	if(DEBUG_LEVEL>=SHOW_PHASE)endTimer(t);
}

void BaseTrainer::result(){
	showProcess("Result");
	if(DEBUG_LEVEL>=SHOW_RESULTS)reportPosNeg();
}

void BaseTrainer::loadData(const char * data_dir){
	double t = startTimer();
	showProcess("Load Data");
	char fn[1024];
	sprintf(fn,"%s/train_positive.dat",data_dir);
	trainPositiveFeatureSet.load(fn);	

	sprintf(fn,"%s/train_negative.dat",data_dir);
	trainNegativeFeatureSet.load(fn);	

	printf("Train data postive : %d\n",trainPositiveFeatureSet.n_cols);
	printf("Train data negtive : %d\n",trainNegativeFeatureSet.n_cols);

	sprintf(fn,"%s/test_positive.dat",data_dir);
	testPositiveFeatureSet.load(fn);	

	sprintf(fn,"%s/test_negative.dat",data_dir);
	testNegativeFeatureSet.load(fn);	

	printf("Test data postive : %d\n",testPositiveFeatureSet.n_cols);
	printf("Test data negtive : %d\n",testNegativeFeatureSet.n_cols);

	if(DEBUG_LEVEL>=SHOW_PHASE)endTimer(t);
}

void BaseTrainer::reportPosNeg(){
	true_pos = 0;
	false_neg = 0;
	int pos_size = positive_result.size();
	for(int i = 0; i<pos_size; i++){
		if(positive_result[i] == 1) true_pos++;
		else false_neg++;
	}

	true_neg = 0;
	false_pos = 0;
	int neg_size = negative_result.size();
	for(int i = 0; i<neg_size; i++){
		if(negative_result[i] == 0) true_neg++;
		else false_pos ++;
	}
	double t_p = true_pos / (double) pos_size * 100; 
	double t_n = true_neg / (double) neg_size * 100; 
	double f_p = 100 - t_n;
	double f_n = 100 - t_p;

	printf(PRINT_GREEN"True Positive	: %d / %d = %lf%%\n"PRINT_END,
			true_pos,pos_size,t_p);
	printf(PRINT_RED  "False Negative	: %d / %d = %lf%%\n"PRINT_END,
			false_neg,pos_size,f_n);
	printf(PRINT_GREEN"True Negative	: %d / %d = %lf%%\n"PRINT_END, 
			true_neg, neg_size, t_n);
	printf(PRINT_RED  "False Positive	: %d / %d = %lf%%\n"PRINT_END,
			false_pos, neg_size ,f_p);
}

void BaseTrainer::setClassifier(Base_M_Classifier* classifier){
	if(this->classifier) delete this->classifier;
	this->classifier = NULL;
	this->classifier = classifier;
}

Base_M_Classifier * BaseTrainer::getClassifier(){
	return classifier;
}

BaseTrainer::BaseTrainer(){
	classifier = NULL;
}

BaseTrainer::~BaseTrainer(){
	if(classifier)delete classifier;
	classifier = NULL;
}


