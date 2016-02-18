#include "CascadeClassifier.h"


//std::vector<Base_M_Classifier*> stage_classifiers;
void CascadeClassifier::getPredictionWeight(const arma::mat& data, arma::mat& prediction)
{
}
		
void addClassifier(const char * classifier_name="navie",
			const char * params="", int num = 1)
{
}

CascadeClassifier::CascadeClassifier(){
}
	
CascadeClassifier::~CascadeClassifier(){
}

bool CascadeClassifier::setParams (const char * params){
	return true;
}

void CascadeClassifier::printParams (){
}

bool CascadeClassifier::save(FILE * fptr)
{
	return true;
}

bool CascadeClassifier::load(FILE * fptr){
	return true;
}


