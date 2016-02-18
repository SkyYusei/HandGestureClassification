#include "ANNClassifier.h"

//std::vector<Base_M_Classifier*> stage_classifiers;
void ANNClassifier::getPredictionWeight(const arma::mat& data, arma::mat& prediction)
{
	assert(train_flag);
}

void ANNClassifier::train(const arma::mat& data, const arma::vec& label){
	//arma::mat output ; // = label
	int output_dimension = net[net.size()-1]->getNodeNum();
	for(int i = 0;i<train_time;i++){
		for(int j = 0; j<data.n_cols;j++){
			forward(data.col(i));
			arma::vec output = arma::zeros(output_dimension);
			// !!!
			output[(int)(label[j]+0.5)] = 1;
			backprop(output);
		}
	}
	train_flag = true;
}

void ANNClassifier::forward(const arma::mat& input){	
	net[0]->forward(input);
	for(int i = 1;i<net.size();i++){
		net[i]->forward(net[i-1]->getA());
	}
}

void ANNClassifier::backprop(const arma::mat& output){	
	for(int i = net.size()-1;i>0;i--){
		net[i]->backprop_updateParam(eta/ output.n_cols, net[i-1]->getA());
	}
}

void ANNClassifier::initParam(){
	// initial structure
	for(int i = 0;i<net.size();i++){
		net[i]->setNodeNum(0);
	}
	// initial parameter to random value
	for(int i = 0;i<net.size();i++){
		net[i]->init(0);
	}
}

ANNClassifier::ANNClassifier(){
}
	
ANNClassifier::~ANNClassifier(){
}

bool ANNClassifier::setParams (const char * params){
	return true;
}

void ANNClassifier::printParams (){
}

bool ANNClassifier::save(FILE * fptr)
{
	assert(train_flag);
	return true;
}

bool ANNClassifier::load(FILE * fptr){
	train_flag = true;
	return true;
}


