#include "ClassifierFactory.h"
#include "Classifier.h"

#include <string.h>
#include <string>
using namespace std;

Base_M_Classifier * createClassifier(const char * classifier_name,const char * params){
	Base_M_Classifier * ret = 0;
		
	if(strcmp(classifier_name,"adaboost")==0)
		ret = new Adaboost_2_Classifier();

	if(strcmp(classifier_name,"naive")==0)
		ret = new Naive_2_WClassifier();

	if(strcmp(classifier_name,"ann")==0)
		ret = new ANNClassifier();
	
	if(strcmp(classifier_name,"svm")==0)
		ret = new SVMClassifier();
		
	if(strlen(params)!=0 &&strcmp(params,"~")!=0)
		if(ret && ret->setParams(params)==false) return 0;
	assert(ret);
	return ret;
}

void listClassifier(){
	vector<string> c;
	puts("Normal Classifier:");
	c.push_back("adaboost");
	c.push_back("naive");
	//c.push_back("cascade");

	for(int i = 0;i<c.size();i++){
		cout << i << "\t" << c[i] << endl;
	}
}



