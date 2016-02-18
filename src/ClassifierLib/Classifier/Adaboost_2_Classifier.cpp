#include "Adaboost_2_Classifier.h"
#include <armadillo>
#include "codelib.h"
#include "ClassifierFactory.h"
#include "Base_2_Classifier.h"
#include <assert.h>

using namespace arma;
using namespace std;

void Adaboost_2_Classifier::train2(const arma::mat& positive,const arma::mat& negative){
	clearWeakClassifier();
	alpha.clear();

	int sample_num = positive.n_cols+negative.n_cols;
	int pos_size = positive.n_cols;
	int neg_size = negative.n_cols;

	int pos_weight = 1;
	int neg_weight = 1;
	weight.ones(pos_size);
	weight = weight * pos_weight;
	weight.insert_cols(weight.n_cols,ones(1,neg_size) * neg_weight);
	weight/=(double)(pos_weight*pos_size+neg_weight*neg_size);

	estimate_sum.zeros(sample_num);

	for(int cid = 0; cid < max_iteration;cid++){
		if(debug_level>=SHOW_PHASE) 
			printf(PRINT_BLUE"\niteration %d\n"PRINT_END,cid);
		addWeakClassifier(weak_classifier_name.c_str(),
			weak_classifier_params.c_str());

		weak_classifiers[cid]->setWeight(&weight);
		if(debug_level>=SHOW_PHASE) 
			printf("training weak classifier\n");
		((Base_2_Classifier*)(weak_classifiers[cid]))->train2(positive,negative);
		if(debug_level>=SHOW_PHASE) 
			printf("training over\n");

		// test
		vec pos_res,neg_res;
		//weak_classifier[cid]->test(positive,pos_res);
		//weak_classifier[cid]->test(negative,neg_res);
		weak_classifiers[cid]->getPredictionLabel(positive,pos_res);
		weak_classifiers[cid]->getPredictionLabel(negative,neg_res);
		
		// get weight error
		double weight_error = 0;
		for(int i = 0;i<pos_size;i++)
			weight_error+= (!pos_res[i])* weight(i);
		for(int i = 0,j=pos_size;i<neg_size;i++,j++)
			weight_error+= neg_res[i] * weight(j);

		//update
		double a = (1/2.0 * log((1-weight_error) / max(weight_error,1e-10)));
		//alpha.push_back(a);
		alpha.insert_rows(alpha.n_rows,1);
		alpha[alpha.n_rows-1]=a;

		for(int i = 0;i<pos_size;i++)
			weight[i] *= exp((1-pos_res[i]*2)*alpha[cid]);
		for(int i = 0,j=pos_size;i<neg_size;i++,j++)
			weight[j] *= exp((neg_res[i]*2-1)*alpha[cid]); 
 		
		//normalize
		weight = weight/sum(weight);
		
		// get total error
		double error = 0;
		for(int i = 0;i<pos_size;i++)
			estimate_sum[i]+= alpha[cid] * (pos_res[i]*2-1);
		for(int i = 0;i<neg_size;i++)
			estimate_sum[pos_size+i]+=alpha[cid] * (neg_res[i]*2-1);

		for(int i = 0;i<sample_num;i++){
			if(estimate_sum[i]<0 && i<pos_size) error++;
			if(estimate_sum[i]>=0 && i>=pos_size) error++;
		}
		error /= estimate_sum.size();

		if(debug_level>=SHOW_PHASE){ 
			printf("\tweight error: %lf\n",weight_error);
			printf("\talpha: %lf\n",alpha[cid]);
			printf("\terror %lf\n",error);
		}


		if(error < error_threshold) break;
	}
	double alpha_sum = sum(alpha);
	for(int i = 0;i<alpha.n_rows;i++)alpha[i]/= alpha_sum;
	train_flag = true;
}

void Adaboost_2_Classifier::getPredictionWeight
	(const arma::mat& data, arma::mat& prediction)
{
	//vector<double> estimate_sum;
	vec weak_prediction;
	prediction = zeros(data.n_cols, 2);

	//estimate_sum.resize(data.n_cols);
	for(int i = 0;i<weak_classifiers.size();i++){
		assert(weak_classifiers[i]);
		weak_classifiers[i]->getPredictionLabel(data,weak_prediction);
		for(int j = 0;j<data.n_cols;j++){
			prediction(j,weak_prediction[j]) += alpha[i];
		}
	}
}

void Adaboost_2_Classifier::addWeakClassifier(
		const char * classifier_name,
		const char * params, int num)
{
	for(int i = 0;i<num;i++){
		Base_M_Classifier * c = createClassifier(classifier_name,params);
		assert(c!=NULL);
		c->setDebug(this->debug_level);
		weak_classifiers.push_back(c);		
	}
}

void Adaboost_2_Classifier::clearWeakClassifier()
{
	for(int i = 0;i<weak_classifiers.size();i++){
		if(weak_classifiers[i])delete weak_classifiers[i];
		weak_classifiers[i]=NULL;
	}
	weak_classifiers.clear();
}

Adaboost_2_Classifier::Adaboost_2_Classifier() {
	max_iteration = 100;
	error_threshold = 0.001;
	weak_classifier_name = "naive";
	weak_classifier_params = "10000";	
}

Adaboost_2_Classifier::~Adaboost_2_Classifier() {
	clearWeakClassifier();
}

void Adaboost_2_Classifier::printParams(){
	printf("adaboost 2 class classifier\n");
	printf("max_iteration: %d\n",max_iteration);
	printf("error_threshold: %lf\n",error_threshold);
	printf("weak_classifier_name: %s\n",weak_classifier_name.c_str());
	printf("weak_classifier_params: %s\n",weak_classifier_params.c_str());
}

bool Adaboost_2_Classifier::setParams(const char* params) {
	vector<string> st;
	split(params,',',st);
	if(st.size()!=4){
		printf(PRINT_RED"ERROR Classifier Parameter Set: %s\n"PRINT_END,params);
		printf("Usage: max_iteration,error_threshold,weak_classifier_name,weak_classifier_params\n");
		return false;
	}
	else{
		max_iteration = atoi(st[0].c_str());
		error_threshold = atof(st[1].c_str());
		weak_classifier_name = st[2];	
		weak_classifier_params = st[3];
	}
	return true;
}

bool Adaboost_2_Classifier::save(FILE * fptr){
	fprintf(fptr,"ClassifierNum: %lu\n",weak_classifiers.size());	
	fprintf(fptr,"SubClassifierName: %s\n",weak_classifier_name.c_str());	

	if(weak_classifier_params.length()==0)
		fprintf(fptr,"SubClassifierParams: ~\n");	
	else
		fprintf(fptr,"SubClassifierParams: %s\n",weak_classifier_params.c_str());	

	fprintf(fptr,"Alpha:\n");
	for(int i = 0;i<alpha.size();i++){
		fprintf(fptr,"%.15lf ",alpha[i]);	
	}fprintf(fptr,"\n");
	for(int i = 0;i<weak_classifiers.size();i++){
		fprintf(fptr,"\nclassifier_%05d\n",i);
		weak_classifiers[i]->save(fptr);
	}
	return true;
}

bool Adaboost_2_Classifier::load(FILE * fptr){
	clearWeakClassifier(); alpha.clear();
	char tmp[256];
	char buf[256];
	int num;
	double a;
	fscanf(fptr,"%s %d",tmp,&num);
	//printf("num %d\n",num);
	fscanf(fptr,"%s %s",tmp,buf);
	weak_classifier_name = buf;

	fscanf(fptr,"%s %s",tmp,buf);
	weak_classifier_params = buf;

	fscanf(fptr,"%s",buf);
	for(int i = 0;i<num;i++){
		fscanf(fptr,"%lf",&a);
		//alpha.push_back(a);
		alpha.insert_rows(alpha.n_rows,1);
		alpha[alpha.n_rows-1]=a;
	}

	for(int i = 0;i<num;i++){
		fscanf(fptr,"%s",buf);
		addWeakClassifier(weak_classifier_name.c_str(), 
			weak_classifier_params.c_str());
		weak_classifiers[i]->load(fptr);
	}
	train_flag = true;
	return true;
}


