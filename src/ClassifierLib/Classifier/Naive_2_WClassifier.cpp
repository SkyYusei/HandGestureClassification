#include "Naive_2_WClassifier.h"
#include <armadillo>
#include <algorithm>
#include <assert.h>
#include "codelib.h"

using namespace std;
using namespace arma;

void Naive_2_WClassifier::train2(const arma::mat& positive,const arma::mat& negative){
	int feature_num = positive.n_rows;
	int sample_num = positive.n_cols + negative.n_cols;

	// set weight
	Row<double> f_weight;
	f_weight.ones(sample_num);
	f_weight/=(double)sample_num;
	if(weight==NULL){	
		weight = & f_weight;
	}

	double * bucket_pos = new double[num_segment];
	double * bucket_neg = new double[num_segment];
	assert(bucket_pos);
	assert(bucket_neg);
	double min_error = -1;
	double sum_neg = 0,sum_pos = 0;
	for(int i = 0;i<positive.n_cols;i++){
		sum_pos+=(*weight)(i);
	}

	for(int i = 0;i<negative.n_cols;i++){
		sum_neg+=(*weight)(i+positive.n_cols);
	}

	for(int r = 0;r<feature_num;r++){
		for(int i = 0;i<num_segment;i++){
			bucket_pos[i] =bucket_neg[i] = 0;
		}

		int cnt=0;
		double max_val = max(max(positive.row(r)),max(negative.row(r)));
		double min_val = min(min(positive.row(r)),min(negative.row(r)));
		double range = max_val-min_val+1e-10;
		// sum in two bucket

		// put value into bucket		
		for(int i = 0;i<positive.n_cols;i++){
			double val = positive(r,i);
			int bid = ceil((val - min_val) / range * (num_segment-1)); 	
			assert(bid>=0 && bid<num_segment);
			bucket_pos[bid] += (*weight)(i);
		}
		for(int i = 0;i<negative.n_cols;i++){
			double val = negative(r,i);
			int bid = floor((val - min_val )/ range * (num_segment-1)); 	
			assert(bid>=0 && bid<num_segment);
			bucket_neg[bid] += (*weight)(i+positive.n_cols) ;
		}

		double tmp_threshold = 0;
		
		int tmp_dir ;
		double tmp_error ; 
		double tmp_error_pos=0;
		double tmp_error_neg=sum_neg;

		for(int i = 1;i<num_segment;i++){
			tmp_error_pos += bucket_pos[i-1];
			tmp_error_neg -= bucket_neg[i-1];	
			tmp_error = tmp_error_pos + tmp_error_neg;
			tmp_dir = 1;
		
			if(tmp_error > 0.5){
				tmp_dir = -1;
				tmp_error = 1 - tmp_error;
			}

			if(min_error<0 || tmp_error<min_error){
				min_error = tmp_error;
				dimension = r;
				direction  = tmp_dir;
				threshold = min_val + i * (max_val-min_val) / num_segment;
			}
		}
	}

	if(bucket_pos)delete bucket_pos;
	if(bucket_neg)delete bucket_neg;
	if(debug_level>=SHOW_ALL_MSG){
		printf("\tdimension: %d\n",dimension);
		printf("\tdirection: %d\n",direction);
		printf("\tthreshold: %lf\n",threshold);
	}
	train_flag=true;
}

void Naive_2_WClassifier::getPredictionWeight(
	const arma::mat& data, arma::mat& prediction)
{
	assert(train_flag);
	assert(dimension<data.n_rows);
	//printf("col %d row %d\n",data.n_cols,data.n_rows);
	//printf("dimension %d\n",dimension);
	prediction=zeros(data.n_cols, 2);
	for(int i = 0;i<data.n_cols;i++){
		bool flag = (data(dimension,i) >= threshold);
		if(direction == 1)
			prediction(i,flag) = 1;		
		else
			prediction(i,!flag) = 1;	
	}
}

Naive_2_WClassifier::Naive_2_WClassifier() {
}

Naive_2_WClassifier::~Naive_2_WClassifier() {
}

bool Naive_2_WClassifier::save(FILE * fptr){
	fprintf(fptr,"direction: %d\n",direction);
	fprintf(fptr,"dimension: %d\n",dimension);
	fprintf(fptr,"threshold: %.15lf\n",threshold);
	return true;
}

bool Naive_2_WClassifier::load(FILE * fptr){
	char buf[256];
	fscanf(fptr,"%s %d",buf,&direction);
	fscanf(fptr,"%s %d",buf,&dimension);
	fscanf(fptr,"%s %lf",buf,&threshold);
	train_flag = true;
	return true;
}

bool Naive_2_WClassifier::setParams (const char * params)
{
	num_segment = atoi(params);
	if(num_segment<=0){
		printf(PRINT_RED"ERROR Classifier Parameter Set: %s\n"PRINT_END,params);
		printf("Usage: num_segment(>0)\n");
		return false;
	}
	return true;
}

void Naive_2_WClassifier::printParams (){
	printf("num_segment %d\n",num_segment);
}

