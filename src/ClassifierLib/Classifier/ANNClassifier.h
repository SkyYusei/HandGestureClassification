#pragma once
#include "Base_M_Classifier.h"
#include <vector>

typedef arma::mat (*MFunc)(const arma::mat & x);
arma::mat sigmoid(const arma::mat & x);
arma::mat sigmoid_d(const arma::mat & x);

class ANNLayer{

	protected:
		int node_num;
		arma::mat weight;
		arma::vec bias;
		arma::vec err;
		arma::mat a;
		arma::mat z;
		MFunc activation ;
		MFunc activation_d ;

	public:
		friend class ANNLayer;

		ANNLayer(){
			node_num=0;
			activation = sigmoid ;
			activation_d = sigmoid_d ;
		}

		const arma::mat& getA(){return this->a;}
		int getNodeNum(){return node_num;}
		void setNodeNum(int n){
			this->node_num = n;
		}

		virtual void init(int input_demension){
			if(input_demension>=0){
				weight = arma::randu(node_num, input_demension);
				bias = arma::randu(node_num);
			}
			else{
				weight.clear();
				bias.clear();
			}
		}

		void forward(const arma::mat & a){
			z = weight * a + bias;
			this->a = activation(z);
		}

		void backprop_calcErr(ANNLayer* nextLayer, const arma::mat& output){
			if(nextLayer){
				err = nextLayer->weight.t() * nextLayer->err * activation_d(z);
			}
			else{
				err =(a-output)*activation_d(z);
			}
		}

		void backprop_updateParam(double eta, const arma::mat& pre_a){
			bias = bias - eta *  err;
			weight = weight - eta  * err * pre_a.t();
		}

};

class ANNClassifier:public Base_M_Classifier {
	protected:
		std::vector<ANNLayer*> net;
		int batch_size;
		int train_time;
		double eta;

	protected:
		virtual void getPredictionWeight(const arma::mat& data, arma::mat& prediction);
		virtual void train(const arma::mat& data, const arma::vec& label);

		//		virtual void activate(const arma::mat& x, arma::mat& y);
		virtual void forward(const arma::mat& input);
		virtual void backprop(const arma::mat& output);	
		virtual void initParam();

	public:
		ANNClassifier() ;
		~ANNClassifier() ;

		virtual bool setParams (const char * params);
		virtual void printParams () ;

		virtual bool save(FILE * fptr);
		virtual bool load(FILE * fptr);
};



