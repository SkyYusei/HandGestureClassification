#include "SVMClassifier.h"
#include <armadillo>
#include "codelib.h"
#include "ClassifierFactory.h"
//#include "WeightClassifier.h"
#include <assert.h>

using namespace arma;
using namespace cv;
using namespace std;

double SVMClassifier::kernel(const arma::vec& v1, const arma::vec& v2) {
	switch (kernel_type) {
		case 0: 
			return dot(v1, v2);
		case 1:
			return pow((gamma*dot(v1,v2)+coef0),degree);
		case 2:
			double k = -2*dot(v1, v2);
			k += sum(square(v1)) + sum(square(v2));
			k = exp(-gamma*k);
			return k;
	}
	return 0;
}

void SVMClassifier::initialize_data(const arma::mat& positive, const arma::mat& negative) {
	X = join_horiz(positive, negative);
	Y = join_vert(ones<vec>(positive.n_cols),-ones<vec>(negative.n_cols));
	sample_num = Y.size();
	sv_num = 0;
	dimension = X.n_rows;
	b = delta_b = 0.0;
	w = zeros<vec>(dimension);
	alpha = zeros<vec>(sample_num);
	error_cache = zeros<vec>(sample_num);

	switch (kernel_type) {
		case 0: 
			K = trans(X)*X;
			break;
		case 1:
			K = pow((gamma*trans(X)*X+coef0),degree);
			break;
		case 2:
			rowvec square_row = sum(square(X),0);
			vec square_col = square_row.t(); 
			K = -2*trans(X)*X;
			for (int i = 0; i < sample_num; ++i) {
				K.row(i) += square_row;
				K.col(i) += square_col;
			}
			K = exp(-gamma*K);
			break;
	}
}

void SVMClassifier::train2(const arma::mat& positive, const arma::mat& negative) {
	printf("Start Training\n");
	initialize_data(positive, negative);
	bool check_all = true;
	int num_changed = 0, k = 0, iteration = 0;
	// alternate between scan all alpha and the non-boundary alpha
	while (num_changed > 0 || check_all) {
		if (++iteration == max_iteration) break;
		num_changed = 0;
		if (check_all) {
			for (k = 0; k < sample_num; ++k)
				num_changed += check_alpha(k);
		}
		else {
			for (k = 0; k < sample_num; ++k)
				if (alpha[k] != 0 && alpha[k] != C)
					num_changed += check_alpha(k);
		}

		if (check_all == true)
			check_all = false;
		else if (num_changed == 0)
			check_all = true;

		printf("Error rate: %lf\n", calculate_err());
		for (k = 0; k < sample_num; ++k)
			if (alpha[k] < 1e-6) alpha[k] = 0;
	}

	// save model
	for (int i = 0; i < sample_num; ++i)
		if (alpha[i] > 0) {
			alpha[sv_num] = alpha[i];
			X.col(sv_num) = X.col(i);
			Y[sv_num] = Y[i];
			++sv_num;
		}
	train_flag = true;
}

int SVMClassifier::check_alpha(int i1) {
	double y1, alpha1, E1, r1;
	y1 = Y[i1];
	alpha1=alpha[i1];
	if (alpha1 > 0 && alpha1 < C)
		E1 = error_cache[i1];
	else
		E1 = predict_function(i1) - y1;

	r1 = y1 * E1;
	// check if the first alpha violate the KKT condition by more than tolerance
	// if it does, search the second alpha and optimize both jointly 
	if ((r1 < -tolerance && alpha1 < C) || (r1 > tolerance && alpha1 > 0)) {
		int k0, k, i2;
		double tmax;
		// try argmax E1-E2
		for (i2 = -1, tmax = 0, k = 0; k < sample_num; ++k) {
			if (alpha[k] > 0 && alpha[k] < C) {
				double E2, tmp;
				E2 = error_cache[k];
				tmp = fabs(E1-E2);
				if (tmp > tmax) {
					tmax = tmp;
					i2 = k;
				}
			}
			if (i2 >= 0)
				if (optimize_alpha(i1,i2)) return 1;
		}

		// try scan non-boundary alpha randomly
		for (k0 = (int)(drand48()*sample_num), k = k0; k < sample_num+k0; ++k) {
			i2 = k % sample_num;
			if (alpha[i2] > 0 && alpha[i2] < C)
				if (optimize_alpha(i1, i2)) return 1;
		}
		// try sacn all alpha
		for (k0 = (int)(drand48()*sample_num), k = k0; k < sample_num+k0; ++k) {
			i2 = k % sample_num;
			if (optimize_alpha(i1, i2)) return 1;
		}
	}
	return 0;
}

bool SVMClassifier::optimize_alpha(int i1, int i2) {
	int y1, y2, s;
	double alpha1, alpha2;
	double a1, a2;
	double E1, E2, L, H, k11, k22, k12, eta, LowObj, HighObj;
	if (i1 == i2) return false;

	alpha1 = alpha[i1];
	alpha2 = alpha[i2];
	y1 = Y[i1];
	y2 = Y[i2];

	if (alpha1 > 0 && alpha1 < C)
		E1 = error_cache[i1];
	else
		E1 = predict_function(i1) - y1;
	if (alpha2 > 0 && alpha2 < C)
		E2 = error_cache[i2];
	else
		E2 = predict_function(i2) - y2;

	s = y1 * y2;
	if (y1 == y2) {
		double r = alpha1+alpha2;
		if (r > C) {
			L = r - C;
			H = C;
		}
		else {
			L = 0;
			H = r;
		}
	}
	else {
		double r = alpha1 -alpha2;
		if (r > 0) {
			L = 0;
			H = C - r;
		}
		else {
			L = -r;
			H = C;
		}
	}

	if (L == H) return false;

	k11 = K(i1, i1);
	k12 = K(i1, i2);
	k22 = K(i2, i2);
	eta = 2*k12-k11-k22;

	if (eta < 0) {
		a2 = alpha2 + y2*(E2-E1)/eta;
		if (a2 < L) a2 = L;
		else if (a2 > H) a2 = H;
	}
	else {
		double c1 = eta/2;
		double c2 = y2*(E1-E2) - eta*alpha2;
		LowObj = c1*L*L + c2*L;
		HighObj = c1*H*H + c2*H;
		if (LowObj > HighObj + eps) a2 = L;
		else if (LowObj < HighObj - eps) a2 = H;
		else a2 = alpha2;
	}

	if (fabs(a2 - alpha2) < eps * (a2 + alpha2 + eps)) return false;
	a1 = alpha1 - s * (a2 - alpha2);
	if (a1 < 0) {
		a2 += s * a1;
		a1 = 0;
	}
	else if (a1 > C) {
		double t = a1 - C;
		a2 += s * t;
		a1 = C;
	}

	double b1, b2, bnew;
	if (a1 > 0 && a1 < C)
		bnew = b + E1 + y1 * (a1 - alpha1) * k11 + y2 * (a2 - alpha2) * k12;
	else if (a2 > 0 && a2 < C)
		bnew = b + E2 + y1 * (a1 - alpha1) * k12 + y2 * (a2 - alpha2) * k22;
	else {
		b1 = b + E1 + y1 * (a1 - alpha1) * k11 + y2 * (a2 - alpha2) * k12;
		b2 = b + E2 + y1 * (a1 - alpha1) * k12 + y2 * (a2 - alpha2) * k22;
		bnew = (b1 + b2) / 2.0;
	}

	delta_b = bnew - b;
	b = bnew;

	double t1 = y1 * (a1 - alpha1);
	double t2 = y2 * (a2 - alpha2);

	if (kernel_type == 0)
		w = w + t1 * X.col(i1) + t2 * X.col(i2);

	for (int i = 0; i < sample_num; ++i)
		if (alpha[i] > 0 && alpha[i] < C)
			error_cache[i] += t1 * K(i1, i) + t2 * K(i2, i) - delta_b;

	// set the Error to zero after successful optimization
	error_cache[i1] = 0;
	error_cache[i2] = 0;

	alpha[i1] = a1;
	alpha[i2] = a2;

	return true;
}

double SVMClassifier::predict_function(int k) {
	double pred = 0;
	for (int i = 0; i < sample_num; ++i)
		if (alpha[i] > 0)
			pred += alpha[i] * Y(i) * K(i,k);
	pred -= b;
	return pred;
}

double SVMClassifier::calculate_err() {
	int num_err = 0;
	for (int i = 0; i < sample_num; ++i)
		if ((predict_function(i) >= 0 && Y[i] < 0) || (predict_function(i) < 0 && Y[i] > 0))
			++num_err;
	return 1.0 * num_err / sample_num;
}

//void SVMClassifier::test(const arma::mat& test_data, std::vector<int>& prediction)
void SVMClassifier::getPredictionWeight(const arma::mat& data, arma::mat& prediction)
{
	printf("start prediction\n");
	double s, y_pred;
	prediction.set_size(data.n_cols,2);
	if (kernel_type == 0) {
		printf("kenerl 0\n");
		for (int i = 0; i < data.n_cols; ++i) {
			s = dot(w, data.col(i)) - b;
			if(s>=0) prediction(i,1) = s;
			else prediction(i,0) = -s;
			//  prediction.push_back(s >= 0);
		}
	}
	else {
		printf("kenerl other\n");
		for (int i = 0; i < data.n_cols; ++i) {
			s = 0;
			for (int j = 0; j < sv_num; ++j)
				s += alpha[j]*Y[j]*kernel(X.col(j), data.col(i));
			s -= b;
			//prediction.push_back(s >= 0);
			if(s>=0) prediction(i,1) = s;
			else prediction(i,0) = -s;
		}
	}
	printf("end prediction\n");
}


SVMClassifier::SVMClassifier() {
	C = 1.0;
	tolerance= 0.0001;
	eps = 0.001;
	gamma = 4.0;
	kernel_type = 2;
	max_iteration = 100;
}

SVMClassifier::~SVMClassifier() {
}

void SVMClassifier::printParams(){
	printf("kernel_type: %d\n", kernel_type);
	printf("degree: %lf\n", degree);
	printf("gamma: %lf\n", gamma);
	printf("coef0: %lf\n", coef0);
	printf("Cvalue: %lf\n", C);
	printf("tolerance: %lf\n", tolerance);
	printf("epsilon: %lf\n", eps);
	printf("max_iteration: %d\n", max_iteration);
}

bool SVMClassifier::setParams(const char* params) {
	printf("%s\n",params);
	vector<string> st;
	split(params,',',st);
	if(st.size()!=8) {
		printf(PRINT_RED"ERROR Classifier Parameter Set: %s\n"PRINT_END,params);
		printf("Usage: kernel_type, degree, gamma, coef0, C, tolerance, epsilon, max_iteration\n");
		return false;
	}
	else {
		kernel_type = atoi(st[0].c_str());
		degree = atof(st[1].c_str());
		gamma = atof(st[2].c_str());
		coef0 = atof(st[3].c_str());
		C = atof(st[4].c_str());
		tolerance = atof(st[5].c_str());
		eps = atof(st[6].c_str());
		max_iteration = atoi(st[7].c_str());
	}
	return true;
}

bool SVMClassifier::save(FILE * fptr){
	fprintf(fptr,"KernelType: %d\n", kernel_type);	
	fprintf(fptr,"b: %lf\n", b);	
	fprintf(fptr,"Gamma: %lf\n", gamma);	
	fprintf(fptr,"Coef0: %lf\n", coef0);	
	fprintf(fptr,"Degree: %lf\n", degree);	
	fprintf(fptr,"VecDimen: %d\n", dimension);
	if (kernel_type == 0) {
		fprintf(fptr,"W:\n");
		for (int i = 0; i < w.size(); ++i)
			fprintf(fptr, "%lf ", w[i]);
		fprintf(fptr, "\n");
	}
	else {
		fprintf(fptr,"SupportVecNum: %d\n", sv_num);
		fprintf(fptr,"Alpha:\n");
		for (int i = 0; i < sv_num; ++i)
			fprintf(fptr, "%lf ", alpha[i]);
		fprintf(fptr, "\n");
		fprintf(fptr,"SupportVec:\n");
		for (int i = 0; i < sv_num; ++i) {
			fprintf(fptr, "%lf", Y[i]);
			for (int j = 0; j < dimension; ++j)
				fprintf(fptr, " %lf ", X(j,i));
			fprintf(fptr, "\n");
		}
	}
	return true;
}

bool SVMClassifier::load(FILE * fptr){
	char tmp[256];
	fscanf(fptr,"%s %d", tmp, &kernel_type);
	fscanf(fptr,"%s %lf", tmp, &b);
	fscanf(fptr,"%s %lf", tmp, &gamma);
	fscanf(fptr,"%s %lf", tmp, &coef0);
	fscanf(fptr,"%s %lf", tmp, &degree);
	fscanf(fptr,"%s %d", tmp, &dimension);
	
	if (kernel_type == 0) {
		w = zeros<vec>(dimension);
		fscanf(fptr,"%s",tmp);
		for (int i = 0; i < dimension; ++i)
			fscanf(fptr, "%lf", &w[i]);
	}
	else {

		fscanf(fptr,"%s %d", tmp, &sv_num);
		fscanf(fptr,"%s",tmp);
		printParams();
		alpha = zeros<vec>(sv_num);
		X = zeros<mat>(dimension, sv_num);
		Y = zeros<vec>(sv_num);

		for (int i = 0; i < sv_num; ++i)
			fscanf(fptr, "%lf", &alpha[i]);
		fscanf(fptr,"%s",tmp);
		for (int i = 0; i < sv_num; ++i) {
			fscanf(fptr, "%lf", &Y[i]);
			for (int j = 0; j < dimension; ++j)
				fscanf(fptr, "%lf", &X(j,i));
		}
	}
	train_flag = true;
	return true;
}


