#include <stdio.h>
#include <armadillo>
#include "codelib.h"
#include "feature.h"

using namespace arma;

static char * result_dir;
static char * result_prefix;

mat D;
bool init = false;
int orig_row ;
int orig_col ;
int eigen_num;

void processFile(const char * file, void * data){
	printf("deal file %s\n",file);
	mat m;
	m.load(file, pgm_binary);
	vec v = normalise(vectorise(m));
	if(init==false){
		orig_row = m.n_rows;
		orig_col = m.n_cols;
		D.set_size(v.n_rows, 0); init = true;
	}
	D.insert_cols(D.n_cols,v);
}

void PCA(){
//	puts("Data normalise ...");
//	D = normalise(D);
	puts("Doing SVD ...");

	mat E;
	pca(D,E,eigen_num);
	
	char fn[128];
	sprintf(fn,"%s/%s.dat",result_dir,result_prefix);
	E.save(fn, raw_ascii);
}


int main(int argc , char ** argv)
{
	printf("This program is a tool for calulate Eigen Face!\n");
	if(argc <= 4){
		printf("Usage: %s data_dir eigen_num result_dir result_prefix\n",argv[0]);
		return 0;
	}

	eigen_num = atoi(argv[2]);
	result_dir = argv[3];
	result_prefix = argv[4];

	puts("Loading data ...");
	travelAllFiles(processFile, argv[1]);	
	PCA();
	return 0;
}


