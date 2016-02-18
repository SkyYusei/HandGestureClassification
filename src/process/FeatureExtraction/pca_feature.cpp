#include <stdio.h>
#include <armadillo>

#include "codelib.h"
#include "feature.h"
using namespace arma;

static char * result_dir;
static char * result_prefix;

mat F,E;

void processFile(const char * file,void * data){
	mat m ; m.load(file);
	//vec img_data = normalise(vectorise(m));
	vec img_data = (vectorise(m));
	vec f;
	pca_feature(img_data, E, f );
	F.insert_cols(F.n_cols,f);
}

int main(int argc , char ** argv)
{
	printf("This program is a tool for calulate PCA feature!\n");
	if(argc <= 4){
		printf("Usage: %s eigen_file result_dir img_dir result_prefix\n",argv[0]);
		return 0;
	}
	result_dir = argv[2];
	result_prefix = argv[4];
	
	E.load(argv[1]);
	travelAllFiles(processFile, argv[3]);	
	char fn[256];sprintf(fn,"%s/%s.dat",result_dir,result_prefix);
	F.save(fn,raw_ascii);
	return 0;
}
