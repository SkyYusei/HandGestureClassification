#include <stdio.h>
#include "codelib.h"
#include "Trainer.h"
#include "Classifier.h"
#include <vector>
using namespace std;

int main(int argc , char ** argv)
{
	printf("This program is a trainer!\n");
	if(argc <= 2){
		printf("Usage: %s classifier_name:param1,param2 data_dir (model_save_file)\n",argv[0]);
		printf("Classifier List:\n");
		listClassifier();	
		return 0;
	}

	BaseTrainer trainer;
	vector<string> st; split(argv[1],':',st);
	Base_M_Classifier * c = createClassifier(st[0].c_str(),st.size()>1?st[1].c_str():"");
	
	if(c){
		//trainer.setDebug(SHOW_RESULTS);
		//trainer.setDebug(SHOW_PROCESS);
		trainer.setDebug(SHOW_ALL_MSG);
		c->setDebug(SHOW_ALL_MSG);
		trainer.setClassifier(c);
		trainer.loadData(argv[2]);
		trainer.run();
		if(argc>3)c->save(argv[3]);
	}
	else{
		printf(PRINT_RED"ERROR classifier \"%s\" or parameter\n"PRINT_END,argv[1]);	
	}
	return 0;
}

