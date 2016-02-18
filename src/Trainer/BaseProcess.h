#pragma once
#include <armadillo>
//#include "BaseClassifier.h"
#include "codelib.h"


class BaseProcess:public BaseDebug
{
private:
protected:
public:
	BaseProcess(){};
	~BaseProcess(){};
	virtual void init(const char * params){};
	virtual void run()=0;
};

