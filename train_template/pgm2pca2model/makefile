BUILD_DIR = ../../bin/
EIGEN_NUM = 30
CLASSIFIER = adaboost:50,0.00001,naive,30000
#CLASSIFIER = adaboost:20,0.0001,naive,10000
#CLASSIFIER = naive:1000
#CLASSIFIER = near

PGM_WIDTH = 64
PGM_HEIGHT = 36

MODEL_NAME = test.model

all: model

# eigen face
eigen: pgm/eigen 
	mkdir -p eigen 
	$(BUILD_DIR)/./pca pgm/eigen $(EIGEN_NUM) eigen/ eigen

# feature
feature: eigen pgm/train_pos pgm/test_pos pgm/train_neg pgm/test_neg
	mkdir -p feature
	$(BUILD_DIR)/./pca_feature eigen/eigen.dat feature pgm/train_pos train_positive
	$(BUILD_DIR)/./pca_feature eigen/eigen.dat feature pgm/train_neg train_negative
	$(BUILD_DIR)/./pca_feature eigen/eigen.dat feature pgm/test_pos test_positive
	$(BUILD_DIR)/./pca_feature eigen/eigen.dat feature pgm/test_neg test_negative

# model
model: feature
	mkdir -p model
	$(BUILD_DIR)/./base_trainer $(CLASSIFIER) feature/ model/$(MODEL_NAME)

clean_model:
	rm -r -f model

clean:
	rm -r -f eigen
	rm -r -f feature
	rm -r -f model


