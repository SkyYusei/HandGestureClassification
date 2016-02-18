cmake_minimum_required(VERSION 3.0)
project(ObjectDetectionFramework)

#################################### LIB
include_directories(/usr/local/include/CodeLib)
find_library(CODELIB CodeLib PATH /usr/local/lib)

include_directories(/usr/local/include)
find_library(ARMA NAMES armadillo PATH /usr/local/lib SHARED)

include_directories(./)

add_subdirectory (ClassifierLib/) 
include_directories(ClassifierLib)
include_directories(ClassifierLib/Classifier)

add_subdirectory (FeatureLib) 
include_directories(FeatureLib)
include_directories(FeatureLib/Feature)

add_subdirectory (Trainer) 
include_directories(Trainer)

#############################################  test
#add_executable(test_matrix test/test_matrix.cpp)
#target_link_libraries (test_matrix ${ARMA})
#target_link_libraries (test_matrix GUI)
#target_link_libraries (test_matrix ${OPENCV})

############################################# process 
add_executable(pca process/FeatureExtraction/pca.cpp)
target_link_libraries (pca ${ARMA})
target_link_libraries (pca ${CODELIB})
target_link_libraries (pca FeatureLib)

add_executable(pca_feature process/FeatureExtraction/pca_feature.cpp)
target_link_libraries (pca_feature ${ARMA})
target_link_libraries (pca_feature ${CODELIB})
target_link_libraries (pca_feature FeatureLib)

add_executable(base_trainer process/ClassiferTrainer/base_trainer.cpp)
target_link_libraries (base_trainer ClassifierLib)
target_link_libraries (base_trainer Trainer)
target_link_libraries (base_trainer ${ARMA})
target_link_libraries (base_trainer ${CODELIB})

