#rm -r -f build
mkdir -p build
cd build
cmake ../src
make

cp FeatureLib/libFeatureLib.dylib /usr/local/lib
cp ClassifierLib/libClassifierLib.dylib /usr/local/lib

mkdir -p /usr/local/include/FeatureLib
mkdir -p /usr/local/include/ClassifierLib
find ../src/FeatureLib -name '*.h' | awk '{print "cp " $0 " /usr/local/include/FeatureLib"}' | sh
find ../src/ClassifierLib -name '*.h' | awk '{print "cp " $0 " /usr/local/include/ClassifierLib"}' | sh

mkdir -p ../bin
cp pca ../bin
cp pca_feature ../bin
cp base_trainer ../bin

