include (Feature/CMakeLists.txt)
include (ImgProcess/CMakeLists.txt)
add_library(FeatureLib
	SHARED
	${Feature} 
	${ImgProcess}
	projection.cpp
)
target_link_libraries(FeatureLib CodeLib)
target_link_libraries(FeatureLib armadillo)





