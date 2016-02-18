include (Classifier/CMakeLists.txt)
add_library(ClassifierLib
	SHARED
	${Classifier}
)

target_link_libraries(ClassifierLib CodeLib)
target_link_libraries(ClassifierLib armadillo)





