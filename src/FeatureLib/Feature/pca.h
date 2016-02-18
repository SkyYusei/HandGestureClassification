#pragma once
#include <armadillo>
// return the pca(eigen face) of the data
// when num <= 0, return all eigen face
// else return only num eigen face
// num must small than the maximum number of the eigen face
void pca(const arma::mat& data, arma::mat& eigen, int num = 0);

// get the pca feature
// save in matrix feature
// when err_term_dim =0, without err_term
// else reconstruct the image and calculate the square error
void pca_feature(const arma::mat& data, const arma::mat& eigen, arma::mat& feature, bool err_term = false);

