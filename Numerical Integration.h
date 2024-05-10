//
// Created by Sahand Adibnia on 4/29/24.
//

#include <iostream>
#include <fstream>
#include <vector>
#include <cmath>
#include <functional>
#include <armadillo>
#include "basis.h"
#include "KE.h"

#ifndef HW6_NUMERICAL_INTEGRATION_H
#define HW6_NUMERICAL_INTEGRATION_H

// More intelligent, allows for nice matrix/vector multiplications!
arma::mat build_grid(double L, int Ng);
arma::vec basisf_at_gridpoints(arma::mat &grid, PB_wavefunction &basis_func, int Ng);
arma::vec customf_at_gridpoints(arma::mat &grid, const std::function<double(double,double,double)> &f, int Ng);
arma::mat basis_grid(arma::mat &grid, std::vector<PB_wavefunction> &basis, int Ng);


// More brute force (takes very very long)
double quad_3D_grid_integration(const std::function<double(double,double,double)> &f, int Ng, double L);

#endif //HW6_NUMERICAL_INTEGRATION_H
