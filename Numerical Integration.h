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

// More intelligent, allows for nice matrix multiplication!
arma::vec grid_int_vec(const std::function<double(double,double,double)> &f, int Ng, double L);

// More brute force
double quad_3D_grid_integration(const std::function<double(double,double,double)> &f, int Ng, double L);

#endif //HW6_NUMERICAL_INTEGRATION_H
