//
// Created by Sahand Adibnia on 4/17/24.
//

#include <iostream>
#include <fstream>
#include <vector>
#include <cmath>
#include <functional>
#include <armadillo>

#ifndef HW6_BASIS_H
#define HW6_BASIS_H

struct PB_wavefunction {
    double n_x, n_y, n_z;
    double L;

    arma::vec n = {n_x, n_y, n_z};
};

double PB_func(double x, double y, double z, PB_wavefunction w);


#endif //HW6_BASIS_H
