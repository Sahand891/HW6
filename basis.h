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
    double n_sq = pow(arma::norm(n),2);
};

struct Atom {
    // General atom data type

    double X,Y,Z;
    int Z_; // atomic number = nuclear charge

    // position vector
    arma::vec pos_vec = {X,Y,Z};

};

double PB_func(double x, double y, double z, const PB_wavefunction &w);


#endif //HW6_BASIS_H
