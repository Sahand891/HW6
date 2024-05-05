//
// Created by Sahand Adibnia on 5/4/24.
//

#include <iostream>
#include <fstream>
#include <vector>
#include <cmath>
#include <functional>
#include <armadillo>
#include "basis.h"
#include "KE.h"
#include "Numerical Integration.h"
#include "Fock Matrix.h"

#ifndef HW6_DIAGONALIZE_CONVERGE_H
#define HW6_DIAGONALIZE_CONVERGE_H

struct iteration_data {

    // Constant for every iteration
    std::vector<PB_wavefunction> basis;
    int Ng;
    double L;
    std::vector<Atom> atoms;
    int p;
    int q;


    // Changers on every iteration
    arma::mat C_alpha_old;
    arma::mat C_beta_old;

    arma::mat fock_alpha;
    arma::mat fock_beta;

    arma::mat C_alpha_new;
    arma::mat C_beta_new;

    int iteration_count;


    // check for convergence just based on alpha coefficient matrix
    bool converged = arma::approx_equal(C_alpha_new,C_alpha_old,"absdiff",1e-6);

    // Other information in case we need it


};

#endif //HW6_DIAGONALIZE_CONVERGE_H
