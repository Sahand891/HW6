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
    arma::mat grid;
    arma::mat whole_grid_basis;
    std::vector<Atom> atoms;
    int p;
    int q;
    arma::vec C_vec_hartree;
    arma::vec C_vec_ext;


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



arma::mat find_MO_coefs(arma::mat &F);

iteration_data initialize_DFT(const std::vector<PB_wavefunction> &basis, int Ng, double L, const std::vector<Atom> &atoms, int p, int q, arma::vec &C_vec_hartree, arma::vec &C_vec_ext, arma::mat& C_guess, arma::mat& F_initial, arma::mat &grid, arma::mat &whole_grid_basis);
std::vector<iteration_data> converge_DFT(const iteration_data &it_data, std::vector<iteration_data> &it_data_vec, std::function<arma::mat(std::vector<PB_wavefunction>&, arma::mat&, arma::vec&, arma::vec&, const std::vector<Atom>&, int, double, int, arma::mat&, arma::mat&)>& F_func);


iteration_data obtain_converged_data(std::vector<iteration_data> &converged_it_data);

#endif //HW6_DIAGONALIZE_CONVERGE_H
