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

arma::mat find_MO_coefs(arma::mat &F);
arma::mat find_eigenvalues(arma::mat &F);

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
    arma::vec C_vec_ext;


    // Changers on every iteration
    arma::mat C_alpha_old;
    arma::mat C_beta_old;
    arma::vec C_vec_hartree_old; // the old C's make the old

    arma::mat fock_alpha; // the old C's + the old C_vec_hartree makes these
    arma::mat fock_beta;

    arma::mat C_alpha_new; // From diagonalization of focks
    arma::mat C_beta_new;
    arma::vec C_vec_hartree_new; // From new C's

    int iteration_count;


    // check for convergence just based on alpha coefficient matrix
    bool converged = arma::approx_equal(C_alpha_new,C_alpha_old,"absdiff",1.14545);



    // Other information in case we need it
    arma::mat V_ext_mat_alpha_old = construct_V_mat_fast(V_ext, C_alpha_old, C_vec_ext, basis, atoms, Ng, L, p, grid, whole_grid_basis);
    arma::mat V_hart_mat_alpha_old = construct_V_mat_fast(V_hartree, C_alpha_old, C_vec_hartree_old, basis, atoms, Ng, L, p, grid, whole_grid_basis);
    arma::mat V_xc_mat_alpha_old = construct_V_mat_fast(V_xc, C_alpha_old, C_vec_hartree_old, basis, atoms, Ng, L, p, grid, whole_grid_basis);
    arma::mat F_alpha_full_old = construct_Fock_matrix(basis, C_alpha_old, C_vec_hartree_old, C_vec_ext, atoms, Ng, L, p, grid, whole_grid_basis);

    arma::mat V_ext_mat_alpha_new = construct_V_mat_fast(V_ext, C_alpha_new, C_vec_ext, basis, atoms, Ng, L, p, grid, whole_grid_basis);
    arma::mat V_hart_mat_alpha_new = construct_V_mat_fast(V_hartree, C_alpha_new, C_vec_hartree_new, basis, atoms, Ng, L, p, grid, whole_grid_basis);
    arma::mat V_xc_mat_alpha_new = construct_V_mat_fast(V_xc, C_alpha_new, C_vec_hartree_new, basis, atoms, Ng, L, p, grid, whole_grid_basis);
    arma::mat F_alpha_full_new = construct_Fock_matrix(basis, C_alpha_new, C_vec_hartree_new, C_vec_ext, atoms, Ng, L, p, grid, whole_grid_basis);

    // Eigenvalues of these matrices
    arma::vec F_alpha_old_eigvals = find_eigenvalues(F_alpha_full_old);
    arma::vec F_alpha_new_eigvals = find_eigenvalues(F_alpha_full_new);

};



iteration_data initialize_DFT(const std::vector<PB_wavefunction> &basis, int Ng, double L, const std::vector<Atom> &atoms, int p, int q, arma::vec &C_vec_hartree, arma::vec &C_vec_ext, arma::mat& C_guess, arma::mat& F_initial, arma::mat &grid, arma::mat &whole_grid_basis);
std::vector<iteration_data> converge_DFT(const iteration_data &it_data, std::vector<iteration_data> &it_data_vec, std::function<arma::mat(std::vector<PB_wavefunction>&, arma::mat&, arma::vec&, arma::vec&, const std::vector<Atom>&, int, double, int, arma::mat&, arma::mat&)>& F_func);


iteration_data obtain_converged_data(std::vector<iteration_data> &converged_it_data);

#endif //HW6_DIAGONALIZE_CONVERGE_H
