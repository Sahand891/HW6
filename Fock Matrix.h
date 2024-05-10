//
// Created by Sahand Adibnia on 4/26/24.
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

#ifndef HW6_FOCK_MATRIX_H
#define HW6_FOCK_MATRIX_H

double density(double x, double y, double z, arma::mat &C, const std::vector<PB_wavefunction> &basis, int num_electrons);

double V_xc(double x, double y, double z, arma::mat &C, arma::vec &C_vec, std::vector<PB_wavefunction> &basis, const std::vector<Atom> &atoms, int Ng, double L, int num_electrons);

arma::vec C_V_ext(const std::vector<PB_wavefunction> &basis, const std::vector<Atom> &atoms, double L);
double V_ext(double x, double y, double z, arma::mat &C, arma::vec &C_vec, std::vector<PB_wavefunction> &basis, const std::vector<Atom> &atoms, int Ng, double L, int num_electrons);

arma::vec C_V_hartree(const std::vector<PB_wavefunction> &basis, const std::vector<Atom> &atoms, arma::mat &C, int Ng, double L, int num_electrons);
double V_hartree(double x, double y, double z, arma::mat &C, arma::vec &C_vec, std::vector<PB_wavefunction> &basis, const std::vector<Atom> &atoms, int Ng, double L, int num_electrons);

arma::mat construct_V_mat_slow(std::function<double(double, double, double, arma::mat&, arma::vec&, std::vector<PB_wavefunction>&, const std::vector<Atom>&, int, double, int)> V, arma::mat &C, arma::vec &C_vec, std::vector<PB_wavefunction> &basis, const std::vector<Atom> &atoms, int Ng, double L, int num_electrons);



arma::mat construct_Fock_matrix(std::vector<PB_wavefunction> &basis, arma::mat &C, arma::vec &C_vec_hartree, arma::vec &C_vec_ext, const std::vector<Atom> &atoms, int Ng, double L, int num_electrons);
arma::mat construct_Fmat_Vext_only(std::vector<PB_wavefunction> &basis, arma::mat &C, arma::vec &C_vec_hartree, arma::vec &C_vec_ext, const std::vector<Atom> &atoms, int Ng, double L, int num_electrons);

#endif //HW6_FOCK_MATRIX_H
