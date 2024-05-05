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

double density(double x, double y, double z, arma::mat &C, const std::vector<PB_wavefunction> &basis);

double V_xc(double x, double y, double z, arma::mat &C, arma::vec &C_vec, std::vector<PB_wavefunction> &basis, const std::vector<Atom> &atoms, double Ng, double L);

arma::vec C_V_ext(const std::vector<PB_wavefunction> &basis, const std::vector<Atom> &atoms, double L);
double V_ext(double x, double y, double z, arma::mat &C, arma::vec &C_vec, std::vector<PB_wavefunction> &basis, const std::vector<Atom> &atoms, double Ng, double L);

arma::vec C_V_hartree(const std::vector<PB_wavefunction> &basis, const std::vector<Atom> &atoms, arma::mat &C, double Ng, double L);
double V_hartree(double x, double y, double z, arma::mat &C, arma::vec &C_vec, std::vector<PB_wavefunction> &basis, const std::vector<Atom> &atoms, double Ng, double L);


arma::mat construct_V_mat(std::function<double(double, double, double, arma::mat&, arma::vec&, std::vector<PB_wavefunction>&, const std::vector<Atom>&, double, double)> V, arma::mat &C, arma::vec &C_vec, std::vector<PB_wavefunction> &basis, const std::vector<Atom> &atoms, double Ng, double L);

#endif //HW6_FOCK_MATRIX_H
