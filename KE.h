//
// Created by Sahand Adibnia on 4/25/24.
//

#include <iostream>
#include <fstream>
#include <vector>
#include <cmath>
#include <functional>
#include <armadillo>
#include "basis.h"

#ifndef HW6_KE_H
#define HW6_KE_H

double T_diagonal_component(PB_wavefunction w);
std::vector<PB_wavefunction> construct_basis(double E_cutoff, double L);
arma::sp_mat construct_T(std::vector<PB_wavefunction> &basis);

#endif //HW6_KE_H
