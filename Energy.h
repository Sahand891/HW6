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
#include "Diagonalize_Converge.h"

#ifndef HW6_ENERGY_H
#define HW6_ENERGY_H


double kinetic_energy(const iteration_data &final_it_data, arma::sp_mat &T);
double external_energy(const iteration_data &final_it_data);
double hartree_energy(const iteration_data &final_it_data);
double xc_energy(const iteration_data &final_it_data);

double total_energy(const iteration_data &final_it_data, arma::sp_mat &T);

#endif //HW6_ENERGY_H
