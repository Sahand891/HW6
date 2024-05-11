//
// Created by Sahand Adibnia on 5/10/24.
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
#include "Energy.h"

#ifndef HW6_PERFORM_DFT_H
#define HW6_PERFORM_DFT_H

void perform_DFT_V_ext_only(std::string atom_name, double L, int Ng, double E_cutoff, std::vector<Atom> &atoms, int p, int q);
void perform_full_DFT(std::string atom_name, double L, int Ng, double E_cutoff, std::vector<Atom> &atoms, int p, int q);

#endif //HW6_PERFORM_DFT_H
