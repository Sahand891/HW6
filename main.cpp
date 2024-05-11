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
#include "Perform_DFT.h"

int main() {

    // Parameters for grid and basis
    double L = 9.4486;
    int Ng = 20; // roughly optimized
    double E_cutoff = 3.67493;


    // Parameters for H atom
    Atom H_atom ={0.0,0.0,0.0,1};
    std::vector<Atom> H_atoms = {H_atom};
    int H_p = 1;
    int H_q = 0;


    perform_DFT_V_ext_only("H", L, Ng, E_cutoff, H_atoms, H_p, H_q);
    perform_full_DFT("H", L, Ng, E_cutoff, H_atoms, H_p, H_q);






    // Parameters for He atom
    Atom He_atom ={0.0,0.0,0.0,2};
    std::vector<Atom> He_atoms = {He_atom};
    int He_p = 1;
    int He_q = 1;

    perform_DFT_V_ext_only("He", L, Ng, E_cutoff, He_atoms, He_p, He_q);
    perform_full_DFT("He", L, Ng, E_cutoff, He_atoms, He_p, He_q);

    return 0;
}