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

int main() {

    // Parameters for H
    double L = 9.4486;
    double Ng = 20; // roughly optimized
    double E_cutoff = 3.67493;
    E_cutoff = 0.5; // to make the basis smaller (7 instead of 214), see if that impacts runtime

    Atom H_atom ={0.001,0.0,0.0,1}; // shift off center a little bit to avoid integrating on a grid point
    std::vector<Atom> atoms = {H_atom};
    std::vector<PB_wavefunction> basis = construct_basis(E_cutoff, L);
    arma::sp_mat T = construct_T(basis);
    arma::mat C(basis.size(), basis.size(), arma::fill::zeros); // make an empty coef matrix just for calculating V_ext
    arma::vec C_vec = C_V_ext(basis, atoms, L);
    C_vec.print();
    arma::mat V_ext_mat = construct_V_mat(V_ext, C, C_vec, basis, atoms, Ng, L);
    V_ext_mat.print();

    return 0;
}
