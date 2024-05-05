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

int main() {

    // Parameters for H
    double L = 9.4486;
    int Ng = 20; // roughly optimized
    double E_cutoff = 3.67493;
    E_cutoff = 0.5; // to make the basis smaller (7 instead of 214) ... drastically decreases runtime

    Atom H_atom ={0.0001,0.0,0.0,1}; // shift off center a little bit to avoid integrating on a grid point
    std::vector<Atom> atoms = {H_atom};
    std::vector<PB_wavefunction> basis = construct_basis(E_cutoff, L);
    int p = 1;
    int q = 0;


    arma::mat C(basis.size(), basis.size(), arma::fill::zeros); // make an empty coef matrix just for calculating V_ext
    arma::vec C_vec = C_V_ext(basis, atoms, L); // make it for the basis, then store it away
    //C_vec.print();


    arma::sp_mat T = construct_T(basis);
    arma::mat V_ext_mat = construct_V_mat(V_ext, C, C_vec, basis, atoms, Ng, L, p);
    //V_ext_mat.print();


    // Now let's try to do the shitty convergence on this bitch!!!
    iteration_data initial_itdata = initialize_DFT(basis, Ng, L, atoms, p, q, C_vec, C_vec, C, V_ext_mat);
    std::vector<iteration_data> start_vec = {initial_itdata};
    std::function<arma::mat(std::vector<PB_wavefunction>&, arma::mat&, arma::vec&, arma::vec&, const std::vector<Atom>&, int, double, int)> func_wrapper = construct_Fmat_Vext_only;
    std::vector<iteration_data> all_itdata = converge_DFT(initial_itdata, start_vec, func_wrapper);
    iteration_data conv_itdata = obtain_converged_data(all_itdata);


    std::cout << "KE: " << kinetic_energy(conv_itdata, T) << std::endl;
    std::cout << "External energy: " << external_energy(conv_itdata) << std::endl;
    std::cout << "Total energy: " << kinetic_energy(conv_itdata, T) + external_energy(conv_itdata) << std::endl;


    return 0;
}
