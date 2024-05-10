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
    //E_cutoff = 2; // to make the basis smaller (7 instead of 214) ... drastically decreases runtime

    Atom H_atom ={0.0,0.0,0.0,1}; // can shift off center a little bit to avoid integrating on a grid point
    std::vector<Atom> atoms = {H_atom};
    std::vector<PB_wavefunction> basis = construct_atom_basis(E_cutoff, L);
    arma::mat grid = build_grid(L, Ng);
    int p = 1;
    int q = 0;


    arma::mat C(basis.size(), basis.size(), arma::fill::zeros); // make an empty coef matrix just for calculating V_ext
    arma::vec C_vec = C_V_ext(basis, atoms, L); // make it for the basis, then store it away


    arma::sp_mat T = construct_T(basis);


    // The limiting factor is this function here for calculating the V matrix!
    // arma::mat V_ext_mat = construct_V_mat_slow(V_ext, C, C_vec, basis, atoms, Ng, L, p);
    // V_ext_mat.print();

    // Making a hopefully faster version of the same function ... in steps
    // 1st step: compute value of V_ext at all grid points - CHECK, THIS WAS VERY FAST!

    auto f = [&](double x, double y, double z) {
        return V_ext(x,y,z,C,C_vec,basis,atoms,Ng,L,p);
    };

    arma::vec V_ext_at_grid = customf_at_gridpoints(grid, f, Ng);
    //V_ext_at_grid.print();

    // 2nd step: multiply the V_ext_at_grid vector by vectors of two basis functions (also evaluated at grid points)
    arma::vec basis_vec1 = basisf_at_gridpoints(grid, basis[0], Ng);
    arma::vec basis_vec2 = basisf_at_gridpoints(grid, basis[1], Ng);
    arma::vec inner_prod_1 = basis_vec1 % V_ext_at_grid % basis_vec2;
    //inner_prod_1.print();

    // 3rd step: accumulate inner product by adding up all the values in the vector, then multiply by the weight to do the complete grid integration!
    double val = arma::accu(inner_prod_1) * pow((L / Ng), 3);
    //std::cout << val << std::endl;

    // NICE! It can do a single matrix element pretty damn quickly!

    // 4th step: Now do this for the ENTIRE V_mat matrix. To start, let's make a matrix of all the basis functions evaluated at all the gridpoints
    arma::mat whole_basis_at_grid = basis_grid(grid, basis, Ng);
    //whole_basis_at_grid.print(); // note that this prints weirdly tho
    //whole_basis_at_grid.col(0).print(); // just to test that it works, and it looks like it does!

    // 5th step: Now take every combination of columns in the whole_basis_at_grid matrix, "inner-product" them element-wise by the V_ext_at_grid vector
    arma::mat V_ext_mat(basis.size(), basis.size(), arma::fill::zeros);
    for (int i=0; i < whole_basis_at_grid.n_cols; i++) {
        for (int j=0; j < whole_basis_at_grid.n_cols; j++) {
            arma::vec bf_vec_1 = whole_basis_at_grid.col(i);
            arma::vec bf_vec_2 = whole_basis_at_grid.col(j);
            arma::vec inner_prod_at_grid = bf_vec_1 % V_ext_at_grid % bf_vec_2;
            V_ext_mat(i,j) = arma::accu(inner_prod_at_grid) * pow((L / Ng), 3);
        }
    }

    V_ext_mat.print(); // it's super fast!!!!

    // Double check that the output is the same: as the other one
    std::cout << "\n\n\n" << std::endl;
    construct_V_mat_slow(V_ext, C, C_vec, basis, atoms, Ng, L, p).print(); // YUP! OUTPUTS THE SAME MATRIX!!!


    return 0;
}


// Old int main::

//int main() {
//
//    // Parameters for H
//    double L = 9.4486;
//    int Ng = 20; // roughly optimized
//    double E_cutoff = 3.67493;
//    //E_cutoff = 2; // to make the basis smaller (7 instead of 214) ... drastically decreases runtime
//
//    Atom H_atom ={0.0,0.0,0.0,1}; // can shift off center a little bit to avoid integrating on a grid point
//    std::vector<Atom> atoms = {H_atom};
//    std::vector<PB_wavefunction> basis = construct_atom_basis(E_cutoff, L);
//    int p = 1;
//    int q = 0;
//
//
//    arma::mat C(basis.size(), basis.size(), arma::fill::zeros); // make an empty coef matrix just for calculating V_ext
//    arma::vec C_vec = C_V_ext(basis, atoms, L); // make it for the basis, then store it away
//    C_vec.print();
//
//
//    arma::sp_mat T = construct_T(basis);
//    arma::mat V_ext_mat = construct_V_mat(V_ext, C, C_vec, basis, atoms, Ng, L, p);
//    V_ext_mat.print();
//
//
//    // Now let's try to do the shitty convergence on this bitch!!!
//    iteration_data initial_itdata = initialize_DFT(basis, Ng, L, atoms, p, q, C_vec, C_vec, C, V_ext_mat);
//    std::vector<iteration_data> start_vec = {initial_itdata};
//    std::function<arma::mat(std::vector<PB_wavefunction>&, arma::mat&, arma::vec&, arma::vec&, const std::vector<Atom>&, int, double, int)> func_wrapper = construct_Fmat_Vext_only;
//    std::vector<iteration_data> all_itdata = converge_DFT(initial_itdata, start_vec, func_wrapper);
//    iteration_data conv_itdata = obtain_converged_data(all_itdata);
//
//    conv_itdata.C_alpha_new.print();
//    conv_itdata.C_beta_new.print();
//
//
//    std::cout << "KE: " << kinetic_energy(conv_itdata, T) << std::endl;
//    std::cout << "External energy: " << external_energy(conv_itdata) << std::endl;
//    std::cout << "Total energy: " << kinetic_energy(conv_itdata, T) + external_energy(conv_itdata) << std::endl;
//
//
//    return 0;
//}
