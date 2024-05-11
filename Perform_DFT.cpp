//
// Created by Sahand Adibnia on 5/10/24.
//

#include "Perform_DFT.h"

// For atoms ONLY

void perform_DFT_V_ext_only(std::string atom_name, double L, int Ng, double E_cutoff, std::vector<Atom> &atoms, int p, int q) {

    // Basis
    std::vector<PB_wavefunction> basis = construct_atom_basis(E_cutoff, L);

    // Grid
    arma::mat grid = build_grid(L, Ng);
    arma::mat whole_basis_grid = basis_grid(grid, basis, Ng);

    // Initial C guess
    arma::mat C(basis.size(), basis.size(), arma::fill::zeros); // make an empty coef matrix just for calculating V_ext

    // Coefs for V_ext density fitting (constant)
    arma::vec C_vec_ext = C_V_ext(basis, atoms, L); // make it for the basis, then store it away

    // T matrix (constant)
    arma::sp_mat T = construct_T(basis);

    // Initial V_ext matrix
    arma::mat V_ext_mat = construct_V_mat_fast(V_ext, C, C_vec_ext, basis, atoms, Ng, L, p, grid, whole_basis_grid);

    // Initial F matrix
    arma::mat F_initial = T + V_ext_mat;

    iteration_data initial_itdata = initialize_DFT(basis, Ng, L, atoms, p, q, C_vec_ext, C_vec_ext, C, F_initial, grid, whole_basis_grid);
    std::vector<iteration_data> start_vec = {initial_itdata};
    std::function<arma::mat(std::vector<PB_wavefunction>&, arma::mat&, arma::vec&, arma::vec&, const std::vector<Atom>&, int, double, int, arma::mat&, arma::mat&)> func_wrapper = construct_Fmat_Vext_only;
    std::vector<iteration_data> all_itdata = converge_DFT(initial_itdata, start_vec, func_wrapper);
    iteration_data conv_itdata = obtain_converged_data(all_itdata);

    std::cout << atom_name << " atom using ONLY T and V_ext:" << std::endl;
    std::cout << "KE: " << kinetic_energy(conv_itdata, T) << std::endl;
    std::cout << "External energy: " << external_energy(conv_itdata) << std::endl;
    std::cout << "Total energy: " << kinetic_energy(conv_itdata, T) + external_energy(conv_itdata) << "\n" << std::endl;

}


void perform_full_DFT(std::string atom_name, double L, int Ng, double E_cutoff, std::vector<Atom> &atoms, int p, int q) {

    // Basis
    std::vector<PB_wavefunction> basis = construct_atom_basis(E_cutoff, L);

    // Grid
    arma::mat grid = build_grid(L, Ng);
    arma::mat whole_basis_grid = basis_grid(grid, basis, Ng);

    // Initial C guess
    arma::mat C(basis.size(), basis.size(), arma::fill::zeros); // make an empty coef matrix just for calculating V_ext

    // Coefs for V_ext density fitting (constant)
    arma::vec C_vec_ext = C_V_ext(basis, atoms, L); // make it for the basis, then store it away

    // T matrix (constant)
    arma::sp_mat T = construct_T(basis);

    // Initial coefs for V_hart density fitting (will change during iterations)
    arma::vec C_vec_hartree = C_V_hartree(basis, C, C, Ng, L, p, q, grid, whole_basis_grid); // make it for the basis, then store it away

    // Initial V matrices
    arma::mat V_ext_mat = construct_V_mat_fast(V_ext, C, C_vec_ext, basis, atoms, Ng, L, p, grid, whole_basis_grid);
    arma::mat V_hartree_mat = construct_V_mat_fast(V_hartree, C, C_vec_hartree, basis, atoms, Ng, L, p, grid, whole_basis_grid);
    arma::mat V_xc_mat = construct_V_mat_fast(V_xc, C, C_vec_hartree, basis, atoms, Ng, L, p, grid, whole_basis_grid);

    // Initial F matrix
    arma::mat F_initial = T + V_ext_mat + V_hartree_mat + V_xc_mat;

    iteration_data initial_itdata = initialize_DFT(basis, Ng, L, atoms, p, q, C_vec_ext, C_vec_ext, C, F_initial, grid, whole_basis_grid);
    std::vector<iteration_data> start_vec = {initial_itdata};
    std::function<arma::mat(std::vector<PB_wavefunction>&, arma::mat&, arma::vec&, arma::vec&, const std::vector<Atom>&, int, double, int, arma::mat&, arma::mat&)> func_wrapper = construct_Fock_matrix;
    std::vector<iteration_data> all_itdata = converge_DFT(initial_itdata, start_vec, func_wrapper);
    iteration_data conv_itdata = obtain_converged_data(all_itdata);

    std::cout << atom_name << " atom using full DFT" << std::endl;
    std::cout << "KE: " << kinetic_energy(conv_itdata, T) << std::endl;
    std::cout << "External energy: " << external_energy(conv_itdata) << std::endl;
    std::cout << "Hartree energy: " << hartree_energy(conv_itdata) << std::endl;
    std::cout << "Exchange-correlation energy: " << xc_energy(conv_itdata) << std::endl;
    std::cout << "Total energy: " << total_energy(conv_itdata, T) << "\n" << std::endl;



}

