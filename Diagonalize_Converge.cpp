//
// Created by Sahand Adibnia on 5/4/24.
//

#include "Diagonalize_Converge.h"

arma::mat find_MO_coefs(arma::mat &F) {
    arma::vec energies;
    arma::mat C;
    arma::eig_sym(energies, C, F); // can use eig_sym because F is guranteed to be symmetric

    return C;
}


arma::mat find_eigenvalues(arma::mat &F) {
    arma::vec energies;
    arma::mat C;
    arma::eig_sym(energies, C, F); // can use eig_sym because F is guranteed to be symmetric

    return energies;
}


// Force both alpha and beta electrons to have the same "first guess" coefficient matrix
iteration_data initialize_DFT(const std::vector<PB_wavefunction> &basis, int Ng, double L, const std::vector<Atom> &atoms, int p, int q, arma::vec &C_vec_hartree, arma::vec &C_vec_ext, arma::mat& C_guess, arma::mat& F_initial, arma::mat &grid, arma::mat &whole_grid_basis) {

    // Obtain the new guesses for the coefficient matrix based on the calculated Fock matrix ... same for alpha and beta
    arma::mat C_new = find_MO_coefs(F_initial);

    // Obtain new C vec hartree for density fitting from the new C's
    arma::vec C_vec_hartree_new = C_V_hartree(basis, C_new, C_new, Ng, L, p, q, grid, whole_grid_basis);

    iteration_data it_data = {basis,
                              Ng,
                              L,
                              grid,
                              whole_grid_basis,
                              atoms,
                              p,
                              q,
                              C_vec_ext,
                              C_guess,
                              C_guess,
                              C_vec_hartree,
                              F_initial,
                              F_initial,
                              C_new,
                              C_new,
                              C_vec_hartree_new,
                              0,
                              false};

    return it_data;

}


// Note to self: the differentiator between alpha and beta electrons comes from the exchange/correlation matrix, which is dependent on the densities!
std::vector<iteration_data> converge_DFT(const iteration_data &it_data, std::vector<iteration_data> &it_data_vec, std::function<arma::mat(std::vector<PB_wavefunction>&, arma::mat&, arma::vec&, arma::vec&, const std::vector<Atom>&, int, double, int, arma::mat&, arma::mat&)>& F_func) {

    // Base case
    if (it_data.converged) {
        return it_data_vec;
    }

    std::cout << "iteration" << std::endl;

    // Extracting constant information for each iteration
    std::vector<PB_wavefunction> basis = it_data.basis;
    int Ng = it_data.Ng;
    double L = it_data.L;
    arma::mat grid = it_data.grid;
    arma::mat whole_grid_basis = it_data.whole_grid_basis;
    std::vector<Atom> atoms = it_data.atoms;
    int p = it_data.p;
    int q = it_data.q;
    arma::vec C_vec_ext = it_data.C_vec_ext;


    // Old coefficient matrices for this iteration are the new ones from the previous iteration
    arma::mat C_alpha_old = it_data.C_alpha_new;
    arma::mat C_beta_old = it_data.C_beta_new;
    arma::vec C_vec_hartree_old = it_data.C_vec_hartree_new;


    // Make the new fock matrix based on the coefficient matrices from the previous iteration
    arma::mat Fock_alpha = F_func(basis, C_alpha_old, C_vec_hartree_old, C_vec_ext, atoms, Ng, L, p, grid, whole_grid_basis);
    arma::mat Fock_beta = F_func(basis, C_beta_old, C_vec_hartree_old, C_vec_ext, atoms, Ng, L, q, grid, whole_grid_basis);


    // Diagonalize the fock matrices and extra MO coefficients
    arma::mat C_alpha_new = find_MO_coefs(Fock_alpha);
    arma::mat C_beta_new = find_MO_coefs(Fock_beta);


    // Obtain new C vec hartree for density fitting from the new C's
    arma::vec C_vec_hartree_new = C_V_hartree(basis, C_alpha_new, C_beta_new, Ng, L, p, q, grid, whole_grid_basis);

    // Just calculate this to check for convergence
    arma::mat actual_old_C_alpha = it_data.C_alpha_old;
    arma::vec actual_old_C_C_vec_hartree = it_data.C_vec_hartree_old;
    arma::mat Fock_alpha_old = F_func(basis, actual_old_C_alpha, actual_old_C_C_vec_hartree, C_vec_ext, atoms, Ng, L, p, grid, whole_grid_basis);


    // Convergence check
    bool is_converged = arma::approx_equal(Fock_alpha,Fock_alpha_old,"absdiff",1e-8);


    // Compile all the appropriate info into a new iteration_data object
    iteration_data new_it_data = {basis,
                                  Ng,
                                  L,
                                  grid,
                                  whole_grid_basis,
                                  atoms,
                                  p,
                                  q,
                                  C_vec_ext,
                                  C_alpha_old,
                                  C_beta_old,
                                  C_vec_hartree_old,
                                  Fock_alpha,
                                  Fock_beta,
                                  C_alpha_new,
                                  C_beta_new,
                                  C_vec_hartree_new,
                                  it_data.iteration_count+1,
                                  is_converged};


//    std::cout << arma::accu(abs(it_data.V_hart_mat_alpha_old - it_data.V_hart_mat_alpha_new)) << std::endl;
//    std::cout << arma::accu(abs(it_data.V_xc_mat_alpha_new - it_data.V_xc_mat_alpha_old)) << std::endl;
//    std::cout << arma::accu(abs(it_data.V_ext_mat_alpha_new - it_data.V_ext_mat_alpha_old)) << std::endl;
//    std::cout << arma::approx_equal(Fock_alpha.t(), Fock_alpha, "absdiff", 1e-3) << " Symmetry check" << std::endl;
//    std::cout << arma::approx_equal(it_data.F_alpha_full_old, Fock_alpha, "absdiff", 1e-3) << " Fock's comparison" << std::endl;
//    std::cout << arma::approx_equal(C_alpha_new, C_alpha_old, "absdiff", 1e-3) << " C's comparison" << std::endl;
//    std::cout << abs(C_alpha_new - C_alpha_old).max() << " Max C's diff" << std::endl;
//    std::cout << " Fock matrix eigenvalues diff - check condition of matrix:" << std::endl;
//    it_data.F_alpha_old_eigvals.print();


    // Adding that new set of iteration data to the original vector of iteration data
    it_data_vec.push_back(new_it_data);


    return converge_DFT(new_it_data, it_data_vec, F_func);

}


iteration_data obtain_converged_data(std::vector<iteration_data> &converged_it_data) {
    return converged_it_data.back();
}



