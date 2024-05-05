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


// Force both alpha and beta electrons to have the same "first guess" coefficient matrix
iteration_data initialize_DFT(const std::vector<PB_wavefunction> &basis, int Ng, double L, const std::vector<Atom> &atoms, int p, int q, arma::vec &C_vec_hartree, arma::vec &C_vec_ext, arma::mat& C_guess, arma::mat& F_initial) {

    // Obtain the new guesses for the coefficient matrix based on the calculated Fock matrix ... same for alpha and beta
    arma::mat C_new = find_MO_coefs(F_initial);

    iteration_data it_data = {basis,
                              Ng,
                              L,
                              atoms,
                              p,
                              q,
                              C_vec_hartree,
                              C_vec_ext,
                              C_guess,
                              C_guess,
                              F_initial,
                              F_initial,
                              C_new,
                              C_new,
                              0,
                              false};

    return it_data;

}


// Note to self: the differentiator between alpha and beta electrons comes from the exchange/correlation matrix, which is dependent on the densities!
std::vector<iteration_data> converge_DFT(const iteration_data &it_data, std::vector<iteration_data> &it_data_vec, std::function<arma::mat(std::vector<PB_wavefunction>&, arma::mat&, arma::vec&, arma::vec&, const std::vector<Atom>&, int, double, int)>& F_func) {

    // Base case
    if (it_data.converged) {
        return it_data_vec;
    }

    // Extracting constant information for each iteration
    std::vector<PB_wavefunction> basis = it_data.basis;
    int Ng = it_data.Ng;
    double L = it_data.L;
    std::vector<Atom> atoms = it_data.atoms;
    int p = it_data.p;
    int q = it_data.q;
    arma::vec C_vec_hartree = it_data.C_vec_hartree;
    arma::vec C_vec_ext = it_data.C_vec_ext;


    // Old coefficient matrices for this iteration are the new ones from the previous iteration
    arma::mat C_alpha_old = it_data.C_alpha_new;
    arma::mat C_beta_old = it_data.C_beta_new;


    // Make the new fock matrix based on the coefficient matrices from the previous iteration
    arma::mat Fock_alpha = F_func(basis, C_alpha_old, C_vec_hartree, C_vec_ext, atoms, Ng, L, p);
    arma::mat Fock_beta = F_func(basis, C_alpha_old, C_vec_hartree, C_vec_ext, atoms, Ng, L, q);


    // Diagonalize the fock matrices and extra MO coefficients
    arma::mat C_alpha_new = find_MO_coefs(Fock_alpha);
    arma::mat C_beta_new = find_MO_coefs(Fock_beta);


    // Compile all the appropriate info into a new iteration_data object
    iteration_data new_it_data = {basis,
                                  Ng,
                                  L,
                                  atoms,
                                  p,
                                  q,
                                  C_vec_hartree,
                                  C_vec_ext,
                                  C_alpha_old,
                                  C_beta_old,
                                  Fock_alpha,
                                  Fock_beta,
                                  C_alpha_new,
                                  C_beta_new,
                                  it_data.iteration_count+1};

    // Adding that new set of iteration data to the original vector of iteration data
    it_data_vec.push_back(new_it_data);


    return converge_DFT(new_it_data, it_data_vec, F_func);

}


iteration_data obtain_converged_data(std::vector<iteration_data> &converged_it_data) {
    return converged_it_data.back();
}



