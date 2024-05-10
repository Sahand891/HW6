//
// Created by Sahand Adibnia on 5/4/24.
//

#include "Energy.h"


arma::mat P(arma::mat &C) {
    return C*C.t();
}

double kinetic_energy(const iteration_data &final_it_data, arma::sp_mat &T) {

    arma::mat C_alpha = final_it_data.C_alpha_new;
    arma::mat C_beta = final_it_data.C_beta_new;

    arma::mat P_mat = P(C_alpha) + P(C_beta);

    return arma::trace(P_mat*T);

}

double external_energy(const iteration_data &final_it_data) {

    std::vector<PB_wavefunction> basis = final_it_data.basis;
    int Ng = final_it_data.Ng;
    double L = final_it_data.L;
    std::vector<Atom> atoms = final_it_data.atoms;
    int p = final_it_data.p;
    arma::vec C_vec_ext = final_it_data.C_vec_ext;

    arma::mat C_alpha = final_it_data.C_alpha_new;
    arma::mat C_beta = final_it_data.C_beta_new;


    arma::mat P_mat = P(C_alpha) + P(C_beta);

    // Note that V_ext doesn't *actually* use C! or even num_electrons, for that matter!
    arma::mat V_ext_mat = construct_V_mat_slow(V_ext, C_alpha, C_vec_ext, basis, atoms, Ng, L, p);

    return arma::trace(P_mat*V_ext_mat);

}


double hartree_energy(const iteration_data &final_it_data) {

    std::vector<PB_wavefunction> basis = final_it_data.basis;
    int Ng = final_it_data.Ng;
    double L = final_it_data.L;
    std::vector<Atom> atoms = final_it_data.atoms;
    int p = final_it_data.p;
    int q = final_it_data.q;
    arma::vec C_vec_hartree = final_it_data.C_vec_hartree;

    arma::mat C_alpha = final_it_data.C_alpha_new;
    arma::mat C_beta = final_it_data.C_beta_new;


    arma::mat P_mat = P(C_alpha) + P(C_beta);

    // Note that V_hartree DOES *actually* use C AND num_electrons!!!
    arma::mat V_hartree_mat_alpha = construct_V_mat_slow(V_hartree, C_alpha, C_vec_hartree, basis, atoms, Ng, L, p);
    arma::mat V_hartree_mat_beta = construct_V_mat_slow(V_hartree, C_beta, C_vec_hartree, basis, atoms, Ng, L, q);

    return 0.5*arma::trace(P_mat*(V_hartree_mat_alpha+V_hartree_mat_beta));

}

double xc_energy(const iteration_data &final_it_data) {

    std::vector<PB_wavefunction> basis = final_it_data.basis;
    int Ng = final_it_data.Ng;
    double L = final_it_data.L;
    std::vector<Atom> atoms = final_it_data.atoms;
    int p = final_it_data.p;
    int q = final_it_data.q;
    arma::vec C_vec_hartree = final_it_data.C_vec_hartree;
    arma::vec C_vec_ext = final_it_data.C_vec_ext;

    arma::mat C_alpha = final_it_data.C_alpha_new;
    arma::mat C_beta = final_it_data.C_beta_new;

    auto f = [&](double x, double y, double z) {
        double density_alpha = density(x,y,z,C_alpha,basis,p);
        double density_beta = density(x,y,z,C_beta,basis,q);
        return pow(density_alpha, 4/3.0) + pow(density_beta, 4/3.0);
    };

    double integral = quad_3D_grid_integration(f, Ng, L);
    double coefficient = -0.75 * pow((3 / M_PI), 1/3.0);

    return coefficient*integral;

}


double total_energy(const iteration_data &final_it_data, arma::sp_mat &T) {
    return kinetic_energy(final_it_data,T) + external_energy(final_it_data) + hartree_energy(final_it_data) + xc_energy(final_it_data);
}
