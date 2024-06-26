//
// Created by Sahand Adibnia on 4/26/24.
//

#include "Fock Matrix.h"

// Density based on JUST ONE C matrix (alpha or beta)
double density(double x, double y, double z, arma::mat &C, const std::vector<PB_wavefunction> &basis, int num_electrons) {

    double sum=0;

    for (int i=0; i < num_electrons; i++) { // each column is an "MO" in C matrix
        for (int u=0; u < basis.size(); u++) { // each row is an "AO" = basis function in C matrix
            double wavefunc_val = PB_func(x,y,z,basis[u]);
            sum += pow(C(u,i)*wavefunc_val,2);
        }
    }

    return sum;
}

// Density based on BOTH alpha and beta C matrices
double total_density(double x, double y, double z, arma::mat &C_alpha, arma::mat &C_beta, const std::vector<PB_wavefunction> &basis, int p, int q) {
    return density(x,y,z,C_alpha,basis,p) + density(x,y,z,C_beta,basis,q);
}



double V_xc(double x, double y, double z, arma::mat &C, arma::vec &C_vec, std::vector<PB_wavefunction> &basis, const std::vector<Atom> &atoms, int Ng, double L, int num_electrons) {
    return -pow((3 / M_PI)*density(x,y,z,C,basis,num_electrons), 1/3.0);
}



// A function to get the coefficients for each basis function for the external potential
arma::vec C_V_ext(const std::vector<PB_wavefunction> &basis, const std::vector<Atom> &atoms, double L) {

    arma::vec C_vec(basis.size(), arma::fill::zeros);


    for (int i=0; i < basis.size(); i++) {
        PB_wavefunction w = basis[i];
        // Now sum over all atoms
        double sum=0;
        for (auto& atom : atoms) {
            sum += atom.Z_ * PB_func(atom.pos_vec[0], atom.pos_vec[1], atom.pos_vec[2], w);
        }
        C_vec[i] = -4 * L*L * sum / (M_PI * w.n_sq);
    }


    return C_vec;

}


double V_ext(double x, double y, double z, arma::mat &C, arma::vec &C_vec, std::vector<PB_wavefunction> &basis, const std::vector<Atom> &atoms, int Ng, double L, int num_electrons) {

    // First make a armadillo vector of values of the wavefunctions at the specified coordinate
    arma::vec w_vals_vec(basis.size(), arma::fill::zeros);
    for (int i=0; i < basis.size(); i++) {
        PB_wavefunction w = basis[i];
        w_vals_vec[i] = PB_func(x,y,z,w);
    }

    return arma::accu(C_vec % w_vals_vec); // % for element-wise multiplication rather than matrix multiplication
}



// A function to get the coefficients for each basis function for the Hartree potential
// Currently this function is very slow, potentially because it's recalculating at every grid point?
arma::vec C_V_hartree(const std::vector<PB_wavefunction> &basis, arma::mat &C_alpha, arma::mat &C_beta, int Ng, double L, int p, int q, arma::mat &grid, arma::mat &whole_basis_grid) {

    arma::vec C_vec(basis.size(), arma::fill::zeros);

    auto f = [&](double x, double y, double z) {
        return total_density(x,y,z,C_alpha,C_beta,basis,p,q);
    };

    // Evaluate density at gridpoints
    arma::vec density_at_grid = customf_at_gridpoints(grid, f, Ng);

    // Go through every basis function
    for (int basis_func_index=0; basis_func_index < basis.size(); basis_func_index++) { // going thru every basis function
        PB_wavefunction w = basis[basis_func_index];
        arma::vec bf_vec = whole_basis_grid.col(basis_func_index);
        arma::vec prod_at_grid = bf_vec % density_at_grid;
        double numerical_int = arma::accu(prod_at_grid) * pow((L / Ng), 3);
        C_vec(basis_func_index) = 4 * L*L * numerical_int / (M_PI * w.n_sq);
    }

    return C_vec;

}


double V_hartree(double x, double y, double z, arma::mat &C, arma::vec &C_vec, std::vector<PB_wavefunction> &basis, const std::vector<Atom> &atoms, int Ng, double L, int num_electrons) {
    // First make a armadillo vector of values of the wavefunctions at the specified coordinate
    arma::vec w_vals_vec(basis.size(), arma::fill::zeros);
    for (int i=0; i < basis.size(); i++) {
        PB_wavefunction w = basis[i];
        w_vals_vec[i] = PB_func(x,y,z,w);
    }

    return arma::accu(C_vec % w_vals_vec);
}



// General function to construct any potential energy matrix for any PE function that depends on a coef matrix and a basis
arma::mat construct_V_mat_slow(std::function<double(double, double, double, arma::mat&, arma::vec&, std::vector<PB_wavefunction>&, const std::vector<Atom>&, int, double, int)> V, arma::mat &C, arma::vec &C_vec, std::vector<PB_wavefunction> &basis, const std::vector<Atom> &atoms, int Ng, double L, int num_electrons) {
    int matrix_size = basis.size();

    // Initialize a matrix of the appropriate size, just filled with zeroes
    arma::mat V_mat(matrix_size, matrix_size, arma::fill::zeros);

    // Iterate through the matrix elements, computing the appropriate numerical integral
    for (int u = 0; u < matrix_size; u++) {
        PB_wavefunction w_u = basis[u];
        for (int v = 0; v < matrix_size; v++) {
            // compute <w,u|V|w,v> using numerical integration, add it to the matrix
            PB_wavefunction w_v = basis[v];

            // Create a function that you can plug in to numerical integration code from before!
            auto f = [&](double x, double y, double z) {
                double term1 = PB_func(x,y,z,w_u);
                double term2 = V(x,y,z,C,C_vec,basis,atoms,Ng,L,num_electrons);
                double term3 = PB_func(x,y,z,w_v);
                return term1*term2*term3;
            };

            // Compute the matrix element with numerical integration
            V_mat(u,v) = quad_3D_grid_integration(f, Ng, L);
        }
    }

    return V_mat;
}




arma::mat construct_V_mat_fast(std::function<double(double, double, double, arma::mat&, arma::vec&, std::vector<PB_wavefunction>&, const std::vector<Atom>&, int, double, int)> V, arma::mat &C, arma::vec &C_vec, std::vector<PB_wavefunction> &basis, const std::vector<Atom> &atoms, int Ng, double L, int num_electrons, arma::mat &grid, arma::mat &whole_basis_grid) {

    // Initialize a matrix of the appropriate size, just filled with zeroes
    arma::mat V_mat(basis.size(), basis.size(), arma::fill::zeros);


    auto f = [&](double x, double y, double z) {
        return V(x,y,z,C,C_vec,basis,atoms,Ng,L,num_electrons);
    };

    // Evaluate potential energy function at gridpoints
    arma::vec V_at_grid = customf_at_gridpoints(grid, f, Ng);

    // 5th step: Now take every combination of columns in the whole_basis_at_grid matrix, "inner-product" them element-wise by the V_ext_at_grid vector
    for (int i=0; i < whole_basis_grid.n_cols; i++) {
        for (int j=0; j < whole_basis_grid.n_cols; j++) {
            arma::vec bf_vec_1 = whole_basis_grid.col(i);
            arma::vec bf_vec_2 = whole_basis_grid.col(j);
            arma::vec inner_prod_at_grid = bf_vec_1 % V_at_grid % bf_vec_2;
            V_mat(i,j) = arma::accu(inner_prod_at_grid) * pow((L / Ng), 3);
        }
    }

    return V_mat;

}








arma::mat construct_Fock_matrix(std::vector<PB_wavefunction> &basis, arma::mat &C, arma::vec &C_vec_hartree, arma::vec &C_vec_ext, const std::vector<Atom> &atoms, int Ng, double L, int num_electrons, arma::mat &grid, arma::mat &whole_basis_grid) {

    arma::sp_mat T_mat = construct_T(basis);
    arma::mat V_ext_mat = construct_V_mat_fast(V_ext, C, C_vec_ext, basis, atoms, Ng, L, num_electrons, grid, whole_basis_grid);
    arma::mat V_hartree_mat = construct_V_mat_fast(V_hartree, C, C_vec_hartree, basis, atoms, Ng, L, num_electrons, grid, whole_basis_grid);
    arma::mat V_xc_mat = construct_V_mat_fast(V_xc, C, C_vec_hartree, basis, atoms, Ng, L, num_electrons, grid, whole_basis_grid);

    return T_mat + V_ext_mat + V_hartree_mat + V_xc_mat;

}



// Other, simpler ways to make fock matrix
arma::mat construct_Fmat_Vext_only(std::vector<PB_wavefunction> &basis, arma::mat &C, arma::vec &C_vec_hartree, arma::vec &C_vec_ext, const std::vector<Atom> &atoms, int Ng, double L, int num_electrons, arma::mat &grid, arma::mat &whole_basis_grid) {

    arma::sp_mat T_mat = construct_T(basis);
    arma::mat V_ext_mat = construct_V_mat_fast(V_ext, C, C_vec_ext, basis, atoms, Ng, L, num_electrons, grid, whole_basis_grid);

    return T_mat + V_ext_mat;

}