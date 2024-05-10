//
// Created by Sahand Adibnia on 4/29/24.
//

#include "Numerical Integration.h"

arma::mat build_grid(double L, int Ng) {
    double width = L/Ng;
    double corner_1D = -L/2;

    arma::mat final_mat(3,pow(Ng,3),arma::fill::zeros); // 1 column per grid point, 1 row per dimension (x,y,z)

    // Build grid
    int grid_point_index = 0;
    for (int i = 0; i < Ng; i++) {
        for (int j = 0; j < Ng; j++) {
            for (int k = 0; k < Ng; k++) {
                double rg_x = corner_1D + width/2 + width*(i); // +width/2 to make gridpoints in center of it all
                double rg_y = corner_1D + width/2 + width*(j);
                double rg_z = corner_1D + width/2 + width*(k);

                final_mat(0,grid_point_index) = rg_x;
                final_mat(1,grid_point_index) = rg_y;
                final_mat(2,grid_point_index) = rg_z;

                grid_point_index += 1;
            }
        }
    }
    return final_mat;
}

// store values of a basis function at grid points
arma::vec basisf_at_gridpoints(arma::mat &grid, PB_wavefunction &basis_func, int Ng) {

    arma::vec final_vec(pow(Ng,3), arma::fill::zeros);

    int grid_point_index=0;
    for (int i=0; i < Ng; i++) {
        for (int j=0; j < Ng; j++) {
            for (int k=0; k < Ng; k++) {
                arma::vec grid_point = grid.col(grid_point_index);
                final_vec(grid_point_index) = PB_func(grid_point(0), grid_point(1), grid_point(2), basis_func);
                grid_point_index += 1;
            }
        }
    }

    return final_vec;

}


// store values of a custom function at grid points
arma::vec customf_at_gridpoints(arma::mat &grid, const std::function<double(double,double,double)> &f, int Ng) {

    arma::vec final_vec(pow(Ng,3), arma::fill::zeros);

    int grid_point_index=0;
    for (int i=0; i < Ng; i++) {
        for (int j=0; j < Ng; j++) {
            for (int k=0; k < Ng; k++) {
                arma::vec grid_point = grid.col(grid_point_index);
                final_vec(grid_point_index) = f(grid_point(0), grid_point(1), grid_point(2));
                grid_point_index += 1;
            }
        }
    }

    return final_vec;

}


// Evaluate entire basis at gridpoints - creates a matrix of N_G x basis_size, each column is a basis function and each row is a grid point
arma::mat basis_grid(arma::mat &grid, std::vector<PB_wavefunction> &basis, int Ng) {

    arma::mat final_mat(pow(Ng, 3), basis.size(), arma::fill::zeros); // 1 column per basis function, 1 row per grid point

    int basis_index=0;
    for (auto& basis_func : basis) {
        arma::vec evaled_basis_func = basisf_at_gridpoints(grid, basis_func, Ng);
        final_mat.col(basis_index) = evaled_basis_func;
        basis_index += 1;
    }


    return final_mat;

}





// Evaluate a potential energy function at gridpoints


double quad_3D_grid_integration(const std::function<double(double,double,double)> &f, int Ng, double L) {

    double width = L / Ng;
    double weight = pow(width, 3);

    double sum = 0;
    int count=0;
    for (int i=0; i < Ng; i++) {
        for (int j=0; j < Ng; j++) {
            for (int k=0; k < Ng; k++) {
                sum += f(width*i - L/2.0, width*j - L/2.0, width*k - L/2.0); // center of box at (0,0,0)
//                std::cout << sum << std::endl;
//                count += 1;
//                std::cout << count << std::endl;
            }
        }
    }

    return weight * sum;
}