//
// Created by Sahand Adibnia on 4/29/24.
//

#include "Numerical Integration.h"

arma::vec grid_int_vec(const std::function<double(double,double,double)> &f, int Ng, double L) {

    double width = L / Ng;
    arma::vec final_vec(Ng, arma::fill::zeros);

    int vector_index=0;
    for (int i=0; i < Ng; i++) {
        for (int j=0; j < Ng; j++) {
            for (int k=0; k < Ng; k++) {
                final_vec(vector_index) = f(width*i - L/2.0, width*j - L/2.0, width*k - L/2.0);
                vector_index += 1;
            }
        }
    }

}


// Evaluate entire basis at gridpoints
arma::vec basis_at_gridpoints(const std::vector<PB_wavefunction> &basis, int Ng, double L) {

    // Create a function that you can plug in to numerical integration code from before!
    auto f = [&](double x, double y, double z) {

    };

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