//
// Created by Sahand Adibnia on 4/25/24.
//

#include "KE.h"

double T_diagonal_component(PB_wavefunction w) {
    return 0.5 * pow((M_PI * arma::norm(w.n) / w.L), 2);
}


// Need to write code that iterates through nx, ny, nz combos and checks if T for that < Ecutoff! If it is, include it in
// a standard vector denoting the entire basis

std::vector<PB_wavefunction> construct_basis(double E_cutoff, double L) {

    std::vector<PB_wavefunction> final_vec;
    double T=0;

    // Arbitrarily chose 100 are largest possible value to avoid running thru loop too much
    for (double x=1; x < 100; x++) {
        for (double y=1; y < 100; y++) {
            for (double z=1; z < 100; z++) {
                PB_wavefunction w = {x, y, z, L};
                T = T_diagonal_component(w);
                if (T <= E_cutoff) {
                    final_vec.push_back(w);
                }
            }
        }
    }

    return final_vec;
}
