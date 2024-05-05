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


