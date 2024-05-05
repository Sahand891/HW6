//
// Created by Sahand Adibnia on 4/17/24.
//

#include "basis.h"

double PB_func(double x, double y, double z, const PB_wavefunction &w) {

    int n_x = w.n_x;
    int n_y = w.n_y;
    int n_z = w.n_z;
    double L = w.L;

    double normalization = pow((2 / L), 3/2.0);
    double x_val = sin(n_x*M_PI*x/L + n_x*M_PI/2);
    double y_val = sin(n_y*M_PI*y/L + n_y*M_PI/2);
    double z_val = sin(n_z*M_PI*z/L + n_z*M_PI/2);

    return normalization*x_val*y_val*z_val;
}