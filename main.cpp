#include <iostream>
#include <fstream>
#include <vector>
#include <cmath>
#include <functional>
#include <armadillo>
#include "basis.h"
#include "KE.h"



int main() {


    std::vector<PB_wavefunction> basis = construct_basis(3.67493, 9.4486);

    return 0;
}
