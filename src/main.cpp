#include "libModel.h"
#include <fstream>
#include <iostream>

int main() {
    constexpr double particleDiameter = 1.0;

   // for (std::size_t nParticles = 2; nParticles <= 100; ++nParticles) {
   //     measureCPUtime(nParticles,1000.0,particleDiameter);
   // }

    
    //for (std::size_t nParticles = 2; nParticles <= 100; ++nParticles) {
    //    doSpectralNormStudy(nParticles, particleDiameter);
    //}

//    for (std::size_t nParticles = 15; nParticles <= 15; ++nParticles) {
//        doSpectralRadiusStudy(nParticles, particleDiameter);
//    }



    doAccuracyStudy(particleDiameter);

    return 0;
}
