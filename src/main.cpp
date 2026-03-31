#include "libModel.h"
#include <fstream>
#include <iostream>

int main() {
    constexpr double particleDiameter = 1.0;

   // for (std::size_t nParticles = 2; nParticles <= 100; ++nParticles) {
   //     measureCPUtime(nParticles,1000.0,particleDiameter);
   // }

    
    for (std::size_t nParticles = 2; nParticles <= 100; ++nParticles) {
        doSpectralNormStudy(nParticles, particleDiameter);
    }

    const char* resultsOutputPath = "results/results.csv";
    std::ofstream resultsFile(resultsOutputPath);
    if (!resultsFile) {
        std::cerr << "Failed to open " << resultsOutputPath << " for writing."
                  << std::endl;
        return 1;
    }

    resultsFile << "nParticles,r,volumeFraction,residualPair,residual1,residual2,residual3,residual4,"
                   "residual5\n";

    for (std::size_t nParticles = 50; nParticles <= 50; ++nParticles) {
        for (std::size_t i = 0; i <= 6000; i++) {
            double r = 10.0 + static_cast<double>(i) / 10.0;
            const ModelResult result =
                runModel(nParticles, r, particleDiameter);

            resultsFile << nParticles << "," << r << ","
                        << result.volumeFraction << ","
                        << result.residualPair << "," << result.residual1
                        << "," << result.residual2 << ","
                        << result.residual3 << "," << result.residual4
                        << "," << result.residual5 << "\n";
        }
    }

    std::cout << "Wrote results to " << resultsOutputPath << std::endl;

    return 0;
}
