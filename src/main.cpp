#include "libModel.h"
#include <fstream>
#include <iostream>

int main() {
    constexpr double particleDiameter = 1.0;
    const char* outputPath = "postprocess/results.csv";
    std::ofstream outputFile(outputPath);

    if (!outputFile) {
        std::cerr << "Failed to open " << outputPath << " for writing."
                  << std::endl;
        return 1;
    }

    outputFile << "nParticles,r,volumeFraction,residual2,residual3,residual4,"
                  "residual5\n";

    for (std::size_t nParticles = 2; nParticles <= 100; ++nParticles) {
        for (int r = 10; r <= 1000; r += 10) {
            const ModelResult result =
                runModel(nParticles, static_cast<double>(r), particleDiameter);

            outputFile << nParticles << "," << r << "," << result.volumeFraction
                       << "," << result.residual2 << "," << result.residual3
                       << "," << result.residual4 << "," << result.residual5
                       << "\n";
        }
    }

    std::cout << "Wrote results to " << outputPath << std::endl;

    return 0;
}
