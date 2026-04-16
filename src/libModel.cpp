#include "libModel.h"

#include "DenseMatrix.h"
#include "libParticles.h"

#include <ctime>
#include <fstream>
#include <string>
#include <cmath>
#include <iostream>
#include <sstream>
#include <vector>
#include <cstdlib>
#include <iomanip>

// Helper function to run ANN inference via Python script
std::vector<std::vector<double>> runANNInference(double phi, std::size_t nParticles) {
    std::ostringstream jsonInput;
    jsonInput << std::fixed << std::setprecision(6) 
              << "{\"phi\": " << phi << ", \"nParticles\": " << nParticles << "}";
    
    // Run Python inference script
    FILE* pipe = popen(("echo '" + jsonInput.str() + "' | python3 scripts/ann_inference.py").c_str(), "r");
    if (!pipe) {
        std::cerr << "  [ERROR] Failed to run ANN inference script" << std::endl;
        return {};
    }
    
    // Read JSON output
    char buffer[16384];
    std::string output;
    while (fgets(buffer, sizeof(buffer), pipe) != nullptr) {
        output += buffer;
    }
    pclose(pipe);
    
    // Simple JSON parsing for nested array [[...],[...],[...]]
    std::vector<std::vector<double>> result;
    result.resize(3);
    
    int row = 0;
    std::string current_value;
    
    for (size_t i = 0; i < output.length() && row < 3; ++i) {
        char c = output[i];
        
        if (c == '[') {
            current_value.clear();
        } else if (c == ',' || c == ']') {
            if (!current_value.empty()) {
                try {
                    double val = std::stod(current_value);
                    result[row].push_back(val);
                    current_value.clear();
                } catch (...) {
                    current_value.clear();
                }
            }
            if (c == ']' && current_value.empty()) {
                row++;
            }
        } else if (c != ' ' && c != '\n' && c != '\r' && c != '\t') {
            current_value += c;
        }
    }
    
    return result;
}

DenseMatrix createPairwiseModel(const std::vector<Vec3>& positions) {
    const std::size_t particleCount = positions.size();
    const std::size_t matrixSize = 3 * particleCount;

    DenseMatrix pairwiseApproximation = DenseMatrix::identity(matrixSize);
    if (particleCount <= 1) {
        return pairwiseApproximation;
    }

    for (std::size_t i = 0; i < particleCount; ++i) {
        for (std::size_t j = i+ 1; j < particleCount; ++j) {
            DenseMatrix pairMobility(0);
            pairMobility.getMobilityMatrix({positions[i], positions[j]});
            DenseMatrix pairResistance = pairMobility.invert();

            const std::size_t idx0 = 3 * i;
            const std::size_t idxj = 3 * j;

            for (std::size_t row = 0; row < 3; ++row) {
                for (std::size_t col = 0; col < 3; ++col) {
                    const double identityEntry = row == col ? 1.0 : 0.0;

                    pairwiseApproximation(idx0 + row, idx0 + col) +=
                        pairResistance(row, col) - identityEntry;
                    pairwiseApproximation(idx0 + row, idxj + col) +=
                        pairResistance(row, 3 + col);
                    pairwiseApproximation(idxj + row, idx0 + col) +=
                        pairResistance(3 + row, col);
                    pairwiseApproximation(idxj + row, idxj + col) +=
                        pairResistance(3 + row, 3 + col) - identityEntry;
                }
            }
        }
    }

    return pairwiseApproximation;
}

DenseMatrix createPairwiseModelTargetParticleOnly(const std::vector<Vec3>& positions) {
    const std::size_t particleCount = positions.size();
    const std::size_t matrixSize = 3 * particleCount;

    DenseMatrix pairwiseApproximation = DenseMatrix::identity(matrixSize);
    if (particleCount <= 1) {
        return pairwiseApproximation;
    }

    for (std::size_t j = 1; j < particleCount; ++j) {
        DenseMatrix pairMobility(0);
        pairMobility.getMobilityMatrix({positions[0], positions[j]});
        DenseMatrix pairResistance = pairMobility.invert();
        const std::size_t idx0 = 0;
        const std::size_t idxj = 3 * j;
        for (std::size_t row = 0; row < 3; ++row) {
            for (std::size_t col = 0; col < 3; ++col) {
                const double identityEntry = row == col ? 1.0 : 0.0;
                pairwiseApproximation(idx0 + row, idx0 + col) += pairResistance(row, col) - identityEntry;
                pairwiseApproximation(idx0 + row, idxj + col) += pairResistance(row, 3 + col);
            }
        }
    }


    return pairwiseApproximation;
}

void doSpectralNormStudy(std::size_t nParticles, double particleDiameter) {
    
    const std::string spectralNormOutputPath =
        "results/spectralNorm_" + std::to_string(nParticles) + ".csv";
    std::ofstream spectralNormFile(spectralNormOutputPath);
    
    spectralNormFile << "nParticles,r,volumeFraction,spectralNorm\n";
    for (std::size_t i = 0; i <= 6000; i++) {
        double r = 10.0 + static_cast<double>(i) / 10.0;
        const KSpectralNormResult result =
            getKSpectralNorm(nParticles, r, particleDiameter);
        spectralNormFile << nParticles << "," << r << ","
                         << result.volumeFraction << ","
                         << result.spectralNorm << "\n";
    }
}


void doAccuracyStudy(double particleDiameter) {
    std::cout << "\n=== Starting Accuracy Study ===" << std::endl;
    std::cout << "Particle diameter: " << particleDiameter << std::endl;

    const char* resultsOutputPath = "results/results.csv";
    std::ofstream resultsFile(resultsOutputPath);
    if (!resultsFile) {
        std::cerr << "[ERROR] Failed to open " << resultsOutputPath << " for writing."
                  << std::endl;
        return;
    }
    std::cout << "Output file: " << resultsOutputPath << std::endl;

    resultsFile << "nParticles,r,volumeFraction,residualPair,residual1,residual2,residual3,residual4,"
                   "residual5,residualANN\n";

    std::clock_t startTime = std::clock();

    for (std::size_t nParticles = 50; nParticles <= 50; ++nParticles) {
        std::cout << "Processing nParticles = " << nParticles  << std::endl;
        int nConfigs = 100;
        for (std::size_t i = 0; i <= nConfigs; i++) {
            if (i % 20 == 0 && i > 0) {
                              
                std::cout << "  Progress: " << i << "/" << nConfigs << " configs " << "of " << nParticles << std::endl;
            }
            double r = 10.0 + static_cast<double>(i) / 10.0;
            const ModelResult result =
                runModel(nParticles, r, particleDiameter);

            resultsFile << nParticles << "," << r << ","
                        << result.volumeFraction << ","
                        << result.residualPair << "," << result.residual1
                        << "," << result.residual2 << ","
                        << result.residual3 << "," << result.residual4
                        << "," << result.residual5 << "," << result.residualANN << "\n";
        }
        std::cout << "  Completed: 6000/6000 configs processed" << std::endl;
    }

    std::clock_t endTime = std::clock();
    double totalTime = static_cast<double>(endTime - startTime) / CLOCKS_PER_SEC;
    
    std::cout << "\n=== Accuracy Study Complete ===" << std::endl;
    std::cout << "Total time: " << std::fixed << std::setprecision(2) << totalTime << " seconds" << std::endl;
    std::cout << "Results written to " << resultsOutputPath << std::endl;

}


void doSpectralRadiusStudy(std::size_t nParticles, double particleDiameter) {
    
    const std::string spectralRadiusOutputPath =
        "results/spectralRadius_" + std::to_string(nParticles) + ".csv";
    std::ofstream spectralRadiusFile(spectralRadiusOutputPath);
    
    spectralRadiusFile << "nParticles,r,volumeFraction,spectralRadius\n";
    for (std::size_t i = 0; i <= 6000; i++) {
        double r = 10.0 + static_cast<double>(i) / 10.0;
        const KSpectralRadiusResult result =
            getKSpectralRadius(nParticles, r, particleDiameter);
        spectralRadiusFile << nParticles << "," << r << ","
                           << result.volumeFraction << ","
                           << result.spectralRadius << "\n"; 

    }
}

void testModel(std::size_t nParticles, double r, double particleDiameter) {
    Particles particles(nParticles, r, particleDiameter);
    DenseMatrix m = createPairwiseModel(particles.xyz);
    std::cout << m << std::endl;
}

KSpectralRadiusResult getKSpectralRadius(std::size_t nParticles, double r, double particleDiameter) {
    const std::size_t n = 3 * nParticles;
    Particles particles(nParticles, r, particleDiameter);
    DenseMatrix mobilityMatrix(n);
    mobilityMatrix.getMobilityMatrix(particles.xyz);
    DenseMatrix Id = DenseMatrix::identity(n);
    DenseMatrix K = mobilityMatrix.substract(Id);
    const double sr = K.spectralRadius();
    //const double sn = K.spectralNorm();
    //std::cout << "nParticles: " << nParticles << ", r: " << r << ", volume fraction: " << particles.getVolumeFraction() << ", spectral radius: " << sr << ", spectral norm: " << sn << std::endl;
    const double vf = particles.getVolumeFraction();

    return {vf, sr};
}


KSpectralNormResult getKSpectralNorm(std::size_t nParticles, double r, double particleDiameter) {
    const std::size_t n = 3 * nParticles;
    Particles particles(nParticles, r, particleDiameter);
    DenseMatrix mobilityMatrix(n);
    mobilityMatrix.getMobilityMatrix(particles.xyz);
    DenseMatrix Id = DenseMatrix::identity(n);
    DenseMatrix K = mobilityMatrix.substract(Id);
    const double sn = K.spectralNorm();
    const double vf = particles.getVolumeFraction();

    return {vf, sn};
}

ModelResult runModel(std::size_t nParticles, double r, double particleDiameter) {
    const std::size_t n = 3 * nParticles;
    constexpr std::size_t nRuns = 1;

    ModelResult averageResult{0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0};

    for (std::size_t run = 0; run < nRuns; ++run) {
        Particles particles(nParticles, r, particleDiameter);

        DenseMatrix mobilityMatrix(n);
        mobilityMatrix.getMobilityMatrix(particles.xyz);
        DenseMatrix resistanceMatrix = mobilityMatrix.invert();

        DenseMatrix Id = DenseMatrix::identity(n);
        DenseMatrix K =  Id.substract(mobilityMatrix); // mobilityMatrix.substract(Id);
        DenseMatrix K2 = K.multiply(K);
        DenseMatrix K3 = K2.multiply(K);
        DenseMatrix K4 = K3.multiply(K);

        DenseMatrix rmAprox1 = Id;
        DenseMatrix rmAproxPair = createPairwiseModel(particles.xyz);

        // ANN augmented calculation via Python inference
        double phi = particles.getVolumeFraction();
        auto dR = runANNInference(phi, nParticles);
        
        if (dR.empty()) {
            std::cerr << "  [WARNING] ANN inference returned empty result at phi=" << phi 
                      << ", nParticles=" << nParticles << std::endl;
        }
        
        DenseMatrix correction(n);
        for (std::size_t i = 0; i < 3 && i < dR.size(); ++i) {
            for (std::size_t j = 0; j < dR[i].size() && j < static_cast<std::size_t>(n); ++j) {
                correction(i, j) = dR[i][j];
            }
        }

        DenseMatrix rmAproxANN = rmAproxPair.add(correction);
        DenseMatrix rmAprox2 = Id.add(K);
        DenseMatrix rmAprox3 = rmAprox2.add(K2);
        DenseMatrix rmAprox4 = rmAprox3.add(K3);
        DenseMatrix rmAprox5 = rmAprox4.add(K4);

        DenseMatrix residualPair = resistanceMatrix.substract(rmAproxPair);
        DenseMatrix residual1 = resistanceMatrix.substract(rmAprox1);
        DenseMatrix residual2 = resistanceMatrix.substract(rmAprox2);
        DenseMatrix residual3 = resistanceMatrix.substract(rmAprox3);
        DenseMatrix residual4 = resistanceMatrix.substract(rmAprox4);
        DenseMatrix residual5 = resistanceMatrix.substract(rmAprox5);

        averageResult.volumeFraction += particles.getVolumeFraction();
        averageResult.residualPair += residualPair.matrix().norm();
        averageResult.residual1 += residual1.matrix().norm();
        averageResult.residual2 += residual2.matrix().norm();
        averageResult.residual3 += residual3.matrix().norm();
        averageResult.residual4 += residual4.matrix().norm();
        averageResult.residual5 += residual5.matrix().norm();

        DenseMatrix residualANN = resistanceMatrix.substract(rmAproxANN);
        averageResult.residualANN += residualANN.matrix().norm();
    }

    const double normalization = 1.0 / static_cast<double>(nRuns);
    averageResult.volumeFraction *= normalization;
    averageResult.residualPair *= normalization;
    averageResult.residual1 *= normalization;
    averageResult.residual2 *= normalization;
    averageResult.residual3 *= normalization;
    averageResult.residual4 *= normalization;
    averageResult.residual5 *= normalization;
    averageResult.residualANN *= normalization;

    return averageResult;
}

void measureCPUtime(std::size_t nParticles, double r, double particleDiameter) {
    const std::size_t n = 3 * nParticles;
    std::size_t nRuns = 1000;
    Particles particles(nParticles, r, particleDiameter);

    std::clock_t start = std::clock();

    for (std::size_t run = 0; run < nRuns; ++run) {
        DenseMatrix mobilityMatrix(n);
        mobilityMatrix.getMobilityMatrix(particles.xyz);
        DenseMatrix Id = DenseMatrix::identity(n);
        DenseMatrix rmAprox2 = Id;
    }    
    std::clock_t end = std::clock();
    double time1 = static_cast<double>(end - start) / CLOCKS_PER_SEC;

    start = std::clock();
    for (std::size_t run = 0; run < nRuns; ++run) {
        DenseMatrix rmAproxPair = createPairwiseModelTargetParticleOnly(particles.xyz);
    }
    end = std::clock();
    double timePair = static_cast<double>(end - start) / CLOCKS_PER_SEC;

    start = std::clock();
    for (std::size_t run = 0; run < nRuns; ++run) {
        DenseMatrix mobilityMatrix(n);
        mobilityMatrix.getMobilityMatrix(particles.xyz);
        DenseMatrix Id = DenseMatrix::identity(n);
        DenseMatrix K = mobilityMatrix.substract(Id);
        DenseMatrix rmAprox2 = Id.substract(K);
    }
    end = std::clock();
    double time2 = static_cast<double>(end - start) / CLOCKS_PER_SEC;

    start = std::clock();
    for (std::size_t run = 0; run < nRuns; ++run) {
        DenseMatrix mobilityMatrix(n);
        mobilityMatrix.getMobilityMatrix(particles.xyz);
        DenseMatrix Id = DenseMatrix::identity(n);
        DenseMatrix K = mobilityMatrix.substract(Id);
        DenseMatrix K2 = K.multiply(K);
        DenseMatrix rmAprox2 = Id.substract(K);
        DenseMatrix rmAprox3 = rmAprox2.add(K2);

    }
    end = std::clock();
    double time3 = static_cast<double>(end - start) / CLOCKS_PER_SEC;


    start = std::clock();
    for (std::size_t run = 0; run < nRuns; ++run) {
        DenseMatrix mobilityMatrix(n);
        mobilityMatrix.getMobilityMatrix(particles.xyz);
        DenseMatrix Id = DenseMatrix::identity(n);
        DenseMatrix K = mobilityMatrix.substract(Id);
        DenseMatrix K2 = K.multiply(K);
        DenseMatrix K3 = K2.multiply(K);
        DenseMatrix rmAprox2 = Id.substract(K);
        DenseMatrix rmAprox3 = rmAprox2.add(K2);
        DenseMatrix rmAprox4 = rmAprox3.substract(K3);        
    }
    end = std::clock();
    double time4 = static_cast<double>(end - start) / CLOCKS_PER_SEC;    


    start = std::clock();
    for (std::size_t run = 0; run < nRuns; ++run) {
        DenseMatrix mobilityMatrix(n);
        mobilityMatrix.getMobilityMatrix(particles.xyz);
        DenseMatrix Id = DenseMatrix::identity(n);
        DenseMatrix K = mobilityMatrix.substract(Id);
        DenseMatrix K2 = K.multiply(K);
        DenseMatrix K3 = K2.multiply(K);
        DenseMatrix K4 = K3.multiply(K);
        DenseMatrix rmAprox2 = Id.substract(K);
        DenseMatrix rmAprox3 = rmAprox2.add(K2);
        DenseMatrix rmAprox4 = rmAprox3.substract(K3);  
        DenseMatrix rmAprox5 = rmAprox4.add(K4);             
    }
    end = std::clock();
    double time5 = static_cast<double>(end - start) / CLOCKS_PER_SEC;  


    start = std::clock();
    for (std::size_t run = 0; run < nRuns; ++run) {
        DenseMatrix mobilityMatrix(n);
        mobilityMatrix.getMobilityMatrix(particles.xyz);
        DenseMatrix resistanceMatrix = mobilityMatrix.invert();
    }
    end = std::clock();
    double timeMm1 = static_cast<double>(end - start) / CLOCKS_PER_SEC;  

    std::ifstream existingFile("results/cpuTime.csv");
    const bool writeHeader = !existingFile.good() || existingFile.peek() == std::ifstream::traits_type::eof();

    std::ofstream cpuTimeFile("results/cpuTime.csv", std::ios::app);
    cpuTimeFile << std::scientific << std::uppercase;
    if (writeHeader) {
        cpuTimeFile << "nParticles, Stokes, PRM, I-K, I-K+K^2, I-K+K^2-K^3, I-K+K^2-K^3+K^4, M^-1" << std::endl;
    }
    cpuTimeFile << nParticles << ", " << time1 << ", " << nParticles * timePair << ", "
                << nParticles * time2 << ", " << nParticles * time3 << ", " << nParticles * time4 << ", " << nParticles * time5
                << ", " << nParticles * timeMm1 << std::endl;

}
