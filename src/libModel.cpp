#include "libModel.h"

#include "DenseMatrix.h"
#include "libParticles.h"

#include <ctime>
#include <fstream>

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


void testModel(std::size_t nParticles, double r, double particleDiameter) {
    Particles particles(nParticles, r, particleDiameter);
    DenseMatrix m = createPairwiseModel(particles.xyz);
    std::cout << m << std::endl;
}


ModelResult runModel(std::size_t nParticles, double r, double particleDiameter) {
    const std::size_t n = 3 * nParticles;
    constexpr std::size_t nRuns = 100;

    ModelResult averageResult{0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0};

    for (std::size_t run = 0; run < nRuns; ++run) {
        Particles particles(nParticles, r, particleDiameter);

        DenseMatrix mobilityMatrix(n);
        mobilityMatrix.getMobilityMatrix(particles.xyz);
        DenseMatrix resistanceMatrix = mobilityMatrix.invert();

        DenseMatrix Id = DenseMatrix::identity(n);
        DenseMatrix K = mobilityMatrix.substract(Id);
        DenseMatrix K2 = K.multiply(K);
        DenseMatrix K3 = K2.multiply(K);
        DenseMatrix K4 = K3.multiply(K);

        DenseMatrix rmAprox1 = Id;
        DenseMatrix rmAproxPair = createPairwiseModel(particles.xyz);
        DenseMatrix rmAprox2 = Id.substract(K);
        DenseMatrix rmAprox3 = rmAprox2.add(K2);
        DenseMatrix rmAprox4 = rmAprox3.substract(K3);
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
    }

    const double normalization = 1.0 / static_cast<double>(nRuns);
    averageResult.volumeFraction *= normalization;
    averageResult.residualPair *= normalization;
    averageResult.residual1 *= normalization;
    averageResult.residual2 *= normalization;
    averageResult.residual3 *= normalization;
    averageResult.residual4 *= normalization;
    averageResult.residual5 *= normalization;

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
