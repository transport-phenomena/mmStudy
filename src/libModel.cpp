#include "libModel.h"

#include "DenseMatrix.h"
#include "libParticles.h"

ModelResult runModel(std::size_t nParticles, double r, double particleDiameter) {
    const std::size_t n = 3 * nParticles;
    constexpr std::size_t nRuns = 10;

    ModelResult averageResult{0.0, 0.0, 0.0, 0.0, 0.0};

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

        DenseMatrix rmAprox2 = Id.substract(K);
        DenseMatrix rmAprox3 = rmAprox2.add(K2);
        DenseMatrix rmAprox4 = rmAprox3.substract(K3);
        DenseMatrix rmAprox5 = rmAprox4.add(K4);

        DenseMatrix residual2 = resistanceMatrix.substract(rmAprox2);
        DenseMatrix residual3 = resistanceMatrix.substract(rmAprox3);
        DenseMatrix residual4 = resistanceMatrix.substract(rmAprox4);
        DenseMatrix residual5 = resistanceMatrix.substract(rmAprox5);

        averageResult.volumeFraction += particles.getVolumeFraction();
        averageResult.residual2 += residual2.matrix().norm();
        averageResult.residual3 += residual3.matrix().norm();
        averageResult.residual4 += residual4.matrix().norm();
        averageResult.residual5 += residual5.matrix().norm();
    }

    const double normalization = 1.0 / static_cast<double>(nRuns);
    averageResult.volumeFraction *= normalization;
    averageResult.residual2 *= normalization;
    averageResult.residual3 *= normalization;
    averageResult.residual4 *= normalization;
    averageResult.residual5 *= normalization;

    return averageResult;
}
