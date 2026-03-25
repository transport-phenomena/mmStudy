#ifndef LIB_MODEL_H
#define LIB_MODEL_H

#include <cstddef>
#include <vector>

#include "DenseMatrix.h"
#include "Vec3.h"

struct ModelResult {
    double volumeFraction;
    double residualPair;
    double residual1;
    double residual2;
    double residual3;
    double residual4;
    double residual5;
};

ModelResult runModel(std::size_t nParticles, double r, double particleDiameter);
void measureCPUtime(std::size_t nParticles, double r, double particleDiameter);
DenseMatrix createPairwiseModel(const std::vector<Vec3>& positions);
void testModel(std::size_t nParticles, double r, double particleDiameter);

#endif
