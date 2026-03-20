#ifndef LIB_MODEL_H
#define LIB_MODEL_H

#include <cstddef>

struct ModelResult {
    double volumeFraction;
    double residual2;
    double residual3;
    double residual4;
    double residual5;
};

ModelResult runModel(std::size_t nParticles, double r, double particleDiameter);

#endif
