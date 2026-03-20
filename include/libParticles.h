#ifndef LIB_PARTICLES_H
#define LIB_PARTICLES_H

#include "Vec3.h"

#include <cstddef>
#include <vector>

class Particles {
public:
    std::vector<Vec3> xyz;

    Particles(std::size_t nParticles, double radius, double particleDiameter);

    double getVolumeFraction() const;

private:
    double radius_;
    double particleDiameter_;

    void generateRandomParticlePositions(std::size_t nParticles);
    double getDistanceToFurthestParticle() const;
};

#endif
