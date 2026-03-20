#include "libParticles.h"

#include <cmath>
#include <random>

namespace {
constexpr double kPi = 3.14159265358979323846;
}

Particles::Particles(std::size_t nParticles, double radius,
                     double particleDiameter)
    : radius_(radius), particleDiameter_(particleDiameter) {
    generateRandomParticlePositions(nParticles);
}

void Particles::generateRandomParticlePositions(std::size_t nParticles) {
    xyz.clear();
    xyz.reserve(nParticles);

    if (nParticles == 0) {
        return;
    }

    xyz.emplace_back(0.0, 0.0, 0.0);

    std::random_device randomDevice;
    std::mt19937 generator(randomDevice());
    std::uniform_real_distribution<double> unitDistribution(0.0, 1.0);

    for (std::size_t i = 1; i < nParticles; ++i) {
        const double u = unitDistribution(generator);
        const double v = unitDistribution(generator);
        const double w = unitDistribution(generator);

        const double theta = 2.0 * kPi * u;
        const double cosPhi = 1.0 - 2.0 * v;
        const double sinPhi = std::sqrt(1.0 - cosPhi * cosPhi);
        const double radius = radius_ * std::cbrt(w);

        const double x = radius * sinPhi * std::cos(theta);
        const double y = radius * sinPhi * std::sin(theta);
        const double z = radius * cosPhi;

        xyz.emplace_back(x, y, z);
    }

}

double Particles::getDistanceToFurthestParticle() const {
    double maxDistance = 0.0;

    for (const Vec3& particle : xyz) {
        maxDistance = std::max(maxDistance, particle.norm());
    }

    return maxDistance;
}

double Particles::getVolumeFraction() const {
    if (xyz.empty()) {
        return 0.0;
    }

    const double distanceToFurthestParticle = getDistanceToFurthestParticle();
    if (distanceToFurthestParticle < 1e-12) {
        return 0.0;
    }

    const double nParticles = static_cast<double>(xyz.size());
    const double particleRadius = 0.5 * particleDiameter_;
    const double particleVolume = (4.0 * kPi / 3.0) * std::pow(particleRadius, 3);
    return (3.0 * nParticles * particleVolume) / (4.0 * kPi * std::pow(distanceToFurthestParticle, 3));

}
