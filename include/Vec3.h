
#pragma once

#include <iostream>
#include <cmath>

class Vec3 {

public:

        double x, y, z;

        Vec3(double x, double y, double z);  // constructor
        Vec3(); // default constructor

        double norm() const { return std::sqrt(x*x + y*y + z*z); }
        double dot(const Vec3& other) const {
            return x*other.x + y*other.y + z*other.z;
        }

        double operator*(const Vec3& other) const { return dot(other); }
        Vec3 operator-(const Vec3& other) const { return Vec3(x - other.x, y - other.y, z - other.z); }
        Vec3 operator+(const Vec3& other) const { return Vec3(x + other.x, y + other.y, z + other.z); }
        Vec3 operator*(double s) const { return Vec3(s * x, s * y, s * z); }
        friend Vec3 operator*(double s, const Vec3& v) { return Vec3(s * v.x, s * v.y, s * v.z); }

        friend std::ostream& operator<<(std::ostream& ioOut, const Vec3& obj);
};
