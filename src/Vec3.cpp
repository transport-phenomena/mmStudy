#include "Vec3.h"

Vec3::Vec3(double xVal, double yVal, double zVal) { x = xVal; y = yVal; z = zVal; }  // constructor
Vec3::Vec3() { x = 0.0; y = 0.0; z = 0.0; } // default constructor

std::ostream& operator<<(std::ostream& ioOut, const Vec3& obj) {
    ioOut << "(" << obj.x << ", " << obj.y << ", " << obj.z << ")";
    return ioOut;
}