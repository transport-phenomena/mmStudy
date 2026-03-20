#ifndef DENSE_MATRIX_H
#define DENSE_MATRIX_H

#include <Eigen/Dense>
#include <cstddef>
#include <iosfwd>
#include <vector>

#include "Vec3.h"

class DenseMatrix {
public:
    using MatrixType = Eigen::MatrixXd;

    explicit DenseMatrix(std::size_t size);
    explicit DenseMatrix(const MatrixType& matrix);

    static DenseMatrix identity(std::size_t size);

    void fillTestData();
    void getMobilityMatrix(const std::vector<Vec3>& particles);
    DenseMatrix invert() const;
    DenseMatrix add(const DenseMatrix& other) const;
    DenseMatrix subtract(const DenseMatrix& other) const;
    DenseMatrix substract(const DenseMatrix& other) const;
    DenseMatrix multiply(const DenseMatrix& other) const;
    std::size_t size() const;
    double operator()(std::size_t row, std::size_t col) const;
    double& operator()(std::size_t row, std::size_t col);
    const MatrixType& matrix() const;

    friend std::ostream& operator<<(std::ostream& os,
                                    const DenseMatrix& denseMatrix);

private:
    MatrixType matrix_;
};

#endif
