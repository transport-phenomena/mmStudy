#include "DenseMatrix.h"

#include <iomanip>
#include <ios>
#include <sstream>
#include <stdexcept>

namespace {
Eigen::Matrix3d oseenTensor(const Vec3& r) {
    const double rNorm = r.norm();
    if (rNorm < 1e-12) {
        return Eigen::Matrix3d::Zero();
    }

    const Eigen::Vector3d rVector(r.x, r.y, r.z);
    const Eigen::Matrix3d identity = Eigen::Matrix3d::Identity();
    const Eigen::Matrix3d rHatOuter =
        (rVector * rVector.transpose()) / (rNorm * rNorm);

    return (3.0 / (8.0 * rNorm)) * (identity + rHatOuter);
}
} // namespace

DenseMatrix::DenseMatrix(std::size_t size)
    : matrix_(MatrixType::Zero(static_cast<Eigen::Index>(size),
                               static_cast<Eigen::Index>(size))) {}

DenseMatrix::DenseMatrix(const MatrixType& matrix) : matrix_(matrix) {}

DenseMatrix DenseMatrix::identity(std::size_t size) {
    return DenseMatrix(MatrixType::Identity(static_cast<Eigen::Index>(size),
                                            static_cast<Eigen::Index>(size)));
}

void DenseMatrix::fillTestData() {
    const Eigen::Index dimension = matrix_.rows();
    matrix_ = MatrixType::Random(dimension, dimension);
    matrix_ += static_cast<double>(dimension) *
               MatrixType::Identity(dimension, dimension);
}

void DenseMatrix::getMobilityMatrix(const std::vector<Vec3>& particles) {
    const Eigen::Index particleCount =
        static_cast<Eigen::Index>(particles.size());
    const Eigen::Index mobilitySize = 3 * particleCount;

    matrix_ = MatrixType::Zero(mobilitySize, mobilitySize);

    for (Eigen::Index i = 0; i < particleCount; ++i) {
        for (Eigen::Index j = 0; j < particleCount; ++j) {
            const Eigen::Index row = 3 * i;
            const Eigen::Index col = 3 * j;

            if (i == j) {
                matrix_.block<3, 3>(row, col) = Eigen::Matrix3d::Identity();
            } else {
                const Vec3 rij = particles[static_cast<std::size_t>(i)] -
                                 particles[static_cast<std::size_t>(j)];
                matrix_.block<3, 3>(row, col) = oseenTensor(rij);
            }
        }
    }
}

DenseMatrix DenseMatrix::invert() const {
    return DenseMatrix(matrix_.inverse());
}

DenseMatrix DenseMatrix::add(const DenseMatrix& other) const {
    if (matrix_.rows() != other.matrix_.rows()) {
        throw std::invalid_argument("Cannot add matrices of different sizes.");
    }

    return DenseMatrix(matrix_ + other.matrix_);
}

DenseMatrix DenseMatrix::subtract(const DenseMatrix& other) const {
    if (matrix_.rows() != other.matrix_.rows()) {
        throw std::invalid_argument(
            "Cannot subtract matrices of different sizes.");
    }

    return DenseMatrix(matrix_ - other.matrix_);
}

DenseMatrix DenseMatrix::substract(const DenseMatrix& other) const {
    return subtract(other);
}

DenseMatrix DenseMatrix::multiply(const DenseMatrix& other) const {
    if (matrix_.cols() != other.matrix_.rows()) {
        throw std::invalid_argument(
            "Cannot multiply matrices with incompatible dimensions.");
    }

    return DenseMatrix(matrix_ * other.matrix_);
}

std::size_t DenseMatrix::size() const {
    return static_cast<std::size_t>(matrix_.rows());
}

double DenseMatrix::operator()(std::size_t row, std::size_t col) const {
    return matrix_(static_cast<Eigen::Index>(row),
                   static_cast<Eigen::Index>(col));
}

double& DenseMatrix::operator()(std::size_t row, std::size_t col) {
    return matrix_(static_cast<Eigen::Index>(row),
                   static_cast<Eigen::Index>(col));
}

const DenseMatrix::MatrixType& DenseMatrix::matrix() const {
    return matrix_;
}

std::ostream& operator<<(std::ostream& os, const DenseMatrix& denseMatrix) {
    std::vector<std::vector<std::string>> formattedRows(
        static_cast<std::size_t>(denseMatrix.matrix_.rows()));
    std::size_t columnWidth = 0;

    for (Eigen::Index i = 0; i < denseMatrix.matrix_.rows(); ++i) {
        formattedRows[static_cast<std::size_t>(i)].reserve(
            static_cast<std::size_t>(denseMatrix.matrix_.cols()));

        for (Eigen::Index j = 0; j < denseMatrix.matrix_.cols(); ++j) {
            std::ostringstream valueStream;
            valueStream << std::fixed << std::setprecision(5)
                        << denseMatrix.matrix_(i, j);
            formattedRows[static_cast<std::size_t>(i)].push_back(
                valueStream.str());
            columnWidth = std::max(
                columnWidth,
                formattedRows[static_cast<std::size_t>(i)].back().size());
        }
    }

    const std::ios::fmtflags originalFlags = os.flags();
    const std::streamsize originalPrecision = os.precision();

    for (std::size_t i = 0; i < formattedRows.size(); ++i) {
        for (std::size_t j = 0; j < formattedRows[i].size(); ++j) {
            os << std::setw(static_cast<int>(columnWidth)) << formattedRows[i][j];
            if (j + 1 < formattedRows[i].size()) {
                os << ' ';
            }
        }
        if (i + 1 < formattedRows.size()) {
            os << '\n';
        }
    }

    os.flags(originalFlags);
    os.precision(originalPrecision);
    return os;
}
