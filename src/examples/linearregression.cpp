#include <iostream>
#include "../../include/cppoptlib/meta.h"
#include "../../include/cppoptlib/problem.h"
#include "../../include/cppoptlib/solver/bfgssolver.h"

// we define a new problem for optimizing the rosenbrock function
// we use a templated-class rather than "auto"-lambda function for a clean architecture
template<typename T>
class LinearRegression : public cppoptlib::Problem<T> {
  public:
    using typename cppoptlib::Problem<T>::VectorType;
    using typename cppoptlib::Problem<T>::MatrixType;

  protected:
    const MatrixType X;
    const VectorType y;
    const MatrixType XX;

  public:
    LinearRegression(const MatrixType &X_, const VectorType &y_) : X(X_), y(y_), XX(X_.transpose()*X_) {}

    T value(const VectorType &beta) {
        return 0.5*(X*beta-y).squaredNorm();
    }

    void gradient(const VectorType &beta, VectorType &grad) {
        grad = XX*beta - X.transpose()*y;
    }
};

int main(int argc, char const *argv[]) {
    typedef LinearRegression<double> TLinearRegression;
    typedef typename TLinearRegression::VectorType VectorType;
    typedef typename TLinearRegression::MatrixType MatrixType;

    // create true model
    VectorType true_beta = VectorType::Random(4);

    // create data
    MatrixType X = MatrixType::Random(50, 4);
    VectorType y = X*true_beta;

    // perform linear regression
    TLinearRegression f(X, y);

    VectorType beta = VectorType::Random(4);
    std::cout << "start in   " << beta.transpose() << std::endl;
    cppoptlib::BfgsSolver<TLinearRegression> solver;
    solver.minimize(f, beta);

    std::cout << "result     " << beta.transpose() << std::endl;
    std::cout << "true model " << true_beta.transpose() << std::endl;

    return 0;
}
