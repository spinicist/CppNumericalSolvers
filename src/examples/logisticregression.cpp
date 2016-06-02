#include <iostream>
#include "../../include/cppoptlib/meta.h"
#include "../../include/cppoptlib/problem.h"
#include "../../include/cppoptlib/solver/bfgssolver.h"

// to use this library just use the namespace "cppoptlib"
namespace cppoptlib {

// we define a new problem for optimizing the rosenbrock function
// we use a templated-class rather than "auto"-lambda function for a clean architecture
template<typename T>
class LogisticRegression : public Problem<T> {
  public:
    using typename Problem<T>::VectorType;
    using typename Problem<T>::MatrixType;
    const MatrixType X;
    const VectorType y;
    const MatrixType XX;

    LogisticRegression(const MatrixType &X_, const VectorType y_) : X(X_), y(y_), XX(X_.transpose()*X_) {}

    T value(const VectorType &beta) {
        return (1.0/(1.0 + exp(-(X*beta).array())) - y.array()).matrix().squaredNorm();
    }

    void gradient(const VectorType &beta, VectorType &grad) {
        const VectorType p = 1.0/(1.0 + exp(-(X*beta).array()));
        grad = X.transpose()*(p-y);
    }
};

}
int main(int argc, char const *argv[]) {
    typedef double T;
    typedef cppoptlib::LogisticRegression<T> LogReg;
    typedef typename LogReg::VectorType VectorType;
    typedef typename LogReg::MatrixType MatrixType;
    srand((unsigned int) time(0));

    // create true model
    VectorType true_beta = VectorType::Random(4);

    // create data
    MatrixType X = MatrixType::Random(50, 4);
    VectorType y = 1.0/(1.0 + exp(-(X*true_beta).array()));

    // perform linear regression
    LogReg f(X, y);

    VectorType beta = VectorType::Random(4);
    std::cout << "start in   " << beta.transpose() << std::endl;
    cppoptlib::BfgsSolver<LogReg> solver;
    solver.minimize(f, beta);

    std::cout << "result     " << beta.transpose() << std::endl;
    std::cout << "true model " << true_beta.transpose() << std::endl;

    return 0;
}
