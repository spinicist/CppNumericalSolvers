// CppNumericalSolver
#ifndef ARMIJO_H_
#define ARMIJO_H_

#include "../meta.h"

namespace cppoptlib {

template<typename ProblemType, int Ord>
class Armijo {
public:
    using Scalar = typename ProblemType::Scalar;
    using VectorType = typename ProblemType::VectorType;
    /**
     * @brief use Armijo Rule for (weak) Wolfe conditiions
     * @details [long description]
     *
     * @param searchDir search direction for next update step
     * @param objFunc handle to problem
     *
     * @return step-width
     */
    static Scalar linesearch(const VectorType &x, const VectorType &searchDir, ProblemType &objFunc, const Scalar alpha_init = 1.0) {
        const Scalar c = 0.2;
        const Scalar rho = 0.9;
        Scalar alpha = alpha_init;
        Scalar f = objFunc.value(x + alpha * searchDir);
        const Scalar f_in = objFunc.value(x);
        VectorType grad(x.rows());
        objFunc.gradient(x, grad);
        const Scalar Cache = c * grad.dot(searchDir);

        while(f > f_in + alpha * Cache) {
            alpha *= rho;
            f = objFunc.value(x + alpha * searchDir);
        }

        return alpha;
    }

};

template<typename ProblemType>
class Armijo<ProblemType, 2> {

 public:
    using typename ProblemType::Scalar;
    using typename ProblemType::VectorType;
    using typename ProblemType::SquareMatrixType;
    /**
     * @brief use Armijo Rule for (weak) Wolfe conditiions
     * @details [long description]
     *
     * @param searchDir search direction for next update step
     * @param objFunc handle to problem
     *
     * @return step-width
     */
    static Scalar linesearch(const VectorType &x, const VectorType &searchDir, ProblemType &objFunc) {
        const Scalar c = 0.2;
        const Scalar rho = 0.9;
        Scalar alpha = 1.0;

        Scalar f = objFunc.value(x + alpha * searchDir);
        const Scalar f_in = objFunc.value(x);
        const SquareMatrixType  hessian(x.rows(), x.rows());
        objFunc.hessian(x, hessian);
        VectorType grad(x.rows());
        objFunc.gradient(x, grad);
        const Scalar Cache = c * grad.dot(searchDir) + 0.5 * c*c * searchDir.transpose() * (hessian * searchDir);

        while(f > f_in + alpha * Cache) {
            alpha *= rho;
            f = objFunc.value(x + alpha * searchDir);
        }
        return alpha;
    }

};

}

#endif /* ARMIJO_H_ */
