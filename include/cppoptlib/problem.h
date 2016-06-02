#ifndef PROBLEM_H
#define PROBLEM_H

#include <vector>
#include <Eigen/Dense>

#include "meta.h"

namespace cppoptlib {
template<typename Scalar_, int Dim_ = Eigen::Dynamic>
class Problem {
 public:
  static const int Dim = Dim_;
  typedef Scalar_ Scalar;
  using VectorType = Eigen::Matrix<Scalar, Dim, 1>;
  using SquareMatrixType = Eigen::Matrix<Scalar, Dim, Dim>;
  using MatrixType = Eigen::Matrix<Scalar, Eigen::Dynamic, Eigen::Dynamic>;

 protected:

  bool hasLowerBound_ = false;
  bool hasUpperBound_ = false;

  VectorType lowerBound_;
  VectorType upperBound_;

 public:

  Problem() {}
  virtual ~Problem()= default;

  virtual bool callback(const Criteria<Scalar> &state, const VectorType &x) {
    return true;
  }

  void setBoxConstraint(VectorType  lb, VectorType  ub) {
    setLowerBound(lb);
    setUpperBound(ub);
  }

  void setLowerBound(VectorType  lb) {
    lowerBound_    = lb;
    hasLowerBound_ = true;
  }

  void setUpperBound(VectorType  ub) {
    upperBound_ = ub;
    hasUpperBound_ = true;
  }

  bool hasLowerBound() {
    return hasLowerBound_;
  }

  bool hasUpperBound() {
    return hasUpperBound_;
  }

  VectorType lowerBound() {
    return lowerBound_;
  }

  VectorType upperBound() {
    return upperBound_;
  }

  /**
   * @brief returns objective value in x
   * @details [long description]
   *
   * @param x [description]
   * @return [description]
   */
  virtual Scalar value(const  VectorType &x) = 0;
  /**
   * @brief overload value for nice syntax
   * @details [long description]
   *
   * @param x [description]
   * @return [description]
   */
  Scalar operator()(const  VectorType &x) {
    return value(x);
  }
  /**
   * @brief returns gradient in x as reference parameter
   * @details should be overwritten by symbolic gradient
   *
   * @param grad [description]
   */
  virtual void gradient(const  VectorType &x,  VectorType &grad) {
    finiteGradient(x, grad);
  }

  /**
   * @brief This computes the hessian
   * @details should be overwritten by symbolic hessian, if solver relies on hessian
   */
  virtual void hessian(const VectorType &x, SquareMatrixType &hessian) {
    finiteHessian(x, hessian);

  }

  virtual bool checkGradient(const VectorType &x, int accuracy = 3) {
    // TODO: check if derived class exists:
    // int(typeid(&Rosenbrock<double>::gradient) == typeid(&Problem<double>::gradient)) == 1 --> overwritten
    const int D = x.rows();
    VectorType actual_grad(D);
    VectorType expected_grad(D);
    gradient(x, actual_grad);
    finiteGradient(x, expected_grad, accuracy);
    for (int d = 0; d < D; ++d) {
      Scalar scale = std::max((std::max(fabs(actual_grad[d]), fabs(expected_grad[d]))), 1.);
      if(fabs(actual_grad[d]-expected_grad[d])>1e-2 * scale)
        return false;
    }
    return true;

  }

  virtual bool checkHessian(const VectorType &x, int accuracy = 3) {
    // TODO: check if derived class exists:
    // int(typeid(&Rosenbrock<double>::gradient) == typeid(&Problem<double>::gradient)) == 1 --> overwritten
    const int D = x.rows();

    SquareMatrixType actual_hessian = SquareMatrixType::Zero(D, D);
    SquareMatrixType expected_hessian = SquareMatrixType::Zero(D, D);
    hessian(x, actual_hessian);
    finiteHessian(x, expected_hessian, accuracy);
    for (int d = 0; d < D; ++d) {
      for (int e = 0; e < D; ++e) {
        Scalar scale = std::max(static_cast<Scalar>(std::max(fabs(actual_hessian(d, e)), fabs(expected_hessian(d, e)))), Scalar(1.));
        if(fabs(actual_hessian(d, e)- expected_hessian(d, e))>1e-1 * scale)
          return false;
      }
    }
    return true;
  }

  virtual void finiteGradient(const  VectorType &x, VectorType &grad, int accuracy = 0) final {
    // accuracy can be 0, 1, 2, 3
    const Scalar eps = 2.2204e-6;
    const std::vector<std::vector<Scalar>> coeff =
    { {1, -1}, {1, -8, 8, -1}, {-1, 9, -45, 45, -9, 1}, {3, -32, 168, -672, 672, -168, 32, -3} };
    const std::vector<std::vector<Scalar>> coeff2 =
    { {1, -1}, {-2, -1, 1, 2}, {-3, -2, -1, 1, 2, 3}, {-4, -3, -2, -1, 1, 2, 3, 4} };
    const std::vector<Scalar> dd = {2, 12, 60, 840};

    VectorType finiteDiff(x.rows());
    for (size_t d = 0; d < x.rows(); d++) {
      finiteDiff[d] = 0;
      for (int s = 0; s < 2*(accuracy+1); ++s)
      {
        VectorType xx = x.eval();
        xx[d] += coeff2[accuracy][s]*eps;
        finiteDiff[d] += coeff[accuracy][s]*value(xx);
      }
      finiteDiff[d] /= (dd[accuracy]* eps);
    }
    grad = finiteDiff;
  }

  virtual void finiteHessian(const VectorType &x, SquareMatrixType &hessian, int accuracy = 0) final {
    const Scalar eps = std::numeric_limits<Scalar>::epsilon()*10e7;

    if(accuracy == 0) {
      for (size_t i = 0; i < x.rows(); i++) {
        for (size_t j = 0; j < x.rows(); j++) {
          VectorType xx = x;
          Scalar f4 = value(xx);
          xx[i] += eps;
          xx[j] += eps;
          Scalar f1 = value(xx);
          xx[j] -= eps;
          Scalar f2 = value(xx);
          xx[j] += eps;
          xx[i] -= eps;
          Scalar f3 = value(xx);
          hessian(i, j) = (f1 - f2 - f3 + f4) / (eps * eps);
        }
      }
    } else {
      /*
        \displaystyle{{\frac{\partial^2{f}}{\partial{x}\partial{y}}}\approx
        \frac{1}{600\,h^2} \left[\begin{matrix}
          -63(f_{1,-2}+f_{2,-1}+f_{-2,1}+f_{-1,2})+\\
          63(f_{-1,-2}+f_{-2,-1}+f_{1,2}+f_{2,1})+\\
          44(f_{2,-2}+f_{-2,2}-f_{-2,-2}-f_{2,2})+\\
          74(f_{-1,-1}+f_{1,1}-f_{1,-1}-f_{-1,1})
        \end{matrix}\right] }
      */
      VectorType xx;
      for (size_t i = 0; i < x.rows(); i++) {
        for (size_t j = 0; j < x.rows(); j++) {

          Scalar term_1 = 0;
          xx = x.eval(); xx[i] += 1*eps;  xx[j] += -2*eps;  term_1 += value(xx);
          xx = x.eval(); xx[i] += 2*eps;  xx[j] += -1*eps;  term_1 += value(xx);
          xx = x.eval(); xx[i] += -2*eps; xx[j] += 1*eps;   term_1 += value(xx);
          xx = x.eval(); xx[i] += -1*eps; xx[j] += 2*eps;   term_1 += value(xx);

          Scalar term_2 = 0;
          xx = x.eval(); xx[i] += -1*eps; xx[j] += -2*eps;  term_2 += value(xx);
          xx = x.eval(); xx[i] += -2*eps; xx[j] += -1*eps;  term_2 += value(xx);
          xx = x.eval(); xx[i] += 1*eps;  xx[j] += 2*eps;   term_2 += value(xx);
          xx = x.eval(); xx[i] += 2*eps;  xx[j] += 1*eps;   term_2 += value(xx);

          Scalar term_3 = 0;
          xx = x.eval(); xx[i] += 2*eps;  xx[j] += -2*eps;  term_3 += value(xx);
          xx = x.eval(); xx[i] += -2*eps; xx[j] += 2*eps;   term_3 += value(xx);
          xx = x.eval(); xx[i] += -2*eps; xx[j] += -2*eps;  term_3 -= value(xx);
          xx = x.eval(); xx[i] += 2*eps;  xx[j] += 2*eps;   term_3 -= value(xx);

          Scalar term_4 = 0;
          xx = x.eval(); xx[i] += -1*eps; xx[j] += -1*eps;  term_4 += value(xx);
          xx = x.eval(); xx[i] += 1*eps;  xx[j] += 1*eps;   term_4 += value(xx);
          xx = x.eval(); xx[i] += 1*eps;  xx[j] += -1*eps;  term_4 -= value(xx);
          xx = x.eval(); xx[i] += -1*eps; xx[j] += 1*eps;   term_4 -= value(xx);

          hessian(i, j) = (-63 * term_1+63 * term_2+44 * term_3+74 * term_4)/(600.0 * eps * eps);

        }
      }
    }

  }

};
}

#endif /* PROBLEM_H */
