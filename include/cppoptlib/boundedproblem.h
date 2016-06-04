/*
 *  This file is part of CppNumericalSolvers
 *
 *  Copyright (C) Tobias Wood @spinicist 2016
 *
 *  This Source Code form is subject to the terms of the MIT license.
 *  Please see the LICENSE file.
 *
 */

#ifndef BOUNDEDPROBLEM_H
#define BOUNDEDPROBLEM_H

#include <vector>
#include <Eigen/Core>

#include "problem.h"

namespace cppoptlib {

template<typename Scalar_, int Dim_ = Eigen::Dynamic>
class BoundedProblem : public Problem<Scalar_, Dim_> {
public:
    using Superclass = Problem<Scalar_, Dim_>;
    using typename Superclass::Scalar;
    using typename Superclass::TVector;

protected:
    TVector m_lowerBound;
    TVector m_upperBound;

public:
    BoundedProblem() : Superclass() {
        m_lowerBound.setConstant(-std::numeric_limits<Scalar>::infinity());
        m_upperBound.setConstant(std::numeric_limits<Scalar>::infinity());
    }

    BoundedProblem(const TVector &l, const TVector &u) :
        Superclass(),
        m_lowerBound(l),
        m_upperBound(u)
    {}

    const TVector &lowerBound() const { return m_lowerBound; }
    void setLowerBound(const TVector &lb) { m_lowerBound = lb; }
    const TVector &upperBound() const { return m_upperBound; }
    void setUpperBound(const TVector &ub) { m_upperBound = ub; }

    void setBoxConstraint(TVector  lb, TVector  ub) {
        setLowerBound(lb);
        setUpperBound(ub);
    }
};

} // end namespace cppoptlib

#endif // BOUNDEDPROBLEM_H
