#include "nloptutility.h"
#include <iostream>

namespace nloptUtility
{

Eigen::VectorXd compute(const Eigen::VectorXd& x_initial,
                        const Eigen::VectorXd& upper,
                        const Eigen::VectorXd& lower,
                        nlopt::vfunc objective_function,
                        void* data,
                        nlopt::algorithm algorithm,
                        int max_evaluations,
                        double tolerance
                        )
{
    const unsigned M = x_initial.rows();

    std::vector<double> l(M);
    std::vector<double> u(M);

    assert(lower.rows() == upper.rows());
    assert(x_initial.rows() == upper.rows());

    Eigen::Map<Eigen::VectorXd>(&l[0], M) = lower;
    Eigen::Map<Eigen::VectorXd>(&u[0], M) = upper;

    nlopt::opt solver(algorithm, M);
    solver.set_lower_bounds(l);
    solver.set_upper_bounds(u);
    solver.set_maxeval(max_evaluations);
    solver.set_xtol_rel(tolerance);
    solver.set_ftol_rel(tolerance);
    solver.set_min_objective(objective_function, data);

    std::vector<double> x_star(M);
    Eigen::Map<Eigen::VectorXd>(&x_star[0], M) = x_initial;

    try
    {
        double d;
        solver.optimize(x_star, d);
    }
    catch (nlopt::roundoff_limited /*e*/)
    {
    }
    catch (std::runtime_error e)
    {
        std::cerr << e.what() << std::endl;
        return x_initial;
    }

    return Eigen::Map<Eigen::VectorXd>(&x_star[0], M);
}

}
