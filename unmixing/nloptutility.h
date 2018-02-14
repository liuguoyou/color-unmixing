#ifndef NLOPTUTILITY_H
#define NLOPTUTILITY_H

#include <nlopt.hpp>
#include <Eigen/Core>

namespace nloptUtility
{

Eigen::VectorXd compute(const Eigen::VectorXd& x_initial,
                        const Eigen::VectorXd& upper,
                        const Eigen::VectorXd& lower,
                        nlopt::vfunc objective_function,
                        void *data = nullptr,
                        nlopt::algorithm algorithm = nlopt::LD_TNEWTON,
                        int max_evaluations = 1000,
                        double tolerance = 1e-12
                        );
}

#endif // NLOPTUTILITY_H
