#ifndef STAN_MATH_MIX_FUNCTOR_LAPLACE_BASE_SOLVE_HPP
#define STAN_MATH_MIX_FUNCTOR_LAPLACE_BASE_SOLVE_HPP

#include <stan/math/prim/fun/Eigen.hpp>
#include <stan/math/mix/functor/laplace_marginal_density.hpp>
#include <stan/math/prim/fun/cholesky_decompose.hpp>

namespace stan {
namespace math {

/**
 * In a latent gaussian model,
 *
 *   theta ~ Normal(theta | 0, Sigma(phi, x))
 *   y ~ pi(y | theta, eta)
 *
 * returns the posterior mean and Cholesky factor
 * from the Laplace approximation of p(theta | y, phi).
 * @tparam LLFunc Type of likelihood function.
 * @tparam LLArgs Tuple of arguments types of likelihood function.
 * \laplace_common_template_args
 * @param ll_fun Likelihood function.
 * @param ll_args Arguments for likelihood function.
 * \laplace_common_args
 * @param options Control parameter for optimizer underlying Laplace approx.
 * \msg_arg
 * @return A tuple containing the posterior mean and Cholesky factor
 */
template <typename LLFunc, typename LLArgs, typename CovarFun,
          typename CovarArgs, bool InitTheta>
inline auto laplace_base_solve(LLFunc&& ll_fun, LLArgs&& ll_args,
                               CovarFun&& covariance_function,
                               CovarArgs&& covar_args,
                               const laplace_options<InitTheta>& options,
                               std::ostream* msgs) {
  Eigen::MatrixXd covariance_train = stan::math::apply(
      [msgs, &covariance_function](auto&&... args) {
        return covariance_function(std::forward<decltype(args)>(args)..., msgs);
      },
      std::forward<CovarArgs>(covar_args));
  auto md_est = internal::laplace_marginal_density_est(
      ll_fun, std::forward<LLArgs>(ll_args), covariance_train, options, msgs);
  Eigen::VectorXd mean_train = covariance_train * md_est.theta_grad;
  Eigen::MatrixXd Sigma;
  if (options.solver == 1 || options.solver == 2) {
    Eigen::MatrixXd V_dec
        = md_est.L.template triangularView<Eigen::Lower>().solve(
            md_est.W_r * covariance_train);
    Sigma = covariance_train - V_dec.transpose() * V_dec;
  } else {
    Sigma = covariance_train
            - covariance_train
                  * (md_est.W_r
                     - md_est.W_r
                           * md_est.LU.solve(covariance_train * md_est.W_r))
                  * covariance_train;
  }
  auto cholesky_factor = cholesky_decompose(Sigma);
  return std::make_tuple(std::move(mean_train), std::move(cholesky_factor));
}

}  // namespace math
}  // namespace stan

#endif
