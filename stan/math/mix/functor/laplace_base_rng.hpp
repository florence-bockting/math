#ifndef STAN_MATH_MIX_FUNCTOR_LAPLACE_BASE_RNG_HPP
#define STAN_MATH_MIX_FUNCTOR_LAPLACE_BASE_RNG_HPP

#include <stan/math/prim/prob/multi_normal_cholesky_rng.hpp>
#include <stan/math/mix/functor/laplace_base_solve.hpp>

namespace stan {
namespace math {

/**
 * In a latent gaussian model,
 *
 *   theta ~ Normal(theta | 0, Sigma(phi, x))
 *   y ~ pi(y | theta, eta)
 *
 * returns a multivariate normal random variate sampled
 * from the Laplace approximation of p(theta_pred | y, phi, x_pred).
 * Note that while the data is observed at x (train_tuple), the new samples
 * are drawn for covariates x_pred (pred_tuple).
 * To sample the "original" theta's, set pred_tuple = train_tuple.
 * @tparam LLFunc Type of likelihood function.
 * @tparam LLArgs Tuple of arguments types of likelihood function.
 * \laplace_common_template_args
 * @tparam RNG A valid boost rng type
 * @param ll_fun Likelihood function.
 * @param ll_args Arguments for likelihood function.
 * \laplace_common_args
 * @param options Control parameter for optimizer underlying Laplace approx.
 * \rng_arg
 * \msg_arg
 */
template <typename LLFunc, typename LLArgs, typename CovarFun,
          typename CovarArgs, bool InitTheta, typename RNG,
          require_t<is_all_arithmetic_scalar<CovarArgs, LLArgs>>* = nullptr>
inline Eigen::VectorXd laplace_base_rng(
    LLFunc&& ll_fun, LLArgs&& ll_args, CovarFun&& covariance_function,
    CovarArgs&& covar_args, const laplace_options<InitTheta>& options, RNG& rng,
    std::ostream* msgs) {
  auto [mean_train, cholesky_factor] = laplace_base_solve(
      ll_fun, ll_args, covariance_function, covar_args, options, msgs);
  return multi_normal_cholesky_rng(std::move(mean_train),
                                   std::move(cholesky_factor), rng);
}

}  // namespace math
}  // namespace stan

#endif
