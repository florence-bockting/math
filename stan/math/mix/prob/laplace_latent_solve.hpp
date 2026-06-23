#ifndef STAN_MATH_MIX_PROB_LAPLACE_LATENT_SOLVE_HPP
#define STAN_MATH_MIX_PROB_LAPLACE_LATENT_SOLVE_HPP

#include <stan/math/mix/functor/laplace_base_solve.hpp>
#include <stan/math/mix/functor/laplace_likelihood.hpp>
#include <stan/math/prim/fun/to_ref.hpp>

namespace stan {
namespace math {

/**
 * In a latent gaussian model,
 *
 *   theta ~ Normal(0, Sigma(phi))
 *   y ~ p(y|theta,phi)
 *
 * returns the posterior mean and Cholesky factor from the Laplace
 * approximation to p(theta|y,phi),
 * where the log likelihood is given by L_f.
 * @tparam LLFunc Type of likelihood function.
 * @tparam LLArgs Type of arguments of likelihood function.
 * \laplace_common_template_args
 * @param[in] L_f Function that returns log likelihood.
 * @param[in] ll_args Arguments for likelihood function.
 * \laplace_common_args
 * @param[in] hessian_block_size Block size for the Hessian approximation with
 * respect to the latent gaussian variable theta.
 * \laplace_options
 * \msg_arg
 */
template <typename LLFunc, typename LLArgs, typename CovarFun,
          typename CovarArgs, typename OpsTuple>
inline auto laplace_latent_tol_solve(LLFunc&& L_f, LLArgs&& ll_args,
                                     int hessian_block_size,
                                     CovarFun&& covariance_function,
                                     CovarArgs&& covar_args, OpsTuple&& ops,
                                     std::ostream* msgs) {
  auto options
      = internal::tuple_to_laplace_options(std::forward<OpsTuple>(ops));
  options.hessian_block_size = hessian_block_size;
  return laplace_base_solve(
      std::forward<LLFunc>(L_f), std::forward<LLArgs>(ll_args),
      std::forward<CovarFun>(covariance_function),
      std::forward<CovarArgs>(covar_args), std::move(options), msgs);
}

/**
 * In a latent gaussian model,
 *
 *   theta ~ Normal(0, Sigma(phi))
 *   y ~ p(y|theta,phi)
 *
 * returns the posterior mean and Cholesky factor
 * from the Laplace approximation of p(theta | y, phi).
 * @tparam LLFunc Type of likelihood function.
 * @tparam LLArgs Type of arguments of likelihood function.
 * \laplace_common_template_args
 * @param[in] L_f Function that returns log likelihood.
 * @param[in] ll_args Arguments for likelihood function.
 * \laplace_common_args
 * @param[in] hessian_block_size Block size for the Hessian approximation with
 * respect to the latent gaussian variable theta.
 * \msg_arg
 */
template <typename LLFunc, typename LLArgs, typename CovarFun,
          typename CovarArgs>
inline auto laplace_latent_solve(LLFunc&& L_f, LLArgs&& ll_args,
                                 int hessian_block_size,
                                 CovarFun&& covariance_function,
                                 CovarArgs&& covar_args, std::ostream* msgs) {
  auto options = laplace_options_default{hessian_block_size};
  return laplace_base_solve(std::forward<LLFunc>(L_f),
                            std::forward<LLArgs>(ll_args),
                            std::forward<CovarFun>(covariance_function),
                            std::forward<CovarArgs>(covar_args), options, msgs);
}

}  // namespace math
}  // namespace stan

#endif
