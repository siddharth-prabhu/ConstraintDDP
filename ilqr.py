import logging
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)
import os
import operator
from typing import NamedTuple, Callable, Tuple, Optional
from datetime import datetime

import jax
jax.config.update("jax_enable_x64", True)
import jax.numpy as jnp
from jax import tree_util


def check_dir(dir : str) -> None :

    if not os.path.exists(dir):
        os.makedirs(dir)

_dir = os.path.join("log", "ilqr", str(datetime.now()))
check_dir(_dir)
logfile = logging.FileHandler(os.path.join(_dir, "solver_stats.txt"))
logger.addHandler(logfile)

# Notation from https://github.com/StanfordASL/AA203-Notes/blob/master/notes.pdf;
# Code from https://colab.research.google.com/github/StanfordASL/AA203-Examples/blob/master/LQR%20Variants.ipynb#scrollTo=prescribed-sharp

# TODO add equality constraints.
# https://mariozanon.wordpress.com/wp-content/uploads/2023/11/sqp.pdf

def debug_print(fmt : str, *args, **kwargs):
    jax.debug.callback(lambda *args, **kwargs : logger.info(fmt.format(*args, **kwargs)), *args, **kwargs)


class LinearDynamics(NamedTuple):
    f_x : jnp.array
    f_u : jnp.array
    f_xx : jnp.array
    f_ux : jnp.array
    f_uu : jnp.array

    def __call__(self, x, u, k = None):
        f_x, f_u, f_xx, f_ux, f_uu = self

        if k is None: 
            return (f_x @ x + f_u @ u + 1/2 * (
                jnp.kron(x, jnp.eye(len(x))) @ jnp.vstack(f_xx) @ x + 
                jnp.kron(u, jnp.eye(len(u))) @ jnp.vstack(f_ux) @ x + 
                jnp.kron(x, jnp.eye(len(u))) @ jnp.vstack(f_uu) @ u)
            )
        else : 
            return self[k](x, u)

    def __getitem__(self, key):
        return tree_util.tree_map(lambda x : x[key], self)


class AffinePolicy(NamedTuple):
    l : jnp.array
    l_x : jnp.array

    def __call__(self, x, k = None):
        l, l_x = self
        return l + l_x@x if k is None else self[k](x)

    def __getitem__(self, k):
        return tree_util.tree_map(lambda x : x[k], self)


class LagrangePolicy(NamedTuple):
    b : jnp.array
    b_x : jnp.array
    b_u : jnp.array
    a : jnp.array
    a_x : jnp.array
    a_u : jnp.array

    def __call__(self, x, u, k = None):
        b, b_x, b_u, a, a_x, a_u = self
        return (b + b_x @ x + b_u @ u, a + a_x @ x + a_u @ u) if k is None else self[k](x, u)
    
    def __getitem__(self, k):
        return tree_util.tree_map(lambda x : x[k], self)


class TerminalLagrangePolicy(NamedTuple):
    b : jnp.array
    b_x : jnp.array
    a : jnp.array
    a_x : jnp.array

    def __call__(self, x):
        b, b_x, a, a_x = self
        return (b + b_x @ x, a + a_x @ x)
    

class QuadraticConstraints(NamedTuple):
    h : jnp.array
    h_x : jnp.array
    h_u : jnp.array
    h_xx : jnp.array
    h_uu : jnp.array
    h_ux : jnp.array

    @classmethod
    def from_pure_quadratic(cls, h_xx, h_uu, h_ux):
        return cls(
            jnp.zeros(h_xx.shape[:-2]),
            jnp.zeros(h_xx.shape[:-1]),
            jnp.zeros(h_uu.shape[:-1]),
            h_xx, 
            h_uu, 
            h_ux
        )

    def __call__(self, x, u, k = None):
        pass

    def __getitem__(self, k):
        return tree_util.tree_map(lambda x : x[k], self)


class LagrangeVariables(NamedTuple):
    s : jnp.array # slack variables for inequality constraints
    lam : jnp.array # lagrange variables for equality constraints

    def __getitem__(self, k):
        return tree_util.tree_map(lambda x : x[k], self)

    def append(self, val):
        return tree_util.tree_map(jnp.vstack, self, val)


class OptimizationConstants(NamedTuple):
    tau : jnp.array
    reg : jnp.array
    alpha : jnp.array

    def __getitem__(self, k):
        return tree_util.tree_map(lambda x : x[k], self)
    
    def append(self, val):
        return tree_util.tree_map(jnp.vstack, self, val)


class QuadraticCost(NamedTuple):
    c : jnp.array
    c_x : jnp.array
    c_u : jnp.array
    c_xx : jnp.array
    c_uu : jnp.array
    c_ux : jnp.array

    @classmethod
    def from_pure_quadratic(cls, c_xx, c_uu, c_ux):
        return cls(
            jnp.zeros(c_xx.shape[:-2]),
            jnp.zeros(c_xx.shape[:-1]),
            jnp.zeros(c_uu.shape[:-1]),
            c_xx, 
            c_uu, 
            c_ux,
        )
    
    def __call__(self, x, u, k = None):
        c, c_x, c_u, c_xx, c_uu, c_ux = self
        return c + c_x @ x + c_u @ u + x @ c_xx @ x /2 + u @ c_uu @ u / 2 + u @ c_ux @ x if k is None else self[k](x, u)

    def __getitem__(self, k):
        return tree_util.tree_map(lambda x : x[k], self)


class QuadraticStateCost(NamedTuple):
    v : jnp.array
    v_x : jnp.array
    v_xx : jnp.array

    @classmethod
    def from_pure_quadratic(cls, v_xx):
        return cls(
            jnp.zeros(v_xx.shape[:-2]),
            jnp.zeros(v_xx.shape[:-1]),
            v_xx
        )

    def __call__(self, x, k = None):
        v, v_x, v_xx = self
        return v + v_x @ x + x @ v_xx @ x / 2 if k is None else self[k](x)

    def __getitem__(self, k):
        return tree_util.tree_map(lambda x : x[k], self)


class AuxData(NamedTuple):
    q_u : jnp.ndarray
    reg : jnp.ndarray
    cost_reduction : jnp.ndarray

    def __call__(self, k = None):
        q_u, *_ = self
        return q_u if k is None else self[k]()

    def __getitem__(self, k):
        return tree_util.tree_map(lambda x : x[k], self)
    

class TotalCost(NamedTuple):
    running_cost : Callable
    terminal_cost : Callable
    running_inequality_constraints_cost : Callable 
    terminal_inequality_constraints_cost : Callable
    running_equality_constraints_cost : Callable
    terminal_equality_constraints_cost : Callable
    
    @classmethod
    def form_cost(cls, running_cost, terminal_cost = None, 
        running_inequality_constraints_cost = None, terminal_inequality_constraints_cost = None, 
        running_equality_constraints_cost = None, terminal_equality_constraints_cost = None):

        class NoCostVec(NamedTuple):
            # No cost returns a empty vector
            def __call__(self, x, *args):
                return jnp.array([])

        class NoCostScl(NamedTuple):
            # No cost returns a empty scalar
            def __call__(self, x, *args):
                return jnp.sum(jnp.array([]))


        return cls(
            running_cost, 
            NoCostScl() if terminal_cost is None else terminal_cost, 
            NoCostVec() if running_inequality_constraints_cost is None else running_inequality_constraints_cost, 
            NoCostVec() if terminal_inequality_constraints_cost is None else terminal_inequality_constraints_cost,
            NoCostVec() if running_equality_constraints_cost is None else running_equality_constraints_cost,
            NoCostVec() if terminal_equality_constraints_cost is None else terminal_equality_constraints_cost
        )

    # augment the cost with constraints. The cost might increase while satisfying some constraints
    def __call__(self, xs, us, s, lam, st, lamt, tau):
        # arguments : state, control inputs, slack variables for running constraints, slack variables for terminal constraints, barrier parameter 
        # returns total cost, total merit function, and max norm kkt conditions
        _running_cost = lambda x, u, s, lam : self.running_cost(x, u) # + (tau/s) @ (self.running_inequality_constraints_cost(x,  u) + s) - jnp.sum(jnp.log(s))*tau + lam @ self.running_equality_constraints_cost(x, u)
        total_running_cost = jnp.sum(jax.vmap(_running_cost)(xs[:-1], us, s, lam))
        total_terminal_cost = self.terminal_cost(xs[-1]) # + (tau/st) @ (self.terminal_inequality_constraints_cost(xs[-1]) + st) - jnp.sum(jnp.log(st))*tau + lamt @ self.terminal_equality_constraints_cost(xs[-1])
        
        _running_inequality_constraints_cost = lambda x, u, s : jnp.abs(self.running_inequality_constraints_cost(x, u) + s)
        _terminal_inequality_constraints_cost = jnp.abs(self.terminal_inequality_constraints_cost(xs[-1]) + st)
        _running_equality_constraints_cost = lambda x, u : jnp.abs(self.running_equality_constraints_cost(x, u))
        _terminal_equality_constraints_cost = jnp.abs(self.terminal_equality_constraints_cost(xs[-1]))
        
        _running_merit_cost = lambda x, u, s : self.running_cost(x, u) - jnp.sum(jnp.log(s))*tau
        total_running_merit_cost = jnp.sum(jax.vmap(_running_merit_cost)(xs[:-1], us, s))
        total_terminal_merit_cost = self.terminal_cost(xs[-1]) - jnp.sum(jnp.log(st))*tau

        # constraints infeasibility
        _running_inequality_constraints_infeasibility = jnp.sum(jax.vmap(_running_inequality_constraints_cost)(xs[:-1], us, s))
        _terminal_inequality_constraints_infeasibility = jnp.sum(_terminal_inequality_constraints_cost)
        _running_equality_constraints_infeasibility = jnp.sum(jax.vmap(_running_equality_constraints_cost)(xs[:-1], us))
        _terminal_equality_constraints_infeasibility = jnp.sum(_terminal_equality_constraints_cost)

        # aggregate all costs
        _total_cost = total_running_cost + total_terminal_cost
        _total_merit_cost = total_running_merit_cost + total_terminal_merit_cost # + _terminal_equality_constraints_infeasibility + _running_equality_constraints_infeasibility
        _total_constraints_infeasibility = (
            _running_inequality_constraints_infeasibility + _terminal_inequality_constraints_infeasibility 
            + _running_equality_constraints_infeasibility + _terminal_equality_constraints_infeasibility
        )

        debug_print("Infeasible equality constraints {}", _running_equality_constraints_infeasibility + _terminal_equality_constraints_infeasibility)
        debug_print("Terminal cost {}", self.terminal_cost(xs[-1]))

        return _total_cost, _total_merit_cost, _total_constraints_infeasibility


def rollout_state_feedback_policy(dynamics : Callable, policy : Callable, lagrange_policy : Callable, terminal_lagrange_policy : Callable, 
                                  x0, step_range, x_nom = None, u_nom = None, s_nom = None, lam_nom = None, st_nom = None, lamt_nom = None):
    # x_nom should be length N + 1
    # u_nom should be length N
    # s_nom should be lenght N

    def body_fun(x, k):
        u = policy(x, k) if x_nom is None else u_nom[k] + policy(x - x_nom[k], k)
        s, lam = lagrange_policy(x, u, k) if x_nom is None else tree_util.tree_map(operator.add, (s_nom[k], lam_nom[k]), lagrange_policy(x - x_nom[k], u - u_nom[k], k))
        xi = dynamics(x, u, k)
        return xi, (xi, u, s, lam)

    _, (xs, us, ss, lams) = jax.lax.scan(body_fun, x0, step_range)
    sst, lamst = terminal_lagrange_policy(xs[-1]) if st_nom is None else tree_util.tree_map(operator.add, (st_nom, lamt_nom), terminal_lagrange_policy(xs[-1] - x_nom[-1]))
    return jnp.vstack((x0, xs)), us, ss, lams, sst, lamst


def ensure_positive_definite(a, reg, max_reg = 1e20, eps = 1e-6):
    # regularize hessian to avoid ill-conditioning
    # restarting regularization instead of using the previous estimated value seems to converge the problem 
    
    def body_fun(val):
        a, _, reg = val
        a -= reg * I
        a += reg * 10 * I
        w, _ = jnp.linalg.eigh(a)
        return a, w, reg * 10

    def cond_fun(val):
        _, w, reg = val
        return (jnp.sum(w < eps) > 0) & (reg <= max_reg)
    
    init_reg = 1e-4
    I = jnp.eye(len(a))

    w, _ = jnp.linalg.eigh(a)
    a = jax.lax.cond(jnp.sum(w < eps) > 0, lambda : a + init_reg*I, lambda : a)
    a, _, chosen_reg = jax.lax.while_loop(cond_fun, body_fun, (a, w, init_reg))
    # debug_print("Chosen hessian regularization {}", chosen_delta/10)
    return a, chosen_reg

def ricatti_step(
    current_step_dynamics : LinearDynamics, current_step_cost : QuadraticCost, current_inequality_constraints : QuadraticConstraints, current_equality_constraints : QuadraticConstraints, 
    next_state_value : QuadraticStateCost, lagrange_variables : LagrangeVariables, regularization : jnp.ndarray, opt_const : OptimizationConstants, approx_hessian : bool = False
    ):

    f_x, f_u, f_xx, f_ux, f_uu = current_step_dynamics
    c, c_x, c_u, c_xx, c_uu, c_ux = current_step_cost
    h, h_x, h_u, h_xx, h_uu, h_ux = current_inequality_constraints
    g, g_x, g_u, g_xx, g_uu, g_ux = current_equality_constraints
    v, v_x, v_xx = next_state_value
    s, lam = lagrange_variables
    tau = opt_const

    # Get shapes
    m, n = c_ux.shape
    v_xx = (v_xx + v_xx.T) / 2. # ensure symmetric

    # update cost to account for inequality constraints
    inv_reg_ineq = jnp.diag(tau / s**2)
    q = c + (tau/s) @ (h + s) - jnp.sum(jnp.log(s))*tau + lam @ g + v
    q_x = c_x + (tau/s) @ h_x + h_x.T @ ((h + s) * (tau/s**2)) + lam @ g_x + f_x.T @ v_x
    q_u = c_u + (tau/s) @ h_u + h_u.T @ ((h + s) * (tau/s**2)) + lam @ g_u + f_u.T @ v_x
    _q_xx = jax.lax.cond(approx_hessian,
        lambda : c_xx + h_x.T @ inv_reg_ineq @ h_x + f_x.T @ v_xx @ f_x, 
        lambda : c_xx + (h_xx.T @ (tau/s)).T + h_x.T @ inv_reg_ineq @ h_x + (g_xx.T @ lam).T + f_x.T @ v_xx @ f_x + jnp.kron(v_x.T, jnp.eye(n)) @ jnp.vstack(f_xx),
    )
    _q_ux = jax.lax.cond(approx_hessian,
        lambda : c_ux + h_u.T @ inv_reg_ineq @ h_x + f_u.T @ v_xx @ f_x, 
        lambda : c_ux + (h_ux.T @ (tau/s)).T + h_u.T @ inv_reg_ineq @ h_x + (g_ux.T @ lam).T + f_u.T @ v_xx @ f_x + jnp.kron(v_x.T, jnp.eye(m)) @ jnp.vstack(f_ux),
    )
    _q_uu = jax.lax.cond(approx_hessian,
        lambda : c_uu + h_u.T @ inv_reg_ineq @ h_u + f_u.T @ v_xx @ f_u, 
        lambda : c_uu + (h_uu.T @ (tau/s)).T + h_u.T @ inv_reg_ineq @ h_u + (g_uu.T @ lam).T + f_u.T @ v_xx @ f_u + jnp.kron(v_x.T, jnp.eye(m)) @ jnp.vstack(f_uu),
    )

    # add regularization to ensure positive definite hessian
    # https://homes.cs.washington.edu/~todorov/papers/TassaIROS12.pdf
    _q_uu += f_u.T @ (regularization*jnp.eye(n)) @ f_u
    _q_ux += f_u.T @ (regularization*jnp.eye(n)) @ f_x
    _q_zz = jnp.block([[_q_xx, _q_ux.T], [_q_ux, _q_uu]]) # transpose is fine since q is scalar

    ## update hessian and gradients to account for equality constraints
    inv_reg_eq = - ( 0.1 / tau) * jnp.diag(jnp.ones_like(g))
    _g_z = jnp.block([g_x, g_u])
    _q_zz += - _g_z.T @ inv_reg_eq @ _g_z
    _q_zz, reg = ensure_positive_definite(_q_zz, regularization)
    q_xx, q_uu, q_ux = _q_zz[:n, :n], _q_zz[-m:, -m:], _q_zz[-m:, :n]

    q_x += - g_x.T @ inv_reg_eq @ g
    q_u += - g_u.T @ inv_reg_eq @ g

    l = -jnp.linalg.solve(q_uu, q_u)
    l_x = -jnp.linalg.solve(q_uu, q_ux)

    current_state_value = QuadraticStateCost(
        q + (l.T @ q_uu @ l) / 2 + l.T @ q_u, # q - (l.T @ q_uu @ l) / 2, # 
        q_x + l_x.T @ q_uu @ l + l_x.T @ q_u + q_ux.T @ l, # q_x - l_x.T @ q_uu @ l, # 
        q_xx + l_x.T @ q_uu @ l_x + l_x.T @ q_ux + q_ux.T @ l_x # q_xx - l_x.T @ q_uu @ l_x, # 
    )

    current_step_optimal_policy = AffinePolicy(l, l_x)
    current_step_optimal_slack = LagrangePolicy(- (h + s), - h_x, - h_u, - inv_reg_eq @ g, - inv_reg_eq @ g_x, - inv_reg_eq @ g_u)
    current_step_aux_data = AuxData(q_u, regularization, - (l.T @ q_uu @ l) / 2)

    return current_state_value, current_step_optimal_policy, current_step_optimal_slack, current_step_aux_data

def iterative_linear_quadratic_regulator(dynamics : Callable, total_cost : Callable, x0 : jnp.ndarray, u_guess : jnp.ndarray, maxiter = 100, 
        atol : float = 1e-9, tol : float = 1e-8, init_tau : Optional[float] = None, approx_hessian : bool = False
    ):

    running_cost, terminal_cost, running_inequality_constraints, terminal_inequality_constraints, running_equality_constraints, terminal_equality_constraints = total_cost
    nx, (N, nu), nh, nht, ng, ngt = (x0.shape[-1], u_guess.shape, running_inequality_constraints(x0, u_guess[-1]).shape[-1], terminal_inequality_constraints(x0).shape[-1], 
        running_equality_constraints(x0, u_guess[-1]).shape[-1], terminal_equality_constraints(x0).shape[-1])  # get shapes
    step_range = jnp.arange(N)

    # current trajectory
    xs, us, ss, lams, sst, lamst = rollout_state_feedback_policy(
        dynamics, 
        lambda x, k : u_guess[k], 
        lambda x, u, k : (running_inequality_constraints(x, u), running_equality_constraints(x, u)), # lagrange policy
        lambda x : (terminal_inequality_constraints(x), terminal_equality_constraints(x)), # terminal lagrange policy
        x0, 
        step_range
    )
    xs_iterates, us_iterates = xs[jnp.newaxis], us[jnp.newaxis]
    
    # current constraints
    hs = jax.vmap(running_inequality_constraints)(xs[:-1], us)
    hst = terminal_inequality_constraints(xs[-1])
    gs = jax.vmap(running_equality_constraints)(xs[:-1], us)
    gst = terminal_equality_constraints(xs[-1])

    hs_iterates, hst_iterates = hs[jnp.newaxis], hst[jnp.newaxis]
    gs_iterates, gst_iterates = gs[jnp.newaxis], gst[jnp.newaxis]

    lagrange_iterates = LagrangeVariables(
        jnp.max(ss, initial = 1.)*jnp.ones((1, N, nh)), # running slack variables
        jnp.ones((1, N, ng)) # running lagrange variables for equality constraints
    )
    terminal_lagrange_iterates = LagrangeVariables(
        jnp.max(sst, initial = 1.)*jnp.ones((1, nht)), # terminal slack variables,
        jnp.ones((1, ngt)) # terminal lagrange variables for equality constraints
    )

    _j_curr, *_ = total_cost(xs, us, *lagrange_iterates[0], *terminal_lagrange_iterates[0], jnp.array(0.))
    opt_const_iterates = OptimizationConstants(
        jnp.maximum(jnp.array([ jnp.abs(_j_curr) / jnp.max(N, initial = 1.) / jnp.max(nh, initial = 1.) ]), 1.) if init_tau is None else jnp.array([init_tau]), # penalty factor
        jnp.zeros((1, N)), # regularization coefficient
        jnp.zeros([1]), # alpha in line search 
    )

    # current cost
    j_curr, m_curr, inf_curr = total_cost(xs, us, *lagrange_iterates[0], *terminal_lagrange_iterates[0], opt_const_iterates[0].tau)
    debug_print("Initial costs, {}", (j_curr, m_curr, inf_curr))
    cost_iterates = (j_curr, m_curr, inf_curr)
    value_functions_iterates = QuadraticStateCost.from_pure_quadratic(jnp.zeros((N + 1, nx, nx)))

    # define running cost functions
    # TODO implement jvp/hvp or vjp/vhp rather than calculating jacobians and hessians
    jac_f = jax.vmap(jax.jacrev(dynamics, argnums = (0, 1))) # dynamics are only reverse mode differentiable
    hess_f = jax.vmap(jax.jacrev(jax.jacrev(dynamics, argnums = (0, 1)), argnums = (0, 1)))
    jac_c = jax.vmap(jax.value_and_grad(running_cost, argnums = (0, 1))) # running cost is scalar
    jac_h = jax.vmap(jax.jacobian(running_inequality_constraints, argnums = (0, 1)))
    jac_g = jax.vmap(jax.jacobian(running_equality_constraints, argnums = (0, 1)))
    hess_c = jax.vmap(jax.hessian(running_cost, argnums = (0, 1))) 
    hess_h = jax.vmap(jax.hessian(running_inequality_constraints, argnums = (0, 1)))
    hess_g = jax.vmap(jax.hessian(running_equality_constraints, argnums = (0, 1)))

    # define termminal cost functions
    jac_ct = jax.value_and_grad(terminal_cost) # terminal cost is scalar
    jac_ht = jax.jacobian(terminal_inequality_constraints)
    jac_gt = jax.jacobian(terminal_equality_constraints)
    hess_ct = jax.hessian(terminal_cost)
    hess_ht = jax.hessian(terminal_inequality_constraints)
    hess_gt = jax.hessian(terminal_equality_constraints) 

    def continuation_criterion(loop_vars):
        i, _, _, _, _, (j_curr, m_curr, inf_curr), *_, opt_const = loop_vars
        return (i < maxiter) and (jnp.maximum(inf_curr, opt_const.tau) > atol)
        
    @jax.jit
    def ilqr_iteration(loop_vars):
        i, xs, us, (hs, hst, gs, gst), (lagrange_variables, terminal_lagrange_variables), (j_curr, m_curr, inf_curr), cost_prev, value_functions_iterates, opt_const = loop_vars
        debug_print("Start of iteration {} -------------------------------------------------------------------------------------------", i)

        # running cost variables
        f_x, f_u = jac_f(xs[:-1], us, step_range)
        (f_xx, _), (f_ux, f_uu) = hess_f(xs[:-1], us, step_range)
        c, (c_x, c_u) = jac_c(xs[:-1], us, step_range)
        h, h_x, h_u = jax.vmap(running_inequality_constraints)(xs[:-1], us, step_range), *jac_h(xs[:-1], us, step_range)
        g, g_x, g_u = jax.vmap(running_equality_constraints)(xs[:-1], us, step_range), *jac_g(xs[:-1], us, step_range)
        (c_xx, _), (c_ux, c_uu) = hess_c(xs[:-1], us, step_range)
        (h_xx, _), (h_ux, h_uu) = hess_h(xs[:-1], us, step_range)
        (g_xx, _), (g_ux, g_uu) = hess_g(xs[:-1], us, step_range)
        
        # terminal cost variables
        s, lam, st, lamt = *lagrange_variables, *terminal_lagrange_variables
        tau = opt_const.tau

        # TODO update terminal value function to account for terminal equality constraints. Add regularization
        ct, ct_x, ct_xx = *jac_ct(xs[-1]), hess_ct(xs[-1])
        ht, ht_x, ht_xx = terminal_inequality_constraints(xs[-1]), jac_ht(xs[-1]), hess_ht(xs[-1])
        gt, gt_x, gt_xx = terminal_equality_constraints(xs[-1]), jac_gt(xs[-1]), hess_gt(xs[-1])
        v = ct + (tau/st) @ (ht + st) - jnp.sum(jnp.log(st))*tau + (lamt @ gt)
        v_x = ct_x + (tau/st) @ ht_x + ht_x.T @ (ht* tau/st**2) + ht_x.T @ (tau/st) + (lamt @ gt_x)
        v_xx = ct_xx + (ht_xx.T @ (tau/st)).T + ht_x.T @ jnp.diag(tau/st**2) @ ht_x + (gt_xx.T @ lamt).T
        
        # find terminal dlam 
        inv_reg = - (0.1 / tau) * jnp.diag(jnp.ones_like(lamt))
        v_xx += - gt_x.T @ inv_reg @ gt_x # check dimensions in linalg.solve
        v_xx, _ = ensure_positive_definite(v_xx, 10.)
        v_x += - gt_x.T @ inv_reg @ gt

        linearized_dynamics = LinearDynamics(f_x, f_u, f_xx, f_ux, f_uu)
        quadratized_running_cost = QuadraticCost(c, c_x, c_u, c_xx, c_uu, c_ux)
        quadratized_terminal_cost = QuadraticStateCost(v, v_x, v_xx)
        quadratized_inequality_constraints = QuadraticConstraints(h, h_x, h_u, h_xx, h_uu, h_ux)
        quadratized_equality_constraints = QuadraticConstraints(g, g_x, g_u, g_xx, g_uu, g_ux)
        terminal_lagrange_policy = TerminalLagrangePolicy(- (ht + st), - ht_x, - inv_reg @ gt, - inv_reg @ gt_x)

        # TODO implement last step inside the scan function
        def scan_fun(next_state_value, current_iterate):
            current_step_dynamics, current_step_inequality_constraints, current_step_equality_constraints, current_step_cost, current_lagrange_variables, current_regularization = current_iterate
            current_state_value, current_step_optimal_policy, current_step_lagrange_policy, current_step_aux_data = ricatti_step(
                current_step_dynamics, current_step_cost, current_step_inequality_constraints, current_step_equality_constraints, 
                next_state_value, current_lagrange_variables, current_regularization, opt_const.tau, approx_hessian = approx_hessian
            )
            return current_state_value, (current_state_value, current_step_optimal_policy, current_step_lagrange_policy, current_step_aux_data)

        _, (value_functions, policy, lagrange_policy, aux_data) = jax.lax.scan(
            scan_fun, 
            quadratized_terminal_cost, 
            (linearized_dynamics, quadratized_inequality_constraints, quadratized_equality_constraints, quadratized_running_cost, lagrange_variables, opt_const.reg), reverse = True)


        def step(x : Tuple, phi : float = 0.995):

            def _cond(x, x_step):
                # reject step if false else accept step
                return tree_util.tree_reduce(operator.and_, tree_util.tree_map(lambda y, z : jnp.all(y >= (1 - phi)*z), x_step, x)) 
            
            def true_fun(val):
                _, b = val
                return b
            
            def false_fun(val):
                # perform binary search

                def cond_fun(_val):
                    a, b, *_ = _val
                    return jnp.abs(b - a) > gold * 1e-16

                def body_fun(_val):
                    left, right, mid = _val
                    
                    *_, _s, _lam, _st, _lamt = rollout_linesearch_policy(mid, policy, lagrange_policy, terminal_lagrange_policy)
                    left, right = jax.lax.cond(_cond(x, (_s, _st)), lambda : (mid, right), lambda : (left, mid))
                    
                    mid = (left + right) / 2 
                    return left, right, mid

                a, b = val
                mid = (a + b) / 2
                left, *_ = jax.lax.while_loop(cond_fun, body_fun, (a, b, mid))
                return left
            
            gold = (jnp.sqrt(5.) + 1.) / 2.
            full_newton_solution = rollout_linesearch_policy(1., policy, lagrange_policy, terminal_lagrange_policy) # full newton
            opt_alpha = jax.lax.cond(_cond(x, (full_newton_solution[-4], full_newton_solution[-2])), true_fun, false_fun, (0., 1.))
            return opt_alpha, rollout_linesearch_policy(opt_alpha, policy, lagrange_policy, terminal_lagrange_policy), full_newton_solution

        def rollout_linesearch_policy(alpha, policy, lagrange_policy, terminal_lagrange_policy):
            # Note that we roll out the true `dynamics`, not the `linearized_dynamics`as done in stagewise newtons method ! 
            l, l_x = policy
            b, b_x, b_u, a, a_x, a_u = lagrange_policy
            bt, bt_x, at, at_x = terminal_lagrange_policy
            
            xs_next, us_next, s_next, lam_next, st_next, lamt_next = rollout_state_feedback_policy(
                dynamics, 
                AffinePolicy(alpha * l, l_x), 
                LagrangePolicy(alpha * b, b_x, b_u, alpha * a, a_x, a_u), 
                TerminalLagrangePolicy(alpha * bt, bt_x, alpha * at, at_x), 
                x0, step_range, xs, us, s, lam, st, lamt
            )
            return xs_next, us_next, s_next, lam_next, st_next, lamt_next

        def backtracking_linesearch():
            """ Backtracking line search to find a solution that leads to a smaller value of the lagrangian """
            
            # simulate for alpha_max. If infeasibility has increased apply solution else do not do anything
            _inf_min = tol # jnp.maximum(1, inf_curr) * 10**-4
            _, max_newton_solution, full_newton_solution = max_alpha, _, (xs_full, us_full, s_full, _, st_full, _) = step((s, st))
            max_newton_cost = total_cost(*max_newton_solution, tau)

            def check_step(val):
                # Accept alpha. Merit function has not increased
                alpha, _, _cost = val
                debug_print("End of backtracking step")
                jax.lax.cond(
                    alpha < tol, 
                    lambda : debug_print("-------------------------------------- WARNING : Step size is smaller than acceptable tolerance of {}", tol), 
                    lambda : jax.lax.cond(
                        (_cost[-1] > _inf_min) & (_cost[-1] > inf_curr),
                        lambda : debug_print("Merit function has decreased, infeasibility has increased"),
                        lambda : debug_print("Merit function has decreased, infeasibility has decreased")
                    )
                )
            
            def cond_fun(val):
                # return True to perform backtracking else False
                alpha, _, _cost = val
                # True to accept step and False to reject step and perfrom backtracking
                accept_step =  jax.lax.cond(
                        _cost[-1] <= _inf_min,
                        lambda : _cost[1] <= m_curr - 1e-5 * inf_curr, # jnp.logical_and(jnp.logical_and(jvp_merit < 0, alpha * (- jvp_merit)**2.3 > inf_curr**1.1), _cost[1] <= m_curr + alpha * 1e-4 * jvp_merit), # _cost[1] <= m_curr + alpha * 1e-4 * jvp_merit, # Armijo condition
                        lambda : jnp.logical_or((_cost[1] <= m_curr - inf_curr * 10**-5), (_cost[-1] <= (1 - 10**-5) * inf_curr))
                    )
                return (alpha >= tol) & jnp.logical_not(accept_step)
            
            def body_fun(val):
                # Do not accept alpha. Peform further backtracking. Merit function has increased. 
                # Implement a simple backtracking algorithm
                alpha, _, _cost = val 
                """
                debug_print("Backtrack. Merit function has increased")
                """
                jax.lax.cond(
                    (_cost[-1] > _inf_min) & (_cost[-1] > inf_curr),
                    lambda : debug_print("While backtracking infeasibility has increased"),
                    lambda : debug_print("While backtracking infeasibility has decreased or below tolerance")
                )

                alpha /= 2. # new alpha
                alpha_solution = rollout_linesearch_policy(alpha, policy, lagrange_policy, terminal_lagrange_policy)   
                alpha_cost = total_cost(*alpha_solution, tau)
                return alpha, alpha_solution, alpha_cost
            
            # TODO need to incorporate feasibility resotration phase
            solution = jax.lax.while_loop(cond_fun, body_fun, (max_alpha, max_newton_solution, max_newton_cost))
            check_step(solution)
            return solution
        
        inf_qu = jnp.max(jnp.abs(aux_data()))
        debug_print("norm of qu {}, norm of cost reduction {}", inf_qu, jnp.abs(jnp.sum(aux_data.cost_reduction)))
        
        alpha, (xs_new, us_new, s_new, lam_new, st_new, lamt_new), cost_new = backtracking_linesearch()
        lagrange_variables_new = LagrangeVariables(s_new, lam_new)
        terminal_lagrange_variables_new = LagrangeVariables(st_new, lamt_new)

        debug_print("chosen_alpha {}", alpha)
        hs_new = jax.vmap(running_inequality_constraints)(xs_new[:-1], us_new)
        hst_new = terminal_inequality_constraints(xs_new[-1])
        gs_new = jax.vmap(running_equality_constraints)(xs_new[:-1], us_new)
        gst_new = terminal_equality_constraints(xs_new[-1])
        _, min_merit, min_inf = cost_new

        new_tau = jnp.where(
            (alpha >= tol) & (jnp.maximum(jnp.maximum(min_inf / 10., inf_qu / 10.), jnp.abs(jnp.sum(aux_data.cost_reduction)) / 10.) <= opt_const.tau), 
            jnp.maximum(atol/10, jnp.minimum(opt_const.tau*0.2, opt_const.tau**1.5)), # decrease tau
            opt_const.tau # no change
        )
        
        # update regularization and alpha
        opt_const_new = OptimizationConstants(
            new_tau,
            jnp.clip(jnp.where(alpha < tol, aux_data.reg*2., aux_data.reg*0.7), a_min = 1e-6, a_max = 1e20),
            alpha
        )
        
        cost_new = (_, min_merit, min_inf)
        
        debug_print("Updated tau {}", new_tau)
        debug_print("New regularization {}", opt_const_new.reg[0])
        debug_print("New cost {}", cost_new)
        # debug_print("End of iteration {} -------------------------------------------------------------------------------------------", i)

        def accept_new_loop_vars():
            debug_print("Accepted new step")
            return [
                i + 1, 
                xs_new, 
                us_new, 
                (hs_new, hst_new, gs_new, gst_new), 
                (lagrange_variables_new, terminal_lagrange_variables_new),
                cost_new, 
                (j_curr, m_curr, inf_curr),
                value_functions_iterates,
                opt_const_new
            ]
        
        def _accept_new_loop_vars():
            debug_print("Rejected new step")
            return [
                i + 1, 
                xs, 
                us, 
                (hs, hst, gs, gst), 
                (lagrange_variables, terminal_lagrange_variables),
                cost_prev, 
                (j_curr, m_curr, inf_curr),
                value_functions_iterates,
                opt_const_new
            ]
        
        return jax.lax.cond(alpha > tol, accept_new_loop_vars, _accept_new_loop_vars)

    loop_vars = (
        0, 
        xs_iterates[0], 
        us_iterates[0], 
        (hs_iterates[0], hst_iterates[0], gs_iterates[0], gst_iterates[0]), 
        (lagrange_iterates[0], terminal_lagrange_iterates[0]), 
        (j_curr, m_curr, inf_curr), 
        (jnp.inf, jnp.inf, jnp.inf), 
        value_functions_iterates, 
        opt_const_iterates[0]
    )

    debug_print("Start of optimization -------------------------------------------------------------------------------------------")
    while continuation_criterion(loop_vars):
        loop_vars = ilqr_iteration(loop_vars)

        # append new values
        (xs_iterates, us_iterates, 
            (hs_iterates, hst_iterates, gs_iterates, gst_iterates), (lagrange_iterates, terminal_lagrange_iterates), 
            cost_iterates, opt_const_iterates) = tree_util.tree_map(lambda x, v : jnp.vstack((x, v[jnp.newaxis])), 
                                                        (xs_iterates, us_iterates, (hs_iterates, hst_iterates, gs_iterates, gst_iterates), (lagrange_iterates, terminal_lagrange_iterates), cost_iterates, opt_const_iterates), 
                                                        (*loop_vars[1:6], loop_vars[-1]))

    debug_print("End of optimization -------------------------------------------------------------------------------------------")
    return {
        "optimal_trajectory" : (xs_iterates[-1], us_iterates[-1], (hs_iterates[-1], hst_iterates[-1], gs_iterates[-1], gst_iterates[-1]), (lagrange_iterates[-1], terminal_lagrange_iterates[-1])),
        "optimal_cost" : tree_util.tree_map(lambda x : x[-1], cost_iterates),
        "num_iterations" : loop_vars[0],
        "trajectory_iterates" : (xs_iterates, us_iterates, (hs_iterates, hst_iterates, gs_iterates, gst_iterates), (lagrange_iterates, terminal_lagrange_iterates)),
        "cost_iterates" : cost_iterates,
        "values_functions_iterates" : value_functions_iterates,
        "optimization_constants" : opt_const_iterates
    }


############################################################################################################
