import os
from typing import NamedTuple, Callable
from pprint import pformat
import logging
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)
from datetime import datetime
import argparse

import jax
jax.config.update("jax_enable_x64", True)
import jax.numpy as jnp
from jax import flatten_util, tree_util
import jax.random as jrandom
from cyipopt import Problem
import matplotlib.pyplot as plt
from matplotlib.patches import Circle

from utils import check_dir, RK4Integrator


parser = argparse.ArgumentParser("MultipleShootingIPOPT")
parser.add_argument("--skip_disp", type = int, choices = [0, 1], default = 1, help = "Skip displacement test ?") 
parser.add_argument("--skip_pend", type = int, choices = [0, 1], default = 1, help = "Skip inverted pendulum test ?") 
parser.add_argument("--skip_unicycle", type = int, choices = [0, 1], default = 1, help = "Skip unicycle test ?") 
parser.add_argument("--skip_parking", type = int, choices = [0, 1], default = 1, help = "Skip car parking test ?") 
parser.add_argument("--skip_obstacle", type = int, choices = [0, 1], default = 1, help = "Skip obstacle test ?") 
parser.add_argument("--skip_cstr", type = int, choices = [0, 1], default = 1, help = "Skip CSTR test ?") 
parser.add_argument("--sparse", type = int, default = 0, choices = [0, 1], help = "Exploit sparsity or not ?")
parser.add_argument("--nfactor", type = int, default = 1, help = "Multiplicative factor of control horizon")
parser.add_argument("--iters", type = int, default = 3000, help = "The maximum ipopt iterations")
parser.add_argument("--tol", type = float, default = 1e-4, help = "Ipopt tolerance")
parser.add_argument("--msg", type = str, default = "", help = "Message")

parser.add_argument("--id", type = str, default = "", help = "Slurm job id")
parser.add_argument("--partition", type = str, default = "", help = "The partition this job is assigned to")
parser.add_argument("--cpus", type = str, default = "", help = "Maximum number of cpus availabe per node")

pargs = parser.parse_args()

_dir = os.path.join("log", "ilqr", str(datetime.now()))
check_dir(_dir)
logfile = logging.FileHandler(os.path.join(_dir, "solver_stats.txt"))
logger.addHandler(logfile)

divider = "--"*50
provide_structure = pargs.sparse
_output_file = os.path.join(_dir, "ipopt_output.txt")
logger.info(pformat(pargs.__dict__))
logger.info(divider)


def sparse_indices(afunc, input_len, output_len, *args):
    # Get sparsity pattern of the output of the function via sampling
    samples = jrandom.normal(key, shape = (10, input_len))
    nonzero_indices = jax.vmap(lambda z : jnp.nonzero(afunc(z, *args), size = output_len, fill_value = -1))(samples)
    nonzero_indices_stacked = jnp.column_stack([*map(lambda z : z.flatten(), nonzero_indices)])
    return tuple([*jnp.unique(nonzero_indices_stacked, axis = 0)[1:].T]) # removes the (-1, -1) index

def solve_nlp(
    obj : Callable, # Objective function
    obj_con : Callable, # Constraints (equality + inequality)
    xu_guess : jnp.ndarray, # Initial guess of variables
    v_lb : jnp.ndarray, # Variable lower bound
    v_ub : jnp.ndarray, # Variable upper bound
    c_lb : jnp.ndarray, # Constraint lower bound
    c_ub : jnp.ndarray # Constraint upper bound
    ) : 
    
    @jax.jit
    def Lagrangian(xu, lam, c) : return c * obj(xu) + jnp.asarray(lam) @ obj_con(xu)

    nvar = len(xu_guess)
    ncon = len(obj_con(xu_guess))

    obj_grad = jax.jit(jax.grad(obj))
    con_jac = jax.jit(jax.jacobian(obj_con))
    Lag_hess = jax.jit(jax.hessian(Lagrangian))

    # TODO Find cheaper alternative for getting sparsity pattern
    # Make sure that numerical zeros dont end up as sparse zeros
    jacobian_sparse_indices = sparse_indices(con_jac, nvar, ncon * nvar)

    _hessian = Lag_hess(xu_guess, jnp.ones(ncon), 1) # hessian of Lagrangian
    hessian_sparse_indices = sparse_indices(Lag_hess, nvar, nvar * nvar, jnp.ones(ncon), 1)
    hessian_dense_indices = jnp.nonzero(jnp.tril(jnp.ones_like(_hessian)))

    class DisplacementExampleProblem():

        def objective(self, x) : return obj(x)
        def gradient(self, x) : return obj_grad(x)

        def constraints(self, x):
            # All equality and inequality constraints
            return obj_con(x)

        def jacobian(self, x):
            if provide_structure :
                return con_jac(x)[jacobian_sparse_indices]
            else :
                return con_jac(x)

        if provide_structure :
            # Structure of Jacobian of constaraints
            def jacobianstructure(self) : return jacobian_sparse_indices
        
        def hessian(self, x, lam, obj_factor):
            _indices = self.hessianstructure()
            return Lag_hess(x, lam, obj_factor)[_indices]

        def hessianstructure(self):
            # Structure of Hessian of Lagrangian
            if provide_structure :
                return hessian_sparse_indices
            else :
                return hessian_dense_indices


    nlp = Problem(
        n = nvar, # Number of variables
        m = ncon, # Number of constraints
        problem_obj = DisplacementExampleProblem(),
        lb = v_lb, # Lower bound on variables
        ub = v_ub, # Upper bound on variables
        cl = c_lb, # Lower bound on constraints
        cu = c_ub # Upper bound on constraints
    )

    options = {
        "tol" : pargs.tol,
        "max_iter" : pargs.iters,
        "output_file" : _output_file, 
        "file_print_level" : 5, 
        "mu_strategy" : "adaptive",
        "print_timing_statistics" : "yes"   
    }

    for _key, _value in options.items() : nlp.add_option(_key, _value)

    xu_opt, info = nlp.solve(xu_guess)

    logger.info(f"{divider}")
    with open(_output_file, "r") as file:
        for line in file : logger.info(line.rstrip())
    logger.info(f"{divider}")

    os.remove(_output_file)
    return xu_opt

def plot_results(xs : jnp.ndarray, us : jnp.ndarray):
    
    _, nu = us.shape
    _, nx = xs.shape

    with plt.style.context(["science", "notebook", "bright"]):
        fig, ax = plt.subplots(1, 2, figsize = (30, 15))
        
        # plotting optimal control inputs
        ax[1].plot(us, "o")
        ax[1].set(xlabel = "Horizon", ylabel = "controls")
        ax[1].legend([f"u{i}" for i in range(nu)])

        # plotting optimal states
        ax[0].plot(xs, "o")
        ax[0].set(xlabel = "Horizon", ylabel = "States")
        ax[0].legend([f"x{i}" for i in range(nx)])

    return ax

def rollout_state_feedback_policy(dynamics : Callable, policy : jnp.ndarray, x0 : jnp.ndarray):

    def body_fun(x, u):
        xi = dynamics(x, u)
        return xi, (xi, u)

    _, (xs, us) = jax.lax.scan(body_fun, x0, policy)
    return xs, us


class TotalCost(NamedTuple):
    running_cost : Callable
    terminal_cost : Callable

    @classmethod
    def form_cost(cls, running_cost, terminal_cost = None):

        class NoCostScl(NamedTuple):
            # No cost returns a empty scalar
            def __call__(self, x, *args):
                return jnp.sum(jnp.array([]))

        return cls(
            running_cost, 
            NoCostScl() if terminal_cost is None else terminal_cost, 
        )
    
    def __call__(self, xs, us):
        _running_cost = lambda x, u : self.running_cost(x, u) 
        total_running_cost = jnp.sum(jax.vmap(_running_cost)(xs[:-1], us))
        total_terminal_cost = self.terminal_cost(xs[-1]) 
        return total_running_cost + total_terminal_cost


if not pargs.skip_disp :

    logger.info("Started displacement test -------------------------------------------------------------------------------------------")

    class DisplacementExampleDynamics(NamedTuple):
        dt : float = 0.1
        m : float = 2.

        def __call__(self, x, u, k = None):
            return jnp.array([
                x[0] + self.dt * x[1],
                x[1] + self.dt * u[0]/self.m 
            ])


    class DisplacementExampleRunningCost(NamedTuple):
        Q : jnp.array = jnp.diag(jnp.array([1., 0.1]))
        R : jnp.array = jnp.diag(jnp.array([0.1]))

        def __call__(self, x, u):
            return (x - jnp.array([10., 0])) @ self.Q @ (x - jnp.array([10., 0])) + u @ self.R @ u
        

    class DisplacementExampleInequalityConstraints(NamedTuple):
        # inequality constraints of the form h(x, u) for each stage

        def __call__(self, x, u):
            return jnp.array([
                u[0]
            ])


    class DisplacementExampleTerminalEqualityConstraints(NamedTuple):
        # terminal equality constraints of the form g(x) = 0 for each stage

        def __call__(self, x):
            return jnp.array([
                x[0] - 10.,
                x[1]
            ])


    class DisplacementExampleDynamicEqualityConstraints(NamedTuple):
        dynamics : Callable

        def __call__(self, xs, us):
            return 


    N = 150 * pargs.nfactor
    dt = 0.1 / pargs.nfactor
    x0 = jnp.array([0., 0.]) # Is given
    key = jrandom.PRNGKey(seed = 10)
    u_guess = 0.001 * jrandom.randint(key, (N, 1), minval = -10, maxval = 10)
    _dynamics = DisplacementExampleDynamics(dt)
    x_guess, _ = rollout_state_feedback_policy(_dynamics, u_guess, x0)
    xu_guess, unravel = flatten_util.ravel_pytree((x_guess, u_guess))
    _total_cost = TotalCost.form_cost(DisplacementExampleRunningCost())

    @jax.jit
    def obj(xu): 
        xs, us = unravel(xu)
        xs = jnp.vstack((x0, xs))
        return _total_cost(xs, us)

    @jax.jit
    def obj_con(xu): 
        xs, us = unravel(xu)
        xs = jnp.vstack((x0, xs))
        return jnp.concatenate((
            jax.vmap(lambda _xsnext, _xs, _us : _xsnext - _dynamics(_xs, _us))(xs[1:], xs[:-1], us).flatten(),
            DisplacementExampleTerminalEqualityConstraints()(xs[-1]).flatten()
        ))

    nvar = len(xu_guess)
    ncon = len(obj_con(xu_guess))

    xu_opt = solve_nlp(
        obj, obj_con, xu_guess, 
        flatten_util.ravel_pytree((- jnp.inf * jnp.ones_like(x_guess), -2 * jnp.ones_like(u_guess)))[0], 
        flatten_util.ravel_pytree(( jnp.inf * jnp.ones_like(x_guess), 2 * jnp.ones_like(u_guess)))[0], 
        jnp.zeros(ncon), 
        jnp.zeros(ncon)  
    )

    x_opt, u_opt = unravel(xu_opt)
    ax = plot_results(jnp.vstack((x0, x_opt)), u_opt)
    plt.savefig(os.path.join(_dir, "ipopt_displacement"))
    plt.close()

if not pargs.skip_pend :
    
    # inverted pendulum example from https://ieeexplore.ieee.org/stamp/stamp.jsp?arnumber=9332234
    logger.info("Started inverted pendulum test -------------------------------------------------------------------------------------------")

    class PendulumExampleDynamics(NamedTuple):
        dt: float = 0.05

        def __call__(self, x, u, k = None):
            return jnp.array([
                x[0] + self.dt * x[1], 
                x[1] + self.dt * jnp.sin(x[0]) + self.dt * u[0]
                ])


    class PendulumExampleRunningCost(NamedTuple):
        q : float = 0.025
        r : float = 0.025

        def __call__(self, x, u, k = None):
            return  self.q * jnp.sum(x**2) + self.r * jnp.sum(u**2) # scalar


    class PendulumExampleTerminalCost(NamedTuple):
        gain: float = 5.
        target: jnp.array = jnp.array([0., 0.])

        def __call__(self, x):
            return self.gain * jnp.sum(jnp.square(x - self.target)) # scalar


    class PendulumExampleInequalityConstraints(NamedTuple):
        # running inequality constraints of the form h(x, u) <= 0

        def __call__(self, x, u, k = None):
            return jnp.array([
                u[0]
            ])

    
    class PendulumExampleTerminalEqualityConstraints(NamedTuple):
        # terminal equality constraints of the form g(x) = 0

        def __call__(self, x):
            return jnp.array([
                x[0],
                x[1]
            ])

    N = 500 * pargs.nfactor
    dt = 0.05 / pargs.nfactor
    x0 = jnp.array([-jnp.pi, 0.])
    key = jrandom.PRNGKey(seed = 5)
    u_guess = 0.001*jrandom.randint(key, (N, 1), minval = -10, maxval = 10)
    _dynamics = PendulumExampleDynamics(dt)
    x_guess, _ = rollout_state_feedback_policy(_dynamics, u_guess, x0)
    xu_guess, unravel = flatten_util.ravel_pytree((x_guess, u_guess))
    _total_cost = TotalCost.form_cost(PendulumExampleRunningCost())

    @jax.jit
    def obj(xu): 
        xs, us = unravel(xu)
        xs = jnp.vstack((x0, xs))
        return _total_cost(xs, us)

    @jax.jit
    def obj_con(xu): 
        xs, us = unravel(xu)
        xs = jnp.vstack((x0, xs))
        return jnp.concatenate((
            jax.vmap(lambda _xsnext, _xs, _us : _xsnext - _dynamics(_xs, _us))(xs[1:], xs[:-1], us).flatten(),
            PendulumExampleTerminalEqualityConstraints()(xs[-1]).flatten()
        ))

    nvar = len(xu_guess)
    ncon = len(obj_con(xu_guess))

    xu_opt = solve_nlp(
        obj, obj_con, xu_guess, 
        flatten_util.ravel_pytree((- jnp.inf * jnp.ones_like(x_guess), - 0.25 * jnp.ones_like(u_guess)))[0], 
        flatten_util.ravel_pytree(( jnp.inf * jnp.ones_like(x_guess), 0.25 * jnp.ones_like(u_guess)))[0], 
        jnp.zeros(ncon), 
        jnp.zeros(ncon)  
    )

    x_opt, u_opt = unravel(xu_opt)
    ax = plot_results(jnp.vstack((x0, x_opt)), u_opt)
    plt.savefig(os.path.join(_dir, "ipopt_pendulum"))
    plt.close()

if not pargs.skip_unicycle :

    # Unicycle motion control example from https://ieeexplore.ieee.org/stamp/stamp.jsp?arnumber=9332234
    logger.info("Started unicycle test -------------------------------------------------------------------------------------------")

    class UnicycleExampleDynamics(NamedTuple):
        dt : float = 0.01
        v : float = 1.5

        def __call__(self, x, u, k = None):
            
            return jnp.array([
                x[0] + self.dt * self.v * jnp.cos(x[2]),
                x[1] + self.dt * self.v * jnp.sin(x[2]),
                x[2] + self.dt * u[0]
            ])


    class UnicycleExampleRunningCost(NamedTuple):
        q : jnp.ndarray = 0.1 * jnp.eye(3)
        r : jnp.ndarray = 0.01 * jnp.eye(1)

        def __call__(self, x, u, k = None):
            return x.T @ self.q @ x + u.T @ self.r @ u


    class UnicycleExampleInequalityConstraints(NamedTuple):
        # inequality constraints of the form h(x, u) <= 0

        def __call__(self, x, u, k = None):

            return jnp.array([ 
                -(x[0] + 5.5)**2 - (x[1] + 1)**2 + 1**2,
                -(x[0] + 8)**2 - (x[1] - 0.2)**2 + 0.5**2,
                -(x[0] + 2.5)**2 - (x[1] - 1)**2 + 1.5**2,
            ])


    class UnicycleExampleTerminalCost(NamedTuple):
        q : jnp.ndarray = 0.1 * jnp.eye(3)

        def __call__(self, x):
            return x.T @ self.q @ x


    N = 650 * pargs.nfactor
    dt = 0.01 / pargs.nfactor
    x0 = jnp.array([-10, 0., 0])
    key = jrandom.PRNGKey(seed = 10)
    u_guess = 0.001*jrandom.randint(key, (N, 1), minval = -10, maxval = 10)
    _dynamics = UnicycleExampleDynamics(dt)
    x_guess, _ = rollout_state_feedback_policy(_dynamics, u_guess, x0)
    xu_guess, unravel = flatten_util.ravel_pytree((x_guess, u_guess))
    _total_cost = TotalCost.form_cost(UnicycleExampleRunningCost(), UnicycleExampleTerminalCost())

    @jax.jit
    def obj(xu): 
        xs, us = unravel(xu)
        xs = jnp.vstack((x0, xs))
        return _total_cost(xs, us)

    @jax.jit
    def obj_con(xu): 
        xs, us = unravel(xu)
        xs = jnp.vstack((x0, xs))
        return jnp.concatenate((
            jax.vmap(lambda _xsnext, _xs, _us : _xsnext - _dynamics(_xs, _us))(xs[1:], xs[:-1], us).flatten(),
            jax.vmap(UnicycleExampleInequalityConstraints())(xs[1:], us).flatten()
        ))

    nvar = len(xu_guess)
    ncon = len(obj_con(xu_guess))

    xu_opt = solve_nlp(
        obj, obj_con, xu_guess, 
        flatten_util.ravel_pytree(( jnp.tile(jnp.array([-jnp.inf, -1, -jnp.inf]), (N, 1)), - 1.5 * jnp.ones_like(u_guess)))[0], 
        flatten_util.ravel_pytree(( jnp.tile(jnp.array([jnp.inf, 1, jnp.inf]), (N, 1)), 1.5 * jnp.ones_like(u_guess)))[0], 
        jnp.concatenate((jnp.zeros(3 * N), - jnp.inf * jnp.zeros(3 * N))), 
        jnp.zeros(ncon)
    )

    x_opt, u_opt = unravel(xu_opt)
    ax = plot_results(jnp.vstack((x0, x_opt)), u_opt)
    
    # plotting optimal states
    with plt.style.context(["science", "notebook", "bright"]):
        ax[0].clear()
        ax[0].plot(x_opt[:, 0], x_opt[:, 1], "o")
        ax[0].set(xlabel = "Position x", ylabel = "Position y")

        circles = [Circle((-5.5, -1), 1, color = "k"), Circle((-8, 0.2), 0.5, color = "k"), Circle((-2.5, 1), 1.5, color = "k")]
        for cir in circles :
            ax[0].add_patch(cir)
    
    plt.savefig(os.path.join(_dir, "ipopt_unicycle"))
    plt.close()

if not pargs.skip_parking :

    # Car parking example from https://ieeexplore.ieee.org/stamp/stamp.jsp?arnumber=9332234
    # https://homes.cs.washington.edu/~todorov/papers/TassaICRA14.pdf
    # https://www.mathworks.com/matlabcentral/fileexchange/52069-ilqg-ddp-trajectory-optimization
    logger.info("Started car parking test -------------------------------------------------------------------------------------------")

    class CarParkingExampleDynamics(NamedTuple):
        dt: float = 0.03
        d : float = 2.

        def __call__(self, x, u, k = None):
            b = lambda v, w : self.d + self.dt * v * jnp.cos(w) - jnp.sqrt(self.d**2 - self.dt**2 * v**2 * jnp.sin(w)**2)

            return jnp.array([
                x[0] + b(x[3], u[0]) * jnp.cos(x[2]), 
                x[1] + b(x[3], u[0]) * jnp.sin(x[2]),
                x[2] + jnp.arcsin(self.dt * x[3] * jnp.sin(u[0]) / self.d),
                x[3] + self.dt * u[1]
                ])


    class CarParkingExampleRunningCost(NamedTuple):
        H : Callable = lambda y, z : jnp.sqrt(y**2 + z**2) - z

        def __call__(self, x, u, k = None):
            return 0.001*(self.H(x[0], 0.1) + self.H(x[1], 0.1) + 10*u[0]**2 + 0.1*u[1]**2)


    class CarParkingExampleTerminalCost(NamedTuple):
        H : Callable = lambda y, z : jnp.sqrt(y**2 + z**2) - z

        def __call__(self, x):
            return 0.1*self.H(x[0], 0.01) + 0.1*self.H(x[1], 0.01) + self.H(x[2], 0.01) + 0.3*self.H(x[3], 1)


    class CarParkingExampleInequalityConstraints(NamedTuple):
        # inequality constraints of the form h(x, u) <= 0

        def __call__(self, x, u, k = None):
            return jnp.array([
                u[0] - 0.5,
                -0.5 - u[0],
                u[1] - 2, 
                -2 - u[1],
            ])

    N = 500 * pargs.nfactor
    dt = 0.03 / pargs.nfactor
    x0 = jnp.array([1, 1, 3*jnp.pi / 2, 0.])
    key = jrandom.PRNGKey(seed = 30)
    u_guess = 0.1*jrandom.normal(key, (N, 2))
    _dynamics = CarParkingExampleDynamics(dt)
    x_guess, _ = rollout_state_feedback_policy(_dynamics, u_guess, x0)
    xu_guess, unravel = flatten_util.ravel_pytree((x_guess, u_guess))
    _total_cost = TotalCost.form_cost(CarParkingExampleRunningCost(), CarParkingExampleTerminalCost())

    @jax.jit
    def obj(xu): 
        xs, us = unravel(xu)
        xs = jnp.vstack((x0, xs))
        return _total_cost(xs, us)

    @jax.jit
    def obj_con(xu): 
        xs, us = unravel(xu)
        xs = jnp.vstack((x0, xs))
        return jax.vmap(lambda _xsnext, _xs, _us : _xsnext - _dynamics(_xs, _us))(xs[1:], xs[:-1], us).flatten()

    nvar = len(xu_guess)
    ncon = len(obj_con(xu_guess))

    xu_opt = solve_nlp(
        obj, obj_con, xu_guess, 
        flatten_util.ravel_pytree(( -jnp.inf * jnp.zeros(N * 4), - jnp.tile(jnp.array([0.5, 2]), (N, 1)) ))[0], 
        flatten_util.ravel_pytree(( jnp.inf * jnp.zeros(N * 4), jnp.tile(jnp.array([0.5, 2]), (N, 1)) ))[0], 
        jnp.zeros(ncon), 
        jnp.zeros(ncon)
    )

    x_opt, u_opt = unravel(xu_opt)
    ax = plot_results(jnp.vstack((x0, x_opt)), u_opt)
    plt.savefig(os.path.join(_dir, "ipopt_car_parking"))
    plt.close()

if not pargs.skip_obstacle : 
    
    # https://github.com/ZhaomingXie/CDDP/blob/master/optimize_car.py
    # https://arxiv.org/pdf/2005.00985
    logger.info("Started car obstacle test -------------------------------------------------------------------------------------------")

    class CarObstacleExampleDynamics(NamedTuple):
        dt: float = 0.05
        
        def __call__(self, x, u, k = None):
            
            return jnp.array([
                x[0] + self.dt * x[3] * jnp.sin(x[2]), 
                x[1] + self.dt * x[3] * jnp.cos(x[2]),
                x[2] + self.dt * u[1] * x[3],
                x[3] + self.dt * u[0]
                ])


    class CarObstacleExampleRunningCost(NamedTuple):
        q : jnp.ndarray = 0 * jnp.eye(4)
        r : jnp.ndarray = 0.05 * jnp.eye(2)

        def __call__(self, x, u, k = None):
            return x.T @ self.q @ x + u.T @ self.r @ u


    class CarObstacleExampleTerminalCost(NamedTuple):
        q : jnp.ndarray = jnp.diag(jnp.array([50, 50, 50, 10.]))
        target : jnp.ndarray = jnp.array([3, 3, jnp.pi / 2, 0.])

        def __call__(self, x):
            _x = x - self.target
            return _x.T @ self.q @ _x


    class CarObstacleExampleInequalityConstraints(NamedTuple):
        # inequality constraints of the form h(x, u) <= 0

        def __call__(self, x, u, k = None):
            return jnp.array([
                0.5**2 - (x[0] - 1)**2 - (x[1] - 1)**2,
                0.5**2 - (x[0] - 1)**2 - (x[1] - 2.5)**2,
                0.5**2 - (x[0] - 2.5)**2 - (x[1] - 2.5)**2
            ])


    N = 200 * pargs.nfactor
    dt = 0.05 / pargs.nfactor
    x0 = jnp.array([0., 0., 0., 0.])
    key = jrandom.PRNGKey(seed = 40)
    u_guess = 0.001 * jrandom.randint(key, (N, 2), minval = -10, maxval = 10)
    _dynamics = CarObstacleExampleDynamics(dt)
    x_guess, _ = rollout_state_feedback_policy(_dynamics, u_guess, x0)
    xu_guess, unravel = flatten_util.ravel_pytree((x_guess, u_guess))
    _total_cost = TotalCost.form_cost(CarObstacleExampleRunningCost(), CarObstacleExampleTerminalCost())

    @jax.jit
    def obj(xu): 
        xs, us = unravel(xu)
        xs = jnp.vstack((x0, xs))
        return _total_cost(xs, us)

    @jax.jit
    def obj_con(xu): 
        xs, us = unravel(xu)
        xs = jnp.vstack((x0, xs))
        return jnp.concatenate((
            jax.vmap(lambda _xsnext, _xs, _us : _xsnext - _dynamics(_xs, _us))(xs[1:], xs[:-1], us).flatten(),
            jax.vmap(CarObstacleExampleInequalityConstraints())(xs[1:], us).flatten()
        ))

    nvar = len(xu_guess)
    ncon = len(obj_con(xu_guess))

    xu_opt = solve_nlp(
        obj, obj_con, xu_guess, 
        flatten_util.ravel_pytree(( - jnp.tile(jnp.array([jnp.inf, jnp.inf, jnp.inf, jnp.inf]), (N, 1)), - jnp.tile(jnp.array([jnp.pi / 2, 10]), (N, 1)) ))[0], 
        flatten_util.ravel_pytree(( jnp.tile(jnp.array([jnp.inf, jnp.inf, jnp.inf, jnp.inf]), (N, 1)), jnp.tile(jnp.array([jnp.pi / 2, 10]), (N, 1)) ))[0], 
        jnp.concatenate((jnp.zeros(4 * N), - jnp.inf * jnp.zeros(3 * N))), 
        jnp.zeros(ncon)
    )

    x_opt, u_opt = unravel(xu_opt)
    ax = plot_results(jnp.vstack((x0, x_opt)), u_opt)
    
    # plotting optimal states
    with plt.style.context(["science", "notebook", "bright"]):
        ax[0].clear()
        ax[0].plot(x_opt[:, 0], x_opt[:, 1], "o")
        ax[0].set(xlabel = "Position x", ylabel = "Position y")

        circles = [Circle((1, 1), 0.5, color = "k"), Circle((1, 2.5), 0.5, color = "k"), Circle((2.5, 2.5), .5, color = "k")]
        for cir in circles :
            ax[0].add_patch(cir)

    plt.savefig(os.path.join(_dir, "ipopt_car_obstacle"))
    plt.close()

if not pargs.skip_cstr :
    
    # cstr example from https://jckantor.github.io/CBE30338/04.11-Implementing-PID-Control-in-Nonlinear-Simulations.html
    # other similar examples can be taken from https://www.do-mpc.com/en/latest/example_gallery/CSTR.html
    logger.info("Started cstr test -------------------------------------------------------------------------------------------")

    class CstrExampleDynamics(NamedTuple):
        Ea : float = 72750    # activation energy J/gmol
        R : float = 8.314     # gas constant J/gmol/K
        K0 : float = 7.2e10   # Arrhenius rate constant 1/min
        V : float = 100.0     # Volume [L]
        D : float = 1000.0  # Density [g/L]
        Cp : float = 0.239    # Heat capacity [J/g/K]
        H : float = -5.0e4  # Enthalpy of reaction [J/mol]
        Ua : float = 5.0e4    # Heat transfer [J/min/K]
        Q : float = 100.0     # Flowrate [L/min]
        Cf : float = 1.0      # Inlet feed concentration [mol/L]
        Tf : float = 300.0    # Inlet feed temperature [K]
        Tcf : float = 300.0   # Coolant feed temperature [K]
        Vc : float = 20.0     # Cooling jacket volume
        
        def __call__(self, x, u, k = None):
            # x[0] reaction concentration [mol/L]
            # x[1] reaction temperature [K]
            # x[2] coolant temperature [K]
            # u[0] coolant flowrate [L/min]

            reaction_rate = self.K0 * jnp.exp(-self.Ea / self.R / x[1])*x[0]

            return jnp.array([
                (self.Q / self.V) * (self.Cf - x[0]) - reaction_rate,
                (self.Q / self.V) * (self.Tf - x[1]) + (- self.H / self.D / self.Cp) * reaction_rate + (self.Ua / self.V / self.D / self.Cp) * (x[2] - x[1]),
                (u[0] / self.Vc) * (self.Tcf - x[2]) + (self.Ua / self.Vc / self.D / self.Cp) * (x[1] - x[2])
                ])


    class CstrExampleRunningCost(NamedTuple):
        q : jnp.ndarray = jnp.diag(jnp.array([0., 1., 0.]))
        r : jnp.ndarray = 0.0 * jnp.eye(1)

        def __call__(self, x, u, k = None):
            return (x - jnp.array([0., 390, 0.])).T @ self.q @ (x - jnp.array([0., 390, 0.])) + u.T @ self.r @ u


    class CstrExampleTerminalCost(NamedTuple):
        q : jnp.ndarray = jnp.diag(jnp.array([0., 50., 0.]))
        target : jnp.ndarray = jnp.array([0, 390, 0.])

        def __call__(self, x):
            return (x - self.target).T @ self.q @ (x - self.target)


    class CstrExampleInequalityConstraints(NamedTuple):
        # inequality constraints of the form h(x, u) <= 0

        def __call__(self, x, u, k = None):
            return jnp.array([
                u[0] - 300,
                - u[0],
            ])


    class CstrExampleTerminalEqualityConstraints(NamedTuple):
        # terminal equality constraints of the form g(x) = 0
        target : jnp.ndarray = jnp.array([390. ])

        def __call__(self, x):
            return jnp.array([
                x[1] - self.target[0],
            ])
    
    
    N = 400 * pargs.nfactor
    dt = 0.01 / pargs.nfactor
    x0 = jnp.array([0.5, 350, 300.])
    key = jrandom.PRNGKey(seed = 40)
    u_guess = 150 * jnp.ones(shape = (N, 1))
    _dynamics = RK4Integrator(CstrExampleDynamics(), dt)
    x_guess, _ = rollout_state_feedback_policy(_dynamics, u_guess, x0)
    xu_guess, unravel = flatten_util.ravel_pytree((x_guess, u_guess))
    _total_cost = TotalCost.form_cost(CstrExampleRunningCost())

    @jax.jit
    def obj(xu): 
        xs, us = unravel(xu)
        xs = jnp.vstack((x0, xs))
        return _total_cost(xs, us)

    @jax.jit
    def obj_con(xu): 
        xs, us = unravel(xu)
        xs = jnp.vstack((x0, xs))
        return jnp.concatenate((
            jax.vmap(lambda _xsnext, _xs, _us : _xsnext - _dynamics(_xs, _us))(xs[1:], xs[:-1], us).flatten(),
            CstrExampleTerminalEqualityConstraints()(xs[-1])
        ))

    nvar = len(xu_guess)
    ncon = len(obj_con(xu_guess))

    xu_opt = solve_nlp(
        obj, obj_con, xu_guess, 
        flatten_util.ravel_pytree(( - jnp.tile(jnp.array([jnp.inf, jnp.inf, jnp.inf]), (N, 1)), jnp.zeros(N) ))[0], 
        flatten_util.ravel_pytree(( jnp.tile(jnp.array([jnp.inf, jnp.inf, jnp.inf]), (N, 1)), 300 * jnp.ones(N) ))[0], 
        jnp.zeros(ncon), 
        jnp.zeros(ncon)
    )

    x_opt, u_opt = unravel(xu_opt)
    ax = plot_results(jnp.vstack((x0, x_opt)), u_opt)

    # plotting only temperature
    with plt.style.context(["science", "notebook", "bright"]):
        ax[0].clear()
        ax[0].plot(x_opt[:, 1], "o")
        ax[0].set(xlabel = "Horizon", ylabel = "Temperature")

    plt.savefig(os.path.join(_dir, "ipopt_cstr"))
    plt.close()
