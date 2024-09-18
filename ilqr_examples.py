from typing import NamedTuple, Callable
import os

import jax
jax.config.update("jax_enable_x64", True)
import jax.numpy as jnp
import jax.random as jrandom
import matplotlib.pyplot as plt
from matplotlib.patches import Circle
import scienceplots

from ilqr import iterative_linear_quadratic_regulator, TotalCost, _dir, logger


skip_disp_test = False # skip displacement test ?
skip_pend_test = False # skip inverted pendulum test ?
skip_unicycle_test = False # skip obstacle test ?
skip_parking_test = False # skip car parking test ?
skip_obstacle_test = False # skip car obstacle test ?
skip_quadrotor_test = True # skip quadrotor test ?
skip_cstr_test = False # skip cstr test ?

consider_equality = False # solve equality constraint optimization ? (default = inequality)
atol = 1e-4 # tolerance
approx_hessian = False


msg = f"testing all with equality constraints as {consider_equality} and {atol} tolerance, with approx_hessian {approx_hessian}" # description of experiment
logger.info(f"MSG : {msg}")

def plot_results(solution : dict):
    
    with plt.style.context(["science", "notebook", "bright"]):
        fig, ax = plt.subplots(3, 3, figsize = (30, 15))
        # plotting cost
        ax[0, 0].plot(solution["cost_iterates"][0], "-") # objective 
        ax[0, 0].plot(solution["cost_iterates"][1], "o") # merit 
        ax[0, 0].set(yscale = "symlog", xlabel = "Iterations", ylabel = "Cost")
        ax[0, 0].legend(["obj", "merit"])
        
        # plotting optimal control inputs
        opt_inputs = solution["optimal_trajectory"][1]
        ax[0, 1].plot(opt_inputs, "o")
        ax[0, 1].set(xlabel = "Horizon", ylabel = "controls")
        ax[0, 1].legend([f"u{i}" for i in range(opt_inputs.shape[-1])])

        # plotting optimal states
        opt_traj = solution["optimal_trajectory"][0]
        ax[0, 2].plot(opt_traj, "o")
        ax[0, 2].set(xlabel = "Horizon", ylabel = "States")
        ax[0, 2].legend([f"x{i}" for i in range(opt_traj.shape[-1])])
        
        # plotting inequality constraint infeasibility
        ax[1, 0].plot(solution["cost_iterates"][2], "o-")
        ax[1, 0].set(yscale = "log", xlabel = "Iterations", ylabel = "Constraint infeasibility")
        
        # plotting alpha used in backtracking
        ax[1, 1].plot(solution["optimization_constants"].alpha, "o-")
        ax[1, 1].set(yscale = "log", xlabel = "Iterations", ylabel = "alpha")
        
        # plotting penalty parameter
        ax[1, 2].plot(solution["optimization_constants"].tau, "o")
        ax[1, 2].set(yscale = "log", xlabel = "Iterations", ylabel = "tau")

        # plotting regularization parameter
        ax[2, 0].plot(solution["optimization_constants"].reg[:, 0], "o")
        ax[2, 0].set(yscale = "log", xlabel = "Iterations", ylabel = "regularization")

    return ax


class RK4Integrator(NamedTuple):
    ode : Callable
    dt : float

    def __call__(self, x, u, k):
        k1 = self.dt * self.ode(x, u)
        k2 = self.dt * self.ode(x + k1/2, u)
        k3 = self.dt * self.ode(x + k2/2, u)
        k4 = self.dt * self.ode(x + k3, u)
        return x + (k1 + 2*k3 + 2*k3 + k4)/6
    

if not skip_disp_test : # and consider_equality : 
    
    # simple example from https://github.com/Bharath2/iLQR/tree/main
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

        def __call__(self, x, u, k = None):
            return (x - jnp.array([10., 0])) @ self.Q @ (x - jnp.array([10., 0])) + u @ self.R @ u
        

    class DisplacementExampleInequalityConstraints(NamedTuple):
        # inequality constraints of the form h(x, u) <= 0

        def __call__(self, x, u, k = None):
            return jnp.array([
                u[0] - 2,
                -2 - u[0]
            ])


    class DisplacementExampleTerminalCost(NamedTuple):
        gain: jnp.array = jnp.diag(jnp.array([10., 10.]))
        target: jnp.array = jnp.array([10., 0.])

        def __call__(self, x):
            return (x - self.target) @ self.gain @ (x - self.target) 


    class DisplacementExampleTerminalEqualityConstraints(NamedTuple):
        # terminal equality constraints of the form g(x) = 0

        def __call__(self, x):
            return jnp.array([
                x[0] - 10.,
                x[1]
            ])
        

    x0 = jnp.array([0., 0.])
    key = jrandom.PRNGKey(seed = 10)
    u_guess = 0.001*jrandom.randint(key, (150, 1), minval = -10, maxval = 10)
    solution = iterative_linear_quadratic_regulator(
            DisplacementExampleDynamics(), 
            TotalCost.form_cost(
                DisplacementExampleRunningCost(), 
                terminal_cost = None if consider_equality else DisplacementExampleTerminalCost(), 
                running_inequality_constraints_cost = DisplacementExampleInequalityConstraints(),
                terminal_inequality_constraints_cost = None,
                running_equality_constraints_cost = None,
                terminal_equality_constraints_cost = DisplacementExampleTerminalEqualityConstraints() if consider_equality else None
            ), 
            x0, u_guess, maxiter = 200, atol = atol, tol = 1e-9, approx_hessian = approx_hessian
        )

    logger.info("-------------------------------------------------------------------------------------------")
    _inf = jnp.max(jnp.abs(solution["optimal_trajectory"][-1][0].s + solution["optimal_trajectory"][2][0]))
    _state = solution["optimal_trajectory"][0][-1]
    logger.info(f"inequality constraint infeasibility : {_inf}")
    logger.info(f"Terminal state : {_state}")
    logger.info("-------------------------------------------------------------------------------------------")

    ax = plot_results(solution)
    plt.savefig(os.path.join(_dir, "solution_displacement"))
    plt.close()
        
if not skip_pend_test :
    
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
                u[0] - 0.25,
                -0.25 - u[0]
            ])

    
    class PendulumExampleTerminalEqualityConstraints(NamedTuple):
        # terminal equality constraints of the form g(x) = 0

        def __call__(self, x):
            return jnp.array([
                x[0],
                x[1]
            ])
        

    x0 = jnp.array([-jnp.pi, 0.])
    key = jrandom.PRNGKey(seed = 5)
    u_guess = 0.001*jrandom.randint(key, (500, 1), minval = -10, maxval = 10)
    # u_guess = jnp.ones(shape = (500, 1))
    solution = iterative_linear_quadratic_regulator(
            PendulumExampleDynamics(), 
            TotalCost.form_cost(
                PendulumExampleRunningCost(), 
                terminal_cost = None if consider_equality else PendulumExampleTerminalCost(),
                running_inequality_constraints_cost = PendulumExampleInequalityConstraints(),
                terminal_inequality_constraints_cost = None,
                running_equality_constraints_cost = None,
                terminal_equality_constraints_cost = PendulumExampleTerminalEqualityConstraints() if consider_equality else None
            ), 
            x0, u_guess, maxiter = 300, atol = atol, approx_hessian = approx_hessian
        )
    
    logger.info("-------------------------------------------------------------------------------------------")
    _inf = jnp.max(jnp.abs(solution["optimal_trajectory"][-1][0].s + solution["optimal_trajectory"][2][0]))
    _state = solution["optimal_trajectory"][0][-1]
    logger.info(f"inequality constraint infeasibility : {_inf}")
    logger.info(f"Terminal state : {_state}")
    logger.info("-------------------------------------------------------------------------------------------")

    ax = plot_results(solution)
    plt.savefig(os.path.join(_dir, "solution_pendulum"))
    plt.close()

if not (skip_unicycle_test or consider_equality) : # "Skipping equality constraints as the final states are not reachable"
    
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
                u[0] - 1.5,
                -1.5 - u[0], 
                x[1] - 1.,
                -1. - x[1],
                -(x[0] + 5.5)**2 - (x[1] + 1)**2 + 1**2,
                -(x[0] + 8)**2 - (x[1] - 0.2)**2 + 0.5**2,
                -(x[0] + 2.5)**2 - (x[1] - 1)**2 + 1.5**2,
            ])


    class UnicycleExampleTerminalCost(NamedTuple):
        q : jnp.ndarray = 0.1 * jnp.eye(3)

        def __call__(self, x):
            return x.T @ self.q @ x
        

    class UnicycleExampleTerminalEqualityConstraints(NamedTuple):
        # terminal equality constraints of the form g(x) = 0

        def __call__(self, x):
            return x


    seed = 10
    x0 = jnp.array([-10, 0., 0])
    key = jrandom.PRNGKey(seed = seed)
    u_guess = 0.001*jrandom.randint(key, (650, 1), minval = -10, maxval = 10)
    solution = iterative_linear_quadratic_regulator(
            UnicycleExampleDynamics(), 
            TotalCost.form_cost(
                UnicycleExampleRunningCost(), 
                terminal_cost = UnicycleExampleTerminalCost(), 
                running_inequality_constraints_cost = UnicycleExampleInequalityConstraints(),
                terminal_inequality_constraints_cost = None, 
                running_equality_constraints_cost = None,
                terminal_equality_constraints_cost = None
            ), 
            x0, u_guess, maxiter = 500, atol = atol, approx_hessian = approx_hessian
        )
    
    logger.info("-------------------------------------------------------------------------------------------")
    _inf = jnp.max(jnp.abs(solution["optimal_trajectory"][-1][0].s + solution["optimal_trajectory"][2][0]))
    _state = solution["optimal_trajectory"][0][-1]
    logger.info(f"inequality constraint infeasibility : {_inf}")
    logger.info(f"Terminal state : {_state}")
    logger.info("-------------------------------------------------------------------------------------------")

    ax = plot_results(solution)
    
    # plotting optimal states
    with plt.style.context(["science", "notebook", "bright"]):
        ax[0, 2].clear()
        opt_traj = solution["optimal_trajectory"][0]
        ax[0, 2].plot(opt_traj[:, 0], opt_traj[:, 1], "o")
        ax[0, 2].set(xlabel = "Position x", ylabel = "Position y")

        circles = [Circle((-5.5, -1), 1, color = "k"), Circle((-8, 0.2), 0.5, color = "k"), Circle((-2.5, 1), 1.5, color = "k")]
        for cir in circles :
            ax[0, 2].add_patch(cir)
    
    plt.savefig(os.path.join(_dir, "solution_unicycle"))
    plt.close()

if not (consider_equality or skip_parking_test) : # "Skipping equality constraints as the final states are not reachable")

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


    class CarParkingExampleTerminalEqualityConstraints(NamedTuple):
        # terminal equality constraints of the form g(x) = 0
        target : jnp.ndarray = jnp.array([.1, .1, 0.01, 1.])

        def __call__(self, x):
            return jnp.array([
                x[0] - self.target[0],
                x[1] - self.target[1],
                x[2] - self.target[2],
                x[3] - self.target[3]
            ])

    
    seed = 30
    x0 = jnp.array([1, 1, 3*jnp.pi / 2, 0.])
    key = jrandom.PRNGKey(seed)
    u_guess = 0.1*jrandom.normal(key, (500, 2))
    solution = iterative_linear_quadratic_regulator(
            CarParkingExampleDynamics(), 
            TotalCost.form_cost(
                CarParkingExampleRunningCost(), 
                terminal_cost = CarParkingExampleTerminalCost(), 
                running_inequality_constraints_cost = CarParkingExampleInequalityConstraints(), 
                terminal_inequality_constraints_cost = None,
                running_equality_constraints_cost = None,
                terminal_equality_constraints_cost = None
            ), 
            x0, u_guess, maxiter = 1000, atol = atol, approx_hessian = approx_hessian
        )
    
    logger.info("-------------------------------------------------------------------------------------------")
    _inf = jnp.max(jnp.abs(solution["optimal_trajectory"][-1][0].s + solution["optimal_trajectory"][2][0]))
    _state = solution["optimal_trajectory"][0][-1]
    logger.info(f"constraint infeasibility : {_inf}")
    logger.info(f"Terminal state : {_state}")
    logger.info("-------------------------------------------------------------------------------------------")

    ax = plot_results(solution)
    
    # plotting optimal states
    with plt.style.context(["science", "notebook", "bright"]):
        opt_traj = solution["optimal_trajectory"][0]
        ax[2, 2].plot(opt_traj[:, 0], opt_traj[:, 1], "o")
        ax[2, 2].set(xlabel = "Position x", ylabel = "Position y")

    plt.savefig(os.path.join(_dir, "solution_car_parking"))
    plt.close()

if not (consider_equality or skip_obstacle_test ) : # "Skipping equality constraints as the final states are not reachable")
    
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
                u[0] - jnp.pi / 2,
                - jnp.pi / 2 - u[0],
                u[1] - 10, 
                - 10 - u[1],
                0.5**2 - (x[0] - 1)**2 - (x[1] - 1)**2,
                0.5**2 - (x[0] - 1)**2 - (x[1] - 2.5)**2,
                0.5**2 - (x[0] - 2.5)**2 - (x[1] - 2.5)**2
            ])


    class CarObstacleExampleTerminalEqualityConstraints(NamedTuple):
        # terminal equality constraints of the form g(x) = 0
        target : jnp.ndarray = jnp.array([3., 3., jnp.pi / 2, 0.])

        def __call__(self, x):
            return jnp.array([
                x[0] - self.target[0],
                x[1] - self.target[1],
            ])


    seed = 40
    x0 = jnp.array([0., 0., 0., 0.])
    key = jrandom.PRNGKey(seed)
    u_guess = 0.001*jrandom.randint(key, (200, 2), minval = -10, maxval = 10)
    
    solution = iterative_linear_quadratic_regulator(
            CarObstacleExampleDynamics(), 
            TotalCost.form_cost(
                CarObstacleExampleRunningCost(), 
                terminal_cost = CarObstacleExampleTerminalCost(), 
                running_inequality_constraints_cost = CarObstacleExampleInequalityConstraints(),
                terminal_inequality_constraints_cost = None, 
                running_equality_constraints_cost = None,
                terminal_equality_constraints_cost = None
            ), 
            x0, u_guess, maxiter = 600, atol = atol, approx_hessian = approx_hessian
        )
    
    logger.info("-------------------------------------------------------------------------------------------")
    _inf = jnp.max(jnp.abs(solution["optimal_trajectory"][-1][0].s + solution["optimal_trajectory"][2][0]))
    _state = solution["optimal_trajectory"][0][-1]
    logger.info(f"inequality constraint infeasibility : {_inf}")
    logger.info(f"Terminal state : {_state}")
    logger.info("-------------------------------------------------------------------------------------------")

    ax = plot_results(solution)
    
    # plotting optimal states
    with plt.style.context(["science", "notebook", "bright"]):
        ax[0, 2].clear()
        opt_traj = solution["optimal_trajectory"][0]
        ax[0, 2].plot(opt_traj[:, 0], opt_traj[:, 1], "o")
        ax[0, 2].set(xlabel = "Position x", ylabel = "Position y")

        circles = [Circle((1, 1), 0.5, color = "k"), Circle((1, 2.5), 0.5, color = "k"), Circle((2.5, 2.5), .5, color = "k")]
        for cir in circles :
            ax[0, 2].add_patch(cir)

    plt.savefig(os.path.join(_dir, "solution_car_obstacle"))
    plt.close()

if not skip_quadrotor_test : # Have not tested yet 
    
    # quadrotor example from https://github.com/ZhaomingXie/CDDP/blob/master/systems.py
    # https://zhaomingxie.github.io/projects/CDDP/CDDP.pdf
    logger.info("Started quadrotor test -------------------------------------------------------------------------------------------")

    class QuadrotorExampleDynamics(NamedTuple):
        dt: float = 0.02
        
        def __call__(self, x, u, k = None):
            
            forces = jnp.array([0, 0, u[0] + u[1] + u[2] + u[3]])
            torques = jnp.array([u[0] - u[2], u[1] - u[3], u[0] - u[1] + u[2] - u[3]])
            rotation_matrix = self.rotation_matrix(x)
            J_omega = self.j_omega(x)
            g = jnp.array([0, 0, -10])
            x_next = jnp.zeros(12)
            x_next[0:3] = x[0:3] + self.dt * x[6:9]
            x_next[6:9] = x[6:9] + self.dt * (g + rotation_matrix.dot(forces) - 0 * x[6:9])
            x_next[3:6] = x[3:6] + self.dt * J_omega.dot(x[9:12])
            x_next[9:12] = x[9:12] + self.dt * torques
            return x_next

        def rotation_matrix(self, x):

            r1 = jnp.cos(x[3]) * jnp.cos(x[5]) - jnp.cos(x[4]) * jnp.sin(x[3]) * jnp.sin(x[5])
            r2 = -jnp.cos(x[3]) * jnp.sin(x[3]) - jnp.cos(x[3]) * jnp.cos(x[4]) * jnp.sin(x[5])
            r3 = jnp.sin(x[4]) * jnp.sin(x[3])
            r4 = jnp.cos(x[4]) * jnp.cos(x[5]) * jnp.sin(x[3]) 
            r5 = jnp.cos(x[3]) * jnp.cos(x[4]) * jnp.cos(x[5]) - jnp.sin(x[3]) * jnp.sin(x[5])
            r6 = -jnp.cos(x[5]) * jnp.sin(x[4])
            r7 = jnp.sin(x[3]) * jnp.sin(x[4])
            r8 = jnp.cos(x[3]) * jnp.sin(x[4])
            r9 = jnp.cos(x[4])
            
            return jnp.array([
                r1, r2, r3, r4, r5, r6, r7, r8, r9
            ]).reshape(3, -1)
        
        def j_omega(self, x):
            return jnp.array([
                1, jnp.sin(x[3]) * jnp.tan(x[4]),   jnp.cos(x[3]) * jnp.tan(x[4]), 
                0, jnp.cos(x[3]),                   - jnp.sin(x[3]), 
                0, jnp.sin(x[3]) / jnp.cos(x[4]),   jnp.cos(x[3]) / jnp.cos(x[4]) 
            ]).reshape(3, -1)


    class QuadrotorExampleRunningCost(NamedTuple):
        q : jnp.ndarray = 0 * jnp.eye(12)
        r : jnp.ndarray = 0.02 * jnp.eye(4)

        def __call__(self, x, u, k = None):
            return u.T @ self.r @ u


    class QuadrotorExampleTerminalCost(NamedTuple):
        q : jnp.ndarray = jnp.diag(jnp.array([50, 50, 50, 2, 2, 2, 1, 1, 1, 1, 1, 1.]))
        target : jnp.ndarray = jnp.array([2.8, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0])

        def __call__(self, x):
            return (x - self.target).T @ self.q @ (x - self.target)


    class QuadrotorExampleInequalityConstraints(NamedTuple):
        # inequality constraints of the form h(x, u) <= 0

        def __call__(self, x, u, k = None):
            return jnp.array([
                - u[0],
                - u[1],
                - u[2], 
                - u[3],
                2**2 - (x[0] - 0)**2 - (x[1] - 0)**2 - (x[2] - 0)**2, # spherical constraints
            ])


    class QuadrotorExampleTerminalEqualityConstraints(NamedTuple):
        # terminal equality constraints of the form g(x) = 0
        target : jnp.ndarray = jnp.array([3., 3., jnp.pi / 2, 0.])

        def __call__(self, x):
            return jnp.array([
            ])


    seed = 40
    x0 = jnp.array([-3.5, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0])
    key = jrandom.PRNGKey(seed)
    u_guess = 0.001*jrandom.randint(key, (200, 4), minval = -10, maxval = 10)
    
    solution = iterative_linear_quadratic_regulator(
            QuadrotorExampleDynamics(), 
            TotalCost.form_cost(
                QuadrotorExampleRunningCost(), 
                terminal_cost = lambda x : 50. * (x[-2] - jnp.pi / 2)**2 + 10 * (x[-1] - 0.)**2 if consider_equality else QuadrotorExampleTerminalCost(), 
                running_inequality_constraints_cost = QuadrotorExampleInequalityConstraints(),
                terminal_inequality_constraints_cost = None, 
                running_equality_constraints_cost = None,
                terminal_equality_constraints_cost = QuadrotorExampleTerminalEqualityConstraints() if consider_equality else None
            ), 
            x0, u_guess, maxiter = 600, atol = atol
        )
    
    
    print("inequality constraint infeasibility", jnp.max(jnp.abs(solution["optimal_trajectory"][-1][0].s + solution["optimal_trajectory"][2][0])))
    print("Terminal state", solution["optimal_trajectory"][0][-1])

    ax = plot_results(solution)
    
    # plotting optimal states 
    # TODO 3d plot to view trajectory
    ax[0, 2].clear()
    opt_traj = solution["optimal_trajectory"][0]
    ax[0, 2].plot(opt_traj[:, 0], opt_traj[:, 1], "o")
    ax[0, 2].set(xlabel = "Position x", ylabel = "Position y")

    # plt.savefig(os.path.join(_dir, "solution_quadrotor"))
    plt.savefig("solution_quadrotor")
    plt.close()

if (not skip_cstr_test) and consider_equality :
    
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


    seed = 40
    x0 = jnp.array([0.5, 350, 300.])
    key = jrandom.PRNGKey(seed)
    u_guess = 150 * jnp.ones(shape = (400, 1))

    solution = iterative_linear_quadratic_regulator(
            RK4Integrator(CstrExampleDynamics(), dt = 0.01), 
            TotalCost.form_cost(
                CstrExampleRunningCost(), 
                terminal_cost = None if consider_equality else CstrExampleTerminalCost(), 
                running_inequality_constraints_cost = CstrExampleInequalityConstraints(),
                terminal_inequality_constraints_cost = None, 
                running_equality_constraints_cost = None,
                terminal_equality_constraints_cost = CstrExampleTerminalEqualityConstraints() if consider_equality else None
            ), 
            x0, u_guess, maxiter = 500, atol = atol, approx_hessian = approx_hessian
        )
    
    logger.info("-------------------------------------------------------------------------------------------")
    _inf = jnp.max(jnp.abs(solution["optimal_trajectory"][-1][0].s + solution["optimal_trajectory"][2][0]))
    _state = solution["optimal_trajectory"][0][-1]
    logger.info(f"inequality constraint infeasibility : {_inf}")
    logger.info(f"Terminal state : {_state}")
    logger.info("-------------------------------------------------------------------------------------------")

    ax = plot_results(solution)

    # plotting only temperature
    with plt.style.context(["science", "notebook", "bright"]):
        ax[0, 2].clear()
        opt_traj = solution["optimal_trajectory"][0]
        ax[0, 2].plot(opt_traj[:, 1], "o")
        ax[0, 2].set(xlabel = "Horizon", ylabel = "Temperature")

    plt.savefig(os.path.join(_dir, "solution_cstr"))
    plt.close()

