from fenics import *
import numpy as np
from scipy.special import zeta

import time
import json

class AffineEquation:
    def __init__(
        self,
        spatial_dimension = 2,
        expectation = None,
        decay_rate = 2,
        f = 10,
        regularity = 1, 
        p = 0.5
    ) -> None:
        self.spatial_dimension = spatial_dimension

        self.decay_rate = decay_rate
        self.t = regularity 

        if expectation == None:
            b = np.sqrt(2) * zeta(self.decay_rate, 1)
            kappa_inverse = 1.1
            self.expectation = b * kappa_inverse 
        else:
            self.expectation = expectation
            
        self.f = f
        self.p = p
        self.sigma_p = p / (1 - p)

        self.qoi = lambda u: assemble(u*dx)

        self.max_level = 1
        self.min_nx = 5
        self.fe_degree = 1
        self.initialised = False

    def diffusion_coefficient(self, x, y):
        coefficient_sum = self.expectation(x) if callable(self.expectation) else self.expectation

        x_norm = x[0] if self.spatial_dimension == 1 else sqrt(x[0]**2 + x[1]**2) if self.spatial_dimension == 2 else sqrt(x[0]**2 + x[1]**2 + x[2]**2) 

        l = 1
        for k in range(len(y)):
            if (k % 2):
                coefficient_sum += y[k] / l**(self.decay_rate) * cos(l * pi * x_norm)
            else: 
                coefficient_sum += y[k] / l**(self.decay_rate) * sin(l * pi * x_norm)
                l += 1
            
        return coefficient_sum
    
    def initialise_spaces(self, max_level, min_nx = None, fe_degree = None):
        assert isinstance(max_level, int) and max_level > 0, "The level must be an integer greater than zero"

        self.max_level = max_level
        self.min_nx = min_nx or self.min_nx
        self.fe_degree = fe_degree or self.fe_degree

        self.mesh_widths = []
        self.meshes = []
        self.spaces = []
        self.functionals = []
        self.boundary_conditions = []

        for level in range(max_level, 0, -1):
            nx = self.min_nx * 2 ** (level - 1)
        
            if self.spatial_dimension == 1:
                mesh = UnitIntervalMesh(nx)
            if self.spatial_dimension == 2:
                mesh = UnitSquareMesh(nx, nx)
            else:
                mesh = UnitCubeMesh(nx, nx, nx)

            V = FunctionSpace(mesh, 'P', self.fe_degree)
            f = self.f(SpatialCoordinate(mesh)) if callable(self.f) else Constant(self.f)
            bc = DirichletBC(V, Constant(0), lambda x, on_boundary: on_boundary)

            self.mesh_widths.append(1 / nx)
            self.meshes.append(mesh)
            self.spaces.append(V)
            self.functionals.append(f)
            self.boundary_conditions.append(bc)

        self.initialised = True

    def solve_equation(self, y, level = None):
        assert self.initialised == True, "function spaces not initialised"

        level = level or self.max_level
        index = self.max_level - level

        V = self.spaces[index]
        mesh = self.meshes[index]
        f = self.functionals[index]
        bc = self.boundary_conditions[index]

        a = self.diffusion_coefficient(SpatialCoordinate(mesh), y)
        
        u = TrialFunction(V)
        v = TestFunction(V)
        equation = a * inner(grad(u), grad(v)) * dx == f * v * dx
        
        u = Function(V)
        solve(equation, u, bc)
        return u
       
    def perform_singlelevel_procedure(self, truncation = 20, c_sparsity = 1, c_M = 1, c_cv = 10):
        
        assert self.initialised == True, "function spaces not initialised"
        
        starting_time = time.process_time()
        print("Start single level procedure")

        G = self.qoi
        
        s = c_sparsity * 2 ** (self.max_level * self.t * self.sigma_p)
        M = s * (truncation + np.log(s)) # * np.log(s) ** 3
        M = c_M * c_cv * int(M // c_cv + 1)

        points = np.random.uniform(low = -1, high = 1, size = (M, truncation))
        values = []

        for m in range(M):
            print(f"[{m + 1}/{M}] Single level L = {self.max_level}")
            y = points[m]

            u = self.solve_equation(y)
            values.append(G(u))

        self.singlelevel_samples = {
            "points": points,
            "values": values
            }
    
        self.singlelevel_parameters = {
            "level": self.max_level,
            "sparsity": s,
            "samplingDuration": time.process_time() - starting_time,
            "meshWidth": self.mesh_widths[0]
        }
    
    def perform_multilevel_procedure(self, truncation = 20, c_sparsity = 1, c_M = 1, c_cv = 10):
        assert self.initialised == True, "function spaces not initialised"

        times = [time.process_time()]
        print("Start multi level procedure")

        G = self.qoi

        self.multilevel_samples = []
        self.multilevel_parameters = {
                "level": [level for level in range(self.max_level, 0, -1)],
                "sparsity": [],
                "samplingDuration": [],
                "meshWidth": self.mesh_widths
            }      
        
        for level in self.multilevel_parameters["level"]:
            index = self.max_level - level

            s = c_sparsity * 2 ** ((index + 1) * self.t * self.sigma_p)
            M = s * (truncation + np.log(s)) # * np.log(s) ** 3
            M = c_M * c_cv * int(M // c_cv + 1)

            points = np.random.uniform(low = -1, high = 1, size = (M, truncation))
            values = []

            for m in range(M):
                print(f"[{m + 1}/{M}] Multi level L = {self.max_level}; l = {level}")
                y = points[m]

                u_fine = self.solve_equation(y, level = level)

                if level > 1:
                    u_coarse = interpolate(u_fine, self.spaces[index + 1])
                    values.append(G(u_fine) - G(u_coarse))
                else:
                    values.append(G(u_fine))

            self.multilevel_samples.append({
                "points": points,
                "values": values
                })

            times.append(time.process_time())
            self.multilevel_parameters["samplingDuration"].append(times[-1] - times[-2])
            self.multilevel_parameters["sparsity"].append(s)      

    def generate_test_set(self, size = 100, truncation = 20):
        assert self.initialised == True, "function spaces not initialised"

        starting_time = time.process_time()
        print("Start generation of test set")

        G = self.qoi
        
        points = np.random.uniform(low = -1, high = 1, size = (size, truncation))
        values = []

        for m in range(size):
            print(f"[{m + 1}/{size}] Test set L = {self.max_level}")
            y = points[m]

            u = self.solve_equation(y)
            values.append(G(u))

        self.test_set = {
            "points": points,
            "values": values
            }
        
        self.test_parameters = {
            "level": self.max_level,
            "meshWidth": self.mesh_widths[0]
        }      

    def save_results(self, filename_singlelevel = None, filename_multilevel = None, filename_test = None):

        if isinstance(filename_singlelevel, str):
            assert hasattr(self, "singlelevel_samples") and hasattr(self, "singlelevel_parameters"), "no generated single level data to save"
            
            np_filename = f"{filename_singlelevel}.npz"            
            open(np_filename, "w")
            np.savez(np_filename, **self.singlelevel_samples)

            json_filename = f"{filename_singlelevel}.json"
            with open(json_filename, "w") as f:
                json.dump(self.singlelevel_parameters, f)

        if isinstance(filename_multilevel, str):
            assert hasattr(self, "multilevel_samples") and hasattr(self, "multilevel_parameters"), "no generated multi level data to save"

            for j in range(len(self.multilevel_samples)):
                np_filename = f"{filename_multilevel}_level{self.max_level - j}.npz"
                open(np_filename, "w")
                np.savez(np_filename, **self.multilevel_samples[j])

            json_filename = f"{filename_multilevel}.json"
            with open(json_filename, "w") as f:
                json.dump(self.multilevel_parameters, f)

        if isinstance(filename_test, str):
            assert hasattr(self, "test_set") and hasattr(self, "test_parameters"), "no generated test data to save"

            np_filename = f"{filename_test}.npz"
            open(np_filename, "w")
            np.savez(np_filename, **self.test_set)

            json_filename = f"{filename_test}.json"
            with open(json_filename, "w") as f:
                json.dump(self.test_parameters, f)

