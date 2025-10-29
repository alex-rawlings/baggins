// bindings.cpp
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

namespace py = pybind11;

// Declare dydt() from dynamics.cpp
std::vector<double> dydt(double t,
                         const std::vector<double>& y,
                         const std::vector<double>& params);

PYBIND11_MODULE(_dynamics, m) {
    m.doc() = "C++ ODE RHS (dydt) interface for SciPy solve_ivp";
    m.def("dydt", &dydt, "Compute time derivative dy/dt",
          py::arg("t"), py::arg("y"), py::arg("params"));
}
