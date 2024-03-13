#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
#include <pybind11/stl.h>

#include "bound_particles.cpp"

namespace py = pybind11;

PYBIND11_MODULE(orbitCXX, m){
    // module docstring
    m.doc() = "Module for heavy-looping of particle data";

    m.def("find_bound_particles", &find_bound_particles, "Find particles bound to a single or binary black hole");

/*
    py::class_<Potential>(m, "Potential")
        // init function
        .def(py::init<int, int, 
            std::vector<double>,
            std::vector<double>,
            std::vector<double>,
            const std::vector<double>&>())
        // TODO lambda function to help with printing
        /*.def("__repr__",
            [](const Potential &a){
                return "<orbit_analysis.Potential object with (n_max, l_max)=" + a.get_nmax() + " " + a.get_lmax() + ">";
            }
        );
        .def("get_nmax", &Potential::get_nmax)
        .def("get_lmax", &Potential::get_lmax)
        .def("get_mmax", &Potential::get_mmax)
        .def("compute_coefficients", &Potential::compute_coefficients,
            "Compute coefficients at some radius",
            py::arg("l"), py::arg("m"), py::arg("r"))
        .def("compute_potential", &Potential::compute_potential, 
            "Compute the potential at some coordinates",
            py::arg("x")=1., py::arg("y")=0., py::arg("z")=0.)
        .def("compute_acceleration", &Potential::compute_acceleration,
            "Compute the acceleration of a particle in the field",
            py::arg("x")=1., py::arg("y")=0., py::arg("z")=0.);


    py::class_<OrbitIntegrator>(m, "OrbitIntegrator")
    // init function
    .def(py::init<Potential&>())
    .def("get_states",
        [](const OrbitIntegrator &o) -> py::array_t<double> {
            return py::cast(o.states);
        })
    .def("get_times",
        [](const OrbitIntegrator &o) -> py::array_t<double> {
            return py::cast(o.times);
        })
    .def("integrate_in_potential", &OrbitIntegrator::integrate_in_potential, 
    "Integrate a particle in a potential",
    py::arg("init_state"), py::arg("t0"), py::arg("tf"), py::arg("output_dt"), py::arg("abs_tol")=1e-15, py::arg("rel_tol")=1e-9);
    */
}