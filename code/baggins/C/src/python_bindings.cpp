#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
#include <pybind11/stl.h>

#include "bound_particles.h"

namespace py = pybind11;

PYBIND11_MODULE(bagginsCXX, m){
    // module docstring
    m.doc() = "Module for heavy-looping of particle data";

    m.def("find_bound_particles", &bound_particles::find_bound_particles, "Find particles bound to a single or binary black hole");
}