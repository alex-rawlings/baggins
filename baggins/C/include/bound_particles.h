#pragma once

#include <iostream>
#include <cmath>
#include <vector>
#include <array>
#include <algorithm>
#include <stdexcept>
#include <omp.h>

namespace bound_particles{

/*
Structure to hold particle data
*/
struct Particle
{
    int ID;
    double mass;
    std::array<double,3> pos;
    std::array<double,3> vel;
    double r;

    void get_distance(){
        r = pos[0]*pos[0] + pos[1]*pos[1] + pos[2]*pos[2];
    }
};


/*
Determine the centre of mass position and centre of mass velocity
@param ps: vector of Particle structures
@return res: array with elements xx, xy, xz, vx, vy, vz, total_mass
*/
std::array<double,7> mass_weighted_coordinates(std::vector<Particle>& ps){
    std::array<double,6> numerator = {0., 0., 0., 0., 0., 0.};
    double denominator = 0.;
    std::array<double,7> res;
    for(auto & p : ps){
        for(int i=0; i<3; ++i){
            numerator[i] += p.mass * p.pos[i];
            numerator[i+3] += p.mass * p.vel[i];
        }
        denominator += p.mass;
    }
    for(int i=0; i<6; ++i){
        res[i] = numerator[i] / denominator;
    }
    // save the total mass
    res[6] = denominator;
    return res;
}


/*
Determine which particles are bound to a central mass (presumably the BH)
@param part_type: particle type from Gadget
@param IDs: particle IDs
@param mass: particles masses
@param pos: particle positions
@param vel: particle velocities
@return bound_IDs: IDs of bound particles, including BHs
*/
std::vector<int> find_bound_particles(
    const std::vector<int>& part_type,
    const std::vector<int>& IDs,
    const std::vector<double>& mass,
    const std::vector<std::array<double, 3>>& pos,
    const std::vector<std::array<double, 3>>& vel
){
    // declare variables
    std::vector<Particle> central;
    std::vector<Particle> candidates;
    std::vector<int> bound_IDs;
    int iters = 0;
    int unbound_count = 0;
    const double G = 4.3009e-6;
    double kinetic_E, potential_E;
    std::array<double,7> _cm_coords;

    // ensure vectors have equal length
    if(part_type.size() != IDs.size() || part_type.size() != mass.size() || 
        part_type.size() != pos.size() || part_type.size() != vel.size()){
            std::invalid_argument("Input vectors must have same length!\n");
    }

    // classify the particles
    for(unsigned int i=0; i<part_type.size(); ++i){
        Particle p;
        p.ID = IDs[i];
        p.mass = mass[i];
        p.pos = pos[i];
        p.vel = vel[i];
        if(part_type[i] == 5){
            // this is a BH, add it to the central object
            central.push_back(p);
        }else{
            // this is a star, add it to the candidates
            candidates.push_back(p);
        }
    }
    // check that we have central and candidate particles
    if(central.empty()){
        std::invalid_argument("There are no particles for the 'central' object!\n");
    }
    if(candidates.empty()){
        std::invalid_argument("There are no particles for the 'candidates' object!\n");
    }

    while(!candidates.empty() && unbound_count<10){
        // print intermittent progress
        if(iters%10000 == 0){
            std::cout << "Iterations: " << iters << "\n";
        }
        // determine CoM
        _cm_coords = mass_weighted_coordinates(central);
        // shift central to CoM coordinates
        #pragma omp parallel for
        for(auto & c : central){
            for(int j=0; j<3; ++j){
                c.pos[j] -= _cm_coords[j];
                c.vel[j] -= _cm_coords[j+3];
            }
        }
        // shift candidates to CoM coordinates
        #pragma omp parallel for
        for(auto & c : candidates){
            for(int j=0; j<3; ++j){
                c.pos[j] -= _cm_coords[j];
                c.vel[j] -= _cm_coords[j+3];
            }
            c.get_distance();
        }

        // determine the index of the candidate star with smallest separation
        auto min_r_part = std::min_element(
            candidates.begin(), candidates.end(),
            [](const Particle& p1, const Particle& p2){
                return p1.r < p2.r;
            }
        );

        // determine the energy of this particle
        kinetic_E = 0.5 * sqrt(
            pow(min_r_part->vel[0], 2.) + 
            pow(min_r_part->vel[1], 2.) + 
            pow(min_r_part->vel[2], 2.)
        );
        potential_E = G * _cm_coords[6] / min_r_part->r;
        if((kinetic_E - potential_E) < 0){
            // this particle is bound
            // add it to the central mass
            central.push_back(*min_r_part);
            // remove it from the list of candidates
            candidates.erase(min_r_part);
        }else{
            // this particle in unbound
            unbound_count++;
        }
        iters++;
    }
    // collect all the IDs of particles that are part of the central object
    // this includes also the BHs that were used to define the original central 
    // object, and not just the added stars
    for(auto & c : central){
        bound_IDs.push_back(c.ID);
    }
    if(candidates.empty()){
        std::cout << "All particles in search radius are bound!\n";
    }
    return bound_IDs;
}

} // namespace bound_particles