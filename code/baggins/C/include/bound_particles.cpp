#pragma once

#include <iostream>
#include <sstream>
#include <stdexcept>
#include <cmath>
#include <vector>
#include <array>
#include <algorithm>


/*
Structure to hold particle data
*/
struct Particle
{
    int ID;
    double mass;
    std::array<double,3> pos;
    std::array<double,3> vel;
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
    for(int i=0; i<ps.size(); ++i){
        for(int j=0; j<3; ++j){
            numerator[j] += ps[i].mass * ps[i].pos[j];
            numerator[j+3] += ps[i].mass * ps[i].vel[j];
        }
        denominator += ps[i].mass;
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
@param ptype: particle type from Gadget
@param IDs: particle IDs
@param mass: particles masses
@param pos: particle positions
@param vel: particle velocities
@return
*/
std::vector<int> find_bound_particles(
    std::vector<int>& ptype,
    std::vector<int>& IDs,
    std::vector<double>& mass,
    std::vector<std::array<double, 3>>& pos,
    std::vector<std::array<double, 3>>& vel
){
    // declare variables
    std::vector<Particle> central;
    std::vector<Particle> candidates(IDs.size());
    std::vector<double> radial_distance(IDs.size());
    std::vector<int> bound_IDs;
    int iters = 0;
    int min_dist_star_idx;
    const double G = 4.3009e-6;
    double kinetic_E, potential_E;
    std::array<double,7> _cm_coords;

    // add the BH particles to the central object
    for(int i=0; i<ptype.size(); ++i){
        Particle p;
        if(ptype[i] == 5){
            p.ID = IDs[i];
            p.mass = mass[i];
            p.pos = pos[i];
            p.vel = vel[i];
            central.push_back(p);
        }
    }

    while(true){
        // shift candidates to CoM coordinates
        _cm_coords = mass_weighted_coordinates(central);
        for(int i=0; i<IDs.size(); ++i){
            for(int j=0; j<3; ++j){
                pos[i][j] -= _cm_coords[j];
                vel[i][j] -= _cm_coords[j+3];
            }
        }

        // identify which particles may be potentially added to the CM object
        for(int i=0; i<central.size(); ++i){
            Particle p;
            if(std::find(IDs.begin(), IDs.end(), central[i].ID) != IDs.end()){
                // this particle is not in the central object
                p.ID = IDs[i];
                p.mass = mass[i];
                p.pos = pos[i];
                p.vel = vel[i];
                candidates[i] = p;
                radial_distance[i] = sqrt(p.pos[0]*p.pos[0] + p.pos[1]*p.pos[1] + p.pos[2]*p.pos[2]);
            }
        }

        // determine the index of the candidate star with smallest separation
        min_dist_star_idx = std::distance(radial_distance.begin(),
                            std::min_element(radial_distance.begin(), radial_distance.end())
                            );

        // determine the energy of this particle
        kinetic_E = 0.5 * sqrt(
            pow(candidates[min_dist_star_idx].vel[0], 2.) + 
            pow(candidates[min_dist_star_idx].vel[1], 2.) + 
            pow(candidates[min_dist_star_idx].vel[2], 2.)
        );
        potential_E = G * _cm_coords[6] / radial_distance[min_dist_star_idx];
        if(kinetic_E - potential_E < 0){
            // this particle is bound
            // undo CoM translation and boost
            for(int i=0; i<3; ++i){
                candidates[min_dist_star_idx].pos[i] += _cm_coords[i];
                candidates[min_dist_star_idx].vel[i] += _cm_coords[i+3];
            }
            // add it to the central mass
            central.push_back(candidates[min_dist_star_idx]);
            // remove it from the list of candidates
            candidates.erase(candidates.begin()+min_dist_star_idx);
            radial_distance.erase(radial_distance.begin()+min_dist_star_idx);
            iters++;
        }else{
            // this particle in unbound
            break;
        }
    }
    // collect all the IDs of particles that are part of the central object
    // this includes also the BHs that were used to define the original central 
    // object, and not just the added stars
    for(int i=0; i<central.size(); ++i){
        bound_IDs.push_back(central[i].ID);
    }
    return bound_IDs;
}
