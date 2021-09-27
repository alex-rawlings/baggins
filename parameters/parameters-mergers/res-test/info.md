# Resolution Tests (Mergers)

This directory contains the parameter files for the merger initial conditions
at different particle mass resolutions to be used in mass resolution testing. 
The conventions of the *isolated resolution tests* are followed here too. 

Mergers are set up using the ICs of the desired mass resolution. There are five
different orbital approaches tested, which are uniquely described with:
- initial separation
- pericentre distance

Both quantities are given in terms of the virial radius of the *larger* of the 
two progenitor galaxies. In all tests, the initial separation is set to
$3\,\mathrm{R}_\mathrm{vir}$, motivated by the way in which the NFW profile of 
the progenitor galaxies is cut ($3\,\mathrm{R}_\mathrm{vir}$ is approximately 
where the cut NFW profile deviates from the true NFW profile by 99%). The 
pericentre distance is varied evenly in $\log_{10}$ space between 
$1.0\times10^{-3}$ and $1.0$ times the virial radius of the larger progenitor.
Hence, the tested pericentre distances are:
1. 0.001
2. 0.005
3. 0.030
4. 0.180
5. 1.000

In all tests so far the galaxies used in the mergers are:
- NGCa0524 (stellar cusp profile)
- NGCa3348 (stellar cored profile)

Which have been renamed A and C, respectively, for brevity (this follows the
somewhat arbitrary choice of the galaxy names in alphabetical order). These 
galaxies are of similar (stellar) mass to one another. 

The naming convention for the merger mass resolution tests follows:  
**AN-CN-I-P**  
where **N** indicates the increased in particle mass (e.g. 02), **I** indicates
the initial separation in terms of the (larger) virial radius (here always 3.0), 
and **P** indicates the pericentre distance in terms of the (larger) virial
radius (e.g. 0.001). The somewhat superfluous use of **N** is included as a
sanity check that indeed both progenitor galaxies are of the same mass
resolution, though this is of course checked by the updated 
`merger-ic-generator` code. 