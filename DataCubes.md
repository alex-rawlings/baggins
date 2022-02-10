# Data Cube Description  
To facilitate the analysis of large quantities of data, each child run has associated with it a data cube that can be accessed to retrieve key quantities of interest. 
The original data can still be retrieved from Allas in the event that a desired quantity is not present in the cube.

## Layout  
```
/  
+-- meta  
|    +-- created //date created  
|    +-- created_by //username of creator  
|    +-- last_accessed //when cube last opened  
|    +-- last_user //username of last person to access  
+-- bh_binary  
|    +-- binary_formation_time //time binary becomes bound  
|    +-- binary_merger_timescale //time to merger from bound time  
|    +-- merger_remnant  
|    |    +-- chi //merger chi value  
|    |    +-- kick_magnitude //merger recoil velocity  
|    |    +-- mass //merger mass  
|    |    +-- merged //bool, has BHB mreged?  
|    +-- r_infl //influence radius [pc]  
|    +-- r_infl_time //time when a=r_infl  
|    +-- r_hard //hardening radius [pc]  
|    +-- r_hard_time //time when a=r_hard  
|    +-- tspan //time since t_ah that analytical fit determined  
|    +-- H //semimajor axis hardening constant  
|    +-- K //eccentricity constant  
|    +-- gw_dominant_semimajor_axis //a where GW emission dominates hardening  
|    +-- gw_dominant_semimajor_axis_time //time when above occurs  
|    +-- predicted_orbital_params  
|    |    +-- a  
|    |    |    +-- lower //a values assuming lower quantile of e during tspan  
|    |    |    +-- estimate //a values assuming median quantile of e during tspan  
|    |    |    +-- upper //a values assuming upper quantile of e during tspan  
|    |    +-- e  
|    |    |    +-- lower //e values assuming lower quantile of e during tspan  
|    |    |    +-- estimate //e values assuming median quantile of e during tspan  
|    |    |    +-- upper //e values assuming upper quantile of e during tspan  
|    |    +-- t  
|    |    |    +-- lower //t values assuming lower quantile of e during tspan  
|    |    |    +-- estimate //t values assuming median quantile of e during tspan  
|    |    |    +-- upper //t values assuming upper quantile of e during tspan  
|    +-- formation_eccentricity_spread //std in e before a=r_infl  
+-- galaxy_properties  
|    +-- stellar_velocity_dispersion //sigma along principal axis for all stars  
|    +-- stellar_velocity_dispersion_projected  
|    |    +-- low //std below mean projected sigma in 1Re  
|    |    +-- estimate //mean projected sigma in 1Re  
|    |    +-- high //std above mean projected sigma in 1Re  
+-- inner_DM_fraction //fraction of DM inside 1Re  
+-- half_mass_radius //radius containing half of stellar mass [pc]  
+-- effective_radius  
|    +-- low //lower estimate for Re  
|    +-- estimate //best estimate for Re  
|    +-- high //upper estimate for Re  
+-- density_profile  
|    +-- 3D //3D density profile  
|    +-- projected  
|    |    +--  
|    |    +--  
|    |    +--  
+-- total_stellar_mass //total stellar mass of system  
+-- virial_info  
|    +-- mass //virial mass  
|    +-- radius //virial radius  
+-- ifu_map_merger //IFU map at time of merger  
|    +-- Vcom //centre of mass velocity  
|    +-- bin_V  //voronoi stats  
|    +-- bin_sigma  //voronoi stats  
|    +-- bin_h3  //voronoi stats  
|    +-- bin_h4  //voronoi stats  
|    +-- bin_mass  //voronoi stats  
|    +-- extent //image extent  
|    +-- img_V  //voronoi image  
|    +-- img_sigma  //voronoi image  
|    +-- img_h3  //voronoi image  
|    +-- img_h4  //voronoi image  
|    +-- xbar  //voronoi grid data  
|    +-- ybar  //voronoi grid data  
|    +-- ycom  //voronoi grid data  
+-- ifu_map_ah //IFU map at time r_hard_time  
|    +-- Vcom //centre of mass velocity  
|    +-- bin_V  //voronoi stats  
|    +-- bin_sigma  //voronoi stats  
|    +-- bin_h3  //voronoi stats  
|    +-- bin_h4  //voronoi stats  
|    +-- bin_mass  //voronoi stats  
|    +-- extent //image extent  
|    +-- img_V  //voronoi image  
|    +-- img_sigma  //voronoi image  
|    +-- img_h3  //voronoi image  
|    +-- img_h4  //voronoi image  
|    +-- xbar  //voronoi grid data  
|    +-- ybar  //voronoi grid data  
|    +-- ycom  //voronoi grid data  
+-- shell_statistics  
|    +-- snapshot_times //times of snapshots  
|    +-- stellar_shell_outflow_velocity //radial velocity of stars through a shell about BHB CoM  
|    +-- bh_binary_watershed_velocity //watershed velocity of BH binary  
+-- beta_r  
|    +-- beta_r //anisotropy parameter as a function of radius  
|    +-- radbins //radial bins used for beta(r)  
|    +-- bincount //number stellar particles per bin  
```

## TODO  
- projected profile  
- ang. mom. of BHB and galaxy at each snap  
- BH binary spin flips (bool)  
- core size  
- triaxiality parameters  
- mass bound to BHs  
- loss cone ang. mom. at merger, or other times?  
- eccentricity at "bound"  
- eccentricity at rh  
- eccentricity at first pericentre (this is from parent, is it appropriate here?)  

