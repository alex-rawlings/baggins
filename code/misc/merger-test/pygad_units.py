import pygad


G1 = pygad.UnitScalar(4.30091e-3, "pc/Msol*(km/s)**2")
print(type(G1))
print(pygad.physics.G.in_units_of("pc/Msol*(km/s)**2"))


F1 = pygad.physics.G * pygad.UnitScalar(10, "Msol")**2 / pygad.UnitScalar(10, "pc")**2
F2 = pygad.physics.G.in_units_of("pc/Msol*(km/s)**2") * pygad.UnitScalar(10, "Msol")**2 / pygad.UnitScalar(10, "pc")**2
print(F1.in_units_of("m*kg/s**2"))
print(F2.in_units_of("m*kg/s**2"))



inner_density = pygad.UnitScalar(0.5, "Msol/pc**3")
inner_sigma = pygad.UnitScalar(230, "km/s")
G_rho_per_sigma = pygad.physics.G * inner_density / inner_sigma #* pygad.UnitScalar(1.02e-6, "pc/km*s/yr")
print(G_rho_per_sigma.in_units_of("pc**-1/yr"))