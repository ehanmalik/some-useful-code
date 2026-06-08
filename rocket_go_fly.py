import math

def blueprint(satelite_mass, num_satelites, area_of_satelites, au, efficiancy, hours):
    print("Total number of fuel needed for given mass and number values.")
    delta_v = 11.2
    #delta v needed to escape earth
    v_e = 4.4
    #velocity from hydrolox thrusters

    rocket_mass = satelite_mass * math.exp(delta_v / v_e)
    fuel_per_satelite = rocket_mass - satelite_mass
    tota_fuel_needed = fuel_per_satelite * num_satelites

    print(f"Weight of each satelite: {satelite_mass:,} kg")
    print(f"Fuel needed per launch: {fuel_per_satelite:,.2f} kg")
    print(f"Total fuel burned for a {num_satelites} satellite fleet: {tota_fuel_needed:,.2f} kg")

    print("Now i will calculate the energy genertated at the specified au.")
    solar_constant_earth = 1361
    irradiance = solar_constant_earth / (au ** 2)
    watts = irradiance * area_of_satelites * efficiancy * num_satelites
    joules = watts * (hours * 3600)

    print(f"Au at {au} generates {watts} watts per {hours} hours at {efficiancy} efficiancy and joules at {joules} joules.")


blueprint(1000, 1000, 100, 0.5, 0.3, 1)
