"""
    N-body simulation.

    Optimized. 

    The runtime after cython improved is 13.8s. 
"""
import cython
import timeit
from itertools import combinations

cpdef void advance(double dt,int iterations,dict BODIES,list body1_body2):
    '''
        advance the system one timestep
    '''
    for _ in range(iterations):
        for(body1,body2) in body1_body2:
            ((x1, y1, z1), v1, m1) = BODIES[body1]
            ((x2, y2, z2), v2, m2) = BODIES[body2]
            (dx, dy, dz) = (x1-x2, y1-y2, z1-z2)
            mag = dt * ((dx * dx + dy * dy + dz * dz) ** (-1.5))
            mag_m1 = m1 * mag
            mag_m2 = m2 * mag
            v1[0] -= dx * mag_m2
            v1[1] -= dy * mag_m2
            v1[2] -= dz * mag_m2
            v2[0] += dx * mag_m1
            v2[1] += dy * mag_m1
            v2[2] += dz * mag_m1

        
        for body in BODIES.keys():
            (r, [vx, vy, vz], m) = BODIES[body]
            r[0] += dt * vx
            r[1] += dt * vy
            r[2] += dt * vz

    
cpdef float report_energy(dict BODIES,list body1_body2,float e=0.0):
    '''
        compute the energy and return it so that it can be printed and reported 

        return e as the energy
    '''

    for(body1,body2) in body1_body2:
        ((x1, y1, z1), v1, m1) = BODIES[body1]
        ((x2, y2, z2), v2, m2) = BODIES[body2]
        (dx, dy, dz) = (x1-x2, y1-y2, z1-z2)
        e -= (m1 * m2) / ((dx * dx + dy * dy + dz * dz) ** 0.5)
        
    for body in BODIES.keys():
        (r, [vx, vy, vz], m) = BODIES[body]
        e += m * (vx * vx + vy * vy + vz * vz) / 2.
        
    return e


cpdef void nbody(int loops,str reference,int iterations):
    '''
        nbody simulation
        loops - number of loops to run
        reference - body at center of system
        iterations - number of timesteps to advance
    '''
    # Set up global state
    cdef float PI = 3.14159265358979323
    cdef float SOLAR_MASS = 4 * PI * PI
    cdef float DAYS_PER_YEAR = 365.24

    cdef dict BODIES = {
    'sun': ([0.0, 0.0, 0.0], [0.0, 0.0, 0.0], SOLAR_MASS),

    'jupiter': ([4.84143144246472090e+00,
                 -1.16032004402742839e+00,
                 -1.03622044471123109e-01],
                [1.66007664274403694e-03 * DAYS_PER_YEAR,
                 7.69901118419740425e-03 * DAYS_PER_YEAR,
                 -6.90460016972063023e-05 * DAYS_PER_YEAR],
                9.54791938424326609e-04 * SOLAR_MASS),

    'saturn': ([8.34336671824457987e+00,
                4.12479856412430479e+00,
                -4.03523417114321381e-01],
               [-2.76742510726862411e-03 * DAYS_PER_YEAR,
                4.99852801234917238e-03 * DAYS_PER_YEAR,
                2.30417297573763929e-05 * DAYS_PER_YEAR],
               2.85885980666130812e-04 * SOLAR_MASS),

    'uranus': ([1.28943695621391310e+01,
                -1.51111514016986312e+01,
                -2.23307578892655734e-01],
               [2.96460137564761618e-03 * DAYS_PER_YEAR,
                2.37847173959480950e-03 * DAYS_PER_YEAR,
                -2.96589568540237556e-05 * DAYS_PER_YEAR],
               4.36624404335156298e-05 * SOLAR_MASS),

    'neptune': ([1.53796971148509165e+01,
                 -2.59193146099879641e+01,
                 1.79258772950371181e-01],
                [2.68067772490389322e-03 * DAYS_PER_YEAR,
                 1.62824170038242295e-03 * DAYS_PER_YEAR,
                 -9.51592254519715870e-05 * DAYS_PER_YEAR],
                5.15138902046611451e-05 * SOLAR_MASS)}
  
    body1_body2_keys = BODIES.keys()
    cdef list body1_body2 = list(combinations(body1_body2_keys, 2))    

    cdef float px,py,pz
    ref = BODIES[reference]
    px=0.0
    py=0.0
    pz=0.0

    for body in BODIES.keys():
        (r, [vx, vy, vz], m) = BODIES[body]
        px -= vx * m
        py -= vy * m
        pz -= vz * m
        
    (r, v, m) = ref
    v[0] = px / m
    v[1] = py / m
    v[2] = pz / m

    for _ in range(loops):
        advance(0.01,iterations,BODIES,body1_body2)
        print(report_energy(BODIES,body1_body2))


if __name__ == '__main__':   
    print(timeit.timeit(lambda:nbody(100, 'sun', 20000), number=1))



