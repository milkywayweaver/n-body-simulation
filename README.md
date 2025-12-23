<h1>Introduction</h1>
<p>N-Body problem is a problem involving the movement of N bodies (in this case celestial objects) due to gravitational forces asserted by each other.
  Unlike two body problem that has analytical solution, N-body problem currently can not be solved analytically. Therefore, numerical approximation is required.
  A challange to overcome when doing numerical integration is to get stable results without exploding the calculation time. This can be done in several ways, including choosing the right integrator.
  In this project, we use basic integrators such as forward euler, leapfrog, and hermite integrator to approximate orbits of star systems.</p>

<p>All of the integrators used in this projects can approximate the obrit of a predefined star systems, such as star systems that results in a circular orbit or star systems that results in a "infinity" orbit.
  But all of the integrators struggles with star systems that iniated at random. This is due to close encounters happening between the stars that explodes the acceleration. There are ways to prevent this but has not yet implemented in this project.</p>
  
<h1>Reference</h1>
Hut, P., & Makino, J. (2007). Moving stars around. The Art of Computational Science.
