# Flight Planner
QGIS plugin for planning photogrammetric flights.

Plugin consists of two parts:
1. design of photogrammetric flight plan. User needs to provide:
   - camera parameters (focal length, sensor pixel size, number of pixels along/across track),
   - desirable GSD,
   - maximum and minimum terrain heights in the Area of Interest,
   - two margins of exceeding photos outside the AoI.
   
   If user loads the DTM, the plugin allows to choose from two additional
methods of flight altitude planning (except to the basic method 'one common altitude for
the entire project'):
   - altitude for each strip,
   - terrain following.
   
   For corridor type flight plan user needs *.shp layer showing axes of flight
and sets buffer size. For block type flight plan user needs *.shp layer presenting Area of Interest
and sets flight direction. The results are 2 layers: projection centers of photos
(with attribute table containing External Orientation parameters)
and photos size at mean terrain height.


2. assessment of the flight (project or already done)
in the form of output:
   - vector layer of real photos coverage (footprint),
   - raster layer of the logical sum of overlapping photos,
   - raster layer of ground sampling distance (GSD).
To conduct assessment user needs to providing projection centers layer
(with External Orientation parameters), camera parameters
and DTM.

[Installation](https://github.com/JMG30/flight_planner/wiki/Installation)

[More detailed Guide](https://github.com/JMG30/flight_planner/wiki/Guide)
