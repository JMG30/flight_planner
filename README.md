# Flight Planner
QGIS plugin for planning photogrammetric flights.

Plugin consists of two parts:
1. Flight Design panel, where user needs to provide:
   - camera parameters (focal length, sensor pixel size, pixel image width and height),
   - desirable GSD or Altitude Above Ground Level,
   - maximum and minimum terrain heights in the Area of Interest (only for 'One Altitude ASL For Entire Flight' mode),
   - two margins of exceeding photos outside the AoI.
   
   If user loaded Digital Terrain Model DTM, the plugin allows to choose from two additional
methods of flight altitude (except the basic method 'One Altitude ASL For The Entire Flight'):
   - Separate Altitude ASL For Each Strip,
   - Terrain Following.
   
   For corridor type flight user needs *.shp layer showing axes of flight
and sets buffer size. For block type flight user needs *.shp layer presenting Area of Interest
and sets flight direction. The results are 4 layers: projection centers of photos
(with attribute table containing External Orientation parameters), photos (size at mean terrain height),
waypoints and flight line.


2. Quality control panel for assessment of the flight (project or already done) in the form of output:
   - vector layer of real photos coverage (footprint),
   - raster layer of the nubmer of overlapping images,
   - raster layer of ground sampling distance (GSD).
To conduct assessment user needs to provide projection centers layer
(with External Orientation parameters), camera parameters and DTM.

[More detailed Guide](https://github.com/JMG30/flight_planner/wiki/Guide)

[Installation](https://github.com/JMG30/flight_planner/wiki/Installation)
