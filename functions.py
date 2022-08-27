"""
Various auxiliary functions for calculations, e.g. projection center
locations, GSD map.
"""

import os
import time
import traceback
from math import (
    acos,
    atan,
    atan2,
    ceil,
    cos,
    fabs,
    pi,
    radians,
    sin,
    sqrt,
    tan
)

import matplotlib.path as mpltPath
import numpy as np
import scipy.ndimage as ndimage
from PyQt5.QtCore import QVariant
from qgis.analysis import QgsZonalStatistics
from qgis.core import (
    QgsFeature,
    QgsField,
    QgsGeometry,
    QgsPointXY,
    QgsPoint,
    QgsProject,
    QgsVectorLayer
)


def add_layers_to_canvas(layers, group_name, counter=1):
    """Add grouped layers to the canvas"""
    root = QgsProject.instance().layerTreeRoot()
    group = root.insertGroup(0, f"{group_name}_{counter}")
    QgsProject.instance().addMapLayers(layers, False)
    for layer in layers:
        layer.setName(layer.name() + f"_{counter}")
        group.addLayer(layer)


def change_layer_style(layer, properties):
    """Change layer style according to the properties."""
    renderer = layer.renderer()
    symbol = renderer.symbol()
    renderer.setSymbol(symbol.createSimple(properties))
    layer.triggerRepaint()


def clip_raster(ds, xyf, R, Xs, Ys, Zs, Z_min, trans_v_r, crs_rst, crs_vct):
    """Return DTM clipped by bounding box of photo. Range of bounding box
     is derived from photo's Exterior Orientation Parameters, camera parameters
     and minimum height of DTM"""

    DTM_array = ds.GetRasterBand(1).ReadAsArray()
    focal = xyf[0, 2]
    img_corners = np.vstack(([0, 0, focal], xyf))

    X_min = Xs + (Z_min - Zs) * np.divide(np.dot(img_corners, R[0]),
                                          np.dot(img_corners, R[2]))
    Y_min = Ys + (Z_min - Zs) * np.divide(np.dot(img_corners, R[1]),
                                          np.dot(img_corners, R[2]))

    X_pc, Y_pc = X_min[0], Y_min[0]
    buffer = max(((X_pc - X_min[1:])**2 + (Y_pc - Y_min[1:])**2)**0.5)

    max_range_X = X_pc + buffer
    min_range_X = X_pc - buffer
    max_range_Y = Y_pc + buffer
    min_range_Y = Y_pc - buffer
    range = np.array([[min_range_X, max_range_Y],
                      [min_range_X, min_range_Y],
                      [max_range_X, min_range_Y],
                      [max_range_X, max_range_Y]
                      ])

    if crs_vct != crs_rst:
        X, Y = transf_coord(trans_v_r, range[:, 0], range[:, 1])
        cols, rows = crs2pixel(ds.GetGeoTransform(), X, Y)
    else:
        cols, rows = crs2pixel(ds.GetGeoTransform(), range[:, 0], range[:, 1])

    upper_left_c, upper_left_r = int(min(cols)//1), int(min(rows)//1)
    bottom_right_c, bottom_right_r = int(max(cols)//1), int(max(rows)//1)

    if upper_left_r < 0:
        upper_left_r = 0
    if upper_left_c < 0:
        upper_left_c = 0

    if bottom_right_r > DTM_array.shape[0]:
        bottom_right_r = DTM_array.shape[0]
    if bottom_right_c > DTM_array.shape[1]:
        bottom_right_c = DTM_array.shape[1]

    x0, y0 = pixel2crs(ds.GetGeoTransform(), upper_left_c, upper_left_r)
    clipped_DTM = np.array(DTM_array[upper_left_r: bottom_right_r+1,
                           upper_left_c: bottom_right_c+1])
    updated_geotransform = list(ds.GetGeoTransform())
    updated_geotransform[0] = x0
    updated_geotransform[3] = y0

    return clipped_DTM, updated_geotransform


def create_flight_line(waypoints_lyr, crs_vect):
    """Create flight line passing through all the waypoints."""
    flight_line = QgsVectorLayer("LineStringZ?crs=" + str(crs_vect),
                                        "flight_line", "memory")
    pr = flight_line.dataProvider()
    waypoints = []
    for w in waypoints_lyr.getFeatures():
        x = w.geometry().asPoint().x()
        y = w.geometry().asPoint().y()
        z = w.attribute('Alt. ASL [m]')
        pnt = QgsPoint(x, y, z)
        waypoints.append(pnt)

    feat = QgsFeature()
    feat.setGeometry(QgsGeometry.fromPolyline(waypoints))
    pr.addFeature(feat)
    flight_line.updateExtents()
    return flight_line


def create_waypoints(projection_centres, crs_vect):
    """Create points where altitude or direction of flight change."""

    waypoints_layer = QgsVectorLayer("Point?crs=" + str(crs_vect),
                              "waypoints", "memory")
    pr = waypoints_layer.dataProvider()
    pr.addAttributes([QgsField("Waypoint Number", QVariant.Int),
                    QgsField("X [m]", QVariant.Double),
                    QgsField("Y [m]", QVariant.Double),
                    QgsField("Alt. ASL [m]", QVariant.Double),
                    QgsField("Alt. AGL [m]", QVariant.Double)])
    waypoints_layer.updateFields()

    strips_nr = int(projection_centres.maximumValue(0))
    feats = projection_centres.getFeatures()
    featList = [feat.attributes()[:6] + [feat.geometry()] for feat in feats]
    featList.sort(key=lambda x: x[1])

    waypoint_nr = 1
    for strip_nr in range(1, strips_nr+1):
        strip = [f for f in featList if int(f[0]) == strip_nr]
        start_waypoint = strip[0]
        end_waypoint = strip[-1]

        x_start = start_waypoint[-1].asPoint().x()
        y_start = start_waypoint[-1].asPoint().y()
        x_end = end_waypoint[-1].asPoint().x()
        y_end = end_waypoint[-1].asPoint().y()

        feat_pnt = QgsFeature()
        pnt_start = QgsPointXY(x_start, y_start)
        feat_pnt.setGeometry(QgsGeometry.fromPointXY(pnt_start))
        feat_pnt.setAttributes([waypoint_nr] + start_waypoint[2:6])
        pr.addFeature(feat_pnt)
        waypoint_nr += 1

        feat_pnt = QgsFeature()
        pnt_end = QgsPointXY(x_end, y_end)
        feat_pnt.setGeometry(QgsGeometry.fromPointXY(pnt_end))
        feat_pnt.setAttributes([waypoint_nr] + end_waypoint[2:6])
        pr.addFeature(feat_pnt)
        waypoint_nr += 1

        waypoints_layer.updateExtents()
    return waypoints_layer


def z_at_3d_line(pnt, start_pnt, end_pnt):
    """Return "z" coordinate for point with known x,y
    lying on line in space defined by start and end points.
    """
    x1, y1, z1 = start_pnt
    x2, y2, z2 = end_pnt
    x, y = pnt[:2]

    if x1 != x2:
        t = (x - x1) / (x2 - x1)
    else:
        t = (y - y1) / (y2 - y1)
    z = t * (z2 - z1) + z1
    return z


def simplify_profile(vertices, epsilon):
    """Reduces the number of vertices in the line, keeping its main shape.
    It is based on the Douglas-Peucker simplification algorithm but
    with the vertical distance instead of perpendicular.
    """
    hmax = 0.0
    index = 0
    for i in range(1, len(vertices) - 1):
        z = z_at_3d_line(vertices[i], vertices[0], vertices[-1])
        h = abs(z - vertices[i][2])
        if h > hmax:
            index = i
            hmax = h

    if hmax >= epsilon:
        results = simplify_profile(vertices[:index+1], epsilon)[:-1]\
                  + simplify_profile(vertices[index:], epsilon)
    else:
        results = [vertices[0], vertices[-1]]

    return results


def distance2d(a, b):
    return sqrt((a[0] - b[0]) ** 2 + (a[1] - b[1]) ** 2)


def points_pixel_centroids(geotransform, shape):
    """Return pixel centroids for the raster."""

    upx = geotransform[0]
    upy = geotransform[3]
    xscale = geotransform[1]
    yscale = geotransform[5]
    xskew = geotransform[2]
    yskew = geotransform[4]
    pc = sqrt(xscale ** 2 + yskew ** 2)
    pr = sqrt(yscale ** 2 + xskew ** 2)
    alpha = acos(xscale / pc)

    x_grid = np.arange(0, shape[1]*pc, pc)
    y_grid = np.arange(0, -shape[0]*pr, -pr)
    xv, yv = np.meshgrid(x_grid[:shape[1]], y_grid[:shape[0]])

    xx = xv.reshape((-1, 1))
    yy = yv.reshape((-1, 1))

    x_start = upx + 1 / 2 * pc
    y_start = upy - 1 / 2 * pr
    centroid_x = x_start + np.cos(-alpha)*xx + np.sin(-alpha)*yy
    centroid_y = y_start + -np.sin(-alpha)*xx + np.cos(-alpha)*yy

    return np.hstack((centroid_x, centroid_y))


def overlap_photo(footprint_vertices, geotransform, clipped_DTM_shape):
    """Return logical array of photo's footprint."""

    raster_centroids = points_pixel_centroids(geotransform, clipped_DTM_shape)

    path = mpltPath.Path(footprint_vertices)
    centroids_inside = path.contains_points(raster_centroids)

    logical_array = centroids_inside.reshape((clipped_DTM_shape[0], -1))

    max_row, max_col = np.argwhere(logical_array > 0).max(axis=0)
    min_row, min_col = np.argwhere(logical_array > 0).min(axis=0)

    trimed_logical_array = logical_array[min_row:max_row+1, min_col:max_col+1]

    upper_left_x, upper_left_y = pixel2crs(geotransform, min_col, min_row)
    trimed_geotransform = geotransform[:]
    trimed_geotransform[0] = upper_left_x
    trimed_geotransform[3] = upper_left_y

    return trimed_logical_array, trimed_geotransform


def gsd(DTM, geotransform, Xs, Ys, Zs, Xs_, Ys_, Zs_, f, size_sensor):
    """Return GSD array."""

    vect_vertical = np.array([0, 0, -1])
    vect_camera_axis = [Xs_ - Xs, Ys_ - Ys, Zs_ - Zs]
    # tilt angle of photo
    t = angle_between_vectors(vect_vertical, vect_camera_axis)
    if t == 0:
        gsd_array = ((Zs - DTM) * size_sensor / f) * 100
    else:
        # photo's greatest fall line
        a, b = line(Ys, Ys_, Xs, Xs_)
        # line perpendicular to photo's greatest fall line
        if a != 0:
            a_l_ = -1 / a
        else:
            a_l_ = -1 / 0.000000000000000001

        pxpy = points_pixel_centroids(geotransform, DTM.shape)
        px = pxpy[:, 0]
        py = pxpy[:, 1]

        b_l_ = py - a_l_ * px
        # projection point on photo's greatest fall line
        ppx, ppy = lines_intersection(a_l_, b_l_, a, b)
        # vector projection center - projection point
        Z = DTM
        vect_S_pp = np.array([ppx - Xs, ppy - Ys, (Z - Zs).flatten()])

        beta = angle_between_vectors(vect_vertical, vect_S_pp)

        direction = angle_between_vectors((vect_camera_axis[0],
                                           vect_camera_axis[1]),
                                          (vect_S_pp[0],
                                           vect_S_pp[1]))

        correct_beta = np.where(direction >= 90, -beta, beta).reshape(Z.shape)

        W = Zs - Z
        gsd_array = size_sensor * (W/f) * np.cos(np.radians(correct_beta - t)) \
                          / np.cos(np.radians(correct_beta)) * 100
    return gsd_array


def ground_edge_points(R, Z, threshold, xyf, Xs, Ys, Zs,
                       Z_DTM, geotransform, crs_DTM, crs_pc, transformer):
    """Return ground coordinates of points representing edges of photo."""

    XY = np.zeros((xyf.shape[0], 2))
    XY_prev = np.ones((xyf.shape[0], 2)) * 1000
    Z = np.ones(xyf.shape[0]) * Z
    counter = 0

    while not threshold_reached(XY, XY_prev, threshold):
        XY_prev = np.array(XY)

        X = Xs + (Z - Zs) * np.divide(np.dot(xyf, R[0]), np.dot(xyf, R[2]))
        Y = Ys + (Z - Zs) * np.divide(np.dot(xyf, R[1]), np.dot(xyf, R[2]))
        XY = np.column_stack((X, Y))

        if crs_DTM == crs_pc:
            column, row = crs2pixel(geotransform, X, Y)
        else:
            X_DTM, Y_DTM = transf_coord(transformer, X, Y)
            column, row = crs2pixel(geotransform, X_DTM, Y_DTM)

        rc_array = np.column_stack((row, column)).T
        Z = ndimage.map_coordinates(Z_DTM, rc_array, output=np.float)

        # protection against too long iteration
        if counter > 100:
            break
        counter += 1

    return XY


def image_edge_points(camera, Z, Zs, mean_res):
    """Create a list of coordinates of points that represent the edges
    of the image in the image's coordinate system."""
    # approximate ground size Lx, Ly of the image
    W = Zs - Z
    Ly = camera.pixels_across_track * camera.sensor_size * W / camera.focal_length
    Lx = camera.pixels_along_track * camera.sensor_size * W / camera.focal_length
    # number of points along the edges of the image
    num_y = Ly / mean_res
    num_x = Lx / mean_res

    # max x and y coordinate of the image
    x_max = camera.sensor_size * camera.pixels_along_track / 2
    y_max = camera.sensor_size * camera.pixels_across_track / 2
    # x or y coordinates to build later [x, y] edges
    y_vertical = np.linspace(-y_max, y_max, int(num_y)).reshape(-1, 1)
    x_horizontal = np.linspace(-x_max, x_max, int(num_x)).reshape(-1, 1)
    x_vertical = np.ones(int(num_y)).reshape(-1, 1)
    y_horizontal = np.ones(int(num_x)).reshape(-1, 1)

    # coordinates of points, that represent 4 edges of image
    left_edge = np.hstack((x_vertical * -x_max, y_vertical))
    top_edge = np.hstack((x_horizontal, y_horizontal * y_max))
    right_edge = np.hstack((x_vertical * x_max, y_vertical * -1))
    bottom_edge = np.hstack((x_horizontal * -1, y_horizontal * -y_max))
    xy = np.vstack((left_edge, top_edge, right_edge, bottom_edge))
    xyf = np.append(xy, np.ones((xy.shape[0], 1)) * -camera.focal_length, axis=1)

    return xyf


def threshold_reached(xy, xy_previous, threshold):
    """Check if distance between coordinates from iteration i and i-1
    is less than given threshold."""
    dx2_dy2 = (xy - xy_previous)**2
    d = (dx2_dy2[:, 0] + dx2_dy2[:, 1])**0.5
    return all(d < threshold)


def angle_between_vectors(v1, v2):
    """Return angle between two 2D vectors."""
    v1v2 = np.dot(v1, v2)
    lenv1 = np.linalg.norm(v1)
    lenv2 = np.linalg.norm(v2, axis=0)
    # cosine of angle g between vectors
    g = np.array(v1v2 / (lenv1 * lenv2))
    g[g > 1] = 1
    g[g < -1] = -1
    angle = np.arccos(g) * 180 / pi
    return angle


def bounding_box_at_angle(alpha, geom):
    """Calculate the two equations of the bounding box at the given
    angle and its dimensions Dx and Dy. The equations describe lines
    that follow the sides of the bounding box."""
    # exception due to tan(90) and tan(270) is equal to infinity
    if alpha != 90 and alpha != 270:
        # variable_ll means parallel, variable_l_ means perpendicular
        # to main direction of flight
        a_ll = tan(alpha * pi / 180)

        # exception for division by zero
        if a_ll != 0:
            a_l_ = -1 / a_ll
        else:
            a_l_ = -1 / 0.000000000000000001

        # parallel and perpendicular line to the direction of flight
        # through the center of the area of interest
        x_centr = geom.centroid().asPoint().x()
        y_centr = geom.centroid().asPoint().y()
        b_ll = y_centr - a_ll * x_centr
        b_l_ = y_centr - a_l_ * x_centr

        # converting the coefficients of both lines y = ax + b 
        # to the form Ax + By + C = 0 
        A_ll = a_ll
        B_ll = -1
        C_ll = b_ll
        A_l_ = a_l_
        B_l_ = -1
        C_l_ = b_l_

        # the purpose of all the code below is to find the equations of
        # the two lines that coincide the sides of the range rectangle
        vrtx_dist_ll = []
        vrtx_dist_l_ = []
        # compute distances from parallel and perpendicular lines
        # to the flight direction to every vertex of geometry
        for vertex in range(len(geom.convertToType(1).asPolyline())):
            vX = geom.vertexAt(vertex).x()
            vY = geom.vertexAt(vertex).y()
            d_ll = (A_ll * vX + B_ll * vY + C_ll) / sqrt(A_ll ** 2 + B_ll ** 2)
            vrtx_dist_ll.append(d_ll)
            d_l_ = (A_l_ * vX + B_l_ * vY + C_l_) / sqrt(A_l_ ** 2 + B_l_ ** 2)
            vrtx_dist_l_.append(d_l_)

        # index of vertices with max and min distance from the lines
        i1_ll = vrtx_dist_ll.index(max(vrtx_dist_ll))
        i2_ll = vrtx_dist_ll.index(min(vrtx_dist_ll))
        i1_l_ = vrtx_dist_l_.index(max(vrtx_dist_l_))
        i2_l_ = vrtx_dist_l_.index(min(vrtx_dist_l_))
        # calculate factors b of equation y = ax + b
        b1_ll = geom.vertexAt(i1_ll).y() - a_ll * geom.vertexAt(i1_ll).x()
        b2_ll = geom.vertexAt(i2_ll).y() - a_ll * geom.vertexAt(i2_ll).x()
        b1_l_ = geom.vertexAt(i1_l_).y() - a_l_ * geom.vertexAt(i1_l_).x()
        b2_l_ = geom.vertexAt(i2_l_).y() - a_l_ * geom.vertexAt(i2_l_).x()
        # calculate dimensions of bounding box
        Dy = fabs(b1_ll - b2_ll) / sqrt(A_ll ** 2 + B_ll ** 2)
        Dx = fabs(b1_l_ - b2_l_) / sqrt(A_l_ ** 2 + B_l_ ** 2)
        # select "b" of line lying "to the left" of flight direction
        if alpha > 90 and alpha < 270:
            b_ll = min(b1_ll, b2_ll)
        else:
            b_ll = max(b1_ll, b2_ll)
        # select "b" of line lying "behind" the direction of flight
        if alpha >= 0 and alpha <= 180:
            b_l_ = min(b1_l_, b2_l_)
        else:
            b_l_ = max(b1_l_, b2_l_)
    else:
        x_max = geom.boundingBox().xMaximum()
        x_min = geom.boundingBox().xMinimum()
        y_max = geom.boundingBox().yMaximum()
        y_min = geom.boundingBox().yMinimum()
        # calculate dimensions of bounding box
        Dx = y_max - y_min
        Dy = x_max - x_min
        # lines lying "to the left" and "behind" of flight direction
        if alpha == 270:
            a_ll, b_ll = line(y_max, y_min, x_max, x_max)
            a_l_, b_l_ = line(y_max, y_max, x_min, x_max)
        else:
            a_ll, b_ll = line(y_max, y_min, x_min, x_min)
            a_l_, b_l_ = line(y_min, y_min, x_min, x_max)
    return a_ll, b_ll, a_l_, b_l_, Dx, Dy


def forward(strip, photo, nr_photos_in_strip):
    """Return dictionary with strip and photo numbers
    for the forward direction of corridor flight."""
    strips_forward = {}
    for seg, n in nr_photos_in_strip.items():
        photos = []
        for _ in range(1, n + 1):
            photos.append(photo)
            photo += 1
        strips_forward[seg] = {strip: photos}
        strip += 1

    return strip, photo, strips_forward


def backward(strip, photo, nr_photos_in_strip):
    """Return dictionary with strip and photo numbers
    for the backward direction of corridor flight."""
    strips_backward = {}
    for seg, n in reversed(nr_photos_in_strip.items()):
        photos = []
        for _ in range(1, n + 1):
            photos.append(photo)
            photo += 1
        strips_backward[seg] = {strip: photos[::-1]}
        strip += 1

    return strip, photo, strips_backward


def corridor_flight_numbering(feats_exp_lines, buff_exp_lines, Bx, By,
    len_across, mult_base, x_percent, segments):
    """Return dictionary with number of strips and photos
    for each segment of corridor flight."""
    nr_photos_in_strip = {}
    for feat_exp in feats_exp_lines:
        x_start = feat_exp.geometry().asPolyline()[0].x()
        y_start = feat_exp.geometry().asPolyline()[0].y()
        x_end = feat_exp.geometry().asPolyline()[1].x()
        y_end = feat_exp.geometry().asPolyline()[1].y()
        # equation of corridor line
        a_line, b_line = line(y_start, y_end, x_start, x_end)
        angle = atan(a_line) * 180 / pi

        if angle < 0:
            angle = angle + 180
        if y_end - y_start < 0:
            angle = angle + 180

        featbuff_exp = buff_exp_lines.getFeature(feat_exp.id())
        # geometry object of line buffer
        geom_line_buf = featbuff_exp.geometry()
        a, b, a2, b2, Dx, Dy = bounding_box_at_angle(angle, geom_line_buf)
        Nx, Ny = strips_projection_centres_number(Dx, Dy, Bx, By,
            len_across, mult_base, x_percent)
        Nx = Nx - 2

        nr_photos_in_strip[f"segment_{feat_exp.id()}"] = Nx

    photo = 1
    strip = 1
    all_directions = []
    for direction in range(1, Ny+1):
        if direction % 2 != 0:
            last_strip, last_photo, strips_in_direction = forward(strip,
                photo, nr_photos_in_strip)
            strip = last_strip
            photo = last_photo
        else:
            last_strip, last_photo, strips_in_direction = backward(strip,
                photo, nr_photos_in_strip)
            strip = last_strip
            photo = last_photo
        all_directions.append(strips_in_direction)

    ordered_segments = {}
    for n in range(1, segments+1):
        segment_list = [d[f'segment_{n}'] for d in all_directions]
        segment_dict = {}
        for strip in segment_list:
            segment_dict.update(strip)
        ordered_segments[f'segment_{n}'] = segment_dict
        
    return ordered_segments


def strips_projection_centres_number(Dx, Dy, Bx, By, Ly, m, x):
    """Return number of strips Ny and projection centres Nx
    for Area of Interst or one segment of corridor flight."""
    # Dx, Dy - dimensions of bounding box at angle,
    # respectively along and across of the flight direction
    # Bx, By - longitudinal, transverse base between center projections

    # optimization number of strips
    Dy_o = Dy - 2 * (0.5 - x / 100) * Ly
    # exceptions if dimension Dy is too thin (e.g. corridors)
    if Dy_o < 0:
        Dy_o = 0
    Ny = ceil(Dy_o / By) + 1

    # number of photos in a strip
    Nx = ceil(Dx / Bx) + 2 * m + 1

    return Nx, Ny


def projection_centres(alpha, geometry, crs_vect, a_ll, b_ll, a_l_, b_l_,
                       Dx, Dy, Bx, By, Lx, Ly, x, m, H, strip_nr, photo_nr):
    """Create QgsVectorLayer of projection centers with attribute table
    and QgsVectorLayer range of photos at average terrain height."""
    # Dx, Dy - dimensions of bounding box at angle,
    # respectively along and across of the flight direction
    # Bx, By - longitudinal, transverse base between center projections

    Nx, Ny = strips_projection_centres_number(Dx, Dy, Bx, By, Ly, m, x)
    # optimization number of strips
    Dy_o = Dy - 2 * (0.5 - x / 100) * Ly
    # exceptions if dimension Dy is too thin (e.g. corridors)
    if Dy_o < 0:
        Dy_o = 0

    if Ny != 1:
        By_o = Dy_o / (Ny - 1)
    else:
        By_o = 0

    # line moved away from line of bounding box, parallel to flight direction
    # converting the coefficients of both lines y = ax + b
    # to the form Ax + By + C = 0
    A = a_ll
    B = -1
    C1 = b_ll
    if alpha > 90 and alpha <= 270:
        C2 = C1 + (0.5 - x / 100) * Ly * sqrt(A ** 2 + B ** 2)
        if Ny == 1:
            C2 = C1 + Dy / 2 * sqrt(A ** 2 + B ** 2)
    else:
        C2 = C1 - (0.5 - x / 100) * Ly * sqrt(A ** 2 + B ** 2)
        if Ny == 1:
            C2 = C1 - Dy / 2 * sqrt(A ** 2 + B ** 2)
    a1 = a_ll
    b1 = C2

    # center projection centers relative to the bounding box
    D = ((ceil(Dx / Bx)) * Bx - Dx) / 2
    A2 = a_l_
    B2 = -1
    C12 = b_l_
    if 0 <= alpha <= 180:
        C22 = C12 - D * sqrt(A2 ** 2 + B2 ** 2)
    else:
        C22 = C12 + D * sqrt(A2 ** 2 + B2 ** 2)
    a2 = a_l_
    b2 = C22
    # coordinates of the center projection of reference 
    x0, y0 = lines_intersection(a1, b1, a2, b2)
    # increments between pictures in strips
    dx = cos(radians(alpha)) * Bx
    dy = sin(radians(alpha)) * Bx
    # increments for subsequent strips 
    dx0 = cos(radians(alpha) - pi / 2) * By_o
    dy0 = sin(radians(alpha) - pi / 2) * By_o
    # create layer of projection centres and its attribute table
    pc_layer = QgsVectorLayer("Point?crs=" + str(crs_vect),
                              "projection centres", "memory")
    pr = pc_layer.dataProvider()
    pr.addAttributes([QgsField("Strip", QVariant.String),
                      QgsField("Photo Number", QVariant.String),
                      QgsField("X [m]", QVariant.Double),
                      QgsField("Y [m]", QVariant.Double),
                      QgsField("Alt. ASL [m]", QVariant.Double),
                      QgsField("Alt. AGL [m]", QVariant.Double),
                      QgsField("Omega [deg]", QVariant.Double),
                      QgsField("Phi [deg]", QVariant.Double),
                      QgsField("Kappa [deg]", QVariant.Double)])
    pc_layer.updateFields()
    # create layer of ground range photos
    photo_layer = QgsVectorLayer("Polygon?crs=" + str(crs_vect),
                                 "photos", "memory")
    prov_photos = photo_layer.dataProvider()
    prov_photos.addAttributes([QgsField("Strip", QVariant.String),
                               QgsField("Photo Number", QVariant.String)])
    photo_layer.updateFields()
    if 0 <= alpha <= 180:
        kappa = alpha
    else:
        kappa = alpha - 360
    d = sqrt((Lx / 2) ** 2 + (Ly / 2) ** 2)
    theta = fabs(atan2(Ly / 2, Lx / 2))

    # calculate projection centers and ground ranges of photos
    for k in range(Ny):
        n_prev = -m - 1
        # coordinates of strip range
        xs1 = x0 + (-m) * dx + cos(radians(alpha) + theta - pi) * d
        ys1 = y0 + (-m) * dy + sin(radians(alpha) + theta - pi) * d
        xs2 = x0 + (-m) * dx + cos(radians(alpha) - theta + pi) * d
        ys2 = y0 + (-m) * dy + sin(radians(alpha) - theta + pi) * d
        xe3 = x0 + (Nx - m - 1) * dx + cos(radians(alpha) + theta) * d
        ye3 = y0 + (Nx - m - 1) * dy + sin(radians(alpha) + theta) * d
        xe4 = x0 + (Nx - m - 1) * dx + cos(radians(alpha) - theta) * d
        ye4 = y0 + (Nx - m - 1) * dy + sin(radians(alpha) - theta) * d
        strip_pnts = [QgsPointXY(xs1, ys1), QgsPointXY(xs2, ys2),
                      QgsPointXY(xe3, ye3), QgsPointXY(xe4, ye4)]
        geom_strip = QgsGeometry.fromPolygonXY([strip_pnts])
        common_part = geom_strip.intersection(geometry)

        for n in range(-m, Nx - m):
            xi = x0 + n * dx
            yi = y0 + n * dy
            # coordinates of ground photo range
            x1 = xi + cos(radians(alpha) + theta - pi) * d
            y1 = yi + sin(radians(alpha) + theta - pi) * d
            x2 = xi + cos(radians(alpha) - theta + pi) * d
            y2 = yi + sin(radians(alpha) - theta + pi) * d
            x3 = xi + cos(radians(alpha) + theta) * d
            y3 = yi + sin(radians(alpha) + theta) * d
            x4 = xi + cos(radians(alpha) - theta) * d
            y4 = yi + sin(radians(alpha) - theta) * d
            feat_poly = QgsFeature()
            pnts = [QgsPointXY(x1, y1), QgsPointXY(x2, y2),
                    QgsPointXY(x3, y3), QgsPointXY(x4, y4)]
            geom_poly = QgsGeometry.fromPolygonXY([pnts])
            xp = xi + cos(radians(alpha) + pi / 2) * Ly / 2
            yp = yi + sin(radians(alpha) + pi / 2) * Ly / 2
            xk = xi + cos(radians(alpha) - pi / 2) * Ly / 2
            yk = yi + sin(radians(alpha) - pi / 2) * Ly / 2
            central_line = QgsGeometry.fromPolylineXY([QgsPointXY(xp, yp),
                                                       QgsPointXY(xk, yk)])
            # check if projection centre can be skipped
            if central_line.distance(common_part) <= m * Bx:
                photo_nr += 1
                if fabs(n - n_prev) != 1:
                    strip_nr += 1
                s_nr = '%(s_nr)04d' % {'s_nr': strip_nr}
                p_nr = '%(p_nr)05d' % {'p_nr': photo_nr}
                n_prev = n
                feat_pnt = QgsFeature()
                pnt = QgsPointXY(xi, yi)
                feat_pnt.setGeometry(QgsGeometry.fromPointXY(pnt))
                feat_pnt.setAttributes([s_nr, p_nr, round(xi, 2), round(yi, 2),
                                        round(H, 2), None, 0, 0, kappa])
                pr.addFeature(feat_pnt)
                pc_layer.updateExtents()
                feat_poly.setGeometry(geom_poly)
                feat_poly.setAttributes([s_nr, p_nr])
                prov_photos.addFeature(feat_poly)
                photo_layer.updateExtents()

        # reverse order of numbering photos of odd strips
        if k % 2 == 0:
            first_p = int(p_nr) + 1
            first_s = int(s_nr) + 1
        else:
            update_order(k, first_p, first_s, p_nr, s_nr, pc_layer)

        x0 = x0 + dx0
        y0 = y0 + dy0
    return pc_layer, photo_layer, strip_nr, photo_nr


def update_order(k, first_p, first_s, p_nr, s_nr, pc_layer):
    list_p = list(range(first_p, int(p_nr) + 1))
    list_s = list(range(first_s, int(s_nr) + 1))
    i = len(list_p) - 1
    j = len(list_s) - 1
    nr_strp_prev = list_s[0]
    for f in pc_layer.getFeatures():
        nr_zdj = int(f.attribute('Photo Number'))
        nr_strp = int(f.attribute('Strip'))

        if nr_zdj in list_p:
            p_nr = '%(p_nr)05d' % {'p_nr': list_p[i]}
            i -= 1
            if nr_strp != nr_strp_prev:
                j -= 1

            s_nr = '%(s_nr)04d' % {'s_nr': list_s[j]}
            pc_layer.startEditing()
            pc_layer.changeAttributeValue(f.id(), 1, p_nr)
            pc_layer.changeAttributeValue(f.id(), 0, s_nr)
            pc_layer.commitChanges()

            nr_strp_prev = nr_strp


def crs2pixel(geo, x, y):
    """Transform coordinates from CRS to pixel coordinates."""
    upx = geo[0]
    upy = geo[3]
    xscale = geo[1]
    yscale = geo[5]
    xskew = geo[2]
    yskew = geo[4]
    pc = sqrt(xscale ** 2 + yskew ** 2)
    pr = sqrt(yscale ** 2 + xskew ** 2)
    alpha = acos(xscale / pc)
    column = (cos(alpha) * (x - upx) + sin(alpha) * (y - upy)) / fabs(pc)
    row = (cos(alpha) * (upy - y) + sin(alpha) * (x - upx)) / fabs(pr)

    return column, row


def pixel2crs(geo, c, r):
    """Transform coordinates from pixel to CRS coordinates."""
    upx = geo[0]
    upy = geo[3]
    xscale = geo[1]
    yscale = geo[5]
    xskew = geo[2]
    yskew = geo[4]
    pc = sqrt(xscale ** 2 + yskew ** 2)
    pr = sqrt(yscale ** 2 + xskew ** 2)
    alpha = acos(xscale / pc)
    x = pc*cos(alpha)*c + pr*sin(alpha)*r + upx
    y = pc*sin(alpha)*c - pr*cos(alpha)*r + upy

    return x, y


def line(ya, yb, xa, xb):
    """Calculate the coefficients a, b of the equation y = ax + b."""
    dy = ya - yb
    dx = xa - xb
    if dx != 0:
        a = dy / dx
        b = ya - (dy / dx) * xa
    else:
        a = dy / 0.000000000000000001
        b = ya - (dy / 0.000000000000000001) * xa
    return a, b


def lines_intersection(a1, b1, a2, b2):
    """Calculate coordinates of intersection two lines."""
    x = (b2 - b1) / (a1 - a2)
    y = (b1 * a2 - b2 * a1) / (a2 - a1)
    return x, y


def rotation_matrix(omega, phi, kappa):
    """Return rotation matrix."""
    phi = radians(phi)
    omega = radians(omega)
    kappa = radians(kappa)

    R = np.array([
        [cos(phi) * cos(kappa), -cos(phi) * sin(kappa), sin(phi)],
        [sin(omega) * sin(phi) * cos(kappa) + cos(omega) * sin(kappa),
         -sin(omega) * sin(phi) * sin(kappa) + cos(omega) * cos(kappa),
         -sin(omega) * cos(phi)],
        [-cos(omega) * sin(phi) * cos(kappa) + sin(omega) * sin(kappa),
         cos(omega) * sin(phi) * sin(kappa) + sin(omega) * cos(kappa),
         cos(omega) * cos(phi)]
    ])
    return R


def transf_coord(transformer, x, y):
    """Transform coordinates between two CRS."""
    x_transformed, y_transformed = transformer.transform(x, y)
    return x_transformed, y_transformed


def minmaxheight(vector, raster):
    """Return max and min value of raster clipped by vector layer."""

    zone_stats = QgsZonalStatistics(vector, raster, 'pre-', 1,
                                    QgsZonalStatistics.Statistics(
                                        QgsZonalStatistics.Min |
                                        QgsZonalStatistics.Max))
    zone_stats.calculateStatistics(None)

    for f in vector.getFeatures():
        min_h = f.attribute('pre-min')
        max_h = f.attribute('pre-max')

    min_idx = vector.fields().lookupField('pre-min')
    max_idx = vector.fields().lookupField('pre-max')
    vector.startEditing()
    vector.deleteAttributes([min_idx, max_idx])
    vector.commitChanges()

    return min_h, max_h


def save_error():
    """Save error traceback as file."""

    error_path = os.path.join(
        os.path.dirname(os.path.abspath(__file__)), 'Error_log.txt')
    with open(error_path, 'a') as error_file:
        error_file.write(time.ctime(time.time()) + '\n')
        error_file.write(traceback.format_exc() + '\n')
