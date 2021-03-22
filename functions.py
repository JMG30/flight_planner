"""
Various auxiliary functions for calculations, e.g. projection center
locations, GSD map.
"""

import os
import time
import traceback
from math import (
    acos,
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

import numpy as np
import scipy.ndimage as ndimage
from osgeo import gdal, osr
from pyproj import Transformer, transform
from PyQt5.QtCore import QVariant
from qgis.analysis import QgsZonalStatistics
from qgis.core import (
    QgsCoordinateReferenceSystem,
    QgsCoordinateTransform,
    QgsFeature,
    QgsField,
    QgsGeometry,
    QgsPointXY,
    QgsProcessingUtils,
    QgsProject,
    QgsRasterBandStats,
    QgsRasterLayer,
    QgsVectorLayer
)


def ground_edge_points(R, Z, threshold, xy, Xs, Ys, Zs,
                       ds, f, crs_DTM, crs_pc, transformer):
    """Return ground coordinates of points representing edges of photo."""

    Z_DTM = ds.GetRasterBand(1).ReadAsArray()
    xyf = np.append(xy, np.ones((xy.shape[0], 1)) * -f, axis=1)
    XY = np.zeros(xy.shape)
    XY_prev = np.ones(xy.shape) * 1000
    Z = np.ones(xy.shape[0]) * Z
    counter = 0
    # interpolating X, Y until given threshold will be reached
    while not threshold_reached(XY, XY_prev, threshold):
        XY_prev = np.array(XY)

        X = Xs + (Z - Zs) * np.divide(np.dot(xyf, R[0]), np.dot(xyf, R[2]))
        Y = Ys + (Z - Zs) * np.divide(np.dot(xyf, R[1]), np.dot(xyf, R[2]))
        XY = np.column_stack((X, Y))

        if crs_DTM == crs_pc:
            # transformation to pixel coordinates
            column, row = crs2pixel(ds.GetGeoTransform(), X, Y)
        else:
            # earlier transformation to DTM CRS
            X_DTM, Y_DTM = transf_coord(transformer, X, Y)
            column, row = crs2pixel(ds.GetGeoTransform(), X_DTM, Y_DTM)

        rc_array = np.column_stack((row, column)).T
        # interpolating heights at X, Y
        Z = ndimage.map_coordinates(Z_DTM, rc_array, output=np.float)

        # protection against too long iteration
        if counter > 100:
            break
        counter += 1

    return XY


def image_edge_points(size_sensor, size_across, size_along, Z, Zs, focal,
                      mean_res):
    """Create a list of coordinates of points that represent the edges
    of the image in the image's coordinate system."""
    # approximate ground size Lx, Ly of the image
    W = Zs - Z
    Ly = size_across * size_sensor * W / focal
    Lx = size_along * size_sensor * W / focal
    # number of points along the edges of the image
    num_y = Ly / mean_res
    num_x = Lx / mean_res

    # max x and y coordinate of the image
    x_max = size_sensor * size_along / 2
    y_max = size_sensor * size_across / 2
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
    all_edges = np.vstack((left_edge, top_edge, right_edge, bottom_edge))

    return all_edges


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
    lenv2 = np.linalg.norm(v2)
    if lenv1 == 0:
        lenv1 = 0.000000000000000001
    if lenv2 == 0:
        lenv2 = 0.000000000000000001
    # cosine of angle g between vectors
    if 1 >= v1v2 / (lenv1 * lenv2) >= -1:
        g = v1v2 / (lenv1 * lenv2)
    elif v1v2 / (lenv1 * lenv2) > 1:
        g = 1
    elif v1v2 / (lenv1 * lenv2) < -1:
        g = -1
    angle = acos(g) * 180 / pi
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


def projection_centres(alpha, geometry, crs_vect, a_ll, b_ll, a_l_, b_l_,
                       Dx, Dy, Bx, By, Lx, Ly, x, m, H, strip_nr, photo_nr):
    """Create QgsVectorLayer of projection centers with attribute table
    and QgsVectorLayer range of photos at average terrain height."""
    # Dx, Dy - dimensions of bounding box at angle,
    # respectively along and across of the flight direction
    # Bx, By - longitudinal, transverse base between center projections

    # optimization number of strips
    Dy_o = Dy - 2 * (0.5 - x / 100) * Ly
    # exceptions if dimension Dy is too thin (e.g. corridors)
    if Dy_o < 0:
        Dy_o = 0
    Ny = ceil(Dy_o / By) + 1

    if Ny != 1:
        By_o = Dy_o / (Ny - 1)
    else:
        By_o = 0

    # number of photos in a strip
    Nx = ceil(Dx / Bx) + 2 * m + 1
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
                                        round(H, 2), 0, 0, kappa])
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
    upx = geo[0]  # top left x
    upy = geo[3]  # top left y
    resx = geo[1]  # W-E pixel resolution
    resy = geo[5]  # N-S pixel resolution
    skewx = geo[2]
    skewy = geo[4]
    pc = sqrt(skewx ** 2 + resx ** 2)
    pr = sqrt(skewy ** 2 + resy ** 2)
    alpha = acos(resx / pc)
    column = (cos(alpha) * (x - upx) + sin(alpha) * (y - upy)) / fabs(pc)
    row = (cos(alpha) * (upy - y) + sin(alpha) * (x - upx)) / fabs(pr)

    return column, row


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


def photo_gsd_overlay(geom_footprint, crs_vect, crs_rst, raster, Xs, Ys, Zs,
                      Xs_, Ys_, Zs_, f, size_sensor):
    """Return dataset raster including 2 bands: GSD map and logical sum
    of overlapping photos."""

    bound_box = geom_footprint.boundingBox()
    trnsfrm = Transformer.from_crs(crs_vect, crs_rst, always_xy=True)
    if crs_vect != crs_rst:
        x_min_bb, y_max_bb = transf_coord(trnsfrm,
                                          bound_box.xMinimum(),
                                          bound_box.yMaximum())
        x_max_bb, y_min_bb = transf_coord(trnsfrm,
                                          bound_box.xMaximum(),
                                          bound_box.yMinimum())
    else:
        x_min_bb, y_max_bb = bound_box.xMinimum(), bound_box.yMaximum()
        x_max_bb, y_min_bb = bound_box.xMaximum(), bound_box.yMinimum()

    geo_rst = raster.GetGeoTransform()
    pxl_width = geo_rst[1]
    pxl_height = -geo_rst[5]
    uplx = geo_rst[0]
    uply = geo_rst[3]
    # transformation to pixel coordinates
    col_min, row_min = crs2pixel(geo_rst, x_min_bb, y_max_bb)
    col_max, row_max = crs2pixel(geo_rst, x_max_bb, y_min_bb)
    col_min, row_min = int(col_min), int(row_min)
    col_max, row_max = int(col_max), int(row_max)
    # output array for logical sum of overlaying photos
    overlap_array = np.zeros((row_max - row_min, col_max - col_min))
    gsd_array = np.ones((row_max - row_min, col_max - col_min)) * 1000
    vect_vertical = [0, 0, -1]
    vect_camera_axis = [Xs_ - Xs, Ys_ - Ys, Zs_ - Zs]
    # tilt angle of photo
    t = angle_between_vectors(vect_vertical, vect_camera_axis)
    band = raster.GetRasterBand(1)
    rst_array = band.ReadAsArray()
    # Get nodata value from the GDAL band object
    nodata = band.GetNoDataValue()
    # Create a masked array for making calculations without nodata values
    rst_array = np.ma.masked_equal(rst_array, nodata)
    r = 0
    y_cell = uply - row_min * pxl_height - pxl_height / 2
    trnsfrm2 = Transformer.from_crs(crs_rst, crs_vect, always_xy=True)
    for row in rst_array[row_min: row_max]:
        c = 0
        x_cell = uplx + col_min * pxl_width + pxl_width / 2
        for Z in row[col_min: col_max]:
            px, py = transf_coord(trnsfrm2, x_cell, y_cell)
            p = QgsGeometry.fromPointXY(QgsPointXY(px, py))
            if geom_footprint.contains(p):
                overlap_array[r, c] = 1
                if t == 0:
                    gsd_array[r, c] = ((Zs - Z) * size_sensor / f) * 100
                else:
                    # photo's greatest fall line
                    a, b = line(Ys, Ys_, Xs, Xs_)
                    # line perpendicular to photo's greatest fall line
                    if a != 0:
                        a_l_ = -1 / a
                    else:
                        a_l_ = -1 / 0.000000000000000001
                    b_l_ = py - a_l_ * px
                    # projection point on photo's greatest fall line
                    ppx, ppy = lines_intersection(a_l_, b_l_, a, b)
                    # vector projection center - projection point
                    vect_S_pp = [ppx - Xs, ppy - Ys, Z - Zs]
                    beta = angle_between_vectors(vect_vertical, vect_S_pp)
                    direction = angle_between_vectors((vect_camera_axis[0],
                                                       vect_camera_axis[1]),
                                                      (vect_S_pp[0],
                                                       vect_S_pp[1]))
                    if direction >= 90:
                        beta = -beta
                    W = Zs - Z
                    gsd_array[r, c] = size_sensor * (W / f) * cos(radians(beta - t)) \
                                      / cos(radians(beta)) * 100
            c = c + 1
            x_cell = x_cell + pxl_width
        r = r + 1
        y_cell = y_cell - pxl_height

    # saving outputs to raster
    temp_raster = os.path.join(QgsProcessingUtils.tempFolder(), 'control.tif')
    driver = gdal.GetDriverByName('GTiff')
    dataset = driver.Create(temp_raster, xsize=len(row[col_min: col_max]),
                            ysize=len(rst_array[row_min: row_max]),
                            bands=2, eType=gdal.GDT_Float32)
    dataset.GetRasterBand(1).WriteArray(overlap_array)
    dataset.GetRasterBand(2).WriteArray(gsd_array)

    # setting georeference
    geot = [uplx + col_min * pxl_width, geo_rst[1], geo_rst[2],
            uply - row_min * pxl_height, geo_rst[4], geo_rst[5]]
    dataset.SetGeoTransform(geot)
    srs = osr.SpatialReference()
    srs.SetWellKnownGeogCS(crs_rst)
    dataset.SetProjection(srs.ExportToWkt())

    return dataset


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
