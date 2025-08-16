import os
import traceback
from math import (
    atan,
    atan2,
    ceil,
    cos,
    fabs,
    pi,
    sin,
    sqrt
)

import scipy.ndimage as ndimage
import numpy as np
from osgeo import gdal, osr
from pyproj import Transformer
from PyQt5.QtCore import pyqtSignal, QObject, QVariant
from PyQt5.QtGui import QColor
from qgis.core import (
    QgsColorRampShader,
    QgsCoordinateReferenceSystem,
    QgsFeature,
    QgsField,
    QgsGeometry,
    QgsPointXY,
    QgsProcessingUtils,
    QgsRasterBandStats,
    QgsRasterLayer,
    QgsRasterShader,
    QgsSingleBandPseudoColorRenderer,
    QgsVectorLayer,
)

from .functions import (
    change_layer_style,
    clip_raster,
    create_flight_line,
    create_waypoints,
    crs2pixel,
    distance2d,
    ground_edge_points,
    gsd,
    image_edge_points,
    minmaxheight,
    overlap_photo,
    simplify_profile,
    rotation_matrix,
    save_error,
    transf_coord,
    z_at_3d_line
)


class Worker(QObject):
    """Maintain hard work to lighten main thread of plugin."""

    finished = pyqtSignal(object, basestring)
    error = pyqtSignal(Exception, basestring)
    progress = pyqtSignal(int)
    enabled = pyqtSignal(bool)

    def __init__(self, **data):
        """Initialize attributes depending on the worker."""
        QObject.__init__(self)

        self.layer = data.get('pointLayer')
        self.crs_vct = data.get('crsVectorLayer')
        self.DTM = data.get('DTM')
        self.raster = data.get('raster')
        self.layer_pol = data.get('polygonLayer')
        self.crs_rst = data.get('crsRasterLayer')
        self.height_is_ASL = data.get('height_is_ASL')
        self.height_f = data.get('hField')
        self.omega_f = data.get('omegaField')
        self.phi_f = data.get('phiField')
        self.kappa_f = data.get('kappaField')
        self.camera = data.get('camera')
        self.overlap_bool = data.get('overlap')
        self.gsd_bool = data.get('gsd')
        self.footprint_bool = data.get('footprint')
        self.threshold = data.get('threshold')
        self.tolerance = data.get('tolerance')
        self.theta = data.get('theta')
        self.dist = data.get('distance')
        self.altitude_AGL = data.get('altitude_AGL')
        self.altitude_ASL = data.get('altitude_ASL')
        self.tab_widg_cor = data.get('tabWidg')
        self.g_line_list = data.get('LineRangeList')
        self.geom_aoi = data.get('Range')
        self.killed = False

    def run_control(self):
        """Do the main work for control methods."""
        result = []
        try:

            if self.crs_rst != self.crs_vct:
                transf_vct_rst = Transformer.from_crs(self.crs_vct,
                                                      self.crs_rst,
                                                      always_xy=True)
                transf_rst_vct = Transformer.from_crs(self.crs_rst,
                                                      self.crs_vct,
                                                      always_xy=True)
            else:
                transf_vct_rst = None
                transf_rst_vct = None

            Z_srtm = self.raster.GetRasterBand(1).ReadAsArray()
            nodata = self.raster.GetRasterBand(1).GetNoDataValue()
            rst_array = np.ma.masked_equal(Z_srtm, nodata)
            Z_min = np.nanmin(rst_array)

            uplx_r, xres_r, xskew_r, \
            uply_r, yskew_r, yres_r = self.raster.GetGeoTransform()

            # calculating metric resolution if crs is geographic
            if QgsCoordinateReferenceSystem(self.crs_rst).isGeographic():
                uplx_r_n = uplx_r + xres_r
                uply_r_n = uply_r + yres_r
                uplx_v, uply_v = transf_coord(transf_rst_vct, uplx_r, uply_r)
                uplx_v_n, uply_v_n = transf_coord(transf_rst_vct, uplx_r_n, uply_r_n)
                xres_r = fabs(uplx_v_n - uplx_v)
                yres_r = fabs(uply_v_n - uply_v)
            mean_res = (fabs(xres_r) + fabs(yres_r)) / 2

            # output footprint layer
            footprint_lay = QgsVectorLayer("Polygon?crs=" + str(self.crs_vct),
                                           "footprint", "memory")
            provider = footprint_lay.dataProvider()
            features = self.layer.getFeatures()
            feat_footprint = QgsFeature()
            ds_list = []
            ulx_list = []
            uly_list = []
            lrx_list = []
            lry_list = []

            xyf_corners = self.camera.image_corners()

            feat_count = self.layer.featureCount()
            progress_c = 0
            step = feat_count // 1000
            # creating footprint, overlapping, GSD maps
            for feature in features:
                if self.killed is True:
                    # kill request received, exit loop early
                    break

                Xs = feature.geometry().asPoint().x()
                Ys = feature.geometry().asPoint().y()
                Zs = feature.attribute(self.height_f)
                if not self.height_is_ASL:
                    x, y = Xs, Ys
                    if self.crs_rst != self.crs_vct:
                        x, y = transf_coord(transf_vct_rst, Xs, Ys)
                    terrain_height, res = self.DTM.dataProvider().sample(QgsPointXY(x, y,), 1)
                    Zs = feature.attribute(self.height_f) + terrain_height

                omega = feature.attribute(self.omega_f)
                phi = feature.attribute(self.phi_f)
                kappa = feature.attribute(self.kappa_f)

                R = rotation_matrix(omega, phi, kappa)
                clipped_DTM, clipped_geot = clip_raster(self.raster, xyf_corners,
                                                        R, Xs, Ys, Zs, Z_min,
                                                        transf_vct_rst,
                                                        self.crs_rst,
                                                        self.crs_vct)

                if self.crs_vct != self.crs_rst:
                    Xs_rast, Ys_rast = transf_coord(transf_vct_rst, Xs, Ys)
                    c, r = crs2pixel(clipped_geot, Xs_rast, Ys_rast)
                else:
                    c, r = crs2pixel(clipped_geot, Xs, Ys)

                # getting Z value from DTM under given center projection point
                Z_under_pc = ndimage.map_coordinates(clipped_DTM,
                                                     np.array([[r, c]]).T)[0]

                # edge points list in image space
                xyf = image_edge_points(self.camera, Z_under_pc, Zs, mean_res)

                # ground coordinates of photo edge points
                footprint_vertices = ground_edge_points(R, Z_under_pc,
                                                        self.threshold, xyf, Xs,
                                                        Ys, Zs, clipped_DTM,
                                                        clipped_geot,
                                                        self.crs_rst,
                                                        self.crs_vct,
                                                        transf_vct_rst)

                footprint_pnts = [QgsPointXY(XY[0], XY[1]) for XY in footprint_vertices]
                geom_footprint = QgsGeometry.fromPolygonXY([footprint_pnts])
                feat_footprint.setGeometry(geom_footprint)
                provider.addFeatures([feat_footprint])
                footprint_lay.updateExtents()

                if self.overlap_bool or self.gsd_bool:
                    if self.crs_vct != self.crs_rst:
                        X_rast, Y_rast = transf_coord(transf_vct_rst,
                                                      footprint_vertices[:, 0],
                                                      footprint_vertices[:, 1])
                        footprint_vertices = np.hstack((X_rast.reshape((-1, 1)),
                                                        Y_rast.reshape((-1, 1))))

                    overlap_arr, overlap_geot = overlap_photo(footprint_vertices,
                                                              clipped_geot,
                                                              clipped_DTM.shape)

                    deltac = int(fabs(round((clipped_geot[0] - overlap_geot[0]) / overlap_geot[1], 0)))
                    deltar = int(fabs(round((clipped_geot[3] - overlap_geot[3]) / overlap_geot[5], 0)))
                    fitted_DTM = clipped_DTM[deltar:overlap_arr.shape[0]+deltar, deltac:overlap_arr.shape[1]+deltac]

                    projection_center = np.array([[Xs], [Ys], [Zs]])
                    img_coords = np.array([[0], [0], [-self.camera.focal_length]])
                    image_crs = np.add(projection_center,
                                       np.dot(R, img_coords))
                    X = image_crs[0][0]
                    Y = image_crs[1][0]
                    Z = image_crs[2][0]

                    gsd_array = gsd(fitted_DTM, overlap_geot, Xs, Ys, Zs, X, Y,
                                    Z, self.camera.focal_length, self.camera.sensor_size)

                    gsd_masked = gsd_array * overlap_arr
                    gsd_masked = np.where(gsd_masked == 0, 1000, gsd_masked)

                    ds_list.append([gsd_masked, overlap_arr, overlap_geot])
                    upx, xres, xskew, upy, yskew, yres = overlap_geot
                    cols = overlap_arr.shape[1]
                    rows = overlap_arr.shape[0]
                    ulx = upx  # upper left x
                    uly = upy  # upper left y
                    lrx = upx + cols * xres + rows * xskew  # lower right x
                    lry = upy + cols * yskew + rows * yres  # lower right y
                    ulx_list.append(ulx)
                    uly_list.append(uly)
                    lrx_list.append(lrx)
                    lry_list.append(lry)

                progress_c += 1
                if step == 0 or progress_c % step == 0:
                    self.progress.emit(int(progress_c / float(feat_count) * 100))

            if self.overlap_bool or self.gsd_bool:
                # range of output raster of 'overlap' and 'gsd' maps
                ulx_fp = min(ulx_list)
                uly_fp = max(uly_list)
                lrx_fp = max(lrx_list)
                lry_fp = min(lry_list)
                cols_fp = int(ceil((lrx_fp - ulx_fp) / xres))
                rows_fp = int(ceil((lry_fp - uly_fp) / yres))
                final_overlay = np.zeros((rows_fp, cols_fp))
                final_gsd = np.ones((rows_fp, cols_fp)) * 1000
                geo = [ulx_fp, xres, xskew, uly_fp, yskew, yres]
                ds_count = len(ds_list)
                progress_c = 0
                step = ds_count // 1000
                # combining the results of all photos
                for gsd_array, overlay_array, geot in ds_list:
                    c, r = crs2pixel(geo, geot[0] + xres / 2,
                                     geot[3] + yres / 2)
                    c = int(c)
                    r = int(r)
                    rows, cols = overlay_array.shape[0], overlay_array.shape[1]

                    final_overlay[r:r+rows, c:c+cols] = final_overlay[r:r+rows, c:c+cols] + overlay_array
                    final_gsd[r:r+rows, c:c+cols] = np.where(gsd_array < final_gsd[r:r+rows, c:c+cols], gsd_array, final_gsd[r:r+rows, c:c+cols])

                    progress_c += 1
                    if step == 0 or progress_c % step == 0:
                        self.progress.emit(int(progress_c / float(ds_count) * 100))
                # saving outputs in temporary folder
                tmp_overlay = os.path.join(QgsProcessingUtils.tempFolder(), 'overlay.tif')
                temp_gsd = os.path.join(QgsProcessingUtils.tempFolder(), 'gsd.tif')
                driver = gdal.GetDriverByName('GTiff')
                ds_overlay = driver.Create(tmp_overlay, xsize=cols_fp,
                                           ysize=rows_fp, bands=1,
                                           eType=gdal.GDT_Float32)
                ds_gsd = driver.Create(temp_gsd, xsize=cols_fp, ysize=rows_fp,
                                     bands=1, eType=gdal.GDT_Float32)
                ds_overlay.GetRasterBand(1).WriteArray(final_overlay)
                ds_overlay.GetRasterBand(1).SetNoDataValue(0)
                ds_gsd.GetRasterBand(1).WriteArray(final_gsd)
                ds_gsd.GetRasterBand(1).SetNoDataValue(1000)
                # setting CRS of the outputs
                ds_overlay.SetGeoTransform(geo)
                ds_gsd.SetGeoTransform(geo)
                srs = osr.SpatialReference()
                srs.ImportFromEPSG(int(self.crs_rst.split(":")[1]))
                srs.SetWellKnownGeogCS(self.crs_rst)
                ds_overlay.SetProjection(srs.ExportToWkt())
                ds_gsd.SetProjection(srs.ExportToWkt())
                ds_overlay = None
                ds_gsd = None
                # changing 'logical sum of overlapping images' layer style
                if self.overlap_bool:
                    overlay_layer = QgsRasterLayer(tmp_overlay, "overlapping")
                    overlay_pr = overlay_layer.dataProvider()
                    stats = overlay_pr.bandStatistics(1,
                                                      QgsRasterBandStats.All)
                    max_v = stats.maximumValue
                    fcn = QgsColorRampShader()
                    fcn.setColorRampType(QgsColorRampShader.Exact)
                    lst = []
                    clr_step = int(255 / max_v)
                    j = 1
                    for i in range(int(max_v)):
                        lst.append(QgsColorRampShader.ColorRampItem(j,
                                                                    QColor(0 + clr_step, 0 + clr_step, 0 + clr_step),
                                                                    str(j)))
                        clr_step = clr_step + int(255 / max_v)
                        j = j + 1
                    fcn.setColorRampItemList(lst)
                    shader = QgsRasterShader()
                    shader.setRasterShaderFunction(fcn)
                    renderer = overlay_layer.renderer()
                    renderer = QgsSingleBandPseudoColorRenderer(overlay_pr,
                                                                1, shader)
                    overlay_layer.setRenderer(renderer)
                    result.append(overlay_layer)
                # changing 'gsd' layer style
                if self.gsd_bool:
                    gsd_layer = QgsRasterLayer(temp_gsd, "gsd_map")
                    gsd_pr = gsd_layer.dataProvider()
                    stats = gsd_pr.bandStatistics(1, QgsRasterBandStats.All)
                    min_v = stats.minimumValue
                    max_v = stats.maximumValue
                    fcn = QgsColorRampShader()
                    fcn.setColorRampType(QgsColorRampShader.Interpolated)
                    lst = [QgsColorRampShader.ColorRampItem(
                        min_v, QColor(0, 255, 0), str(min_v)),
                        QgsColorRampShader.ColorRampItem(max_v,
                                                         QColor(255, 0, 0), str(max_v))]
                    fcn.setColorRampItemList(lst)
                    shader = QgsRasterShader()
                    shader.setRasterShaderFunction(fcn)
                    renderer = gsd_layer.renderer()
                    renderer = QgsSingleBandPseudoColorRenderer(gsd_pr,
                                                                1, shader)
                    gsd_layer.setRenderer(renderer)
                    result.append(gsd_layer)
            # changing 'footprint' layer style
            if self.footprint_bool:
                prop = {'color': '255,0,0,30', 'color_border': '#000000',
                        'width_border': '0.2'}
                change_layer_style(footprint_lay, prop)
                result.append(footprint_lay)
            if self.killed is False:
                self.progress.emit(100)
        except Exception as e:
            self.error.emit(e, traceback.format_exc())
            save_error()
        self.finished.emit(result, "quality_control")
        self.enabled.emit(True)


    def run_followingTerrain(self):
        """Update altitude ASL and AGL for Following Terrain altitude type."""
        result = []
        try:
            geotransf = self.raster.GetGeoTransform()
            raster_array = self.raster.GetRasterBand(1).ReadAsArray()
            nodata = self.raster.GetRasterBand(1).GetNoDataValue()
            DTM_array = np.ma.masked_equal(raster_array, nodata)

            pix_width = geotransf[1]
            pix_height = -geotransf[5]
            if self.crs_rst != self.crs_vct:
                transf_vct_rst = Transformer.from_crs(self.crs_vct,
                                                      self.crs_rst,
                                                      always_xy=True)
                transf_rst_vct = Transformer.from_crs(self.crs_rst,
                                                      self.crs_vct,
                                                      always_xy=True)
                uplx = geotransf[0]
                uply = geotransf[3]
                uplx_n = uplx + pix_width
                uply_n = uply + pix_height
                xo, yo = transf_coord(transf_rst_vct, uplx, uply)
                xo1, yo1 = transf_coord(transf_rst_vct, uplx_n, uply_n)
                pix_width = distance2d((xo, yo), (xo1, yo))
                pix_height = distance2d((xo, yo), (xo, yo1))
            raster_diagonal_angle = atan(pix_height/pix_width)*180/pi

            waypoints_layer = QgsVectorLayer("Point?crs=" + str(self.crs_vct),
                                    "waypoints", "memory")
            pr = waypoints_layer.dataProvider()
            pr.addAttributes([QgsField("Waypoint Number", QVariant.Int),
                            QgsField("X [m]", QVariant.Double),
                            QgsField("Y [m]", QVariant.Double),
                            QgsField("Alt. ASL [m]", QVariant.Double),
                            QgsField("Alt. AGL [m]", QVariant.Double)])
            waypoints_layer.updateFields()

            strips_nr = int(self.layer.maximumValue(0))
            feats = self.layer.getFeatures()
            proj_cent_list = [f.attributes()[:2] + [f.id(), f.geometry()] for f in feats]
            proj_cent_list.sort(key=lambda x: x[1])

            progress_c = 0
            step = int(self.layer.maximumValue(0)) // 1000
            waypoint_nr = 1
            for strip_nr in range(1, strips_nr+1):
                if self.killed is True:
                    # kill request received, exit loop early
                    break

                strip_proj_centres = [f for f in proj_cent_list if int(f[0]) == strip_nr]
                pc_coords = [(p[-1].asPoint().x(), p[-1].asPoint().y()) for p in strip_proj_centres]
                pc_coords = np.array(pc_coords)
                first_pc = strip_proj_centres[0]
                last_pc = strip_proj_centres[-1]

                first_x = first_pc[-1].asPoint().x()
                first_y = first_pc[-1].asPoint().y()
                last_x = last_pc[-1].asPoint().x()
                last_y = last_pc[-1].asPoint().y()

                direction = 90
                if last_x-first_x != 0:
                    direction = abs(atan((last_y-first_y) / (last_x-first_x))*180/pi)

                if direction < raster_diagonal_angle:
                    step_profile = pix_width / cos(direction*pi/180)
                else:
                    step_profile = pix_height / sin(direction*pi/180)

                vertices_count = ceil(distance2d((first_x, first_y), (last_x, last_y))/step_profile)
                profile_x = np.linspace(first_x, last_x, vertices_count, endpoint=True)
                profile_y = np.linspace(first_y, last_y, vertices_count, endpoint=True)

                if self.crs_vct != self.crs_rst:
                    prf_x_rst, prf_y_rst = transf_coord(transf_vct_rst, profile_x, profile_y)
                    prf_c, prf_r = crs2pixel(geotransf, prf_x_rst, prf_y_rst)
                    pc_x_rst, pc_y_rst = transf_coord(transf_vct_rst, pc_coords[:, 0], pc_coords[:, 1])
                    pc_c, pc_r = crs2pixel(geotransf, pc_x_rst, pc_y_rst)
                else:
                    prf_c, prf_r = crs2pixel(geotransf, profile_x, profile_y)
                    pc_c, pc_r = crs2pixel(geotransf, pc_coords[:, 0], pc_coords[:, 1])

                profile_z = DTM_array[prf_r.astype(int), prf_c.astype(int)]
                pc_z = DTM_array[pc_r.astype(int), pc_c.astype(int)]

                profile_coords = np.column_stack((profile_x, profile_y, profile_z))
                profile_coords = list(map(list, profile_coords.reshape((-1, 3))))
                simplified_profile = simplify_profile(profile_coords, self.tolerance)
                waypoints_coords = [w[:2] + [w[2] + self.altitude_AGL] for w in simplified_profile]

                strip_proj_centres = [f_pc + [pc_z[e]] for e, f_pc in enumerate(strip_proj_centres)]
                for i in range(len(waypoints_coords[:-1])):
                    start_w = waypoints_coords[i]
                    end_w = waypoints_coords[i+1]

                    waypoint_x = float(start_w[0])
                    waypoint_y = float(start_w[1])
                    waypoint_ASL = float(start_w[-1])
                    waypoint_AGL = round(self.altitude_AGL,2)

                    feat_waypnt = QgsFeature()
                    waypnt = QgsPointXY(waypoint_x, waypoint_y)
                    feat_waypnt.setGeometry(QgsGeometry.fromPointXY(waypnt))
                    feat_waypnt.setAttributes([waypoint_nr, round(waypoint_x,2),
                        round(waypoint_y,2), round(waypoint_ASL,2), waypoint_AGL])
                    pr.addFeature(feat_waypnt)

                    for proj_centre in strip_proj_centres:
                        pc_x = proj_centre[-2].asPoint().x()
                        pc_y = proj_centre[-2].asPoint().y()
                        pc_z_ground = float(proj_centre[-1])

                        if min(start_w[0], end_w[0]) <= pc_x <= max(start_w[0], end_w[0]) \
                        and min(start_w[1], end_w[1]) <= pc_y <= max(start_w[1], end_w[1]):
                            id = int(proj_centre[2])
                            new_ASL =  float(z_at_3d_line((pc_x, pc_y), start_w, end_w))
                            new_AGL = new_ASL - pc_z_ground
                            self.layer.startEditing()
                            self.layer.changeAttributeValue(id, 4, round(new_ASL, 2))
                            self.layer.changeAttributeValue(id, 5, round(new_AGL, 2))
                            self.layer.commitChanges()
                    waypoint_nr += 1

                waypoint_x = float(end_w[0])
                waypoint_y = float(end_w[1])
                waypoint_ASL = float(end_w[-1])

                feat_waypnt = QgsFeature()
                waypnt = QgsPointXY(waypoint_x, waypoint_y)
                feat_waypnt.setGeometry(QgsGeometry.fromPointXY(waypnt))
                feat_waypnt.setAttributes([waypoint_nr, round(waypoint_x,2),
                        round(waypoint_y,2), round(waypoint_ASL,2), waypoint_AGL])
                pr.addFeature(feat_waypnt)
                waypoints_layer.updateExtents()
                waypoint_nr += 1

                # increment progress
                progress_c += 1
                if step == 0 or progress_c % step == 0:
                    self.progress.emit(int(progress_c / float(strips_nr) * 100))
            if self.killed is False:
                self.progress.emit(100)
                flight_line = create_flight_line(waypoints_layer, self.crs_vct)
                style_path = os.path.join(
                        os.path.dirname(os.path.abspath(__file__)), 'flight_line_style.qml')
                flight_line.loadNamedStyle(style_path)
                # deleting reduntant fields
                self.layer.startEditing()
                self.layer.deleteAttributes([9, 10, 11])
                self.layer.commitChanges()
                self.layer_pol.startEditing()
                self.layer_pol.deleteAttributes([2, 3])
                self.layer_pol.commitChanges()
                # changing layer style
                prop = {'color': '200,200,200,30', 'color_border': '#000000',
                        'width_border': '0.2'}
                change_layer_style(self.layer_pol, prop)
                change_layer_style(self.layer, {'size': '1.0'})
                self.layer_pol.setName('photos')
                self.layer.setName('projection centres')
                result.append(self.layer)
                result.append(flight_line)
                result.append(waypoints_layer)
                result.append(self.layer_pol)
        except Exception as e:
            # forward the exception upstream
            self.error.emit(e, traceback.format_exc())
            save_error()
        self.finished.emit(result, "flight_design")
        self.enabled.emit(True)

    def run_altitudeStrip(self):
        result = []
        try:
            strips_count = int(self.layer.maximumValue(0))
            progress_c = 0
            step = strips_count // 1000
            feat_strip = QgsFeature()

            if self.crs_rst != self.crs_vct:
                transf_vct_rst = Transformer.from_crs(self.crs_vct,
                                                      self.crs_rst,
                                                      always_xy=True)
            for t in range(1, strips_count + 1):

                if self.killed is True:
                    # kill request received, exit loop early
                    break

                strip_nr = '%(StripNr)04d' % {'StripNr': t}
                feats = self.layer.getFeatures('"Strip" = ' + str(strip_nr))
                nrP_max = 0
                nrP_min = 1000000
                # finding first and last photo of strip
                for f in feats:
                    if int(f.attribute('Photo Number')) > nrP_max:
                        nrP_max = int(f.attribute('Photo Number'))
                    if int(f.attribute('Photo Number')) < nrP_min:
                        nrP_min = int(f.attribute('Photo Number'))
                    if self.tab_widg_cor:
                        BuffNr = int(f.attribute('BuffNr'))

                # range of the strip
                photos_list = [f.attributes()[:2] + [f.id(), f.geometry()] for f in self.layer_pol.getFeatures()]
                photos_list.sort(key=lambda x: x[1])
                strip_photos = [f for f in photos_list if int(f[0]) == t]

                first_photo = [p for p in strip_photos[0][-1].asPolygon()[0]]
                last_photo = [p for p in strip_photos[-1][-1].asPolygon()[0]]
                points = first_photo + last_photo

                x_pnt = np.array([pnt.x() for pnt in points]).reshape(-1, 1)
                y_pnt = np.array([pnt.y() for pnt in points]).reshape(-1, 1)
                pnts = np.hstack((x_pnt, y_pnt))

                pnt1 = pnts[np.argmin(pnts[:, 0])]
                pnt2 = pnts[np.argmin(pnts[:, 1])]
                pnt3 = pnts[np.argmax(pnts[:, 0])]
                pnt4 = pnts[np.argmax(pnts[:, 1])]

                one_strip = [QgsPointXY(pnt1[0], pnt1[1]),
                             QgsPointXY(pnt2[0], pnt2[1]),
                             QgsPointXY(pnt3[0], pnt3[1]),
                             QgsPointXY(pnt4[0], pnt4[1])]
                g_strip = QgsGeometry.fromPolygonXY([one_strip])
                kappa = float(f.attribute('Kappa [deg]'))
                if kappa in [-90, 0, 90, 180]:
                    g_strip = QgsGeometry.fromPolygonXY([points])
                    g_strip = QgsGeometry.fromRect(g_strip.boundingBox())

                # common part of strip and Area of Interest
                if self.tab_widg_cor:
                    common = g_strip.intersection(self.g_line_list[BuffNr - 1])
                else:
                    common = g_strip.intersection(self.geom_aoi)

                feat_strip.setGeometry(common)
                common_lay = QgsVectorLayer("Polygon?crs=" + str(self.crs_vct),
                                            "row", "memory")
                prov_com = common_lay.dataProvider()
                prov_com.addFeature(feat_strip)

                h_min, h_max = minmaxheight(common_lay, self.DTM)
                avg_terrain_height = h_max - (h_max - h_min) / 3
                altitude_ASL = self.altitude_AGL + avg_terrain_height

                # update altitude flight
                self.layer.startEditing()
                for k in range(nrP_min, nrP_max + 1):
                    photo_nr = '%(Photo Number)05d' % {'Photo Number': k}
                    ph_nr_iterator = self.layer.getFeatures('"Photo Number" = ' + str(photo_nr))
                    for f in ph_nr_iterator:
                        ph_nr = f.id()

                    x = f.geometry().asPoint().x()
                    y = f.geometry().asPoint().y()
                    if self.crs_rst != self.crs_vct:
                        x, y = transf_coord(transf_vct_rst, x, y)

                    terrain_height, res = self.DTM.dataProvider().sample(QgsPointXY(x, y,), 1)
                    altitude_AGL = altitude_ASL - terrain_height
                    self.layer.changeAttributeValue(ph_nr, 5, round(altitude_AGL, 2))
                    self.layer.changeAttributeValue(ph_nr, 4, round(altitude_ASL, 2))
                self.layer.commitChanges()
                # increment progress
                progress_c += 1
                if step == 0 or progress_c % step == 0:
                    self.progress.emit(int(progress_c / strips_count * 100))

            waypoints_layer = create_waypoints(self.layer, self.crs_vct)
            if self.killed is False:
                self.progress.emit(100)
                flight_line = create_flight_line(waypoints_layer, self.crs_vct)
                style_path = os.path.join(
                        os.path.dirname(os.path.abspath(__file__)), 'flight_line_style.qml')
                flight_line.loadNamedStyle(style_path)
                # deleting redundant fields
                if self.tab_widg_cor:
                    self.layer.startEditing()
                    self.layer.deleteAttributes([9, 10, 11])
                    self.layer.commitChanges()
                    self.layer_pol.startEditing()
                    self.layer_pol.deleteAttributes([2, 3])
                    self.layer_pol.commitChanges()

                # changing layer style
                prop = {'color': '200,200,200,30', 'color_border': '#000000',
                        'width_border': '0.2'}
                change_layer_style(self.layer_pol, prop)
                change_layer_style(self.layer, {'size': '1.0'})
                # changing layers name
                self.layer_pol.setName('photos')
                self.layer.setName('projection centres')
                result.append(self.layer)
                result.append(flight_line)
                result.append(waypoints_layer)
                result.append(self.layer_pol)
        except Exception as e:
            # forward the exception upstream
            self.error.emit(e, traceback.format_exc())
            save_error()
        self.finished.emit(result, "flight_design")
        self.enabled.emit(True)

    def kill(self):
        self.killed = True
