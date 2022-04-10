import os
import traceback
from math import (
    ceil,
    cos,
    fabs,
    pi,
    sin
)

import scipy.ndimage as ndimage
import numpy as np
from osgeo import gdal, osr
from pyproj import Transformer
from PyQt5.QtCore import pyqtSignal, QObject
from PyQt5.QtGui import QColor
from qgis.core import (
    QgsColorRampShader,
    QgsCoordinateReferenceSystem,
    QgsFeature,
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
    ground_edge_points,
    image_edge_points,
    crs2pixel,
    rotation_matrix,
    transf_coord,
    minmaxheight,
    save_error,
    clip_raster,
    overlap_photo,
    gsd
)


class Worker(QObject):
    """Maintain hard work to lighten main thread of plugin."""

    finished = pyqtSignal(object)
    error = pyqtSignal(Exception, basestring)
    progress = pyqtSignal(float)
    enabled = pyqtSignal(bool)

    def __init__(self, **data):
        """Initialize attributes depending on the worker."""
        QObject.__init__(self)

        self.layer = data.get('pointLayer')
        self.crs_vct = data.get('crsVectorLayer')
        self.DTM = data.get('DTM')
        self.layer_pol = data.get('polygonLayer')
        self.crs_rst = data.get('crsRasterLayer')
        self.height_f = data.get('hField')
        self.omega_f = data.get('omegaField')
        self.phi_f = data.get('phiField')
        self.kappa_f = data.get('kappaField')
        self.camera = data.get('camera')
        self.overlap_bool = data.get('overlap')
        self.gsd_bool = data.get('gsd')
        self.footprint_bool = data.get('footprint')
        self.threshold = data.get('threshold')
        self.theta = data.get('theta')
        self.dist = data.get('distance')
        self.w = data.get('height')
        self.s = data.get('strips')
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

            Z_srtm = self.DTM.GetRasterBand(1).ReadAsArray()
            nodata = self.DTM.GetRasterBand(1).GetNoDataValue()
            rst_array = np.ma.masked_equal(Z_srtm, nodata)
            Z_min = np.nanmin(rst_array)

            uplx_r, xres_r, xskew_r, \
            uply_r, yskew_r, yres_r = self.DTM.GetGeoTransform()

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
                omega = feature.attribute(self.omega_f)
                phi = feature.attribute(self.phi_f)
                kappa = feature.attribute(self.kappa_f)

                R = rotation_matrix(omega, phi, kappa)
                clipped_DTM, clipped_geot = clip_raster(self.DTM, xyf_corners,
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
                    self.progress.emit(progress_c / float(feat_count) * 100)

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
                        self.progress.emit(progress_c / float(ds_count) * 100)
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
                renderer = footprint_lay.renderer()
                symbol = renderer.symbol()
                prop = {'color': '255,0,0,30', 'color_border': '#000000',
                        'width_border': '0.2'}
                my_symbol = symbol.createSimple(prop)
                renderer.setSymbol(my_symbol)
                footprint_lay.triggerRepaint()
                result.append(footprint_lay)
            if self.killed is False:
                self.progress.emit(100)
        except Exception as e:
            self.error.emit(e, traceback.format_exc())
            save_error()
        self.finished.emit(result)
        self.enabled.emit(True)


    def run_followingTerrain(self):
        result = []
        try:
            photo_layer = QgsVectorLayer("Polygon?crs=" + str(self.crs_vct),
                                         "photo", "memory")
            provPhoto = photo_layer.dataProvider()
            featPhoto = QgsFeature()
            feat_count = self.layer.featureCount()
            progress_c = 0
            step = feat_count // 1000
            feats = self.layer.getFeatures()
            for f in feats:
                if self.killed is True:
                    # kill request received, exit loop early
                    break
                # projection center coordinates
                xg = f.geometry().asPoint().x()
                yg = f.geometry().asPoint().y()
                kappa = float(f.attribute('Kappa [deg]')) * pi / 180
                # range of photo
                x1 = xg + cos(kappa + self.theta - pi) * self.dist
                y1 = yg + sin(kappa + self.theta - pi) * self.dist
                x2 = xg + cos(kappa - self.theta + pi) * self.dist
                y2 = yg + sin(kappa - self.theta + pi) * self.dist
                x3 = xg + cos(kappa + self.theta) * self.dist
                y3 = yg + sin(kappa + self.theta) * self.dist
                x4 = xg + cos(kappa - self.theta) * self.dist
                y4 = yg + sin(kappa - self.theta) * self.dist
                photo = [QgsPointXY(x1, y1), QgsPointXY(x2, y2),
                         QgsPointXY(x3, y3), QgsPointXY(x4, y4)]
                geomPhoto = QgsGeometry.fromPolygonXY([photo])
                featPhoto.setGeometry(geomPhoto)
                provPhoto.addFeature(featPhoto)
                # min and max height DTM in the range of photo
                hMin, hMax = minmaxheight(photo_layer, self.DTM)
                mean_h = (hMax + hMin) / 2
                w0 = self.w + mean_h
                self.layer.startEditing()
                self.layer.changeAttributeValue(f.id(), 4, round(w0, 2))
                self.layer.commitChanges()
                photo_layer.startEditing()
                photo_layer.deleteFeature(f.id())
                photo_layer.commitChanges()
                # increment progress
                progress_c += 1
                if step == 0 or progress_c % step == 0:
                    self.progress.emit(progress_c / float(feat_count) * 100)
            if self.killed is False:
                self.progress.emit(100)
                # deleting reduntant fields
                self.layer.startEditing()
                self.layer.deleteAttributes([8, 9, 10])
                self.layer.commitChanges()
                self.layer_pol.startEditing()
                self.layer_pol.deleteAttributes([2, 3])
                self.layer_pol.commitChanges()
                # changing layer style
                renderer = self.layer_pol.renderer()
                symbol = renderer.symbol()
                prop = {'color': '200,200,200,30', 'color_border': '#000000',
                        'width_border': '0.2'}
                my_symbol = symbol.createSimple(prop)
                renderer.setSymbol(my_symbol)
                self.layer_pol.triggerRepaint()
                self.layer_pol.setName('photos')
                self.layer.setName('projection centres')
                result.append(self.layer)
                result.append(self.layer_pol)
        except Exception as e:
            # forward the exception upstream
            self.error.emit(e, traceback.format_exc())
            save_error()
        self.finished.emit(result)
        self.enabled.emit(True)

    def run_altitudeStrip(self):
        result = []
        try:
            progress_c = 0
            step = self.s // 1000
            feat_strip = QgsFeature()
            for t in range(1, self.s + 1):

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
                        xg_max = f.geometry().asPoint().x()
                        yg_max = f.geometry().asPoint().y()
                    if int(f.attribute('Photo Number')) < nrP_min:
                        nrP_min = int(f.attribute('Photo Number'))
                        xg_min = f.geometry().asPoint().x()
                        yg_min = f.geometry().asPoint().y()
                    kappa = float(f.attribute('Kappa [deg]')) * pi / 180
                    if self.tab_widg_cor:
                        BuffNr = int(f.attribute('BuffNr'))

                # range of the strip
                # ZLY ZASIEG!!!!!! (UCINA NIEWIELKA CZESC STRIP PRZEZ ZLE WIERZCHOLKI)
                xmin1 = xg_min + cos(kappa + self.theta - pi) * self.dist
                ymin1 = yg_min + sin(kappa + self.theta - pi) * self.dist
                xmin2 = xg_min + cos(kappa - self.theta + pi) * self.dist
                ymin2 = yg_min + sin(kappa - self.theta + pi) * self.dist
                xmax1 = xg_max + cos(kappa + self.theta) * self.dist
                ymax1 = yg_max + sin(kappa + self.theta) * self.dist
                xmax2 = xg_max + cos(kappa - self.theta) * self.dist
                ymax2 = yg_max + sin(kappa - self.theta) * self.dist
                one_strip = [QgsPointXY(xmin1, ymin1),
                             QgsPointXY(xmin2, ymin2),
                             QgsPointXY(xmax1, ymax1),
                             QgsPointXY(xmax2, ymax2)]
                g_strip = QgsGeometry.fromPolygonXY([one_strip])

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
                mean_h = h_max - (h_max - h_min) / 3
                w0 = self.w + mean_h

                # update altitude flight
                self.layer.startEditing()
                for k in range(nrP_min, nrP_max + 1):
                    photo_nr = '%(Photo Number)05d' % {'Photo Number': k}
                    ph_nr_iterator = self.layer.getFeatures('"Photo Number" = ' + str(photo_nr))
                    for f in ph_nr_iterator:
                        ph_nr = f.id()
                    self.layer.changeAttributeValue(ph_nr, 4, round(w0, 2))
                self.layer.commitChanges()
                # increment progress
                progress_c += 1
                if step == 0 or progress_c % step == 0:
                    self.progress.emit(progress_c / self.s * 100)
            if self.killed is False:
                self.progress.emit(100)
                # deleting redundant fields
                if self.tab_widg_cor:
                    self.layer.startEditing()
                    self.layer.deleteAttributes([8, 9, 10])
                    self.layer.commitChanges()
                    self.layer_pol.startEditing()
                    self.layer_pol.deleteAttributes([2, 3])
                    self.layer_pol.commitChanges()

                # changing layer style
                renderer = self.layer_pol.renderer()
                symbol = renderer.symbol()
                prop = {'color': '200,200,200,30', 'color_border': '#000000',
                        'width_border': '0.2'}
                my_symbol = symbol.createSimple(prop)
                renderer.setSymbol(my_symbol)
                self.layer_pol.triggerRepaint()
                # changing layers name
                self.layer_pol.setName('photos')
                self.layer.setName('projection centres')
                result.append(self.layer)
                result.append(self.layer_pol)
        except Exception as e:
            # forward the exception upstream
            self.error.emit(e, traceback.format_exc())
            save_error()
        self.finished.emit(result)
        self.enabled.emit(True)

    def kill(self):
        self.killed = True
