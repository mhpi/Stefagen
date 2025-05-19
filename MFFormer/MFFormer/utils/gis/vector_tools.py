from osgeo import ogr
import os
import glob
import numpy as np
import pandas as pd
from MFFormer.utils.sys_tools import re_folder, split_path

# os.environ["PROJ_LIB"] = "/data/jxl6499/anaconda3/pkgs/proj-9.0.0-h93bde94_1/share/proj"
os.environ["PROJ_LIB"] = "/data/jql6620/software_install/anaconda3/envs/TimesNet/lib/python3.8/site-packages/pyproj/proj_dir/share/proj"


class VectorTools(object):

    def __init__(self):
        pass

    def create_shp_from_pts(self,
                            opt_shp,
                            srs,
                            df=None,
                            x_field="lon",
                            y_field="lat",
                            inp_csv=None,
                            lon_list=None,
                            lat_list=None,
                            *args,
                            **kwargs,
                            ):
        """
        create shp from pts(csv, dataframe, lon/lat list)
        It is a specific function for csv2other_format()

        Deprecated Parameters: srs_kwt, save_as_WGS84, EPSG

        Parameters like csv2other_format()

        Character limit for shapefile field:
            No more than 10 characters, otherwise only the first 10 characters are saved.
        """
        self.csv2other_format(opt_file=opt_shp, srs=srs, df=df, x_field=x_field, y_field=y_field, inp_csv=inp_csv,
                              lon_list=lon_list, lat_list=lat_list)

    def clip_shp(self, inp_shp, opt_shp, inp_clip=None, bx=None):
        """
        e.g.
        clip("./datasets/pts.shp", "./datasets/polygon.shp", "./datasets/pts_clip.shp")

        If shapefile is too complex or topological errors then the following problems will occur:
        ERROR 1: TopologyException: Input geom 1 is invalid: Self-intersection at or near point
        don't know how to solve it
        """
        opt_path = os.path.split(opt_shp)[0]
        tmp_path = os.path.join(opt_path, "tmp")
        re_folder(tmp_path)

        if not bx is None:
            inp_clip = os.path.join(tmp_path, "bx.shp")
            self.create_shp_from_bx(opt_shp=inp_clip, bx=bx)

        assert not inp_clip is None

        cmd = "ogr2ogr -overwrite -clipsrc  {} {} {}".format(inp_clip, opt_shp, inp_shp)
        os.system(cmd)

    def pts_within_shp(self,
                       inp_shp,
                       opt_path,
                       EPSG=4326,
                       lon_list=None,
                       lat_list=None,
                       srs=4326):

        """
        e.g.
        df / pts.csv:
        lats, lon, values1, values2
        35,   -110,   10, 20
        36,   -112,   11, 24

        id_list =pts_within_shp("./datasets/polygon.shp", y_field="lats", inp_csv="./pts.csv")
        id_list =pts_within_shp("./datasets/polygon.shp", y_field="lats", df=df)

        e.g.
        lon = [-102, -110]
        lat = [35, 32]
        id_list = pts_within_shp("/datasets/sdb/polygon.shp", lon_list, lat_list)
        """
        opt_path = os.path.join(opt_path, "tmp")
        re_folder(opt_path)
        tmp_shp = os.path.join(opt_path, "clip_pts_by_shp.shp")

        # generate pts.shp
        df = pd.DataFrame({"lon": lon_list, "lat": lat_list})

        df["clip_id"] = np.arange(len(df))
        self.create_shp_from_pts(tmp_shp, srs=srs, df=df, x_field="lon", y_field="lat", EPSG=EPSG)

        # clip
        inp_clip = inp_shp
        opt_clip = os.path.join(opt_path, "clip_pts_by_shp_clip.shp")

        self.clip_shp(inp_shp=tmp_shp, opt_shp=opt_clip, inp_clip=inp_clip)

        # get index
        shp = ogr.Open(opt_clip)
        layer = shp.GetLayer(0)
        id_list = []  # lon, lon
        for feature in layer:
            id_list.append(int(feature.GetField("clip_id")))

        return id_list

    def csv2other_format(self,
                         opt_file,
                         srs,
                         df=None,
                         x_field="lon",
                         y_field="lat",
                         inp_csv=None,
                         lon_list=None,
                         lat_list=None,
                         ):
        """
        convert crd csv file to other format. e.g. shp, geojson, kml

        Parameters
            opt_file: other format file name
            srs: The coordinate reference systems that can be passed are anything supported by the OGRSpatialReference class.
                e.g. "EPSG:4326", WKT CRS file, PROJ.4 string, *.prj file, etc.
            df: crd dataframe including x/lon and y/lat
            x_field: x/lon field name
            y_field: y/lat field name
            inp_csv: crd csv file
            lon_list: list of lon
            lat_list: list of lat

        Returns
            Other format file

        Syntax:
            df / pts.csv:
            lats, lon, values1, values2
            35,   -110,   10, 20
            36,   -112,   11, 24

            csv2other("./datasets/pts.shp", y_field="lats", inp_csv="./pts.csv", srs=4326)
            csv2other("./datasets/pts.shp", y_field="lats", df=df, srs=4326)

            lon = [-102, -110]
            lat = [35, 32]
            csv2other("/datasets/sdb/pts.shp", lon_list, lat_list, srs=4326)
        """
        opt_path, shp_basename, shp_suffix = split_path(opt_file)
        driver = self._suffix2driver(inp_suffix=shp_suffix)

        tmp_path = os.path.join(opt_path, "tmp")
        re_folder(tmp_path)

        if not lon_list is None:
            df = pd.DataFrame({"lon": lon_list, "lat": lat_list, "id": np.arange(len(lon_list))})

        if not df is None:
            inp_csv = os.path.join(tmp_path, "tmp_csv_shp.csv")
            df.to_csv(inp_csv)
        assert not inp_csv is None

        a_srs = self.format_srs(srs)

        cmd = 'ogr2ogr -overwrite -f "{}" -a_srs "{}" {} {} -oo X_POSSIBLE_NAMES={} -oo Y_POSSIBLE_NAMES={}'.format(
            driver, a_srs, opt_file, inp_csv, x_field, y_field)
        os.system(cmd)

    def _suffix2driver(self, inp_suffix):
        """
        Get driver name from suffix
        Not include all drivers.
        https://gdal.org/drivers/vector/index.html

        Parameters
            inp_suffix: str e.g. ".shp"

        Returns
            driver_name: str e.g. "ESRI Shapefile"

        Syntax:
            _suffix2driver(".shp")
        """
        # cmd = "ogrinfo --formats | sort"
        driver_dict = {
            ".shp": "ESRI Shapefile",
            ".csv": "CSV",
            ".kml": "KML",
            ".json": "GeoJSON",
            ".geojson": "GeoJSON",
            ".gml": "GML",
            # ".mdb": "Geomedia",
            ".dwg": "CAD",
            ".gpkg": "GPKG",
        }
        assert inp_suffix in driver_dict.keys()
        driver = driver_dict[inp_suffix]
        return driver

    def format_srs(self, inp_srs):
        """
        inp_srs = "/data/basin.shp", "/data/basin.tif", "/data/basin.kwt", 4326
        """
        # get input tyep
        inp_type = type(inp_srs)
        if inp_type in [int]:
            srs = 'EPSG: {}'.format(inp_srs)
        elif inp_type in [str]:
            if "EPSG" in inp_srs:
                return inp_srs
            srs = self.search_proj_file(inp_ds=inp_srs)
        return srs

    def search_proj_file(self, inp_ds):
        """
        inp_ds: "/data/basin.shp", "/data/basin.tif",  "/data/basin.kwt"
        :return "/data/basin.prj", "/data/basin.twf", "/data/basin.kwt",
        """
        inp_path, basename, suffix = split_path(inp_file=inp_ds)
        if suffix in [".prj"]:
            return inp_ds
        if suffix in [".shp"]:
            prj_file = glob.glob(os.path.join(inp_path, basename + ".prj"))
            assert len(prj_file), ".prj file does not exist!"
            return prj_file[0]

        if suffix in [".wkt"]:
            return inp_ds

        if suffix in [".tif"]:
            prj_file = glob.glob(os.path.join(inp_path, basename + ".tfw"))
            assert len(prj_file), ".tfw file does not exist!"
            return prj_file[0]
