#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
File Name: load_nc.py
Description:






@author: Jiangtao Liu
liujiangtao3@gmail.com
Anaconda 3 64bit, Python 3.7.0
------------------------------------------
Change Activities:



"""
import numpy as np
import pandas as pd
import xarray as xr
import os
import json

class NetCDFDataset():

    def __init__(self):
        self.ds = None  # This will store the generated dataset

    def array2nc(self,
                 station_ids=None,
                 lat=None,
                 lon=None,
                 time_range=None,
                 time_series_data=None,
                 time_series_variables=None,
                 time_series_variables_units=None,
                 static_data=None,
                 static_variables=None,
                 static_variables_units=None,
                 meta_data_dict={},
                 ):
        """
        Generate a xarray dataset from the given data.
        Input:
            station_ids: list of station ids
            lat: list of latitudes
            lon: list of longitudes
            time_range: list of time stamps
            time_series_data: 3D array of time series data: [station_id, time, variable]
            time_series_variables: list of variable names
            time_series_variables_units: list of variable units
            static_data: 2D array of static data: [station_id, variable]
            static_variables: list of variable names
            static_variables_units: list of variable units
            meta_data_dict: data description, e.g. metadata = {
                'description': "This dataset contains synthetic temperature data for demonstration purposes.",
                'source': "Simulated Data Generator v1.0",
                # 'notes': "This is a note",  # Uncomment to add the note
                'contact': "contact@example.com"                }
        Output:
            ds: xarray dataset
        Sytntax:
            ds = generate_nc_data(station_ids=station_ids,

        Example:
            station_count = 10
            time_points = 5
            time_series_data = np.random.rand(station_count, time_points, 3)
            static_data = np.random.rand(station_count, 2)
            lat = np.random.rand(station_count) * (90) - 45  # Random latitudes between -45 and 45
            lon = np.random.rand(station_count) * (180) - 90  # Random longitudes between -90 and 90
            time_series_variables = ["temperature", "precipitation", "radiation"]
            time_series_variables_units = ["°C", "mm", "W/m^2"]
            static_variables = ["elevation", "land_cover_type"]
            static_variables_units = ["m", "category"]
            ds = generate_nc_data(
                station_ids=np.arange(station_count),
                lat=lat,
                lon=lon,
                time_range=pd.date_range("2000-01-01", periods=time_points, freq="D"),
                time_series_data=time_series_data,
                time_series_variables=time_series_variables,
                time_series_variables_units=time_series_variables_units,
                static_data=static_data,
                static_variables=static_variables,
                static_variables_units=static_variables_units,
    )

        """
        data_dict = {}

        if station_ids is None:
            station_ids = np.arange(len(time_series_data))

        coords = {
            "station_ids": station_ids,
            "lat": (("station_ids",), lat) if lat is not None else None,
            "lon": (("station_ids",), lon) if lon is not None else None
        }

        if time_series_data is not None:
            for i, var_name in enumerate(time_series_variables):
                data_dict[var_name] = (("station_ids", "time"), time_series_data[:, :, i])
            times = time_range
            coords["time"] = times

        if static_data is not None:
            for i, var_name in enumerate(static_variables):
                data_dict[var_name] = (("station_ids",), static_data[:, i])

        ds = xr.Dataset(data_dict, coords=coords)

        if time_series_variables_units:
            for var_name, unit in zip(time_series_variables, time_series_variables_units):
                if var_name in ds:
                    ds[var_name].attrs['units'] = unit

        if static_variables_units:
            for var_name, unit in zip(static_variables, static_variables_units):
                if var_name in ds:
                    ds[var_name].attrs['units'] = unit

        # Store the variable names in the dataset attributes
        ds.attrs['time_series_variables'] = time_series_variables if time_series_variables else []
        ds.attrs['static_variables'] = static_variables if static_variables else []
        # units
        ds.attrs['time_series_variables_units'] = time_series_variables_units if time_series_variables_units else []
        ds.attrs['static_variables_units'] = static_variables_units if static_variables_units else []

        ds.attrs["metadata"] = json.dumps(meta_data_dict)

        self.ds = ds

        return ds

    def save_to_file(self, filename):
        """Saves the dataset to a netCDF file."""
        if self.ds:
            self.ds.to_netcdf(filename, format='NETCDF4')
        else:
            print("Dataset is empty. Generate a dataset first using array2nc method.")

    def load_nc(self,
                nc_file,
                station_ids=None,
                time_range=None,
                time_series_variables=None,
                static_variables=None,
                ):
        """
        Load a netcdf file into a xarray dataset and select specific data based on input.

        Input:
            nc_file: path to the netcdf file
            station_ids: list of station ids to select
            time_range: 2-element list of start and end times to select
            time_series_variables: list of time series variable names to select
            static_variables: list of static variable names to select

        Output:
            ds: xarray dataset with selected data
        """
        ds = xr.open_dataset(nc_file)

        # Select based on provided station_ids
        if station_ids is not None:
            ds = ds.sel(station_ids=station_ids)

        # Select based on provided time range
        if time_range is not None:
            ds = ds.sel(time=slice(*time_range))

        # Subset the dataset for the provided time series and static variables
        selected_vars = []
        if time_series_variables:
            selected_vars.extend(time_series_variables)
        if static_variables:
            selected_vars.extend(static_variables)
        if selected_vars:
            ds = ds[selected_vars]

        self.ds = ds  # Store the selected dataset in the class attribute

        return ds

    def nc2array(self,
                 nc_file,
                 station_ids=None,
                 time_range=None,
                 time_series_variables=None,
                 static_variables=None,
                 warmup_days=0,
                 add_coords=False,
                 *args,
                 **kwargs,
                 ):
        """
        Load a netcdf file into a xarray dataset and select specific data based on input.
        Input:
            nc_file: path to the netcdf file
            station_ids: list of station ids to select
            time_range: 2-element list of start and end times to select
            time_series_variables: list of time series variable names to select
            static_variables: list of static variable names to select
            warmup_days: number of days to add to the start of the time range to account for warmup, add to the left of
                            the time range
        Output:
            time_series_data: list of time series data arrays
            static_data: list of static data arrays

        """
        self.add_coords = add_coords
        # if warmup_days > 0:
        time_range = [(pd.to_datetime(time_range[0]) - pd.DateOffset(days=warmup_days)).strftime("%Y-%m-%d"),
                      time_range[-1]]

        select_ds = self.load_nc(
            nc_file,
            station_ids=station_ids,
            time_range=time_range,
            time_series_variables=time_series_variables,
            static_variables=static_variables
        )
        assert select_ds.equals(self.ds), "Loaded dataset doesn't match the stored dataset."

        # Extract time series data
        time_series_data = []
        if time_series_variables:
            for var in time_series_variables:
                if var in select_ds:
                    time_series_data.append(select_ds[var].values)
                else:
                    raise ValueError("Variable {} not found in the dataset.".format(var))

        if time_series_data:
            time_series_data = np.stack(time_series_data, axis=-1)  # Convert list of arrays to a single array
        else:
            time_series_data = None

        # Extract static data
        static_data = []
        if static_variables:
            for var in static_variables:
                if var in select_ds:
                    static_data.append(select_ds[var].values)
                else:
                    raise ValueError("Variable {} not found in the dataset.".format(var))

        if self.add_coords:
            # Try to get lat/lon as data variables first, then as coordinates if they don't exist as variables
            try:
                static_data.append(select_ds['lat'].values)
            except KeyError:
                if 'lat' in select_ds.coords:
                    static_data.append(select_ds.coords['lat'].values)
                else:
                    raise KeyError("'lat' not found in dataset variables or coordinates")
                
            try:
                static_data.append(select_ds['lon'].values)
            except KeyError:
                if 'lon' in select_ds.coords:
                    static_data.append(select_ds.coords['lon'].values)
                else:
                    raise KeyError("'lon' not found in dataset variables or coordinates")

        if static_data:
            static_data = np.stack(static_data, axis=-1)  # Convert list of arrays to a single array
        else:
            static_data = None

        # Extract the time range if present
        extracted_time_range = pd.to_datetime(select_ds['time'].values) if 'time' in select_ds else None

        return time_series_data, static_data, extracted_time_range

    def nc2array_grid(self,
                      nc_file,
                      station_ids=None,
                      time_range=None,
                      time_series_variables=None,
                      static_variables=None,
                      warmup_days=0,
                      add_coords=False,
                      sampling_interval=None,
                      ):
        """


        """
        assert station_ids ==None, "station_ids is not supported for grid data."
        # if warmup_days > 0:
        time_range = [(pd.to_datetime(time_range[0]) - pd.DateOffset(days=warmup_days)).strftime("%Y-%m-%d"),
                      time_range[-1]]

        select_ds = self.load_nc(
            nc_file,
            station_ids=station_ids,
            time_range=time_range,
            time_series_variables=time_series_variables,
            static_variables=static_variables + ["mask"]
        )
        assert select_ds.equals(self.ds), "Loaded dataset doesn't match the stored dataset."

        if sampling_interval is not None:
            select_ds = select_ds.sel(lon=slice(None, None, sampling_interval), lat=slice(None, None, sampling_interval))

        masking_index = select_ds["mask"].data == 1
        valid_lat_idx, valid_lon_idx = np.where(masking_index)
        masked_lats = select_ds["lat"].data[valid_lat_idx]
        masked_lons = select_ds["lon"].data[valid_lon_idx]

        # Extract time series data
        time_series_data = []
        if time_series_variables:
            for var in time_series_variables:
                if var in select_ds:

                    masked_values = select_ds[var].data[..., masking_index]  # time, grid
                    masked_values = np.transpose(masked_values, (1, 0))  # grid, time

                    assert np.isnan(masked_values).sum() <= 0

                    time_series_data.append(masked_values)

                else:
                    raise ValueError("Variable {} not found in the dataset.".format(var))

        if time_series_data:
            time_series_data = np.stack(time_series_data, axis=-1)  # Convert list of arrays to a single array
        else:
            time_series_data = None

        # Extract static data
        static_data = []
        if static_variables:
            for var in static_variables:
                if var in select_ds:
                    masked_values = select_ds[var].data[..., masking_index]  # grid

                    if np.isnan(masked_values).sum() > 0:
                        print(f"{var} NaN ratio: {np.isnan(masked_values).sum() / masked_values.size * 100:.2f} %. Fill with mean.")
                        # fill nan with mean
                        masked_values[np.isnan(masked_values)] = np.nanmean(masked_values)

                    static_data.append(masked_values)
                else:
                    raise ValueError("Variable {} not found in the dataset.".format(var))

        if add_coords:
            static_data.append(masked_lats)
            static_data.append(masked_lons)

        if static_data:
            static_data = np.stack(static_data, axis=-1)  # Convert list of arrays to a single array
        else:
            static_data = None

        # Extract the time range if present
        extracted_time_range = pd.to_datetime(select_ds['time'].values) if 'time' in select_ds else None
        self.masked_lons = masked_lons
        self.masked_lats = masked_lats

        return time_series_data, static_data, extracted_time_range
