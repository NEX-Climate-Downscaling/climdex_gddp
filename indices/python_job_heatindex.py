import xarray as xr
import numpy as np
import dask.array as da
import matplotlib.pyplot as plt
import rasterio
from rasterio.crs import CRS
import os
from dask.distributed import Client, LocalCluster
from datetime import datetime, timedelta
import glob
import indices_function as ifun
import warnings
warnings.filterwarnings('ignore')

def heat_index(elev,sph,temp):
    """
    Calculates the NOAA Heat Index (HI). It combines air temperature and 
    relative humidity to determine an apparent temperature. This is the 
    method from Lans P. Rothfusz 1990: https://www.weather.gov/media/ffc/ta_htindx.PDF 
    and Steadman, 1979. Full equation in S5 in Schwingshackl et al., 2021: 
    https://agupubs.onlinelibrary.wiley.com/doi/full/10.1029/2020EF001885

    Arguments:
        temp (float or array-like): dry bulb air temperature in [°F]
        rh (float or array-like): relative humidity [%]


    Returns:
        hi (float or array-like): Heat Index in [°C]

    """
    pair = 101.3 * (((293 - 0.0065 * elev) / 293) ** (9.8 / (0.0065 * 286.9)))
    ea = sph * pair / (0.622 + 0.378 * sph)
    temp_c=(temp - 32) * 5/9
    es =0.6108 * np.exp(17.27 * temp_c / (temp_c + 237.3))
    rh=((ea/es)*100).clip(0,100)
    
    # based on NOAA approximation which states original eqn not valid for HI < 80 F
    
    approx = 0.5 * (temp + 61 + 1.2 * (temp - 68) + 0.094 * rh)
    hi1 = -42.379 + 2.04901523*temp + 10.14333127*rh - 0.22475541*temp*rh\
        - 6.83783e-3*temp**2 - 5.481717e-2*rh**2 + 1.22874e-3*temp**2*rh\
        + 8.5282e-4*temp*rh**2 - 1.99e-6*temp**2*rh**2

    # modifications  from 1990, NOAA
    condition = (approx + temp) / 2 < 80
    condition_1= (rh < 13) & (80 <= temp) & (temp < 112)
    condition_2= (rh > 85) & (80 <= temp) & (temp < 87)
    if condition.any():
        hi = np.where(condition, approx, hi1) 
    elif condition_1.any():
        hia1 = np.where(condition_1, ((13 - rh) / 4) * np.sqrt((17 - np.abs(temp - 95))/17), 0)
        hi = hi1 - hia1
    elif condition_2.any():
        hia2 = np.where(condition, ((rh - 85) / 10) * (87 - temp) / 5, 0)
        hi = hi1 + hia2
    else:
        hi = hi1
        
    return hi

def main(heat_index):
    cluster = LocalCluster(
        n_workers=40,
        threads_per_worker=1,
        timeout='3600s',
        memory_limit='50GB',
    )
    client = Client(cluster)
    models=['INM-CM4-8','GFDL-CM4','CMCC-ESM2','BCC-CSM2-MR','MIROC6','ACCESS-ESM1-5','FGOALS-g3','EC-Earth3-Veg-LR','ACCESS-CM2',
 'CMCC-CM2-SR5','CNRM-CM6-1','GFDL-ESM4','IPSL-CM6A-LR','MIROC-ES2L',
 'MRI-ESM2-0','TaiESM1','CNRM-ESM2-1','KIOST-ESM','NorESM2-MM','MPI-ESM1-2-HR','INM-CM5-0','EC-Earth3',
 'GISS-E2-1-G','GFDL-CM4_gr2','MPI-ESM1-2-LR','UKESM1-0-LL','NESM3','HadGEM3-GC31-LL','HadGEM3-GC31-MM','KACE-1-0-G','NorESM2-LM']
    projections=["historical","ssp126", "ssp245","ssp370","ssp585"]
    print("start the jobs")
    for model in models[:16]:
        for project in projections:
            try:
                os.system("mkdir -p /nobackupp28/skhajehe/gddp-indices/"+model+"/"+project+"/heat")
                if glob.glob("/nex/datapool/nex-gddp-cmip6/"+model+"/"+project+"/*/tasmax/*v1.2.nc"):
                    netcdf_list_temp=glob.glob("/nex/datapool/nex-gddp-cmip6/"+model+"/"+project+"/*/tasmax/*v1.2.nc")
                elif glob.glob("/nex/datapool/nex-gddp-cmip6/"+model+"/"+project+"/*/tasmax/*v1.1.nc"):
                    netcdf_list_temp=glob.glob("/nex/datapool/nex-gddp-cmip6/"+model+"/"+project+"/*/tasmax/*v1.1.nc")
                else:
                    netcdf_list_temp=glob.glob("/nex/datapool/nex-gddp-cmip6/"+model+"/"+project+"/*/tasmax/*.nc")
                complete_proj_temp=xr.open_mfdataset(netcdf_list_temp)
                complete_proj_temp['lon'] = (complete_proj_temp['lon'] + 180) % 360 - 180
                complete_proj_temp = complete_proj_temp.sortby(complete_proj_temp.lon)
                complete_proj_temp=ifun.kelvin_to_fahrenheit(complete_proj_temp)
                # complete_proj_temp_rechunk=complete_proj_temp.chunk({"lat": 20, "lon": 20, "time": 365})

                netcdf_list_sph=glob.glob("/nex/datapool/nex-gddp-cmip6/"+model+"/"+project+"/*/huss/*.nc")
                complete_proj_sph=xr.open_mfdataset(netcdf_list_sph)
                complete_proj_sph['lon'] = (complete_proj_sph['lon'] + 180) % 360 - 180
                complete_proj_sph = complete_proj_sph.sortby(complete_proj_sph.lon)
                # complete_proj_sph_rechunk=complete_proj_sph.chunk({"lat": 20, "lon": 20, "time": 365})

                elevation=xr.open_dataset("nex_dem.nc").sortby('lat', ascending=True)
                # elevation_rechunck=elevation.isel(time=0).chunk({"lat": 20, "lon": 20})

                heat_index_max= xr.apply_ufunc(
                    heat_index,
                    elevation.isel(time=0),
                    complete_proj_sph["huss"],
                    complete_proj_temp["tasmax"],
                    output_dtypes=[complete_proj_temp.tasmax.dtype],
                    dask="parallelized")

                heat_index_monthly=(heat_index_max).resample(time="1MS").mean(dim="time",skipna=True)
                heat_index_monthly.to_netcdf("/nobackupp28/skhajehe/gddp-indices/"+model+"/"+project+"/heat/heatmax.nc")
            # os.system("mv heatmax.nc /nobackupp28/skhajehe/gddp-indices/"+model+"/"+project+"/heat/heatmax.nc")  
                print(model+"-"+project+" is Done!")

            except:
                print(model+"-"+project+" doesn't exist!")
                pass

if __name__ == "__main__":
    main(heat_index)

