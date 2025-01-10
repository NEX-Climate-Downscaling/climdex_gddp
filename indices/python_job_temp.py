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

def main():
    cluster = LocalCluster(
        n_workers=20,
        threads_per_worker=1,
        timeout='3600s',
        memory_limit='20GB',
    )
    client = Client(cluster)
    
    models=['INM-CM4-8','GFDL-CM4','CanESM5','CMCC-ESM2','BCC-CSM2-MR','MIROC6','ACCESS-ESM1-5','FGOALS-g3','EC-Earth3-Veg-LR','ACCESS-CM2',
 'CMCC-CM2-SR5','CNRM-CM6-1','GFDL-ESM4','IPSL-CM6A-LR','MIROC-ES2L',
 'MRI-ESM2-0','TaiESM1','CNRM-ESM2-1','KIOST-ESM','NorESM2-MM','MPI-ESM1-2-HR','INM-CM5-0','EC-Earth3',
 'GISS-E2-1-G','GFDL-CM4_gr2','MPI-ESM1-2-LR','UKESM1-0-LL','NESM3','HadGEM3-GC31-LL','HadGEM3-GC31-MM','KACE-1-0-G','NorESM2-LM']
    projections=["historical","ssp126", "ssp245","ssp370","ssp585"]
    print("start the jobs")
    for model in models[:8]:
        for project in projections:
            try:
                os.system("mkdir -p /nobackupp28/skhajehe/gddp-indices/"+model+"/"+project+"/tasmax")

                if glob.glob("/nex/datapool/nex-gddp-cmip6/"+model+"/"+project+"/*/tasmax/*v1.2.nc"):
                    netcdf_list_temp=glob.glob("/nex/datapool/nex-gddp-cmip6/"+model+"/"+project+"/*/tasmax/*v1.2.nc")
                elif glob.glob("/nex/datapool/nex-gddp-cmip6/"+model+"/"+project+"/*/tasmax/*v1.1.nc"):
                    netcdf_list_temp=glob.glob("/nex/datapool/nex-gddp-cmip6/"+model+"/"+project+"/*/tasmax/*v1.1.nc")
                else:
                    netcdf_list_temp=glob.glob("/nex/datapool/nex-gddp-cmip6/"+model+"/"+project+"/*/tasmax/*.nc")
                complete_proj=xr.open_mfdataset(netcdf_list_temp)
                complete_proj['lon'] = (complete_proj['lon'] + 180) % 360 - 180
                complete_proj = complete_proj.sortby(complete_proj.lon)
                complete_proj=ifun.kelvin_to_fahrenheit(complete_proj)

                complete_proj_monthly=(complete_proj['tasmax']).resample(time="1MS").max(dim="time",skipna=True)
                complete_proj_monthly=complete_proj_monthly.where(complete_proj['tasmax'].isel(time=0).notnull())
                # os.system("mkdir -p /nobackupp28/skhajehe/gddp-indices/"+model+"/"+project+"/tasmax")
                complete_proj_monthly.to_dataset().to_netcdf("/nobackupp28/skhajehe/gddp-indices/"+model+"/"+project+"/tasmax/monthly_max.nc")
                print(model+"-"+project+" is Done!")
            except:
                print(model+"-"+project+" doesn't exist!")
                pass
if __name__ == "__main__":
    main()
