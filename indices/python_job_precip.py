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
        n_workers=80,
        threads_per_worker=1,
        timeout='3600s',
        memory_limit='50GB',
    )
    client = Client(cluster)
    models=['INM-CM4-8','GFDL-CM4','CanESM5','CMCC-ESM2','BCC-CSM2-MR','MIROC6','ACCESS-ESM1-5','FGOALS-g3','EC-Earth3-Veg-LR','ACCESS-CM2',
 'CMCC-CM2-SR5','CNRM-CM6-1','GFDL-ESM4','IPSL-CM6A-LR','MIROC-ES2L',
 'MRI-ESM2-0','TaiESM1','CNRM-ESM2-1','KIOST-ESM','NorESM2-MM','MPI-ESM1-2-HR','INM-CM5-0','EC-Earth3',
 'GISS-E2-1-G','GFDL-CM4_gr2','MPI-ESM1-2-LR','UKESM1-0-LL','NESM3','HadGEM3-GC31-LL','HadGEM3-GC31-MM','KACE-1-0-G','NorESM2-LM']
    projections=["historical","ssp126", "ssp245","ssp370","ssp585"]
    print("start the jobs")
    for model in models:
        for project in projections:
            try:
                os.system("mkdir -p /nobackupp27/skhajehe/gddp-indices/"+model+"/"+project+"/pr")
                netcdf_list=glob.glob("/nex/datapool/nex-gddp-cmip6/"+model+"/"+project+"/*/pr/*.nc")
                complete_proj=xr.open_mfdataset(netcdf_list)
                complete_proj['lon'] = (complete_proj['lon'] + 180) % 360 - 180
                complete_proj = complete_proj.sortby(complete_proj.lon)
                complete_proj=ifun.mm_inch(complete_proj*86400.0)
                prgtzero=ifun.days_gt_threshold(complete_proj,0)
                prgtzero=prgtzero.where(complete_proj['pr'].isel(time=0).notnull())
                prgtzero.to_netcdf("prgtzero.nc")
                os.system("mv prgtzero.nc /nobackupp27/skhajehe/gddp-indices/"+model+"/"+project+"/pr/prgtzero.nc")  
            except:
                print(model+"-"+project+" doesn't exist!")
                pass

if __name__ == "__main__":
    main()

