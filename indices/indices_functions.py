
def fahrenheit_to_kelvin(temp_F):
    '''
    Convert Fahrenheit to Kelvin
    
    '''
    temp_K=(temp_F-32)*5/9 + 273.15
    
    return temp_K

def kelvin_to_fahrenheit(temp_K):
    '''
    Convert Kelvin to Fahrenheit
    
    '''
    temp_F=(temp_K-273.15)*9/5 + 32
    
    return temp_F

def mm_inch(precip_mm):
    '''
    Convert mm to inch
    
    '''
    precip_inch=precip_mm*0.039
    
    return precip_inch


def days_gt_threshold(da, threshold):
    '''
    Count the number of days where the values in each grid cell are greater
    than or equal to the threshold value.
    '''
    
    # Calculate the count over time of all the values greater than or equal to
    # the threshold.
    #
    result = da.where(da > = threshold).resample(time="1Y").count(dim='time')

    # Return the result.
    #
    return result

def days_lt_threshold(da, threshold):
    '''
    Count the number of days where the values in each grid cell are smaller
    than or equal to the threshold value.
    '''
    
    # Calculate the count over time of all the values greater than or equal to
    # the threshold.
    #
    result = da.where(da < = threshold).resample(time="1Y").count(dim='time')

    # Return the result.
    #
    return result

def max_ndays(da, days):
    '''
    Annual highest value of a certain variable averaged over a n-day period
    '''

    # Get the maximum sums over the days and take the mean.
    #
    maxndays= da.rolling(time=days,min_periods=1,center=True).mean().resample(time="1Y").max(dim='time')

    # Return the result.
    #
    return maxndays

def min_ndays(da, days):
    '''
    Annual lowest value of a certain variable averaged over a n-day period
    '''

    # Get the maximum sums over the days and take the mean.
    #
    minndays= da.rolling(time=days,min_periods=1,center=True).mean().resample(time="1Y").min(dim='time')

    # Return the result.
    #
    return minndays



def cooling_degree_days(da,threshold):
    
    '''
    Cooling degree days (annual cumulative number of degrees by which the daily average temperature is greater than x°F)
    '''
    diff=(da-threshold).clip(min=0)
    cdd=diff.resample(time="1Y").sum(dim="time")
    
    return cdd

def heating_degree_days(da,threshold):
    
    '''
    Cooling degree days (annual cumulative number of degrees by which the daily average temperature is greater than x°F)
    '''
    diff=(threshold-da).clip(min=0)
    hdd=diff.resample(time="1Y").sum(dim="time")
    
    return hdd

def heating_degree_days(da,threshold):
    
    '''
    Cooling degree days (annual cumulative number of degrees by which the daily average temperature is greater than x°F)
    '''
    diff=(threshold-da).clip(min=0)
    hdd=diff.resample(time="1Y").sum(dim="time")
    
    return hdd

def modified_growing_degree_days(da,threshold):
    
    '''
    Modified growing degree days, base 50 (annual cumulative number of degrees by which the daily average temperature is greater than 50°F; before calculating the daily average temperatures, daily maximum temperatures above 86°F and daily minimum temperatures below 50°F are set to those values)
    '''
    da['tasmax']=da['tasmax'].where(da['tasmax']<86, 86)
    da['tasmin']=da['tasmin'].where(da['tasmin']>50, 50)
    da_avg = da.assign(tavg=lambda x: (x.tasmin + x.tasmax)/2)['tavg']

    diff=(da_avg-threshold).clip(min=0)
    gddmod=diff.resample(time="1Y").sum(dim="time")
    
    return gddmod

def pr_agg(da,agg_fun):
    '''
    Sum the values in each grid cell (monthly, yearly). Convert the result to mm/day from mm/sec.
    
    
    '''
    
    # Calculate the sum of the values.
    #
    result = da.resample(time=agg_fun).sum(dim=("time"))
  
    # Convert the results to mm/day by multiplying by 86400.0.
    #
    result *= 86400.0
    
    # Return the result.
    #
    return result

def cons_dd(data,threshold=0.01):
    '''
    Annual maximum number of consecutive dry days (days with total precipitation less than 0.01 inches)
    '''
    data_group=data
    dry_days = data_group<threshold
    consecutive_dry_periods = (dry_days == dry_days.shift(time=1))
    cs = consecutive_dry_periods.cumsum(dim="time")
    cs=cs.assign_coords({"time": consecutive_dry_periods["time"].values})
    cs2 = cs.where(consecutive_dry_periods == 0)  
    cs2 = cs2.where(cs2.time!=0, 0)
    cs3 = cs2.ffill(dim="time")  
    consecutive_dry_periods_time = cs - cs3
    dry_run_max = consecutive_dry_periods_time.max(dim='time')
    
    return dry_run_max

def cons_wd(data,threshold):
    '''
    Annual maximum number of consecutive dry days (days with total precipitation less than 0.01 inches)
    '''
    dry_days = data>threshold
    consecutive_wet_periods = (dry_days == dry_days.shift(time=1))
    cs = consecutive_wet_periods.cumsum(dim="time")  
    cs=cs.assign_coords({"time": consecutive_wet_periods["time"].values})
    cs2 = cs.where(consecutive_wet_periods == 0)  
    cs2 = cs2.where(cs2.time!=0, 0)
    cs3 = cs2.ffill(dim="time")  
    consecutive_wet_periods_time = cs - cs3
    wet_run_max = consecutive_wet_periods_time.resample(time="1Y").max(dim='time')
        
    return wet_run_max

def prmax5day(da, days):
    '''
    days    [in] The number of days to sum over.
    returns      An array of maximum sums.
    '''

    maxsum = da.rolling(time=days,min_periods=1,center=True).sum().resample(time="1Y").max(dim='time')

    return maxsum

def nonzero_quantiles(hist, threshold_list):
    '''
    threshold    [in] quantile=10,20,30,40,50,60, 70, 80, 90, 95, 99
    returns      An array of maximum sums.
    '''

    quant=hist.quantile(threshold_list, dim="time", method='linear', skipna=True)

    return quant
