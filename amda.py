from __future__ import annotations
import argparse
import json
from pathlib import Path
import numpy as np
import pandas as pd
from scipy.ndimage import uniform_filter1d
from scipy.stats import theilslopes, kendalltau, norm


def clean_calendar(df):
    x=df.copy(); x['date']=pd.to_datetime(x['date']); x=x.sort_values('date')
    x=x[~((x.date.dt.month==2)&(x.date.dt.day==29))].drop_duplicates('date')
    x=x.set_index('date').asfreq('D')
    x=x[~((x.index.month==2)&(x.index.day==29))]
    return x


def weighted_smooth(x,w,width,mode='nearest'):
    width=max(1,int(width))
    good=np.isfinite(x)&np.isfinite(w)&(w>0)
    numerator=uniform_filter1d(np.where(good,x*w,0.),width,mode=mode)*width
    denominator=uniform_filter1d(np.where(good,w,0.),width,mode=mode)*width
    return np.divide(numerator,denominator,out=np.full(len(x),np.nan),where=denominator>1e-10)


def seasonal_decompose(temp, years, doy, h=11, iterations=3):
    """Weighted day-of-year seasonal smoother plus 365-day trend; STL-type, not STL."""
    y=np.asarray(temp,float); yr=np.asarray(years,int); doy=np.asarray(doy,int)
    distinct=np.unique(yr); matrix=np.full((len(distinct),365),np.nan)
    for yi,year in enumerate(distinct):
        sel=yr==year; matrix[yi,doy[sel]-1]=y[sel]
    weights=np.isfinite(matrix).astype(float)
    matrix=np.nan_to_num(matrix,nan=0.)
    tr=np.zeros(len(y)); season=np.zeros(len(y))
    for iteration in range(iterations):
        detrended=matrix.copy()
        for yi,year in enumerate(distinct):
            sel=yr==year; detrended[yi,doy[sel]-1]-=tr[sel]
        num=uniform_filter1d(detrended*weights,size=h,axis=0,mode='nearest')
        den=uniform_filter1d(weights,size=h,axis=0,mode='nearest')
        by_doy=np.divide(num,den,out=np.zeros_like(num),where=den>1e-10)
        # circular day-of-year smoothing; each year retains a slowly varying season
        by_doy=uniform_filter1d(by_doy,size=15,axis=1,mode='wrap')
        by_doy-=by_doy.mean(axis=1,keepdims=True)
        season=np.array([by_doy[np.searchsorted(distinct, yy), dd-1] for yy,dd in zip(yr,doy)])
        tr=weighted_smooth(y-season, np.array([weights[np.searchsorted(distinct, yy),dd-1] for yy,dd in zip(yr,doy)]),365)
        residual=y-season-tr
        finite=np.isfinite(residual)
        mad=np.median(np.abs(residual[finite]-np.median(residual[finite]))) if finite.any() else 0.
        scale=max(1e-8,6*1.4826*mad)
        robust=np.square(np.maximum(0,1-np.square(np.nan_to_num(residual,nan=scale*2)/scale)))
        for yi,year in enumerate(distinct):
            sel=yr==year; weights[yi,doy[sel]-1]=np.isfinite(y[sel])*robust[sel]
    return tr,season,y-tr-season


def choose_window(y,year,doy,candidates=(5,11,21)):
    """Three strictly forward one-year holdouts, when there is enough history."""
    unique=np.unique(year)
    origins=unique[-3:] if len(unique)>=9 else unique[-1:]
    scores={}
    for h in candidates:
        errors=[]
        for origin in origins:
            train=year<origin; test=year==origin
            if train.sum()<365*5 or test.sum()==0: continue
            tr,season,_=seasonal_decompose(y[train],year[train],doy[train],h=h)
            season_by_doy=pd.Series(season).groupby(doy[train]).mean()
            forecast=tr[np.where(np.isfinite(tr))[0][-1]]+season_by_doy.reindex(doy[test]).to_numpy()
            valid=np.isfinite(y[test]) & np.isfinite(forecast)
            if valid.any(): errors.extend((y[test][valid]-forecast[valid])**2)
        scores[h]=float(np.mean(errors)) if errors else float('inf')
    return min(scores,key=scores.get),scores


def quality_control(x,k=4.5,h=None):
    y=x['station'].to_numpy(float); year=x.index.year.to_numpy()
    doy=(x.index.dayofyear.to_numpy()-((x.index.is_leap_year)&(x.index.month>2)).astype(int))
    physical=np.isfinite(y)&((y< -60)|(y>50))
    persistence=np.zeros(len(y),bool); run_start=0
    for i in range(1,len(y)+1):
        if i<len(y) and np.isfinite(y[i]) and np.isfinite(y[i-1]) and y[i]==y[i-1] and (x.index[i]-x.index[i-1]).days==1:
            continue
        if i-run_start>=5: persistence[run_start+1:i]=True
        run_start=i
    initial=physical|persistence; cleaned=y.copy(); cleaned[initial]=np.nan
    if h is None: h,scores=choose_window(cleaned,year,doy)
    else: scores={}
    trend,season,res=seasonal_decompose(cleaned,year,doy,h=h)
    # local circular day-of-year pooled median / MAD, excluding rejected observations
    med=np.full(365,np.nan); sd=np.full(365,np.nan)
    for d in range(1,366):
        distance=np.abs(doy-d); pool=res[np.minimum(distance,365-distance)<=15]
        pool=pool[np.isfinite(pool)]
        if len(pool)>=10:
            med[d-1]=np.median(pool); sd[d-1]=max(1e-7,1.4826*np.median(np.abs(pool-med[d-1])))
    z=(res-med[doy-1])/sd[doy-1]
    residual_flag=np.isfinite(z)&(np.abs(z)>k)
    flags=initial|residual_flag
    return flags, {'h':int(h),'h_cv_mse':scores,'physical':int(physical.sum()),'persistence':int(persistence.sum()),'residual':int(residual_flag.sum())}


def monthly_anomalies(x,flags):
    station=x['station'].copy(); station.iloc[np.where(flags)[0]]=np.nan
    station_month=station.resample('MS').agg(['mean','count'])
    reference_month=x['reference'].resample('MS').agg(['mean','count'])
    station_month.loc[station_month['count']<20,'mean']=np.nan
    reference_month.loc[reference_month['count']<20,'mean']=np.nan
    # Station-reference difference is sufficient to cancel the common seasonal cycle.
    # Both series' monthly climatologies are subtracted independently.
    s=station_month['mean']; r=reference_month['mean']
    sa=s-s.groupby(s.index.month).transform('mean'); ra=r-r.groupby(r.index.month).transform('mean')
    return pd.DataFrame({'station_anomaly':sa,'reference_anomaly':ra,'difference':sa-ra})


def _segment_cost(prefix, prefix_sq, start, end):
    """Within-segment SSE, with half-open indices [start, end)."""
    length = end-start
    total = prefix[end]-prefix[start]
    return float(max(0.,prefix_sq[end]-prefix_sq[start]-total*total/length))


def exact_changepoints(y, beta, min_size=24):
    """PELT: exact penalised segmentation WITH candidate pruning (Killick et al.).

    Objective: sum of segment SSE + beta * number of changepoints.
    PELT inequality uses K=0 for SSE segment cost. `min_size` enforces a
    minimum segment length. Full objective is checked by regression tests
    against exhaustive unpruned dynamic programming.
    """
    y = np.asarray(y,dtype=float)
    if not np.isfinite(y).all():
        raise ValueError('PELT needs a finite contiguous sequence')
    if beta < 0 or min_size < 1:
        raise ValueError('beta >= 0 and min_size >= 1 required')
    n = len(y)
    if n < min_size: return []
    prefix=np.r_[0.,np.cumsum(y)]
    prefix_sq=np.r_[0.,np.cumsum(y*y)]
    best=np.full(n+1,np.inf)
    best[0]=-beta
    previous=np.full(n+1,-1,int)
    admissible=[]
    remove_after={}
    for end in range(min_size,n+1):
        # With a minimum segment length, a candidate dominated at time t
        # can only be removed for endpoints >= t + min_size.
        admissible=[start for start in admissible if remove_after.get(start, n+1)>end]
        new_start=end-min_size
        if np.isfinite(best[new_start]):
            admissible.append(new_start)
        if not admissible:continue
        costs=[best[start]+_segment_cost(prefix,prefix_sq,start,end)+beta
               for start in admissible]
        index=int(np.argmin(costs))
        best[end]=costs[index]
        previous[end]=admissible[index]
        # PELT pruning: candidates dominated at end cannot be optimal later.
        # Cost of a whole segment >= cost of its two subsegments (SSE).
        for start,cost in zip(admissible,costs):
            if cost > best[end]+beta+1e-10:
                remove_after[start]=min(remove_after.get(start,n+1),end+min_size)
    if not np.isfinite(best[n]):return []
    points=[];end=n
    while end>0:
        start=int(previous[end])
        if start < 0:raise RuntimeError('Unable to reconstruct PELT path')
        if start:points.append(start)
        end=start
    return sorted(points)


def mann_kendall(y):
    """Two-sided classical Mann-Kendall with tie-corrected variance."""
    y=np.asarray(y,float)
    y=y[np.isfinite(y)]
    n=len(y)
    if n<3: return {'S':np.nan,'z':np.nan,'p':np.nan,'tau':np.nan}
    S=sum(int(np.sign(y[j]-y[i])) for i in range(n-1) for j in range(i+1,n))
    _, counts=np.unique(y,return_counts=True)
    variance=(n*(n-1)*(2*n+5)-sum(t*(t-1)*(2*t+5) for t in counts if t>1))/18.
    z=(S-1)/np.sqrt(variance) if S>0 and variance>0 else ((S+1)/np.sqrt(variance) if S<0 and variance>0 else 0.)
    p=float(2*norm.sf(abs(z)))
    tau=float(S/(n*(n-1)/2))
    return {'S':int(S),'z':float(z),'p':p,'tau':tau}


def trend_free_prewhitened_mk(values, years):
    """Yue et al. (2002)-style trend-free pre-whitening then Mann-Kendall.

    1. Theil-Sen slope is estimated from the original annual series.
    2. Subtract trend; estimate lag-one correlation of detrended residuals.
    3. Form whitened residuals e[t] - rho*e[t-1].
    4. Restore original fitted trend to whitened residuals and run MK.

    This is the single-pass TFPW approach, not iterative TFPW-WS.
    """
    values=np.asarray(values,float)
    years=np.asarray(years,float)
    if len(values)!=len(years) or not np.isfinite(values).all() or not np.isfinite(years).all():
        raise ValueError('Annual values and years must be finite and aligned')
    if len(values)<5:
        return {'warning':'TFPW needs at least 5 annual values'}
    slope, intercept, _, _=theilslopes(values,years,alpha=.95)
    detrended=values-slope*(years-years[0])
    # Only consecutive years form lag-1 pairs. Gaps cannot be treated as one year.
    adjacent=np.diff(years)==1
    before=detrended[:-1][adjacent]
    after=detrended[1:][adjacent]
    if len(before)<3:return {'warning':'insufficient consecutive years for AR(1)'}
    rho=float(np.corrcoef(before,after)[0,1]) if np.std(before)>0 and np.std(after)>0 else 0.
    if not np.isfinite(rho):rho=0.
    # Common TFPW practice: apply prewhitening only for positive rho;
    # negative autocorrelation need not be inflated in the same way.
    rho_used=max(0.,min(0.99,rho))
    prewhitened=after-rho_used*before
    restored=prewhitened+slope*(years[1:][adjacent]-years[0])
    result=mann_kendall(restored)
    result.update({'rho_detrended':rho,'rho_used':rho_used,
                   'sen_slope_C_per_year':float(slope),
                   'n_prewhitened':int(len(restored)),
                   'method':'trend-free pre-whitening + Mann-Kendall (Yue et al. 2002 style)'})
    return result


def homogenize(monthly,c=3.,delta_min=0.5):
    d=monthly['difference'].to_numpy(float)
    valid=np.isfinite(d)
    # Analyze contiguous valid sequence; missing months are not imputed as measurements.
    v=d[valid]; months=monthly.index[valid]
    if len(v)<48: return monthly.assign(adjusted=monthly.station_anomaly),[],{'warning':'<48 valid months'}
    dif=np.diff(v); sigma2=(1.4826*np.median(np.abs(dif-np.median(dif))))**2/2
    detr=v-pd.Series(v).rolling(25,center=True,min_periods=1).median().to_numpy()
    rho=float(np.corrcoef(detr[:-1],detr[1:])[0,1]) if len(v)>2 else 0.
    if not np.isfinite(rho): rho=0.
    rho=np.clip(rho,-0.95,0.95)
    beta=float(c*np.log(len(v))*max(sigma2,1e-6)*(1+rho)/(1-rho))
    cps=exact_changepoints(v,beta,min_size=24)
    # A detected boundary is retained only when neighboring raw-segment means differ sufficiently.
    bounds=[0]+cps+[len(v)]
    means=[float(np.mean(v[a:b])) for a,b in zip(bounds[:-1],bounds[1:])]
    retained=[p for j,p in enumerate(cps) if abs(means[j+1]-means[j])>=delta_min]
    bounds=[0]+retained+[len(v)]
    means=[float(np.mean(v[a:b])) for a,b in zip(bounds[:-1],bounds[1:])]
    correction=np.zeros(len(v))
    for j,(a,b) in enumerate(zip(bounds[:-1],bounds[1:])): correction[a:b]=means[-1]-means[j]
    adj=monthly.station_anomaly.copy()
    adj.loc[months]=adj.loc[months].to_numpy()+correction
    changes=[{'month':str(months[p].date()),'difference_C':round(means[j+1]-means[j],4)} for j,p in enumerate(retained)]
    return monthly.assign(adjusted=adj),changes,{'c':c,'delta_min_C':delta_min,'rho':rho,'beta':beta,'sigma2':sigma2}


def annual_trend(monthly):
    ann=monthly['adjusted'].resample('YS').agg(['mean','count'])
    ann=ann.loc[ann['count']>=10,'mean'].dropna()
    if len(ann)<5:return {'warning':'fewer than 5 complete annual means'}
    years=ann.index.year.to_numpy(); values=ann.to_numpy(float)
    slope,intercept,low,high=theilslopes(values,years,alpha=.95)
    raw=mann_kendall(values)
    tfpw=trend_free_prewhitened_mk(values,years)
    return {'trend_C_per_decade':float(slope*10),
            'CI95_C_per_decade':[float(low*10),float(high*10)],
            'n_years':len(ann),
            'mann_kendall_TFPW':tfpw,
            'mann_kendall_unadjusted_descriptive':raw,
            'kendall_tau_descriptive':raw['tau'],
            'kendall_p_unadjusted':raw['p']}


def analyze(df,k=4.5,c=3,delta_min=.5,h=None):
    x=clean_calendar(df)
    if not {'station','reference'}<=set(x):raise ValueError('Input must have date, station, reference columns')
    flags,q=quality_control(x,k=k,h=h)
    monthly=monthly_anomalies(x,flags)
    result,changes,meta=homogenize(monthly,c=c,delta_min=delta_min)
    return {'quality_control':q,'homogenization':meta,'changes':changes,'trend':annual_trend(result)},pd.DataFrame({'station':x.station,'reference':x.reference,'flag':flags},index=x.index),result


def simulate(seed,years=40):
    """Illustrative synthetic generator informed by manuscript Table 2; not original generator."""
    rng=np.random.default_rng(seed); n=years*365; day=np.tile(np.arange(1,366),years)
    dates=pd.date_range('1981-01-01',periods=years*365+12,freq='D')
    dates=dates[~((dates.month==2)&(dates.day==29))][:n]
    amplitude=2.5+20*np.cos(2*np.pi*(day-200)/365)
    slope=rng.uniform(.2,.6)/10; base=amplitude+slope*np.arange(n)/365
    yearly=rng.normal(0,.7,years).repeat(365)
    sd=3.7+1.1*np.cos(2*np.pi*(day-15)/365)
    z=np.zeros(n); refz=np.zeros(n)
    for i in range(1,n):
        z[i]=.7*z[i-1]+np.sqrt(1-.7**2)*rng.normal()
        refz[i]=.7*refz[i-1]+np.sqrt(1-.7**2)*(.85*z[i]+np.sqrt(1-.85**2)*rng.normal())
    true=base+yearly+sd*z
    reference=base+yearly+sd*(.85*z+np.sqrt(1-.85**2)*refz)+rng.normal(0,.15,n)
    observed=true.copy(); nshift=min(3,rng.poisson(1.2)); shift_times=[]
    if nshift:
        possibilities=np.arange(8*365,n-8*365,365)
        rng.shuffle(possibilities)
        for pos in possibilities:
            if all(abs(pos-zpos)>=8*365 for zpos in shift_times):shift_times.append(int(pos))
            if len(shift_times)>=nshift:break
        for pos in shift_times:observed[pos:]+=rng.choice([-1,1])*rng.uniform(.7,1.8)
    labels=np.zeros(n,bool)
    for fraction,kind in [(.002,'spike'),(.001,'sign')]:
        indices=rng.choice(n,int(n*fraction),replace=False)
        if kind=='spike':observed[indices]+=rng.choice([-1,1],len(indices))*rng.uniform(10,25,len(indices))
        else:observed[indices]*=-1
        labels[indices]=True
    for _ in range(rng.poisson(years*.25)):
        pos=int(rng.integers(1,n-16)); length=int(rng.integers(5,16))
        observed[pos:pos+length]=observed[pos-1]; labels[pos:pos+length]=True
    observed=np.round(observed,1); reference=np.round(reference,1)
    missing=rng.random(n)<.03
    for _ in range(int(years*.7)):
        pos=int(rng.integers(0,n-30)); missing[pos:pos+int(rng.integers(10,31))]=True
    observed[missing]=np.nan
    return pd.DataFrame({'date':dates,'station':observed,'reference':reference}),labels,sorted(shift_times),slope*10


def sensitivity(seeds=range(1000,1012), values=(.3,.5,.7),c_values=(1.5,3.,6.),k=4.5,h=11):
    """Runs a fresh independent tuning experiment; never substitutes for original results."""
    rows=[]
    for seed in seeds:
        df,truth,shift_positions,slope=simulate(seed)
        x=clean_calendar(df); flags,_=quality_control(x,k=k,h=h)
        month=monthly_anomalies(x,flags)
        for c in c_values:
            for delta in values:
                m,changes,_=homogenize(month,c=c,delta_min=delta)
                t=annual_trend(m)
                rows.append({'seed':seed,'c':c,'delta_min':delta,'n_changes':len(changes),
                             'abs_trend_error_C_decade':abs(t.get('trend_C_per_decade',np.nan)-slope),
                             'interval_contains_truth':int(t['CI95_C_per_decade'][0]<=slope<=t['CI95_C_per_decade'][1]) if 'CI95_C_per_decade' in t else np.nan})
    return pd.DataFrame(rows)


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--mode',choices=['demo','analyze','tune'],default='demo')
    p.add_argument('--input',help='CSV with date,station,reference columns')
    p.add_argument('--output-dir',default='outputs')
    p.add_argument('--n-seeds',type=int,default=12)
    p.add_argument('--h',type=int,default=None)
    p.add_argument('--delta-min',type=float,default=.5)
    p.add_argument('--c',type=float,default=3.)
    args=p.parse_args(); out=Path(args.output_dir);out.mkdir(exist_ok=True,parents=True)
    if args.mode=='tune':
        res=sensitivity(seeds=range(1000,1000+args.n_seeds))
        res.to_csv(out/'tuning_individual_runs.csv',index=False)
        summary=res.groupby(['c','delta_min']).agg(mean_abs_trend_error=('abs_trend_error_C_decade','mean'),coverage=('interval_contains_truth','mean'),mean_changes=('n_changes','mean')).reset_index()
        summary.to_csv(out/'tuning_summary.csv',index=False)
        print(summary.to_string(index=False));return
    if args.mode=='analyze':
        if not args.input:p.error('--input required in analyze mode')
        df=pd.read_csv(args.input)
    else:
        df,labels,shifts,slope=simulate(42)
        df.to_csv(out/'simulated_station.csv',index=False)
    res,daily,monthly=analyze(df,c=args.c,delta_min=args.delta_min,h=args.h)
    daily.to_csv(out/'daily_flags.csv');monthly.to_csv(out/'monthly_adjusted.csv')
    (out/'summary.json').write_text(json.dumps(res,indent=2,default=float),encoding='utf8')
    print(json.dumps(res,indent=2,default=float))

if __name__=='__main__':main()
