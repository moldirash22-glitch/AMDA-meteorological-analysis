import numpy as np
import pandas as pd
from amda import exact_changepoints,simulate,clean_calendar,annual_trend,mann_kendall,trend_free_prewhitened_mk


def brute_changepoints(y,beta,min_size):
    n=len(y); best=np.full(n+1,np.inf);best[0]=-beta
    prev=np.full(n+1,-1,int)
    for end in range(min_size,n+1):
        for start in range(end-min_size+1):
            if not np.isfinite(best[start]):continue
            seg=y[start:end]
            value=best[start]+np.sum((seg-seg.mean())**2)+beta
            if value<best[end]: best[end]=value;prev[end]=start
    output=[];pos=n
    while pos>0 and prev[pos]>=0:
        start=prev[pos]
        if start:output.append(int(start))
        pos=start
    return sorted(output)


def test_pelt_step():
    a=np.r_[np.zeros(40),np.ones(40)*2]
    assert exact_changepoints(a,beta=5,min_size=24)==[40]


def test_pelt_matches_unpruned_optimum():
    for seed in range(30):
        rng=np.random.default_rng(seed)
        y=rng.normal(size=75)
        if seed%2==0:y[35:]+=2
        for penalty in [0.5,3,8,30]:
            assert exact_changepoints(y,penalty,min_size=6)==brute_changepoints(y,penalty,6)


def test_generator_and_calendar():
    x,_,_,slope=simulate(17,years=12)
    assert len(x)==12*365
    assert len(clean_calendar(x))==len(x)
    assert .2<=slope<=.6


def test_trend_and_tfpw():
    ix=pd.date_range('2000-01-01',periods=240,freq='MS')
    m=pd.DataFrame({'adjusted':np.repeat(np.arange(20)*.05,12)},index=ix)
    r=annual_trend(m)
    assert abs(r['trend_C_per_decade']-.5)<1e-8
    assert 'p' in r['mann_kendall_TFPW']
    assert r['mann_kendall_TFPW']['method'].startswith('trend-free')


def test_mk_monotone_and_constant():
    assert mann_kendall(np.arange(20.))['p']<.01
    assert mann_kendall(np.ones(20))['p']==1


def test_tfpw_detects_positive_autocorrelation():
    rng=np.random.default_rng(7)
    noise=np.zeros(50)
    for i in range(1,50):noise[i]=.7*noise[i-1]+rng.normal(0,.3)
    out=trend_free_prewhitened_mk(.03*np.arange(50)+noise,np.arange(1980,2030))
    assert -1<out['rho_detrended']<1
    assert out['n_prewhitened']==49
