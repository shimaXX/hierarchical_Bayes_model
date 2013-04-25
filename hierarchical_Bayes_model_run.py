# coding: utf8
'''
Created on 2013/02/25

@author: n_shimada
'''
import time
import sys
import numpy.random as npr
from numpy.oldnumeric.linear_algebra import inverse
import scipy
from scipy import linalg
import random
import scipy.stats.distributions as dis
from scipy.stats import uniform,chi2,invgamma
import scipy.stats as ss
import numpy as np
import math
from scipy.linalg import inv,cholesky,det
import scipy.optimize as so
import hierarchical_Bayes_model as hbm
from multiprocessing import *
from numpy import genfromtxt
reload(hbm)
#
#
# データインポートファイルパスの定義
inpt_D_filename = 'C:/Users/n_shimada/Desktop/real_stage/Galster_CRM_D_sliced_rev4.csv'
inpt_Z_filename = 'C:/Users/n_shimada/Desktop/real_stage/Galster_CRM_visit_sliced_rev4.csv'
inpt_payment_filename = 'C:/Users/n_shimada/Desktop/real_stage/Galster_CRM_pymData_sliced_rev4.csv'
inpt_C_filename = 'C:/Users/n_shimada/Desktop/real_stage/Galster_CRM_C_data_sliced_rev4.csv'
inpt_visit_filename = 'C:/Users/n_shimada/Desktop/real_stage/Galster_CRM_sliced_rev4.csv'
#
### データinput-------------------
# Z1=訪問間隔データの取得
#Z_visit_span = genfromtxt(inpt_Z_filename, delimiter=',')        
# Dデータの取得（共通性を使って個人毎のパラメータを推定するのに必要なデータ）
D = genfromtxt(inpt_D_filename, delimiter=',')
# Cデータの取得（日平均消費金額）
C = genfromtxt(inpt_C_filename, delimiter=',')
# PMデータの取得（ INVを求めるための購買日別の金額 ）
PM = genfromtxt(inpt_payment_filename, delimiter=',')
# yデータの取得（ 訪問flagによって切断正規分布のa,bを求める ）
y = genfromtxt(inpt_visit_filename, delimiter=',')
print "done input"
### ------------------------------
#
# ランダムシードの発生
random.seed(555)
#
##--定数の定義-------
TIMES = len(PM[0,:])
nhh = len(PM[:,0])
SEASONALYTY = 7 #季節変動の周期
RP = 50  # サンプリング数
keep = 10
nz = 2
nD = len(D[0,:])
m = 1 + 1 + nz
nvar = 1 # 個人属性で求める対象の数（生鮮食品とそれ以外の日曜品とかだと2）
limy = 1e20 # 欠測とみなす数値の境界
k = 1 # トレンド成分モデルの次数
zc = k+1+ SEASONALYTY-2 + nz
pr = 4 #プロセス数
##-------------------
#
### data発生プロセス-----------------------------------
## 前回来店からの日数の対数,イベントの有無（ダミー変数）------
## 値引きの有無（ダミー変数）
Ztld=np.zeros((zc,TIMES,nhh))
## ----------------------
# Ztldの整形
Ztld[0,:,:] = [[1]*nhh]*TIMES
Ztld[1,:,:] = [[1]*nhh]*TIMES
Ztld[zc-2,:,:] = genfromtxt(inpt_Z_filename, delimiter=',').T #サイト訪問間隔
#
Ztld[(k-1)+SEASONALYTY,:,:] = hbm.standardization(Ztld[(k-1)+SEASONALYTY,:,:])
Ztld[(k-1)+SEASONALYTY+nz-1,:,:] = hbm.calc_INV(TIMES, PM, C, nhh)
Ztld[(k-1)+SEASONALYTY+nz-1,:,:] = hbm.standardization(Ztld[(k-1)+SEASONALYTY+nz-1,:,:])
print "done data shape"
###------------------------------------------------------------
#
## 事前分布のパラメタ-----------------
# '''step1：潜在効用サンプリング用'''
A = 0.01 * np.identity(zc) ## AはB_0の逆数
b0 = np.zeros((zc,nvar),dtype=np.float)
#
# step3：システムノイズの分散サンプリング用
mu0 = 0; kaps0 = 25
nu0 = 0.02; s0 = 0.02
Sita_sys0 = np.array([np.float(10)]*m)
#
# step5：消費者異質性の回帰パラメタHのデータ枠
m0 = np.zeros((nD,nvar),dtype=np.float)
A0 = 0.01 * np.identity(nD)  #樋口本ではA
#
# step6：消費者異質性の回帰パラメタVのデータ枠
f0 = nvar+3
V0 = f0 * np.identity(nvar)
##------------------------------------
#
#
## 事後分布サンプリングに必要なデータの枠を作成-----------
# step1：潜在効用サンプリングのデータ枠 
u = np.zeros((nhh,TIMES),dtype=np.float)
#
# step2:状態ベクトルの算出のデータ枠
# 処理の都合上、初期値の設定部分で設定
#
# step3：システムノイズの分散サンプリングのデータ枠
Sita_sys = np.array([np.float(10)*np.identity(m) for i in xrange(nhh)])    # 3次元のためmatrix化できないので、計算時にmatrix化
#
# step4：擬似家庭内在庫を規定するためのパラメタサンプリングのためのデータ枠
# θが2変数を持つ時はベクトルではなくmatrixにすること
Lsita_dlt = np.zeros((nhh),dtype=np.float)
Lsita_lmbd = np.zeros((nhh),dtype=np.float)
Hdlt = np.zeros((nvar,nD),dtype=np.float)
Hlmbd = np.zeros((nvar,nD),dtype=np.float)
Vsita_dlt = 0.01*np.identity(nvar)
Vsita_lmbd = 0.01*np.identity(nvar)
Sita_dlt = np.zeros((nhh),dtype=np.float)
Sita_lmbd = np.zeros((nhh),dtype=np.float)
sigma_dlt = 1.5*np.identity(nvar)
sigma_lmbd = 1.5*np.identity(nvar)
rej_dlt = np.zeros((nhh),dtype=np.float)
rej_lmbd = np.zeros((nhh),dtype=np.float)
##---------------------------------------------------------
#
#
## 初期値の設定------------------------------
# step1用
Xs = np.zeros((nhh, zc, TIMES),dtype=np.float) #人,変数数,time
sigma = 1.0
#
## step2用
param = hbm.FGHset(0, 1, SEASONALYTY, 0, nz)
L = 1
R = np.identity(L)
F = np.array(param['MatF'])
G = np.array(param['MatG'])
# システムモデルの分散を個人ごとに格納する枠
Q0 =np.array(param['MatQ'])
#Q = np.array(Q0*nhh)
#
# step3用
mu = 0.
sigs = 1.
##-------------------------------------------
## 切断範囲の指定
at = np.array([[-100 if y[hh,t]==0 else 0 for t in xrange(TIMES)] for hh in xrange(nhh)])
bt = np.array([[0 if y[hh,t]==0 else 100 for t in xrange(TIMES)] for hh in xrange(nhh)])
y = None
#
##-------------------
udraw = np.zeros((nhh,1,RP), dtype=np.float)
#
def calculate_utility((hh, Xs, Ztld)):
    # step1--------------------------------------------------
    # uのサンプリング(ループはさせていないが個人ごとの計算)
    mu = np.sum(Ztld[:,:,hh] * Xs[hh,:,:], axis = 0)
    u[hh,:] = hbm.rtnorm(mu, sigma, at[hh,:], bt[hh,:])
    return u[hh,:]
    #------------------------------------------------------------
#
def Kalman_filtering((hh, u, Sita_sys, Ztld)):
    ## step2のシステムモデルパラメータの計算----------------------    
    # TAU2の最尤推定を求める数値計算------------------------------
    ISW = 0; XPS=0;
    #mybounds=[(1e-4,1e2),(1e-4,1e2),(1e-4,1e2),(1e-4,1e2),(1e-4,1e2)]
    #LLF1 = so.fmin_l_bfgs_b(hbm.LogL, x0=tau0,
    #                        args=(np.array(u[hh,:]), F, np.array(Ztld[:,:,hh]), G, R, limy, ISW, zc, m, TIMES, Q0),
    #                        bounds=mybounds, approx_grad=True)         
    # TAU2の最尤推定
# 
    # カルマンフィルタ
    #Q = hbm.Qset(Q0 ,TAU2);
    XF0 = [0]*zc
    VF0 = np.float(10) * np.identity(zc); OSW = 1
    LLF2 = hbm.KF(u[hh,:], XF0, VF0, F, Ztld[:,:,hh], G, Sita_sys[hh,:,:], R, limy, ISW, OSW, zc, TIMES)
    XPS = LLF2['XPS']; XFS = LLF2['XFS']
    VPS = LLF2['VPS']; VFS = LLF2['VFS']
    SIG2 = LLF2['Ovar']; GSIG2 = 1
    # 平滑化 ----------------------------------------------------------
    LLF3 = hbm.SMO(XPS, XFS, VPS, VFS, F, GSIG2, 1, SEASONALYTY, 1, zc, TIMES)
    Xs[hh,:,:] = np.array(LLF3['XSS']).T #型を合わすために無理やり変換
    return Xs[hh,:,:]
    #------------------------------------------------------------
#
def calculate_difference((hh, Xs)):
    # step3の階差の計算--------------------------------------
    # step3の階差計算の和の計算で使用する変数の初期化
    dift = 0.
    difw = 0.
    difbeta = np.array([np.float(0)]*nz)
#
    dift = sum( (Xs[hh,0,1:TIMES] - Xs[hh,0,0:TIMES-1])**2 )
    difw = sum( (Xs[hh,k,1:TIMES]+sum(Xs[hh, k:(k-1)+SEASONALYTY-1, 0:TIMES-1]))**2 )
    for d in xrange(nz):
        difbeta[d] = sum( (Xs[hh, (k-1)+SEASONALYTY+d, 1:TIMES] 
                                    - Xs[hh, (k-1)+ SEASONALYTY+d, 0:TIMES-1])**2 )      
    # step3--------------------------------------
    Sita_sys[hh,0,0] = invgamma.rvs((nu0+TIMES)/2, scale=(s0+dift)/2, size=1)[0]
    Sita_sys[hh,1,1] = invgamma.rvs((nu0+TIMES)/2, scale=(s0+difw)/2, size=1)[0]
    for d in xrange(nz):
        Sita_sys[hh,2+d,2+d] = invgamma.rvs((nu0+TIMES)/2, scale=(s0+difbeta[d])/2, size=1)[0]
    return Sita_sys[hh,:,:]
    #--------------------------------------------
#
def calculate_difference_m(Xs): # 人間すべてに対応
    # step3の階差の計算--------------------------------------
    # step3の階差計算の和の計算で使用する変数の初期化
    dift = np.zeros(nhh, dtype=np.float)
    difw = np.zeros(nhh, dtype=np.float)
    difbeta = np.zeros((nhh,nz), dtype=np.float)
#
    dift = np.sum( (Xs[:,0,1:TIMES] - Xs[:,0,0:TIMES-1])**2, axis=1 )
    difw = np.sum( (Xs[:,k,1:TIMES]+np.sum(Xs[:, k:(k-1)+SEASONALYTY-1, 0:TIMES-1], axis=1))**2, axis=1 )
    for d in xrange(nz):
        difbeta[:,d] = np.sum( (Xs[:, (k-1)+SEASONALYTY+d, 1:TIMES] 
                                    - Xs[:, (k-1)+ SEASONALYTY+d, 0:TIMES-1])**2, axis=1 )      
    # step3--------------------------------------
    Sita_sys[:,0,0] = invgamma.rvs((nu0+TIMES)/2, scale=(s0+dift)/2, size=nhh)
    Sita_sys[:,1,1] = invgamma.rvs((nu0+TIMES)/2, scale=(s0+difw)/2, size=nhh)
    for d in xrange(nz):
        Sita_sys[:,2+d,2+d] = invgamma.rvs((nu0+TIMES)/2, scale=(s0+difbeta[:,d])/2, size=nhh)
    return Sita_sys[:,:,:]
    #--------------------------------------------
#
def calculate_spending_habits_param_delta((hh, u, Xs, Sita_dlt, Hdlt, Vsita_dlt, Ztld)):  # step4のΘ_δの計算
    ### step4--------------------------------------
    ## '''dlt側の計算'''
    # step4のθの事後分布カーネルの第一項の和計算時使用する変数の初期化
    Lsita = 0.
    # 効用値の誤差計算(θの尤度計算)
    Lsita = sum( (u[hh,:] - np.diag(np.dot(Ztld[:,:,hh].T, Xs[hh,:,:])))**2  )
    # 現状のθを確保する
    old_sita_dlt = Sita_dlt[hh]
    # 新しいθをサンプリング（酔歩サンプリング）
    new_sita_dlt = Sita_dlt[hh] + ss.norm.rvs(loc=0, scale=sigma_dlt,size=1)[0]
#    
    # 尤度の計算（対数尤度の場合はヤコビアンで調整）
    new_Lsita_dlt = Lsita + hbm.Nkernel(new_sita_dlt, Hdlt, D[hh,:], Vsita_dlt)
    new_Lsita_dlt = math.exp(-0.5*new_Lsita_dlt)
    #new_Lsita_dlt = new_Lsita_dlt*(1/math.exp(new_Lsita_dlt))
    old_Lsita_dlt = Lsita + hbm.Nkernel(old_sita_dlt, Hdlt, D[hh,:], Vsita_dlt)
    old_Lsita_dlt = math.exp(-0.5*old_Lsita_dlt)
    #old_Lsita_dlt = old_Lsita_dlt*(1/math.exp(old_Lsita_dlt))
    if old_Lsita_dlt == 0: old_Lsita_dlt=1e-6
#        
    # MHステップ
    alpha = min(1, new_Lsita_dlt/old_Lsita_dlt)
    if alpha==None: alpha = -1
    uni = ss.uniform.rvs(loc=0 , scale=1, size=1)
    if uni < alpha and new_sita_dlt > 0:
        Sita_dlt[hh] = new_sita_dlt
    else:
        rej_dlt[hh] = rej_dlt[hh] + 1
    return Sita_dlt[hh], rej_dlt[hh]    #list[[Sita_dlt],[rej_dlt]]
#
def calculate_spending_habits_param_delta_m((u, Xs, Sita_dlt, Hdlt, Vsita_dlt, Ztld, rej)):  # step4-多変量版
    ### step4--------------------------------------
    ## '''dlt側の計算'''
    # step4のθの事後分布カーネルの第一項の和計算時使用する変数の初期化
    Lsita = np.zeros(nhh,dtype=np.float)
    # 効用値の誤差計算(θの尤度計算)
    Lsita = [np.sum( (u[hh,:] - np.sum(Ztld[:,:,hh]*Xs[hh,:,:], axis = 0) )**2 ) for hh in xrange(nhh)]
    # 現状のθを確保する
    old_sita_dlt = Sita_dlt[:]
    # 新しいθをサンプリング（酔歩サンプリング）
    new_sita_dlt = Sita_dlt[:] + ss.norm.rvs(loc=0, scale=sigma_dlt, size=nhh)
#   
    # 尤度の計算（対数尤度の場合はヤコビアンで調整）
    new_Lsita_dlt = Lsita + hbm.Nkernel(new_sita_dlt, Hdlt, D[:,:], Vsita_dlt)
    new_Lsita_dlt = np.exp(-0.5*new_Lsita_dlt)
    #new_Lsita_dlt = new_Lsita_dlt*(1/math.exp(new_Lsita_dlt))
    old_Lsita_dlt = Lsita + hbm.Nkernel(old_sita_dlt, Hdlt, D[:,:], Vsita_dlt)
    old_Lsita_dlt = np.exp(-0.5*old_Lsita_dlt)
    #old_Lsita_dlt = old_Lsita_dlt*(1/math.exp(old_Lsita_dlt))
    old_Lsita_dlt[old_Lsita_dlt == 0] = 1e-6
#
    # MHステップ
    alpha = np.array(new_Lsita_dlt/old_Lsita_dlt)
    alpha = alpha*(alpha<1) # 1より大きければここでは0が入る.1より小さければそのままの値.
    alpha[alpha==0] = 1 # 1より大きいものは1とする,をここで実行
    alpha[alpha!=alpha] = -1  # NaNであった部分を-1とする
    uni = ss.uniform.rvs(loc=0 , scale=1, size=nhh)
    tmp = alpha - uni
    Sita_tmp = new_sita_dlt*(tmp>0) + old_sita_dlt*(tmp<=0) # MHの採択条件
    Sita_dlt = Sita_tmp*(Sita_tmp>=0) + old_sita_dlt*(Sita_tmp<0) # この分析ではδは非負であるという条件 
    tmp = Sita_dlt - old_sita_dlt #  
    rej += 1*(tmp[0,:]==0) # 計算の結果が初期値と同じであればリジェクトに1をプラス
    #return np.ndarray.flatten(Sita_dlt), np.ndarray.flatten(rej)
    return Sita_dlt[0,:], rej 
#
def calculate_spending_habits_param_lambd((hh, u, Xs, Sita_lmbd, Hlmbd, Vsita_lmbd, Ztld)):  # step4のΘ_λの計算
    ### step4--------------------------------------
    ## '''lmbd側の計算'''
    # step4のθの事後分布カーネルの第一項の和計算時使用する変数の初期化
    Lsita = 0.
    # 効用値の誤差計算(θの尤度計算)
    Lsita = sum( (u[hh,:] - np.diag(np.dot(Ztld[:,:,hh].T, Xs[hh,:,:])))**2  )
    # 現状のθを確保する
    old_sita_lmbd = Sita_lmbd[hh]
    # 新しいθをサンプリング（酔歩サンプリング）
    new_sita_lmbd = Sita_lmbd[hh] + ss.norm.rvs(loc=0, scale=sigma_lmbd, size=1)[0]
#        
    # 尤度の計算（対数尤度の場合はヤコビアンで調整）
    new_Lsita_lmbd = Lsita + hbm.Nkernel(new_sita_lmbd, Hlmbd, D[hh,:], Vsita_lmbd)
    new_Lsita_lmbd = math.exp(-0.5*new_Lsita_lmbd)
    old_Lsita_lmbd = Lsita + hbm.Nkernel(old_sita_lmbd, Hlmbd, D[hh,:], Vsita_lmbd)
    old_Lsita_lmbd = math.exp(-0.5*old_Lsita_lmbd)
    if old_Lsita_lmbd == 0: old_Lsita_lmbd=1e-6
#        
    # MHステップ
    alpha = min(1, new_Lsita_lmbd/old_Lsita_lmbd)
    if alpha==None: alpha = -1
    uni = ss.uniform.rvs(loc=0, scale=1, size=1)
    if uni < alpha:
        Sita_lmbd[hh] = new_sita_lmbd
    else:
        rej_lmbd[hh] = rej_lmbd[hh] + 1
    return Sita_lmbd[hh], rej_lmbd[hh]
#
def calculate_spending_habits_param_lambd_m((u, Xs, Sita_lmbd, Hlmbd, Vsita_lmbd, Ztld, rej)):  # step4-多変量
    ### step4--------------------------------------
    ## '''lmbd側の計算'''
    # step4のθの事後分布カーネルの第一項の和計算時使用する変数の初期化
    Lsita = np.zeros(nhh,dtype=np.float)
    # 効用値の誤差計算(θの尤度計算)
    Lsita = [np.sum( (u[hh,:] - np.sum(Ztld[:,:,hh]*Xs[hh,:,:], axis = 0) )**2 ) for hh in xrange(nhh)]
    # 現状のθを確保する
    old_sita_lmbd = Sita_lmbd[:]
    # 新しいθをサンプリング（酔歩サンプリング）
    new_sita_lmbd = Sita_lmbd[:] + ss.norm.rvs(loc=0, scale=sigma_lmbd, size=nhh)
#   
    # 尤度の計算（対数尤度の場合はヤコビアンで調整）
    new_Lsita_lmbd = Lsita + hbm.Nkernel(new_sita_lmbd, Hlmbd, D[:,:], Vsita_lmbd)
    new_Lsita_lmbd = np.exp(-0.5*new_Lsita_lmbd)
    old_Lsita_lmbd = Lsita + hbm.Nkernel(old_sita_lmbd, Hlmbd, D[:,:], Vsita_lmbd)
    old_Lsita_lmbd = np.exp(-0.5*old_Lsita_lmbd)
    old_Lsita_lmbd[old_Lsita_lmbd == 0] = 1e-6
#
    # MHステップ
    alpha = np.array(new_Lsita_lmbd/old_Lsita_lmbd)
    alpha = alpha*(alpha<1) # 1より大きければここでは0が入る.1より小さければそのままの値.
    alpha[alpha==0] = 1 # 1より大きいものは1とする,をここで実行
    alpha[alpha!=alpha] = -1  # NaNであった部分を-1とする
    uni = ss.uniform.rvs(loc=0 , scale=1, size=nhh)
    tmp = alpha - uni
    Sita_lmbd = new_sita_lmbd*(tmp>0) + old_sita_lmbd*(tmp<=0) # MHの採択条件
    rej += 1*(tmp[0,:]<=0) # 計算の結果が初期値と同じであればリジェクトに1をプラス
    return Sita_lmbd[0,:], rej  
#
## step7-----------------------------------------------------------
def calc_INV((hh, dlt, lmbd)):
    INV = np.zeros(TIMES,dtype = np.double)
    for t in xrange(1,TIMES):
        if abs(INV[t-1]) < 1e-6 : Cons = 0.0
        else:
            denom = dlt[hh]*C[hh] + (INV[t-1])**lmbd[hh]
            if denom == 0.0: denom = 1e-6
            Cons = INV[t-1]*dlt[hh]*C[hh]/denom
        INV[t] = INV[t-1] + PM[hh, t-1]- Cons
    return INV
#
def calc_INV_m((dlt, lmbd)):
    INV = np.zeros((TIMES,nhh), dtype = np.double)
    Cons = np.zeros(nhh, dtype = np.double)
    for t in xrange(1,TIMES):
        tmp = INV[t-1,:]*(abs(INV[t-1,:])>=1e-6) + 1e-6*(abs(INV[t-1,:])<1e-6) # INVが1e-6以上ではINVの値を採用、未満の場合は分子が小さくなりすぎる（warnigが出る）ので1e-6とする 
        denom = np.array(dlt[:]*C[:] + (tmp)**lmbd[:])
        denom[denom == 0.0] = 1e-6
        Cons = (INV[t-1,:]*dlt*C/denom)*(abs(INV[t-1,:])>=1e-6) # INV[t-1]が1e-6以下の時は0とする
        INV[t] = INV[t-1,:] + PM[:,t-1]- Cons
    return INV
## ----------------------------------------------------------------
#
def pool_map(func, itr, args):
    pool = Pool(processes=pr)
    result = pool.map( func, ((i, args) for i in range(itr)) )
    pool.close()
    pool.join()
    return result
#-------------------------------------------- 
print "done ini"   
## サンプリングのループ
if __name__ == "__main__":
    rej_dlt = [0]*nhh
    rej_lmbd = [0]*nhh
#
    #start = time.clock()
    for nd in xrange(RP):
        # step1--calculate utility
        pool = Pool(processes=pr)
        u = np.array( pool.map(calculate_utility, ((hh, Xs, Ztld) for hh in xrange(nhh))) )
        pool.close()
        pool.join()     
        udraw[:,:,nd] = np.matrix(u[:,10]).T
        # step2--calculate implicit system v variable=Xa(do kalman filter)
        pool = Pool(processes=pr)
        Xs = np.array( pool.map(Kalman_filtering, ((hh, u, Sita_sys, Ztld) for hh in xrange(nhh))) ) 
        pool.close()
        pool.join()
        # step3--calculate difference
        Sita_sys = calculate_difference_m(Xs)
        # step4--calculate_spending_habits_param_delta=Sita_dlt
        Sita_dlt, rej_dlt = calculate_spending_habits_param_delta_m((u, Xs, Sita_dlt, Hdlt, Vsita_dlt, Ztld, rej_dlt))                
        # step4--calculate_spending_habits_param_lambd=Sita_lmbd
        Sita_lmbd, rej_lmbd = calculate_spending_habits_param_lambd_m((u, Xs,  Sita_lmbd, Hlmbd, Vsita_lmbd, Ztld, rej_lmbd))
        ### step5--------------------------------------
        ## dlt側の算出----
        # 多変量正規分布のパラメタの算出
        D2 = np.dot(D.T, D)
        D2pA0 = D2 + A0
        Hhat_dlt = np.ndarray.flatten( np.dot(np.dot(inverse(D2), D.T) , Sita_dlt.T) )
        Dtld = np.dot( inverse(D2pA0) , (np.dot(D2, Hhat_dlt) + np.dot(A0, np.ndarray.flatten(m0))) )
        rtld = np.ndarray.flatten(Dtld)
        sig =  np.array( np.kron(Vsita_dlt, inverse(D2pA0)).T )
        # 多変量正規分布でサンプリング
        Hdlt = np.ndarray.flatten( hbm.randn_multivariate(rtld, np.matrix(sig), n=nvar) )
        ##-----------------
        ## lmbd側の算出----
        # 多変量正規分布のパラメタの算出
        Hhat_lmbd = np.ndarray.flatten( np.dot( np.dot(inverse(D2), D.T) , Sita_lmbd.T) )
        Dtld = np.dot( inverse(D2pA0) , (np.dot(D2, Hhat_lmbd) + np.dot(A0, np.ndarray.flatten(m0))) )
        rtld = np.ndarray.flatten(Dtld)
        sig =  np.array( np.kron(Vsita_lmbd, inverse(D2pA0)).T )
        # 多変量正規分布でサンプリング
        Hlmbd = np.ndarray.flatten( hbm.randn_multivariate(rtld, np.matrix(sig), n=nvar) ) 
        ##-----------------
        #--------------------------------------------
#    
        ### step6--------------------------------------
        ##dlt側の算出
        # 逆ウィッシャート分布のパラメタの算出
        #div = np.array(Sita_dlt) - np.dot( D, np.ndarray.flatten(Hdlt).T ).T
        div = np.array(Sita_dlt) - np.dot( D, Hdlt.T ).T
        S = np.dot(div, div.T) # 上の計算でdivは横ベクトルになるのでSをスカラにするためにTは後ろ
        # 逆ウィッシャート分布でサンプリング
        Vsita_dlt = hbm.invwishartrand(f0 + nhh, V0 + S)  
        ##------------
        ##lmbd側の算出
        # 逆ウィッシャート分布のパラメタの算出
        div = np.array(Sita_lmbd) - np.dot( D, Hlmbd.T ).T
        S = np.dot(div, div.T)      
        # 逆ウィッシャート分布でサンプリング
        Vsita_lmbd = hbm.invwishartrand(f0 + nhh, V0 + S)  
        ##------------  
        #--------------------------------------------
#
        ## step7-------------------------------------------
        ##Zの消費金額を更新
        INV = calc_INV_m((Sita_dlt, Sita_lmbd))
        Ztld[(k-1)+SEASONALYTY+nz-1,:,:] = hbm.standardization(INV)       
        ##-------------------------------------------
        print "nd=%d", nd
    #stop = time.clock()
    #print stop - start
    ##-------------------------------------------
#    cProfile.run("worker(int(sys.argv[1]), int(sys.argv[2]))")
    ## output area---------------------------------------------------------
    outf = open('C:/Users/n_shimada/Desktop/test_udraw3014.csv','w')
    nums = []
    for hh in xrange(nhh):
        prsd = np.std(udraw[hh,0, 0:math.ceil(0.1*RP)]); posd = np.std(udraw[hh,0,RP-math.floor(0.5*RP):])
        prav = np.average(udraw[hh,0,0:math.ceil(0.1*RP)]); poav = np.average(udraw[hh,0,RP-math.floor(0.5*RP):])
        prf = np.sum(abs(udraw[hh,0,0:math.ceil(0.1*RP)])); pof = np.sum(abs(udraw[hh,0,RP-math.floor(0.5*RP):]))
        Z = math.sqrt(RP)*(prav - poav)/math.sqrt( prsd**2/0.1 + posd**2/0.5 )
        outf.write("UsrNon="+ str(hh) + ",Z="+ str(Z) +"\n")
        for t in xrange(1):
            nums = [str(udraw[hh,t,p/keep]) for p in xrange(RP) if p%keep == 0 ]
            outf.write((",".join(nums))+"\n") 
    outf.close()
    ## --------------------------------------------------------------------
    ## output area---------------------------------------------------------
    outf = open('C:/Users/n_shimada/Desktop/test_crm3014.csv','w')
    nums = []
    for hh in xrange(nhh):
        outf.write("UsrNon="+ str(hh) +"\n")
        for r in xrange(zc):
            nums = [str(Xs[hh,r,t]) for t in xrange(TIMES)]
            outf.write((",".join(nums))+"\n")
    outf.close()
    ## --------------------------------------------------------------------
    ## output area---------------------------------------------------------
    outf01 = open('C:/Users/n_shimada/Desktop/test_u3014.csv','w')
    outf02 = open('C:/Users/n_shimada/Desktop/test_INV3014.csv','w')
    outf03 = open('C:/Users/n_shimada/Desktop/test_rej_dlt3014.csv','w')
    outf04 = open('C:/Users/n_shimada/Desktop/test_rej_lmbd3014.csv','w')
    outf05 = open('C:/Users/n_shimada/Desktop/test_sita_dlt3014.csv','w')
    outf06 = open('C:/Users/n_shimada/Desktop/test_sita_lmbd3014.csv','w')
    nums01 = []; nums02 = []; nums03 = []; nums04 = []; nums05 = []; nums06 = []
    for hh in xrange(nhh):
        #順序付きで格納
        nums01 = [str(u[hh,t]) for t in xrange(TIMES)]
        nums02 = [str(INV[t,hh]) for t in xrange(TIMES)]
        nums03 = [str(rej_dlt[hh])]
        nums04 = [str(rej_lmbd[hh])]
        nums05 = [str(Sita_dlt[hh])]
        nums06 = [str(Sita_lmbd[hh])]
#
        # 書き込み
        outf01.write((",".join(nums01))+"\n")
        outf02.write((",".join(nums02))+"\n")
        outf03.write((",".join(nums03))+"\n")
        outf04.write((",".join(nums04))+"\n")
        outf05.write((",".join(nums05))+"\n")
        outf06.write((",".join(nums06))+"\n")
    outf01.close()
    outf02.close()
    outf03.close()
    outf04.close()
    outf05.close()
    outf06.close()
    ## --------------------------------------------------------------------
#
    print("that's all")