# coding: utf8
'''
Created on 2013/02/25

@author: n_shimada
'''
import time
import cProfile
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
import hierarchical_Bayes_model_rev1 as hbm
from multiprocessing import *
from threading import *
reload(hbm)
#
#
inpt_filename = 'C:/Users/n_shimada/Desktop/test.csv'
#
#
# ランダムシードの定義
random.seed(555)
#
##--定数の定義-------
TIMES = 1344
nhh = 1
SEASONALYTY = 7 #季節変動の周期
RP = 11000  # サンプリング数
keep = 100
nz = 3
nD = 3
zc = 1+1+ SEASONALYTY-2 + nz
m = 1 + 1 + nz
nvar = 1 # 個人属性で求める対象の数（生鮮食品とそれ以外の日曜品とかだと2）
limy = 1e20 # 欠測とみなす数値の境界
k = 1 # トレンド成分モデルの次数
#thread = 4  # 並列処理のプロセス数
##-------------------
#
### data発生プロセス
## 前回来店からの日数の対数,イベントの有無（ダミー変数）------
## 値引きの有無（ダミー変数）
Ztld=[]
## 顧客の消費傾向Data
INV = np.array([[np.double(0)]*TIMES for hh in xrange(nhh)])
PM = np.array([np.random.uniform(0, 3000, TIMES) for i in range(nhh)])
C = np.array([539.7260274]*nhh)
dlt = np.array([np.float(2)]*nhh)
lmbd = np.array([np.float(1.2)]*nhh)
#INV = hbm.calc_INV(TIMES, PM, C, INV, dlt, lmbd)
# Z全体にマージ
Z=[]
tmp = []
y = []
with open(inpt_filename, 'r') as f:
    for hh in xrange(nhh):
        for line in f:
            inpt = line.strip().split(',')
            y.append(int(inpt[0]))
            # 時系列パラメタ部分
            tmp = [0]*(SEASONALYTY-2)
            # Zの季節調整成分を追加
            tmp.insert(0,1)
            # Zのトレンド成分を追加
            for i in xrange(k):
                tmp.insert(0,1)
            # 消費者の数値を導入
            tmp.append(int(inpt[1]))
            tmp.append(int(inpt[2]))
            tmp.append(int(inpt[3]))
            Z.append(tmp)
        Ztld.append(Z) # 人,time,変数数.転置後は変数数,時間,人
Ztld = np.array(Ztld).T
p = np.array( np.ndarray.flatten((Ztld[(k-1)+SEASONALYTY+nz-1,:,:])))
Ztld[(k-1)+SEASONALYTY,:,:] = hbm.standardization(Ztld[(k-1)+SEASONALYTY,:,:])
Ztld[(k-1)+SEASONALYTY+nz-1,:,:] = hbm.calc_INV(TIMES, p, C, INV, dlt, lmbd).T
Ztld[(k-1)+SEASONALYTY+nz-1,:,:] = hbm.standardization(Ztld[(k-1)+SEASONALYTY+nz-1,:,:])
#
## 顧客共通部分のdata
D = np.array([[np.float(0)]*3 for i in xrange(1)])
D[:,:] = [1, 2.201636905, 3.302585093]
##------------------------------------------------------------
#
## 事前分布のパラメタ-----------------
# '''step1：潜在効用サンプリング用'''
A = 0.01 * np.identity(zc) ## AはB_0の逆数
b0 = np.array([[np.float(0) for j in xrange(1)] for i in xrange(zc)])
#
# step3：システムノイズの分散サンプリング用
mu0 = 0; kaps0 = 25
nu0 = 0.02; s0 = 0.02
Sita_sys0 = np.array([np.float(10)]*m)
#
# step5：消費者異質性の回帰パラメタHのデータ枠
m0 = np.array([[np.float(0) for j in xrange(nvar)] for i in xrange(nD)])
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
#ZpZ = np.zeros(TIMES,zc,zc)
u = np.array( [[np.float(0)]*TIMES]*nhh )
#
# step2:状態ベクトルの算出のデータ枠
# 処理の都合上、初期値の設定部分で設定
#
# step3：システムノイズの分散サンプリングのデータ枠
Sita_sys = np.array([np.float(10)*np.identity(m) for i in xrange(nhh)])    # 3次元のためmatrix化できないので、計算時にmatrix化
#
# step4：擬似家庭内在庫を規定するためのパラメタサンプリングのためのデータ枠
# θが2変数を持つ時はベクトルではなくmatrixにすること
Lsita_dlt = np.array([np.float(0)]*nhh)
Lsita_lmbd = np.array([np.float(0)]*nhh)
Hdlt = np.array([[np.float(0)]*nD]*nvar)
Hlmbd = np.array([[np.float(0)]*nD]*nvar)
Vsita_dlt = 0.01*np.identity(nvar)
Vsita_lmbd = 0.01*np.identity(nvar)
Sita_dlt = np.array([np.float(0)]*nhh)
Sita_lmbd = np.array([np.float(0)]*nhh)
sigma_dlt = 0.01*np.identity(nvar)
sigma_lmbd = 0.01*np.identity(nvar)
rej_dlt = np.array([np.float(0)]*nhh)
rej_lmbd = np.array([np.float(0)]*nhh)
##---------------------------------------------------------
#
#
## 初期値の設定------------------------------
# step1用
Xs = np.array([[[np.float(0)]*TIMES]*zc]*nhh) #人,変数数,time
sigma = 1.0
# utはat,btを発生させるために作った擬似データ.本来は必要ない
ut = np.array([ss.uniform.rvs(loc=-2, scale=4,size=TIMES) for i in xrange(nhh)])
#
## step2用
param = hbm.FGHset(0, 1, SEASONALYTY, 0, nz)
L = 1
R = np.identity(L)
F = np.array(param['MatF'])
G = np.array(param['MatG'])
# システムモデルの分散を個人ごとに格納する枠
Q0 =np.array(param['MatQ'])
Q = np.array(Q0*nhh)
#
# step3用
mu = 0.
sigs = 1.
##-------------------------------------------
## 切断範囲の指定
at = np.array([[-100 if y==0 else 0 for t in xrange(TIMES)] for hh in xrange(nhh)])
bt = np.array([[0 if y==0 else 100 for t in xrange(TIMES)] for hh in xrange(nhh)])
#
##-------------------
udraw = np.array([[[np.float(0) for l in xrange(TIMES)] for j in xrange(nhh)] for i in xrange(RP)])
#
def calculate_utility((hh, Xs)):
    for t in xrange(TIMES):
        # step1--------------------------------------------------
        # uのサンプリング(ループはさせていないが個人ごとの計算)
        u[hh,t] = hbm.rtnorm(np.dot(Ztld[:,t,hh], Xs[hh,:,t]), sigma, at[hh,t], bt[hh,t])[0]
    return u[hh,:]
    #------------------------------------------------------------
#
def Kalman_filtering((hh, u, Sita_sys)):
 #   try:
        ## step2のシステムモデルパラメータの計算----------------------    
        # TAU2の最尤推定を求める数値計算------------------------------
        ISW = 0; XPS=0;
        tau0 = []
        for i in range(m):
            tau0.append(Sita_sys[hh,i,i]) 
        #mybounds=[(1e-4,1e2),(1e-4,1e2),(1e-4,1e2),(1e-4,1e2),(1e-4,1e2)]
        #LLF1 = so.fmin_l_bfgs_b(hbm.LogL, x0=tau0,
        #                        args=(np.array(u[hh,:]), F, np.array(Ztld[:,:,hh]), G, R, limy, ISW, zc, m, TIMES, Q0),
        #                        bounds=mybounds, approx_grad=True)         
        # TAU2の最尤推定
        TAU2 = tau0
# 
        # カルマンフィルタ
        Q = hbm.Qset(Q0 ,TAU2); XF0 = [0]*zc
        VF0 = np.float(10) * np.identity(zc); OSW = 1
        LLF2 = hbm.KF(u[hh,:], XF0, VF0, F, Ztld[:,:,hh], G, Q, R, limy, ISW, OSW, zc, TIMES)
        XPS = LLF2['XPS']; XFS = LLF2['XFS']
        VPS = LLF2['VPS']; VFS = LLF2['VFS']
        SIG2 = LLF2['Ovar']; GSIG2 = 1
        # 平滑化 ----------------------------------------------------------
        LLF3 = hbm.SMO(XPS, XFS, VPS, VFS, F, GSIG2, 1, SEASONALYTY, 1, zc, TIMES)
        Xs[hh,:,:] = np.array(LLF3['XSS']).T #型を合わすために無理やり変換
        return Xs[hh,:,:]
        #------------------------------------------------------------
#    except Exception, e:
#        import sys
#        import traceback
#        sys.stderr.write(traceback.format_exc())
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
def calculate_spending_habits_param_delta((hh, u, Xs, Sita_dlt, Hdlt, Vsita_dlt)):  # step4のΘ_δの計算
    try:
        ### step4--------------------------------------
        ## '''dlt側の計算'''
        #print "Sita_dlt", Sita_dlt
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
        old_Lsita_dlt = Lsita + hbm.Nkernel(old_sita_dlt, Hdlt, D[hh,:], Vsita_dlt)
        old_Lsita_dlt = math.exp(-0.5*old_Lsita_dlt)
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
        #print "rej", rej_dlt[hh]
        return Sita_dlt[hh], rej_dlt[hh]    #list[[Sita_dlt],[rej_dlt]]
    except Exception, e:
        import sys
        import traceback
        sys.stderr.write(traceback.format_exc())
#
def calculate_spending_habits_param_lambd((hh, u, Xs, Sita_lmbd, Hlmbd, Vsita_lmbd)):  # step4のΘ_λの計算
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
## step7-----------------------------------------------------------
def calc_INV((hh, INV, dlt, lmbd)):
    for t in xrange(1,TIMES):
        Cons = INV[hh,t-1]*( dlt[hh]*C[hh]/(dlt[hh] + (INV[hh,t-1])**lmbd[hh]) )
        INV[hh, t] = INV[hh, t-1] + PM[hh, t-1]- Cons
    return INV[hh,:]
## ----------------------------------------------------------------
#
def pool_map(func, itr, args):
    pool = Pool(processes=4)
    result = pool.map( func, ((i, args) for i in range(itr)) )
    pool.close()
    pool.join()
    return result
#--------------------------------------------    
## サンプリングのループ
if __name__ == "__main__":
    start = time.clock()
    for nd in xrange(RP):
        rej_dlt = [0]*nhh
        rej_lmbd = [0]*nhh
        # step1--calculate utility
        pool = Pool(processes=4)
        u = np.array( pool.map(calculate_utility, ((hh, Xs) for hh in range(nhh))) )
        pool.close()
        pool.join()    
        udraw[nd,:,:] = u
        # step2--calculate implicit system v variable=Xa(do kalman filter)
        pool = Pool(processes=4)
        Xs = np.array( pool.map(Kalman_filtering, ((hh, u, Sita_sys) for hh in range(nhh))) ) 
        pool.close()
        pool.join()            
        # step3--calculate difference
        pool = Pool(processes=4)
        Sita_sys = np.array( pool.map(calculate_difference, ((hh, Xs) for hh in range(nhh))) )
        pool.close()
        pool.join()            
        # step4--calculate_spending_habits_param_delta=Sita_dlt
        pool = Pool(processes=4)
        tmp = np.array( pool.map(calculate_spending_habits_param_delta, 
                                 ((hh, u, Xs, Sita_dlt, Hdlt, Vsita_dlt) for hh in range(nhh))) )                
        pool.close()
        pool.join()            
        Sita_dlt = tmp[:,0]
        rej_dlt += tmp[:,1]
        # step4--calculate_spending_habits_param_lambd=Sita_lmbd
        pool = Pool(processes=4)
        tmp = np.array( pool.map(calculate_spending_habits_param_lambd, 
                                 ((hh, u, Xs,  Sita_lmbd, Hlmbd, Vsita_lmbd) for hh in range(nhh))) )
        pool.close()
        pool.join()            
        Sita_lmbd = tmp[:,0]
        rej_lmbd += tmp[:,1]
        ### step5--------------------------------------
        ## dlt側の算出----
        # 多変量正規分布のパラメタの算出
        D2 = np.dot(D.T, D)
        D2pA0 = D2 + A0
        Hhat_dlt = np.dot(np.dot(inverse(D2), D.T) , Sita_dlt)
        Dtld = np.dot( inverse(D2pA0) , (np.dot(D2, Hhat_dlt) + np.dot(A0, np.ndarray.flatten(m0))) )
        rtld = np.ndarray.flatten(Dtld)
        sig =  np.array( np.kron(Vsita_dlt, inverse(D2pA0)).T )
        # 多変量正規分布でサンプリング
        Hdlt = np.ndarray.flatten( hbm.randn_multivariate(rtld, np.matrix(sig), n=nvar) )
        ##-----------------
        ## lmbd側の算出----
        # 多変量正規分布のパラメタの算出
        Hhat_lmbd = np.dot( np.dot(inverse(D2), D.T) , Sita_lmbd)
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
        div = np.array(Sita_dlt) - np.dot( D, np.ndarray.flatten(Hdlt).T ).T
        S = np.dot(div, div.T) # 上の計算でdivは横ベクトルになるのでSをスカラにするためにTは後ろ
        # 逆ウィッシャート分布でサンプリング
        Vsita_dlt = hbm.invwishartrand(f0 + nhh, V0 + S)  
        ##------------
        ##lmbd側の算出
        # 逆ウィッシャート分布のパラメタの算出
        div = np.array(Sita_lmbd) - np.dot( D, np.ndarray.flatten(Hlmbd).T ).T
        S = np.dot(div, div.T)      
        # 逆ウィッシャート分布でサンプリング
        Vsita_lmbd = hbm.invwishartrand(f0 + nhh, V0 + S)  
        ##------------  
        #print "loop=%d" % nd
        #--------------------------------------------
#
        ## step7-------------------------------------------
        ##Zの消費金額を更新
        pool = Pool(processes=4)
        INV = np.array( pool.map(calc_INV, 
                         ((hh, INV, np.exp(Sita_dlt), Sita_lmbd) for hh in xrange(nhh))) )
        pool.close()
        pool.join()
        Ztld[(k-1)+SEASONALYTY+nz-1,:,:] = hbm.standardization(INV.T)       
        ##-------------------------------------------
        print "nd=%d", nd
    stop = time.clock()
    ##-------------------------------------------
#    cProfile.run("worker(int(sys.argv[1]), int(sys.argv[2]))")
    print stop - start
    print Xs
    ## output area---------------------------------------------------------
    outf = 'C:/Users/n_shimada/Desktop/test_crm3014.csv'
    with file(outf, 'w') as outfile:
        for slice_2d in Xs:
            np.savetxt(outfile, slice_2d)
    ## --------------------------------------------------------------------
#
    print("that's all")