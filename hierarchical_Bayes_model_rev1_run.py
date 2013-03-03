# coding: utf8
'''
Created on 2013/02/25

@author: n_shimada
'''
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
reload(hbm)
#
# ランダムシードの定義
random.seed(555)
#
##--定数の定義-------
TIMES = 100
nhh = 15
SEASONALYTY = 7 #季節変動の周期
RP = 1  # サンプリング数
keep = 100
nz = 3
nD = 5
zc = 1+1+ SEASONALYTY-2 + nz
m = 1 + 1 + nz
nvar = 1 # 個人属性で求める対象の数（生鮮食品とそれ以外の日曜品とかだと2）
limy = 1e20 # 欠測とみなす数値の境界
k = 1 # トレンド成分モデルの次数
##-------------------
#
### data発生プロセス
## 前回来店からの日数の対数,イベントの有無（ダミー変数）------
## 値引きの有無（ダミー変数）
Ztld=[]
# Z全体にマージ
for hh in range(nhh):
    Z=[]
    for t in range(TIMES):
        # 時系列パラメタ部分
        tmp = [0]*(SEASONALYTY-2)
        # Zのトレンド成分を追加
        for i in range(k):
            tmp.insert(0,1)
        # Zの季節調整成分を追加
        tmp.insert(0,1)
        tmp.append(int(random.uniform(0,1)*30))
        tmp.append(int(random.uniform(0,2)))
        tmp.append(int(random.uniform(0,2)))
        Z.append(tmp)
    Ztld.append(Z) # 人,time,変数数.転置後は変数数,時間,人
Ztld = np.array(Ztld).T
Ztld[(k-1)+SEASONALYTY,:,:] = hbm.standardization(Ztld[(k-1)+SEASONALYTY,:,:])
#
## 顧客共通部分のdata
D = []
for i in range(nhh):
    Dtmp = [1]
    Dtmp.append( math.log(random.normalvariate(7, 2)) )
    Dtmp.append( math.log(random.normalvariate(3000, 800)/1000) )
    Dtmp.append( math.log(random.normalvariate(400, 100)/100) )
    Dtmp.append( math.log(random.normalvariate(30, 5)/10) )
    D.append(Dtmp)
D = np.array(D) #人,変数数
##------------------------------------------------------------
#
## 事前分布のパラメタ-----------------
# '''step1：潜在効用サンプリング用'''
A = 0.01 * np.identity(zc) ## AはB_0の逆数
b0 = np.array([[np.float(0) for j in range(1)] for i in range(zc)])
#
# step3：システムノイズの分散サンプリング用
mu0 = 0; kaps0 = 25
nu0 = 0.02; s0 = 0.02
Sita_sys0 = np.array([np.float(10)]*m)
#
# step5：消費者異質性の回帰パラメタHのデータ枠
m0 = np.array([[np.float(0) for j in range(nvar)] for i in range(nD)])
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
Sita_sys = np.array([np.float(10)*np.identity(m) for i in range(nhh)])    # 3次元のためmatrix化できないので、計算時にmatrix化
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
ut = np.array([ss.uniform.rvs(loc=-2, scale=4,size=TIMES) for i in range(nhh)])
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
at = np.array([[-100 if ut[hh,t]<0 else 0 for t in range(TIMES)] for hh in range(nhh)])
bt = np.array([[0 if ut[hh,t]<0 else 100 for t in range(TIMES)] for hh in range(nhh)])
#
##-------------------
udraw = np.array([[[np.float(0) for l in range(TIMES)] for j in range(nhh)] for i in range(RP)])
#
## サンプリングのループ
for nd in range(RP):
    for hh in range(nhh):
        # step3の階差計算の和の計算で使用する変数の初期化
        dift = 0.
        difw = 0.
        difbeta = np.array([np.float(0)]*nz)
#   
        # step4のθの事後分布カーネルの第一項の和計算時使用する変数の初期化
        Lsita = 0.
#
        for t in range(TIMES):
            # step1--------------------------------------------------
            # uのサンプリング(ループはさせていないが個人ごとの計算)
            u[hh,t] = hbm.rtnorm(np.dot(Ztld[:,t,hh], Xs[hh,:,t]), sigma, at[hh,t], bt[hh,t])[0]
            udraw[nd,hh,t] = u[hh,t]
            #------------------------------------------------------------
#           
        ## step2のシステムモデルパラメータの計算----------------------    
        # TAU2の最尤推定を求める数値計算------------------------------
        ISW = 0
        mybounds=[(1e-4,1e2),(1e-4,1e2),(1e-4,1e2),(1e-4,1e2),(1e-4,1e2)]
        LLF1 = so.fmin_l_bfgs_b(hbm.LogL, x0=Sita_sys0,
                                args=(np.array(u[hh,:]), F, np.array(Ztld[:,:,hh]), G, R, limy, ISW, zc, m, TIMES, Q0),
                                bounds=mybounds, approx_grad=True)         
        # TAU2の最尤推定
        TAU2 = LLF1[0]
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
        #------------------------------------------------------------
#        
        # step3の階差の計算--------------------------------------
        dift = sum( (Xs[hh,0,1:TIMES] - Xs[hh,0,0:TIMES-1])**2 )
        difw = sum( (Xs[hh,k,1:TIMES]+sum(Xs[hh, k:(k-1)+SEASONALYTY-1, 0:TIMES-1]))**2 )
        for d in range(nz):
            difbeta[d] = sum( (Xs[hh, (k-1)+SEASONALYTY+d, 1:TIMES] 
                                        - Xs[hh, (k-1)+ SEASONALYTY+d, 0:TIMES-1])**2 )
        #--------------------------------------------------------
#     
        # step4の効用値の誤差計算(step4のθの尤度計算)------------
        Lsita = sum( (u[hh,:] - np.diag(np.dot(Ztld[:,:,hh].T, Xs[hh,:,:])))**2  )
        #--------------------------------------------------------
#           
        # step3--------------------------------------
        Sita_sys[hh,0,0] = invgamma.rvs((nu0+TIMES)/2, scale=(s0+dift)/2, size=1)[0]
        Sita_sys[hh,1,1] = invgamma.rvs((nu0+TIMES)/2, scale=(s0+difw)/2, size=1)[0]
        for d in range(nz):
            Sita_sys[hh,2+d,2+d] = invgamma.rvs((nu0+TIMES)/2, scale=(s0+difbeta[d])/2, size=1)[0]
        #--------------------------------------------
#       
        ### step4--------------------------------------
        ## '''dlt側の計算'''
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
#        
        # MHステップ
        alpha = min(1, new_Lsita_dlt/old_Lsita_dlt)
        if alpha==None: alpha = -1
        uni = ss.uniform.rvs(loc=0 , scale=1, size=1)
        if uni < alpha:
            Sita_dlt[hh] = new_sita_dlt
        else:
            rej_dlt[hh] = rej_dlt[hh] + 1
#    
        ## lmbd側の計算
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
#        
        # MHステップ
        alpha = min(1, new_Lsita_lmbd/old_Lsita_lmbd)
        if alpha==None: alpha = -1
        uni = ss.uniform.rvs(loc=0, scale=1, size=1)
        if uni < alpha:
            Sita_lmbd[hh] = new_sita_lmbd
        else:
            rej_lmbd[hh] = rej_lmbd[hh] + 1
        #--------------------------------------------    
#             
    ### step5--------------------------------------
    ## dlt側の算出----
    # 多変量正規分布のパラメタの算出
    D2 = np.dot(D.T, D)
    D2pA0 = D2 + A0
    Hhat_dlt = np.dot(np.dot(inverse(D2), D.T) , Sita_dlt)
    Dtld = np.dot( inverse(D2pA0) , (np.dot(D2, Hhat_dlt) + np.dot(A0, np.ndarray.flatten(m0))) )
    rtld = np.ndarray.flatten(Dtld)
    sig =  np.array( [D2pA0*Vsita_dlt[i] for i in range(Vsita_dlt.shape[0])] )
    # 多変量正規分布でサンプリング
    Hdlt = np.ndarray.flatten( hbm.randn_multivariate(rtld, np.matrix(sig), n=nvar) )
    ##-----------------
    ## lmbd側の算出----
    # 多変量正規分布のパラメタの算出
    Hhat_lmbd = np.dot( np.dot(inverse(D2), D.T) , Sita_lmbd)
    Dtld = np.dot( inverse(D2pA0) , (np.dot(D2, Hhat_lmbd) + np.dot(A0, np.ndarray.flatten(m0))) )
    # Dtldをベクトルにバラす
    Dtld_ary = np.array(Dtld) # arrayじゃないと要素で操作できないのでarrayへ
    rtld = np.ndarray.flatten(Dtld_ary)
    sig =  np.array( [[D2pA0]*Vsita_lmbd[i] for i in range(Vsita_lmbd.shape[0])] )
    # 多変量正規分布でサンプリング
    Hlmbd = np.ndarray.flatten( hbm.randn_multivariate(rtld, np.matrix(sig), n=nvar) ) 
    ##-----------------
    #--------------------------------------------
#    
    ### step6--------------------------------------
    ##dlt側の算出
    # 逆ウィッシャート分布のパラメタの算出
    div = np.array(Sita_dlt) - np.dot(D, 
            np.array([np.array([Hdlt[j+i*nD] for j in range(nD)]) for i in range(nvar)]).T).T
    S = np.dot(div, div.T) # 上の計算でdivは横ベクトルになるのでSをスカラにするためにTは後ろ
    # 逆ウィッシャート分布でサンプリング
    Vsita_dlt = hbm.invwishartrand(f0 + nhh, V0 + S)  
    ##------------
    ##lmbd側の算出
    # 逆ウィッシャート分布のパラメタの算出
    div = np.array(Sita_lmbd) - np.dot(D,
            np.array([np.array([Hlmbd[j+i*nD] for j in range(nD)]) for i in range(nvar)]).T).T
    S = np.dot(div, div.T)
    # 逆ウィッシャート分布でサンプリング
    Vsita_lmbd = hbm.invwishartrand(f0 + nhh, V0 + S)  
    ##------------  
    #--------------------------------------------
##-------------------------------------------