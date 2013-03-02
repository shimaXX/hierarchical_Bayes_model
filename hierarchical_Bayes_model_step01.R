## マーケティングモデルの断片
options(encoding="UTF-8")
rm(list=ls())
setwd("C:/RW")

random.seed(555)

#install.packages("mvtnorm")
#install.packages("MCMCpack")
#install.packages("FKF")
#install.packages("sspir")

### package読み込み
library(bayesm)
library(mvtnorm)
library(MCMCpack)
library(FKF)
#--------------------

##--定数の定義-------
TIMES <- 500
nhh <-15
SEASONALYTY <- 7 #季節変動の周期
RP <- 11000  # サンプリング数
keep <- 100
nz <- 3
nD <- 5
zc <- 1+1+ SEASONALYTY-2 + nz
m <- 1 + 1 + nz
nvar <- 1 # 個人属性で求める対象の数（生鮮食品とそれ以外の日曜品とかだと2）
limy <- 1e20 # 欠測とみなす数値の境界
k <- 1 # トレンド成分モデルの次数
##-------------------

### data発生プロセス
## 前回来店からの日数の対数,イベントの有無（ダミー変数）------
## 値引きの有無（ダミー変数）
Z <- NULL

# 時系列パラメタ部分
tmp <- c(1, 1, rep(0,SEASONALYTY-2))

# Z全体にマージ
for(t in 1:TIMES){
  Z[[t]] <- list(
          Z0 = t(matrix(tmp, length(tmp), nhh)),
          Z1 = as.integer(runif(nhh)*30),
          Z2 = rep(ifelse(runif(1)>0.5,1,0), nhh),
          Z3 = rep(ifelse(runif(1)>0.5,0,1), nhh))
}

## 顧客共通部分のdata
D <- matrix(
    c(
      rep(1,nhh),
      log( rnorm(nhh, 7, 2) ),
      log( rnorm(nhh, 3000, 800)/1000 ),
      log( rnorm(nhh, 400, 100)/100 ),
      log( rnorm(nhh, 30, 5)/10 )
    ),
    nhh, nD
  )
##------------------------------------------------------------

### 関数の定義------------------------------------
# 2項プロビットモデルのベイズ推定(step1用)
rtnorm <- function(mu, sigma, a, b){
  FA <- pnorm(a, mu, sigma)
  FB <- pnorm(b, mu, sigma)
  result <- qnorm(runif(length(mu)) * (FB - FA) + FA, mu, sigma)
  if(result == -Inf ) {result <- -100}
  if(result == Inf ) {result <- 100}
  return(result)
}

# Zの値を基準化する関数(step1用)
standardization <- function(z){
  return(log(0.001 + (z - min(z))/(max(z) - min(z))*(0.999 - 0.001)))
}

# 逆ガンマ分布のサンプリング関数(step3用)
irgamma <- function(n, shp, rt){
  return(1/rgamma(n, shape=shp, scale=1/rt))
}

# 単変量正規分布のカーネル部分の乗数の部分の計算(step4用)
Nkernel <- function(sita, H, D, Vsita){
  return( ((sita - H%*%D)^2)/Vsita )
}

# 多変量正規分布のカーネル部分の乗数の部分の計算(step4用)
NMkernel <- function(sita, H, D, Vsita){
  return(
    t(sita - t(H)%*%D) %*% solve(Vsita) %*% (sita - t(H)%*%D)
  )
}

### 季節調整モデルの状態空間表現の行列設定
FGHset <- function(al, k, p, q, nz){  #alはARモデルのαベクトル、k;p;qはトレンド、季節、AR
  m <- k + p + q + nz -1
  if(q>0){G <- matrix(0,m,3+nz)} # 状態モデルでトレンド、季節、ARの3つを含む場合
  else{G <- matrix(0,m,2+nz)}    #AR成分を含まない場合(q=0)
  F <- matrix(0,m,m)
  #H <- matrix(0,1,m)          #Hの代わりにZtldを使うので、Hは不要
  
  ## トレンドモデルのブロック行列の構築
  G[1,1] <- 1
  #H[1,1] <- 1
  if(k==1) {F[1,1] <- 1}
  if(k==2) {F[1,1] <- 2; F[1,2] <- -1; F[2,1] <- 1}
  if(k==3) {F[1,1] <- 3; F[1,2] <- -3; F[1,3] <- 1
            F[2,1] <- 1; F[3,2] <- 1}
  LS <- k
  NS <- 2;
  
  ## 季節調整成分のブロック行列の構築
  G[LS+1, NS] <- 1
  #H[1,LS+1] <- 1
  for(i in 1:(p-1)) {F[LS+1, LS+i] <- -1}
  for(i in 1:(p-2)) {F[LS+i+1, LS+i] <- 1}

  ## Z成分のブロック行列の構築
  LS <- LS + p -1
  NS <- 2
  for(i in 1:(nz)) {F[LS+i, LS+i] <- 1}
  for(i in 1:(nz)) {G[LS+i, NS+i] <- 1}
  
  if(q>0){
    NS <- NS +1
    G[LS+1, NS] <- 1
    #H[1,LS+1] <- 1
    for(i in 1:q) {F[LS+1, LS+i] <- al[i]}
    if(q>1) {
      for(i in 1:(q-1)) {F[LS+i+1, LS+i] <- 1}
    }
  }
  
  ## シスムモデルの分散共分散行列Qの枠の算出
  Q <- diag(NS+nz)
  
  return(list(m=m, MatF=F, MatG=G, MatQ=Q))
}

# 状態空間表現における行列Qの設定------------------------
Qset <- function(Q0 ,parm){
  NS <- ncol(Q0)
  Q <- Q0
  # シスムモデルの分散共分散行列Qの枠の算出
  for(i in 1:NS) {
    Q[i,i] <- parm[i]
  }
  return(Q)
}

# カルマンフィルタの関数 ------------------------------------------
KF <- function(y, XF0, VF0, F, H, G, Q, R, limy, ISW, OSW, m, N)  
{
  if (OSW == 1)
  {
    XPS <- matrix(0, m, N); XFS <- matrix(0, m, N)
    VPS <- array(dim = c(m, m, N)); VFS <- array(dim = c(m, m, N))
  }
  XF <- XF0; VF <- VF0; NSUM <- 0; SIG2 <- 0; LDET <- 0
  for (n in 1:N)
  {
    # 1期先予測
    XP <- F %*% XF
    VP <- F %*% VF %*% t(F) +  G %*% Q %*% t(G)
    # フィルタ
    if (y[n] < limy) 
    {
      NSUM <- NSUM + 1
      B <- matrix(H[,n],1,) %*% VP %*% t(matrix(H[,n],1,)) + R  # Hは数学的には横ベクトルなので回転させる
      B1 <- solve(B)
      K <- VP %*% t(matrix(H[,n],1,)) %*% B1
      e <- y[n] - matrix(H[,n],1,) %*% XP
      XF <- XP + K %*% e
      VF <- VP - K %*% matrix(H[,n],1,) %*% VP
      SIG2 <- SIG2 + t(e) %*% B1 %*% e
      LDET <- LDET + log(det(B))
    }
    else
    {
      XF <- XP; VF <- VP
    }
    if (OSW == 1)
    {
      XPS[,n] <- XP; XFS[,n] <- XF; VPS[,,n] <- VP; VFS[,,n] <- VF
    }   
  }
  SIG2 <- SIG2 / NSUM
  if (ISW == 0)
  {
    FF <- -0.5 * (NSUM * (log(2 * pi * SIG2) + 1) + LDET)
  }
  else
  {
    FF <- -0.5 * (NSUM * (log(2 * pi) + SIG2) + LDET)
  }
  if (OSW == 0)
  {
    return(list(LLF=FF, Ovar=SIG2))
  }
  if (OSW == 1)
  {
    return(list(XPS=XPS, XFS=XFS, VPS=VPS, VFS=VFS, LLF=FF, Ovar=SIG2))
  }
}

# 平滑化の関数 ----------------------------------------------------
SMO <- function(XPS, XFS, VPS, VFS, F, GSIG2, k, p, q, m, N)
{
  XSS <- matrix(0, m, N); VSS <- array(dim = c(m, m, N))
  XS1 <- XFS[,N]; VS1 <- VFS[,,N]
  XSS[,N] <- XS1; VSS[,,N] <- VS1
  for (n1 in 1:(N-1))
  {
    n <- N - n1; XP <- XPS[,n+1]; XF <- XFS[,n]
    VP <- VPS[,,n+1]; VF <- VFS[,,n]; VPI <- solve(VP)
    A <- VF %*% t(F) %*% VPI
    XS2 <- XF + A %*% (XS1 - XP)
    VS2 <- VF + A %*% (VS1 - VP) %*% t(A)
    XS1 <- XS2; VS1 <- VS2
    XSS[,n] <- XS1; VSS[,,n] <- VS1
  }
  return(list(XSS=XSS, VSS=VSS))
}

# TAU2xの対数尤度関数の定義 ----------------------------------------
LogL <- function(parm, y, F, H, G, R, limy, ISW, k, m, N, Q0,...) 
{
  Q <- Qset(Q0 ,parm)
  XF0 <- numeric(k); VF0 <- 10 * diag(k); OSW <- 0
  LLF <- KF(y, XF0, VF0, F, H, G, Q, R, limy, ISW, OSW, k, N)
  LL <- LLF$LLF
  return(LL)
}
# -------------------------------------------------------
##-----------------------------------


## 事前分布のパラメタ-----------------
# step1：潜在効用サンプリング用
A <- 0.01 * diag(zc) ## AはB_0の逆数
b0 <- matrix(0, nrow=zc, ncol=1)  ## カルマンフィルタで算出し、渡す必要あるかも

# step3：システムノイズの分散サンプリング用
mu0 <- 0; kaps0 <- 25;
nu0 <- 0.02; s0 <- 0.02

# step5：消費者異質性の回帰パラメタHのデータ枠
m0 <- matrix(rep(0,nD*nvar), nD, nvar)
A0 <- 0.01*diag(nD)            #樋口本ではA

# step6：消費者異質性の回帰パラメタVのデータ枠
f0 <- nvar　+3
V0 <- f0 * diag(nvar)
##------------------------------------


##↓Rのループの直下に仕込まないと駄目では？
## 事後分布サンプリングに必要なデータの枠を作成-----------
# step1：潜在効用サンプリングのデータ枠 
ZpZ <- array(double(zc*zc*TIMES),dim=c(zc, zc, TIMES))
Ztld <- array(double(nhh*zc*TIMES),dim=c(nhh, zc, TIMES))
for(t in 1:TIMES){
  Ztld[,,t] <- cbind(Z[[t]]$Z0, Z[[t]]$Z1, Z[[t]]$Z2, Z[[t]]$Z3)
  ZpZ[,,t] <- crossprod(Ztld[,,t])
}
u <- array(runif(nhh*TIMES, min=-1, max=1), dim=c(nhh, TIMES))
Zpu <- matrix(double(zc*TIMES), zc, TIMES)

# step2:状態ベクトルの算出のデータ枠
# 処理の都合上、初期値の設定部分で処理

# step3：システムノイズの分散サンプリングのデータ枠
Sita.sys <- array(rep(10*diag(m),nhh), dim=c(m,m,nhh)) 

# step4：擬似家庭内在庫を規定するためのパラメタサンプリングのためのデータ枠
# θが2変数を持つ時はベクトルではなくmatrixにすること
Lsita.dlt <- rep(0,nhh)
Lsita.lmbd <- rep(0,nhh)
Hdlt <- matrix(rep(0,nD),nvar,)
Hlmbd <- matrix(rep(0,nD),nvar,)
Vsita.dlt <- 0.01*diag(nvar)
Vsita.lmbd <- 0.01*diag(nvar)
Sita.dlt <- rep(0,nhh)
Sita.lmbd <- rep(0,nhh) 
sigma.dlt <- 0.01*diag(nvar)
sigma.lmbd <- 0.01*diag(nvar)
rej.dlt <-rep(0,nhh)
rej.lmbd <-rep(0,nhh)
##---------------------------------------------------------


## 初期値の設定------------------------------
# step1用
Xs <- array(double(nhh*zc*TIMES), dim=c(nhh, zc, TIMES))
sigma <- 1.0
# utはat,btを発生させるために作った擬似データ.本来は必要ない
ut <- array(runif(nhh*TIMES, min=-1, max=1), dim=c(nhh, TIMES))

# step2用
param <- FGHset(0, 1, SEASONALYTY, 0, nz)
L <- 1
R <- diag(L)
F <- param$MatF
G <- param$MatG
 #システムモデルの分散を個人ごとに格納する枠
 Q0 <-param$MatQ %o% rep(1,nhh)
 Q <- Q0

# step3用
mu <- 0
sigs <- 1
##-------------------------------------------

## 切断範囲の指定
at <- ifelse(ut<0, -100, 0)
bt <- ifelse(ut<0, 0, 100)


##-------------------
udraw <- array(double(nhh*TIMES*RP),dim=c(nhh,TIMES,RP))

## サンプリングのループ
for(nd in 1:RP){
  for(hh in 1:nhh){
    # step3の階差計算の和の計算で使用する変数の初期化
    dift <- 0
    difw <- 0
    difbeta <- rep(0,nz)
    
    # step4のθの事後分布カーネルの第一項の和計算時使用する変数の初期化
    Lsita <- 0
    
    for(t in 1:TIMES){
      # step1--------------------------------------------------
      u[hh,t] <- rtnorm(Ztld[hh,,t]%*%Xs[hh,,t], sigma, at[hh,t], bt[hh,t])
      
      udraw[hh,t,nd] <- u[hh,t]
      
      # betaのサンプリング
      Zpu[,t] <- crossprod(t(Ztld[hh,,t]),u[hh,t])
      IB <- solve(ZpZ[,,t] + A)
      btilde <- IB %*% (Zpu[,t] + A%*% b0)
      Xs[hh,,t] <- btilde + chol(IB) %*% rnorm(zc)

      #------------------------------------------------------------
    }
    
    ## step2のシステムモデルパラメータの計算----------------------    
    # TAU2の最尤推定を求める数値計算------------------------------
    ISW <- 0
    tau0 <- c(TAU21=Sita.sys[1,1,hh],TAU22=Sita.sys[2,2,hh],
              TAU23=Sita.sys[3,3,hh],TAU24=Sita.sys[4,4,hh],
              TAU25=Sita.sys[5,5,hh])
    LLF1 <- optim(tau0, fn=LogL, y=u[hh,], F=F, H=Ztld[hh,,], G=G, R=R,
                  limy=limy, ISW=ISW, k=zc, m=m , N=TIMES , Q0=Q0[,,hh],
                  method ="L-BFGS-B",
                  lower = 1e-4, upper = 1e2,
                  control=list(fnscale=-1))
    # TAU2の最尤推定
    TAU2 <- LLF1$par
    
    # カルマンフィルタ
    Q <- Qset(Q0[,,hh] ,TAU2); XF0 <- numeric(zc)
    VF0 <- 10 * diag(zc); OSW <- 1
    LLF2 <- KF(u[hh,], XF0, VF0, F, Ztld[hh,,], G, Q, R, limy, ISW, OSW, zc, TIMES)
    XPS <- LLF2$XPS; XFS <- LLF2$XFS
    VPS <- LLF2$VPS; VFS <- LLF2$VFS
    SIG2 <- LLF2$Ovar; GSIG2 <- 1
    # 平滑化 ----------------------------------------------------------
    LLF3 <- SMO(XPS, XFS, VPS, VFS, F, GSIG2, 1, SEASONALYTY, 1, zc, TIMES)
    Xs[hh,,] <- LLF3$XSS
    #------------------------------------------------------------
    
    for(t in 1:TIMES){  
      # step3の階差の計算--------------------------------------
      if(t>1){
        dift <- dift + (Xs[hh,1,t] - Xs[hh,1,t-1])^2
        difw <- difw + (Xs[hh, 2, t]+sum(Xs[hh, 2:SEASONALYTY, t-1]))^2
        
        for(d in 1:nz){
          difbeta[d] <- difbeta[d] + (Xs[hh,SEASONALYTY+d,t] 
                                        - Xs[hh,SEASONALYTY+d,t-1])^2
        }
      }
      #--------------------------------------------------------
      
      # step4の効用値の誤差計算(step4のθの尤度計算)------------
      Lsita <- Lsita + (u[hh,t] - t(Ztld[hh,,t])%*%Xs[hh,,t])^2
      
      #--------------------------------------------------------
    }
    
    # step3--------------------------------------
    Sita.sys[1,1,hh] <- irgamma(1, (nu0+TIMES)/2, (s0+dift)/2)
    Sita.sys[2,2,hh] <- irgamma(1, (nu0+TIMES)/2, (s0+difw)/2)
    for(d in 1:nz){
      Sita.sys[2+d,2+d,hh] <- irgamma(1, (nu0+TIMES)/2, (s0+difbeta[d])/2)
    }
    #--------------------------------------------
    
    ### step4--------------------------------------
    ## dlt側の計算
    # 現状のθを確保する
    old.sita.dlt <- Sita.dlt[hh]
    # 新しいθをサンプリング（酔歩サンプリング）
    new.sita.dlt <- Sita.dlt[hh] + rnorm(1, 0, sigma.dlt)

    # 尤度の計算（対数尤度の場合はヤコビアンで調整）
    new.Lsita.dlt <- Lsita + Nkernel(new.sita.dlt, Hdlt, D[hh,], Vsita.dlt)
    new.Lsita.dlt <- exp(-0.5*new.Lsita.dlt)
    old.Lsita.dlt <- Lsita + Nkernel(old.sita.dlt, Hdlt, D[hh,], Vsita.dlt)
    old.Lsita.dlt <- exp(-0.5*old.Lsita.dlt)
    
    # MHステップ
    alpha <- min(1, new.Lsita.dlt/old.Lsita.dlt)
    if(alpha=='NaN') alpha <- -1
    uni <- runif(1)
    if(uni < alpha){
      Sita.dlt[hh] <- new.sita.dlt
    }else{
      rej.dlt[hh] <- rej.dlt[hh] + 1
    }

    ## lmbd側の計算
    # 現状のθを確保する
    old.sita.lmbd <- Sita.lmbd[hh]
    # 新しいθをサンプリング（酔歩サンプリング）
    new.sita.lmbd <- Sita.lmbd[hh] + rnorm(1, 0, sigma.lmbd)
    
    # 尤度の計算（対数尤度の場合はヤコビアンで調整）
    new.Lsita.lmbd <- Lsita + Nkernel(new.sita.lmbd, Hlmbd, D[hh,], Vsita.lmbd)
    new.Lsita.lmbd <- exp(-0.5*new.Lsita.lmbd)
    old.Lsita.lmbd <- Lsita + Nkernel(old.sita.lmbd, Hlmbd, D[hh,], Vsita.lmbd)
    old.Lsita.lmbd <- exp(-0.5*old.Lsita.lmbd)
    
    # MHステップ
    alpha <- min(1, new.Lsita.lmbd/old.Lsita.lmbd)
    if(alpha=='NaN') alpha <- -1
    uni <- runif(1)
    if(uni < alpha){
      Sita.lmbd[hh] <- new.sita.lmbd
    }else{
      rej.lmbd[hh] <- rej.lmbd[hh] + 1
    }    
    #--------------------------------------------    
  }
  ### step5--------------------------------------
  ## dlt側の算出----
  # 多変量正規分布のパラメタの算出
  Hhat.dlt <- solve(crossprod(D)) %*% t(D) %*% Sita.dlt
  Dtld <- solve(crossprod(D) + A0) %*% (crossprod(D) %*% Hhat.dlt + A0%*%m0)
  rtld <- as.vector(Dtld)
  sig <-  (crossprod(D) + A0) %o% Vsita.dlt # %o%の項の前後を変えるのはダメ
  # 多変量正規分布でサンプリング
  Hdlt <- rmvnorm(nvar, rtld, as.matrix(data.frame(sig))) 
  ##-----------------
  ## lmbd側の算出----
  # 多変量正規分布のパラメタの算出
  Hhat.lmbd <- solve(crossprod(D)) %*% t(D) %*% Sita.lmbd
  Dtld <- solve(crossprod(D) + A0) %*% (crossprod(D) %*% Hhat.lmbd + A0%*%m0)
  rtld <- as.vector(Dtld)
  sig <-  (crossprod(D) + A0) %o% Vsita.lmbd # %o%の項の前後を変えるのはダメ
  # 多変量正規分布でサンプリング
  Hlmbd <- rmvnorm(nvar, rtld, as.matrix(data.frame(sig))) 
  ##-----------------
  #--------------------------------------------
  
  ### step6--------------------------------------
  ##dlt側の算出
  # 逆ウィッシャート分布のパラメタの算出
  div <- (Sita.dlt - D%*%matrix(Hdlt,nD,nvar))
  S <- crossprod(div)
  # 逆ウィッシャート分布でサンプリング
  Vsita.dlt <- riwish(f0 + nhh, V0 + S)  
  ##------------
  ##lmbd側の算出
  # 逆ウィッシャート分布のパラメタの算出
  div <- (Sita.lmbd - D%*%matrix(Hlmbd,nD,nvar))
  S <- crossprod(div)
  # 逆ウィッシャート分布でサンプリング
  Vsita.lmbd <- riwish(f0 + nhh, V0 + S)  
  ##------------  
  #--------------------------------------------
}
##-------------------------------------------