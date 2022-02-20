假设森林平均碳存量与树平均年龄有关：

在t时刻：

$V_t=f_t(y_t)$

森林每年增长率为$G(y_t)=\frac{\part V_t}{\part y_t}$

可写出每年森林增长方程：

$V_{t+1}=V_t+G(y_t)-H_t$

其中$H_t$是每年砍伐量（假设在年末进行）

由$y_t=f^{-1}(V_t)$,设砍伐量与$V_t$相关

$H_t=x_tV_t$

代入得：

$V_{t+1}=V_t +G(f^{-1}(V_t))-x_tV_t$

方程只与Vt相关

设砍树时损耗率为$\varphi$

产品腐烂率为$\theta$

森林产品净CO2去除量(假设全用作产品）：

$M_t=(1-\varphi)H_t-\theta{\sum_{\tau=0}^{t-1}(1-\theta)^{t-1-\tau}F_{\tau} }  $

Model1:

设产品使用不释放CO2：（$\theta=0$）

总CO2吸收量（每年）：

$A_t=M_t+V_{t+1}-V_t=G(f^{-1}(V_t))-\varphi x_tV_t$

拟合曲线

$V_t=\frac{a}{1+be^{-ky_t}}$

$y_t=-\frac{1}{k}ln\frac{a-V_t}{bV_t}$

要想CO2吸收量高，有：

- $V_{t+1}=V_t$,即$G(f^{-1}(V_t))=x_tV_t$(稳态)

- $\frac{\part A_t}{\part V_t}=0$

  

代入得：

$x_t=\frac{k(a-V_t)}{a}$

$V_t=\frac{a}{2}$

最大吸收CO2量：

$A_tmax=\frac{(1-\varphi)ka}{4}$