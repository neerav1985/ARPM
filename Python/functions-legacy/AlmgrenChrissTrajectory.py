import matplotlib.pyplot as plt
from numpy import sinh, cosh, ones, sqrt

plt.style.use('seaborn')


def AlmgrenChrissTrajectory(q,h_start,h_end,lam,eta,sigma,theta=None):
   #This function computes the optimal trading trajectory and its first order derivative according to the
   #Almgren-Chriss model
   # INPUTS:  
   #  q        [vector] (1 x k_) grid of volume times
   #  h_start [scalar]  initial holdings
   #  h_end   [scalar]  final holdings
   #  lam   [scalar] parameter for the mean-variance trade-off
   #  eta      [scalar] parameter for the slippage component of the underlying price dynamics
   #  sigma    [scalar] diffusion coefficient of the underlying price dynamics
   #  theta    [scalar] drift coefficient of the underlying price dynamics
   #  OPS:
   #  h        [vector] (1 x k_) Almgren-Chriss trading trajectory at each point of the time grid q
   #  h_speed  [vector] (1 x k_) Almgren-Chriss trading trajectory first order derivative at each point of the time grid q 
   #  NOTE: If not specified, theta is by default set equal to
   #        2*lam@sigma***hq_end (value that ensures monotonicity of the trajectories)
   #        If lam=0, then the VWAP trading trajectory is computed.
   
   # For details on the exercise, see here .
   ## Code
   if lam !=0:
      coeff2=2*lam*sigma**2
      if theta is None:
         theta=coeff2*h_end 
      coeff1=sqrt(lam/eta)*sigma
      coeff=sinh(coeff1*(q[-1]-q[0]))
      h_q=(h_start-theta/coeff2)*sinh(coeff1*(q[-1]-q))/coeff+(h_end-theta/coeff2)*sinh(coeff1*(q-q[0]))/coeff+theta/coeff2
      hspeed_q=-coeff1*(h_start-theta/coeff2)*cosh(coeff1*(q[-1]-q))/coeff+coeff1*(h_end-theta/coeff2)*cosh(coeff1*(q-q[0]))/coeff
   else:
      h_q=q*(h_end-h_start)/(q[-1]-q[0])+(h_start*q[-1]-h_end*q[0])/(q[-1]-q[0])
      hspeed_q=(h_end-h_start)/(q[-1]-q[0])*ones((1,len(q)))
   
   return h_q,hspeed_q


