%% Online Summer Course on Advance Optimization Techniques and Hands-on with MATLAB
%% Lab IV- Particle Swarm Optimization for function Optimization- By Dr. S. J. Nanda, MNIT Jaipur
%%% Problem : To determine the minimum value of Sin(x) in the range [0,2*pi]. 
clc; clear all; close all; 
N=20; %%%Number of Particles
P=rand(N,1)*2*pi; %% Initial Position of Particles
V=rand(N,1); %% Initial Velocity of Particles
c1=2.05; c2=2.05; % acceleration factor
W=0.4; % Inertia weight
Lbest=P; %%% LocalBest Positions
f=sin(P); %%% Fitness Evaluation
[a,b]=min(f);
Gbest=P(b,1); %% Global Best Position
%Iter
for z=1:20
%%%Update position and velocity
for i=1:N
   NV(i,1)=W*V(i,1)+c1*rand*(Lbest(i,1)-P(i,1))+c2*rand*(Gbest-P(i,1));
   NP(i,1)=P(i,1)+NV(i,1);
   if (NP(i,1)>2*pi) || (NP(i,1)<0) %%% Bounding within the range of Sin(x)
      NP(i,1)=P(i,1);
   end   
end
%%%%%Local best Calculation
 nf=sin(NP);
 for i=1:N
     if nf(i,1)<f(i,1)
         Lbest(i,1)=NP(i,1);
     end
 end
 %%%Global Best Calculation
 nf1=sin(Lbest);
 [an,bn]=min(nf1);
 Gbest=Lbest(bn,1);
 %%%Next generation
 P=Lbest;%%%Lbest%%NP
 f=nf1;%%%nf1%%nf
 V=NV;
 ff(z)=an;
 end
plot(ff)
xlabel('Iteration');
ylabel('Fitness function Sin(x) Min. value'); 
title('PSO convergence characteristic')

        