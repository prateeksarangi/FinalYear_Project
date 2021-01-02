%% Online Summer Course on Advance Optimization Techniques and Hands-on with MATLAB
%% Lab IV- Differential Evolution for function Optimization- By Dr. S. J. Nanda, MNIT Jaipur
%%% Problem : To determine the minimum value of Sin(x) in the range [0,2*pi]. 
clc; clear all; close all; 
N=20; %%%Number of Initial Vector 
P=rand(N,1)*2*pi; %% Initial Position of Target Vector
 F=0.5;%1
 CR=0.5; %%%Crossover Rate
 itmax=20;
 tic
 for itr=1:itmax
for i =1:N
    r1 = round(rand*(N-1)+1);% Random Intiger Value [1,20]
    r2 = round(rand*(N-1)+1);
    r3 = round(rand*(N-1)+1);
    if r2 == r1
        r2 = round(rand*(N-1)+1);
    end
    if (r3 == r2||r3 == r1)
        r3 = round(rand*(N-1)+1);
    end
 %%%%Creating Trail Vector with Mutation
 V(i,1)=P(r1,1)+F*(P(r2,1)-P(r3,1)); 
 if (V(i,1)>2*pi) || (V(i,1)<0) %%% Bounding within the range of Sin(x)
     V(i,1)=P(i,1);
 end
end
%%% Crossover 
for i=1:N
    rx(i)=rand;
    if (rx(i)< CR)||(i==randi([1,N]))
        U(i,1)=V(i,1);
    else
        U(i,1)=P(i,1);
    end
end
%%%%Selection
f1=sin(P);
f2=sin(U);
for i=1:N
    if f2(i)<f1(i)
        P(i,1)=U(i,1);
    end
end
 Fb=sin(P);
 a(itr)=min(Fb);
 end
 toc
 plot(a)
 xlabel('Iteration');
 ylabel('Fitness function Sin(x) Min. value'); 
 title('DE convergence characteristic')

        