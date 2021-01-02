%% Online Summer Course on Advance Optimization Techniques and Hands-on with MATLAB
%% Lab III- Binary Genetic Algorithm for function Optimization- By Dr. S. J. Nanda, MNIT Jaipur
%%% Problem : To determine the maximum value of Sin(x) in the range [0,2*pi]. 
clc; close all; clear all;
tic
N=10;%no of population 
B=10;%no of bit
x=hardlim(rand(N,B)-.5) %10 chromozone (every chromozone has 10 bits)
% bring the binary into decimal
t=B-1:-1:0 
t1=2.^t
% multipy with x for creating decimal value
dx=x*t1' % decimal value
dx1=(dx*2*pi)/(2^B-1) %convert between 0 to 2pi
% given fitness function 
f=sin(dx1) %evaluated fitness
% fitness if sort from maximum to mimimum-coz maximum problem taken
s=sort(f,'descend')    %sortng from maximum to minimum
 for i=1:N
     for j=1:B
         if s(i)==f(j)
             sx(i,:)=x(j,:)
         end
     end
 end
 %sx1=sx(1:0.8*N,:) %%keeping 80% of population -selection
for itr=1:20
%%%%  cross over(Single point crossover)
pc=0.8 %probability of crossover
tn=pc*N/2;%%% Total number of crossover
C_child1=[];
for i=1:tn
    r1=ceil(rand(1,2)*(N-1));    %random selection of prarent for crossover
    r2=ceil(rand*(B-1));   % random crossover point
    c1=sx(r1(1,1),1:B);
    c2=sx(r1(1,2),1:B);
    chd1=[c1(1,1:r2),c2(1,r2+1:B)]
    chd2=[c2(1,1:r2),c1(1,r2+1:B)]
    chd=[chd1;chd2];
    C_child1=[C_child1;chd];
end 
%%%%%mutation %
 mp=0.2%probability of mutaion
 no_mutation=N*B*mp %% total no of muteted bits
 C_child2=C_child1;
 for k=1:no_mutation
     r3=ceil(rand*(2*tn-1));
     r4=ceil(rand*(B-1));
     C_child2(r3,r4)=~C_child2(r3,r4); %mutation
 end  
    %%%% reselection
    new=[sx;C_child1;C_child2];
    p=(B-1):-1:0;
    dec2=new*(2.^p)'; %Decimal conversion
    x2=(dec2/2^B-1)*2*pi;%convert between 0 to 2pi
    %%%%Again fitness
    f2=sin(x2);
    %%%%%%%%%%%%%Again sorting
    s0=sort(f2,'descend');
    for i=1:size(new,1)
        for j=1:size(new,1)
            if s0(i)==f2(j)
                s11(i,:)=new(j,:);
            end 
        end 
    end 
    sx=s11(1:10,:); %%% Reassignment as parent
    fit(itr)=s0(1,1)
end 
toc
plot(fit)
grid on
xlabel('generation');ylabel('Sin(x) Maximum Value'); title('Genetic Algorithm Convergence')