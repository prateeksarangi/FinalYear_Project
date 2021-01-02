%Bowl Plot%
clc;
clear all;
close all;
N=5;
x=[0 0];
y=[0 0];
for n=1:2
    for k=1:N
        x(n)=x(n)+(1/N)*(sin(2*pi*k/N)*sin(2*pi*(k-n+1)/(N)));
        y(n)=y(n)+(2/N)*(cos(2*pi*k/N)*sin(2*pi*(k-n+1)/(N)));
   end
end
R=[x(1) x(2);x(2) x(1)];
P=[y(1) y(2)];
w=[inv(R)]*P';
d=2*cos(2*pi*k/N);
for w1=-50:1:50
    for w2=-50:1:50
        MSE(w1+51,w2+51)=0.5*(w1^2+w2^2)+(w1*w2*cos(2*pi/N)+2*w2*sin(2*pi/N))+2;
    end
end
[p q]=meshgrid(w1,w2);
figure;
mesh(MSE);%surf
xlabel('w2');ylabel('w1');zlabel('J');title('MMSE error surface');
%hold on;
figure
contour(MSE,w1)
[u,v]=gradient(MSE,0.5);
xlabel('w2');ylabel('w1');title('Contours of the error performance surface');
%hold on
w1=-50:1:50;
w2=-50:1:50;
figure
quiver(w1,w2,u,v)

