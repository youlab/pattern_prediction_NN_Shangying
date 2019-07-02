%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%% To integrate the diffusion part of the equations
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

function out=Diffusion(t,Ce,param)

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%% The parameters used in this function        %%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
N=param.N;
h=param.L/N;
G1=param.G1;

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%% The FD stencils
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

Cep=[Ce(2:N); Ce(N-1)]; 
Cem=[Ce(2); Ce(1:N-1)];

radius_vec=linspace(0,param.L,param.N);

%% The rhs of the discretized PDE (diffusion part)

out=G1/h^2*(Cep-2*Ce+Cem)+ G1/2/h./radius_vec'.*(Cep-Cem);
out(1)=2*G1/h^2*(Cep(1)-2*Ce(1)+Cem(1)); % origin is special: no 1/r term (singular), but factor 2

% the factor two can be understood as follows: go back to cartesian
% coordinates where the Laplacian becomes u_xx+u_yy, then do partial
% derivatives along both directions, but u(0,deltaR)=u(deltaR,0) by radial
% symmetry, so that we just end up having two times the Laplacian in
% r-direction.

%out=G1/h^2*(Cep-2*Ce+Cem);







