%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%% To integrate the advection part of the equations
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

function out=Advect(t,vec,Ce,param)

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%% The parameters used in this function        %%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
N=param.N;
length=param.L;
h=length/N;
G1=param.G1;
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% reassemble the vectors
L=vec(1:N);   
T=vec(N+1:2*N);
P=vec(2*N+1:3*N);
RFP=vec(3*N+1:4*N);
CFP=vec(4*N+1:5*N);

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%% The FD stencils
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
Lp=[L(2:N); L(N-1)]; % shifted to right (note L(N+1)=L(N-1) because of no-flux condition
Lm=[L(2); L(1:N-1)]; % shifted to left  (note L(0)=L(2) because of no-flux condition

Tp=[T(2:N); T(N-1)];
Tm=[T(2); T(1:N-1)];

Cep=[Ce(2:N); Ce(N-1)];
Cem=[Ce(2); Ce(1:N-1)];

Pp=[P(2:N); P(N-1)];
Pm=[P(2); P(1:N-1)];

RFPp=[RFP(2:N); RFP(N-1)];
RFPm=[RFP(2); RFP(1:N-1)];

CFPp=[CFP(2:N); CFP(N-1)];
CFPm=[CFP(2); CFP(1:N-1)];



%% The rhs of the discretized PDE (advection parts)
%% Note the 1e-5 is there to avoid troubles with zero-values of Ce
F1=G1./(Ce+1e-5).*1/(4*h^2).*(Lp-Lm).*(Cep-Cem);
F2=G1./(Ce+1e-5).*1/(4*h^2).*(Tp-Tm).*(Cep-Cem);
F3=G1./(Ce+1e-5).*1/(4*h^2).*(Pp-Pm).*(Cep-Cem);
F4=G1./(Ce+1e-5).*1/(4*h^2).*(RFPp-RFPm).*(Cep-Cem);
F5=G1./(Ce+1e-5).*1/(4*h^2).*(CFPp-CFPm).*(Cep-Cem);


%% send it back to the main function

out=[F1;F2;F3;F4;F5];




