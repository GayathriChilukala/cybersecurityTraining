clear; close all; clc
N= 14;
%% Loading IEEE 14 Bus Case in MATPOWER
define_constants;
mpc = loadcase('case14');

%% Power Flow Calculations in MATPOWER
results = runpf(mpc);

%% Building Graph: Type I
%Coordinates of the buses
xloc = [150;250;693;685;325;370;727;830;715;570;460;22;385;560]; 
yloc = -[363;583;615;350;430;300;287;280;225;220;175;130;70;130];

% Edges
s = mpc.branch(:,1);  % From Bus
t = mpc.branch(:,2);  % To Bus

weights = 1 ./ sqrt((xloc(s)-xloc(t)).^2 + (yloc(s)-yloc(t)).^2 );
weights = normalize(weights, 'range')+0.01;
G = simplify(graph(s,t,weights));
%L = laplacian(G);

%% Building Graph: Type II
L = makeYbus(mpc);

%% Computing the GFT basis functions
% Eigen-decomposition of the Graph Laplacian
[V,D] = eig(full(L));

% Sorting Eigen-vectors and Eigen-values
[lambda,inds] = sort(diag(D),'ascend');
V = V(:,inds);

%% Original graph signal
f_n = results.bus(:,VA);

%% Plotting the Graph Signal

figure(1), h = plot(G, 'XData', xloc, 'YData', yloc, ...
                   'MarkerSize', 30, 'NodeLabel',{}); 
colormap jet;
xticks([]), yticks([]); 
h.NodeCData = f_n;  % Graph Signal
c = colorbar; % Graph Signal
c.Label.String = 'Voltage Angle (in Degrees)';
c.Label.Interpreter = 'latex';
c.TickLabelInterpreter= 'latex';
c.FontSize = 30;
h.LineWidth = 3;
h.EdgeColor = 'k';

for i=1:N, nodeID{i}=num2str(i); end
figure(1),
text(h.XData-0.1, h.YData, nodeID, 'HorizontalAlignment',...
     'center', 'FontSize', 15);
 


%% False Data Injection Attack
Attack_Bus_Number = 14;
False_Measurement = 5;
f_nCorrupted = f_n; 
f_nCorrupted(Attack_Bus_Number) = False_Measurement;


%% GFT of the original and corrupted signals
F = (V'*f_n);
F_Corrupted = (V'*f_nCorrupted);


%% Plotting the GFTs

% normalised graph frequency (From 0 to 1)
norm_freq = (lambda-min(lambda)) / (max(lambda)-min(lambda));

figure(2), h = plot(norm_freq,abs(F),'-d','linewidth',4); hold on
ax = ancestor(h, 'axes');
xrule = ax.XAxis; yrule = ax.YAxis;
xrule.FontSize = 24; yrule.FontSize = 24;
h = plot(norm_freq,abs(F_Corrupted),'-o','linewidth',4); hold on
xlabel('Normalized Frequency (0 to 1)', 'FontSize', 24)
ylabel('Magnitude response', 'FontSize', 24)
legend('Original signal','Corrupted Signal', 'FontSize', 24)

%% High-pass graph filter

% Cut-off normalised graph frequency 
lambda_cut_off = 0.1;

% High-pass filter
H = zeros(length(lambda),1);
H(norm_freq>lambda_cut_off)= 1;


%% High-Pass Filtering 
F_HighPass = F.*H;
F_Corrupted_HighPass = F_Corrupted.*H;

%% Amount of High-frequency Component, gamma
gammaOriginal = sum(abs(F_HighPass));
gammaCorrupted = sum(abs(F_Corrupted_HighPass))

%% Threshold Selection 
Threshold = 20;

%% Attack Declaration
Attack_Flag = gammaCorrupted >= Threshold;
disp(Attack_Flag)

