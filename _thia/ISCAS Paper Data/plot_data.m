data = grasp_data.data;
V = grasp_data.V;
gsra = grasp_data.gsra;
gref = grasp_data.gref;
T = grasp_data.T;
freq = grasp_data.freq;
spikes = grasp_data.spikes;
dt =  grasp_data.parameters.dt;

close all

%initialize figure and axes
figure(1)

ax1 = subplot(4,1,1);
ax2 = subplot(4,1,2);
ax3 = subplot(4,1,3);
ax4 = subplot(4,1,4);

subplot(4,1,1);
plot(T(1:(length(T))),data(:,6));
ylabel('Grip Force (N)');
hold on;
subplot(4,1,2);
plot(T(1:(length(T))),V);
ylabel('Neuron Potential (mV)')
hold on;
subplot(4,1,3);
plot(T(1:length(T)),data(:,8),'r');
ylabel('EMG Gain');
hold on
% plot(T(1:(length(T))),gsra);
% ylabel('Neuron Conductance (mV)'); 
% hold on
% plot(T(1:(length(T))),gref,'r')
% hold on
subplot(4,1,4);
plot(T(1:(length(T))),freq);
ylabel('Theoretical Spike Rate (Hz)');

linkaxes([ax1,ax2,ax3, ax4],'x');

%% ISCAS Fig 3

close 
a = 44345;
b = 44465;
Vth = 0.019;

v = V(a:b);
t = T(a:b)-T(a);

for i = 1:length(v)
    if v(i) >= Vth
        spike(i,1) = 1;
    else 
        spike(i,1) = 0;
    end
end

window = round(0.06/dt);
val = spike(1:window);

for i = window/2:(length(t)-window/2)
    val=vertcat(val,spike(i));
    val(1)=[];
    spike_rate(i) = sum(val)/(window*dt);
end

spike_rate(1:window/2) = 0;

figure()
title('Neuromorphic Tactile Feedback')
n = 6;
ax1 = subplot(n,1,1:2);
ax2 = subplot(n,1,3:4);
ax3 = subplot(n,1,n-1);
ax4 = subplot(n,1,n);

subplot(n,1,1:2);
plot(t,data(a:b,6),'LineWidth',3,'Color','k');
ylabel('Grip Force (N)','FontWeight','bold');
ax1.YLim=([0 15]);
ax1.YTick=([0 10]);
ax1.XTick=[];
hold on

subplot(n,1,3:4)
plot(t,V(a:b),'LineWidth',2,'Color','k');
hold on
ylabel('Neuron Potential (mV)','FontWeight','bold');
ax2.YLim=([-0.09 0.03]);
ax2.YTick=([-0.05 0]);
ax2.XTick=[];
hold on

subplot(n,1,5)
plot(t,spike_rate,'LineWidth',2,'Color','k');
ylabel('Spike Rate (Hz)','FontWeight','bold');
ax3.YLim=([0 60]);
ax3.XTick=[];
hold on


subplot(n,1,6)
plot(t,data(a:b,8),'LineWidth',2,'Color','k');
ylabel('EMG Gain','FontWeight','bold');
ax4.YLim=([-0.1 1.1]);
ax4.YTick=([0 1]);
xlabel('Time (s)','FontSize',12,'FontWeight','bold');
hold on

suptitle('Neuromorphic Tactile Response')

linkaxes([ax1,ax2,ax3, ax4],'x');
ax1.XLim=([0 0.6]);

%% ISCAS Fig 4 - able-body experiment results -- broken objects

if exist('User1', 'var') == 0
    load User1
end
close all

y(1,1) = 100*User1.average(1,1);
y(1,2) = 100*User1.average(1,3);
y(1,3) = 100*User1.average(1,2);
SEM(1,1) = 100*User1.SEM(1,1);
SEM(1,2) = 100*User1.SEM(1,3);
SEM(1,3) = 100*User1.SEM(1,2);

fig = barwitherr(SEM,y);
fig.FaceColor = [0.5 0.5 0.5];
title('Prosthesis Grasping Experiment');
ax = gca;
ax.FontSize = 12;
ax.YTick = [0 10 20 30];
ax.XTickLabel = {'No Feedback','Grip Force Feedback','Spike Rate Feedback'};
ylabel('Broken Objects (%)')

%% ISCAS Fig 4 - amputee experiment results -- broken objects

if exist('Amp1', 'var') == 0
    load Amp1
end
close all

user = Amp1;

y(1,1) = 100*user.average(1,1);
y(1,2) = 100*user.average(1,3);
y(1,3) = 100*user.average(1,2);
SEM(1,1) = 100*user.SEM(1,1);
SEM(1,2) = 100*user.SEM(1,3);
SEM(1,3) = 100*user.SEM(1,2);

fig = barwitherr(SEM,y);
fig.FaceColor = [0.5 0.5 0.5];
title('Prosthesis Grasping Experiment');
ax = gca;
ax.FontSize = 12;
ax.YTick = [0 10 20 30];
ax.XTickLabel = {'No Feedback','Grip Force Feedback','Spike Rate Feedback'};
ylabel('Broken Objects (%)')

%% ISCAS Fig 4 - combined experiment results -- broken objects

if exist('All', 'var') == 0
    load All
end
close all

user = All;

y(1,1) = 100*user.average(1,1);     %No Feedback
y(1,2) = 100*user.average(1,3);     %Compliant Grasping Feedback
y(1,3) = 100*user.average(1,2);     %Neuromorphic Rate Feedback
SEM(1,1) = 100*user.SEM(1,1);
SEM(1,2) = 100*user.SEM(1,3);
SEM(1,3) = 100*user.SEM(1,2);

fig = barwitherr(SEM,y);
fig.FaceColor = [0.5 0.5 0.5];
title('Prosthesis Grasping Experiment');
ax = gca;
ax.FontSize = 12;
ax.YTick = [0 10 20 30];
ax.XTickLabel = {'No Feedback','Grip Force Feedback','Spike Rate Feedback'};
ylabel('Broken Objects (%)')

