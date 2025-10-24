%% ================================================================
%  cacfar_2targets_fracdelay_rcmc.m
%  2目标 | 分数延时 + 物理速度(range walk) | 距离压缩后做 RCMC(Keystone)
%  然后慢时间窗 -> 慢时间FFT -> 2D CA-CFAR + 5x5 NMS
%  导出：rd_for_sod.jpg、rd_meta.mat、rd_power.mat（供后续映射/CFAR用）
% ================================================================
clear; clc; close all; rng(0);

%% ---------------- 体制与场景 ----------------
c  = 3e8;                   % 光速
fc = 10e9;                  % 载频
lambda = c/fc;

B   = 20e6;                 % LFM 带宽
Tp  = 20e-6;                % 脉宽
Fs  = 40e6;                 % 采样率
PRF = 8e3;                  % PRF
Np  = 1000;                 % CPI 脉冲数
T_CPI = Np/PRF;

Ns  = round(Tp*Fs);         % 发射脉冲采样点数（TX）
t_fast_tx = (0:Ns-1)/Fs;    % TX 快时间
t_slow = (0:Np-1)/PRF;      % 慢时间（脉序）

%% ---------------- 目标设置（2个，物理速度→多普勒） ----------------
R0    = [ 902.1, 2102.7 ];     % m (off-grid range)
Vmag  = [ 45,    30     ];     % 速度幅值 m/s
theta = [ 35,    58     ];     % 与LOS夹角 deg
v_rad = Vmag .* cosd(theta);    % 径向速度 m/s
amp0  = ones(1,numel(R0));      % 幅度同量级

targets = struct('R', num2cell(R0), 'v', num2cell(v_rad), 'amp', num2cell(amp0));

%% ---------------- CA-CFAR & NMS 参数 ----------------
Gr=8; Gd=1;         % 守护半径（距/多普勒）
Rr=12; Rd2=4;       % 参考半径（距/多普勒）
Pfa = 1e-6;         % 标称虚警率
kappa = 1.3;        % NMS 阈上裕量倍数
NBHD  = 2;          % NMS 5x5 邻域半径

%% ------------- 生成信号（分数延时 + range walk）、脉压、RCMC、RD -------------
t_all = tic;

% 发射 LFM + 快时间 Kaiser(β=10~12)
Kr = B/Tp;
betaFT = 12;
s_base = exp(1j*pi*Kr*(t_fast_tx.^2));
wR = kaiser_local(Ns, betaFT).';
s  = s_base .* wR;                   % 上窗后的发射
mf = conj(flip(s));                  % 匹配滤波器（与发射一致）

% -------- 接收快时间拉长，避免远距回波被截断 --------
tau_max = 2*( max(R0) + max(abs(v_rad))*T_CPI ) / c;   % 最大片上时延(含 range walk)
Ns_rx   = Ns + ceil(tau_max*Fs) + 8;    % 适当余量
t_fast_rx = (0:Ns_rx-1)/Fs;

% —— 分数延时 + range walk 生成回波（每脉冲用 tau_p）——
RX = zeros(Np, Ns_rx);
for p=1:Np
    tslow = t_slow(p);
    r = zeros(1,Ns_rx);
    for k=1:numel(targets)
        v  = targets(k).v;
        fD = 2*v / lambda;                          % 多普勒(Hz)
        tau_p = 2*(targets(k).R + v*tslow) / c;     % ★ 每脉冲分数时延
        echo  = frac_echo(t_fast_rx, tau_p, Tp, Kr, wR);  % ★ 用长接收轴
        r = r + targets(k).amp * exp(1j*2*pi*fD*tslow) .* echo;
    end
    RX(p,:) = r;
end

% 加噪（复高斯）
SNR_dB = 10;
sigpow = mean(abs(RX(:)).^2);
noipow = sigpow/10^(SNR_dB/10);
RX = RX + sqrt(noipow/2)*(randn(size(RX)) + 1j*randn(size(RX)));

% 距离向匹配滤波（脉冲压缩）：full -> 取“正时延段” [0..Ns_rx-1]
Yfull = zeros(Np, Ns_rx + Ns - 1);
for p=1:Np
    Yfull(p,:) = conv(RX(p,:), mf, 'full');
end
Y = Yfull(:, Ns : Ns + Ns_rx - 1);     % 对应回波 0..(Ns_rx-1) 的延迟（已压缩）

%% ---------------- Keystone RCMC（距离压缩后） ----------------
% 1) 对距离向做FFT得到“距离频率”（列方向）
Yf  = fftshift(fft(Y, [], 2), 2);              % Np x Ns_rx，沿列FFT
fr  = (-floor(Ns_rx/2):ceil(Ns_rx/2)-1)*(Fs/Ns_rx);   % 距离频率(Hz)，与列一一对应

% 2) 对每个距离频率bin，按 s = fc / (fc + fr) 缩放慢时间（Keystone）
Yf_rcmc = zeros(size(Yf));
for k=1:Ns_rx
    s = fc/(fc + fr(k));                        % Keystone 缩放因子（fc >> fr, 近似~1）
    tin  = t_slow;                              % 原慢时间网格
    tout = t_slow * s;                          % 缩放后的采样位置
    col  = Yf(:,k);                             % Np×1
    Yf_rcmc(:,k) = interp1(tin, col, tout, 'linear', 0);  % 超界补0
end

% 3) 逆FFT回到距离时域，得到 RCMC 后的距离压缩数据
Y_rcmc = ifft(ifftshift(Yf_rcmc, 2), [], 2);    % Np x Ns_rx

%% ---------------- 慢时间窗 + 慢时间FFT（形成 RD） ----------------
winD = hann_local(Np);                          % 慢时间窗
Yw   = Y_rcmc .* repmat(winD,1,Ns_rx);
RD   = fftshift(fft(Yw, Np, 1), 1);             % 复杂 RD（行=多普勒，列=距离）
P    = abs(RD).^2;                              % 线性功率

%% ---------------- 导出 RD 灰度图（功率dB，峰值归一），供SOD ----------------
dbSpan = [-40 -20];
RDdB = 10*log10(P./max(P(:)) + eps);
RDdB = min(dbSpan(2), max(dbSpan(1), RDdB));
img01 = (RDdB - dbSpan(1)) / diff(dbSpan);
img8  = uint8(round(255 * img01));          % 行=多普勒, 列=距离
img_adj = imadjust(img8, stretchlim(img8, [0.02 0.98]), []);
imwrite(img_adj, 'rd_for_sod.jpg', 'jpg', 'Quality', 95);

%% ---------------- 坐标轴 ----------------
r_axis  = (0:Ns_rx-1) * (c/(2*Fs));              % 距离轴（m）
fd_axis = (-floor(Np/2):ceil(Np/2)-1) * (PRF/Np); % 多普勒轴（Hz）
fd_axis = fd_axis(:);
r_axis  = r_axis(:);
v_axis  = fd_axis * (lambda/2);

% 保存轴信息（供 Python 管线映射）
meta = struct('Nd',size(P,1),'Nr',size(P,2),...
              'fd_axis',fd_axis(:),'r_axis',r_axis(:));
save('rd_meta.mat','meta');
save('rd_power.mat','P');

[Nd, Nr] = size(P);
t_upto_cfar = toc(t_all);

%% ---------------- 朴素 CA-CFAR（四重for，仅演示） ----------------
blkH = 2*(Rd2+Gd)+1;  blkW = 2*(Rr+Gr)+1;
gH   = 2*Gd+1;        gW   = 2*Gr+1;
numRef = blkH*blkW - gH*gW;
alpha  = numRef * (Pfa^(-1/numRef) - 1);

rStart = Rr+Gr+1; rEnd = Nr-(Rr+Gr);
dStart = Rd2+Gd+1; dEnd = Nd-(Rd2+Gd);

detMap = false(Nd,Nr);
thrMap = nan(Nd,Nr);

t_cfar = tic;
for di = dStart:dEnd
    for ri = rStart:rEnd
        r_lo = ri-(Rr+Gr); r_hi = ri+(Rr+Gr);
        d_lo = di-(Rd2+Gd); d_hi = di+(Rd2+Gd);
        rg_lo = ri-Gr; rg_hi = ri+Gr;
        dg_lo = di-Gd; dg_hi = di+Gd;

        sumRef = 0.0; cntRef = 0;
        for dj = d_lo:d_hi
            for rj = r_lo:r_hi
                if ~(dj>=dg_lo && dj<=dg_hi && rj>=rg_lo && rj<=rg_hi)
                    sumRef = sumRef + P(dj, rj);
                    cntRef = cntRef + 1;
                end
            end
        end
        noise_est = sumRef / cntRef;
        T = alpha * noise_est;
        thrMap(di,ri) = T;
        detMap(di,ri) = (P(di,ri) > T);
    end
end
t_cfar = toc(t_cfar);

%% ---------------- 5x5 NMS（局部极大 + 阈上裕量） ----------------
t_nms = tic;
peakMask = false(Nd,Nr);
cand = detMap & (P > kappa * thrMap);
for di = (dStart+NBHD):(dEnd-NBHD)
    for ri = (rStart+NBHD):(rEnd-NBHD)
        if cand(di,ri)
            nb = P(di-NBHD:di+NBHD, ri-NBHD:ri+NBHD);
            if P(di,ri) >= max(nb(:))
                peakMask(di,ri) = true;
            end
        end
    end
end
t_nms = toc(t_nms);
%% ---- 导出 CFAR+NMS 的检测清单（数值） ----
[idx_d, idx_r] = find(peakMask);                    % 行=多普勒(索引)，列=距离(索引)
det_range_m = r_axis(idx_r);                        % 距离 (m)
det_fd_Hz   = fd_axis(idx_d);                       % 多普勒 (Hz)
det_vel_mps = det_fd_Hz * (lambda/2);               % 速度 (m/s)

% 像素功率、门限、CFAR 比值
linIdx      = sub2ind(size(P), idx_d, idx_r);
det_power   = P(linIdx);                            % 线性功率
det_thr     = thrMap(linIdx);                       % CFAR 局部门限
cfar_ratio  = det_power ./ max(det_thr, eps);       % P / T（检测统计量）
maxP        = max(P(:));
det_power_dB = 10*log10(det_power./maxP + eps);     % 相对峰值 dB（用于可视化的一致性）

% 可选：只保留 0~3000 m 的结果
rngMax = 3000;
keep = (det_range_m >= 0) & (det_range_m <= rngMax);
det_range_m = det_range_m(keep);
det_fd_Hz   = det_fd_Hz(keep);
det_vel_mps = det_vel_mps(keep);
det_power   = det_power(keep);
det_thr     = det_thr(keep);
cfar_ratio  = cfar_ratio(keep);
det_power_dB= det_power_dB(keep);
idx_d = idx_d(keep); idx_r = idx_r(keep);

% 排序：按功率从大到小
[~, ord] = sort(det_power, 'descend');
det_range_m = det_range_m(ord);
det_fd_Hz   = det_fd_Hz(ord);
det_vel_mps = det_vel_mps(ord);
det_power   = det_power(ord);
det_thr     = det_thr(ord);
cfar_ratio  = cfar_ratio(ord);
det_power_dB= det_power_dB(ord);
idx_d = idx_d(ord); idx_r = idx_r(ord);

% 打印到控制台
fprintf('\n[CFAR+NMS] 检测到 %d 个峰（已按功率排序，显示前 10 条）：\n', numel(det_range_m));
nshow = min(10, numel(det_range_m));
fprintf('  #%3s  row  col     Range(m)     fd(Hz)      Vel(m/s)     Power      Thr       P/T      P(dB)\n','');
for i = 1:nshow
    fprintf('  #%3d  %4d %4d  %10.3f  %10.3f  %10.3f  %9.3g  %9.3g  %8.3f  %7.2f\n', ...
        i, idx_d(i), idx_r(i), det_range_m(i), det_fd_Hz(i), det_vel_mps(i), ...
        det_power(i), det_thr(i), cfar_ratio(i), det_power_dB(i));
end

% 保存为 CSV & MAT
T = table(idx_d, idx_r, det_range_m, det_fd_Hz, det_vel_mps, ...
          det_power, det_thr, cfar_ratio, det_power_dB, ...
          'VariableNames', {'row','col','range_m','fd_Hz','vel_mps', ...
                            'power','thr','cfar_ratio','power_dB'});
writetable(T, 'cfar_detections.csv');
save('cfar_detections.mat', 'idx_d','idx_r','det_range_m','det_fd_Hz','det_vel_mps', ...
                           'det_power','det_thr','cfar_ratio','det_power_dB');
fprintf('[OK] 检测结果已保存：cfar_detections.csv / cfar_detections.mat\n');

%% ---------------- 仅画 0~3000 m ----------------
rngMax = 3000;

figure('Color','w');
imagesc(r_axis, fd_axis, RDdB); axis xy;
cb = colorbar;                 % 取得色标句柄
title(cb, 'dB');               % 在色标条顶上加单位
xlim([0 rngMax]); caxis([-40 0]);
xlabel('Range (m)'); ylabel('Doppler (Hz)');
title('RD Amplitude squared','FontName','Times New Roman','FontSize',12,'FontWeight', 'bold');
set(gca,'FontName','Times New Roman','FontSize',12,'FontWeight', 'bold');
ylim([-2000 3000]);                 % 仅显示 -3000~3000 Hz 的多普勒范围
set(gca,'YTick',[ -2000 -1000 0 500 1000 1500 2000  2500 3000]);    % 只打这几个刻度（若只要两端就用 [-3000 3000]）

% 图2：朴素 CA-CFAR 掩码
figure('Color','w');
imagesc(r_axis, v_axis, detMap); axis xy; colorbar;
xlim([0 rngMax]);
xlabel('Range (m)'); ylabel('Velocity (m/s)');
title('Naive CA-CFAR detections','FontName','Times New Roman','FontSize',12,'FontWeight', 'bold');

% 图3：在 RD（峰值归一 dB）上圈出 NMS 峰
[di_idx, ri_idx] = find(peakMask);
det_ranges = r_axis(ri_idx);
det_vels   = v_axis(di_idx);

figure('Color','w');
imagesc(r_axis, v_axis, RDdB); axis xy; hold on; 
cb = colorbar;                 % 取得色标句柄
title(cb, 'dB');               % 在色标条顶上加单位
xlim([0 rngMax]); caxis([-40 0]);
xlabel('Range (m)'); ylabel('Velocity (m/s)');
title('CA-CFAR Detections','FontName','Times New Roman','FontSize',12,'FontWeight', 'bold');
plot(det_ranges, det_vels, 'wo', 'MarkerSize', 6, 'LineWidth', 1.2);
set(gca,'FontName','Times New Roman','FontSize',12,'FontWeight', 'bold');
hold off;

%% ---------------- 打印计时与结果 ----------------
fprintf('\n=========== 2-targets | 分数延时 + RCMC(Keystone) | Kaiser(beta=%.1f) ===========\n', betaFT);
fprintf('Data size (Nd x Nr): %d x %d  (pixels = %d)\n', Nd, Nr, Nd*Nr);
fprintf('Ref window: %dx%d, Guard: %dx%d, N_ref=%d\n', 2*(Rd2+Gd)+1, 2*(Rr+Gr)+1, 2*Gd+1, 2*Gr+1, numRef);
fprintf('Pfa=%.1e, alpha=%.6f\n', Pfa, alpha);
fprintf('[Up to CFAR] 生成->MF->RCMC->FFT: %.3f s\n', t_upto_cfar);
fprintf('[CFAR only] 朴素CFAR     : %.3f s  (ROI吞吐≈ %.2f Mcells/s)\n', ...
        t_cfar, ((dEnd-dStart+1)*(rEnd-rStart+1)/1e6)/max(t_cfar,eps) );
fprintf('[NMS     ] 峰值去重       : %.3f s\n', t_nms);
fprintf('[Full    ] 全流程总用时   : %.3f s\n', t_upto_cfar + t_cfar + t_nms);

fprintf('\nTargets (physical):\n');
for k=1:numel(targets)
    fD = 2*targets(k).v/lambda;
    fprintf('  #%d  R0=%7.2f m  V=%6.2f m/s  theta=%6.1f deg  vr=%7.2f m/s  fD=%.2f Hz\n', ...
        k, targets(k).R, Vmag(k), theta(k), targets(k).v, fD);
end

%% ================== 本地函数 ==================
function y = frac_echo(t_fast, tau, Tp, Kr, wR)
    % 分数延时的LFM回波：s(t - tau)，并与发射端同窗（PSF一致）
    u  = t_fast - tau;                         % 相对时间
    g  = (u>=0 & u<=Tp);                       % 脉冲门
    y0 = exp(1j*pi*Kr*u.^2) .* g;              % 连续时间LFM
    Ns = numel(wR);
    nrm = (u/Tp)*(Ns-1)+1;                     % 映射到 [1,Ns]
    w   = interp1(1:Ns, wR, nrm, 'linear', 0); % 线性插值，窗外置0
    y   = y0 .* w;
end

function w = kaiser_local(N,beta)
    if N<=1, w=ones(N,1); return; end
    n=(0:N-1).'; a=(N-1)/2; t=(n-a)./a;
    w = besseli(0,beta*sqrt(1-t.^2)) / besseli(0,beta);
end

function w = hann_local(N)
    if N<=1, w=ones(N,1); return; end
    n=(0:N-1).'; w=0.5 - 0.5*cos(2*pi*n/(N-1));
end

function w = blackmanharris_local(N)
    if N<=1, w=ones(N,1); return; end
    n=(0:N-1)'; M=N-1;
    a0=0.35875; a1=0.48829; a2=0.14128; a3=0.01168;
    w = a0 - a1*cos(2*pi*n/M) + a2*cos(4*pi*n/M) - a3*cos(6*pi*n/M);
end
