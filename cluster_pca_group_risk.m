function cluster_pca_group_risk()
%% =========================
% 0. 基本配置
% ==========================
dataPath = 'E:\project1\CaseStudy_health_lifestyle_dataset.csv';

% 连续变量（10 个）
contVars = { ...
    'age', ...
    'daily_steps', ...
    'sleep_hours', ...
    'water_intake_l', ...
    'calories_consumed', ...
    'bmi', ...
    'resting_hr', ...
    'systolic_bp', ...
    'diastolic_bp', ...
    'cholesterol'};

% 原始性别字符串列名（CSV 里有）
genderStrVar = 'gender';

% 二元变量（4 个，gender_num 如果不存在就用 gender 映射）
binVars = {'smoker', 'alcohol', 'family_history', 'gender_num'};

% 风险列
riskVar = 'disease_risk';

% 聚类 segment 数
Kseg = 10;

%% =========================
% 1. 读数据 & 生成 gender_num（如果没有）
% ==========================
Traw = readtable(dataPath);

% 如果 gender_num 不存在，但有 gender 字符串列，则映射生成
if ~ismember('gender_num', Traw.Properties.VariableNames)
    if ~ismember(genderStrVar, Traw.Properties.VariableNames)
        error('数据中既没有 gender_num，也没有 gender，无法生成性别变量。');
    end
    
    gStr = string(Traw.(genderStrVar));
    gender_num = nan(size(gStr));
    
    % 映射规则：Male/M -> 1, Female/F -> 0
    gender_num(ismember(lower(gStr), ["male","m"]))   = 1;
    gender_num(ismember(lower(gStr), ["female","f"])) = 0;
    
    Traw.gender_num = gender_num;
    fprintf('已根据 "%s" 列生成 gender_num (Male/M -> 1, Female/F -> 0)。\n', genderStrVar);
end

% 需要的全部列（先不含合并指标）
needVars = [contVars, binVars, {riskVar}];

% 检查缺失列
missing = setdiff(needVars, Traw.Properties.VariableNames);
if ~isempty(missing)
    error('以下列在 CSV 中缺失：\n%s', strjoin(missing, ', '));
end

% 只取需要的列
T = Traw(:, needVars);

% 去掉缺失
T = rmmissing(T);
n = height(T);
fprintf('有效样本数: %d\n', n);

%% =========================
% 2. 二元变量组合 → patternCode（第一层降阶）
% ==========================
patTbl = T(:, binVars);

codeStr = strcat( ...
    string(patTbl.smoker), ...
    string(patTbl.alcohol), ...
    string(patTbl.family_history), ...
    string(patTbl.gender_num));

[~, ~, patternCode] = unique(codeStr);   % 1..Kpat
T.patternCode = patternCode;

% 看一下典型模式（patternCode）本身的风险和布尔特征均值
tmpTbl = T(:, [binVars, {riskVar, 'patternCode'}]);
patSummary = groupsummary(tmpTbl, 'patternCode', 'mean', [riskVar, binVars]);

disp('===== 二元组合模式 patternCode 汇总（第一层降阶结果） =====');
disp(patSummary);

%% =========================
% 3. PCA + KMeans 在“连续Z + pattern one-hot”上做分群
% ==========================

% 连续变量 → Z-score
Xcont  = T{:, contVars};
XcontZ = zscore(Xcont);

% patternCode → one-hot
patternDummy = dummyvar(patternCode);  % 需要 Statistics and ML Toolbox
% 拼在一起：连续Z + one-hot 模式
X_for_cluster = [XcontZ, patternDummy];

% PCA
[coeff, score, ~, ~, expl] = pca(X_for_cluster);
cumExpl = cumsum(expl);
kPC = find(cumExpl >= 80, 1);   % 至少解释 80% 方差
if isempty(kPC)
    kPC = min(8, numel(expl));
end
fprintf('PCA 选取前 %d 个主成分，累计解释方差 = %.2f%%\n', ...
    kPC, cumExpl(kPC));

Xpc = score(:, 1:kPC);

% KMeans 聚类
rng(0);
opts = statset('MaxIter', 1000);
[segId, ~, sumd] = kmeans(Xpc, Kseg, ...
    'Replicates', 20, ...
    'Options'   , opts);

T.segment = segId;

fprintf('KMeans 完成，%d 个 segment，总距离向量：\n', Kseg);
disp(sumd');

%% =========================
% 4. 在表里增加“合并指标”：Any of smoking / alcohol / family history
% ==========================
% 只要抽烟/喝酒/有家族史之一为 1，则该合并变量为 1
T.any_unhealthy = double( T.smoker | T.alcohol | T.family_history );

% 现在一共 10 连续 + 4 原始二元 + 1 合并二元 = 15 个特征
allFeatCodes = [contVars, binVars, {'any_unhealthy'}];

% 对应用于展示的“文档名”：
featDisplayNames = { ...
   'Age', ...
   'Daily steps', ...
   'Sleep hours', ...
   'Water intake (L)', ...
   'Calories consumed', ...
   'Body mass index (BMI)', ...
   'Resting heart rate', ...
   'Systolic blood pressure', ...
   'Diastolic blood pressure', ...
   'Cholesterol level', ...
   'Smoking (1 = yes)', ...
   'Alcohol use (1 = yes)', ...
   'Family history (1 = yes)', ...
   'Gender (male = 1)', ...
   'Any of smoking / alcohol / family history'};

%% =========================
% 5. segment 层面的风险和特征均值
% ==========================
% 5.1 每个 segment 的样本数、风险
segCount    = accumarray(segId, 1, [Kseg, 1]);
segRiskMean = accumarray(segId, T.(riskVar), [Kseg, 1], @mean);
segPopShare = segCount / n;

segSummaryRaw = table((1:Kseg)', segCount, segPopShare, segRiskMean, ...
    'VariableNames', {'segment','n','pop_share','risk_mean'});

% 按风险从高到低排列
segSummaryPlot = sortrows(segSummaryRaw, 'risk_mean', 'descend');

overallRisk = mean(T.(riskVar));

fprintf('\n===== segment 概览（按风险从高到低）=====\n');
disp(segSummaryPlot);

% 5.2 每个 segment 的所有特征均值
meanDataVars = allFeatCodes;
segFeatMeans = groupsummary(T, 'segment', 'mean', meanDataVars);
segFeatMeans = sortrows(segFeatMeans, 'segment');   % 保证按 segment 1..K 排

meanColNames = strcat('mean_', allFeatCodes);       % groupsummary 的列名

% 整体矩阵 (全部分组)
M_allSeg = segFeatMeans{:, meanColNames};           % nSeg × 15

%% =========================
% 6. Figure 1: risk line plot + population pie chart
% ==========================
figure('Name','Segments risk & population','Position',[100 100 1400 500]);

% ----- Left: risk line plot -----
nSeg = height(segSummaryPlot);

x = 1:nSeg;
plot(x, segSummaryPlot.risk_mean, '-o', ...
    'LineWidth', 1.5, 'MarkerSize', 6);
hold on;
yline(overallRisk, 'r--', 'LineWidth', 1.2);

% text labels above each point
for i = 1:nSeg
    text(x(i), segSummaryPlot.risk_mean(i) + 3e-4, ...
        sprintf('%.3f', segSummaryPlot.risk_mean(i)), ...
        'HorizontalAlignment','center', ...
        'FontSize', 8);
end

hold off;
segLabels = strcat('S', string(segSummaryPlot.segment));
set(gca, 'XTick', 1:nSeg, 'XTickLabel', segLabels);

xlabel('Segment');
ylabel('P(disease\_risk = 1)');
title('Disease risk rate by segment');
legend({'Segment-wise P(risk=1)', ...
        sprintf('Overall mean = %.3f', overallRisk)}, ...
       'Location', 'southwest');


%% =========================
% 7. 图二：Top 50% 高风险分组的特征热力图 + 3D surf（按 15 个特征）
% ==========================

% 7.1 选出 top 50% 高风险的分组（向上取整）
nSegTop   = ceil(nSeg / 2);
topSegIds = segSummaryPlot.segment(1:nSegTop);
topLabels = strcat('S', string(topSegIds));

% 找到这些 segment 在 segFeatMeans 里的行索引
[~, locTop] = ismember(topSegIds, segFeatMeans.segment);
M_top = M_allSeg(locTop, :);                % nSegTop × 15

% 对这些分组，在每个特征上做列 z-score
M_topZ = zscore(M_top, 0, 1);               % nSegTop × 15

figure('Name','Top-risk segments: feature pattern','Position',[100 100 1400 600]);

% 左：热力图

imagesc(M_topZ);
colormap(parula);
colorbar;
caxis([-1.5 1.5]);
set(gca, 'XTick', 1:numel(allFeatCodes), ...
         'XTickLabel', featDisplayNames, ...
         'XTickLabelRotation', 45, ...
         'TickLabelInterpreter','none');
set(gca, 'YTick', 1:nSegTop, ...
         'YTickLabel', topLabels, ...
         'YDir', 'reverse');
xlabel('Feature');
ylabel('Segment (top 50% by risk)');
title('Top-risk segments: feature pattern (z-score)');


%% =========================
% 8. 图三：3D surf —— “大类 × 分组”的降阶结果
%    大类定义（5 个）：
%    1) Daily routine                : age, steps, sleep, water, calories
%    2) Metabolic status             : BMI, resting HR, cholesterol
%    3) Cardiovascular status        : systolic BP, diastolic BP
%    4) Demographics                 : gender_num
%    5) Any of smoking/alcohol/FH    : 合并指标 any_unhealthy
% ==========================

bigCatNames = { ...
    'Daily routine', ...
    'Metabolic status', ...
    'Cardiovascular status', ...
    'Demographics', ...
    'Any of smoking / alcohol / family history'};

bigCatFeatCodes = { ...
    {'age','daily_steps','sleep_hours','water_intake_l','calories_consumed'}, ...
    {'bmi','resting_hr','cholesterol'}, ...
    {'systolic_bp','diastolic_bp'}, ...
    {'gender_num'}, ...
    {'any_unhealthy'}};   % 这里就是你说的“把 smoker/alcohol/family_history 合并成一个大类”

% 8.1 在所有分组上，对 15 个特征做 z-score（列方向）
M_allSegZ = zscore(M_allSeg, 0, 1);      % nSeg × 15

% 8.2 把 15 维特征压到 5 个“大类”上
nCat     = numel(bigCatNames);
catScore = zeros(nSeg, nCat);            % 行：segment，列：大类
for j = 1:nCat
    codesInCat = bigCatFeatCodes{j};
    idx = find(ismember(allFeatCodes, codesInCat));
    % 这一大类的“得分”：该大类中所有特征 z-score 的均值
    catScore(:, j) = mean(M_allSegZ(:, idx), 2);
end

% 8.3 按“风险从高到低”的顺序重排行
[~, locSegInAll] = ismember(segSummaryPlot.segment, segFeatMeans.segment);
catScore_sorted  = catScore(locSegInAll, :);   % nSeg × nCat，行顺序 = 风险从高到低

% 8.4 画 3D surf：X=大类，Y=segment，Z=平均 z-score
[XS, YS] = meshgrid(1:nCat, 1:nSeg);
Zsurf = catScore_sorted;

figure('Name','Segments × Big-category (dimension-reduced surf)', ...
       'Color','w','Position',[100 100 1200 600]);

surf(XS, YS, Zsurf, ...
    'EdgeColor','none', ...
    'FaceAlpha',0.95);
shading interp;
colormap(parula);
colorbar;
caxis([-2 2]);
grid on;
view(135, 30);

set(gca, 'XTick', 1:nCat, ...
         'XTickLabel', bigCatNames, ...
         'XTickLabelRotation', 0, ...
         'TickLabelInterpreter','none');
set(gca, 'YTick', 1:nSeg, ...
         'YTickLabel', segLabels, ...      % 和图一一样的顺序（高→低风险）
         'YDir','reverse');


ylabel('Segment (high \rightarrow low risk)');
zlabel('Average z-score within category');
title({'Segments \times Category (dimension-reduced 3D surf)', ...
       'Each big-category aggregates multiple original features; Z-axis shows overall level per segment'});

%% =========================
% 9. 小结
% ==========================
fprintf('\n===== 本次降阶+分群模型说明 =====\n');
fprintf('1) 聚类使用的连续变量（z-score）：\n');
disp(contVars(:));

fprintf('2) 第一层降阶：四个二元变量组合成 patternCode（smoker, alcohol, family_history, gender_num）。\n');
fprintf('   在 PCA + KMeans 中，用 patternCode 的 one-hot 与连续变量一起做分群。\n');
fprintf('3) 第二层降阶：把 10 连续 + 4 二元 + 1 合并指标 共 15 个特征，聚合成 5 个“大类”：\n');
disp(bigCatNames(:));
fprintf('   其中最后一个大类就是你要求的 “任意 抽烟/喝酒/家族史” 合并指标。\n');
fprintf('4) 图一：segment 的风险条形图 + 占比–风险散点图（按整体风险从高到低排序）。\n');
fprintf('5) 图二：Top 50%% 高风险分组在 15 个特征上的 z-score 热力图 + 3D surf。\n');
fprintf('6) 图三：分组 × 大类 的 3D surf，直观展示降阶后的主成分结构。\n\n');

end
