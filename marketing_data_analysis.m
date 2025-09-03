%% MATLAB Data Analysis Mini-Project
% Goal: Clean, explore, visualize, detect anomalies, and model a simple dataset.
% Theme: Daily website Visits explained by AdSpend (with noise, missing values, and outliers).

clear; close all; clc;

%% 1) Create a realistic synthetic dataset and save to CSV (simulates a real file)
startDate = datetime(2024,1,1);
endDate   = datetime(2024,3,31);
Date = (startDate:endDate)';                 % daily dates
n = numel(Date);

rng(42);                                     % reproducible
AdSpend = 200 + 20*sin((1:n)'/6) + 30*randn(n,1);  % cyclical + noise
AdSpend = max(0, AdSpend);                   % no negatives

% True relationship: Visits = 1200 + 3.8*AdSpend + weekday lift + noise
isWeekend = ismember(weekday(Date), [1 7]);  % Sunday=1, Saturday=7
weekendLift = -150 * double(isWeekend);      % fewer visits on weekends
Visits = 1200 + 3.8*AdSpend + weekendLift + 120*randn(n,1);

% Inject some missing values & outliers to make it real
missIdx = randsample(n, 8);
outIdx  = randsample(setdiff((1:n)', missIdx), 5);
Visits(missIdx)  = NaN;
Visits(outIdx)   = Visits(outIdx) + 1500 .* sign(randn(numel(outIdx),1)); % spikes/dips

T = table(Date, AdSpend, Visits);
writetable(T, 'marketing_data.csv');         % pretend this is provided to you

%% 2) Load the data (as you would in a real project)
D = readtable('marketing_data.csv');

%% 3) Basic inspect & sort
D = sortrows(D, 'Date');
disp('Head:'); disp(D(1:5,:))
fprintf('Rows: %d  |  Missing Visits: %d\n', height(D), sum(ismissing(D.Visits)));

%% 4) Clean: handle missing data, enforce types
% Fill missing Visits via linear interpolation on the time axis
% (alternatives: 'movmedian', 'nearest', or remove rows)
D.Visits = fillmissing(D.Visits, 'linear', 'EndValues','nearest');

% Remove truly impossible AdSpend values (defensive)
D.AdSpend(D.AdSpend < 0) = 0;

%% 5) Feature engineering
D.DayOfWeek = categorical(day(D.Date,'name'));            % Monday, Tuesday, ...
D.IsWeekend = ismember(weekday(D.Date), [1 7]);            % boolean
D.Visits7dMA = movmean(D.Visits, [6 0]);                   % trailing 7-day moving average
D.Spend7dMA  = movmean(D.AdSpend, [6 0]);

%% 6) Exploratory plots
figure('Name','Time Series: Visits & 7-day MA');
plot(D.Date, D.Visits, '.', 'DisplayName','Visits'); hold on;
plot(D.Date, D.Visits7dMA, 'LineWidth', 1.5, 'DisplayName','Visits 7-day MA');
xlabel('Date'); ylabel('Visits'); title('Visits Over Time'); legend boxoff;

figure('Name','Scatter: AdSpend vs Visits');
scatter(D.AdSpend, D.Visits, 12, 'filled'); grid on;
xlabel('Ad Spend'); ylabel('Visits'); title('Ad Spend vs Visits');

figure('Name','Boxplot: Visits by DayOfWeek');
boxplot(D.Visits, D.DayOfWeek);
ylabel('Visits'); title('Visits Distribution by Day of Week');

%% 7) Outlier detection (median absolute deviation)
isOut = isoutlier(D.Visits, 'median');
D.IsOutlier = isOut;

figure('Name','Visits with Outliers Highlighted');
plot(D.Date, D.Visits, '-', 'DisplayName','Visits'); hold on;
plot(D.Date(isOut), D.Visits(isOut), 'o', 'MarkerSize',6, 'DisplayName','Outliers');
xlabel('Date'); ylabel('Visits'); title('Outlier Detection'); legend boxoff; grid on;

fprintf('Detected %d outliers in Visits.\n', sum(isOut));

%% 8) Simple regression (base MATLAB, no toolboxes)
% Model: Visits ~ 1 + AdSpend + IsWeekend
y = D.Visits;
X = [ones(height(D),1), D.AdSpend, double(D.IsWeekend)];
beta = X \ y;                                  % OLS via backslash
yhat = X * beta;

% Goodness of fit
SSR = sum((yhat - mean(y)).^2);
SSE = sum((y - yhat).^2);
R2  = SSR / (SSR + SSE);

% Error metrics
MAE = mean(abs(y - yhat));
RMSE = sqrt(mean((y - yhat).^2));

fprintf('OLS Coefficients [Intercept, AdSpend, IsWeekend]:\n'); disp(beta.');
fprintf('R^2 = %.3f   |   MAE = %.1f   |   RMSE = %.1f\n', R2, MAE, RMSE);

% Plot fit vs actual
figure('Name','Model Fit');
scatter(D.Date, y, 10, 'filled', 'DisplayName','Actual'); hold on;
plot(D.Date, yhat, 'LineWidth',1.5, 'DisplayName','Predicted');
xlabel('Date'); ylabel('Visits'); title('Actual vs Predicted Visits'); legend boxoff; grid on;

%% 9) (Optional) Robust fit to reduce outlier influence (still base MATLAB)
% Use weighted least squares with Huber-like weights
resid = y - yhat;
s = 1.4826 * median(abs(resid - median(resid)));      % robust scale estimate
c = 1.345 * s;                                        % Huber constant
w = min(1, c ./ max(c, abs(resid)));                  % simple Huber weights in [0,1]
W = diag(w);
beta_w = (X' * W * X) \ (X' * W * y);
yhat_w = X * beta_w;

SSR_w = sum((yhat_w - mean(y)).^2);
SSE_w = sum((y - yhat_w).^2);
R2_w  = SSR_w / (SSR_w + SSE_w);
MAE_w = mean(abs(y - yhat_w));
RMSE_w= sqrt(mean((y - yhat_w).^2));

fprintf('Weighted OLS (robust) R^2=%.3f | MAE=%.1f | RMSE=%.1f\n', R2_w, MAE_w, RMSE_w);

figure('Name','Predictions: OLS vs Robust');
plot(D.Date, y, '-', 'DisplayName','Actual'); hold on;
plot(D.Date, yhat, 'LineWidth',1.2, 'DisplayName','OLS');
plot(D.Date, yhat_w, 'LineWidth',1.2, 'DisplayName','Robust');
xlabel('Date'); ylabel('Visits'); title('OLS vs Robust Regression'); legend boxoff; grid on;

%% 10) Save cleaned/enriched data
writetable(D, 'marketing_data_cleaned.csv');
disp('Saved cleaned dataset to marketing_data_cleaned.csv');
