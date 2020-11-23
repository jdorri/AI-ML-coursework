# requires Python==3.7.5

from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler

from aif360.datasets import AdultDataset, GermanDataset, CompasDataset
from aif360.metrics import ClassificationMetric, BinaryLabelDatasetMetric
from aif360.explainers import MetricTextExplainer
from aif360.algorithms.preprocessing.reweighing import Reweighing

import numpy as np
import matplotlib.pyplot as plt

import pandas as pd

def load_dataset(name):
    if name == 'Adult':
        ds = AdultDataset()
    elif name == 'German':
        ds = GermanDataset()
    elif name == 'Compas':
        ds = CompasDataset()
    return ds, name

def standardize(train, test):
    scaler = StandardScaler()
    X_train = scaler.fit_transform(train.features)
    y_train = train.labels.ravel()
    X_test = scaler.fit_transform(test.features)
    y_test = test.labels.ravel()
    return X_train, y_train, X_test, y_test

# load dataset
ds, name = load_dataset('Compas') # enter 'Adult', 'Compas' or 'German'

# split into train and test partitions + standardize
ds_tr, ds_te = ds.split([0.7], shuffle=True, seed=1)
X_train, y_train, X_test, _ = standardize(ds_tr, ds_te)

# look into training dataset
print(name + " training dataset shape:")
print(ds_tr.features.shape)

# define privileged and unprivileged groups
priv = [{'sex': 1}] # Male
unpriv = [{'sex': 0}] # Female

# check for existing bias in training data
metric_tr = BinaryLabelDatasetMetric(ds_tr,
    unprivileged_groups=unpriv, privileged_groups=priv)
text_exp_tr = MetricTextExplainer(metric_tr)
print('Checking for bias in the training data...')
print(text_exp_tr.statistical_parity_difference())

def train_and_predict(X_train, y_train, X_test, c, penalty, sample_weight=None):
    learner = LogisticRegression(solver='liblinear', penalty=penalty, random_state=1)
    learner.set_params(C=c)
    learner.fit(X_train, y_train, sample_weight=sample_weight)
    return learner.predict(X_test), learner

def plot_results(lambdas, norm_type, accuracy, mean_diff, avg_odds_diff, equal_opp_diff, exp_name):
    fig, ax1 = plt.subplots()#figsize=(8.0,8.0))

    lns1 = ax1.plot(lambdas, accuracy, color='tab:blue', label='Accuracy')
    ax1.set_xscale('log')
    ax1.set_xlabel('Lambda (log scale, '+norm_type+'-norm)')
    ax1.set_ylabel('Accuracy')

    ax2 = ax1.twinx()
    lns2 = ax2.plot(lambdas, mean_diff, color='tab:orange', label='Mean Difference')
    lns3 = ax2.plot(lambdas, avg_odds_diff, color='tab:red', label='Avg. Odds Difference')
    lns4 = ax2.plot(lambdas, equal_opp_diff, color='tab:green', label='Equal Opp. Difference')
    ax2.invert_yaxis()
    ax2.set_ylabel('Fairness (inverted)')

    leg = lns1 + lns2 + lns3 + lns4
    labs = [l.get_label() for l in leg]
    ax2.legend(leg, labs, loc=0)

    plt.tight_layout()
    #plt.savefig(name+'_figs/'+exp_name+'.pdf')
    plt.show()

# perform regularization-versus-fairness experiment (main loop)
C = [0.000001, 0.00001, 0.0001, 0.001, 0.01, 0.1, 1, 10, 100]
#C = np.logspace(np.log10(0.000001), np.log10(100), 25)
norm_type = 'l2'
accuracy = []
mean_diff = []
average_odds_diff = []
equal_opp_diff = []
for c in C:
    predictions, _ = train_and_predict(X_train, y_train, X_test, c, norm_type)

    ds_te_pred = ds_te.copy()
    ds_te_pred.labels = predictions

    metric_te = ClassificationMetric(ds_te, ds_te_pred,
                    unprivileged_groups=unpriv, privileged_groups=priv)

    BACC = 0.5*(metric_te.true_positive_rate()\
        +metric_te.true_negative_rate())
    metric_1 = metric_te.statistical_parity_difference()
    metric_2 = metric_te.average_odds_difference()
    metric_3 = metric_te.equal_opportunity_difference()

    accuracy.append(BACC)
    mean_diff.append(metric_1)
    average_odds_diff.append(metric_2)
    equal_opp_diff.append(metric_3)

# save plots
plot_results(C, norm_type, accuracy, mean_diff, average_odds_diff, \
    equal_opp_diff, name+'_all_metrics_'+norm_type)

def results_table(C, accuracy, mean_diff, avg_odds_diff, equal_opp_diff):
    results = pd.DataFrame()
    results['c'] = C
    results['bACC'] = accuracy
    results['mean_diff'] = mean_diff
    results['avg_odds_diff'] = avg_odds_diff
    results['equal_opp_diff'] = equal_opp_diff
    return results

def arg_best(results):
    metrics = ['bACC', 'mean_diff', 'avg_odds_diff', 'equal_opp_diff']
    for metric in metrics:
        if metric == 'bACC':
            best = round(results[metric].max(),3)
            c = results['c'][results[metric].idxmax()]
        else:
            best = round(results[metric].abs().min(),3)
            c = results['c'][results[metric].abs().idxmin()]
        print('Best '+metric+': '+str(best)+', corresponding reg. strength (c): '+str(c))

# print results table and 'arg-best' analysis
results = results_table(C, accuracy, mean_diff, average_odds_diff, equal_opp_diff)
print(results)
arg_best(results)

def reweight(ds, priv, unpriv):
    rw = Reweighing(unprivileged_groups=unpriv,
                    privileged_groups=priv)
    ds_transf = rw.fit_transform(ds)
    return ds_transf

# apply re-weighting bias mitigation algorithm
ds_tr_transf = reweight(ds_tr, priv, unpriv)

# re-check for bias
metric_tr_transf = BinaryLabelDatasetMetric(ds_tr_transf,
    unprivileged_groups=unpriv, privileged_groups=priv)
text_exp_tr_transf = MetricTextExplainer(metric_tr_transf)
print('----'*10)
print('Checking for bias in the pre-processed training data...')
print(text_exp_tr_transf.statistical_parity_difference()) # should be extremely close to 0

# repeat experiment loop with fair training dataset
accuracy_transf = []
mean_diff_transf = []
average_odds_diff_transf = []
equal_opp_diff_transf = []
for c in C:
    predictions_transf, _ = train_and_predict(X_train, y_train, X_test, c, \
        norm_type, sample_weight=ds_tr_transf.instance_weights) # re-train classifier

    ds_te_pred_transf = ds_te.copy()
    ds_te_pred_transf.labels = predictions_transf

    metric_te_transf = ClassificationMetric(ds_te, ds_te_pred_transf,
                    unprivileged_groups=unpriv, privileged_groups=priv)

    BACC_transf = 0.5*(metric_te_transf.true_positive_rate()\
        +metric_te_transf.true_negative_rate())
    metric_1_transf = metric_te_transf.statistical_parity_difference()
    metric_2_transf = metric_te_transf.average_odds_difference()
    metric_3_transf = metric_te_transf.equal_opportunity_difference()

    accuracy_transf.append(BACC_transf)
    mean_diff_transf.append(metric_1_transf)
    average_odds_diff_transf.append(metric_2_transf)
    equal_opp_diff_transf.append(metric_3_transf)

# re-generate plot post-bias mitigation
plot_results(C, norm_type, accuracy_transf, mean_diff_transf, average_odds_diff_transf, \
    equal_opp_diff_transf, name+'_all_metrics_transf_'+norm_type)

# re-display results/arg-best analysis
results_transf = results_table(C, accuracy_transf, mean_diff_transf, \
    average_odds_diff_transf, equal_opp_diff_transf)
print(results_transf)
arg_best(results_transf)


print('----'*10)
value = input('Enter fairness metric to optimise in addition to accuracy \
(1 = mean difference, 2 = average odds difference, 3 = equal opportunity difference): ')
print('Performing model selection...')

# model selection
accuracy_final = []
fairness_final = []
#optimal_c = []
for i in range(5):
    ds_tr, ds_te = ds.split([0.7], shuffle=True)

    ds_tr, ds_val = ds_tr.split([0.7], shuffle=True)

    X_train, y_train, X_val, _ = standardize(ds_tr, ds_val)
    ds_tr_transf = reweight(ds_tr, priv, unpriv)

    scores = []
    models = []
    for c in C:
        predictions, model = train_and_predict(X_train, y_train, X_val, c, \
            norm_type, sample_weight=ds_tr_transf.instance_weights)

        ds_val_pred = ds_val.copy()
        ds_val_pred.labels = predictions

        metric_val = ClassificationMetric(ds_val, ds_val_pred,
                        unprivileged_groups=unpriv, privileged_groups=priv)

        BACC = 0.5*(metric_val.true_positive_rate()\
            +metric_val.true_negative_rate())

        if value == '1':
            metric = metric_val.statistical_parity_difference()
            label = 'Mean Difference'
        elif value == '2':
            metric = metric_val.average_odds_difference()
            label = 'Avg. Odds Difference'
        elif value == '3':
            metric = metric_val.equal_opportunity_difference()
            label = 'Equal Opp. Difference'

        if metric == 0:
            score = float('inf')
        else:
            score = 0.5*BACC + 0.5*1/abs(metric)

        scores.append(score)
        models.append(model)

    idx = scores.index(max(scores))
    best_model = models[idx]

    #optimal_c.append(best_model.get_params()['C'])

    scaler = StandardScaler()
    X_test = scaler.fit_transform(ds_te.features)
    predictions = best_model.predict(X_test)

    ds_te_pred = ds_te.copy()
    ds_te_pred.labels = predictions

    metric_te = ClassificationMetric(ds_te, ds_te_pred,
                    unprivileged_groups=unpriv, privileged_groups=priv)

    BACC = 0.5*(metric_te.true_positive_rate()\
        +metric_te.true_negative_rate())
    if value == '1':
        metric = metric_te.statistical_parity_difference()
    elif value == '2':
        metric = metric_te.average_odds_difference()
    elif value == '3':
        metric = metric_te.equal_opportunity_difference()

    accuracy_final.append(BACC)
    fairness_final.append(metric)

def mean_and_stdv(metric):
    metric = np.array(metric)
    avg = metric.mean()
    stdv = metric.std()
    return avg, stdv

# compute statistics
acc_avg, acc_stdv = mean_and_stdv(accuracy_final)
fairness_avg, fairness_stdv = mean_and_stdv(fairness_final)

# plot results
metrics = ['Accuracy', label]
x_pos = [0,0.4]
avg = [acc_avg, fairness_avg]
error = [acc_stdv, fairness_stdv]

fig, ax = plt.subplots(figsize=(6,3))
ax.barh(x_pos, avg, xerr=error, height=0.3)
ax.set_xlabel('Average')
ax.set_yticks(x_pos)
ax.set_yticklabels(metrics)
ax.get_children()[2].set_color('tab:orange')
ax.grid(True)
plt.tight_layout()
#plt.savefig(name+'_figs/'+name+'_model_selection_'+value+'.pdf')
plt.show()

#print(optimal_c)
print('Model selection averages:')
print("bACC: %0.3f (+/- %0.3f)" % (acc_avg, acc_stdv))
print(label + ": %0.3f (+/- %0.3f)" % (fairness_avg, fairness_stdv))
