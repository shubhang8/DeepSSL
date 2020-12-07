# from comet_ml import Experiment

# # Create an experiment with your api key:
# experiment = Experiment(
#     api_key="pTKIV6STi3dM8kJmovb0xt0Ed",
#     project_name="deepssl",
#     workspace="suchzheng2",
# )


from sklearn.datasets import make_classification
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_curve
from sklearn.metrics import roc_auc_score
from matplotlib import pyplot

from sklearn.metrics import precision_recall_curve
from sklearn.metrics import f1_score
from sklearn.metrics import auc



def AUROC(experiment, targets, predict, img_path, curve_name):
	# targets = [1,1,1,1,1,1,0,0,0,0,0,0,0,0,1,1,1,1,1]
	# predict = [0.9,0.8,0.7,0.2,0.2,1.0,0.9,0.8,0.3,0.3,0.4,0.9,0.1,0.1,0.9,0.7,0.9,0.3,0.9]
	auc = roc_auc_score(targets, predict)
	
	fpr, tpr, _ = roc_curve(targets, predict)
	experiment.log_metric("AUROC",auc)
	experiment.log_curve(curve_name,fpr,tpr)

	# plot the roc curve for the model

	pyplot.plot(fpr, tpr, linestyle='--', label= curve_name)
	# axis labels
	pyplot.title("AUROC")
	pyplot.xlabel('False Positive Rate')
	pyplot.ylabel('True Positive Rate')
	# show the legend
	pyplot.legend()
	pyplot.savefig(img_path)

	experiment.log_image(img_path)

def AUPRC(experiment, targets, predict, img_path, curve_name):
	precision, recall, _ = precision_recall_curve(targets, predict)
	auc_val = auc(recall, precision)
	experiment.log_metric("AUPRC",auc_val)
	experiment.log_curve(curve_name,recall, precision)

	pyplot.plot(recall, precision, marker='.', label=curve_name)
	# axis labels
	pyplot.title("AUPRC")
	pyplot.xlabel('Recall')
	pyplot.ylabel('Precision')
	# show the legend
	pyplot.legend()
	# show the plot
	# pyplot.show()
	

	pyplot.savefig(img_path)
	experiment.log_image(img_path)


def F1(experiment, targets, predict):
	f_1 = f1_score(targets,predict)
	experiment.log_metric("F1",f_1)












