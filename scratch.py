from models_hmm import RampModelHMM
import matplotlib.pyplot as plt
import w3_2
fn = '0.25GGML_D60_T10'

# w3_2.plot_confusion_matrix('./results/' + fn + '.csv' , r'Conf. Matrix (Gaussian gen/prior), $\sigma_{frac}=\frac{1}{4}$',
#                            save_name='./plots/task_3_2_3_' + fn + '_confmat',
#                            fig_size_factor=0.8)

w3_2.plot_heatmap('./results/UU_D240_T3.csv', title='Bayes Factor Heatmap', vertical=True)
# w3_2.plot_heatmap('./results/0.5GU_D240_T3.csv')
# w3_2.plot_heatmap('./results/0.25GU_D240_T3.csv')
# w3_2.plot_heatmap('./results/0.125GU_D240_T3.csv')