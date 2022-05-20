import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

have_legend = False

plt.rcParams["pdf.fonttype"] = 42
plt.rcParams["ps.fonttype"] = 42

data = pd.read_csv('summary.csv')

y_label = "QD-Score"

plt.figure(figsize = (12,12))

# Color mapping for algorithms
palette ={
    "CMA-MAE": "C1",
    "CMA-ME": "C2",
    "MAP-Elites": "C3",
    "MAP-Elites (line)": "C4",
}

sns.set(font_scale=4)
with sns.axes_style("white"):
    sns.set_style("white",{'font.family':'serif','font.serif':'Palatino'})
    sns.set_palette("colorblind")
    
    # Plot the responses for different events and regions
    sns_plot = sns.lineplot(x="Iteration", 
                            y=y_label, 
                            linewidth=3.0,
                            hue="Algorithm",
                            data=data, 
                            legend=have_legend,
                            palette=palette,
                           )
    plt.xticks([0, 5000, 10000])
    plt.yticks([0, 80])
    plt.xlabel("Iterations")
    plt.ylabel(y_label)
    #sns_plot.yaxis.set_label_coords(0.5,0.0)
    #sns_plot.xaxis.set_label_coords(0.5,-0.15)


    if have_legend:
        legend = plt.legend(loc="lower right", frameon=False, prop={'size': 45})
        #legend.set_bbox_to_anchor((0.60, 0.40))
        for line in legend.get_lines():
            line.set_linewidth(4.0)
        
        frame = legend.get_frame()
        frame.set_facecolor('white')
    
    sns_plot.figure.savefig("qd_score.pdf", bbox_inches='tight', dpi=100)
