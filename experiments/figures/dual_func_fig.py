# Generates a fig showing the f and f_A gaps.
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd
import functools

def sampleFunction(f, start, end, numSamples):
    xs = []
    ys = []
    for i in range(numSamples):
        x = (end-start) * (i * 1.0 / numSamples) + start
        xs.append(x)
        ys.append(f(x))
    return xs, ys

def fact(n):
    if n == 0:
        return 1
    return functools.reduce(lambda a, b: a*b, range(1, n+1))

@functools.lru_cache(maxsize=None)
def binom(n, k):
    if n == 0 or k == 0 or n == k:
        return 1
    return binom(n-1, k) + binom(n-1, k-1)

def generate_bates_pdf(n, offset_x, offset_y, offset_mag):
    def f(x):
        x = x - offset_x
        coeff = n / (2.0*fact(n-1))

        summation = 0
        lead = 1.0
        for k in range(n+1):
            v1 = binom(n, k)
            v2 = (n*x-k) ** (n-1)
            nx = n * x
            s = 0
            if nx < k:
                s = -1
            else:
                s = 1
            cur = lead * v1 * v2 * s
            summation += cur
            lead *= -1
        return coeff * summation * offset_mag + offset_y

    return f

plt.rcParams["pdf.fonttype"] = 42
plt.rcParams["ps.fonttype"] = 42
#sns.set(style="white", palette="muted", color_codes=True)

start_x = 0.0
end_x = 1.0
num_samples = 10000

def gen_obj():
    f1 = generate_bates_pdf(20, 0.2, 0.3, 1.5)
    f2 = generate_bates_pdf(10, -0.3, 0.3, 0.7)

    def f(x):
        return f1(x) + f2(x)
    return f

def gen_fa():
    f1 = generate_bates_pdf(20, 0.2, 0.3, 1.4)
    f2 = generate_bates_pdf(10, -0.3, 0.0, 0.1)

    def f(x):
        return f1(x) + f2(x)
    return f

def gen_gap():
    f1 = gen_obj()
    f2 = gen_fa()

    def f(x):
        return f1(x) - f2(x)
    return f

def sim_qd(alpha, min_f, lam, n, num_iters, lo_bound, hi_bound, f):
    archive = np.full(n, min_f)

    for i in range(num_iters):
        
        pos = np.random.uniform(lo_bound, hi_bound, lam)
        gap = hi_bound - lo_bound

        f_vals = [f(v) for v in pos]
        cells = n * (pos - lo_bound) / gap

        for obj_v, cell in zip(f_vals, cells):

            obj_v = obj_v * np.random.uniform(0.75, 1.0)
            cell_id = int(cell)
            archive[cell_id] = (1-alpha) * archive[cell_id] + alpha * obj_v
    
    # Archives are choppy, so smooth the f_A function.
    for j in range(5):
        width = 50
        smooth_archive = np.zeros(n)
        for i in range(len(archive)):
            x = max(0, i - width)
            y = min(n-1, i + width)
            v = np.mean(archive[x:y])
            smooth_archive[i] = v
        archive = smooth_archive

    return None, archive


def make_dual_func():
    filepath = 'dual.pdf'

    data = {}
    column_order = []

    # The objective function f
    xs, ys = sampleFunction(gen_obj(), start_x, end_x, num_samples)
    label = '$f$'
    column_order.append(label)
    data[label] = ys

    f_a = gen_fa()
    alphas = [1.0, 0.01, 0.0]
    for alpha in alphas:
        print('start a =', alpha)
        _, ys = sim_qd(alpha, 0.0, 36, num_samples, 10_000, start_x, end_x, f_a)
        print('end a =', alpha)

        label = '$f_A (\\alpha={})$'.format(alpha)
        column_order.append(label)
        data[label] = ys

    xs, _ = sampleFunction(lambda x: 1, start_x, end_x, num_samples)
    xlabels = {x:xs[x] for x in range(num_samples)}

    data = pd.DataFrame(data)
    data = data.rename(index=xlabels)
    data = data[column_order]

    with sns.axes_style("white"):
        sns.set_style("white",{'font.family':'serif','font.serif':'Palatino'})

        # Draw plot 1 with uniform lines accross a value
        palette_list = ['Blue'] + ['Red'] * (len(column_order)-1)
        p = sns.color_palette(palette=palette_list)
        p1 = sns.lineplot(
                data=data, 
                palette=p, 
                linewidth=2.5, 
                legend=True, 
                #dashes=False,
        )

        p1.axis('off')

        plt.ylim(0, 10.0)
        plt.legend(prop={'size':25})
        plt.tight_layout()
        #plt.show()
        p1.figure.savefig(filepath, bbox_inches='tight')

        plt.close('all')

def make_gap_func(alpha):
    filepath = 'gap_{}.pdf'.format(alpha)

    data = {}
    column_order = []

    # The gap function
    _, ys_f = sampleFunction(gen_obj(), start_x, end_x, num_samples)
    #column_order.append(label)
    #data[label] = ys

    f_a = gen_fa()
    print('start a =', alpha)
    _, ys = sim_qd(alpha, 0.0, 36, num_samples, 10_000, start_x, end_x, f_a)
    print('end a =', alpha)

    ys = ys_f - ys

    label = 'gap'
    column_order.append(label)
    data[label] = ys

    xs, _ = sampleFunction(lambda x: 1, start_x, end_x, num_samples)
    xlabels = {x:xs[x] for x in range(num_samples)}

    data = pd.DataFrame(data)
    data = data.rename(index=xlabels)
    data = data[column_order]

    with sns.axes_style("white"):
        sns.set_style("white",{'font.family':'serif','font.serif':'Palatino'})

        # Draw plot 1 with uniform lines accross a value
        p = sns.color_palette(palette=['Black'])
        p1 = sns.lineplot(data=data, palette=p, linewidth=2.5, legend=False, dashes=False)

        p1.axis('off')

        plt.ylim(0, 10.0)
        #plt.legend(prop={'size':20})
        plt.tight_layout()
        #plt.show()
        p1.figure.savefig(filepath, bbox_inches='tight')

        plt.close('all')

make_dual_func()
make_gap_func(1.0)
make_gap_func(0.01)
make_gap_func(0.0)
