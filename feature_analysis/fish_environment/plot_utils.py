from feature_analysis.fish_environment.feature_utils import get_y, save_fig_fixname
from feature_analysis.fish_environment.fish_processed_data import pixels_mm_converters

import os
import numpy as np
from numpy import format_float_scientific
import matplotlib.pyplot as plt
from matplotlib_scalebar.scalebar import ScaleBar
import matplotlib.colors as colors
import matplotlib.cm as cm
import seaborn as sns
import copy
import logging
from tqdm import tqdm
import pandas as pd
import traceback
import warnings
from scipy import stats
import scipy

FIG_SIZE = (8, 6)  # 6 height


def heatmap_plot(f, np_map, name, title, plot_dir, max_val=None, ax: plt.Axes = [], xspace=50, yspace=50, n_mm=2,
                 with_cbar=True, visible_ticks=True,
                 xy_tick_values=[-8.0, -4.0, 0.0, 4.0, 8.0], is_abort=False):
    _, one_pixel_in_mm = pixels_mm_converters()
    if max_val is None:
        max_val = np.max(np_map)

    palette = colors.LinearSegmentedColormap.from_list('rg', ["w", "r"], N=256)  # from white to red

    c = ax.pcolormesh(np.flipud(np_map), cmap=palette, vmax=max_val, vmin=0)  # 0.000001
    if with_cbar:
        f.colorbar(c, ax=ax.invert_yaxis())
    ax.set_title(title)
    ax.add_artist(ScaleBar(one_pixel_in_mm, "mm", location='lower left', color='k', box_alpha=0,
                           fixed_value=n_mm))  # font_properties={"size": 16},
    if not is_abort:
        ax.annotate(".", (0.5, 0.5), xycoords='axes fraction', ha='center', color='b', size=44)
    else:
        ax.annotate(".", (np_map.shape[0] // 2, np_map.shape[1] // 2), ha='center', color='b', size=44)

    if not visible_ticks:
        if xy_tick_values != []:  # opposite of tick labels conversion
            rx = (np.array(xy_tick_values) / one_pixel_in_mm) + np_map.shape[1] // 2
            ry = (np.array(xy_tick_values) / one_pixel_in_mm) + np_map.shape[0] // 2
        else:
            rx = np.arange(0, np_map.shape[1] + 1, xspace)
            ry = np.arange(0, np_map.shape[0] + 1, yspace)
        ax.get_xaxis().set_ticks(rx)
        ax.get_yaxis().set_ticks(ry)
        ax.get_xaxis().set_ticklabels(["{0:.1f}".format(a * one_pixel_in_mm) for a in rx - np_map.shape[1] // 2])
        ax.get_yaxis().set_ticklabels(["{0:.1f}".format(a * one_pixel_in_mm) for a in ry - np_map.shape[0] // 2])
    else:
        ax.get_xaxis().set_visible(False)
        ax.get_yaxis().set_visible(False)
        ax.get_xaxis().set_ticks([])
        ax.get_yaxis().set_ticks([])
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.set_xlim(np_map.shape[1] // 4 + np_map.shape[1] // 8, np_map.shape[1] // 8 + np_map.shape[1] // 2)
    if is_abort:
        ax.set_ylim(np_map.shape[0] // 4 + np_map.shape[0] // 8, np_map.shape[0] - np_map.shape[0] // 4)
    else:
        ax.set_ylim(np_map.shape[0] // 4 + np_map.shape[0] // 8, np_map.shape[1] // 8 + np_map.shape[1] // 2)
    # ax.get_xaxis().set_visible(False)
    # ax.get_yaxis().set_visible(False)


def barplot_annotate_brackets(num1, num2, data, center, height, yerr=None, dh=.05, barh=.05, fs=None, maxasterix=None,
                              ax=None, color='k', index_append_up=1):
    """
    Annotate barplot with p-values.

    :param num1: number of left bar to put bracket over
    :param num2: number of right bar to put bracket over
    :param data: string to write or number for generating asterixes
    :param center: centers of all bars (like plt.bar() input)
    :param height: heights of all bars (like plt.bar() input)
    :param yerr: yerrs of all bars (like plt.bar() input)
    :param dh: height offset over bar / bar + yerr in axes coordinates (0 to 1)
    :param barh: bar height in axes coordinates (0 to 1)
    :param fs: font size
    :param maxasterix: maximum number of asterixes to write (for very small p-values)
    """

    if type(data) is str:
        text = data
    else:
        # * is p < 0.05, ** is p < 0.005, *** is p < 0.0005, etc.
        text = ''
        p = .05

        while data < p:
            text += '*'
            p /= 10.
            if maxasterix and len(text) == maxasterix:
                break
        if len(text) == 0:
            text = 'n. s.'

    lx, ly = center[num1], height[num1]
    rx, ry = center[num2], height[num2]

    if yerr:
        ly += yerr[num1]
        ry += yerr[num2]

    if ax is None:
        ax = plt.gca()
    ax_y0, ax_y1 = ax.get_ylim()
    dh *= (ax_y1 - ax_y0)
    barh *= (ax_y1 - ax_y0)

    y = max(ly, ry) + dh * index_append_up

    barx = [lx, lx, rx, rx]
    bary = [y, y + barh, y + barh, y]
    mid = ((lx + rx) / 2, y + barh)

    ax.plot(barx, bary, c=color, label='_nolegend_')  # todo add legend?

    kwargs = dict(ha='center', va='bottom', color=color)
    if fs is not None:
        kwargs['fontsize'] = fs

    ax.text(*mid, text, **kwargs, label='_nolegend_')

    _, ax_y11 = ax.get_ylim()
    ax.set_ylim([ax_y0, max(ax_y11, y + barh + dh * 10)])  # adjust to not have annotation outside


def plot_scatter_bar(add_name, age_names, outcome_names, counters, key, y_label, curr_title, filename, dpi,
                     is_combine_age, plot_dir, with_lines=True, with_p_values=True, fig_size=FIG_SIZE, dh=.01,
                     is_significant_only=False, split_per_outcome=True, max_value=None, is_subplot=False,
                     is_cut_bottom=True, with_numbers=True,
                     title_rename_map={}, xlabel_rename_map={}, with_title=True, f=lambda x: x,
                     color_map={}, color=None, with_sup_title=False, p_val_color=None, colormap=cm.jet):
    image_paths = []
    statistics = {}
    space = 0

    for i, age in enumerate(age_names):
        outcomes_dict = counters[age]
        for j, outcome in enumerate(outcome_names):
            if not split_per_outcome and not is_subplot:  # is_combine_age:  # plot all on the same graph
                index = i * len(outcome_names) + j + space
            else:  # diff plots per outer key
                index = j if with_lines else i
            values_dict = outcomes_dict.get(outcome, {})
            if values_dict == {}:
                per_fish = []
            else:
                if key != "":
                    per_fish = [_ for _ in values_dict[key]]
                else:
                    per_fish = values_dict
            if not with_lines:
                if outcome not in statistics.keys():
                    statistics[outcome] = {}
                statistics[outcome][age] = {'x': index, 'y': copy.deepcopy(per_fish)}
            else:
                if age not in statistics.keys():
                    statistics[age] = {}
                statistics[age][outcome] = {'x': index, 'y': copy.deepcopy(per_fish)}
        space += 1

    if is_subplot:
        fig, axes = plt.subplots(1, len(statistics.keys()), figsize=fig_size, dpi=dpi, sharex=True, sharey=True)
        if not isinstance(axes, (list, np.ndarray)):
            axes = [axes]
        for a in axes:
            a.spines["top"].set_visible(False)
            a.spines["right"].set_visible(False)
    else:
        fig, ax = plt.subplots(1, 1, figsize=fig_size, dpi=dpi)
    normalize = colors.Normalize(vmin=0, vmax=len(outcome_names) - 1)
    added_labels = []  # make sure legend is unique
    for outer_ind, (outer_key, dict_value) in enumerate(statistics.items()):
        numbers = []
        if split_per_outcome:  # not is_combine_age  # plot all on the same graph
            fig, ax = plt.subplots(1, 1, figsize=fig_size, dpi=dpi)
            added_labels = []  # make sure legend is unique
            ax.spines["top"].set_visible(False)
            ax.spines["right"].set_visible(False)
        elif is_subplot:
            ax = axes[outer_ind]
        for j, (inner_key, data_dict) in enumerate(dict_value.items()):
            x, y = data_dict['x'], get_y(data_dict, f=f)
            numbers.append(len([_ for _ in y if not np.isnan(_)]))
            # y = y[y <= 5]
            outcome = inner_key if with_lines else outer_key
            outcome_ind = j if with_lines else outer_ind
            label = outcome if outcome not in added_labels else '_nolegend_'
            if label in title_rename_map.keys():
                label = title_rename_map[label]
            n = np.random.normal(0, 0.12, len(y))
            c = color if color is not None else colormap(inner_key, outer_key)
            c = color_map[outcome] if outcome in color_map.keys() else c
            x_range = np.ones(len(y)) * x + n
            ax.scatter(x_range, y, label=label, color=c, alpha=0.5)
            added_labels.append(label)
            if not np.isnan(y).all():  # len(y) > 0 and np.mean(y) > 0:
                yerr = scipy.stats.sem(y, nan_policy="omit")
                if isinstance(yerr, np.ma.core.MaskedConstant):
                    continue
                if False:
                    ax.plot([min(x_range), max(x_range)], [np.mean(y), np.mean(y)], label='_nolegend_', color='k')
                    ax.errorbar(x, np.nanmean(y), label='_nolegend_', linestyle='None',
                                yerr=yerr, ecolor='black', capsize=20, elinewidth=1)
                else:
                    ax.bar(x, np.nanmean(y), label='_nolegend_', fill=False, edgecolor="black",
                           yerr=yerr, align='center', ecolor='black', capsize=10)

        if with_p_values:
            ages_list = list(dict_value.keys())
            paired_ages = [(ages_list[p1], ages_list[p2]) for p1 in range(len(ages_list))
                           for p2 in range(p1 + 1, len(ages_list))]
            normalize2 = colors.Normalize(vmin=0, vmax=len(paired_ages) - 1)
            heights = [np.nanmax(
                np.array(get_y(dict_value[ages_list[p1]], f=f))) + dh * 5
                       for p1 in range(len(ages_list)) if len(get_y(dict_value[ages_list[p1]], f=f)) > 0]
            if len(heights) > 0:
                heights = [np.nanmax(heights), np.nanmax(heights)]
            for inner_pair_ind, (age_1, age_2) in enumerate(paired_ages):
                y1, y2 = get_y(dict_value[age_1], f=f), get_y(dict_value[age_2], f=f)
                if len(y1) > 0 and len(y2) > 0:
                    outcome_ind = inner_pair_ind if with_lines else outer_ind
                    p_val = stats.ttest_ind(y1, y2, nan_policy='omit')[1]
                    centers = [dict_value[age_1]['x'], dict_value[age_2]['x']]
                    if not is_significant_only or (is_significant_only and p_val <= .05):
                        c = p_val_color if p_val_color is not None else colormap(inner_key, outer_key)
                        barplot_annotate_brackets(0, 1, format_pval(p_val), centers, heights, dh=dh, ax=ax,
                                                  color=c, index_append_up=inner_pair_ind * 10)

        # if with_lines:  # add lines between types
        #     outcome_list = list(dict_value.keys())
        #     paired_outcomes = [(outcome_list[p1], outcome_list[p2])
        #                        for p1 in range(len(outcome_list)) for p2 in range(p1 + 1, len(outcome_list))]
        #     normalize2 = colors.Normalize(vmin=0, vmax=len(paired_outcomes) - 1)
        #     for pair_ind, (outcome_1, outcome_2) in enumerate(paired_outcomes):
        #         x1, x2 = dict_value[outcome_1]['x'], dict_value[outcome_2]['x']
        #         y1, y2 = dict_value[outcome_1]['y'], dict_value[outcome_2]['y']
        #         if len(y1) == len(y2) and len(y1) > 0:
        #             lines = np.c_[np.ones_like(y1) * x1, y1, np.ones_like(y2) * x2, y2].reshape(len(y2), 2, 2)
        #             ax.add_collection(LineCollection(lines, colors=colormap(normalize2(pair_ind))))
        #         else:
        #             print("Error. diff number of values for {0}: {1}, {2}".format(outer_key, outcome_1, outcome_2))

        if split_per_outcome or is_subplot:  # not is_combine_age:  # plot all on the same graph
            t_str = outer_key
            if t_str in title_rename_map.keys():
                t_str = title_rename_map[t_str]
            ages_list = list(dict_value.keys())
            ages_list = [xlabel_rename_map.get(k, k) for k in ages_list]
            if with_numbers and len(numbers) == len(ages_list) and sum(numbers) > 0:
                ages_list = [ages_list[i] + " \n(n={0})".format(numbers[i]) for i in range(len(ages_list))]
            ax.set_xticks(range(len(ages_list)))
            ax.set_xticklabels(ages_list)
            ax.set_xlabel("Outcome" if with_lines else "Age (dpf)")
            if not is_subplot or (is_subplot and outer_ind == 0):
                ax.set_ylabel(y_label)
            if is_cut_bottom:
                ax.set_ylim(bottom=0)
            # if max_value is not None:
            #     ax.set_ylim(top=max(max_value, ax.get_ylim()[1]))
            if not is_subplot:
                # ax.legend(loc='best', facecolor='none', framealpha=0.0)
                ax.set_title(t_str)
                image_fish_scatter_path = \
                    os.path.join(plot_dir, save_fig_fixname(add_name + "_{0}".format(outer_key) + filename))
                try:
                    plt.subplots_adjust(bottom=0.15)
                    # plt.tight_layout()
                    plt.savefig(image_fish_scatter_path, bbox_inches='tight')
                    image_paths.append(image_fish_scatter_path)
                except Exception as e:
                    print(e)
                    traceback.print_tb(e.__traceback__)
                plt.close()
                print("cloaed", image_fish_scatter_path)
            else:
                if with_title:
                    ax.set_title("{0}".format(t_str))

    if not split_per_outcome and not is_subplot:  # is_combine_age:  # plot all on the same graph
        ax.legend()
        spaces = (len(outcome_names) + 1)
        plt.xticks(range(spaces // 2 - 1, len(age_names) * spaces, spaces), age_names, rotation=0)
        ax.set_xlabel("age (dpf)")
        ax.set_ylabel(y_label)
        ax.set_ylim(bottom=0)
        if max_value is not None:
            ax.set_ylim(top=max(max_value, ax.get_ylim()[1]))
        # if max_value is not None:
        #     ax.set_ylim(top=max_value)
        if with_title:
            ax.set_title(curr_title)
        image_fish_scatter_path = os.path.join(plot_dir, save_fig_fixname(add_name + filename))
        try:
            plt.tight_layout()
            plt.savefig(image_fish_scatter_path, bbox_inches='tight')
            image_paths.append(image_fish_scatter_path)
        except Exception as e:
            print(e)
            traceback.print_tb(e.__traceback__)
        plt.close()
    if is_subplot:
        if with_sup_title:
            plt.suptitle(curr_title)
        image_fish_scatter_path = os.path.join(plot_dir, save_fig_fixname(add_name + filename))
        try:
            plt.subplots_adjust(bottom=0.15)
            # plt.tight_layout()
            plt.savefig(image_fish_scatter_path, bbox_inches='tight')
            image_paths.append(image_fish_scatter_path)
        except Exception as e:
            print(e)
            traceback.print_tb(e.__traceback__)
        plt.close()

    logging.info(image_paths)
    return image_paths


def format_pval(p_val):
    """Make sure if value is too small, it will be printed as 1e... and otherwise, it will be rounded to constant digits
    """
    return 'p = {0}'.format('{0:.3f}'.format(p_val) if p_val > 1e-3 else format_float_scientific(p_val, precision=2))


def plot_densities(general_filename, plot_dir, per_age_statistics, title_add, what, add_name, palette="Set2",
                   is_title=False, is_cum_line=False, d_fake=None,
                   x_label="Distance in mm", y_label="Paramecia", is_legend_outside=False, is_box=False, is_rel=False,
                   is_joint=True,
                   colormap=cm.jet, dpi=600, fig_size=FIG_SIZE, is_kde=True, is_hist=True, is_outer_keys_subplots=False,
                   is_cdf=False, all_together=False,
                   outer_keys_list=[], inner_keys_list=[], outer_keys_map={}, inner_keys_map={}, f=lambda x: x):
    image_paths = []
    out_keys = outer_keys_list if len(outer_keys_list) > 0 and \
                                  np.array([a in per_age_statistics.keys() for a in outer_keys_list]).all() else list(
        per_age_statistics.keys())

    dict_value = per_age_statistics[out_keys[0]]
    age_names = list(dict_value.keys())
    age_names = inner_keys_list if len(inner_keys_list) > 0 and \
                                   np.array([a in age_names for a in inner_keys_list]).all() else age_names
    print("age", age_names)
    if len(age_names) == 3:
        paired_ages = [(p1, p2, p3, age_names[p1], age_names[p2], age_names[p3])
                       for p1 in range(len(age_names)) for p2 in range(p1 + 1, len(age_names)) for p3 in
                       range(p2 + 1, len(age_names))]
    else:
        paired_ages = [(p1, p2, None, age_names[p1], age_names[p2], None)
                       for p1 in range(len(age_names)) for p2 in range(p1 + 1, len(age_names))]

    if is_outer_keys_subplots:
        print("sub", len(paired_ages), len(out_keys))
        fig, axes = plt.subplots(len(paired_ages), len(out_keys), figsize=fig_size, dpi=dpi, sharex=True, sharey=True)
        axes = axes if isinstance(axes, (list, np.ndarray)) else [axes]
        print(axes)
        for a in axes:
            if isinstance(a, (list, np.ndarray)):
                for aa in a:
                    aa.spines["top"].set_visible(False)
                    aa.spines["right"].set_visible(False)
            else:
                a.spines["top"].set_visible(False)
                a.spines["right"].set_visible(False)
    elif all_together:
        fig, axes = plt.subplots(1, 1, figsize=fig_size, dpi=dpi, sharex=True, sharey=True)
        axes = axes if isinstance(axes, (list, np.ndarray)) else [axes]
        for a in axes:
            a.spines["top"].set_visible(False)
            a.spines["right"].set_visible(False)

    data = []
    yy_label = y_label
    logging.info(out_keys)
    for ind, outcome in tqdm(enumerate(out_keys), desc="{0}".format(out_keys)):
        dict_value = per_age_statistics[outcome]
        age_names = list(dict_value.keys())
        age_names = inner_keys_list if len(inner_keys_list) > 0 and \
                                       np.array([a in age_names for a in inner_keys_list]).all() else age_names
        if len(age_names) == 3:
            paired_ages = [(p1, p2, p3, age_names[p1], age_names[p2], age_names[p3])
                           for p1 in range(len(age_names)) for p2 in range(p1 + 1, len(age_names)) for p3 in range(p2 + 1, len(age_names))]
        else:
            paired_ages = [(p1, p2, None, age_names[p1], age_names[p2], None)
                           for p1 in range(len(age_names)) for p2 in range(p1 + 1, len(age_names))]
        logging.info(paired_ages)
        if len(paired_ages) == 0:
            continue
        if not is_outer_keys_subplots and not all_together:
            if len(paired_ages) > 3:
                fig, axes = plt.subplots(2, int(np.ceil(len(paired_ages) / 2)), figsize=fig_size, dpi=dpi,
                                         sharex=True, sharey=True)
            else:
                fig, axes = plt.subplots(1, len(paired_ages), figsize=fig_size, dpi=dpi, sharex=True, sharey=True)
            if not isinstance(axes, (list, np.ndarray)):
                axes = [axes]
            print(axes)
        filename = general_filename + "_" + what + "_" + outcome.replace("-", "_")
        filename += "_kde" if is_kde else ""
        filename += "_cdf" if is_cdf else ""
        filename += "_hist" if is_hist else ""
        curr_title = x_label + title_add + " ({1}: {0})".format(outcome, what)
        normalize = colors.Normalize(vmin=0, vmax=len(paired_ages))

        for pair_ind, (p1, p2, p3, age_1, age_2, age_3) in tqdm(reversed([_ for _ in enumerate(paired_ages)]),
                                                                desc="ages", disable=True):
            if len(axes) == 1:
                ax = axes[0]
            elif len(axes.shape) > 1:
                if not is_outer_keys_subplots:
                    ax = axes[pair_ind % 2, pair_ind // 2]
                else:
                    ax = axes[pair_ind, ind]
            else:
                if len(out_keys) == 1:
                    ax = axes[pair_ind]
                else:
                    ax = axes[ind]

            y1, y2, y3 = get_y(dict_value[age_1]), get_y(dict_value[age_2]), \
                         get_y(dict_value[age_3]) if age_3 is not None else []
            age_1_s = inner_keys_map[age_1] if age_1 in inner_keys_map.keys() else age_1
            age_2_s = inner_keys_map[age_2] if age_2 in inner_keys_map.keys() else age_2
            age_3_s = inner_keys_map[age_3] if age_3 in inner_keys_map.keys() else age_3
            leg = 'brief' if not is_outer_keys_subplots or (pair_ind == len(paired_ages) - 1 and \
                                                            ind == len(out_keys) - 1) else False
            if is_kde:
                yy_label = "" + "Probability density"
                y1 = np.array(y1)
                y2 = np.array(y2)
                print("sapir kde ", y1, y2)
                sns.kdeplot(y1, fill=True, common_norm=False, alpha=.1, linewidth=1, label=age_1_s, ax=ax,
                            color=colormap(age_1_s, outcome))  # cumulative
                sns.kdeplot(y2, fill=True, common_norm=False, alpha=.1, linewidth=1, label=age_2_s, ax=ax,
                            color=colormap(age_2_s, outcome))
                if age_3 is not None:
                    sns.kdeplot(y3, fill=True, common_norm=False, alpha=.1, linewidth=1, label=age_3_s, ax=ax,
                                color=colormap(age_3_s, outcome))
                # sns.kdeplot(y1[(y1 >= 3) & (y1 <= 6)], fill=False, common_norm=False, alpha=.5, linewidth=1,
                #             label=age_1_s + outcome, ax=ax)  # cumulative
                # sns.kdeplot(y2[(y2 >= 3) & (y2 <= 6)], fill=False, common_norm=False, alpha=.5, linewidth=1,
                #             label=age_2_s + outcome, ax=ax)
                plt.yticks([], [])
            elif is_cdf:
                y1 = np.array(y1)
                y2 = np.array(y2)
                sns.kdeplot(y1, fill=False, cumulative=True, common_norm=False, label=age_1_s, ax=ax,
                            color=colormap(age_1_s, outcome))  # cumulative
                sns.kdeplot(y2, fill=False, cumulative=True, common_norm=False, label=age_2_s, ax=ax,
                            color=colormap(age_2_s, outcome))
                # sns.kdeplot(y1[(y1 >= 3) & (y1 <= 6)], fill=False, cumulative=True, common_norm=False, label=age_1_s + outcome, ax=ax)  # cumulative
                # sns.kdeplot(y2[(y2 >= 3) & (y2 <= 6)], fill=False, cumulative=True, common_norm=False, label=age_2_s + outcome, ax=ax)
                yy_label = "" + "Cumulative density"
                # myPlot = sns.ecdfplot(y1, label=age_1_s, ax=ax)  # cumulative
                # lines2D2 = [obj for obj in myPlot.findobj() if str(type(obj)) == "<class 'matplotlib.lines.Line2D'>"]
                # x, y = lines2D2[0].get_data()[0], lines2D2[0].get_data()[1]
                # df = pd.DataFrame.from_dict({'x': x, 'y': y})
                # print(outcome, age_1_s, df[df['x'].abs() <= 0.2])
                # myPlot = sns.ecdfplot(y2, label=age_2_s + outcome, ax=ax)
                # lines2D2 = [obj for obj in myPlot.findobj() if str(type(obj)) == "<class 'matplotlib.lines.Line2D'>"]
                # x, y = lines2D2[1].get_data()[0], lines2D2[1].get_data()[1]
                # df = pd.DataFrame.from_dict({'x': x, 'y': y})
                # print(outcome, age_2_s, df[df['x'].abs() <= 0.2])
                # plt.yticks([], [])
            elif is_hist:
                yy_label = y_label + " number"
                if len(y1) > 0:
                    sns.histplot(y1, label=age_1_s, ax=ax, kde=True, discrete=True, color=colormap(age_1_s, outcome))
                if len(y2) > 0:
                    sns.histplot(y2, label=age_2_s, ax=ax, kde=True, discrete=True, color=colormap(age_2_s, outcome))
            elif is_box:
                df = pd.concat([pd.DataFrame.from_dict({what: age_1_s, x_label: y1}),
                                pd.DataFrame.from_dict({what: age_2_s, x_label: y2})])
                sns.boxplot(data=df, y=x_label, x=what, ax=ax, palette=palette)
            elif is_rel:
                y12, y22 = get_y(dict_value[age_1], k='y2'), get_y(dict_value[age_2], k='y2')
                df = pd.concat([pd.DataFrame.from_dict({what: age_1_s, x_label: f(y1), y_label: f(y12)}),
                                pd.DataFrame.from_dict({what: age_2_s, x_label: f(y2), y_label: f(y22)})])

                if is_joint:
                    sns.kdeplot(x=f(y1), y=f(y12), label=age_1_s, color=colormap(age_1_s, outcome),
                                thresh=.2, ax=ax)
                    sns.kdeplot(x=f(y2), y=f(y22), label=age_2_s, color=colormap(age_2_s, outcome),
                                thresh=.2, ax=ax)
                    # sns.kdeplot(data=df, x=x_label, y=y_label,hue=what,
                    #             fill=True, common_norm=False, alpha=.5, linewidth=0,# label=age_1_s,
                    #             levels=5, thresh=.2,
                    #             palette=[colormap(normalize(p1)), colormap(normalize(p2))], legend=leg,
                    #             ax=ax)  # cumulative
                    # sns.jointplot(data=df, x=x_label, y=y_label, hue=what, ax=ax,
                    #                         palette=[colormap(normalize(p1)), colormap(normalize(p2))], legend=leg)
                else:
                    sns.scatterplot(data=df, x=x_label, y=y_label, hue=what, style=what, ax=ax,
                                    palette=[colormap(age_1_s, outcome),
                                             colormap(age_2_s, outcome)], legend=leg)
            else:
                # is_legend_outside = False  # cumulative
                yy_label = "Mean " + y_label
                ci = True
                for curr_y, curr_age, age_s in zip([y1, y2, y3], [age_1, age_2, age_3], [age_1_s, age_2_s, age_3_s]):
                    if len(curr_y) > 0:
                        x = dict_value[curr_age]['x']
                        if isinstance(x, list) and isinstance(x[0], dict):
                            x = [0]['ibi__1']
                        df = pd.DataFrame.from_dict({'x': x, 'y': curr_y})
                        x_u = np.unique(x)
                        if is_cum_line:
                            ci = False
                            y_mean = df.groupby('x').mean().cumsum()['y']
                            ax.plot(x_u, y_mean, label=age_s,
                                    color=colormap(age_s, outcome))
                            ax.errorbar(x_u, y_mean, label='_nolegend_', linestyle='None',
                                        yerr=df.groupby('x').sem()['y'], ecolor=colormap(age_s, outcome), capsize=1,
                                        elinewidth=1)
                            ii = np.concatenate([x_u[:-5:3], x_u[-5:]])
                            ax.set_xticks(ii)
                            ax.set_xticklabels(["{:.1f}".format(_) for _ in ii])
                        else:
                            y_mean = df.groupby('x').mean()['y']
                            print(age_s, outcome, len(curr_y), df.groupby('x').sem())
                            sns.lineplot(x=x, y=curr_y, label=age_s, ax=ax,
                                         color=colormap(age_s, outcome),
                                         legend=leg, err_style="bars", ci=68)
                            if len(x_u) > 13:
                                ii = np.concatenate([x_u[:-5:3], x_u[-5:]])
                            else:
                                ii = np.concatenate([x_u[:-5:2], x_u[-5:]])
                            ax.set_xticks(ii)
                            ax.set_xticklabels(["{:.1f}".format(_) for _ in ii])
                            if not ci:
                                ax.errorbar(y_mean.index, y_mean, label='_nolegend_', linestyle='None',
                                            yerr=df.groupby('x').sem()['y'], ecolor=colormap(age_s, outcome), capsize=1,
                                            elinewidth=1)

                if d_fake is not None:
                    random_data = list(d_fake.values())[0]
                    random_data = list(random_data.values())[0]
                    df = pd.DataFrame.from_dict({'x': random_data['x'], 'y': random_data['y']})
                    x = np.unique(random_data['x'])
                    if is_cum_line:
                        ci = False
                        y_mean = df.groupby('x').mean().cumsum()['y']
                        ax.plot(x, y_mean, label="random", color='k')
                        ax.errorbar(x, y_mean, label='_nolegend_', linestyle='None',
                                    yerr=df.groupby('x').sem()['y'], ecolor='k', capsize=1, elinewidth=1)
                    else:
                        y_mean = df.groupby('x').mean()['y']
                        sns.lineplot(x=random_data['x'], y=random_data['y'], label="random", ax=ax,
                                     color='k', legend=leg, err_style="bars", ci=68)

            if not is_outer_keys_subplots and not all_together:
                if pair_ind == 1 or len(paired_ages) == 1:
                    ax.set_xlabel(x_label)
                if pair_ind == 0:
                    ax.set_ylabel(yy_label)
            else:
                if not is_box:
                    ax.set_ylabel("")
                    ax.set_xlabel("")

            add_legend = (pair_ind == len(paired_ages) - 1 and \
                          ind == len(out_keys) - 1)

            if add_legend or all_together:
                if is_legend_outside:
                    ax.legend(loc='center left', bbox_to_anchor=(0.55, 0.9), facecolor='none', framealpha=0.0)
                else:
                    ax.legend(loc='best', facecolor='none', framealpha=0.0)

            if not is_outer_keys_subplots and is_title:
                ax.set_title("{0} vs {1}".format(age_1_s, age_2_s))
            else:
                out_s = outer_keys_map[outcome] if outcome in outer_keys_map.keys() else outcome
                if len(axes) > 1:
                    ax.set_title(out_s)

        if not is_outer_keys_subplots and not all_together:
            if is_legend_outside:
                plt.legend(loc='center left', bbox_to_anchor=(0.55, 0.9), facecolor='none', framealpha=0.0,
                           frameon=False)
            else:
                plt.legend(loc='best', facecolor='none', framealpha=0.0, frameon=False)

            if is_title:
                plt.suptitle(curr_title)
            # fig.tight_layout()
            filename += "_kde" if is_kde else ""
            filename += "_hist" if is_hist else ""
            filename += "_box" if is_box else ""
            filename += "_rel" if is_rel and not is_joint else ""
            filename += "_relj2" if is_rel and is_joint else ""
            filename += "_cum" if is_cum_line else ""
            image_fish_scatter_path = os.path.join(plot_dir, save_fig_fixname(add_name + filename + ".pdf"))
            try:
                plt.savefig(image_fish_scatter_path)
                image_paths.append(image_fish_scatter_path)
            except Exception as e:
                print(e)
                traceback.print_tb(e.__traceback__)
            plt.close()
    if is_outer_keys_subplots or all_together:
        filename = general_filename + "_" + what + "_" + "all".replace("-", "_")
        filename += "_kde" if is_kde else ""
        filename += "_hist" if is_hist else ""
        filename += "_box" if is_box else ""
        filename += "_rel" if is_rel and not is_joint else ""
        filename += "_relj2" if is_rel and is_joint else ""
        filename += "_cum" if is_cum_line else ""
        from matplotlib import rcParams
        if not is_box:
            fig.text(0.5, 0.02, x_label, va='center', ha='center', fontsize=rcParams['axes.labelsize'])
            fig.text(0.02, 0.5, yy_label, va='center', ha='center', rotation='vertical',
                     fontsize=rcParams['axes.labelsize'])

        # plt.suptitle("")
        plt.legend(loc='best')
        plt.subplots_adjust(left=0.2)
        fig.tight_layout()
        image_fish_scatter_path = os.path.join(plot_dir, save_fig_fixname(add_name + filename + ".pdf"))
        try:
            plt.savefig(image_fish_scatter_path)
            image_paths.append(image_fish_scatter_path)
        except Exception as e:
            print(e)
            traceback.print_tb(e.__traceback__)
        plt.close()

    return image_paths
