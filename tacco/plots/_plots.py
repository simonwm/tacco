import pandas as pd
import numpy as np
import anndata as ad
import scanpy as sc
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import matplotlib.lines as mlines
from matplotlib.colors import to_rgb, to_rgba, rgb_to_hsv, hsv_to_rgb, to_rgba_array, PowerNorm, LinearSegmentedColormap, to_hex, ListedColormap, LogNorm, Normalize
from matplotlib.cm import ScalarMappable
import seaborn as sns
import scipy.stats
import scipy.cluster
from ..tools._enrichments import get_compositions, get_contributions, enrichments
from .. import utils
from .. import tools
from .. import get
from .. import preprocessing
import joblib
import gc
from numba import njit

def get_min_max(vector, log=True):
    if log:
        maximum = np.log(np.nanmax(vector))
        minimum = np.log(np.nanmin(vector[vector > 0]))
        delta = (maximum - minimum) * 0.1
        maximum += delta
        minimum -= delta
        maximum = np.exp(maximum)
        minimum = np.exp(minimum)
    else:
        maximum = np.nanmax(vector)
        minimum = np.nanmin(vector)
        #delta = (maximum - minimum) * 0.1
        #maximum += delta
        #minimum -= delta
    return minimum, maximum
        
def _correlations(x, y, log=False):
    if log:
        x, y = np.log(x), np.log(y)
    x_y_finite = np.isfinite(x) & np.isfinite(y)
    x, y = x[x_y_finite], y[x_y_finite]
    nDOF = len(x) - 1
    x2 = np.sum((x-y-x.mean()+y.mean())**2) / (np.var(y) * nDOF)
    pc = scipy.stats.pearsonr(x,y)[0]
    return pc, x2

def _scatter_plot(ax,x,y,sizes=None,colors=None,alpha=1.0,marker='o',log=False):
    ax.scatter(x=x,y=y,s=sizes,c=colors,alpha=alpha,marker=marker)
    pc, x2 = _correlations(x, y, log=log)
    ax.annotate('r=%.2f'%(pc), (0.05,0.95), xycoords='axes fraction', horizontalalignment='left', verticalalignment='top')
    ax.annotate('$\\chi^2_m$=%.2f'%(x2), (0.95,0.00), xycoords='axes fraction', horizontalalignment='right', verticalalignment='bottom')
    if log:
        ax.set_xscale('log')
        ax.set_yscale('log')

def _composition_bar(compositions, colors, horizontal=True, ax=None, legend=True):
    n_freqs = len(compositions.index)
    
    x = np.arange(compositions.shape[0])
    fig = None
    if horizontal:
        if ax is None:
            fig, ax = plt.subplots(figsize=(11,1*(n_freqs+2)))
        bottom = np.full_like(x,0)
        for t in colors.index[::1]:
            if t in compositions:
                ax.barh(x, compositions[t], height=0.55, label=t, left=bottom, color=colors[t])
                bottom = bottom + np.array(compositions[t])
            else:
                ax.barh(x, np.zeros(compositions.shape[0]), height=0.55, label=t, left=bottom, color=colors[t])

        ax.set_xlim([0, 1])
        ax.set_xlabel('composition')
        ax.set_ylim(x.min()-0.75,x.max()+0.75)
        ax.set_yticks(x)
        ax.set_yticklabels(compositions.index, ha='right')
        if legend:
            ax.legend(bbox_to_anchor=(-0.0, 1.0), loc='lower left', ncol=6)
    else:
        if ax is None:
            fig, ax = plt.subplots(figsize=(1*(n_freqs+2),8))
        bottom = np.array(x)*0+1
        for t in colors.index[::-1]:
            if t in compositions:
                bottom = bottom - np.array(compositions[t])
                ax.bar(x, compositions[t], width=0.55, label=t, bottom=bottom, color=colors[t])
            else:
                ax.bar(x, np.zeros(compositions.shape[0]), width=0.55, label=t, bottom=bottom, color=colors[t])

        ax.set_ylim([0, 1])
        ax.set_ylabel('composition')
        ax.set_xlim(x.min()-0.75,x.max()+0.75)
        ax.set_xticks(x)
        ax.set_xticklabels(compositions.index, rotation=30,va='top',ha='right')
        if legend:
            ax.legend(bbox_to_anchor=(1, 1), loc='upper left', ncol=1)

    if fig is not None:
        fig.tight_layout()
    
    return fig

def _complement_color(r,g,b):
    h, s, v = rgb_to_hsv((r,g,b))
    _r, _g, _b = hsv_to_rgb(((h+0.5)%1,s,v))
    return _r, _g, _b

def _scatter_frame(df, colors, ax_, id_=None, cmap=None, cmap_vmin_vmax=None, out_min_max=None, joint=True, point_size=3, grid=False, margin=0.25, scale=None, title='', rasterized=True):
    
    pie = False
    if not isinstance(rasterized, bool):
        if rasterized != 'pie':
            raise ValueError(f'`rasterized` has to be boolean or "pie"!')
        if cmap is not None:
            raise ValueError(f'`rasterized=="pie"` cannot use a `cmap`!')
        pie = True
        rasterized = False
        
        color_array = np.array([colors[c] for c in df.columns.difference(['x','y'])])

        dpi = matplotlib.rcParams['figure.dpi']
        point_radius = np.sqrt(point_size / np.pi) / 2 * dpi
        
        # convert from pixel to axes units
        point_radius = (ax_[0].transAxes.inverted().transform((point_radius, 0)) - ax_[0].transAxes.inverted().transform((0, 0)))[0]
    
    if id_ is None:
        n_cols = len(colors)
        id_ = np.full(fill_value=-1,shape=(n_cols,2),dtype=object)
    
    n_types = len(colors)
        
    for ax in ax_:
        ax.set_aspect(1)
        ax.grid(grid)
        
    if joint is None:
        _ax_ = ax_[1:]
        _id_ = id_[1:]
    elif joint:
        _ax_ = [None] * n_types
        _id_ = [None] * n_types
    else:
        _ax_ = ax_
        _id_ = id_
    
    axsize = ax_[0].get_window_extent().transformed(ax_[0].get_figure().dpi_scale_trans.inverted()).size
    canvas_size = axsize
    
    coords = df[['x','y']].to_numpy()
    
    # use the largest possible amount of space of the axes
    coords_min = coords.min(axis=0) if len(coords) > 0 else np.nan
    _coords = coords - coords_min
    coords_range = _coords.max(axis=0) if len(coords) > 0 else np.nan
    scale_0 = (canvas_size[0]-2*margin) / coords_range[0]
    scale_1 = (canvas_size[1]-2*margin) / coords_range[1]
    if scale is None:
        scale = min(scale_0, scale_1)
    # center the coords
    offset_0 = (canvas_size[0] - coords_range[0] * scale) / 2
    offset_1 = (canvas_size[1] - coords_range[1] * scale) / 2

    # coords_min is mapped to (offset_0,offset_1)
    # => coords_min-(offset_0,offset_1)/scale is mapped to (0,0)
    # canvas_size / scale is the range
    # => extent_min = coords_min - (offset_0,offset_1) / scale
    # => extent_max = extent_min + canvas_size / scale
    extent_min = coords_min - np.array((offset_0,offset_1)) / scale
    extent_max = extent_min + canvas_size / scale
    extent = [extent_min[0], extent_max[0], extent_min[1], extent_max[1]]
    
    x, y = df['x'], df['y']
    for (t,c),ax,id in zip(colors.items(),_ax_,_id_):
        if pie:
            only_color_array = color_array.copy()
            only_color_array[only_color_array != c] = '#fff0' # transparent white
            
            weights = df[df.columns.difference(['x','y'])].to_numpy()
            
            def plotit(ax, id, color_array):
                for _weights,_x,_y in zip(weights, x, y):
                    ax.pie(_weights,colors=color_array,radius=point_radius,center=(_x,_y), frame=True)

                ax.set_xlim(extent[0:2])
                ax.set_ylim(extent[2:4])

                if out_min_max is not None: # get values for backpropagation of the value range
                    out_min_max[id[0],id[1]] = vals.min(), vals.max()

            if joint is None or not joint:
                plotit(ax, id, only_color_array)
                ax.set_title(f'{title}: {t}' if title is not None and title != '' else f'{t}')
            if joint is None or joint:
                if (t,c) == list(iter(colors.items()))[0]: # plot full pies only once
                    plotit(ax_[0], id_[0],color_array)
            
            
        elif cmap is None:
            #scale = 200.0 * np.random.rand(n)
            r, g, b = to_rgb(c)
            _r, _g, _b = _complement_color(r,g,b) # complement colors for negative values
            if t in df.columns:
                vals = np.maximum(np.minimum(df[t].to_numpy(),1),-1)
                # dont plot invisible points
                visible = vals != 0
                vals, _x, _y = vals[visible], x[visible], y[visible]
                # separately plot efficient cases
                which1p = vals >= +1
                vals1p, _x1p, _y1p = vals[which1p], _x[which1p], _y[which1p]
                which1m = vals <= -1
                vals1m, _x1m, _y1m = vals[which1m], _x[which1m], _y[which1m]
                # and the rest
                remaining = (vals > -1) & (vals < 1)
                vals, _x, _y = vals[remaining], _x[remaining], _y[remaining]
                color1p = [r,g,b]
                color1m = [_r,_g,_b]
                color = np.hstack([np.array([[r,_r],[g,_g],[b,_b]]).T[(1-np.sign(vals).astype(np.int8))//2],np.abs(vals)[:,None]])
                def plotit(ax, id):
                    if len(_x) > 0:
                        ax.scatter(_x, _y, color=color, s=point_size, edgecolors='none', rasterized=rasterized)
                    if len(_x1p) > 0:
                        ax.scatter(_x1p, _y1p, color=color1p, s=point_size, edgecolors='none', rasterized=rasterized)
                    if len(_x1m) > 0:
                        ax.scatter(_x1m, _y1m, color=color1m, s=point_size, edgecolors='none', rasterized=rasterized)
                    
                    ax.set_xlim(extent[0:2])
                    ax.set_ylim(extent[2:4])

                    if out_min_max is not None: # get values for backpropagation of the value range
                        out_min_max[id[0],id[1]] = vals.min(), vals.max()

                if joint is None or not joint:
                    plotit(ax, id)
                    ax.set_title(f'{title}: {t}' if title is not None and title != '' else f'{t}')
                if joint is None or joint:
                    plotit(ax_[0], id_[0])
        else:
            if t in df.columns:
                vals = df[t].to_numpy()
                def plotit(ax, id):

                    if cmap_vmin_vmax is not None:
                        vmin, vmax = cmap_vmin_vmax
                        norm = Normalize(vmin=vmin, vmax=vmax)
                    else:
                        vmin, vmax = vals.min(), vals.max()
                        norm = None
                    
                    ax.scatter(x, y, c=vals, s=point_size, edgecolors='none', cmap=cmap, norm=norm, rasterized=rasterized)
                    
                    ax.set_xlim(extent[0:2])
                    ax.set_ylim(extent[2:4])

                    if out_min_max is not None: # get values for backpropagation of the value range
                        out_min_max[id[0],id[1]] = vmin, vmax

                if joint is None or not joint:
                    plotit(ax, id)
                    ax.set_title(f'{title}: {t}' if title is not None and title != '' else f'{t}')
                if joint is None or joint:
                    plotit(ax_[0], id_[0])

@njit(cache=True)
def _make_stencil(point_radius):
    extra_size = int(np.ceil(point_radius - 0.5))
    stencil_size = 1 + 2 * extra_size
    stencil = np.zeros((stencil_size,stencil_size))
    # monte carlo evaluate density profile of a disc... There have to be better options, but maybe not more straightforward ones...
    # for one quadrant
    np.random.seed(42)
    point_radius2 = point_radius**2
    n_samples = int(point_radius2*10000)
    samples = np.random.uniform(0,1,size=(n_samples,2)) * point_radius
    quadrant = np.zeros((1+extra_size,1+extra_size))
    for sample in samples:
        d2 = (sample**2).sum()
        if d2 < point_radius2:
            _sample = sample+0.5
            quadrant[int(_sample[0]),int(_sample[1])] += 1
    # symmetrize octant
    quadrant = quadrant + quadrant.T # using += with numba and matrices gives wrong results: https://github.com/numba/numba/issues/6949 
    # replicate to other quadrants
    stencil[:-extra_size,:-extra_size] += quadrant[::-1,::-1]
    stencil[extra_size:,:-extra_size] += quadrant[::,::-1]
    stencil[:-extra_size,extra_size:] += quadrant[::-1,::]
    stencil[extra_size:,extra_size:] += quadrant[::,::]
    # normalize
    stencil /= stencil.sum()
    return stencil
@njit(cache=True)
def _draw_weights(canvas_size, weights, data, offset_x, offset_y, stencil):
    extra_size = (stencil.shape[0] - 1) // 2
    canvas = np.zeros(canvas_size)
    for w,(x,y) in zip(weights,data):
        xo,yo = int(x+offset_x),int(y+offset_y)
        canvas[(xo-extra_size):(xo+extra_size+1),(yo-extra_size):(yo+extra_size+1)] += w * stencil
    return canvas
                    
def _render_frame(df, colors, ax_, id_=None, cmap=None, cmap_vmin_vmax=None, out_min_max=None, joint=True, point_size=3, grid=False, margin=0.25, scale=None, title='', color_mix_mode='xyv'):
    
    typing = df[df.columns.intersection(colors.index)]
    coords = df[['x','y']].to_numpy()
    
    coords_min = coords.min(axis=0) if len(coords) > 0 else np.nan
    coords = coords - coords_min
    coords_range = coords.max(axis=0) if len(coords) > 0 else np.nan
    
    if cmap is None: # empty dots dont get rendered
        nonzero = np.abs(typing.to_numpy()).sum(axis=1) > 0
        typing = typing.loc[nonzero]
        coords = coords[nonzero]
    
    n_types = len(typing.columns)
    weights = typing
    
    dpi = matplotlib.rcParams['figure.dpi']
    
    if id_ is None:
        n_cols = len(colors)
        id_ = np.full(fill_value=-1,shape=(n_cols,2),dtype=object)
        
    for ax in ax_:
        ax.grid(grid)
        
    if joint is None:
        _ax_ = ax_[1:]
        _id_ = id_[1:]
    elif joint:
        _ax_ = [None] * n_types
        _id_ = [None] * n_types
    else:
        _ax_ = ax_
        _id_ = id_
    
    point_radius = np.sqrt(point_size / np.pi) / 72 * dpi
    
    margin = int(np.ceil(margin * dpi + point_radius))
    axsize = ax_[0].get_window_extent().transformed(ax_[0].get_figure().dpi_scale_trans.inverted()).size
    canvas_size = ((axsize)*dpi).astype(int)
    
    # use the largest possible amount of space of the axes
    scale_0 = (canvas_size[0]-1-2*margin) / coords_range[0]
    scale_1 = (canvas_size[1]-1-2*margin) / coords_range[1]
    if scale is None:
        scale = min(scale_0, scale_1)
    else:
        scale = scale * dpi
    # center the coords
    offset_0 = (canvas_size[0]-1 - coords_range[0] * scale) / 2
    offset_1 = (canvas_size[1]-1 - coords_range[1] * scale) / 2

    # coords_min is mapped to (offset_0,offset_1)
    # => coords_min-(offset_0,offset_1)/scale is mapped to (0,0)
    # canvas_size / scale is the range
    # => extent_min = coords_min - (offset_0,offset_1) / scale
    # => extent_max = extent_min + canvas_size / scale
    extent_min = coords_min - np.array((offset_0,offset_1)) / scale
    extent_max = extent_min + canvas_size / scale
    extent = [extent_min[0], extent_max[0], extent_min[1], extent_max[1]]
    
    coords = coords * scale
    
    stencil = _make_stencil(point_radius)
    
    canvases = { c: _draw_weights((canvas_size[0],canvas_size[1]), weights[c].to_numpy(), coords, offset_0, offset_1, stencil) for c in weights.columns}
    
    sum_canvas = sum(canvases.values())
    finite_sum = sum_canvas!=0
    
    colors = {i:np.array(to_rgb(colors[i])) for i in colors.index}
    
    # alpha just tells us whether there is data on the pixel or not
    def get_alpha(norm_canvas, stencil):
        alpha = np.log1p(np.abs(norm_canvas)) / np.log1p(stencil.max())
        alpha[alpha>1] = 1 # cut off values with too high alpha
        return alpha
    def add_alpha(canvas, norm_canvas, stencil):
        canvasA = np.zeros_like(canvas, shape=(*canvas.shape[:-1],4))
        canvasA[...,:-1] = canvas
        canvasA[...,-1] = get_alpha(norm_canvas, stencil)
        return canvasA

    if cmap is None:
        #norm_canvas = _draw_weights((canvas_size[0],canvas_size[1]), np.ones(shape=len(weights)), coords, offset_0, offset_1, stencil)
        norm_canvas = sum([np.abs(canvas) for canvas in canvases.values()])
        #finite_norm = norm_canvas!=0
        
        if joint is None or joint:
            canvas = mix_base_colors(np.stack([canvases[t] for t in canvases],axis=-1), np.array([colors[t] for t in canvases]), mode=color_mix_mode)
            for i in range(3): # remove zero weight colors
                canvas[...,i][~finite_sum] = 1
            canvas[canvas>1] = 1 # numerical accuracy issues

            canvasA = add_alpha(canvas, norm_canvas, stencil)

            ax_[0].imshow(canvasA.swapaxes(0,1), origin='lower', extent=extent)
            
            if out_min_max is not None: # get values for backpropagation of the value range
                out_min_max[id_[0][0],id_[0][1]] = sum(canvases.values()).min()/stencil.max(), sum(canvases.values()).max()/stencil.max()
        
        if joint is None or not joint:
            for (t,c),ax,id in zip(colors.items(),_ax_,_id_):
                if t in canvases:
                    canvas  = canvases[t][...,None] * colors[t]
                    finite_t = canvases[t]!=0
                    for i in range(3): # normalize the colors by the weights
                        canvas[...,i][finite_t] = canvas[...,i][finite_t] / np.abs(canvases[t][finite_t])
                    canvas[canvas>1] = 1 # numerical accuracy issues

                    canvasA = add_alpha(canvas, np.abs(canvases[t]), stencil)

                    ax.imshow(canvasA.swapaxes(0,1), origin='lower', extent=extent)
                    ax.set_title(f'{title}: {t}' if title is not None and title != '' else f'{t}')

                    if out_min_max is not None: # get values for backpropagation of the value range
                        out_min_max[id[0],id[1]] = canvases[t].min()/stencil.max(), canvases[t].max()/stencil.max()
    else:
        norm_canvas = _draw_weights((canvas_size[0],canvas_size[1]), np.ones(shape=len(weights)), coords, offset_0, offset_1, stencil)
        finite_norm = norm_canvas!=0
        
        if joint is None or joint:
            canvas = sum_canvas.copy()
            canvas[finite_norm] = canvas[finite_norm] / norm_canvas[finite_norm] # normalize the values by the weights

            alpha = get_alpha(norm_canvas, stencil)

            if cmap_vmin_vmax is not None:
                vmin, vmax = cmap_vmin_vmax
            else:
                vmin, vmax = canvas[finite_norm].min(), canvas[finite_norm].max()
            norm = Normalize(vmin=vmin, vmax=vmax)

            ax_[0].imshow(canvas.swapaxes(0,1), alpha=alpha.swapaxes(0,1), origin='lower', extent=extent, cmap=cmap, norm=norm)
            
            if out_min_max is not None: # get values for backpropagation of the value range
                out_min_max[id_[0][0],id_[0][1]] = vmin, vmax
        
        if joint is None or not joint:
            for (t,c),ax,id in zip(colors.items(),_ax_,_id_):
                if t in canvases:
                    canvas = canvases[t].copy()
                    canvas[finite_norm] = canvas[finite_norm] / norm_canvas[finite_norm] # normalize the values by the weights

                    alpha = get_alpha(norm_canvas, stencil)

                    if cmap_vmin_vmax is not None:
                        vmin, vmax = cmap_vmin_vmax
                        norm = Normalize(vmin=vmin, vmax=vmax)
                    else:
                        vmin, vmax = canvas[finite_norm].min(), canvas[finite_norm].max()
                        norm = None

                    ax.imshow(canvas.swapaxes(0,1), alpha=alpha.swapaxes(0,1), origin='lower', extent=extent, cmap=cmap, norm=norm)
                    ax.set_title(f'{title}: {t}' if title is not None and title != '' else f'{t}')

                    if out_min_max is not None: # get values for backpropagation of the value range
                        out_min_max[id[0],id[1]] = vmin, vmax

def _set_axes_labels(ax, axes_labels):
    if axes_labels is not None:
        if not pd.api.types.is_list_like(axes_labels) or len(axes_labels) != 2:
            raise ValueError(f'`axes_labels` {axes_labels!r} is not a list-like of 2 elements!')
        ax.set_xlabel(axes_labels[0])
        ax.set_ylabel(axes_labels[1])
                        
def spatial_distribution_plot(typing_data, coords, colors, n_cols=1, axs=None, joint=None, normalize=True, point_size=3, cmap=None, cmap_vmin_vmax=None, out_min_max=None, scale=None, grid=False, margin=0.25, render=False, rasterized=True, noticks=False, axes_labels=None, on_data_legend=None):
    n_types = len(colors.index)
    if joint is None:
        n_y = (n_types + n_cols) // n_cols
    elif joint:
        n_cols = 1
        n_y = 1
    else:
        n_y = (n_types + n_cols - 1) // n_cols
    n_solutions = len(typing_data)
    if axs is None:
        fig, axs = plt.subplots(n_y,n_solutions*n_cols,figsize=(7*n_solutions*n_cols,7*n_y), squeeze=False)
    else:
        fig = None
        if axs.shape != (n_y,n_solutions*n_cols):
            raise Exception('spatial_distribution_plot got axs of wrong dimensions: need %s, got %s' % ((n_y,n_solutions*n_cols),axs.shape))
    _axs = np.full(fill_value=None,shape=(n_solutions,n_cols*n_y),dtype=object)
    _idx = np.full(fill_value=-1,shape=(n_solutions,n_cols*n_y,2),dtype=object)
    counter = 0
    for i in range(n_cols):
        for j in range(n_y):
            for c in range(n_solutions):
                _axs[c,counter] = axs[j,i*n_solutions+c]
                _idx[c,counter,:] = np.array([j,i*n_solutions+c])
            counter += 1
    
    for id_,ax_,df,title in zip(_idx,_axs,typing_data,typing_data.index):
        df = df.copy() # dont change the original df
        df = df[colors.index.intersection(df.columns)]
        if normalize: # normalize values per column to give alphas between 0 and 1
            df = df - df.min(axis=0).to_numpy()
            df = df / df.max(axis=0).to_numpy()
            
        df[['x','y']] = coords
        
        df = df.loc[:,~df.isna().all(axis=0)] # remove all-nan columns
        df = df.loc[~df.isna().any(axis=1)] # remove any-nan rows
        
        if isinstance(render, bool) and not render:
            _scatter_frame(df, colors, ax_, id_=id_, cmap=cmap, cmap_vmin_vmax=cmap_vmin_vmax, out_min_max=out_min_max, joint=joint, point_size=point_size, grid=grid, margin=margin, scale=scale, title=title, rasterized=rasterized)
        else:
            if not rasterized:
                raise ValueError(f'`render!=False` only works when `rasterized==True`')
            if isinstance(render, bool):
                color_mix_mode = 'xyv'
            else:
                color_mix_mode = render
            _render_frame (df, colors, ax_, id_=id_, cmap=cmap, cmap_vmin_vmax=cmap_vmin_vmax, out_min_max=out_min_max, joint=joint, point_size=point_size, grid=grid, margin=margin, scale=scale, title=title, color_mix_mode=color_mix_mode)
        
        for ax in ax_:
            
            _set_axes_labels(ax, axes_labels)

            if noticks:
                ax.set_xticks([])
                ax.set_yticks([])
                    
        if joint is None or joint:
            ax_[0].set_title(title)
        
        if on_data_legend is not None:
            def weighted_median(df, value_col, weights_col):
                df = df[[value_col, weights_col]].sort_values(value_col)
                return df[value_col][df[weights_col].cumsum() >= 0.5 * df[weights_col].sum()].iloc[0]
            def find_closest_point(center, coords, weights):
                coords = coords.loc[weights >= weights.max() * 0.9].to_numpy() # only consider points with weights above 90% of the max weight
                dists = utils.cdist(np.array(center).reshape((1,2)),coords)
                return coords[np.argmin(dists.flatten()).flatten()[0]]
            
            for annotation in [c for c in df.columns if c not in ['x','y']]:
                medians = []
                for direction in ['x','y']:
                    medians.append(weighted_median(df, direction, annotation))
                closest = find_closest_point(medians, df[['x','y']], df[annotation])
                for ax in ax_:
                    ax.text(*closest, on_data_legend[annotation] if annotation in on_data_legend else annotation, ha='center', va='center')
    
    return fig

def write_results_to_adata(adata, typing_data, pfn_factors=None, pfn_key='platform_normalization_factors'):
    if pfn_key in adata.varm:
        pfns = adata.varm[pfn_key]
    else:
        pfns = pd.DataFrame(index=adata.var.index)
    for typing, data in typing_data.items():
        adata.obsm[typing] = data.reindex(adata.obs.index)
        if pfn_factors is not None and typing in pfn_factors and pfn_factors[typing] is not None:
            pfns[typing] = pfn_factors[typing]
    if len(pfns.columns) > 0:
        adata.varm[pfn_key] = pfns

def get_default_colors(n, offset=0):
    """\
    Chooses default colors.
    This is a convenience wrapper around :func:`seaborn.color_palette` which
    provides a quasi-periodicity of 10, i.e. every 10 colors, the colors are
    related.
    
    Parameters
    ----------
    n
        Number of colors to choose OR a list of keys to choose colors for.
    offset
        The number of chosen colors to skip before starting to use them. This
        is useful for generating different sets of colors.
    
    Returns
    -------
    If `n` is a number, returns a list of colors. If `n` is a list-like,\
    returns a mapping of the elements of `n` to colors.
    
    """
  
    if pd.api.types.is_list_like(n):
        default_colors = get_default_colors(len(n), offset)
        return {name:color for name,color in zip(n,default_colors)}
    else:
        default_colors = [ to_hex(c) for c in [*sns.color_palette("bright"),*sns.color_palette("deep"),*sns.color_palette("dark"),*sns.color_palette("pastel")] ]
        default_colors *= ((n+offset) // len(default_colors) + 1)
        return default_colors[offset:(n+offset)]

def mix_base_colors(weights, base_colors_rgb, mode='xyv'):
    """\
    Mix colors "additively". In contrast to weighted averages over "rgb" values
    (which results in quite dark colors), the average can be done in "xyv"
    space, which is "hsv" with the "hs" part converted from polar to cartesian
    coordinates.
    
    Parameters
    ----------
    weights
        A weight tensor with last dimension `n_base` describing mixtures of
        `n_base` colors; this can be a :class:`~numpy.ndarray` or a
        :class:`~pandas.DataFrame`.
    base_colors_rgb
        An `n_base X 3` matrix defining the base colors in rgb space; this must
        be a :class:`~numpy.ndarray`.
    mode
        The mixing mode; available are:
        
        - 'rgb': average the rgb values in the rgb cube
        - 'xyv': average the xyz values in the hsv cylider
        
    Returns
    -------
    Returns the color mixtures depending on the type of `weights` either as\
    :class:`~numpy.ndarray` or :class:`~pandas.DataFrame`.
    
    """
    
    weights_index = None
    if hasattr(weights, 'to_numpy'):
        weights_index = weights.index
        weights = weights.to_numpy()
    weights = weights / weights.sum(axis=-1)[...,None]
    if mode == 'xyv':
        base_colors_hsv = np.array([matplotlib.colors.rgb_to_hsv(base_color_rgb[:3]) for base_color_rgb in base_colors_rgb])
        base_colors_xyv = np.array([
            np.cos(base_colors_hsv[:,0] * 2 * np.pi) * base_colors_hsv[:,1],
            np.sin(base_colors_hsv[:,0] * 2 * np.pi) * base_colors_hsv[:,1],
            base_colors_hsv[:,2]
        ]).T
        mixed_colors_xyv = weights @ base_colors_xyv
        mixed_colors_hsv = np.stack([
            np.arctan2(mixed_colors_xyv[...,1], mixed_colors_xyv[...,0])/(2 * np.pi),
            np.sqrt(mixed_colors_xyv[...,0]**2 + mixed_colors_xyv[...,1]**2),
            mixed_colors_xyv[...,2]
        ],axis=-1)
        mixed_colors_hsv[mixed_colors_hsv[...,0]<0,0] += 1
        mixed_colors_rgb = matplotlib.colors.hsv_to_rgb(mixed_colors_hsv)
        if weights_index is not None:
            mixed_colors_rgb = pd.DataFrame(mixed_colors_rgb, index=weights_index)
    elif mode == 'rgb':
        mixed_colors_rgb = weights @ base_colors_rgb
    else:
        raise ValueError(f'The mode "{mode}" is not implemented!')
    return mixed_colors_rgb

def _filter_types(typing_data, types, colors, show_only):
    if show_only is not None:
        show_only = pd.Index(show_only)
        if show_only.isin(types).all():
            colors = colors[show_only]
            types = colors.index
        else:
            raise Exception('Not all selected types %s are available in the data %s!' % (show_only, types))
    
    typing_data = typing_data.map(lambda data: data.reindex(columns=types))
    
    return typing_data, types, colors


def _get_colors(colors, types):
    if colors is not None:
        colors = pd.Series(colors)
        if types.isin(colors.index).all():
            #colors = colors[colors.index.intersection(types)]
            types = colors.index
        else:
            raise Exception('Not all types %s are given colors with %s!' % (types, colors))
    else:
        colors = pd.Series(get_default_colors(len(types)), index=types)
    
    return colors, types

def _get_adatas(adata):
    if isinstance(adata, ad.AnnData):
        adatas = pd.Series(index=[''],dtype=object)
        adatas[''] = adata
    elif isinstance(adata, pd.DataFrame):
        adatas = pd.Series(index=[''],dtype=object)
        adatas[''] = utils.dataframe2anndata(adata, None, None)
    elif isinstance(adata, dict):
        #adatas = pd.Series(adata) # it could be so simple - but it does not work for adatas...
        adatas = pd.Series(index=adata.keys(),dtype=object)
        for k,v in adata.items():
            if isinstance(v, pd.DataFrame):
                v = utils.dataframe2anndata(v, None, None)
            adatas[k] = v
    else:
        adatas = adata
    
    if (adatas.index.value_counts() != 1).any():
        raise ValueError('The series of adatas has non-unique indices: %s!', adatas.index)
    
    return adatas

def _validate_args(adata, keys, colors, show_only, reads=False, method_labels=None, counts_location=None, compositional=True):
    
    adatas = _get_adatas(adata)
    
    if isinstance(keys, str):
        methods = [keys]
    else:
        methods = keys
    try:
        #methods = pd.Series({k:v for k,v in methods.items()}) # check basically whether methods is a dict or pd.Series (having an items method), mapping sample names to methods
        # check basically whether methods is a dict or pd.Series (having an items method), mapping sample names to methods
        _methods = {}
        for k,v in methods.items():
            if isinstance(v,str):
                v = [v]
            _methods[k] = v
        methods = pd.Series(_methods)
    except:
        methods = pd.Series([methods]*len(adatas), index=adatas.index) # assume same methods available for all samples
    
    typing_data = []
    types = pd.Index([],dtype=object)
    for sample, adata in adatas.items():
            
        for method in methods[sample]:
            if pd.api.types.is_list_like(method):
                data = pd.DataFrame(index=adata.obs.index)
                for element in method:
                    if element in adata.obs:
                        data[element] = adata.obs[element]
                        if not pd.api.types.is_numeric_dtype(data[element]):
                            raise Exception(f'The adata obs key {element} did contain non-numeric data%s!' % ('' if sample == '' else (' for sample "%s"' % sample)))
                    elif element in adata.var.index:
                        _data = adata[:,element].X
                        if scipy.sparse.issparse(_data):
                            _data = _data.A
                        data[element] = _data.flatten()
                    else:
                        raise Exception(f'The key {element} is neither in obs.columns nor in var.index%s!' % ('' if sample == '' else (' for sample "%s"' % sample)))
            elif method in adata.obsm:
                data = adata.obsm[method].copy()
            elif method in adata.obs:
                iscat = hasattr(adata.obs[method], 'cat')
                number = pd.api.types.is_numeric_dtype(adata.obs[method])
                if iscat or not number:
                    data = pd.get_dummies(adata.obs[method])
                    if data.shape[1] > 100:
                        print(f'More than 30 categories were discovered in column `.obs[{method}]`, which probably leads to slow plotting! (This message can be removed, if the column is made a categorical column).')
                else:
                    data = pd.DataFrame({method:adata.obs[method]})
            else:
                raise Exception(f'The method key {method} is neither obs(m) key nor a list-like%s!' % ('' if sample == '' else (' for sample "%s"' % sample)))
                
            if compositional:
                
                data *= (data>=0).to_numpy() # some methods may export negative values... (RCTD did that once)
                data = (data / data.fillna(0).sum(axis=1).to_numpy()[:,None]).fillna(0)
                if reads: # scale by read count
                    counts = get.counts(adata, counts_location=counts_location)[adata.obs.index]
                    if counts.shape[1] != 0: # ...but only if there is data for that
                        data *= np.array(counts.X.sum(axis=1)).flatten()[:,None]
                
                if compositional == 'catsize':
                    sizes = get_cellsize(adata, data)
                    data /= sizes[data.columns].to_numpy()
                    data = (data / data.fillna(0).sum(axis=1).to_numpy()[:,None]).fillna(0)
            
            types = types.union(data.columns,sort=False)
            method_label = method if ((method_labels is None) or (method not in method_labels)) else method_labels[method]
            if sample == '' and method_label == '':
                name = ''
            elif sample == '':
                name = method_label
            elif isinstance(method_label, str) and method_label == '':
                name = sample
            else:
                name = f'{sample}; {method_label}'
            typing_data.append((name, sample, method, data))
            
            
    typing_data = pd.DataFrame(typing_data, columns=['name','sample','method','data']).set_index('name')
    
    colors, types = _get_colors(colors, types)
    
    if compositional and len(types) < 1:
        print(f'`compositional==True`, but there were less than 2 categories: {types!r}')
    
    typing_data['data'], types, colors = _filter_types(typing_data['data'], types, colors, show_only)
    
    return typing_data, adatas, methods, types, colors

def _validate_scatter_args(adata, position_key, keys, colors, show_only, method_labels=None, counts_location=None, compositional=True):
    typing_data, adatas, methods, types, colors = _validate_args(adata, keys, colors, show_only, method_labels=method_labels, counts_location=counts_location, compositional=compositional)
    
    coords = {}
    for sample, adata in adatas.items():
        _coords = get.positions(adata, position_key)
        _coords.columns = ['x','y']
        coords[sample] = _coords#.rename(columns={_coords.columns[0]:'x',_coords.columns[1]:'y'})
    
    return typing_data, adatas, methods, types, colors, coords

def subplots(
    n_x=1,
    n_y=1,
    axsize=(5,5),
    hspace=0.15,
    wspace=0.15,
    x_padding=None,
    y_padding=None,
    title=None,
    sharex='none',
    sharey='none',
    width_ratios=None,
    height_ratios=None,
    x_shifts=None,
    y_shifts=None,
):
    """\
    Creates a new figure with a grid of subplots.
    This is a convenience wrapper around :func:`matplotlib.pyplot.subplots`
    with parameters for axis instead of figure and absolute instead of relative
    units.
    
    Parameters
    ----------
    n_x
        Number of plots in horizontal/x direction
    n_y
        Number of plots in vertical/y direction
    axsize
        Size of a single axis in the plot
    hspace
        Relative vertical spacing between plots
    wspace
        Relative horizontal spacing between plots
    x_padding
        Absolute horizontal spacing between plots; this setting overrides
        `wspace`; if `None`, use the value from `wspace`
    y_padding
        Absolute vertical spacing between plots; this setting overrides
        `hspace`; if `None`, use the value from `hspace`
    title
        Sets the figure suptitle
    sharex
        Parameter for sharing the x-axes between the subplots; see the
        documentation of :func:`matplotlib.pyplot.subplots`
    sharey
        Parameter for sharing the y-axes between the subplots; see the
        documentation of :func:`matplotlib.pyplot.subplots`
    width_ratios
        Sets the ratios of widths for the columns of the subplot grid keeping
        the width of the widest column; if `None`, all columns have the same
        width
    height_ratios
        Sets the ratios of heights for the rows of the subplot grid keeping the
        height of the highest row; if `None`, all rows have the same height
    x_shifts
        The absolute shifts in position in horizontal/x direction per column of
        subplots; if `None`, the columns are not shifted
    y_shifts
        The absolute shifts in position in vertical/y direction per row of
        subplots; if `None`, the rows are not shifted
        
    Returns
    -------
    A pair of the created :class:`~matplotlib.figure.Figure` and an 2d\
    :class:`~numpy.ndarray` of :class:`~matplotlib.axes.Axes`.
    
    """
  
    if x_padding is not None:
        wspace = x_padding / axsize[0]
    if y_padding is not None:
        hspace = y_padding / axsize[1]

    if width_ratios is None:
        effective_n_x = n_x
        effective_wspace = wspace
    else:
        effective_n_x = sum(width_ratios) / max(width_ratios)
        effective_wspace = wspace * n_x / effective_n_x
        
    if height_ratios is None:
        effective_n_y = n_y
        effective_hspace = hspace
    else:
        effective_n_y = sum(height_ratios) / max(height_ratios)
        effective_hspace = hspace * n_y / effective_n_y
    
    # find the values of the figure size which will keep the axis size identical for all numbers of columns and rows...
    fig_height = axsize[1] * (effective_n_y + hspace * (n_y - 1))
    fig_width = axsize[0] * (effective_n_x + wspace * (n_x - 1))
    
    top = 1.0
    if title != None:
        title_space = 0.75
        fig_height += title_space
        top = 1 - title_space / fig_height
    
    fig, axs = plt.subplots(n_y,n_x,figsize=(fig_width,fig_height), squeeze=False, sharex=sharex, sharey=sharey, gridspec_kw={'wspace':effective_wspace,'hspace':effective_hspace,'left':0,'right':1,'top':top,'bottom':0,'width_ratios': width_ratios,'height_ratios': height_ratios})
    
    if title is not None:
        fig.suptitle(title, fontsize=16, y=1)
    
    if x_shifts is not None or y_shifts is not None:
        if x_shifts is None:
            x_shifts = [0.0] * n_x
        else:
            y_shifts = [0.0] * n_y
        for i_x in range(n_x):
            for i_y in range(n_y):
                [left,bottom,width,height] = axs[i_y,i_x].get_position().bounds
                axs[i_y,i_x].set_position([left+x_shifts[i_x],bottom+y_shifts[i_y],width,height])
    
    return fig, axs

def _add_legend_or_colorbars(fig, axs, colors, cmap=None, min_max=None, scale_legend=1.0):
    
    if cmap is None:
        axs[0,-1].legend(handles=[mpatches.Patch(color=color, label=ind) for (ind, color) in colors.items() ],
            bbox_to_anchor=(1, 1), loc='upper left', ncol=1)
        
    elif cmap is not None:
        rel_dpi_factor = matplotlib.rcParams['figure.dpi'] / 72
        height_pxl = 200 * rel_dpi_factor * scale_legend
        width_pxl = 15 * rel_dpi_factor * scale_legend
        offset_top_pxl = 0 * rel_dpi_factor * scale_legend
        offset_left_pxl = 10 * rel_dpi_factor * scale_legend

        for irow in range(axs.shape[0]):
            for jcol in range(axs.shape[1]):
                ax = axs[irow,jcol]
                left,bottom = fig.transFigure.inverted().transform(ax.transAxes.transform((1,1))+np.array([offset_left_pxl,-offset_top_pxl-height_pxl]))
                width,height = fig.transFigure.inverted().transform(fig.transFigure.transform((0,0))+np.array([width_pxl,height_pxl]))
                cax = fig.add_axes((left, bottom, width, height))
                norm = Normalize(vmin=min_max[irow,jcol,0], vmax=min_max[irow,jcol,1])
                fig.colorbar(ScalarMappable(norm=norm, cmap=cmap), cax=cax)

def scatter(
    adata,
    keys,
    position_key=('x','y'),
    group_key=None,
    colors=None,
    show_only=None,
    axsize=(5,5),
    padding=0.5,
    margin=0.0,
    sharex=False,
    sharey=False,
    share_scaling=True,
    n_cols=1,
    joint=True,
    method_labels=None,
    counts_location=None,
    compositional=False,
    normalize=False,
    point_size=3,
    cmap=None,
    cmap_vmin_vmax=None,
    legend=True,
    on_data_legend=None,
    title=None,
    render=True,
    rasterized=True,
    background_color=None,
    grid=False,
    noticks=False,
    axes_labels=None,
    ax=None,
):
    
    """\
    Scatter plots of annotation.
    
    Parameters
    ----------
    adata
        An :class:`~anndata.AnnData` including annotation in `.obs` and/or
        `.obsm`. Can also be a mapping of labels to :class:`~anndata.AnnData`
        to specify multiple datasets. The :class:`~anndata.AnnData` instances
        can be replaced also by :class:`~pandas.DataFrame`, which are then
        treated like the `.obs` of an :class:`~anndata.AnnData`.
    keys
        The `.obs`/`.obsm` annotation keys to compare. Can be a single string,
        or a list of strings, or a mapping of the labels of `adata` to strings
        or lists of strings. In the list or mapping variant, categorical `.obs`
        keys can be replaced by list-likes of numerical `.obs` keys or gene
        names can be used.
    position_key
        The `.obsm` key or array-like of `.obs` keys with the position space
        coordinates.
    group_key
        An `.obs` key with categorical group information to split `adata` prior
        to plotting. This works only if `adata` is a single
        :class:`~anndata.AnnData` instance.
    colors
        A mapping of annotation values to colors. If `None`, default colors are
        used.
    show_only
        A subset of annotation values to restrict the plotting to.
    axsize
        Tuple of width and height of a single axis. If one of them is `None`,
        it is determined from the aspect ratio of the data. If it is a single
        scalar value, then this is interpreted as a conversion factor from
        data units to axsize units and `share_scaling` is ignored.
    padding
        The absolute padding between the plots.
    margin
        The absolute margin between the outermost data points and the boundary
        of the plot
    sharex
        Whether to use common x axis for all axes.
    sharey
        Whether to use common y axis for all axes.
    share_scaling
        Whether to have the units in all plots be of the same size in pixels
    n_cols
        Number of "columns" to plot: If larger than 1 splits columns of plots
        into `n_cols` columns.
    joint
        Whether to plot only one scatter plot with all annotation categories or
        only the scatter plots with one annotation category per plot. If
        `None`, plot both.
    method_labels
        A mapping from the strings in `keys` and `basis_keys` to (shorter)
        names to show in the plot.
    counts_location
        A string or tuple specifying where the count matrix is stored, e.g.
        `'X'`, `('raw','X')`, `('raw','obsm','my_counts_key')`,
        `('layer','my_counts_key')`, ... For details see
        :func:`~tacco.get.counts`.
    compositional
        Whether the annotation is to be interpreted as compositional data or as
        arbitrary numbers. Compositional data is normalized to sum to 1 per
        observation. Can also be 'catsize', which rescales the compositions by
        the average observed size of an annotation category in terms of the
        contents of `.X`, e.g. reads.
    normalize
        Whether to shift the data to non-negative values and normalize them by
        their maximum.
    point_size
        The size of the points in the plot. Like in matplotlib, this is a
        measure of point area and provided in units of "squared points"
        corresponding to (1/72)^2 inch^2 = (25.4/72)^2 mm^2.
    cmap
        A string/colormap to override the `colors` with. Makes sense mostly for
        numeric data.
    cmap_vmin_vmax
        A tuple giving the range of values for the colormap.
    legend
        Whether to plot a legend
    on_data_legend
        A mapping from annotation values to (shortened) versions of the labels
        to use for labels on the plot at the center of the annotation;
        annotations not occurring in the mapping are used as is; if `None`, no
        annotation is plotted on the data.
    title
        The title of the figure
    render
        Whether the scatterplot should be custom rendered using the dpi setting
        from matplotlib or plotted using a matplotlib's scatterplot. If `True`,
        the different annotations from the same (and overlapping) positions are
        added up symmetrically, if `False`, they are plottet on top of each
        other using an alpha channel proportional to the weight. `True` also
        has the advantage that only the scatter part of the figure will be
        exported as pixelated version if the plot is exported as vectorgraphic,
        with the rest like labels and axes being exported as a vectorgraphic.
        This parameter provides control over the type of color averaging in the
        process of the rendering by specifying one of the modes available in
        :func:`~tacco.plots.mix_base_colors`, e.g. "xyv" or "rgb", with "xyv"
        being equivalent to setting `True`.
    rasterized
        Whether to rasterize the interior of the plot, even when exported later
        as vectorgraphic. This leads to much smaller plots for many (data)
        points. `rasterized==False` is incompatible with `render==True` or
        string.
        This parameter provides experimental support for plotting pie charts
        per dot via ´rasterized=="pie"´ and ´render==False´. This is much
        slower, so only usable for very few points.
    background_color
        The background color to draw the points on.
    grid
        Whether to draw a grid
    noticks
        Whether to switch off ticks on the axes.
    axes_labels
        Labels to write on the axes as an list-like of the two labels.
    ax
        The 2d array of :class:`~matplotlib.axes.Axes` instances to plot on.
        The array dimensions have to agree with the number of axes which would
        be created automatically if `ax` was not supplied. If it is a single
        instance it is treated as a 1x1 array.
        
    Returns
    -------
    A :class:`~matplotlib.figure.Figure`.
    
    """
    
    if group_key is not None:
        if not isinstance(adata, ad.AnnData):
            raise ValueError(f'The `group_key` {group_key!r} is not `None`, but `adata` is not a single `AnnData` instance!')
        if group_key not in adata.obs:
            raise ValueError(f'The `group_key` {group_key!r} is not available in `adata.obs`!')
        adata = { c: adata[df.index] for c, df in adata.obs.groupby(group_key) if len(df) > 0 }
    
    typing_data, adatas, methods, types, colors, coords = _validate_scatter_args(adata, position_key, keys, colors, show_only, method_labels=method_labels, counts_location=counts_location, compositional=compositional)
    n_solutions, n_samples, n_types = len(typing_data), len(adatas), len(types)
    
    if joint is None:
        n_types = n_types + 1
    elif joint:
        n_types = 1
        n_cols = 1
    n_x = n_solutions*n_cols
    n_y = (n_types + n_cols - 1) // n_cols
    
    axsize = np.array(axsize)
    scale = None
    
    # special treatment of various semi-automatic axsizes
    if len(axsize.shape) == 0 or axsize[0] is None or axsize[1] is None or ax is not None or share_scaling:
        minxs, maxxs = [], []
        minys, maxys = [], []    
        for i_sample, sample in enumerate(typing_data['sample'].unique()):
            minx,maxx = get_min_max(coords[sample].iloc[:,0], log=False)
            miny,maxy = get_min_max(coords[sample].iloc[:,1], log=False)
            minxs.append(minx), maxxs.append(maxx)
            minys.append(miny), maxys.append(maxy)
        minxs, maxxs = np.array(minxs), np.array(maxxs)
        minys, maxys = np.array(minys), np.array(maxys)
        sizexs = maxxs-minxs
        sizeys = maxys-minys
        maxsizex = sizexs.max()
        maxsizey = sizeys.max()
        minx, maxx = minxs.min(), maxxs.max()
        miny, maxy = minys.min(), maxys.max()
        sizex = maxx-minx
        sizey = maxy-miny
        
        if ax is not None: # If `ax` is given, use it. This also ensures that axs is always defined below
            if isinstance(ax, matplotlib.axes.Axes):
                axs = np.array([[ax]])
            else:
                axs = ax
            if axs.shape != (n_y,n_x):
                raise ValueError(f'The `ax` argument got the wrong shape of axes: needed is {(n_y,n_x)!r} supplied was {axs.shape!r}!')
            axsize = axs[0,0].get_window_extent().transformed(axs[0,0].get_figure().dpi_scale_trans.inverted()).size
            
        elif len(axsize.shape) == 0: # if it is a single scalar value, use that as a global scale
            # this also implies that all axes have to have a common scaling
            if not (sharex or sharey):
                share_scaling = True
            
            if sharex:
                axsizex = sizex * axsize
            else:
                axsizex = maxsizex * axsize
            if sharey:
                axsizey = sizey * axsize
            else:
                axsizey = maxsizey * axsize
                
            axsize = np.array([axsizex, axsizey])
            
        elif axsize[0] is None or axsize[1] is None:
            
            # determine the missing axsize
            if axsize[0] is None and axsize[1] is None:
                raise ValueError(f'The parameter `axsize` got `(None, None)`, while only one of the entries of the tuple can be `None`!')
                
            aspect_ratio = (sizey if sharey else sizeys) / (sizex if sharex else sizexs)
            
            if not (sharex and sharey): # if aspect_ratio is not yet fixed and still a vector
            
                #aspect_ratio = max(aspect_ratio)
                #aspect_ratio = min(aspect_ratio)
                
                #np.argmax(np.abs(aspect_ratio - 1))
                
                if sharex: # x is fixed: make y as large as needed
                    aspect_ratio = max(aspect_ratio)
                elif sharey: # y is fixed: make x as large as needed
                    aspect_ratio = min(aspect_ratio)
                else:
                    aspect_ratio = aspect_ratio.mean() # we are free to take the one which fits best - in some arbitrary sense...
                
            if axsize[0] is None:
                axsize[0] = axsize[1] / aspect_ratio
            else: # axsize[1] is None
                axsize[1] = axsize[0] * aspect_ratio
        
        if share_scaling:
            scale_0 = (axsize[0]-2*margin) / maxsizex
            scale_1 = (axsize[1]-2*margin) / maxsizey
            scale = min(scale_0, scale_1)
    
    aspect_ratio = axsize[1] / axsize[0]
    
    wspace, hspace = padding / axsize
    
    if ax is None:
        fig, axs = subplots(n_x,n_y,axsize=axsize,sharex=sharex,sharey=sharey,wspace=wspace,hspace=hspace,title=title)
    else:
        fig = axs[0,0].get_figure()
    
    min_max = np.zeros((*axs.shape,2))
    for i_sample, sample in enumerate(typing_data['sample'].unique()):
        sub = typing_data[typing_data['sample']==sample]
        n_methods = len(sub)
        ax = axs[:,(i_sample*n_methods*n_cols):((i_sample+1)*n_methods*n_cols)]
        _min_max = None if cmap is None else min_max[:,(i_sample*n_methods*n_cols):((i_sample+1)*n_methods*n_cols)]
        spatial_distribution_plot(sub['data'], coords[sample], colors, axs=ax, n_cols=n_cols, joint=joint, normalize=normalize, point_size=point_size, cmap=cmap, cmap_vmin_vmax=cmap_vmin_vmax, out_min_max=_min_max, scale=scale, grid=grid, margin=margin, render=render, rasterized=rasterized, noticks=noticks, axes_labels=axes_labels, on_data_legend=on_data_legend)
        if background_color is not None:
            for _ax in ax.flatten():
                _ax.set_facecolor(background_color)
    
    if share_scaling:
    
        fig.canvas.draw() # make the transformations have well defined values
        
        ax_heights = []
        ax_widths = []
        for _ax in axs.flatten():
            ax_x_low,ax_y_low = _ax.transData.inverted().transform(_ax.transAxes.transform((0,0)))
            ax_x_high,ax_y_high = _ax.transData.inverted().transform(_ax.transAxes.transform((1,1)))
            ax_height,ax_width = (ax_y_high-ax_y_low),(ax_x_high-ax_x_low)
            ax_heights.append(ax_height)
            ax_widths.append(ax_width)
        max_height = max(ax_heights)
        max_width = max(ax_widths)

        for _ax in axs.flatten():
            ax_x_low,ax_y_low = _ax.transData.inverted().transform(_ax.transAxes.transform((0,0)))
            ax_x_high,ax_y_high = _ax.transData.inverted().transform(_ax.transAxes.transform((1,1)))
            axes_ratio = (ax_y_high-ax_y_low)/(ax_x_high-ax_x_low)

            delta = max_height - (ax_y_high - ax_y_low)
            ax_y_high += delta / 2
            ax_y_low -= delta / 2
            _ax.set_ylim((ax_y_low,ax_y_high))

            delta = max_width - (ax_x_high - ax_x_low)
            ax_x_high += delta / 2
            ax_x_low -= delta / 2
            _ax.set_xlim((ax_x_low,ax_x_high))

    fig.canvas.draw() # make the transformations have well defined values
    
    # make axes use the full axsize
    for _ax in axs.flatten():
        ax_x_low,ax_y_low = _ax.transData.inverted().transform(_ax.transAxes.transform((0,0)))
        ax_x_high,ax_y_high = _ax.transData.inverted().transform(_ax.transAxes.transform((1,1)))
        axes_ratio = (ax_y_high-ax_y_low)/(ax_x_high-ax_x_low)
        if aspect_ratio > axes_ratio:
            # increase height
            delta = (aspect_ratio / axes_ratio - 1) * (ax_y_high - ax_y_low)
            ax_y_high += delta / 2
            ax_y_low -= delta / 2
            _ax.set_ylim((ax_y_low,ax_y_high))
        else:
            # increase width
            delta = (axes_ratio / aspect_ratio - 1) * (ax_x_high - ax_x_low)
            ax_x_high += delta / 2
            ax_x_low -= delta / 2
            _ax.set_xlim((ax_x_low,ax_x_high))
    
    if legend:
        _add_legend_or_colorbars(fig, axs, colors, cmap=cmap, min_max=min_max)
    
    return fig


def get_cellsize(adata, key='OTT', reference_adata=None, reference_key='OTT', pfn_key=None, counts_location=None):
    if isinstance(key, pd.DataFrame):
        solution = key
    elif key in adata.obsm:
        solution = adata.obsm[key]
    else:
        solution = pd.get_dummies(adata.obs[key])
    
    if reference_adata is not None:
        counts = get.counts(adata, counts_location=counts_location)[adata.obs.index,adata.var.index]
        pfn = adata.var.index.isin(reference_adata.var.index).astype(float) # basic level of rescaling: using common genes only
        bare_read_count = pd.Series(np.array(counts.X @ pfn).flatten(), index=adata.obs.index)
        #print('get_cell_size0:',pfn.sum())
        if pfn_key is not None:
            if pfn_key in adata.varm and key in adata.varm[pfn_key]:
                pfn = pfn * adata.varm[pfn_key][key].to_numpy()
                #print('multiplying', adata.varm[pfn_key][key].to_numpy())
            if pfn_key in reference_adata.varm and reference_key in reference_adata.varm[pfn_key]:
                pfn = pfn / reference_adata.varm[pfn_key][reference_key].reindex(adata.var.index).to_numpy()
                #print('dividing', reference_adata.varm[pfn_key][reference_key].reindex(adata.var.index).to_numpy())
        #print('get_cell_size1:',pfn.sum())
        pfn = np.nan_to_num(pfn).astype(float)
        #print('get_cell_size2:',pfn.sum())
        read_count = pd.Series(np.array(counts.X @ pfn).flatten(), index=adata.obs.index)
        read_count *= bare_read_count.sum() / read_count.sum()
    else:
        counts = get.counts(adata, counts_location=counts_location)[adata.obs.index]
        read_count = pd.Series(np.array(counts.X.sum(axis=1)).flatten(), index=adata.obs.index)
    
    by_cells = solution / solution.sum(axis=1).to_numpy()[:,None]
    by_reads = by_cells * read_count[solution.index].to_numpy()[:,None]
    cellsize = by_reads.sum(axis=0) / by_cells.sum(axis=0).to_numpy()
    
    return cellsize

def _validate_cross_args(adata, keys, colors, show_only, basis_adata, basis_keys, reads, method_labels, counts_location=None):
    typing_data, adatas, methods, types, colors = _validate_args(adata, keys, colors, show_only, reads=reads, method_labels=method_labels, counts_location=counts_location)
    
    if basis_adata is None:
        basis_adata = adata
    if basis_keys is None:
        basis_keys = keys
    
    basis_typing_data, basis_adatas, basis_methods, basis_types, basis_colors = _validate_args(basis_adata, basis_keys, colors, show_only, reads=reads, method_labels=method_labels, counts_location=counts_location)
    
    if len(types.intersection(basis_types)) != len(types):
        raise Exception('The available types provided in adata %s and basis adata %s dont agree!' % (types, basis_types))
    
    return typing_data, adatas, methods, types, colors, basis_typing_data, basis_adatas, basis_methods

def _cross_scatter(data, axs, colors, marker='o'):
    if data.shape != axs.shape:
        raise Exception('The shapes of the data and axs arrays %s and %s dont agree!' % (data.shape, axs.shape))
    n_y, n_x = data.shape
    for x in range(n_x):
        for y in range(n_y):
            x_data,y_data = data.iloc[y,x]
            ax = axs[y,x]
            ax.grid(True)
            common = colors.index
            common = common.intersection(x_data[x_data > 0].index)
            common = common.intersection(y_data[y_data > 0].index)
            
            _scatter_plot(ax,x=x_data[common],y=y_data[common],
                       colors=colors[common],
                       marker=marker, log=True)
            if y == n_y-1:
                ax.set_xlabel(data.columns[x],rotation=15,va='top',ha='right')
                ax.tick_params(axis='x', rotation=45, which='both')
            if x == 0:
                ax.set_ylabel(data.index[y],rotation='horizontal',ha='right')
                        
    axs[0,n_x-1].legend(handles=[mpatches.Patch(color=color, label=ind) for (ind, color) in colors.items() ],
              bbox_to_anchor=(1, 1), loc='upper left', ncol=1)

def cellsize(
    adata,
    keys,
    colors=None,
    show_only=None,
    axsize=(1.5,1.5),
    basis_adata=None,
    basis_keys=None,
    pfn_key=None,
    use_reference=False,
    method_labels=None,
    counts_location=None
):

    """\
    Scatter plots of average cell sizes in an annotation category in the whole
    dataset against some reference dataset. These cell sizes are given as the
    average number of counts per unit cell.
    
    Parameters
    ----------
    adata
        An :class:`~anndata.AnnData` including annotation in `.obs` and/or `.obsm`.
        Can also be a mapping of labels to :class:`~anndata.AnnData` to specify
        multiple datasets.
    keys
        The `.obs`/`.obsm` annotation keys to compare. Can be a single string, or a
        list of strings, or a mapping of the labels of `adata` to strings or lists
        of strings.
    colors
        A mapping of annotation values to colors. If `None`, default colors are used.
    show_only
        A subset of annotation values to restrict the plotting to.
    axsize
        Tuple of width and size of a single axis.
    basis_adata
        like `adata`, but for reference data.
    basis_keys
        like `adata`, but for reference data.
    pfn_key
        A `.varm` key containing platform normalization factors. Ignored if platform
        normalization is not requested via `use_reference`.
    use_reference
        Whether and how to use reference data for the determination of cell sizes in
        non-reference data.
        Can be a single choice or an array of choices. Possible choices are:
        
        - `False`: Dont use the reference
        - `True`: Determine the cell sizes on the set of common genes with the
          reference data
        - 'pfn': Like `True`, but use additionally platform normalization factors.
        - `None`: Use all settings which are available, i.e. includes `True`, `False,
          and if `pfn_key!=None` also 'pfn'.
    method_labels
        A mapping from the strings in `keys` and `basis_keys` to (shorter) names
        to show in the plot.
    counts_location
        A string or tuple specifying where the count matrix is stored, e.g. `'X'`,
        `('raw','X')`, `('raw','obsm','my_counts_key')`, `('layer','my_counts_key')`,
        ... For details see :func:`~tacco.get.counts`.
        
    Returns
    -------
    A :class:`~matplotlib.figure.Figure`.
    
    """
    
    typing_data, adatas, methods, types, colors, basis_typing_data, basis_adatas, basis_methods = _validate_cross_args(adata, keys, colors, show_only, basis_adata, basis_keys, reads=False, method_labels=method_labels, counts_location=counts_location)
    n_solutions, n_samples, n_types = len(typing_data), len(adatas), len(types)
    n_solutions_basis, n_samples_basis = len(basis_typing_data), len(basis_adatas)
    
    possible_use_refs = [False,True,'pfn']
    if isinstance(use_reference, (str,bool)):
        use_reference = [use_reference]
    elif use_reference is None:
        if pfn_key is None:
            use_reference = [False,True]
        else:
            use_reference = possible_use_refs
    if 'pfn' in use_reference and pfn_key is None:
        raise Exception('If platform normalization is desired (as indicated by specifying "pfn" in the "use_reference" argument), the argument "pfn_key" has to be specified!')
    use_ref_labels = []
    for use_ref in use_reference:
        if use_ref not in possible_use_refs:
            raise Exception('The argument use_reference got the unknown value "%s"! Only %s are possible.' % (use_ref, possible_use_refs))
        if isinstance(use_ref,str):
            if use_ref == 'pfn':
                use_ref_label = ' - platform corrected'
        elif use_ref: # ignore pfn for cellsize estimation
            use_ref_label = ' - common genes'
        else: # ignore reference for cellsize estimation
            use_ref_label = ''
        use_ref_labels.append(use_ref_label)
        
    n_x = n_solutions
    n_y = n_solutions_basis * len(use_reference)
    
    fig, axs = subplots(n_x,n_y,axsize=axsize,sharex='all',sharey='all',wspace=0,hspace=0)
    
    data = pd.DataFrame(dtype=object,columns=typing_data.index,index=[basis_typing_data.index[y % n_solutions_basis] + use_ref_labels[y // n_solutions_basis] for y in range(n_y)])

    for x in range(n_x):
        for y in range(n_y):
            x_key = typing_data['method'].iloc[x]
            x_adata = adatas[typing_data['sample'].iloc[x]]
            
            y_key = basis_typing_data['method'].iloc[y % n_solutions_basis]
            y_adata = basis_adatas[basis_typing_data['sample'].iloc[y % n_solutions_basis]]
            
            use_ref = use_reference[y // n_solutions_basis]
            if isinstance(use_ref,str):
                if use_ref == 'pfn':
                    cell_sizes_x = get_cellsize(x_adata, x_key, reference_adata=y_adata, reference_key=y_key, counts_location=counts_location)#, pfn_key=None) # no platform normalization effects, only on common genes
                    cell_sizes_y = get_cellsize(y_adata, y_key, reference_adata=x_adata, reference_key=x_key, counts_location=counts_location, pfn_key=pfn_key) # include platform normalization effects for both adatas in these size estimations
            elif use_ref: # ignore pfn for cellsize estimation
                cell_sizes_x = get_cellsize(x_adata, x_key, reference_adata=y_adata, counts_location=counts_location) # no platform normalization effects, only on common genes
                cell_sizes_y = get_cellsize(y_adata, y_key, reference_adata=x_adata, counts_location=counts_location) # include platform normalization effects for both adatas in these size estimations
            else: # ignore reference for cellsize estimation
                cell_sizes_x = get_cellsize(x_adata, x_key, counts_location=counts_location)
                cell_sizes_y = get_cellsize(y_adata, y_key, counts_location=counts_location)
                
            data.iloc[y,x] = (cell_sizes_x,cell_sizes_y)
    
    _cross_scatter(data, axs, colors, marker='x')
    
    return fig

def _validate_frequency_args(adata, keys, colors, show_only, basis_adata, basis_keys, reads, method_labels=None, counts_location=None):
    typing_data, adatas, methods, types, _colors = _validate_args(adata, keys, colors, show_only, reads, method_labels=method_labels, counts_location=counts_location)
    
    if basis_adata is not None:
        basis_typing_data, basis_adatas, basis_methods, basis_types, _colors = _validate_args(basis_adata, basis_keys, colors, show_only, reads, method_labels=method_labels, counts_location=counts_location)
    
        if len(types.intersection(basis_types)) != len(types):
            raise Exception('The available types provided in adata %s and basis adata %s dont agree!' % (types, basis_types))
            
        typing_data = pd.concat([basis_typing_data, typing_data])
    
    return typing_data, adatas, methods, types, _colors

def frequency_bar(
    adata,
    keys,
    colors=None,
    show_only=None,
    axsize=None,
    basis_adata=None,
    basis_keys=None,
    horizontal=True,
    reads=False,
    method_labels=None,
    counts_location=None,
    ax=None,
):

    """\
    Bar plots of the total frequency of annotation in the whole dataset against
    some reference dataset.
    
    Parameters
    ----------
    adata
        An :class:`~anndata.AnnData` including annotation in `.obs` and/or
        `.obsm`. Can also be a mapping of labels to :class:`~anndata.AnnData`
        to specify multiple datasets. The :class:`~anndata.AnnData` instances
        can be replaced also by :class:`~pandas.DataFrame`, which are then
        treated like the `.obs` of an :class:`~anndata.AnnData`.
    keys
        The `.obs`/`.obsm` annotation keys to compare. Can be a single string,
        or a list of strings, or a mapping of the labels of `adata` to strings
        or lists of strings.
    colors
        A mapping of annotation values to colors. If `None`, default colors are
        used.
    show_only
        A subset of annotation values to restrict the plotting to.
    axsize
        Tuple of width and size of a single axis. If `None`, a default size is
        chosen automatically.
    basis_adata
        like `adata`, but for reference data.
    basis_keys
        like `adata`, but for reference data.
    horizontal
        Whether to draw the bars horizontally.
    reads
        Whether to work with read or cell count fractions.
    method_labels
        A mapping from the strings in `keys` and `basis_keys` to (shorter)
        names to show in the plot.
    counts_location
        A string or tuple specifying where the count matrix is stored, e.g.
        `'X'`, `('raw','X')`, `('raw','obsm','my_counts_key')`,
        `('layer','my_counts_key')`, ... For details see
        :func:`~tacco.get.counts`.
    ax
        The :class:`~matplotlib.axes.Axes` to plot on. If `None`, creates a
        fresh figure for plotting.
        
    Returns
    -------
    The :class:`~matplotlib.figure.Figure` containing the plot.
    
    """
    
    typing_data, adatas, methods, types, colors = _validate_frequency_args(adata, keys, colors, show_only, basis_adata, basis_keys, reads, method_labels=method_labels, counts_location=counts_location)
        
    n_solutions = len(typing_data)
    
    if ax is None:
        if axsize is None:
            if horizontal:
                axsize=(9.1,0.7*n_solutions)
            else:
                axsize=(0.7*n_solutions,5)
        fig, axs = subplots(axsize=axsize)
        ax = axs[0,0]
    else:
        fig = ax.figure
    
    type_freqs = pd.DataFrame({ method: data.sum(axis=0) for method, data in typing_data['data'].items() })
    
    norm_tf = (type_freqs/type_freqs.sum(axis=0)).T
    norm_tf = norm_tf.fillna(0)
    
    _composition_bar(norm_tf, colors, horizontal=horizontal, ax=ax)

    return fig

def frequency(
    adata,
    keys,
    colors=None,
    show_only=None,
    axsize=(1.5,1.5),
    basis_adata=None,
    basis_keys=None,
    reads=False,
    method_labels=None,
    counts_location=None
):

    """\
    Scatter plots of the total frequency of annotation in the whole dataset
    against some reference dataset.
    
    Parameters
    ----------
    adata
        An :class:`~anndata.AnnData` including annotation in `.obs` and/or
        `.obsm`. Can also be a mapping of labels to :class:`~anndata.AnnData`
        to specify multiple datasets. The :class:`~anndata.AnnData` instances
        can be replaced also by :class:`~pandas.DataFrame`, which are then
        treated like the `.obs` of an :class:`~anndata.AnnData`.
    keys
        The `.obs`/`.obsm` annotation keys to compare. Can be a single string,
        or a list of strings, or a mapping of the labels of `adata` to strings
        or lists of strings.
    colors
        A mapping of annotation values to colors. If `None`, default colors are
        used.
    show_only
        A subset of annotation values to restrict the plotting to.
    axsize
        Tuple of width and size of a single axis.
    basis_adata
        like `adata`, but for reference data.
    basis_keys
        like `adata`, but for reference data.
    reads
        Whether to work with read or cell count fractions.
    method_labels
        A mapping from the strings in `keys` and `basis_keys` to (shorter)
        names to show in the plot.
    counts_location
        A string or tuple specifying where the count matrix is stored, e.g.
        `'X'`, `('raw','X')`, `('raw','obsm','my_counts_key')`,
        `('layer','my_counts_key')`, ... For details see
        :func:`~tacco.get.counts`.
        
    Returns
    -------
    A :class:`~matplotlib.figure.Figure`.
    
    """

    typing_data, adatas, methods, types, colors, basis_typing_data, basis_adatas, basis_methods = _validate_cross_args(adata, keys, colors, show_only, basis_adata, basis_keys, reads=reads, method_labels=method_labels, counts_location=counts_location)
    n_solutions, n_samples, n_types = len(typing_data), len(adatas), len(types)
    n_solutions_basis, n_samples_basis = len(basis_typing_data), len(basis_adatas)
    
    n_x = n_solutions
    n_y = n_solutions_basis
    
    fig, axs = subplots(n_x,n_y,axsize=axsize,sharex='all',sharey='all',wspace=0,hspace=0)
    
    type_freqs = pd.DataFrame({ method: data.sum(axis=0) for method, data in typing_data['data'].items() })
    type_freqs = type_freqs / type_freqs.sum(axis=0) # normalize freqs
    
    basis_type_freqs = pd.DataFrame({ method: data.sum(axis=0) for method, data in basis_typing_data['data'].items() })
    basis_type_freqs = basis_type_freqs / basis_type_freqs.sum(axis=0) # normalize freqs
    
    data = pd.DataFrame(dtype=object,columns=typing_data.index,index=basis_typing_data.index)
    
    for x in range(n_x):
        for y in range(n_y):
            x_data = type_freqs.iloc[:,x]
            y_data = basis_type_freqs.iloc[:,y]
            data.iloc[y,x] = (x_data,y_data)
        
    _cross_scatter(data, axs, colors, marker='o')

    return fig

def comparison(
    adata,
    keys,
    colors=None,
    show_only=None,
    axsize=(2.0,2.0),
    basis_adata=None,
    basis_keys=None,
    method_labels=None,
    counts_location=None,
    point_size=2,
    joint=None
):

    """\
    Scatterplots of the annotation fractions of different annotations, e.g. of
    different annotation methods or of methods and a ground truth.
    
    Parameters
    ----------
    adata
        An :class:`~anndata.AnnData` including annotation in `.obs` and/or `.obsm`.
        Can also be a mapping of labels to :class:`~anndata.AnnData` to specify
        multiple datasets.
    keys
        The `.obs`/`.obsm` annotation keys to compare. Can be a single string, or a
        list of strings, or a mapping of the labels of `adata` to strings or lists
        of strings.
    colors
        A mapping of annotation values to colors. If `None`, default colors are used.
    show_only
        A subset of annotation values to restrict the plotting to.
    axsize
        Tuple of width and size of a single axis.
    basis_adata
        like `adata`, but for reference data.
    basis_keys
        like `adata`, but for reference data.
    method_labels
        A mapping from the strings in `keys` and `basis_keys` to (shorter) names
        to show in the plot.
    counts_location
        A string or tuple specifying where the count matrix is stored, e.g. `'X'`,
        `('raw','X')`, `('raw','obsm','my_counts_key')`, `('layer','my_counts_key')`,
        ... For details see :func:`~tacco.get.counts`.
    point_size
        The size of the points in the plot.
    joint
        Whether to plot only one scatter plot with all annotation categories or only
        the scatter plots with one annotation category per plot. If `None`, plot both.
        
    Returns
    -------
    A :class:`~matplotlib.figure.Figure`.
    
    """

    typing_data, adatas, methods, types, colors, basis_typing_data, basis_adatas, basis_methods = _validate_cross_args(adata, keys, colors, show_only, basis_adata, basis_keys, reads=False, method_labels=method_labels, counts_location=counts_location)
    n_solutions, n_samples, n_types = len(typing_data), len(adatas), len(types)
    n_solutions_basis, n_samples_basis = len(basis_typing_data), len(basis_adatas)
    
    n_types = len(colors.index)
    n_x = n_solutions
    if joint is None:
        n_y = n_types + 1
    elif joint:
        n_y = 1
    else:
        n_y = n_types
    n_y = n_solutions_basis * n_y
    
    fig, axs = subplots(n_x,n_y,axsize=axsize,sharex='all',sharey='all',wspace=0.25,hspace=0.25)
    for ax in axs.flatten():
        #ax.grid(False)
        #ax.axes.xaxis.set_ticks([])
        #ax.axes.yaxis.set_ticks([])
        pass
    
    for i,basis_name in enumerate(basis_typing_data.index):
        basis_sample, basis_type_freqs = basis_typing_data.loc[basis_name,['sample', 'data']]
        basis_type_freqs /= basis_type_freqs.sum(axis=1).to_numpy()[:,None]
        
        for j,name in enumerate(typing_data.index):
        
            sample, type_freqs = typing_data.loc[name,['sample', 'data']]
            type_freqs /= type_freqs.sum(axis=1).to_numpy()[:,None]
            
            common = type_freqs.index.intersection(basis_type_freqs.index)
            type_freqs = type_freqs.loc[common]
            _basis_type_freqs = basis_type_freqs.loc[common]
            
            def plotit(t, ax):
                x = type_freqs[t]
                y = _basis_type_freqs[t]
                ax.scatter(x, y, color=colors[t], s=point_size, label=t, edgecolors='none')
            
            if joint is None:
                joint_ax = axs[i * (n_types + 1),j]
                for it,t in enumerate(colors.index):
                    plotit(t, joint_ax)
                    ax = axs[i * (n_types + 1) + it + 1, j]
                    plotit(t, ax)
                    pc, x2 = _correlations(type_freqs[t], _basis_type_freqs[t], log=False)
                    #ax.set_title('%s\nr=%.2f $\\chi^2_m$=%.2f'%(t, pc, x2))
                    ax.set_title('r=%.2f $\\chi^2_m$=%.2f'%(pc, x2))
                pc, x2 = _correlations(type_freqs.to_numpy().flatten(), _basis_type_freqs.to_numpy().flatten(), log=False)
                joint_ax.set_title('r=%.2f $\\chi^2_m$=%.2f'%(pc, x2))
            elif joint:
                joint_ax = axs[i,j]
                for t in colors.index:
                    plotit(t, joint_ax)
                pc, x2 = _correlations(type_freqs.to_numpy().flatten(), _basis_type_freqs.to_numpy().flatten(), log=False)
                joint_ax.set_title('r=%.2f $\\chi^2_m$=%.2f'%(pc, x2))
            else:
                for it,t in enumerate(colors.index):
                    ax = axs[i * n_types + it, j]
                    plotit(t, ax)
                    pc, x2 = _correlations(type_freqs[t], _basis_type_freqs[t], log=False)
                    #ax.set_title('%s\nr=%.2f $\\chi^2_m$=%.2f'%(t, pc, x2))
                    ax.set_title('r=%.2f $\\chi^2_m$=%.2f'%(pc, x2))
            
    for ax,name in zip(axs[n_y-1,:],typing_data.index):
        ax.set_xlabel(name,rotation=15,va='top',ha='right')
        #ax.tick_params(axis='x', rotation=45, which='both')
    for ax,basis_name in zip(axs[:,0],list(basis_typing_data.index) * (n_y//n_solutions_basis)):
        ax.set_ylabel(basis_name,rotation='horizontal',ha='right')
                        
    axs[0,n_x-1].legend(handles=[mpatches.Patch(color=color, label=ind) for (ind, color) in colors.items() ],
              bbox_to_anchor=(1, 1), loc='upper left', ncol=1)
            
            #typing = pd.Series({'%s (%s)' % (name, basis_name): pd.get_dummies(mdata.obs['split_type'])})
            
    return fig


def compositions(
    adata,
    value_key,
    group_key,
    basis_adata=None,
    basis_value_key=None,
    basis_group_key=None,
    fillna=None,
    restrict_groups=None,
    restrict_values=None,
    basis_restrict_groups=None,
    basis_restrict_values=None,
    reads=False,
    colors=None,
    horizontal=False,
    axsize=None,
    ax=None,
    legend=True,
):

    """\
    Plot compositions of groups. In contrast to :func:`~tacco.plots.contribution`, compositions
    have to add up to one.
    
    Parameters
    ----------
    adata
        An :class:`~anndata.AnnData` with annotation in `.obs`.
    value_key
        The `.obs` or `.obsm` key with the values to determine the
        enrichment for.
    group_key
        The `.obs` key with categorical group information.
    basis_adata
        Another :class:`~anndata.AnnData` with annotation in `.obs`
        to compare. If `None`, only the `adata` composition is shown.
    basis_value_key
        The `.obs` or `.obsm` key for `basis_adata` with the values
        to determine the enrichment for. If `None`, `value_key` is used.
    basis_group_key
        The `.obs` key with categorical group information for
        `basis_adata`. If `None`, `value_key` is used.
    fillna
        If `None`, observation containing NA in the values are filtered.
        Else, NA values are replaced with this value.
    restrict_groups
        A list-like containing the groups within which the enrichment analysis is
        to be done. If `None`, all groups are included.
    restrict_values
        A list-like containing the values within which the enrichment analysis is
        to be done. If `None`, all values are included. Works only for categorical
        values.
    basis_restrict_groups
        Like `restrict_groups` but for `basis_adata`.
    basis_restrict_values
        Like `restrict_values` but for `basis_adata`.
    reads
        Whether to weight the values by the total count per observation
    colors
        The mapping of value names to colors. If `None`, a set of
        standard colors is used.
    horizontal
        Whether to plot the bar plot horizontally.
    axsize
        Tuple of width and size of a single axis. If `None`, use
        automatic values.
    ax
        The :class:`~matplotlib.axes.Axes` to plot on. If `None`, creates a
        fresh figure for plotting.
    legend
        Whether to include the legend
        
    Returns
    -------
    A :class:`~matplotlib.figure.Figure` if `ax` is `None`, else `None`.
    
    """
    
    compositions = get_compositions(adata=adata, value_key=value_key, group_key=group_key, fillna=fillna, restrict_groups=restrict_groups, restrict_values=restrict_values, reads=reads)
    
    colors, types = _get_colors(colors, compositions.columns)
    
    if basis_adata is not None:
        if basis_value_key is None:
            basis_value_key = value_key
        if basis_group_key is None:
            basis_group_key = group_key
        basis_compositions = get_compositions(adata=basis_adata, value_key=basis_value_key, group_key=basis_group_key, fillna=fillna, restrict_groups=basis_restrict_groups, restrict_values=basis_restrict_values, reads=reads)
    
        basis_compositions.index = basis_compositions.index.astype(str) + ' (reference)'
        compositions = pd.concat([basis_compositions,compositions])
    
    n_solutions = len(compositions.index)
    
    if ax is not None:
        fig = None
    else:
        if axsize is None:
            if horizontal:
                axsize=(9.1,0.7*n_solutions)
            else:
                axsize=(0.7*n_solutions,5)
        fig, axs = subplots(axsize=axsize)
        ax = axs[0,0]
    
    _composition_bar(compositions, colors, horizontal=horizontal, ax=ax, legend=legend)
    
    return fig

def _prep_contributions(
    adata,
    value_key,
    group_key,
    sample_key=None,
    basis_adata=None,
    basis_value_key=None,
    basis_group_key=None,
    basis_sample_key=None,
    fillna=None,
    restrict_groups=None,
    restrict_values=None,
    basis_restrict_groups=None,
    basis_restrict_values=None,
    reduction='sum',
    normalization=None,
    assume_counts=None,
    reads=False,
    colors=None,
):
    """\
    Prepares contribution data.
    
    Parameters
    ----------
    adata
        An :class:`~anndata.AnnData` with annotation in `.obs`.
    value_key
        The `.obs` or `.obsm` key with the values to determine the
        enrichment for.
    group_key
        The `.obs` key with categorical group information.
    sample_key
        The `.obs` key with categorical sample information. If `None`,
        only the aggregated data is plotted. Otherwise the data is aggregated
        per sample and total and per-sample values are plotted.
    basis_adata
        Another :class:`~anndata.AnnData` with annotation in `.obs`
        to compare. If `None`, only the `adata` composition is shown.
    basis_value_key
        The `.obs` or `.obsm` key for `basis_adata` with the values
        to determine the enrichment for. If `None`, `value_key` is used.
    basis_group_key
        The `.obs` key with categorical group information for
        `basis_adata`. If `None`, `value_key` is used.
    basis_sample_key
        The `.obs` key with categorical sample information for
        `basis_adata`. If `None`, `sample_key` is used.
    fillna
        If `None`, observation containing NA in the values are filtered.
        Else, NA values are replaced with this value.
    restrict_groups
        A list-like containing the groups within which the enrichment analysis
        is to be done. If `None`, all groups are included.
    restrict_values
        A list-like containing the values within which the enrichment analysis
        is to be done. If `None`, all values are included. Works only for
        categorical values.
    basis_restrict_groups
        Like `restrict_groups` but for `basis_adata`.
    basis_restrict_values
        Like `restrict_values` but for `basis_adata`.
    reduction
        The reduction to apply on each (group,sample) subset of the data.
        Possible values are:
        
        - 'sum': sum of the values over observations
        - 'mean': mean of the values over observations
        - 'median': median of the values over observations
        - a callable mapping a :class:`~pandas.DataFrame` to its reduced
          counterpart
    
    normalization
        The normalization to apply on each reduced (group,sample) subset of the
        data. Possible values are:
        
        - 'sum': normalize values by their sum (yields fractions)
        - 'percent': like 'sum' scaled by 100 (yields percentages)
        - 'gmean': normalize values by their geometric mean (yields
          contributions which make more sense for enrichments than fractions,
          due to zero-sum issue; see :func:`~tacco.tools.enrichments`)
        - 'clr': "Center logratio transform"; like 'gmean' with additional log
          transform; makes the distribution more normal and better suited for t
          tests
        - `None`: no normalization
        - a value name from `value_key`: all values are normalized to this
          contribution
        - a callable mapping a :class:`~pandas.DataFrame` to its normalized
          counterpart
    
    assume_counts
        Ony relevant for `normalization=='gmean'` and `normalization=='clr'`;
        whether to regularize zeros by adding a pseudo count of 1 or by
        replacing them by 1e-3 of the minimum value. If `None`, check whether
        the data are consistent with count data and assume counts accordingly,
        except if `reads==True`, then also `assume_counts==True`.
    reads
        Whether to weight the values by the total count per observation
    colors
        The mapping of value names to colors. If `None`, a set of
        standard colors is used.
        
    Returns
    -------
    A tuple containing contributions, detailed_contributions, colors, types.
    
    """
    contributions = get_contributions(adata=adata, value_key=value_key, group_key=group_key, sample_key=None, fillna=fillna, restrict_groups=restrict_groups, restrict_values=restrict_values, reads=reads, reduction=reduction, normalization=normalization, assume_counts=assume_counts)
    detailed_contributions = None
    if sample_key is not None:
        detailed_contributions = get_contributions(adata=adata, value_key=value_key, group_key=group_key, sample_key=sample_key, fillna=fillna, restrict_groups=restrict_groups, restrict_values=restrict_values, reads=reads, reduction=reduction, normalization=normalization, assume_counts=assume_counts)
    
    colors, types = _get_colors(colors, contributions.columns)
    
    if basis_adata is not None:
        if basis_value_key is None:
            basis_value_key = value_key
        if basis_group_key is None:
            basis_group_key = group_key
        if basis_sample_key is None:
            basis_sample_key = sample_key
        basis_contributions = get_contributions(adata=basis_adata, value_key=basis_value_key, group_key=basis_group_key, sample_key=None, fillna=fillna, restrict_groups=basis_restrict_groups, restrict_values=basis_restrict_values, reads=reads, reduction=reduction, normalization=normalization, assume_counts=assume_counts)
    
        basis_contributions.index = basis_contributions.index.astype(str) + ' (reference)'
        contributions = pd.concat([basis_contributions,contributions])
        
        if sample_key is not None:
            if basis_sample_key is None:
                basis_sample_key = sample_key
            detailed_basis_contributions = get_contributions(adata=basis_adata, value_key=basis_value_key, group_key=basis_group_key, sample_key=basis_sample_key, fillna=fillna, restrict_groups=basis_restrict_groups, restrict_values=basis_restrict_values, reads=reads, reduction=reduction, normalization=normalization, assume_counts=assume_counts)
    
            detailed_basis_contributions.index = detailed_basis_contributions.index.set_levels([detailed_basis_contributions.index.levels[0].astype(str) + ' (reference)',detailed_basis_contributions.index.levels[1].astype(str) ])
            detailed_basis_contributions.index.names = detailed_contributions.index.names
            detailed_contributions = pd.concat([detailed_basis_contributions,detailed_contributions])
    
    return contributions, detailed_contributions, colors, types

def contribution(
    adata,
    value_key,
    group_key,
    sample_key=None,
    basis_adata=None,
    basis_value_key=None,
    basis_group_key=None,
    basis_sample_key=None,
    fillna=None,
    restrict_groups=None,
    restrict_values=None,
    basis_restrict_groups=None,
    basis_restrict_values=None,
    reduction='sum',
    normalization='gmean',
    assume_counts=None,
    reads=False,
    colors=None,
    axsize=None,
    log=True,
    ax=None,
):

    """\
    Plot contribution to groups. In contrast to :func:`~tacco.plots.composition`,
    contributions dont have to add up to one.
    
    Parameters
    ----------
    adata
        An :class:`~anndata.AnnData` with annotation in `.obs`.
    value_key
        The `.obs` or `.obsm` key with the values to determine the
        enrichment for.
    group_key
        The `.obs` key with categorical group information.
    sample_key
        The `.obs` key with categorical sample information. If `None`,
        only the aggregated data is plotted. Otherwise the data is aggregated
        per sample and total and per-sample values are plotted.
    basis_adata
        Another :class:`~anndata.AnnData` with annotation in `.obs`
        to compare. If `None`, only the `adata` composition is shown.
    basis_value_key
        The `.obs` or `.obsm` key for `basis_adata` with the values
        to determine the enrichment for. If `None`, `value_key` is used.
    basis_group_key
        The `.obs` key with categorical group information for
        `basis_adata`. If `None`, `value_key` is used.
    basis_sample_key
        The `.obs` key with categorical sample information for
        `basis_adata`. If `None`, `sample_key` is used.
    fillna
        If `None`, observation containing NA in the values are filtered.
        Else, NA values are replaced with this value.
    restrict_groups
        A list-like containing the groups within which the enrichment analysis
        is to be done. If `None`, all groups are included.
    restrict_values
        A list-like containing the values within which the enrichment analysis
        is to be done. If `None`, all values are included. Works only for
        categorical values.
    basis_restrict_groups
        Like `restrict_groups` but for `basis_adata`.
    basis_restrict_values
        Like `restrict_values` but for `basis_adata`.
    reduction
        The reduction to apply on each (group,sample) subset of the data.
        Possible values are:
        
        - 'sum': sum of the values over observations
        - 'mean': mean of the values over observations
        - 'median': median of the values over observations
        - a callable mapping a :class:`~pandas.DataFrame` to its reduced
          counterpart
    
    normalization
        The normalization to apply on each reduced (group,sample) subset of the
        data. Possible values are:
        
        - 'sum': normalize values by their sum (yields fractions)
        - 'percent': like 'sum' scaled by 100 (yields percentages)
        - 'gmean': normalize values by their geometric mean (yields
          contributions which make more sense for enrichments than fractions,
          due to zero-sum issue; see :func:`~tacco.tools.enrichments`)
        - 'clr': "Center logratio transform"; like 'gmean' with additional log
          transform; makes the distribution more normal and better suited for t
          tests
        - `None`: no normalization
        - a value name from `value_key`: all values are normalized to this
          contribution
        - a callable mapping a :class:`~pandas.DataFrame` to its normalized
          counterpart
    
    assume_counts
        Ony relevant for `normalization=='gmean'` and `normalization=='clr'`;
        whether to regularize zeros by adding a pseudo count of 1 or by
        replacing them by 1e-3 of the minimum value. If `None`, check whether
        the data are consistent with count data and assume counts accordingly,
        except if `reads==True`, then also `assume_counts==True`.
    reads
        Whether to weight the values by the total count per observation
    colors
        The mapping of value names to colors. If `None`, a set of
        standard colors is used.
    axsize
        Tuple of width and size of a single axis. If `None`, use
        automatic values.
    log
        Whether to plot on the log scale.
    ax
        The :class:`~matplotlib.axes.Axes` to plot on. If `None`, creates a
        fresh figure for plotting.
        
    Returns
    -------
    A :class:`~matplotlib.figure.Figure` if `ax` is `None`, else `None`.
    
    """
    
    contributions, detailed_contributions, colors, types = _prep_contributions(
        adata=adata,
        value_key=value_key,
        group_key=group_key,
        sample_key=sample_key,
        basis_adata=basis_adata,
        basis_value_key=basis_value_key,
        basis_group_key=basis_group_key,
        basis_sample_key=basis_sample_key,
        fillna=fillna,
        restrict_groups=restrict_groups,
        restrict_values=restrict_values,
        basis_restrict_groups=basis_restrict_groups,
        basis_restrict_values=basis_restrict_values,
        reduction=reduction,
        normalization=normalization,
        assume_counts=assume_counts,
        reads=reads,
        colors=colors,
    )
    
    labels = contributions.columns.astype(str)

    total_bars_width = 0.8

    n_states = len(contributions.index)
    bars_separation = total_bars_width / (n_states * 3)
    bar_separation = total_bars_width / (n_states * 5)

    n_states = len(contributions.index)
    
    if ax is not None:
        fig = None
    else:
        if axsize is None:
            axsize = (len(labels) * (0.3 * n_states + .2), 4)
        fig, ax = subplots(axsize=axsize)
        ax = ax[0,0]

    ax.set_axisbelow(True)
    ax.yaxis.grid()#color='gray', linestyle='dashed')

    alpha = 1 if detailed_contributions is None else 0.5
    
    minor_x_labeling = {'position':[],'label':[]}
    
    for i_state, state in enumerate(contributions.index):
        for i_column, column in enumerate(contributions.columns):
            bar_width = (total_bars_width-n_states*bar_separation)/n_states
            bar_start = i_column - 0.5*total_bars_width + i_state * total_bars_width/n_states + 0.5*bar_separation
            ax.bar(bar_start, contributions.loc[state,column], bar_width, color=colors[column], align='edge', alpha=alpha)

            minor_x_labeling['position'].append(i_column - 0.5*total_bars_width + (i_state+0.5)/(n_states) * total_bars_width)
            minor_x_labeling['label'].append(state)
    
    if detailed_contributions is not None:
        for i_state, state in enumerate(contributions.index):
            df = detailed_contributions[detailed_contributions.index.get_level_values(0) == state]
            n_samples = df.shape[0]
            for i_column, column in enumerate(contributions.columns):
                heights = df[column].sort_values(ascending=False)
                bar_start = i_column - 0.5*total_bars_width + 0.5*bars_separation + i_state/n_states * total_bars_width
                bar_width = (total_bars_width - (n_states - 0) * bars_separation) / (n_states * n_samples)
                ax.bar(bar_start + np.arange(0,(n_samples-0.5)*bar_width,bar_width), heights, bar_width, color=colors[column], align='edge')

    # Add some text for labels, title and custom x-axis tick labels, etc.
    ax.set_ylabel('contribution')
    if log:
        ax.set_yscale('log')

    x = np.arange(len(labels))  # the label locations
    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    
    ax.set_xlim(x[0]-0.5,x[-1]+0.5)

    minor_x_labeling = pd.DataFrame(minor_x_labeling).sort_values('position')
    minor_x_labeling['position'] += 1e-2 # tiny shift to make the central minor annotation not be superseded by the major annotation...

    ax.set_xticks(minor_x_labeling['position'],minor=True)
    ax.set_xticklabels(minor_x_labeling['label'], rotation=50, ha='right',minor=True)
    for t in ax.get_xticklabels(minor=True):
        t.set_y(-0.05)

    ax.tick_params( axis='x', which='major', bottom=False, top=False )
    ax.tick_params( axis='x', which='minor', bottom=False, top=False )
    
    return fig

def heatmap(
    adata,
    value_key,
    group_key,
    basis_adata=None,
    basis_value_key=None,
    basis_group_key=None,
    fillna=None,
    restrict_groups=None,
    restrict_values=None,
    basis_restrict_groups=None,
    basis_restrict_values=None,
    reduction='sum',
    normalization=None,
    assume_counts=None,
    reads=False,
    colors=None,
    alpha=None,
    axsize=None,
    axes_labels=None,
    annotation=None,
    value_cluster=False,
    group_cluster=False,
    value_dendrogram=False,
    group_dendrogram=False,
    value_order=None,
    group_order=None,
    group_labels_rotation=None,
    ax=None,
    cmap=None,
    cmap_center=0,
    cmap_vmin_vmax=None,
    complement_colors=True,
    colorbar=True,
    colorbar_label=None,
):

    """\
    Plot heatmap of contribution to groups.
    
    Parameters
    ----------
    adata
        An :class:`~anndata.AnnData` with annotation in `.obs`. Can also be a
        :class:`~pandas.DataFrame` with the data to show in the heatmap. In the
        latter case all adata processing arguments are ignored.
    value_key
        The `.obs` or `.obsm` key with the values to determine the
        enrichment for.
    group_key
        The `.obs` key with categorical group information.
    basis_adata
        Another :class:`~anndata.AnnData` with annotation in `.obs`
        to compare. If `None`, only the `adata` composition is shown.
    basis_value_key
        The `.obs` or `.obsm` key for `basis_adata` with the values
        to determine the enrichment for. If `None`, `value_key` is used.
    basis_group_key
        The `.obs` key with categorical group information for
        `basis_adata`. If `None`, `value_key` is used.
    fillna
        If `None`, observation containing NA in the values are filtered.
        Else, NA values are replaced with this value.
    restrict_groups
        A list-like containing the groups within which the enrichment analysis
        is to be done. If `None`, all groups are included.
    restrict_values
        A list-like containing the values within which the enrichment analysis
        is to be done. If `None`, all values are included. Works only for
        categorical values.
    basis_restrict_groups
        Like `restrict_groups` but for `basis_adata`.
    basis_restrict_values
        Like `restrict_values` but for `basis_adata`.
    reduction
        The reduction to apply on each (group,sample) subset of the data.
        Possible values are:
        
        - 'sum': sum of the values over observations
        - 'mean': mean of the values over observations
        - 'median': median of the values over observations
        - a callable mapping a :class:`~pandas.DataFrame` to its reduced
          counterpart
    
    normalization
        The normalization to apply on each reduced (group,sample) subset of the
        data. Possible values are:
        
        - 'sum': normalize values by their sum (yields fractions)
        - 'percent': like 'sum' scaled by 100 (yields percentages)
        - 'gmean': normalize values by their geometric mean (yields
          contributions which make more sense for enrichments than fractions,
          due to zero-sum issue; see :func:`~tacco.tools.enrichments`)
        - 'clr': "Center logratio transform"; like 'gmean' with additional log
          transform; makes the distribution more normal and better suited for t
          tests
        - `None`: no normalization
        - a value name from `value_key`: all values are normalized to this
          contribution
        - a callable mapping a :class:`~pandas.DataFrame` to its normalized
          counterpart
    
    assume_counts
        Ony relevant for `normalization=='gmean'` and `normalization=='clr'`;
        whether to regularize zeros by adding a pseudo count of 1 or by
        replacing them by 1e-3 of the minimum value. If `None`, check whether
        the data are consistent with count data and assume counts accordingly,
        except if `reads==True`, then also `assume_counts==True`.
    reads
        Whether to weight the values by the total count per observation
    colors
        The mapping of value names to colors. If `None`, a set of
        standard colors is used.
    alpha
        A value-group-dataframe specifying a separate alpha value for all cells
        in the histogram. If `None`, no transparency is used.
    axsize
        Tuple of width and size of a single axis. If `None`, use
        automatic values.
    axes_labels
        Labels to write on the axes as an list-like of the two labels.
    annotation
        A :class:`~pandas.DataFrame` containing annotation for each heatmap
        cell. If "value", annotate by the values. If `None`, don't annotate.
        If a tuple of "value" and a :class:`~pandas.DataFrame`, append the
        annotation from the dataframe to the values.
    value_cluster
        Whether to cluster and reorder the values.
    group_cluster
        Whether to cluster and reorder the groups.
    value_dendrogram
        Whether to draw a dendrogram for the values. If `True`, this implies
        `value_cluster=True`.
    group_dendrogram
        Whether to draw a dendrogram for the groups. If `True`, this implies
        `group_cluster=True`.
    value_order
        Set the order of the values explicitly with a list or to be close to
        diagonal by specifying "diag"; this option is incompatible with
        `value_cluster` and `value_dendrogram`.
    group_order
        Set the order of the groups explicitly with a list or to be close to
        diagonal by specifying "diag"; this option is incompatible with
        `group_cluster` and `group_dendrogram`.
    group_labels_rotation
        Adjusts the rotation of the group labels in degree.
    ax
        The :class:`~matplotlib.axes.Axes` to plot on. If `None`, creates a
        fresh figure for plotting. Incompatible with dendrogram plotting.
    cmap
        A string/colormap to override the `colors` with.
    cmap_center
        A value to use as center of the colormap. E.g. choosing `0` sets the
        central color to `0` for every colormap in the plot (i.e. `0` will get
        white, positive and negative colors the color and complement color
        given by `colors` for `cmap` is `None`, and whatever the central color
        of the supplied colormap is if `cmap` is not `None`). If `None`, the
        colormap spans the entire value range.
    cmap_vmin_vmax
        A tuple giving the range of values for the colormap. This can be
        modfied by `cmap_center`.
    complement_colors
        Whether to use complement colors for values below `cmap_center` if
        `cmap==None`.
    colorbar
        Whether to draw a colorbar; only available if `cmap` is not `None`.
    colorbar_label
        The label to use for the colorbar; only available if `colorbar` is
        `True`.
        
    Returns
    -------
    A :class:`~matplotlib.figure.Figure` if `ax` is `None`, else `None`.
    
    """
    
    if isinstance(adata, pd.DataFrame):
        contributions = adata.fillna(0)
        types = contributions.index
    else:
        contributions, detailed_contributions, colors, types = _prep_contributions(
            adata=adata,
            value_key=value_key,
            group_key=group_key,
            sample_key=None,
            basis_adata=basis_adata,
            basis_value_key=basis_value_key,
            basis_group_key=basis_group_key,
            basis_sample_key=None,
            fillna=fillna,
            restrict_groups=restrict_groups,
            restrict_values=restrict_values,
            basis_restrict_groups=basis_restrict_groups,
            basis_restrict_values=basis_restrict_values,
            reduction=reduction,
            normalization=normalization,
            assume_counts=assume_counts,
            reads=reads,
            colors=colors,
        )
    
    value_cluster = value_cluster or value_dendrogram
    group_cluster = group_cluster or group_dendrogram
    
    if value_order is not None and value_cluster:
        raise ValueError('The options `value_cluster` and `value_dendrogram` are incompatible with the option `value_order`.')
    if group_order is not None and group_cluster:
        raise ValueError('The options `group_cluster` and `group_dendrogram` are incompatible with the option `group_order`.')
    group_order_diag = isinstance(group_order, str) and group_order == 'diag'
    value_order_diag = isinstance(value_order, str) and value_order == 'diag'
    if group_order_diag and value_order is not None and not value_order_diag:
        raise ValueError('The option `group_order=="diag"` is incompatible with non-´None´ or "diag" values for `value_order`.')
    if value_order_diag and group_order is not None and not group_order:
        raise ValueError('The option `value_order=="diag"` is incompatible with non-´None´ or "diag" values for `group_order`.')
    if group_order_diag and group_cluster:
        raise ValueError('The option `group_order=="diag"` is incompatible with the options `value_cluster` and `value_dendrogram`.')
    if value_order_diag and value_cluster:
        raise ValueError('The option `value_order=="diag"` is incompatible with the options `group_cluster` and `group_dendrogram`.')
    
    if ax is not None:
        if value_dendrogram or group_dendrogram:
            raise ValueError('The options `value_dendrogram` and `group_dendrogram` are incompatible with the option `ax`.')
        fig = ax.get_figure()
    else:
        n_ax_x, n_ax_y = 1+value_dendrogram, 1+group_dendrogram
        if axsize is None:
            y_per_value = 0.25
            x_per_group = y_per_value if annotation is None else 0.7
            plot_size_y = contributions.shape[1] * y_per_value
            plot_size_x = contributions.shape[0] * x_per_group
            axsize = (plot_size_x,plot_size_y)
        else:
            plot_size_x,plot_size_y = axsize
        width_ratios = [plot_size_x,0.5*np.log(contributions.shape[1])] if value_dendrogram else None
        height_ratios = [0.5*np.log(contributions.shape[0]),plot_size_y] if group_dendrogram else None
        fig,axs = subplots(n_ax_x, n_ax_y, axsize=axsize, wspace=0, hspace=0, width_ratios=width_ratios, height_ratios=height_ratios)
        ax = axs[-1,0]
    
    if group_labels_rotation is None:
        if annotation is None:
            group_labels_rotation = 45
        else:
            group_labels_rotation = 30
    
    if value_cluster:
        Z = scipy.cluster.hierarchy.linkage(contributions.T, method='average', metric='cosine')
        dn = scipy.cluster.hierarchy.dendrogram(Z, ax=(axs[-1,-1] if value_dendrogram else None), orientation='right', color_threshold=0, above_threshold_color='tab:gray', no_plot=(not value_dendrogram))
        if value_dendrogram:
            axs[-1,-1].set_axis_off()
        
        reordering = pd.Series(dn['ivl']).astype(np.int).to_numpy()
        contributions = contributions.iloc[:,reordering]
    
    if group_cluster:
        Z = scipy.cluster.hierarchy.linkage(contributions, method='average', metric='cosine')
        dn = scipy.cluster.hierarchy.dendrogram(Z, ax=(axs[0,0] if group_dendrogram else None), orientation='top', color_threshold=0, above_threshold_color='tab:gray', no_plot=(not group_dendrogram))
        if group_dendrogram:
            axs[0,0].set_axis_off()
        
        reordering = pd.Series(dn['ivl']).astype(np.int).to_numpy()
        contributions = contributions.iloc[reordering]
        
    if value_dendrogram and group_dendrogram:
        axs[0,-1].set_axis_off()
    
    if isinstance(value_order,str) and value_order  == 'diag':
        # permute towards diagonal
        for i in range(10):
            contributions = contributions.iloc[:,np.argsort(np.argmax(contributions.to_numpy(),axis=0))]
            contributions = contributions.iloc[np.argsort(np.argmax(contributions.to_numpy().T,axis=0))]
    else:
        if value_order is not None:
            contributions = contributions.loc[:,value_order]
        if group_order is not None:
            contributions = contributions.loc[group_order]
    
    _x, _y = [np.arange(-0.5,s+0.5,1) for s in contributions.shape]
    x, y = (_x[:-1] + _x[1:]) / 2, (_y[:-1] + _y[1:]) / 2
    
    def _vmin_vmax(vmin, vmax):
        if cmap_vmin_vmax is not None:
            vmin, vmax = cmap_vmin_vmax
        if cmap_center is not None:
            delta_max = vmax - cmap_center
            delta_min = -(vmin - cmap_center)
            delta = max(delta_max, delta_min)
            vmax = cmap_center + delta
            vmin = cmap_center - delta
        return vmin, vmax
    
    if cmap is None:
        rgba = np.zeros((*contributions.shape,4))
        for j in range(contributions.shape[1]):

            r, g, b = to_rgb(colors[contributions.columns[j]])
            _r, _g, _b = _complement_color(r,g,b) # complement colors for negative values
            vmin, vmax = _vmin_vmax(contributions.iloc[:,j].min(), contributions.iloc[:,j].max())
            norm = Normalize(vmin=vmin, vmax=vmax)
            if complement_colors:
                cmap_j=LinearSegmentedColormap.from_list(contributions.columns[j], [(0,(_r, _g, _b)),(0.5,(1, 1, 1)),(1,(r, g, b))])
            else:
                cmap_j=LinearSegmentedColormap.from_list(contributions.columns[j], [(0,(1, 1, 1)),(1,(r, g, b))])
            mapper = ScalarMappable(norm=norm, cmap=cmap_j)
            rgba[:,j,:] = mapper.to_rgba(contributions.iloc[:,j].to_numpy())
    else:
        vmin, vmax = _vmin_vmax(contributions.to_numpy().min(), contributions.to_numpy().max())

        norm = Normalize(vmin=vmin, vmax=vmax)
        mapper = ScalarMappable(norm=norm, cmap=cmap)
        rgba = mapper.to_rgba(contributions.to_numpy())

        if colorbar:
            height_pxl = 200
            width_pxl = 15
            offset_top_pxl = 0
            offset_left_pxl = 20

            left,bottom = fig.transFigure.inverted().transform(ax.transAxes.transform((1,1))+np.array([offset_left_pxl,-offset_top_pxl-height_pxl]))
            width,height = fig.transFigure.inverted().transform(fig.transFigure.transform((0,0))+np.array([width_pxl,height_pxl]))
            cax = fig.add_axes((left, bottom, width, height))
            cb = fig.colorbar(mapper, cax=cax, label=colorbar_label)
    
    if alpha is not None:
        alpha = alpha.reindex(index=contributions.index, columns=contributions.columns)
        rgba[...,-1] = alpha
    ax.imshow(np.swapaxes(rgba, 0, 1), origin='lower', aspect='auto')
        
    if annotation is not None:
        for i,ind in enumerate(contributions.index):
            for j,col in enumerate(contributions.columns):
                if isinstance(annotation, str):
                    if annotation == 'value':
                        ann = f'{contributions.loc[ind,col]:.2}'
                    else:
                        raise ValueError(f'`annotation` got unknown string argument "{annotation}"')
                elif (hasattr(annotation, 'shape') and annotation.shape == (2,)) or (not hasattr(annotation, 'shape') and len(annotation) == 2):
                    if isinstance(annotation[0], str) and annotation[0] == 'value':
                        if ind in annotation[1].index and col in annotation[1].columns:
                            val = annotation[1].loc[ind,col]
                        else:
                            val = 0#np.nan
                        ann = f'{contributions.loc[ind,col]:.2}{val}'
                    else:
                        raise ValueError('`annotation` got a tuple argument where the first entry is not "value"')
                else:
                    ann = annotation.loc[ind,col]
                ax.annotate(ann, xy=(x[i],y[j]), ha='center', va='center')
    
    ax.set_xticks(x)
    ax.set_xticklabels(contributions.index, rotation=group_labels_rotation, ha=('right' if group_labels_rotation not in [0,90] else 'center'))
    ax.set_yticks(y)
    ax.set_yticklabels(contributions.columns)
    
    _set_axes_labels(ax, axes_labels)
    
    return fig

def _asterisks_from_pvals(pvals):
    pvals = pvals.astype(object)
    anstr = pvals.astype(str)
    anstr.loc[:,:] = ''
    anstr[(pvals > 0) & (np.abs(pvals) <= 0.05)] = '$^{\\ast}$'
    anstr[(pvals > 0) & (np.abs(pvals) <= 0.01)] = '${^{\\ast}}{^{\\ast}}$'
    anstr[(pvals > 0) & (np.abs(pvals) <= 0.001)] = '${^{\\ast}}{^{\\ast}}{^{\\ast}}$'
    anstr[(pvals < 0) & (np.abs(pvals) <= 0.05)] = '$_{\\ast}$'
    anstr[(pvals < 0) & (np.abs(pvals) <= 0.01)] = '${_{\\ast}}{_{\\ast}}$'
    anstr[(pvals < 0) & (np.abs(pvals) <= 0.001)] = '${_{\\ast}}{_{\\ast}}{_{\\ast}}$'
    return anstr

def sigmap(
    adata,
    value_key,
    group_key,
    sample_key=None,
    position_key=None,
    position_split=2,
    min_obs=0,
    basis_adata=None,
    basis_value_key=None,
    basis_group_key=None,
    basis_sample_key=None,
    basis_position_key=None,
    basis_position_split=None,
    basis_min_obs=None,
    fillna=None,
    restrict_groups=None,
    restrict_values=None,
    basis_restrict_groups=None,
    basis_restrict_values=None,
    p_corr='fdr_bh',
    method='mwu',
    reduction=None,
    normalization=None,
    assume_counts=None,
    reads=False,
    colors=None,
    axsize=None,
    value_dendrogram=False,
    group_dendrogram=False,
    value_order=None,
    group_order=None,
    ax=None,
):

    """\
    Plot heatmap of contribution to groups and mark significant differences
    with asterisks.
    
    Parameters
    ----------
    adata
        An :class:`~anndata.AnnData` with annotation in `.obs`.
    value_key
        The `.obs` or `.obsm` key with the values to determine the
        enrichment for.
    group_key
        The `.obs` key with categorical group information.
    sample_key
        The `.obs` key with categorical sample information for p-value
        determination. If `None`, use only the aggregated data is plotted.
    position_key
        The `.obsm` key or array-like of `.obs` keys with the position space
        coordinates. If `None`, no position splits are performed.
    position_split
        The number of splits per spatial dimension before enrichment. Can be a
        tuple with the spatial dimension as length to assign a different split
        per dimension. If `None`, no position splits are performed. See also
        `min_obs`.
    min_obs
        The minimum number of observations per sample: if less observations are
        available, the sample is not used. This also limits the number of
        `position_split` to stop splitting if the split would decrease the
        number of observations below this threshold.
    basis_adata
        Another :class:`~anndata.AnnData` with annotation in `.obs`
        to compare. If `None`, only the `adata` composition is shown.
    basis_value_key
        The `.obs` or `.obsm` key for `basis_adata` with the values
        to determine the enrichment for. If `None`, `value_key` is used.
    basis_group_key
        The `.obs` key with categorical group information for
        `basis_adata`. If `None`, `value_key` is used.
    basis_sample_key
        The `.obs` key with categorical sample information for
        `basis_adata`. If `None`, `sample_key` is used.
    basis_position_key
        Like `position_key` but for `basis_adata`. If `None`, no position
        splits are performed.
    basis_position_split
        Like `position_split` but for `basis_adata`. If `None`,
        `position_split` is used.
    basis_min_obs
        Like `min_obs` but for `basis_adata`. If `None`, `min_obs` is used.
    fillna
        If `None`, observation containing NA in the values are filtered.
        Else, NA values are replaced with this value.
    restrict_groups
        A list-like containing the groups within which the enrichment analysis
        is to be done. If `None`, all groups are included.
    restrict_values
        A list-like containing the values within which the enrichment analysis
        is to be done. If `None`, all values are included. Works only for
        categorical values.
    basis_restrict_groups
        Like `restrict_groups` but for `basis_adata`.
    basis_restrict_values
        Like `restrict_values` but for `basis_adata`.
    p_corr
        The name of the p-value correction method to use. Possible values are
        the ones available in
        :func:`~statsmodels.stats.multitest.multipletests`. If `None`, no
        p-value correction is performed.
    method
        Specification of methods to use for enrichment. Available are:
        
        - 'fisher': Fishers exact test; only for categorical values. Ignores
          the `reduction` and `normalization` arguments.
        - 'mwu': MannWhitneyU test
        
    reduction
        The reduction to apply on each (group,sample) subset of the data.
        Possible values are:
        
        - 'sum': sum of the values over observations
        - 'mean': mean of the values over observations
        - 'median': median of the values over observations
        - `None`: use observations directly
        - a callable mapping a :class:`~pandas.DataFrame` to its reduced
          counterpart
    
    normalization
        The normalization to apply on each reduced (group,sample) subset of the
        data. Possible values are:
        
        - 'sum': normalize values by their sum (yields fractions)
        - 'percent': like 'sum' scaled by 100 (yields percentages)
        - 'gmean': normalize values by their geometric mean (yields
          contributions which make more sense for enrichments than fractions,
          due to zero-sum issue; see :func:`~tacco.tools.enrichments`)
        - 'clr': "Center logratio transform"; like 'gmean' with additional log
          transform; makes the distribution more normal and better suited for t
          tests
        - `None`: no normalization
        - a value name from `value_key`: all values are normalized to this
          contribution
        - a callable mapping a :class:`~pandas.DataFrame` to its normalized
          counterpart
    
    assume_counts
        Ony relevant for `normalization=='gmean'` and `normalization=='clr'`;
        whether to regularize zeros by adding a pseudo count of 1 or by
        replacing them by 1e-3 of the minimum value. If `None`, check whether
        the data are consistent with count data and assume counts accordingly,
        except if `reads==True`, then also `assume_counts==True`.
    reads
        Whether to weight the values by the total count per observation
    colors
        The mapping of value names to colors. If `None`, a set of standard
        colors is used.
    axsize
        Tuple of width and size of a single axis. If `None`, use automatic
        values.
    value_dendrogram
        Whether to draw a dendrogram for the values
    group_dendrogram
        Whether to draw a dendrogram for the groups
    value_order
        Set the order of the values explicitly; this option is incompatible
        with `value_dendrogram`.
    group_order
        Set the order of the groups explicitly; this option is incompatible
        with `group_dendrogram`.
    ax
        The :class:`~matplotlib.axes.Axes` to plot on. If `None`, creates a
        fresh figure for plotting. Incompatible with dendrogram plotting.
        
    Returns
    -------
    A :class:`~matplotlib.figure.Figure` if `ax` is `None`, else `None`.
    
    """
    
    pvals = enrichments(
        adata=adata,value_key=value_key,group_key=group_key,sample_key=sample_key, method=method,fillna=fillna,
        position_key=position_key, position_split=position_split,min_obs=min_obs,p_corr=p_corr,
        restrict_groups=restrict_groups,restrict_values=restrict_values, reads=reads,
        reduction=reduction,normalization=normalization,assume_counts=assume_counts,
    )
    if basis_adata is not None:
        if basis_position_split is None:
            basis_position_split = position_split
        if basis_min_obs is None:
            basis_min_obs = min_obs
        if basis_sample_key is None:
            basis_sample_key = sample_key
        if basis_group_key is None:
            basis_group_key = group_key
        if basis_value_key is None:
            basis_value_key = value_key
        basis_pvals = enrichments(
            adata=basis_adata,value_key=basis_value_key,group_key=basis_group_key,sample_key=basis_sample_key, method=method,fillna=fillna,
            position_key=basis_position_key, position_split=basis_position_split,min_obs=basis_min_obs,p_corr=p_corr,
            restrict_groups=basis_restrict_groups,restrict_values=basis_restrict_values, reads=reads,
            reduction=reduction,normalization=normalization,assume_counts=assume_counts,
        )
        basis_pvals.rename(columns={basis_value_key:value_key}, inplace=True)
        basis_pvals[basis_group_key] = basis_pvals[basis_group_key].cat.rename_categories(lambda c: c + ' (reference)')
        basis_pvals.rename(columns={basis_group_key:group_key}, inplace=True)
        joint_categories = [*list(basis_pvals[group_key].cat.categories), *list(pvals[group_key].cat.categories)]
        pvals = pd.concat([basis_pvals,pvals])
        pvals[group_key] = pvals[group_key].astype(pd.CategoricalDtype(joint_categories,ordered=True))

    ann = pd.pivot_table(pvals[pvals['enrichment']=='enriched'],values=f'p_{method}_{p_corr}',index=group_key,columns=value_key)
    annp = pd.pivot_table(pvals[pvals['enrichment']!='enriched'],values=f'p_{method}_{p_corr}',index=group_key,columns=value_key)
    ann[annp<ann] = -annp
    anstr = _asterisks_from_pvals(ann)

    fig = heatmap(
        adata=adata,value_key=value_key,group_key=group_key,
        basis_adata=basis_adata,basis_value_key=basis_value_key,basis_group_key=basis_group_key,
        fillna=fillna,restrict_groups=restrict_groups,restrict_values=restrict_values,
        basis_restrict_groups=basis_restrict_groups,basis_restrict_values=basis_restrict_values,
        reduction=reduction,normalization=normalization,reads=reads,colors=colors,axsize=axsize,annotation=('value',anstr),
        value_dendrogram=value_dendrogram, group_dendrogram=group_dendrogram,assume_counts=assume_counts,
        ax=ax, colorbar=False, complement_colors=False, cmap_center=None,
        value_order=value_order,group_order=group_order,
    );
    
    return fig

def significances(
    significances,
    p_key,
    value_key,
    group_key,
    pmax=0.05,
    pmin=1e-5,
    annotate_pvalues=True,
    value_cluster=False,
    group_cluster=False,
    value_order=None,
    group_order=None,
    axsize=None,
    ax = None,
    scale_legend=1.0
):

    """\
    Plot enrichment significances.
    
    Parameters
    ----------
    significances
        A :class:`~pandas.DataFrame` with p-values and their annotation..
    p_key
        The key with the p-values.
    value_key
        The key with the values for which the enrichment was determined.
    group_key
        The key with the groups in which the enrichment was determined.
    pmax
        The maximum p-value to show.
    pmin
        The minimum p-value on the color scale.
    annotate_pvalues
        Whether to annotate p-values
    value_cluster
        Whether to cluster and reorder the values.
    group_cluster
        Whether to cluster and reorder the groups.
    value_order
        Set the order of the values explicitly with a list or to be close to
        diagonal by specifying "diag"; this option is incompatible with
        `value_cluster`.
    group_order
        Set the order of the groups explicitly with a list or to be close to
        diagonal by specifying "diag"; this option is incompatible with
        `group_cluster`.
    axsize
        Tuple of width and size of a single axis. If `None`, use
        automatic values.
    ax
        The :class:`~matplotlib.axes.Axes` to plot on. If `None`, creates a
        fresh figure for plotting. Incompatible with dendrogram plotting.
    scale_legend
        Set to scale height and width of legend.  
        
    Returns
    -------
    A :class:`~matplotlib.figure.Figure`.
    
    """
    
    enr_e = pd.pivot(significances[significances['enrichment']=='enriched'], value_key, group_key, p_key)
    enr_p = pd.pivot(significances[significances['enrichment']!='enriched'], value_key, group_key, p_key)

    small_value = 1e-300
    max_log = -np.log(pmin)
    min_log = -np.log(pmax)
    
    enr_e = np.maximum(enr_e,small_value)
    enr_p = np.maximum(enr_p,small_value)

    enr_p = enr_p.reindex_like(enr_e)
    enr = pd.DataFrame(np.where(enr_e < enr_p, -np.log(enr_e), np.log(enr_p)),index=enr_e.index,columns=enr_e.columns)
    
    if annotate_pvalues:
        ann = pd.DataFrame(np.where(enr_e < enr_p, enr_e, enr_p),index=enr_e.index,columns=enr_e.columns)
        ann = pd.DataFrame(np.where(ann <= pmax, ann.applymap(lambda x: f'{x:.2}'), ''),index=enr_e.index,columns=enr_e.columns)
        ann = ann.T
    else:
        ann = None

    # setup the plotting
    enriched_color = (1.0, 0.07058823529411765, 0.09019607843137255)
    depleted_color = (0.30196078431372547, 0.5215686274509804, 0.7098039215686275)
    null_color = (0.9,0.9,0.9)
    slightly_weight = 0.2
    slightly_enriched_color, slightly_depleted_color = mix_base_colors(
        np.array([[slightly_weight,1-slightly_weight,0.0],[0.0,1-slightly_weight,slightly_weight],]),
        np.array([list(enriched_color),list(null_color),list(depleted_color)])
    )
    ct1 = 0.5 * (1 - min_log/max_log)
    ct2 = 0.5 * (1 + min_log/max_log)
    cdict = {'red': [[0.0,  depleted_color[0], depleted_color[0]],
                     [ct1,  slightly_depleted_color[0], null_color[0]],
                     [ct2,  null_color[0], slightly_enriched_color[0]],
                     [1.0,  enriched_color[0], enriched_color[0]]],
           'green': [[0.0,  depleted_color[1], depleted_color[1]],
                     [ct1,  slightly_depleted_color[1], null_color[1]],
                     [ct2,  null_color[1], slightly_enriched_color[1]],
                     [1.0,  enriched_color[1], enriched_color[1]]],
            'blue': [[0.0,  depleted_color[2], depleted_color[2]],
                     [ct1,  slightly_depleted_color[2], null_color[2]],
                     [ct2,  null_color[2], slightly_enriched_color[2]],
                     [1.0,  enriched_color[2], enriched_color[2]]]}
    cmap = LinearSegmentedColormap('sigmap', segmentdata=cdict, N=256)

    fig = heatmap(enr.T, None, None, cmap=cmap, cmap_vmin_vmax=(-max_log,max_log), annotation=ann, colorbar=False, value_cluster=value_cluster, group_cluster=group_cluster, value_order=value_order, group_order=group_order, axsize=axsize, ax=ax);

    rel_dpi_factor = matplotlib.rcParams['figure.dpi'] / 72
    height_pxl = 200 * rel_dpi_factor * scale_legend
    width_pxl = 15 * rel_dpi_factor * scale_legend
    offset_top_pxl = 0 * rel_dpi_factor * scale_legend
    offset_left_pxl = 30 * rel_dpi_factor * scale_legend

    ax = fig.axes[0]
    left,bottom = fig.transFigure.inverted().transform(ax.transAxes.transform((1,1))+np.array([offset_left_pxl,-offset_top_pxl-height_pxl]))
    width,height = fig.transFigure.inverted().transform(fig.transFigure.transform((0,0))+np.array([width_pxl,height_pxl]))
    cax = fig.add_axes((left, bottom, width, height))
    norm = Normalize(vmin=-max_log, vmax=max_log)
    cb = fig.colorbar(ScalarMappable(norm=norm, cmap=cmap), cax=cax)
    cb.set_ticks([-max_log,-min_log,min_log,max_log])
    cb.set_ticklabels([pmin,pmax,pmax,pmin])
    cb.ax.annotate('enriched', xy=(0, 1), xycoords='axes fraction', xytext=(-3, -5), textcoords='offset pixels', horizontalalignment='right', verticalalignment='top', rotation=90, fontsize=10*scale_legend)
    cb.ax.annotate('insignificant', xy=(0, 0.5), xycoords='axes fraction', xytext=(-3, 5), textcoords='offset pixels', horizontalalignment='right', verticalalignment='center', rotation=90, fontsize=10*scale_legend)
    cb.ax.annotate('depleted', xy=(0, 0), xycoords='axes fraction', xytext=(-3, 5), textcoords='offset pixels', horizontalalignment='right', verticalalignment='bottom', rotation=90, fontsize=10*scale_legend)

    return fig

def _escape_math_special_characters(string):
    string = string.replace('_','\_')
    return string

def _get_co_occurrence(adata, analysis_key, show_only, show_only_center, colors, score_key, log_base):

    if analysis_key not in adata.uns:
        raise ValueError(f'`analysis_key` "{analysis_key}" is not found in `adata.uns`! Make sure to run tc.tl.co_occurrence first.')

    if score_key in adata.uns[analysis_key]:
        mean_scores = adata.uns[analysis_key][score_key]
        if score_key.startswith('log'):
            if log_base is not None:
                mean_scores = mean_scores / np.log(log_base)
    else:
        raise ValueError(f'The score_key {score_key!r} was not found!')
    
    if mean_scores is None:
        raise ValueError(f'The score {score_key!r} in `adata.uns[{analysis_key}]` is None!')
    
    intervals = adata.uns[analysis_key]['interval']
    annotation = adata.uns[analysis_key]['annotation']
    center = adata.uns[analysis_key]['center']
    
    if show_only is not None:
        if isinstance(show_only, str):
            show_only = [show_only]
        select = annotation.isin(show_only)
        if select.sum() < len(show_only):
            raise ValueError(f'The `show_only` categories {[s for s in show_only if s not in annotation]!r} are not available in the data!')
        annotation = annotation[select]
        mean_scores = mean_scores[select,:,:]

    if show_only_center is not None:
        if isinstance(show_only_center, str):
            show_only_center = [show_only_center]
        select = center.isin(show_only_center)
        if select.sum() < len(show_only_center):
            raise ValueError(f'The `show_only_center` categories {[s for s in show_only_center if s not in center]!r} are not available in the data!')
        center = center[select]
        mean_scores = mean_scores[:,select,:]

    colors, types = _get_colors(colors, pd.Series(annotation))
    
    return mean_scores, intervals, annotation, center, colors, types

def _get_cooc_expression_label(score_key,log_base):
    if score_key == 'occ':
        expression = '$\\frac{p(anno|center)}{p(anno)}$'
    elif score_key == 'log_occ':
        base_str = '' if log_base is None else f'_{log_base}'
        expression = '$log' + base_str + '\\left(\\frac{p(anno|center)}{p(anno)}\\right)$'
    elif score_key == 'z':
        expression = '$\\frac{log(N(anno,center))-random expectation}{standard deviation}$'
    elif score_key == 'composition':
        expression = '$p(anno|center)$'
    elif score_key == 'log_composition':
        base_str = '' if log_base is None else f'_{log_base}'
        expression = '$log' + base_str + '(p(anno|center))$'
    elif score_key == 'distance_distribution':
        expression = '$p(dist|anno,center)$'
    elif score_key == 'log_distance_distribution':
        base_str = '' if log_base is None else f'_{log_base}'
        expression = '$log' + base_str + '(p(dist|anno,center))$'
    elif score_key == 'relative_distance_distribution':
        expression = '$\\frac{p(dist|anno,center)}{p(dist|*,center)}$'
    elif score_key == 'log_relative_distance_distribution':
        base_str = '' if log_base is None else f'_{log_base}'
        expression = '$log' + base_str + '\\left(\\frac{p(dist|anno,center)}{p(dist|*,center)}\\right)$'
    else:
        base_str = '' if log_base is None else f'_{log_base}'
        expression = '$log' + base_str + '\\left(\\frac{p(anno|center)/p(anno)}{gmean\\left(p(anno|center)/p(anno)\\right)}\\right)$'
    return expression

def co_occurrence(
    adata,
    analysis_key,
    score_key='log_occ',
    log_base=None,
    colors=None,
    show_only=None,
    show_only_center=None,
    axsize=(4,3),
    sharex=True,
    sharey='col',
    wspace=0.15,
    hspace=0.3,
    legend=True,
    grid=True,
    merged=False,
    ax=None
):

    """\
    Plot co-occurrence as determined by :func:`~tacco.tools.co_occurrence`.
    
    Parameters
    ----------
    adata
        An :class:`~anndata.AnnData` with the co-occurence analysis in `.uns`.
        Can also be a mapping of labels to :class:`~anndata.AnnData` to specify
        multiple datasets.
    analysis_key
        The `.uns` key with the co-occurence analysis result.
    score_key
        The `.uns[analysis_key]` key of the score to use or the
        `.uns[analysis_key]['comparisons']` sub-key specifying the comparison
        to plot. Available keys `.uns[analysis_key]` include:
        
        - 'occ': co-occurrence
        - 'log_occ': logarithm of the co-occurrence
        - 'log2_occ': base-2-logarithm of the co-occurrence; this is a not a
          real key but a convenience function to rescale the 'log_occ' values
        - 'z': z-score of the log of the neighbourship counts with respect to
          random neighbourships
        - 'composition': distance dependent composition, `p(anno|center, dist)`
        - 'log_composition': log of 'composition'
        - 'distance_distribution': distribution of distances between anno and
          center ´p(dist|anno,center)´
        - 'log_distance_distribution': log of 'distance_distribution'
        - 'relative_distance_distribution': 'distance_distribution' normalized
          to `p(dist|*,center)`, the distance distribution of any annotation to
          the center
        - 'log_relative_distance_distribution': log of
          'relative_distance_distribution'
    log_base
        The base of the logarithm to use for plotting if `score_key` is
        'log_occ' or a comparison key. If `None`, use the natural logarithm.
    colors
        The mapping of value names to colors. If `None`, a set of
        standard colors is used.
    show_only
        A subset of annotation values to restrict the plotting to.
    show_only_center
        A subset of the center annotation values to restrict the plotting
        to.
    axsize
        Tuple of width and size of a single axis.
    sharex, sharey
        Whether and how to use common x/y axis. Options include `True`,
        `False`, "col", "row", "none", and "all".
    wspace, hspace
        Control the spacing between the plots.
    legend
        Whether to include the legend
    grid
        Whether to plot a grid
    merged
        Whether to merge the plots for all :class:`~anndata.AnnData` instances
        into a single row of plots. This makes only sense if more instances are
        provided in `adata`.
    ax
        The :class:`~matplotlib.axes.Axes` to plot on. If `None`, creates a
        fresh figure for plotting. Incompatible with dendrogram plotting.
        
    Returns
    -------
    A :class:`~matplotlib.figure.Figure`.
    
    """
    
    adatas = _get_adatas(adata)
    
    fig = None
    linestyles = ['solid','dashed','dotted','dashdot']
    
    for adata_i, (adata_name, adata) in enumerate(adatas.items()):

        mean_scores, intervals, annotation, center, colors, types = _get_co_occurrence(adata, analysis_key, show_only, show_only_center, colors, score_key, log_base)
        
        if merged:
            
            if len(adatas) > len(linestyles):
                raise ValueError(f'`merged==True` is ony possible with up to {len(linestyles)} andatas!')
            
            
            #if fig is None:
            #    fig, axs = subplots(len(center), 1, axsize=axsize, hspace=hspace, wspace=wspace, sharex=sharex, sharey=sharey)

            if ax is not None:
                        if isinstance(ax, matplotlib.axes.Axes):
                            axs = np.array([[ax]])
                        else:
                            axs = ax
                        if axs.shape != (len(center), 1):
                            raise ValueError(f'The `ax` argument got the wrong shape of axes: needed is {(len(center), 1)!r} supplied was {axs.shape!r}!')
                        axsize = axs[0,0].get_window_extent().transformed(axs[0,0].get_figure().dpi_scale_trans.inverted()).size
                        fig = axs[0,0].get_figure()
            else:
                fig, axs = subplots(len(center), 1, axsize=axsize, hspace=hspace, wspace=wspace, sharex=sharex, sharey=sharey, )


            for ir, nr in enumerate(center):
                x = (intervals[1:] + intervals[:-1]) / 2
                for ja, na in enumerate(annotation):
                    y = mean_scores[ja,ir,:]
                    linestyle = linestyles[adata_i]
                    axs[0,ir].plot(x, y, color=colors[na], linestyle=linestyle)
                
                if adata_i == len(adatas) - 1:
                    axs[0,ir].set_xlabel('distance')
                    adata_title = f'center {center.name}={nr}: '
                    anno_title = 'annotation: ' + annotation.name
                    axs[0,ir].set_xlabel('distance')
                    axs[0,ir].set_ylabel(anno_title)
                    expression = _get_cooc_expression_label(score_key,log_base)
                    axs[0,ir].set_title(adata_title + expression)
                    axs[0,ir].grid(grid)

        else:
            
            #if fig is None:
            #    fig, axs = subplots(len(center), len(adatas), axsize=axsize, hspace=hspace, wspace=wspace, sharex=sharex, sharey=sharey)

            if ax is not None:
                if isinstance(ax, matplotlib.axes.Axes):
                    axs = np.array([[ax]])
                else:
                    axs = ax
                if axs.shape != (len(center), len(adatas)):
                    raise ValueError(f'The `ax` argument got the wrong shape of axes: needed is {(len(center), len(adatas))!r} supplied was {axs.shape!r}!')
                axsize = axs[0,0].get_window_extent().transformed(axs[0,0].get_figure().dpi_scale_trans.inverted()).size
                fig = axs[0,0].get_figure()
            else:
                fig, axs = subplots(len(center), len(adatas), axsize=axsize, hspace=hspace, wspace=wspace, sharex=sharex, sharey=sharey, )

            for ir, nr in enumerate(center):
                x = (intervals[1:] + intervals[:-1]) / 2
                for ja, na in enumerate(annotation):
                    y = mean_scores[ja,ir,:]
                    linestyle = linestyles[nr==na]
                    axs[adata_i,ir].plot(x, y, color=colors[na], linestyle=linestyle)

                axs[adata_i,ir].set_xlabel('distance')

                adata_title = f'{adata_name}, center {center.name}={nr}: ' if adata_name != '' else f'center {center.name}={nr}: '
                anno_title = 'annotation: ' + annotation.name
                axs[adata_i,ir].set_xlabel('distance')
                axs[adata_i,ir].set_ylabel(anno_title)

                expression = _get_cooc_expression_label(score_key,log_base)

                axs[adata_i,ir].set_title(adata_title + expression)

                axs[adata_i,ir].grid(grid)

    if legend:
        
        handles = []

        if merged:
            handles.extend([mlines.Line2D([], [], color='gray', label=adata_name, linestyle=linestyle) for ((adata_name, adata), linestyle) in zip(adatas.items(), linestyles) ])
        
        handles.extend([mpatches.Patch(color=color, label=ind) for (ind, color) in zip(annotation, colors[annotation]) ])
        
        axs[0,len(center)-1].legend(handles=handles, bbox_to_anchor=(1, 1), loc='upper left', ncol=1)
            
    
    return fig

def co_occurrence_matrix(
    adata,
    analysis_key,
    score_key='log_occ',
    log_base=None,
    colors=None,
    show_only=None,
    show_only_center=None,
    axsize=None,
    hspace=None,
    wspace=None,
    x_padding=2.0,
    y_padding=2.0,
    value_cluster=False,
    group_cluster=False,
    restrict_intervals=None,
    p_corr='fdr_bh',
    cmap='bwr',
    cmap_vmin_vmax=None,
    legend=True,
    ax = None,
    scale_legend=1.0
):
    """\
    Plot co-occurrence as determined by :func:`~tacco.tools.co_occurrence` or
    :func:`~tacco.tools.co_occurrence_matrix` as a matrix.
    
    Parameters
    ----------
    adata
        An :class:`~anndata.AnnData` with the co-occurence analysis in `.uns`.
        Can also be a mapping of labels to :class:`~anndata.AnnData` to specify
        multiple datasets.
    analysis_key
        The `.uns` key with the co-occurence analysis result.
    score_key
        The `.uns[analysis_key]` key of the score to use. Available keys
        `.uns[analysis_key]` include:
        
        - 'occ': co-occurrence
        - 'log_occ': logarithm of the co-occurrence
        - 'log2_occ': base-2-logarithm of the co-occurrence; this is a not a
          real key but a convenience function to rescale the 'log_occ' values
        - 'z': z-score of the log of the neighbourship counts with respect to
          random neighbourships
        - 'composition': distance dependent composition, `p(anno|center, dist)`
        - 'log_composition': log of 'composition'
        - 'distance_distribution': distribution of distances between anno and
          center ´p(dist|anno,center)´
        - 'log_distance_distribution': log of 'distance_distribution'
        - 'relative_distance_distribution': 'distance_distribution' normalized
          to `p(dist|*,center)`, the distance distribution of any annotation to
          the center
        - 'log_relative_distance_distribution': log of
          'relative_distance_distribution'
    log_base
        The base of the logarithm to use for plotting if `score_key` is a log
        quantity. If `None`, use the natural logarithm.
    colors
        The mapping of value names to colors. If `None`, a set of
        standard colors is used.
    show_only
        A subset of annotation values to restrict the plotting to.
    show_only_center
        A subset of the center annotation values to restrict the plotting
        to.
    axsize
        Tuple of width and size of a single axis. If `None`, some heuristic
        value is used.
    hspace, vspace
        Relative horizontal and vertical spacing between plots
    x_padding, y_padding
        Absolute horizontal and vertical spacing between plots; this setting
        overrides `hspace` and `vspace`; if `None`, use the value from `hspace`
        `vspace`; if `None`, use the value from `vspace`
    value_cluster
        Whether to cluster and reorder the values.
    group_cluster
        Whether to cluster and reorder the groups.
    restrict_intervals
        A list-like containing the indices of the intervals to plot. If `None`,
        all intervals are included.
    cmap
        A string/colormap to override the `colors` with globally.
    cmap_vmin_vmax
        A tuple giving the range of values for the colormap.
    legend
        Whether to include the legend
    scale_legend
        Set to scale height and width of legend.
    ax
        The :class:`~matplotlib.axes.Axes` to plot on. If `None`, creates a
        fresh figure for plotting. Incompatible with dendrogram plotting.
        
    Returns
    -------
    A :class:`~matplotlib.figure.Figure`.
    
    """
    
    adatas = _get_adatas(adata)
    
    min_max = None

    if ax is not None:
        if isinstance(ax, matplotlib.axes.Axes):
            axs = np.array([[ax]])
        else:
            axs = ax
        if axs.shape != (len(restrict_intervals), len(adatas)):
            raise ValueError(f'The `ax` argument got the wrong shape of axes: needed is {(len(restrict_intervals), len(adatas))!r} supplied was {axs.shape!r}!')
        axsize = axs[0,0].get_window_extent().transformed(axs[0,0].get_figure().dpi_scale_trans.inverted()).size
        fig = axs[0,0].get_figure()
    else:
        fig, axs = subplots(len(restrict_intervals), len(adatas), axsize=axsize, hspace=hspace, wspace=wspace, x_padding=x_padding, y_padding=y_padding, )

    # first pass through the data to get global min/max of the values for colormap
    for adata_i, (adata_name, adata) in enumerate(adatas.items()):

        mean_scores, intervals, annotation, center, colors, types = _get_co_occurrence(adata, analysis_key, show_only, show_only_center, colors, score_key=score_key, log_base=log_base)
        
        if min_max is None:
            if restrict_intervals is None:
                restrict_intervals = np.arange(len(intervals)-1)
            min_max = np.zeros((len(adatas),len(restrict_intervals),2))
        
        data = mean_scores[:,:,restrict_intervals]
        
        min_max[adata_i,:,:] = data.min(),data.max()
    
    min_max[:,:,0] = min_max[:,:,0].min()
    min_max[:,:,1] = min_max[:,:,1].max()
        
    if cmap_vmin_vmax is not None:
        min_max[:,:,:] = np.array(cmap_vmin_vmax)
    
    if axsize is None:
        axsize = (0.2*len(center),0.2*len(annotation))
    
    # second pass for actual plotting
    for adata_i, (adata_name, adata) in enumerate(adatas.items()):

        mean_scores, intervals, annotation, center, colors, types = _get_co_occurrence(adata, analysis_key, show_only, show_only_center, colors, score_key=score_key, log_base=log_base)
        
        data = mean_scores[:,:,restrict_intervals]
        
        for _ii, rii in enumerate(restrict_intervals):
            interval = f'({intervals[rii]},{intervals[rii+1]})'
            
            ax = axs[adata_i,_ii]

            data = mean_scores[:,:,_ii]

            data = pd.DataFrame(data, index=annotation, columns=center).T

            heatmap(
                data,
                value_key=None, group_key=None,
                colors=colors,
                value_cluster=value_cluster, group_cluster=group_cluster,
                ax=ax,
                cmap=cmap,
                cmap_center=None,#(0 if log else None),
                cmap_vmin_vmax=min_max[adata_i,_ii],
                group_labels_rotation=90,
                colorbar=False,
            );

            adata_title = f'{adata_name}, interval {interval}: ' if adata_name != '' else f'interval {interval}: '

            expression = _get_cooc_expression_label(score_key,log_base)
                
            ax.set_title(adata_title + expression)

            anno_title = 'annotation: ' + _escape_math_special_characters(annotation.name)
            anno_center_title = 'center: ' + _escape_math_special_characters(center.name)
            ax.set_ylabel(anno_title)
            ax.set_xlabel(anno_center_title)

    if legend:
        _add_legend_or_colorbars(fig, axs, colors, cmap=cmap, min_max=min_max, scale_legend=scale_legend)
    
    return fig

def annotated_heatmap(
    adata,
    obs_key=None,
    var_key=None,
    n_genes=None,
    var_highlight=None,
    obs_colors=None,
    var_colors=None,
    cmap='bwr',
    cmap_center=0,
    cmap_vmin_vmax=(-2,2),
    trafo=None,
    axsize=(4,4),
):
    """\
    Plots a heatmap of cells and genes grouped by categorical annotations.
    
    Parameters
    ----------
    adata
        An :class:`~anndata.AnnData` with annotation in `.obs` and/or `.var`.
    obs_key
        The `.obs` key with the categorical `obs` annotation to use. If `None`,
        the observations are not grouped and plotted in the order in which they
        appear in `adata`.
    var_key
        The `.var` key with the categorical `var` annotation to use. Can also
        be a mapping of annotations to list-likes of var names. If `None`, the
        genes are not grouped and plotted in the order in which they appear in
        `adata`. If `n_genes` is set, the meaning of this key is modified.
    n_genes
        The number of differentially expressed genes to find for the groups of
        observations. The differentially exressed genes will be used in place
        of a categorical `var` annotation. Setting `n_genes` changes the
        behaviour of `var_key`: It is interpreted as a categorical `.obs` key
        defining the groups of observations for which to derive the
        differentially expressed genes. If `var_key` is `None`, `obs_key` is
        used for that.
    var_highlight
        A list-like of var names to annotate.
    obs_colors
        A dict-like with the colors to use for the `obs_key` annotation. If
        `None`, default colors are used.
    var_colors
        A dict-like with the colors to use for the `var_key` annotation. If
        `None`, default colors are used, except if `n_genes` is set which
        triggers the usage of `obs_colors` for `var_colors`.
    cmap
        A string/colormap to use in the heatmap.
    cmap_center
        A value to use as center of the colormap. E.g. choosing `0` sets the
        central color to `0`. If `None`, the colormap spans the entire value
        range.
    cmap_vmin_vmax
        A tuple giving the range of values for the colormap. This can be
        modfied by `cmap_center`.
    trafo
        Whether to normalize, logarithmize, and scale the data prior to
        plotting. This makes sense if bare count data is supplied. If the data
        is already preprocessed, this can be set to `False`. If `None`, a
        heuristic tries to figure out whether count data was supplied, and if
        so performs the preprocessing.
    axsize
        Tuple of width and size of the main axis.
        
    Returns
    -------
    A :class:`~matplotlib.figure.Figure`.
    
    """
    
    if trafo is None:
        got_counts = True
        try:
            preprocessing.check_counts_validity(adata.X)
        except:
            got_counts = False
        if got_counts:
            print('`adata` looks like it contains bare counts and will be normalized, logarithmized, and scaled for plotting. If that is not desired (and to get rid of this message) explicitly set `trafo` to something else than `None`.')
            trafo = True
        else:
            trafo = False
    
    if trafo or n_genes:
        adata = adata.copy()
    if trafo:
        sc.pp.normalize_total(adata, target_sum=1e6)
        sc.pp.log1p(adata)
    
    if var_key is not None or n_genes is not None:
        if n_genes is not None:
            if var_key is not None:
                if hasattr(var_key, 'items'):
                    raise ValueError(f'`var_key` cannot be a dict-like if `n_genes` is set!')
                if var_key not in adata.obs:
                    raise ValueError(f'`var_key` {var_key!r} is not a column of `adata.obs` even though `n_genes` is set!')
                if not hasattr(adata.obs[var_key], 'cat'):
                    print(f'WARNING: `var_key` {var_key!r} is not a categorical column of `adata.obs` even though `n_genes` is set! Treating it as a categorical column...')
                group_key = var_key
            elif obs_key is not None:
                if obs_key not in adata.obs:
                    raise ValueError(f'`obs_key` {obs_key!r} is not a column of `adata.obs`!')
                if not hasattr(adata.obs[obs_key], 'cat'):
                    print(f'WARNING: `obs_key` {obs_key!r} is not a categorical column of `adata.obs`! Treating it as a categorical column...')
                group_key = obs_key
                if var_colors is None:
                    var_colors = obs_colors
            else:
                raise ValueError(f'`n_genes` can only be used if at least one of [`obs_key`, `var_key`] is not `None`!')
                
            ukey = utils.find_unused_key(adata.uns)
            sc.tl.rank_genes_groups(adata, group_key, key_added=ukey, n_genes=n_genes,)
            marker = pd.DataFrame(adata.uns[ukey]['names'])
            del adata.uns[ukey]
                
        else: # if var_key is not None:
            if hasattr(var_key, 'items'):
                marker = var_key
            else:
                if var_key not in adata.var:
                    raise ValueError(f'`var_key` {var_key!r} is not a column of `adata.var`!')
                if not hasattr(adata.var[var_key], 'cat'):
                    print(f'WARNING: `var_key` {var_key!r} is not a categorical column of `adata.var`! Treating it as a categorical column...')
                marker = {c: df.index for c,df in adata.var.groupby(var_key)}

        all_marker = [c for l,m in marker.items() for c in m]
        # reorder genes to represent the annotation
        adata = adata[:,all_marker]
        
        if var_colors is None:
            var_colors = to_rgba_array(get_default_colors(len(marker)))
            var_colors = {cat:col for cat,col in zip(marker.keys(),var_colors)}
        else:
            var_colors = {cat:to_rgba(col) for cat,col in var_colors.items()}
            
        all_marker_colors = [var_colors[l] for l,m in marker.items() for g in m]
        
        all_marker_labels = [l for l,m in marker.items() for c in m]
        marker_centers = { l: np.median(np.arange(len(all_marker_labels))[np.array(all_marker_labels) == l]) for l in marker.keys() }
    
    if obs_key is not None:
        if obs_key not in adata.obs:
            raise ValueError(f'`obs_key` {obs_key!r} is not a column of `adata.obs`!')
        if not hasattr(adata.obs[obs_key], 'cat'):
            print(f'WARNING: `obs_key` {obs_key!r} is not a categorical column of `adata.obs`! Treating it as a categorical column...')

        cells = {c: df.index for c,df in adata.obs.groupby(obs_key)}
        all_cells = [c for l,m in cells.items() for c in m]
        # reorder cells to represent the annotation
        adata = adata[all_cells]
        
        if obs_colors is None:
            obs_colors = to_rgba_array(get_default_colors(len(cells)))
            obs_colors = {cat:col for cat,col in zip(cells.keys(),obs_colors)}
        else:
            obs_colors = {cat:to_rgba(col) for cat,col in obs_colors.items()}
            
        all_cell_colors = [obs_colors[l] for l,m in cells.items() for c in m]
        
        all_cell_labels = [l for l,m in cells.items() for c in m]
        cell_centers = { l: np.median(np.arange(len(all_cell_labels))[np.array(all_cell_labels) == l]) for l in cells.keys() }
    
    if var_highlight is not None:
        # reorder highlighted genes to simplify labelling
        var_highlight = pd.Series(adata.var.index)[adata.var.index.isin(var_highlight)]
        highlight_centers = pd.Series(var_highlight.index, index=var_highlight )

    if trafo:
        if adata.is_view:
            adata = adata.copy()
        sc.pp.scale(adata)
    
    data = adata.X
    if scipy.sparse.issparse(data):
        data = data.A
    
    if cmap_vmin_vmax is None:
        cmap_vmin_vmax = [data.min(),data.max()]
    cmap_vmin_vmax = np.array(cmap_vmin_vmax)
    if cmap_center is not None:
        shifted = cmap_vmin_vmax - cmap_center
        abs_max = np.max(np.abs(shifted))
        shifted[:] = [-abs_max,abs_max]
        cmap_vmin_vmax = shifted + cmap_center

    # plotting
    fig,axs=subplots(2,2,axsize=axsize,width_ratios=[0.03,axsize[0]/4],height_ratios=[0.03,axsize[1]/4],x_padding=0.05,y_padding=0.05)
    axs[0,0].axis('off')
    im = axs[1,1].imshow(data.T,aspect='auto',cmap=cmap, vmin=cmap_vmin_vmax[0], vmax=cmap_vmin_vmax[1])
    axs[1,1].set_xticks([])
    axs[1,1].set_yticks([])
    rel_dpi_factor = matplotlib.rcParams['figure.dpi'] / 72
    cax_width = 100 * rel_dpi_factor # color bar width in pixel
    cax_height = 10 * rel_dpi_factor # color bar height in pixel
    cax_offset = 10 * rel_dpi_factor # color bar y offset in pixel
    cax_l = 1 - fig.transFigure.inverted().transform([cax_width,0])[0] + fig.transFigure.inverted().transform([0,0])[0]
    cax_b = 0 - fig.transFigure.inverted().transform([0,cax_height+cax_offset])[1] + fig.transFigure.inverted().transform([0,0])[1]
    cax_w = fig.transFigure.inverted().transform([cax_width,0])[0] - fig.transFigure.inverted().transform([0,0])[0]
    cax_h = fig.transFigure.inverted().transform([0,cax_height])[1] - fig.transFigure.inverted().transform([0,0])[1]
    cax = fig.add_axes((cax_l,cax_b,cax_w,cax_h))
    fig.colorbar(im, cax=cax, orientation='horizontal')

    # labelling
    def _collides(ann1, ann2, offset1, offset2, fig, direction):
        if offset1 != offset2:
            return False
        extent1 = ann1.get_window_extent(fig.canvas.get_renderer())
        extent2 = ann2.get_window_extent(fig.canvas.get_renderer())
        if direction == 'x':
            return extent2.x1 > extent1.x0
        else:
            return extent1.y1 > extent2.y0

    def _collides_any(ann, off, anns, offs, fig, direction):
        for ann2, off2 in zip(anns, offs):
            if _collides(ann, ann2, off, off2, fig, direction):
                return True
        return False
    def _find_offset(ann, anns, offs, fig, direction):
        off = 0
        while _collides_any(ann, off, anns, offs, fig, direction):
            off += 1
        return off
    def _find_shift(ann, anns, fig, direction):
        if len(anns) == 0:
            return 0
        extent1 = ann.get_window_extent(fig.canvas.get_renderer())
        extent2 = anns[-1].get_window_extent(fig.canvas.get_renderer())
        if direction == 'x':
            delta = extent2.x1 - extent1.x0
        else:
            delta = extent1.y1 - extent2.y0
        if delta > 0:
            return delta
        else:
            return 0

    if obs_key is not None:
        axs[0,1].imshow(np.array([all_cell_colors]),aspect='auto')
        axs[0,1].set_xticks([])
        axs[0,1].set_yticks([])

        anns = []
        offs = []
        
        bar = (axs[0,1].transData.inverted().transform([0,-15*rel_dpi_factor])[1] - axs[0,1].transData.inverted().transform([0,0])[1])
        for l,c in cell_centers.items():
            ann = axs[0,1].annotate(l, (c, -0.5), (c, -0.5-bar), ha="center", va="center", rotation=0, size=10, arrowprops={'arrowstyle':'-'},)
            offset = _find_offset(ann, anns, offs, fig, direction='x')
            ann.xyann = (ann.xyann[0],ann.xyann[1]-offset*1.8)
            anns.append(ann)
            offs.append(offset)
    else:
        axs[0,1].axis('off')

    if var_key is not None or n_genes is not None:
        axs[1,0].imshow(np.array([all_marker_colors]).swapaxes(0,1),aspect='auto')
        axs[1,0].set_xticks([])
        axs[1,0].set_yticks([])

        anns=[]
        bar = (axs[1,0].transData.inverted().transform([15*rel_dpi_factor,0])[0] - axs[1,0].transData.inverted().transform([0,0])[0])
        for l,c in marker_centers.items():
            ann = axs[1,0].annotate(l, (-0.5, c), (-0.5-bar, c), ha="right", va="center", rotation=0, size=10, arrowprops={'arrowstyle':'-'},)
            shift = _find_shift(ann, anns, fig, direction='y')
            shift = (axs[1,0].transData.inverted().transform([0,shift])[1] - axs[1,0].transData.inverted().transform([0,0])[1])
            ann.xyann = (ann.xyann[0],ann.xyann[1]-shift)
            anns.append(ann)
    else:
        axs[1,0].axis('off')

    if var_highlight is not None:
        anns=[]
        bar = (axs[1,1].transData.inverted().transform([15*rel_dpi_factor,0])[0] - axs[1,1].transData.inverted().transform([0,0])[0])
        offset = (axs[1,1].transData.inverted().transform([0*rel_dpi_factor,0])[0] - axs[1,1].transData.inverted().transform([0,0])[0]) - 0.5
        for l,c in highlight_centers.items():
            ann = axs[1,1].annotate(l, (len(adata.obs.index)+offset, c), (len(adata.obs.index)+offset+bar, c), ha="left", va="center", rotation=0, size=10, arrowprops={'arrowstyle':'-'}, annotation_clip=False,)
            shift = _find_shift(ann, anns, fig, direction='y')
            shift = (axs[1,1].transData.inverted().transform([0,shift])[1] - axs[1,1].transData.inverted().transform([0,0])[1])
            ann.xyann = (ann.xyann[0],ann.xyann[1]-shift)
            anns.append(ann)
    
    return fig

@njit(parallel=False,fastmath=True,cache=True)
def _anno_hist(mist, intervals, weights, anno):
    Nobs,Nanno = anno.shape
    assert(Nobs==len(mist))
    assert(Nobs==len(weights))
    Nd = len(intervals)-1
    hist = np.zeros((Nd,))
    sums = np.zeros((Nd,Nanno))
    for i in range(Nobs):
        di = mist[i]
        _di = np.argmax(di <= intervals) - 1
        if di > 0:
            hist[_di] += weights[i]
            sums[_di] += weights[i] * anno[i]
    for d in range(Nd):
        sums[d] /= hist[d]
    return sums

def annotation_coordinate(
    adata,
    annotation_key,
    coordinate_key,
    group_key=None,
    reference_key=None,
    max_coordinate=None,
    delta_coordinate=None,
    axsize=None,
    colors=None,
    stacked=True,
    verbose=1,
):
    """\
    Plots an annotation density with respect to a scalar coordinate.
    
    Parameters
    ----------
    adata
        A :class:`~anndata.AnnData`.
    annotation_key
        The `.obs` or `.obsm` key to plot.
    coordinate_key
        The `.obs` key or (`.obsm` key, column name) pair with the scalar
        coordinate(s).
    group_key
        A categorical group annotation. The plot is done separately per group.
        If `None`, plots for only one group are generated.
    reference_key
        The `.obs` key to use as weights (i.e. the weight to use for
        calculating the annotation density). If `None`, use `1` per
        observation, which makes sense if the annotation is categorical or
        fractional annotations which should sum to `1`.
    max_coordinate
        The maximum coordinate to use. If `None` or `np.inf`, uses the maximum
        coordinate in the data.
    delta_coordinate
        The width in coordinate for coordinate discretization. If `None`, takes
        `max_coordinate/100`.
    axsize
        Size of a single axis in the plot
    colors
        A mapping of annotation values to colors. If `None`, default colors are
        used.
    stacked
        Whether to plot the different annotations on different scales on
        separate stacked plots or on the same scale in a single plot.
    verbose
        Level of verbosity, with `0` (no output), `1` (some output), ...
        
    Returns
    -------
    A :class:`~matplotlib.figure.Figure`.
    
    """
    
    if group_key is None:
        group_adatas = {'':adata}
    else:
        group_adatas = {group:adata[df.index] for group,df in adata.obs.groupby(group_key) if len(df)>0 }
    
    if annotation_key in adata.obs:
        annotation = adata.obs[annotation_key]
        if hasattr(annotation, 'cat'):
            annotation = pd.get_dummies(annotation)
        else:
            annotation = pd.DataFrame(annotation)
    elif annotation_key in adata.obsm:
        annotation = adata.obsm[annotation_key].copy()
    else:
        raise ValueError(f'The `annotation_key` {annotation_key!r} is neither in `adata.obs` nor `adata.obsm`!')
    
    if pd.api.types.is_list_like(coordinate_key):
        if len(coordinate_key) == 2 and coordinate_key[0] in adata.obsm and coordinate_key[1] in adata.obsm[coordinate_key[0]]:
            coordinates = adata.obsm[coordinate_key[0]][coordinate_key[1]]
        else:
            raise ValueError(f'The `coordinate_key` {coordinate_key!r} is list/like, but not something of length 2 containing a `adata.obsm` key and a column name therein!')
        #coordinate_key = f'{coordinate_key[0]}:{coordinate_key[1]}'
        coordinate_key = f'{coordinate_key[1]}'
    elif coordinate_key in adata.obs:
        coordinates = adata.obs[coordinate_key]
    else:
        raise ValueError(f'The `coordinate_key` {coordinate_key!r} is not in `adata.obs` and not a valid specification for something in `adata.obsm`!')

    if max_coordinate is None or max_coordinate == np.inf:
        max_coordinate = coordinates.to_numpy().max()
    max_coordinate = float(max_coordinate)
    
    if delta_coordinate is None:
        delta_coordinate = max_coordinate / 100
    n_intervals = int(max_coordinate / delta_coordinate)
    max_coordinate = n_intervals * delta_coordinate
    intervals = np.arange(0,max_coordinate+delta_coordinate*0.5,delta_coordinate)
    midpoints = (intervals[1:] + intervals[:-1]) * 0.5
    
    if reference_key is None:
        reference_weights = pd.Series(np.ones(len(adata.obs.index),dtype=float),index=adata.obs.index)
    elif reference_key in adata.obs:
        reference_weights = adata.obs[reference_key]
    else:
        raise ValueError(f'The `reference_key` {reference_key!r} is not in `adata.obs`!')
    
    annotation_categories = annotation.columns
    colors,annotation_categories = _get_colors(colors, annotation_categories)
    
    if stacked:
        if axsize is None:
            axsize = (6,0.7)
        fig,axs = subplots(len(group_adatas),len(annotation.columns),axsize=axsize,sharex=True,sharey='row',y_padding=0,x_padding=0)
        
    else:
        if axsize is None:
            axsize = (6,4)
        fig,axs = subplots(len(group_adatas),1,axsize=axsize,sharex=True,sharey='row',y_padding=0,x_padding=0)


    for gi,(group,group_adata) in enumerate(group_adatas.items()):

        group_annotation = annotation.loc[group_adata.obs.index].to_numpy()
        group_coordinates = coordinates.loc[group_adata.obs.index].to_numpy()
        group_reference_weights = reference_weights.loc[group_adata.obs.index].to_numpy()

        no_nan = ~np.isnan(group_annotation).any(axis=1)
        anno = _anno_hist(group_coordinates[no_nan], intervals, group_reference_weights[no_nan], group_annotation[no_nan])
        anno = pd.DataFrame(anno,index=midpoints,columns=annotation.columns)

        if stacked:
            for i,c in enumerate(colors.index):
                axs[i,gi].plot(anno[c],c=colors[c],label=c)
                axs[i,gi].xaxis.grid(True)
        else:
            for i,c in enumerate(colors.index):
                axs[0,gi].plot(anno[c],c=colors[c],label=c)
            axs[0,gi].xaxis.grid(True)

        group_string = f' in {group}' if group != '' else ''
        axs[0,gi].set_title(f'{annotation_key} VS distance from {coordinate_key}{group_string}')
        
    axs[0,-1].legend(handles=[mpatches.Patch(color=color, label=ind) for (ind, color) in colors.items() ],
        bbox_to_anchor=(1, 1), loc='upper left', ncol=1)
    
    return fig

def dotplot(
    adata,
    genes,
    group_key,
    log1p=True,
    marks=None,
    marks_colors=None,
):
    
    """\
    Dot plot of expression values.
    
    This is similar to :func:`scanpy.pl.dotplot` with customizations, e.g. the
    option to mark selected dots.
    
    Parameters
    ----------
    adata
        An :class:`~anndata.AnnData` including annotation in `.obs`.
    genes
        The `.var` index values to compare use as a list-like.
    group_key
        An `.obs` key with categorical group information.
    log1p
        Whether to log1p-transform the data prior to plotting.
    marks
        A :class:`pandas.DataFrame` containing categorical markers for the dots
        with genes in the rows and groups in the columns.
    marks_colors
        A mapping from the categories in `marks` to colors; if `None`, default
        colors are used.
        
    Returns
    -------
    A :class:`~matplotlib.figure.Figure`.
    
    """
    
    if not pd.Index(genes).isin(adata.var.index).all():
        raise ValueError(f'The genes {pd.Index(genes).difference(adata.var.index)!r} are not available in `adata.var`!')
        
    markers = genes[::-1]
    
    if group_key not in adata.obs.columns:
        raise ValueError(f'The `group_key` {group_key!r} is not available in `adata.obs`!')
    
    if hasattr(adata.obs[group_key], 'cat'):
        cluster = adata.obs[group_key].cat.categories
    else:
        cluster = adata.obs[group_key].unique()
    
    fig,axs = subplots(axsize=0.25*np.array([len(cluster),len(markers)]))

    x = np.arange(len(cluster))  # the label locations
    axs[0,0].set_xticks(x)
    axs[0,0].set_xticklabels(cluster, rotation=45, ha='right',)
    y = np.arange(len(markers))  # the label locations
    axs[0,0].set_yticks(y)
    axs[0,0].set_yticklabels(markers)
    axs[0,0].set_xlim((x.min()-0.5,x.max()+0.5))
    axs[0,0].set_ylim((y.min()-0.5,y.max()+0.5))

    axs[0,0].set_axisbelow(True)
    axs[0,0].grid(True)
    
    marker_counts = adata[:,markers].to_df()
    if log1p:
        marker_counts = np.log1p(marker_counts)
    mean_exp = pd.DataFrame({c: marker_counts.loc[df.index].mean(axis=0) for c,df in adata.obs.groupby(group_key) })
    mean_pos = pd.DataFrame({c: (marker_counts.loc[df.index] != 0).mean(axis=0) for c,df in adata.obs.groupby(group_key) })
    
    if marks is not None:
        marks = marks.reindex_like(mean_pos)

    mean_exp_index_name = 'index' if mean_exp.index.name is None else mean_exp.index.name
    mean_pos_index_name = 'index' if mean_pos.index.name is None else mean_pos.index.name
    mean_exp = pd.melt(mean_exp, ignore_index=False).reset_index().rename(columns={mean_exp_index_name:'value','variable':'cluster','value':'mean_exp'})
    mean_pos = pd.melt(mean_pos, ignore_index=False).reset_index().rename(columns={mean_pos_index_name:'value','variable':'cluster','value':'mean_pos'})
    
    if marks is not None:
        marks.index.name = None
        marks.columns.name = None
        marks_index_name = 'index' if marks.index.name is None else marks.index.name
        marks = pd.melt(marks, ignore_index=False).reset_index().rename(columns={marks_index_name:'value','variable':'cluster','value':'marks'})
        if marks_colors is None:
            marks_colors = get_default_colors(marks['marks'].unique())

    all_df = pd.merge(mean_exp, mean_pos, on=['value', 'cluster'])
    
    if marks is not None:
        all_df = pd.merge(all_df, marks, on=['value', 'cluster'], how='outer')
        
    all_df['x'] = all_df['cluster'].map(pd.Series(x,index=cluster))
    all_df['y'] = all_df['value'].map(pd.Series(y,index=markers))

    legend_items = []
    
    mean_exp_min, mean_exp_max = all_df['mean_exp'].min(), all_df['mean_exp'].max()
    norm = Normalize(vmin=mean_exp_min, vmax=mean_exp_max)
    cmap='Reds'#LinearSegmentedColormap.from_list('mean_exp', [(0,(1, 1, 1)),(1,(1, g, b))])
    mapper = ScalarMappable(norm=norm, cmap=cmap)
    color = [ tuple(x) for x in mapper.to_rgba(all_df['mean_exp'].to_numpy()) ]
    
    legend_items.append(mpatches.Patch(color='#0000', label='mean expression'))
    mean_exp_for_legend = np.linspace(mean_exp_min, mean_exp_max, 4)
    legend_items.extend([mpatches.Patch(color=color, label=f'{ind:.2f}') for color,ind in zip(mapper.to_rgba(mean_exp_for_legend),mean_exp_for_legend)])

    mean_pos_min, mean_pos_max = all_df['mean_pos'].min(), all_df['mean_pos'].max()
    def size_map(x):
        return (x/mean_pos_max * 14)**2
    size = size_map(all_df['mean_pos'])
    
    legend_items.append(mpatches.Patch(color='#0000', label='fraction of expressing cells'))
    mean_pos_for_legend = np.linspace(mean_pos_min, mean_pos_max, 5)[1:]
    legend_items.extend([mlines.Line2D([], [], color='#aaa', linestyle='none', marker='o', markersize=np.sqrt(size_map(ind)), label=f'{ind:.2f}') for ind in mean_pos_for_legend])

    edgecolors = '#aaa' if marks is None else all_df['marks'].map(marks_colors)
    
    if marks is not None:
        marks_name = marks_colors.name if hasattr(marks_colors, 'name') else ''
        legend_items.append(mpatches.Patch(color='#0000', label=marks_name))
        legend_items.extend([mlines.Line2D([], [], color='#aaa', linestyle='none', fillstyle='none', markeredgecolor=color, marker='o', markersize=np.sqrt(size_map(mean_pos_for_legend[-2])), label=f'{ind}') for ind,color in marks_colors.items()])

    axs[0,0].scatter(all_df['x'], all_df['y'], c=color, s=size, edgecolors=edgecolors)
    
    axs[0,0].legend(handles=legend_items, bbox_to_anchor=(1, 1), loc='upper left', ncol=1)
    
    return fig
