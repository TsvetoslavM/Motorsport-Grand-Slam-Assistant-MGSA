import argparse
import pandas as pd
import altair as alt
import numpy as np
import os
from functools import reduce

MODERN_STYLE = '''
<style>
body {
    font-family: 'Inter', 'Segoe UI', Arial, sans-serif;
    background: #f5f8fa;
    margin: 0;
}
.header {
    background: #060e23;
    color: #fff;
    padding: 2rem 0 1.2rem 0;
    text-align: center;
    box-shadow: 0 4px 28px rgba(0,0,0,0.07);
    letter-spacing: .02em;
}
.container {
    max-width: 620px;
    margin: 2em auto 3em auto;
    padding: 2em 2.3em 2em 2.3em;
    background: #fff;
    border-radius: 22px;
    box-shadow: 0 2px 28px rgba(29,69,120,0.09);
}
@media (max-width: 600px) {
  .container {padding: .6em;}
}
h1 {margin:0 0 0.7em 0;font-size:2.3em;font-weight:800;letter-spacing:-0.03em;}
p.title-info{color:#757aa3;margin-top:0;font-size:1.13em;}
</style>
<link href="https://fonts.googleapis.com/css2?family=Inter:wght@400;700;800&display=swap" rel="stylesheet">
'''

def read_and_prepare(lines):
    dfs = []
    for fname, label in lines:
        df = pd.read_csv(fname)
        if 's_m' not in df:
            dx = np.diff(df['x_m'].values)
            dy = np.diff(df['y_m'].values)
            ds = np.sqrt(dx*dx + dy*dy)
            s = np.concatenate(([0.0], np.cumsum(ds)))
            df['s_m'] = s
        df['label'] = label
        df['index'] = np.arange(len(df))
        dfs.append(df)
    ref_s = dfs[0]['s_m'].values
    dfs_interp = []
    for df in dfs:
        interp = pd.DataFrame({'s_m': ref_s})
        for col in ['x_m', 'y_m', 'speed_kmh', 'time_s']:
            interp[col] = np.interp(ref_s, df['s_m'].values, df[col].values)
        interp['label'] = df['label'].iloc[0]
        interp['index'] = np.arange(len(interp))
        dfs_interp.append(interp)
    return pd.concat(dfs_interp, ignore_index=True)

def load_track_edges(track_csv):
    df = pd.read_csv(track_csv, comment="#", names=["x_m","y_m","w_tr_right_m","w_tr_left_m"])
    x = df['x_m'].values
    y = df['y_m'].values
    dx = np.gradient(x)
    dy = np.gradient(y)
    norm = np.sqrt(dx**2 + dy**2)
    tx = dx / norm
    ty = dy / norm
    nx = -ty
    ny = tx
    x_left = x + nx * df['w_tr_left_m'].values
    y_left = y + ny * df['w_tr_left_m'].values
    x_right = x - nx * df['w_tr_right_m'].values
    y_right = y - ny * df['w_tr_right_m'].values
    left_df = pd.DataFrame({'x': x_left, 'y': y_left})
    right_df = pd.DataFrame({'x': x_right, 'y': y_right})
    return left_df, right_df

def add_delay_info(bigdf, base_label):
    """Add delay information relative to the base (first) line"""
    # Create pivot table for time at each distance
    pivot = bigdf.pivot(index='s_m', columns='label', values='time_s').sort_index().interpolate()
    base_times = pivot[base_label]
    
    # Calculate delays for each line
    delay_data = []
    for label in pivot.columns:
        if label != base_label:
            delays = pivot[label] - base_times
            delay_data.append(pd.DataFrame({
                's_m': pivot.index,
                'label': label,
                'delay_s': delays
            }))
    
    if delay_data:
        delay_df = pd.concat(delay_data, ignore_index=True)
        # Merge delays back into bigdf
        bigdf = bigdf.merge(delay_df, on=['s_m', 'label'], how='left')
        # For the base label, delay is 0
        bigdf.loc[bigdf['label'] == base_label, 'delay_s'] = 0.0
    else:
        bigdf['delay_s'] = 0.0
    
    return bigdf

def format_time_display(seconds):
    """Format seconds as MM:SS.SS"""
    minutes = int(seconds // 60)
    secs = seconds % 60
    return f"{minutes}:{secs:05.2f}"

def create_track_overlay(bigdf, left_edge=None, right_edge=None):
    # Add formatted columns for tooltips
    bigdf['Speed'] = bigdf['speed_kmh'].apply(lambda x: f"{x:.2f} km/h")
    bigdf['Time'] = bigdf['time_s'].apply(format_time_display)
    bigdf['Distance'] = bigdf['s_m'].apply(lambda x: f"{x:.2f} m")
    
    # Format delay with +/- sign
    bigdf['Delay'] = bigdf['delay_s'].apply(
        lambda x: f"{x:+.2f} sec" if pd.notna(x) and x != 0 else "0.00 sec (reference)"
    )
    
    elements = []
    # Draw left and right edges as ordered lines, no fills
    if left_edge is not None:
        elements.append(
            alt.Chart(left_edge.reset_index()).mark_line(
                color='#808080', strokeWidth=2, opacity=0.25, strokeDash=[1,0], tooltip=False
            ).encode(
                x='x', y='y', order='index:Q'
            )
        )
    if right_edge is not None:
        elements.append(
            alt.Chart(right_edge.reset_index()).mark_line(
                color='#808080', strokeWidth=2, opacity=0.25, strokeDash=[1,0], tooltip=False
            ).encode(
                x='x', y='y', order='index:Q'
            )
        )
    
    base = alt.Chart(bigdf).mark_line().encode(
        x=alt.X('x_m', axis=alt.Axis(title='x_m')),
        y=alt.Y('y_m', axis=alt.Axis(title='y_m')),
        color=alt.Color('label:N', title='Line'),
        detail='label',
        order='index',
        tooltip=['label:N', 'Speed:N', 'Distance:N', 'Time:N', 'Delay:N']
    )
    
    firsts = bigdf.loc[bigdf['index'] == 0]
    start_pts = alt.Chart(firsts).mark_point(filled=True, color='lime', stroke='black', size=70).encode(
        x='x_m', y='y_m', tooltip=['label']
    )
    elements.append(base + start_pts)
    chart = alt.layer(*elements).properties(width=450, height=450, title='Track overlay (edges + lines)').interactive()
    return chart

def create_speed_vs_distance(bigdf):
    # Add formatted columns
    bigdf['Speed'] = bigdf['speed_kmh'].apply(lambda x: f"{x:.2f} km/h")
    bigdf['Distance'] = bigdf['s_m'].apply(lambda x: f"{x:.2f} m")
    bigdf['Delay'] = bigdf['delay_s'].apply(
        lambda x: f"{x:+.2f} sec" if pd.notna(x) and x != 0 else "0.00 sec (reference)"
    )
    
    base = alt.Chart(bigdf).encode(
        x=alt.X('s_m', title='Distance [m]'), color='label:N')
    lines = base.mark_line().encode(
        y=alt.Y('speed_kmh', title='Speed [km/h]'), 
        tooltip=['label:N', 'Distance:N', 'Speed:N', 'Delay:N']
    )
    return lines.properties(width=350, height=200, title='Speed vs Distance')

def create_time_vs_distance(bigdf):
    # Add formatted columns
    bigdf['Time'] = bigdf['time_s'].apply(format_time_display)
    bigdf['Distance'] = bigdf['s_m'].apply(lambda x: f"{x:.2f} m")
    bigdf['Delay'] = bigdf['delay_s'].apply(
        lambda x: f"{x:+.2f} sec" if pd.notna(x) and x != 0 else "0.00 sec (reference)"
    )
    
    base = alt.Chart(bigdf).encode(
        x=alt.X('s_m', title='Distance [m]'), color='label:N')
    lines = base.mark_line().encode(
        y=alt.Y('time_s', title='Cumulative time [s]'), 
        tooltip=['label:N', 'Distance:N', 'Time:N', 'Delay:N']
    )
    return lines.properties(width=350, height=200, title='Cumulative Time vs Distance')

def create_delta_time_vs_distance(bigdf, base_label):
    pivot = bigdf.pivot(index='s_m', columns='label', values='time_s').sort_index().interpolate()
    base = pivot[base_label]
    delta_df = pivot.subtract(base, axis=0).reset_index()
    melt = delta_df.melt(id_vars=['s_m'], var_name='label', value_name='delta_time_s')
    melt = melt[melt['label'] != base_label]
    
    # Add formatted columns
    melt['Distance'] = melt['s_m'].apply(lambda x: f"{x:.2f} m")
    melt['Delay'] = melt['delta_time_s'].apply(lambda x: f"{x:+.2f} sec")
    
    base_chart = alt.Chart(melt).mark_line().encode(
        x=alt.X('s_m', title='Distance [m]'), 
        y=alt.Y('delta_time_s', title=f'Δ time to {base_label} [s]'),
        color='label:N', 
        tooltip=['label:N', 'Distance:N', 'Delay:N']
    )
    zero_line = alt.Chart(pd.DataFrame({'y': [0.0]})).mark_rule(color='white', opacity=0.5).encode(y='y')
    return (base_chart + zero_line).properties(width=350, height=200, title='Delta Time vs Distance')

def save_with_modern_style(vega_html, output_html):
    header = '''<div class="header"><h1>Racing Line Comparison</h1><p class="title-info">Visual analytics for lap data &mdash; modern style Altair plots</p></div>\n'''
    container_start = '<div class="container">\n'
    container_end = '\n</div>'
    with open(output_html, 'w', encoding='utf8') as f:
        f.write('<!DOCTYPE html>\n<html lang="en">\n<head>\n<meta charset="utf-8">\n<meta name="viewport" content="width=device-width,initial-scale=1">\n' + MODERN_STYLE + '\n<title>Racing Line Comparison</title>\n</head>\n<body>\n')
        f.write(header)
        f.write(container_start)
        f.write(vega_html)
        f.write(container_end)
        f.write('\n</body>\n</html>')

def main():
    parser = argparse.ArgumentParser(description='Create static Altair HTML comparison of N racing lines.')
    parser.add_argument('--inputs', nargs='+', type=str, required=True, help='Pairs: csv label csv label ...')
    parser.add_argument('--output_html', type=str, required=True, help='Output HTML file path.')
    parser.add_argument('--track_csv', type=str, default=None, help='(Optional) path to track reference CSV (for underlying track edges)')
    args = parser.parse_args()
    if len(args.inputs) < 2 or len(args.inputs) % 2 != 0:
        raise ValueError('Must provide pairs: csv label csv label ... (at least two lines)')
    lines = list(zip(args.inputs[0::2], args.inputs[1::2]))
    bigdf = read_and_prepare(lines)
    base_label = lines[0][1]
    
    # Add delay information relative to first line
    bigdf = add_delay_info(bigdf, base_label)
    
    left_edge = right_edge = None
    if args.track_csv:
        left_edge, right_edge = load_track_edges(args.track_csv)
    
    c1 = create_track_overlay(bigdf, left_edge, right_edge)
    c2 = create_speed_vs_distance(bigdf)
    c3 = create_time_vs_distance(bigdf)
    c4 = create_delta_time_vs_distance(bigdf, base_label)
    final = alt.vconcat(c1, c2, c3, c4).resolve_legend(color='independent')

    import tempfile
    with tempfile.NamedTemporaryFile('w+', delete=False, suffix='.html', encoding='utf8') as tmpf:
        final.save(tmpf.name)
        tmpf.seek(0)
        vega_html = tmpf.read()
    save_with_modern_style(vega_html, args.output_html)
    print(f'Wrote: {os.path.abspath(args.output_html)}')

if __name__ == '__main__':
    main()