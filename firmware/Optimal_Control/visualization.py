import numpy as np
import matplotlib.pyplot as plt
from matplotlib.collections import LineCollection

def plot_f1_results(
    x_center, y_center, w_left, w_right, normals,
    n_opt, v_opt, a_lon_opt, a_lat_opt, ds_array,
    vehicle, lap_time_seconds, track_length, N
):
    """
    Създава пълна визуализация на F1 оптимизацията.
    
    Parameters:
    -----------
    x_center, y_center : array
        Координати на централната линия на трасето
    w_left, w_right : array
        Ширини на лявата/дясна граница
    normals : array
        Нормални вектори към трасето
    n_opt : array
        Оптимално отклонение от централната линия
    v_opt : array
        Оптимална скорост
    a_lon_opt : array
        Оптимално надлъжно ускорение
    a_lat_opt : array
        Оптимално странично ускорение
    ds_array : array
        Дължини на сегментите
    vehicle : VehicleParams
        Параметри на автомобила
    lap_time_seconds : float
        Време за обиколка
    track_length : float
        Дължина на трасето
    N : int
        Брой сегменти
    """
    
    # Изчисляване на допълнителни метрики
    x_opt = x_center + n_opt * normals[:, 0]
    y_opt = y_center + n_opt * normals[:, 1]
    
    s = np.cumsum(np.concatenate([[0], ds_array[:-1]]))
    a_total = np.sqrt(a_lon_opt**2 + a_lat_opt**2)
    
    downforce_g = vehicle.k_aero() * v_opt**2 / vehicle.gravity
    F_drag = vehicle.k_drag() * v_opt**2
    power_used = vehicle.mass_kg * a_lon_opt * v_opt + F_drag * v_opt
    
    # Sector анализ
    sectors = 3
    sector_times = []
    sector_speeds = []
    for sector in range(sectors):
        start_idx = int(sector * N / sectors)
        end_idx = int((sector + 1) * N / sectors)
        sector_time = sum(ds_array[start_idx:end_idx] / v_opt[start_idx:end_idx])
        sector_avg_speed = np.mean(v_opt[start_idx:end_idx])
        sector_times.append(sector_time)
        sector_speeds.append(sector_avg_speed * 3.6)
    
    # Границы на трасето
    x_left_bound = x_center - w_left * normals[:, 0]
    y_left_bound = y_center - w_left * normals[:, 1]
    x_right_bound = x_center + w_right * normals[:, 0]
    y_right_bound = y_center + w_right * normals[:, 1]
    
    # ===== СЪЗДАВАНЕ НА ФИГУРАТА =====
    plt.style.use('dark_background')
    fig = plt.figure(figsize=(16, 12))
    gs = fig.add_gridspec(3, 4, hspace=0.35, wspace=0.30)
    
    # ===== ROW 1-2, COL 1-2: ОПТИМАЛНА ЛИНИЯ =====
    ax1 = fig.add_subplot(gs[0:2, 0:2])
    ax1.fill(np.concatenate([x_left_bound, x_right_bound[::-1]]),
             np.concatenate([y_left_bound, y_right_bound[::-1]]),
             color='#1a1a1a', edgecolor='white', linewidth=2, alpha=0.8)
    
    points = np.array([x_opt, y_opt]).T.reshape(-1, 1, 2)
    segments = np.concatenate([points[:-1], points[1:]], axis=1)
    lc = LineCollection(segments, cmap='plasma', linewidth=4)
    lc.set_array(v_opt[:-1] * 3.6)
    line = ax1.add_collection(lc)
    cbar = plt.colorbar(line, ax=ax1, pad=0.02)
    cbar.set_label('Speed [km/h]', rotation=270, labelpad=20, fontsize=11)
    
    ax1.scatter(x_opt[0], y_opt[0], c='lime', s=300, marker='s', 
               edgecolors='white', linewidth=2, zorder=10, label='Start/Finish')
    
    arrow_spacing = max(N // 12, 1)
    for i in range(0, N, arrow_spacing):
        if i < N-1:
            dx = x_opt[(i+1)%N] - x_opt[i]
            dy = y_opt[(i+1)%N] - y_opt[i]
            norm = np.sqrt(dx**2 + dy**2) + 1e-6
            ax1.arrow(x_opt[i], y_opt[i], 3*dx/norm, 3*dy/norm, 
                     head_width=2, head_length=2.5, fc='cyan', ec='cyan', 
                     alpha=0.6, linewidth=1.5)
    
    ax1.set_xlabel("X [m]", fontsize=13, fontweight='bold')
    ax1.set_ylabel("Y [m]", fontsize=13, fontweight='bold')
    ax1.axis('equal')
    ax1.legend(fontsize=11, loc='upper right')
    ax1.set_title("F1 Optimal Racing Line", fontsize=15, fontweight='bold', pad=15)
    ax1.grid(True, alpha=0.2, linestyle='--')
    
    # ===== ROW 1-2, COL 3: СТАТИСТИКА =====
    ax_stats = fig.add_subplot(gs[0:2, 2])
    ax_stats.axis('off')
    
    stats_text = f"""F1 PERFORMANCE
SUMMARY

LAP TIME
{lap_time_seconds:.2f} sec

SPEED
Avg: {np.mean(v_opt)*3.6:.1f} km/h
Max: {np.max(v_opt)*3.6:.1f} km/h
Min: {np.min(v_opt)*3.6:.1f} km/h

G-FORCES
Long: {np.max(np.abs(a_lon_opt))/vehicle.gravity:.2f}G
Lat: {np.max(a_lat_opt)/vehicle.gravity:.2f}G
Total: {np.max(a_total)/vehicle.gravity:.2f}G

FORCES
Downforce: {np.max(downforce_g):.2f}G
Track: {track_length:.1f}m

SPECS
Mass: {vehicle.mass_kg:.0f}kg
Power: {vehicle.engine_power_watts/1000:.0f}kW
μ: {vehicle.mu_friction:.2f}
C_L: {vehicle.cL_downforce:.2f}
C_D: {vehicle.cD_drag:.2f}"""
    
    ax_stats.text(0.05, 0.95, stats_text, transform=ax_stats.transAxes,
                 fontsize=10, family='monospace', verticalalignment='top',
                 bbox=dict(boxstyle='round', facecolor='#1a1a1a', 
                          edgecolor='cyan', linewidth=2, pad=1))
    
    # ===== ROW 1, COL 4: ИНФОРМАЦИЯ ЗА ТРАСЕТО =====
    ax7 = fig.add_subplot(gs[0, 3])
    ax7.axis('off')
    
    ds_avg = np.mean(ds_array)
    curvature = np.zeros(N)
    for i in range(N):
        i_prev = (i - 1) % N
        i_next = (i + 1) % N
        x0, y0 = x_center[i_prev], y_center[i_prev]
        x1, y1 = x_center[i], y_center[i]
        x2, y2 = x_center[i_next], y_center[i_next]
        dx1, dy1 = x1 - x0, y1 - y0
        dx2, dy2 = x2 - x1, y2 - y1
        cross = dx1*dy2 - dy1*dx2
        ds1 = np.sqrt(dx1**2 + dy1**2) + 1e-9
        ds2 = np.sqrt(dx2**2 + dy2**2) + 1e-9
        curvature[i] = 2 * cross / (ds1 * ds2 * (ds1 + ds2) + 1e-9)
    
    track_info = f"""TRACK INFO

Length: {track_length:.1f}m
Segments: {N}
Avg Δs: {ds_avg:.2f}m

Max Curvature:
{np.max(np.abs(curvature)):.4f} m⁻¹"""
    
    ax7.text(0.1, 0.5, track_info, transform=ax7.transAxes,
             fontsize=10, family='monospace', verticalalignment='center',
             bbox=dict(boxstyle='round', facecolor='#1a1a1a', 
                      edgecolor='orange', linewidth=2, pad=1))
    
    # ===== ROW 2, COL 4: DOWNFORCE =====
    ax6 = fig.add_subplot(gs[1, 3])
    ax6.fill_between(s, 0, downforce_g, color='purple', alpha=0.4)
    ax6.plot(s, downforce_g, color='purple', linewidth=2.5)
    ax6.set_xlabel("Distance [m]", fontsize=11, fontweight='bold')
    ax6.set_ylabel("Downforce [G]", fontsize=11, fontweight='bold')
    ax6.set_title("Aerodynamic Downforce", fontsize=13, fontweight='bold')
    ax6.grid(True, alpha=0.2)
    
    # ===== ROW 3, COL 1: ПРОФИЛ НА СКОРОСТТА =====
    ax2 = fig.add_subplot(gs[2, 0])
    ax2.fill_between(s, 0, v_opt*3.6, color='cyan', alpha=0.3)
    ax2.plot(s, v_opt*3.6, color='cyan', linewidth=2.5)
    ax2.axhline(np.mean(v_opt)*3.6, color='yellow', linestyle='--', 
               linewidth=1.5, alpha=0.7, label=f'Avg: {np.mean(v_opt)*3.6:.1f}')
    ax2.set_xlabel("Distance [m]", fontsize=10, fontweight='bold')
    ax2.set_ylabel("Speed [km/h]", fontsize=10, fontweight='bold')
    ax2.set_title("Speed Profile", fontsize=12, fontweight='bold')
    ax2.legend(fontsize=8)
    ax2.grid(True, alpha=0.2)
    ax2.set_ylim([0, max(v_opt*3.6)*1.1])
    
    # ===== ROW 3, COL 2: G-СИЛИ =====
    ax3 = fig.add_subplot(gs[2, 1])
    ax3.plot(s, a_lon_opt/vehicle.gravity, color='orange', linewidth=2, 
             label='Longitudinal', alpha=0.8)
    ax3.plot(s, (a_lat_opt/vehicle.gravity)/100, color='magenta', linewidth=2, 
             label='Lateral', alpha=0.8)
    ax3.plot(s, (a_total/vehicle.gravity)/100, color='red', linewidth=2.5, 
             label='Total', linestyle='--')
    ax3.axhline(vehicle.mu_friction, color='lime', linestyle=':', 
               linewidth=2, alpha=0.6, label='Grip')
    ax3.set_xlabel("Distance [m]", fontsize=10, fontweight='bold')
    ax3.set_ylabel("G-Force", fontsize=10, fontweight='bold')
    ax3.set_title("G-Force Profile", fontsize=12, fontweight='bold')
    ax3.legend(fontsize=8, loc='upper right')
    ax3.grid(True, alpha=0.2)
    ax3.set_ylim([-6, 6])
    
    # ===== ROW 3, COL 3: МОЩНОСТ =====
    ax4 = fig.add_subplot(gs[2, 2])
    ax4.fill_between(s, 0, np.maximum(power_used/1000, 0), 
                     color='green', alpha=0.4, label='Accel')
    ax4.fill_between(s, 0, np.minimum(power_used/1000, 0), 
                     color='red', alpha=0.4, label='Brake')
    ax4.plot(s, power_used/1000, color='white', linewidth=2)
    ax4.axhline(vehicle.engine_power_watts/1000, color='green', 
               linestyle='--', alpha=0.7, linewidth=1.5)
    ax4.axhline(-vehicle.brake_power_watts/1000, color='red', 
               linestyle='--', alpha=0.7, linewidth=1.5)
    ax4.set_xlabel("Distance [m]", fontsize=10, fontweight='bold')
    ax4.set_ylabel("Power [kW]", fontsize=10, fontweight='bold')
    ax4.set_title("Power Usage", fontsize=12, fontweight='bold')
    ax4.legend(fontsize=8, loc='upper right')
    ax4.grid(True, alpha=0.2)
    
    # ===== ROW 3, COL 4: ВРЕМЕНА ПО СЕКТОРИ =====
    ax5 = fig.add_subplot(gs[2, 3])
    colors = ['gold', 'silver', '#CD7F32']
    bars = ax5.bar(range(1, sectors+1), sector_times, color=colors, 
                   alpha=0.8, edgecolor='white', linewidth=1.5)
    ax5.set_xlabel("Sector", fontsize=11, fontweight='bold')
    ax5.set_ylabel("Time [s]", fontsize=11, fontweight='bold')
    ax5.set_title("Sector Times", fontsize=13, fontweight='bold')
    ax5.set_xticks(range(1, sectors+1))
    
    for bar, time, speed in zip(bars, sector_times, sector_speeds):
        height = bar.get_height()
        ax5.text(bar.get_x() + bar.get_width()/2., height,
                f'{time:.2f}s\n{speed:.0f}km/h',
                ha='center', va='bottom', fontsize=9)
    ax5.set_ylim([30, 55])
    ax5.grid(True, alpha=0.2, axis='y')
    
    # ===== ЗАГЛАВИЕ =====
    plt.suptitle("F1 LAP TIME OPTIMIZATION - COMPLETE ANALYSIS", 
                fontsize=18, fontweight='bold', y=0.995)
    
    plt.tight_layout(rect=[0, 0, 1, 0.97])

    # ===== HOVER POPUP ON RACING LINE =====
    import mplcursors
    cursor = mplcursors.cursor(lc, hover=True)

    @cursor.connect("add")
    def on_hover(sel):
        # LineCollection returns tuple (segment_index, vertex_index)
        i = sel.index
        if isinstance(i, tuple):
            i = i[0]

        i = max(0, min(i, len(v_opt) - 1))

        # Compose hover text
        text = (
            f"Segment: {i}\n"
            f"Speed: {v_opt[i]*3.6:.1f} km/h\n"
            f"Long. Accel: {a_lon_opt[i]:.2f} m/s²\n"
            f"Lat. Accel: {a_lat_opt[i]:.2f} m/s²\n"
            f"Total G: {np.sqrt(a_lon_opt[i]**2 + a_lat_opt[i]**2)/vehicle.gravity:.2f} G"
        )
        sel.annotation.set_text(text)
        sel.annotation.arrow_patch.set(arrowstyle="simple", alpha=0.0)  # remove arrow
        sel.annotation.get_bbox_patch().set(fc="black", alpha=0.7, edgecolor='cyan')


    return fig



def print_summary(v_opt, a_lon_opt, a_lat_opt, vehicle, lap_time_seconds, track_length):
    """Принтира резюме на оптимизацията."""
    a_total = np.sqrt(a_lon_opt**2 + a_lat_opt**2)
    downforce_g = vehicle.k_aero() * v_opt**2 / vehicle.gravity
    
    print(f"\n{'='*60}")
    print(f"F1 LAP TIME OPTIMIZATION COMPLETE")
    print(f"{'='*60}")
    print(f"LAP TIME:       {lap_time_seconds:.2f} seconds")
    print(f"Average speed:  {np.mean(v_opt):.2f} m/s ({np.mean(v_opt)*3.6:.1f} km/h)")
    print(f"Top speed:      {np.max(v_opt):.2f} m/s ({np.max(v_opt)*3.6:.1f} km/h)")
    print(f"Min speed:      {np.min(v_opt):.2f} m/s ({np.min(v_opt)*3.6:.1f} km/h)")
    print(f"{'='*60}")
    print(f"Max acceleration:  {np.max(a_lon_opt):.2f} m/s² ({np.max(a_lon_opt)/vehicle.gravity:.2f}G)")
    print(f"Max braking:       {abs(np.min(a_lon_opt)):.2f} m/s² ({abs(np.min(a_lon_opt))/vehicle.gravity:.2f}G)")
    print(f"Max lateral G:     {np.max(a_lat_opt)/vehicle.gravity:.2f}G")
    print(f"Max combined G:    {np.max(a_total)/vehicle.gravity:.2f}G")
    print(f"Max downforce:     {np.max(downforce_g):.2f}G")
    print(f"Track length:      {track_length:.1f}m")
    print(f"{'='*60}\n")