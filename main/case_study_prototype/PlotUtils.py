from main.case_study_prototype.PhysicalTwin import PhysicalTwin
from main.case_study_prototype.DigitalTwin import DigitalTwin
from src.bayesian_updating.ERADist import ERADist
from typing import List, Tuple, Optional, Dict
from matplotlib import pyplot as plt
import numpy as np

def plot_state_history(cur_time: int, physical_twin: PhysicalTwin, 
                        digital_twin: DigitalTwin, 
                        timeframe: np.ndarray,
                        max_moment: float,
                        measured_moments: Optional[Dict[str, List[float]]] = None, 
                        measured_corrosions: Optional[Dict[str, List[float]]] = None,
                        measured_soil_samples: Optional[Dict[str, List[float]]] = None,
                        plot_dir: str = None):
    """
    Plot the state history of the physical twin and digital twin
    """
    # create figure and axes
    nr_cols = 3
    nr_plots = (len(physical_twin.parameter_names)+2) // nr_cols + 1  # +2 for displacement and pf plots
    fig, ax = plt.subplots(nr_plots, nr_cols, figsize=(nr_cols*10, nr_plots*5))
    
    # plot displacement history
    # _plot_displacement_history(ax[0, 0], timeframe, physical_twin.displacement_history, digital_twin.received_displacements)
    
    # plot moment history
    _plot_moment_history(ax[0, 0], timeframe, cur_time, physical_twin.moment_history, physical_twin.capacity_history, 
                         digital_twin.moment_dict['load_history'], digital_twin.moment_dict['capacity_history'],
                         measured_moments, max_moment)

    # plot pf history
    # _plot_pf_history(ax[0,2], timeframe, physical_twin.pf_history, digital_twin.pf_history)
    
    # plot beta history
    _plot_beta_history(ax[0,2], timeframe, cur_time, physical_twin.beta_history, digital_twin.moment_dict['capacity_history'])

    # plot moment history
    # _plot_moment_history(ax[1], timeframe, physical_twin.moment_history, digital_twin.moment_history)

    # plot water level
    physical_water_state_history = np.array(physical_twin.water_state_history)
    _plot_water_level_history(ax[0,1], timeframe, cur_time, physical_water_state_history, digital_twin.water_level_properties)

    # plot corrosion rate history
    _plot_corrosion_rate_history(ax[1,1], cur_time, timeframe, physical_twin.corrosion_rate_history, digital_twin.corrosion_dict, measured_corrosions=measured_corrosions)
    
    # # digital_state_history = np.array(digital_twin.state_history)
    # # plot parameter history
    physical_state_history = np.array(physical_twin.state_history)
    axes = ax[2:, :].flatten()
    _plot_parameter_history(axes, timeframe, cur_time, physical_state_history,
                            digital_twin.soil_properties, measured_soil_samples=measured_soil_samples)
    
    fontsize = 18
    # set fontsize for all axes
    for ax in ax.flatten():
        ax.tick_params(axis='both', which='major', labelsize=fontsize)
        ax.set_ylabel(ax.get_ylabel(), fontsize=fontsize)
        ax.set_xlabel(ax.get_xlabel(), fontsize=fontsize)
        ax.set_title(ax.get_title(), fontsize=fontsize+3)
        ax.legend(fontsize=fontsize)


    plt.tight_layout()
    # save figure
    if plot_dir is not None:    
        plt.savefig(plot_dir + f'/state_at_{timeframe[cur_time]}.png', dpi=300, bbox_inches='tight')
    # plt.show()
    plt.close()

def _plot_corrosion_rate_history(ax, cur_time: int, timeframe: np.ndarray, physical_twin_corrosion_rate: List[float], digital_twin_corrosion_rate: List[float], measured_corrosions: Optional[Dict[str, List[float]]] = None):
    """
    Plot the corrosion rate history of the physical twin and digital twin
    """
    # Plot physical twin displacement history (deterministic)
    ax.plot(timeframe[:cur_time+1], physical_twin_corrosion_rate[:cur_time+1]*100, 'b-', linewidth=2, label='True Values', marker='o')
    ax.plot(timeframe[cur_time:], physical_twin_corrosion_rate[cur_time:]*100, '--', linewidth=2, marker='o')

    corrosion_rate_dict = digital_twin_corrosion_rate['corrosion_rate_history']
    # dt_stds = np.array(dt_stds)
    dt_means = []
    dt_stds = []
    for keys, values in corrosion_rate_dict.items():
        dt_means.append(values['mean']*100)
        dt_stds.append(values['std']*100)
    dt_means = np.array(dt_means)
    dt_stds = np.array(dt_stds)
    # Plot mean with 95% confidence intervals
    ax.plot(timeframe[:cur_time+1], dt_means[:cur_time+1], '-', color='orange', linewidth=2, label='Predicted Corrosion Rate', marker='s')
    ax.fill_between(timeframe[:cur_time+1], 
                     dt_means[:cur_time+1] - 1.96 * dt_stds[:cur_time+1], 
                     dt_means[:cur_time+1] + 1.96 * dt_stds[:cur_time+1], 
                     alpha=0.2, color='lightblue')
    
    ax.plot(timeframe[cur_time:], dt_means[cur_time:], '--', color='blue', linewidth=2, marker='s')
    ax.fill_between(timeframe[cur_time:], 
                     dt_means[cur_time:] - 1.96 * dt_stds[cur_time:], 
                     dt_means[cur_time:] + 1.96 * dt_stds[cur_time:], 
                     alpha=0.8, color='lightblue')

    # Plot error bars instead of boxplot
    if measured_corrosions is not None:
        measured_corrosion_means = np.array(measured_corrosions['value']).flatten()
        measured_corrosion_sigmas = np.array(measured_corrosions['sigma']).flatten() 
        std = measured_corrosion_sigmas * measured_corrosion_means
        ax.errorbar(timeframe[measured_corrosions['time']], measured_corrosion_means*100, yerr=std*100, fmt='x', color='m', 
                capsize=5, capthick=2, elinewidth=2,
                label='Measured Corrosion Rates')

    # plot vertical line at cur_time
    ax.axvline(x=timeframe[cur_time], color='k', linestyle='--', linewidth=2, label='Current Time')
    
    ax.set_xlabel('Time [years]')
    ax.set_ylabel('Corrosion Rate [%]')
    ax.set_title('Corrosion Rate Evolution')
    ax.legend(loc='upper left')
    ax.set_xlim(timeframe[0], timeframe[-1])
    ax.set_xticks(timeframe)
    ax.grid(True, alpha=0.3)

def _plot_moment_history(ax, timeframe: np.ndarray, cur_time: int, physical_twin_load: List[float], physical_twin_capacity: List[float],
                         digital_twin_load: List[float], digital_twin_capacity: List[float],
                         measured_moments: Optional[Dict[str, List[float]]] = None, 
                         max_moment: float = 40):
    """
    Plot the moment history of the physical twin and digital twin
    """
    # cur_timeframe = timeframe[:len(physical_twin_displacement)+1]
    nr_samples = len(digital_twin_load)
    cur_timeframe_real = timeframe[:nr_samples]

    # Plot physical twin displacement history (deterministic)
    # ax.plot(cur_timeframe_real, physical_twin_capacity[:nr_samples], 'b-', linewidth=2, label='True Capacity', marker='o')
    # ax.plot(timeframe[nr_samples-1:], physical_twin_capacity[nr_samples-1:], '--', linewidth=2, marker='o')

    # Plot physical twin displacement history (deterministic)
    # ax.plot(cur_timeframe_real, physical_twin_load[:nr_samples], 'b-', linewidth=2, label='True Load', marker='o')
    # ax.plot(timeframe[nr_samples-1:], physical_twin_load[nr_samples-1:], linewidth=2, marker='o')
    cur_timeframe = timeframe[:nr_samples]

    dt_c_means = []
    dt_c_stds = []
    for keys, values in digital_twin_capacity.items():
        dt_c_means.append(values['mean'])
        dt_c_stds.append(values['std'])
    dt_c_means = np.array(dt_c_means)
    dt_c_stds = np.array(dt_c_stds)
    
    # Plot mean with 95% confidence intervals of capacity
    ax.plot(timeframe[:cur_time+1], dt_c_means[:cur_time+1], 'g-', linewidth=2, label='Predicted Capacity', marker='s')
    ax.fill_between(timeframe[:cur_time+1], 
                     np.array(dt_c_means[:cur_time+1]) - 1.96 * np.array(dt_c_stds[:cur_time+1]), 
                     np.array(dt_c_means[:cur_time+1]) + 1.96 * np.array(dt_c_stds[:cur_time+1]), 
                     alpha=0.08, color='green')
    
    ax.plot(timeframe[cur_time:], dt_c_means[cur_time:], 'g-', linewidth=2, marker='s')
    ax.fill_between(timeframe[cur_time:], 
                     np.array(dt_c_means[cur_time:]) - 1.96 * np.array(dt_c_stds[cur_time:]), 
                     np.array(dt_c_means[cur_time:]) + 1.96 * np.array(dt_c_stds[cur_time:]), 
                     alpha=0.3, color='green')
    

    dt_m_means = []
    dt_m_stds = []
    for keys, values in digital_twin_load.items():
        dt_m_means.append(np.mean(values['moment_samples']))
        dt_m_stds.append(np.std(values['moment_samples']))
    dt_m_means = np.array(dt_m_means)
    dt_m_stds = np.array(dt_m_stds)

    # Plot mean with 95% confidence intervals of load/moment
    ax.plot(timeframe[:cur_time+1], dt_m_means[:cur_time+1], 'r-', linewidth=2, label='Predicted Load', marker='s')
    ax.fill_between(timeframe[:cur_time+1], 
                     np.array(dt_m_means[:cur_time+1]) - 1.96 * np.array(dt_m_stds[:cur_time+1]), 
                     np.array(dt_m_means[:cur_time+1]) + 1.96 * np.array(dt_m_stds[:cur_time+1]), 
                     alpha=0.08, color='red')

    ax.plot(timeframe[cur_time:], dt_m_means[cur_time:], 'r-', linewidth=2, marker='s')
    ax.fill_between(timeframe[cur_time:], 
                     np.array(dt_m_means[cur_time:]) - 1.96 * np.array(dt_m_stds[cur_time:]), 
                     np.array(dt_m_means[cur_time:]) + 1.96 * np.array(dt_m_stds[cur_time:]), 
                     alpha=0.3, color='red')

    ax.axvline(x=timeframe[cur_time], color='k', linestyle='--', linewidth=2, label='Current Time')
    # # Plot error bars instead of boxplot
    
    if measured_moments is not None:
        # std = np.multiply(np.array(measured_sigmas), np.array(measured_moments).T)
        measured_means = np.array(measured_moments['value']).flatten()
        measured_sigmas = np.array(measured_moments['sigma']).flatten()
        std = measured_sigmas * measured_means
        ax.errorbar(timeframe[measured_moments['time']], measured_means, yerr=std, fmt='x', color='m', 
                capsize=5, capthick=2, elinewidth=2, alpha=0.3,
                label='Measured Moments')
    
    # plot horizontal line at 40
    # ax.axhline(y=max_moment, color='k', linestyle='--', linewidth=2, label='Design Moment')
    
    ax.set_xlabel('Time [years]')
    ax.set_ylabel('Moment [kNm]')
    ax.set_title('Moment Evolution')
    ax.legend(loc='lower right')
    ax.set_xlim(timeframe[0], timeframe[-1])
    ax.set_xticks(timeframe)
    ax.grid(True, alpha=0.3)

def _plot_beta_history(ax, timeframe: np.ndarray, cur_time: int, physical_twin_beta: List[float], digital_twin_capacity: List[float]):
    """
    Plot the beta history of the physical twin and digital twin using step plots with dual y-axes
    """
    keys = list(digital_twin_capacity.keys())
    digital_twin_beta = []
    digital_twin_pf = []
    for i_key, key in enumerate(keys):
        digital_twin_beta.append(digital_twin_capacity[key]['beta'])
        digital_twin_pf.append(digital_twin_capacity[key]['pf'])
    digital_twin_beta = np.array(digital_twin_beta)
    digital_twin_pf = np.array(digital_twin_pf)
    
    nr_samples = len(digital_twin_beta)
    # cur_timeframe = timeframe[:nr_samples]

    # Plot beta on primary (left) y-axis
    # line1 = ax.plot(timeframe[:cur_time+1], physical_twin_beta[:cur_time+1], 'b-', linewidth=2, 
    #                 label='Physical Twin', marker='o')
    # line2 = ax.plot(timeframe[cur_time:], physical_twin_beta[cur_time:], '--', linewidth=2, 
    #                 label='Physical Twin', marker='o')
    line1 = ax.plot(timeframe[:cur_time+1], digital_twin_beta[:cur_time+1], 'r-', linewidth=4, alpha=0.3, 
                    label='Predicted Beta', marker='s')
    line2 = ax.plot(timeframe[cur_time:], digital_twin_beta[cur_time:], 'r-', linewidth=4, marker='s')
    ax.set_ylabel('Reliability Index $\\beta$', color='r')
    ax.tick_params(axis='y', labelcolor='r')
    ax.set_ylim(0, 5)

    # Create secondary (right) y-axis for pf
    ax2 = ax.twinx()
    line3 = ax2.plot(timeframe[:cur_time+1], digital_twin_pf[:cur_time+1], 'b-', linewidth=4, alpha=0.3, 
                     label='Predicted Pf', marker='s')
    line4 = ax2.plot(timeframe[cur_time:], digital_twin_pf[cur_time:], 'b-', linewidth=4, marker='s')
    ax2.set_ylabel('Probability of Failure', color='b')
    ax2.tick_params(axis='y', labelcolor='b')
    ax2.set_ylim(0, 1)

    ax.axvline(x=timeframe[cur_time], color='k', linestyle='--', linewidth=2, label='Current Time')

    ax.set_xlabel('Time [years]')
    ax.set_title('Reliability Index Evolution')
    
    # Combine legends from both axes
    lines = line1 + line3 
    labels = [l.get_label() for l in lines]
    ax.legend(lines, labels, loc='upper right')
    
    ax.grid(True, alpha=0.3)
    ax.set_xlim(timeframe[0], timeframe[-1])
    ax.set_xticks(timeframe)

def _plot_pf_history(ax, timeframe: np.ndarray, physical_twin_pf: List[float], digital_twin_pf: List[float]):
    """
    Plot the pf history of the physical twin and digital twin
    """
    # cur_timeframe = timeframe[:len(physical_twin_pf)+1]
    nr_samples = len(digital_twin_pf)
    cur_timeframe = timeframe[:nr_samples]

    # Plot physical twin pf history
    ax.plot(cur_timeframe, physical_twin_pf[:nr_samples], 'b-', linewidth=2, label='Physical Twin', marker='o')
    ax.plot(timeframe[nr_samples-1:], physical_twin_pf[nr_samples-1:], '--', linewidth=2, marker='o')
    

    ax.plot(cur_timeframe, digital_twin_pf, 'r-', linewidth=2, label='Digital Twin', marker='s')
    # ax.plot(timeframe[nr_samples-1:], digital_twin_pf[nr_samples-1:], '--', linewidth=2, label='Future Digital Twin', marker='s')

    # digital_twin_pf = np.array(digital_twin_pf)
    # mean = np.mean(digital_twin_pf, axis=1)
    # std = np.std(digital_twin_pf, axis=1)
    # ax.plot(cur_timeframe, mean, 'r-', linewidth=2, label='Digital Twin', marker='s')
    # ax.fill_between(cur_timeframe, mean - std, mean + std, alpha=0.3, color='red', label='Digital Twin (±1σ)')
    
    ax.set_xlabel('Time [years]')
    ax.set_ylabel('Probability of Failure')
    ax.set_title('Probability of Failure History')
    ax.legend()
    ax.grid(True, alpha=0.3)
    ax.set_xlim(timeframe[0], timeframe[-1])
    ax.set_xticks(timeframe)
    # ax.set_ylim(0, max(max(physical_twin_pf), max(mean))*1.5)
    ax.set_ylim(0, 0.5)

def _plot_parameter_history(ax_list, timeframe: np.ndarray, cur_time: int,
                            physical_twin_state_history: List[List[float]], 
                            digital_twin_soil_properties: Dict,
                            measured_soil_samples: Optional[Dict[str, List[float]]] = None):
    """
    Plot the parameter history of the physical twin and digital twin
    """
    soil_property_state_history = digital_twin_soil_properties['state_history']
    soil_property_parameter_names = digital_twin_soil_properties['parameter_names']
    # cur_timeframe = timeframe[:len(physical_twin_state_history)+1]
    keys = list(soil_property_state_history.keys())
    nr_samples = len(keys)
    cur_timeframe = timeframe[:nr_samples]
    
    dt_param_means = []
    dt_param_stds = []
    for i, key in enumerate(keys):
        dt_param_means.append(soil_property_state_history[key]['mean'])
        dt_param_stds.append(soil_property_state_history[key]['std'])
    dt_param_means = np.array(dt_param_means)
    dt_param_stds = np.array(dt_param_stds)

    past_dt_param_means = np.vstack([dt_param_means[:-1,:], dt_param_means[-2,:]])
    past_dt_param_stds = np.vstack([dt_param_stds[:-1,:], dt_param_stds[-2,:]])

    predicted_dt_param_mean = np.repeat(dt_param_means[-1,:].reshape(-1, 1), len(timeframe[cur_time:]), axis=1).T
    predicted_dt_param_std = np.repeat(dt_param_stds[-1,:].reshape(-1, 1), len(timeframe[cur_time:]), axis=1).T

    for i, param_name in enumerate(soil_property_parameter_names):
        ax = ax_list[i]
        ax.plot(timeframe[:cur_time+1], past_dt_param_means[:,i], 'r-', linewidth=4, alpha=0.3, label='Digital Twin (Mean)', marker='s')
        ax.fill_between(cur_timeframe, 
                        past_dt_param_means[:,i] - 1.96 * past_dt_param_stds[:,i], 
                        past_dt_param_means[:,i] + 1.96 * past_dt_param_stds[:,i], 
                        alpha=0.08, color='orange')
        
        ax.plot(timeframe[cur_time:], predicted_dt_param_mean[:,i], 'r-', linewidth=4, marker='s')
        ax.fill_between(timeframe[cur_time:], 
                        predicted_dt_param_mean[:,i] - 1.96 * predicted_dt_param_std[:,i], 
                        predicted_dt_param_mean[:,i] + 1.96 * predicted_dt_param_std[:,i], 
                        alpha=0.5, color='orange')
        
        pt_param_values = [state[i] for state in physical_twin_state_history]
        ax.plot(cur_timeframe, pt_param_values[:nr_samples], 'b-', linewidth=4, alpha=0.3, label='Physical Twin', marker='o')
        ax.plot(timeframe[nr_samples-1:], pt_param_values[nr_samples-1:], 'b-', linewidth=4, marker='o')

        if measured_soil_samples is not None:
            # print(50*'-')
            # print(measured_soil_samples)
            nr_measurements = len(measured_soil_samples['time'])
            for i_measurement in range(nr_measurements):
                cur_measurement_time = measured_soil_samples['time'][i_measurement]
                cur_measurement_value = measured_soil_samples['value'][i_measurement]
                cur_measurement_sigma = measured_soil_samples['sigma'][i_measurement]
                ax.errorbar(timeframe[cur_measurement_time[i]], cur_measurement_value[i], yerr=cur_measurement_sigma[i], fmt='x', color='m', 
                        capsize=5, capthick=2, elinewidth=2, alpha=0.3,
                        label='Measurement')

        ax.axvline(x=timeframe[cur_time], color='k', linestyle='--', linewidth=2, label='Current Time')

        plot_param_name = param_name.replace('_', ' ')
        plot_param_name = plot_param_name.replace('SheetPilingElementEI', 'EI (kNm)')
        plot_param_name = plot_param_name.replace('soilphi', 'Phi')
        plot_param_name = plot_param_name.replace('soilcohesion', 'Cohesion')
        plot_param_name = plot_param_name.replace('soilcurkb1', 'Stiffness')
        ax.set_ylabel(plot_param_name)
        ax.set_title(f'{plot_param_name}')
        ax.set_xlim(timeframe[0], timeframe[-1])
        ax.set_xticks(timeframe)
        ax.legend()
        ax.grid(True, alpha=0.3)

def _plot_water_level_history(ax, timeframe: np.ndarray, cur_time: int, physical_twin_water_level: List[List[float]], dt_water_level: Dict):
    """
    Plot the water level history of the physical twin and digital twin
    """
    # print(50*'-')
    # print(physical_twin_water_level)
    ax.plot(timeframe, physical_twin_water_level[:,0], 'r--', linewidth=2, label='True Soil Water Level', marker='o')
    ax.plot(timeframe, physical_twin_water_level[:,1], 'g-', linewidth=2, label='True Canal Water Level', marker='o')

    water_samples = dt_water_level['samples']
    
    water_samples_means = np.mean(water_samples, axis=0)
    water_samples_stds = np.std(water_samples, axis=0)

    soil_water_level = np.repeat(water_samples_means[0], len(timeframe))
    canal_water_level = np.repeat(water_samples_means[1], len(timeframe))
    soil_water_level_stds = np.repeat(water_samples_stds[0], len(timeframe))
    canal_water_level_stds = np.repeat(water_samples_stds[1], len(timeframe))

    ax.plot(timeframe[:cur_time+1], soil_water_level[:cur_time+1], '-', color='grey', linewidth=2, label='DT Soil Water Level', marker='s')
    ax.fill_between(timeframe[:cur_time+1], 
                    soil_water_level[:cur_time+1] - 1.96 * soil_water_level_stds[:cur_time+1], 
                    soil_water_level[:cur_time+1] + 1.96 * soil_water_level_stds[:cur_time+1], 
                    alpha=0.08, color='red')
    
    ax.plot(timeframe[:cur_time+1], canal_water_level[:cur_time+1], '-', color='grey', linewidth=2, label='DT Canal Water Level', marker='s')
    ax.fill_between(timeframe[:cur_time+1], 
                    canal_water_level[:cur_time+1] - 1.96 * canal_water_level_stds[:cur_time+1], 
                    canal_water_level[:cur_time+1] + 1.96 * canal_water_level_stds[:cur_time+1], 
                    alpha=0.08, color='green')

    ax.plot(timeframe[cur_time:], soil_water_level[cur_time:], '-', color='grey', linewidth=2, marker='s')
    ax.fill_between(timeframe[cur_time:], 
                    soil_water_level[cur_time:] - 1.96 * soil_water_level_stds[cur_time:], 
                    soil_water_level[cur_time:] + 1.96 * soil_water_level_stds[cur_time:], 
                    alpha=0.2, color='red')
    
    ax.plot(timeframe[cur_time:], canal_water_level[cur_time:], '-', color='grey', linewidth=2, marker='s')
    ax.fill_between(timeframe[cur_time:], 
                    canal_water_level[cur_time:] - 1.96 * canal_water_level_stds[cur_time:], 
                    canal_water_level[cur_time:] + 1.96 * canal_water_level_stds[cur_time:], 
                    alpha=0.2, color='green')
    
        
    ax.axvline(x=timeframe[cur_time], color='k', linestyle='--', linewidth=2, label='Current Time')
        
    # ax.plot(timeframe, digital_twin_water_level, 'r-', linewidth=2, label='Digital Twin', marker='s')
    ax.set_xlabel('Time [years]')
    ax.set_ylabel('Water Level [m+NAP]')
    ax.set_ylim(-0.95, -0)
    ax.set_xlim(timeframe[0], timeframe[-1])
    ax.set_xticks(timeframe)
    ax.set_title('Water Level Evolution')
    ax.grid(True, alpha=0.3)
    ax.legend()

if __name__ == "__main__":
    pass