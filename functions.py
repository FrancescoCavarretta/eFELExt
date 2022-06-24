import efel
import numpy
import copy
from .exp2_fit import membrane_time_constant

def _trace_trim(trace, tpad=200):
  idx = numpy.logical_and(trace['T'] >= (trace['stim_start'][0] - tpad),
                          trace['T'] <= (trace['stim_end'][0] + tpad))
  return { 'T':trace['T'][idx],
           'V':trace['V'][idx],
           'stim_start':trace['stim_start'],
           'stim_end':trace['stim_end'] }


def AP_count(trace):
  trace = copy.deepcopy(trace)

  if hasattr(trace['T'], 'to_numpy'):
    trace['T'] = trace['T'].to_numpy()
    
  if hasattr(trace['V'], 'to_numpy'):
    trace['V'] = trace['V'].to_numpy()
    
  trace = _trace_trim(trace)
  ef = efel.getFeatureValues([trace], ['AP_begin_voltage'])[0]    
  
  try:
    return ef['AP_begin_voltage'].shape[0]
  except:
    pass
  return 0


def AP1_amp_rev(trace):
  trace = copy.deepcopy(trace)

  if hasattr(trace['T'], 'to_numpy'):
    trace['T'] = trace['T'].to_numpy()
    
  if hasattr(trace['V'], 'to_numpy'):
    trace['V'] = trace['V'].to_numpy()
    
  trace = _trace_trim(trace)
  ef = efel.getFeatureValues([trace], ['AP_begin_voltage', 'AP1_peak'])[0]
  
  try:
    return ef['AP1_peak'][0] - ef['AP_begin_voltage'][0]
  except:
    pass
  return numpy.nan


def AP2_amp_rev(trace):
  trace = copy.deepcopy(trace)

  if hasattr(trace['T'], 'to_numpy'):
    trace['T'] = trace['T'].to_numpy()
    
  if hasattr(trace['V'], 'to_numpy'):
    trace['V'] = trace['V'].to_numpy()
    
  trace = _trace_trim(trace)
  ef = efel.getFeatureValues([trace], ['AP_begin_voltage', 'AP2_peak'])[0]
  
  try:
    return ef['AP2_peak'][0] - ef['AP_begin_voltage'][1]
  except:
    pass
  return numpy.nan


def AP_count_after_stim(trace):
  trace = copy.deepcopy(trace)

  if hasattr(trace['T'], 'to_numpy'):
    trace['T'] = trace['T'].to_numpy()
    
  if hasattr(trace['V'], 'to_numpy'):
    trace['V'] = trace['V'].to_numpy()
    
  trace['stim_start'][0] = trace['stim_end'][0]
  trace['stim_end'][0] = trace['T'][-1]
  trace = _trace_trim(trace, tpad=0)
  return AP_count(trace)


def AP_count_before_stim(trace):
  trace = copy.deepcopy(trace)

  if hasattr(trace['T'], 'to_numpy'):
    trace['T'] = trace['T'].to_numpy()
    
  if hasattr(trace['V'], 'to_numpy'):
    trace['V'] = trace['V'].to_numpy()
    
  trace['stim_end'][0] = trace['stim_start'][0]
  trace['stim_start'][0] = trace['T'][0]
  trace = _trace_trim(trace, tpad=0)
  return AP_count(trace)



def clustering_index(trace):
  trace = copy.deepcopy(trace)

  if hasattr(trace['T'], 'to_numpy'):
    trace['T'] = trace['T'].to_numpy()
    
  if hasattr(trace['V'], 'to_numpy'):
    trace['V'] = trace['V'].to_numpy()
    
  ef = efel.getFeatureValues([trace], ['all_ISI_values', 'time_to_first_spike'])[0]
  
  try:
    tspk = numpy.concatenate((numpy.array([0.]), numpy.cumsum(ef['all_ISI_values']))) + ef['time_to_first_spike']
    half_stim_dur = (trace['stim_end'][0] - trace['stim_start'][0]) / 2.0
    return len(tspk[tspk < half_stim_dur])/(0.+len(tspk))
  except:
    pass
  return None




def input_resistance(trace, tpad=500):
  trace = copy.deepcopy(trace)

  if hasattr(trace['T'], 'to_numpy'):
    trace['T'] = trace['T'].to_numpy()
    
  if hasattr(trace['V'], 'to_numpy'):
    trace['V'] = trace['V'].to_numpy()
  
  baseline = numpy.mean(trace['V'][ trace['T'] < tpad ])

  trace = _trace_trim(trace, tpad=0)
  Vpeak = trace['V'][-1] - baseline
  retval = (Vpeak * 1e-3) / (-10.0 * 1e-12) * 1e-6
  
  return retval



def decay_time_constant_after_stim2(trace):
  trace = copy.deepcopy(trace)

  if hasattr(trace['T'], 'to_numpy'):
    trace['T'] = trace['T'].to_numpy()
    
  if hasattr(trace['V'], 'to_numpy'):
    trace['V'] = trace['V'].to_numpy()
    
  trace['stim_end'][0] = trace['T'][-1]
  trace = _trace_trim(trace, tpad=0)
  return membrane_time_constant(trace['T'],  trace['V']) 






  
function = { 'clustering_index':clustering_index,
             'input_resistance':input_resistance,
             'AP_count':AP_count,
             'AP_count_after_stim':AP_count_after_stim,
             'AP_count_before_stim':AP_count_before_stim,
             'decay_time_constant_after_stim2':decay_time_constant_after_stim2,
             'AP1_amp_rev':AP1_amp_rev,
             'AP2_amp_rev':AP2_amp_rev }

  

def getFeatureValues(packed_trace, efeature_name):
  """
  Return the features values
  """
  ret_val = {}
  
  for _efeature_name in efeature_name: 
      
    if _efeature_name in function:
      val = function[_efeature_name](packed_trace)
    else:
      val = efel.getFeatureValues([ packed_trace ], [ _efeature_name ])[0][_efeature_name]

    # type checking
    if val is None:
      val = numpy.nan
    elif isinstance(val, numpy.ndarray):
      if val.shape[0] == 0:
        if 'AP_count' in _efeature_name:
          val = 0
        else:
          val = numpy.nan
      else:
        val = val[0]
      
    if numpy.isinf(val):
      val = numpy.nan
      
    ret_val[_efeature_name] = val
    
  return ret_val

