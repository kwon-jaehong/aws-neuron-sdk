#!/usr/bin/env python3

""" neuron-monitor frontend which acts as a Prometheus instance
"""
import sys
import json
import argparse
import signal


try:
    from prometheus_client import start_http_server, Gauge, Counter, Info
except:
    print("Missing package - please install prometheus_client and then try again. 'pip install prometheus_client'")
    sys.exit(1)



def get_instance_labels(instance_info):
    instance_labels = {
        'instance_name': instance_info['instance_name'],
        'instance_id': instance_info['instance_id'],
        'instance_type': instance_info['instance_type'],
        'availability_zone': instance_info['instance_availability_zone'],
        'region': instance_info['instance_region'],
        'subnet_id': instance_info['subnet_id']
    }
    return instance_labels


def get_runtime_labels(instance_info, runtime_tag):
    label_dict = instance_info.copy()
    label_dict['runtime_tag'] = runtime_tag
    return label_dict

## 함수 수정
def process_neuroncore_counters(group_obj, data, labels):   
    ## 환경변수 파싱, 수정 부분 시작
    in_use_neroncore = str(os.getenv('NEURON_RT_VISIBLE_CORES'))
    use_core_list = []
    if '-' in in_use_neroncore:
        use_core_range = [int(x) for x in in_use_neroncore.split('-')]
        use_core_list = [*range(use_core_range[0],use_core_range[1]+1)]
    elif ',' in in_use_neroncore:
        use_core_list = [int(x) for x in in_use_neroncore.split(',')]
    else:
        pass    
    use_core_list_flg = [None] * len(data['neuroncores_in_use'])
    for i in use_core_list:
        use_core_list_flg[i]=True
    ## 수정 부분 끝
        
    gauge_name = 'neuroncore_utilization_ratio'
    labels['neuroncore'] = None
    labels['using'] = None # using 변수 추가
    if gauge_name not in group_obj:
        group_obj[gauge_name] = Gauge(gauge_name, 'NeuronCore utilization ratio', labels.keys())
    for nc_idx, nc_data in data['neuroncores_in_use'].items():
        labels['neuroncore'] = int(nc_idx)
        labels['using']= use_core_list_flg[int(nc_idx)] # 레이블 정보 추가
        group_obj[gauge_name].labels(**labels).set(nc_data['neuroncore_utilization'] / 100.0)


def process_neuron_runtime_vcpu_usage(group_obj, data, labels):
    gauge_name = 'neuron_runtime_vcpu_usage_ratio'
    labels['usage_type'] = None
    if gauge_name not in group_obj:
        group_obj[gauge_name] = Gauge(gauge_name, 'Runtime vCPU utilization ratio', labels.keys())
    cpu_usage_fields = ['user', 'system']
    for field in cpu_usage_fields:
        labels['usage_type'] = field
        group_obj[gauge_name].labels(**labels).set(data['vcpu_usage'][field] / 100.0)


def process_memory_used(group_obj, data, labels):
    gauge_name = 'neuron_runtime_memory_used_bytes'
    labels['memory_location'] = None
    if gauge_name not in group_obj:
        group_obj[gauge_name] = Gauge(gauge_name, 'Runtime memory used bytes', labels.keys())
    mem_locations = ['host', 'neuron_device']
    for mem_location_type in mem_locations:
        labels['memory_location'] = mem_location_type
        group_obj[gauge_name].labels(**labels).set(data['neuron_runtime_used_bytes'][mem_location_type])


def process_inference_stats(group_obj, data, labels):
    counter_name = 'inference_errors_total'
    err_labels = labels.copy()
    err_labels['error_type'] = None
    if counter_name not in group_obj:
        group_obj[counter_name] = Counter(counter_name, 'Inference errors total', err_labels.keys())
    error_summary = data['error_summary']
    for error_type in error_summary:
        err_labels['error_type'] = error_type
        group_obj[counter_name].labels(**err_labels).inc(error_summary[error_type])

    counter_name = 'inference_status_total'
    status_labels = labels.copy()
    status_labels['status_type'] = None
    if counter_name not in group_obj:
        group_obj[counter_name] = Counter(counter_name, 'Inference status total', status_labels.keys())
    inference_summary = data['inference_summary']
    for inference_outcome in inference_summary:
        status_labels['status_type'] = inference_outcome
        group_obj[counter_name].labels(**status_labels).inc(inference_summary[inference_outcome])

    gauge_name = 'inference_latency_seconds'
    latency_labels = labels.copy()
    latency_labels['percentile'] = None
    if gauge_name not in group_obj:
        group_obj[gauge_name] = Gauge(gauge_name, 'Inference latency in seconds', latency_labels.keys())
    latency_stats = data['latency_stats']
    if latency_stats['total_latency'] is not None:
        for percentile in latency_stats['total_latency']:
            latency_labels['percentile'] = percentile
            group_obj[gauge_name].labels(**latency_labels).set(latency_stats['total_latency'][percentile])


def process_neuron_hw_counters(group_obj, data, labels):
    counter_name = 'hardware_ecc_events_total'
    labels['event_type'] = None
    labels['neuron_device_index'] = None
    if counter_name not in group_obj:
        group_obj[counter_name] = Counter(counter_name, 'Hardware ecc events total', labels.keys())
    hw_counters = ['mem_ecc_corrected', 'mem_ecc_uncorrected', 'sram_ecc_uncorrected']
    for device in data['neuron_devices']:
        for counter in hw_counters:
            labels['event_type'] = counter
            labels['neuron_device_index'] = device['neuron_device_index']
            group_obj[counter_name].labels(**labels).inc(device[counter])


def process_vcpu_usage(group_obj, data, labels):
    cpu_usage_aggregation = {
        'user': ['user', 'nice'],
        'system': ['system', 'io_wait', 'irq', 'soft_irq']
    }
    gauge_name = 'system_vcpu_count'
    if gauge_name not in group_obj:
        group_obj[gauge_name] = Gauge(gauge_name, 'System vCPU count', labels.keys())
    group_obj[gauge_name].labels(**labels).set(len(data['usage_data']))

    labels['usage_type'] = None
    gauge_name = 'system_vcpu_usage_ratio'
    if gauge_name not in group_obj:
        group_obj[gauge_name] = Gauge(gauge_name, 'System CPU utilization ratio', labels.keys())
    for field, aggregated in cpu_usage_aggregation.items():
        aggregate_value = sum([data['average_usage'][item] for item in aggregated])
        aggregate_value = min(aggregate_value, 100.0)
        labels['usage_type'] = field
        group_obj[gauge_name].labels(**labels).set(aggregate_value / 100.0)


def process_memory_info(group_obj, data, labels):
    for entries in [('memory', 'system_memory'), ('swap', 'system_swap')]:
        for stat in ['total_bytes', 'used_bytes']:
            gauge_name = '{}_{}'.format(entries[1], stat)
            if gauge_name not in group_obj:
                group_obj[gauge_name] = Gauge(gauge_name,
                                              'System {} {} bytes'.format(entries[0], stat), labels.keys())
            src_entry = '{}_{}'.format(entries[0], stat)
            group_obj[gauge_name].labels(**labels).set(data[src_entry])


def process_instance_info(metric_objects, instance_data):
    if 'instance_info' not in metric_objects:
        metric_objects['instance_info'] = Info('instance', 'EC2 instance information')
        metric_objects['instance_info'].info(instance_data)


def process_report_entries(metric_objects, report_entries, labels, runtime_tag=None):
    for metric_group_name, metric_group_data in report_entries.items():
        handler_name = 'process_{}'.format(metric_group_name)
        if handler_name in globals():
            crt_error = metric_group_data['error']
            if crt_error == '':
                if metric_group_name not in metric_objects:
                    metric_objects[metric_group_name] = {}
                metric_group_object = metric_objects[metric_group_name]
                globals()[handler_name](metric_group_object, metric_group_data, labels.copy())
            else:
                if runtime_tag is not None:
                    print('Error getting {} for runtime tag {}: {}'.format(
                        metric_group_name, runtime_tag, crt_error), file=sys.stderr)
                else:
                    print('Error getting {}: {}'.format(metric_group_name, crt_error), file=sys.stderr)


def process_data(metric_objects, monitor_data, instance_info):
    if monitor_data['neuron_runtime_data'] is not None:
        for runtime in monitor_data['neuron_runtime_data']:
            runtime_tag = runtime['neuron_runtime_tag']

            if runtime['error'] != '':
                print('Runtime {} error: {}'.format(runtime_tag, runtime['error']), file=sys.stderr)
                continue

            process_report_entries(metric_objects, runtime['report'],
                                   get_runtime_labels(instance_info, runtime_tag), runtime_tag)
    if monitor_data['system_data'] is not None:
        process_report_entries(metric_objects, monitor_data['system_data'], instance_info)
    process_instance_info(metric_objects, instance_info)


def update_loop():
    running = True

    def signal_handler(*_):
        nonlocal running
        running = False
    signal.signal(signal.SIGINT, signal_handler)

    """ Dictionary containing all prometheus client objects, first by metric group and
        then by metric, for example, for neuroncore_counters:
        all_metric_objects['neuroncore_counters']['neuroncore_utilization_ratio'] = Gauge(...)
    """
    all_metric_objects = {}
    instance_labels = None
    while running:
        line = sys.stdin.readline()
        if len(line) == 0:
            continue
        try:
            monitor_data = json.loads(line)
        except Exception as exc:
            print('Unable to decode JSON {}'.format(exc))
            continue
        if instance_labels is None:
            instance_labels = get_instance_labels(monitor_data['instance_info'])
        process_data(all_metric_objects, monitor_data, instance_labels)


def main():
    arg_parser = argparse.ArgumentParser()
    arg_parser.add_argument('-p', '--port', default=8000,
                            type=int, help='HTTP port on which to run the server')
    args = arg_parser.parse_args()

    start_http_server(args.port)
    update_loop()


if __name__ == '__main__':
    main()
