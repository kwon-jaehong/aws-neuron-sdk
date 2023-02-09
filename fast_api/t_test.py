""" neuron-monitor frontend which acts as a Prometheus instance
"""
import sys
import json
import argparse
import signal
import os


try:
    from prometheus_client import start_http_server, Gauge, Counter, Info
except:
    print("Missing package - please install prometheus_client and then try again. 'pip install prometheus_client'")
    sys.exit(1)

dictfilt = lambda x, y: dict([ (i,x[i]) for i in x if i in set(y) ])

# core_list = Gauge('neuroncore_utilization_ratio', 'NeuronCore utilization ratio', labels.keys())
core_list = Gauge('neuroncore_utilization_ratio', 'NeuronCore utilization ratio',['instance_name', 'instance_id', 'instance_type', 'availability_zone', 'region', 'using', 'neuroncore'])

def get_instance_labels(instance_info,neuron_hardware_info):
    instance_labels = {
        'instance_name': instance_info['instance_name'],
        'instance_id': instance_info['instance_id'],
        'instance_type': instance_info['instance_type'],
        'availability_zone': instance_info['instance_availability_zone'],
        'region': instance_info['instance_region'],
        'subnet_id': instance_info['subnet_id'],
        'neuron_device_count': neuron_hardware_info['neuron_device_count'],
        'neuroncore_per_device_count': neuron_hardware_info['neuroncore_per_device_count'],
        'neuron_core_sum_count': neuron_hardware_info['neuron_device_count'] * neuron_hardware_info['neuroncore_per_device_count']
    }
    return instance_labels


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
    elif in_use_neroncore!=None:
        use_core_list.append(int(in_use_neroncore))
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

def process_data(monitor_data, instance_info,flg):
    
    ## 환경변수 파싱, 수정 부분 시작
    in_use_neroncore = str(os.getenv('NEURON_RT_VISIBLE_CORES'))
    use_core_list = []
    if '-' in in_use_neroncore:
        use_core_range = [int(x) for x in in_use_neroncore.split('-')]
        use_core_list = [*range(use_core_range[0],use_core_range[1]+1)]
    elif ',' in in_use_neroncore:
        use_core_list = [int(x) for x in in_use_neroncore.split(',')]
    elif in_use_neroncore!=None:
        use_core_list.append(int(in_use_neroncore))
    else:
        pass
    use_core_list_flg = [None] * instance_info['neuron_core_sum_count']
    for i in use_core_list:
        use_core_list_flg[i]=True
    ## 수정 부분 끝
    
    
    ## 뉴런코어 사용률을 총 계산할 리스트 선언
    neuron_core_riato_list = [0] * instance_info['neuron_core_sum_count']
    if monitor_data['neuron_runtime_data'] is not None:
        for runtime in monitor_data['neuron_runtime_data']:
            for core_number,use_ratio in runtime['report']['neuroncore_counters']['neuroncores_in_use'].items():
                neuron_core_riato_list[int(core_number)] += use_ratio['neuroncore_utilization']

    select_key =['instance_name','instance_id','instance_type','availability_zone','region']
    labels = dictfilt(instance_info,select_key)
    labels['using'] = None
    labels['neuroncore'] = None

    for ind in range(0,instance_info['neuron_core_sum_count']):
        labels['neuroncore'] = int(ind)
        labels['using']= use_core_list_flg[int(ind)] # 레이블 정보 추가
        core_list.labels(**labels).set(neuron_core_riato_list[ind])


def update_loop(flg):
    running = True

    def signal_handler(*_):
        nonlocal running
        running = False
    signal.signal(signal.SIGINT, signal_handler)
      
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
            instance_labels = get_instance_labels(monitor_data['instance_info'],monitor_data['neuron_hardware_info'])
        process_data(monitor_data, instance_labels,flg)


def main():
    arg_parser = argparse.ArgumentParser()
    arg_parser.add_argument('-p', '--port', default=9000,
                            type=int, help='HTTP port on which to run the server')
    arg_parser.add_argument('-m', '--mode', default=False,
                            type=bool, help='사용중인 게이지만 출력')
    args = arg_parser.parse_args()

    start_http_server(args.port)
    update_loop(args.mode)


if __name__ == '__main__':
    main()
